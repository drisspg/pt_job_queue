from __future__ import annotations

from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import MagicMock, patch

import pytest

from ptq.application.rebase_service import rebase
from ptq.domain.models import (
    JobRecord,
    PtqError,
    RebaseInfo,
    RebaseState,
)
from ptq.infrastructure.job_repository import JobRepository


def _make_repo(tmp_path: Path) -> tuple[JobRepository, str]:
    repo = JobRepository(tmp_path / "jobs.json")
    repo.save(
        JobRecord(
            job_id="20260217-42",
            issue=42,
            machine="gpu-dev",
            workspace="~/ptq_workspace",
        )
    )
    return repo, "20260217-42"


def _ok(stdout: str = "") -> CompletedProcess[str]:
    return CompletedProcess("", 0, stdout, "")


def _fail(stderr: str = "") -> CompletedProcess[str]:
    return CompletedProcess("", 1, "", stderr)


class TestRebaseInfoRoundtrip:
    def test_idle_to_dict_empty(self):
        assert RebaseInfo().to_dict() == {}

    def test_roundtrip(self):
        info = RebaseInfo(
            state=RebaseState.SUCCEEDED,
            target_ref="origin/main",
            before_sha="aaa",
            after_sha="bbb",
            attempts=2,
        )
        restored = RebaseInfo.from_dict(info.to_dict())
        assert restored.state == RebaseState.SUCCEEDED
        assert restored.before_sha == "aaa"
        assert restored.after_sha == "bbb"
        assert restored.attempts == 2

    def test_from_empty_dict(self):
        assert RebaseInfo.from_dict({}).state == RebaseState.IDLE

    def test_none_returns_default(self):
        assert RebaseInfo.from_dict(None).state == RebaseState.IDLE


class TestJobRecordRebase:
    def test_rebase_omitted_when_none(self):
        d = JobRecord(job_id="j").to_dict()
        assert "rebase" not in d

    def test_rebase_omitted_when_idle(self):
        d = JobRecord(job_id="j", rebase=RebaseInfo()).to_dict()
        assert "rebase" not in d

    def test_rebase_persisted_when_active(self):
        info = RebaseInfo(state=RebaseState.SUCCEEDED, after_sha="abc")
        d = JobRecord(job_id="j", rebase=info).to_dict()
        assert "rebase" in d
        assert d["rebase"]["state"] == "succeeded"

    def test_rebase_roundtrip(self):
        info = RebaseInfo(
            state=RebaseState.NEEDS_HUMAN,
            target_ref="origin/main",
            error="conflicts remain",
        )
        record = JobRecord(job_id="j", rebase=info)
        d = record.to_dict()
        restored = JobRecord.from_dict("j", d)
        assert restored.rebase is not None
        assert restored.rebase.state == RebaseState.NEEDS_HUMAN
        assert restored.rebase.error == "conflicts remain"

    def test_rebase_info_property(self):
        record = JobRecord(job_id="j")
        assert record.rebase is None
        ri = record.rebase_info
        assert ri.state == RebaseState.IDLE
        assert record.rebase is ri


class TestRebaseRepository:
    def test_save_rebase(self, tmp_path):
        repo = JobRepository(tmp_path / "jobs.json")
        repo.save(JobRecord(job_id="j1", machine="gpu-dev", workspace="~/ws"))
        repo.save_rebase("j1", {"state": "running", "target_ref": "origin/main"})
        job = repo.get("j1")
        assert job.rebase is not None
        assert job.rebase.state == RebaseState.RUNNING

    def test_clear_rebase(self, tmp_path):
        repo = JobRepository(tmp_path / "jobs.json")
        repo.save(
            JobRecord(
                job_id="j1",
                machine="gpu-dev",
                workspace="~/ws",
                rebase=RebaseInfo(state=RebaseState.SUCCEEDED),
            )
        )
        repo.save_rebase("j1", {})
        job = repo.get("j1")
        assert job.rebase is None

    def test_save_rebase_missing_job(self, tmp_path):
        repo = JobRepository(tmp_path / "jobs.json")
        repo.save_rebase("nonexistent", {"state": "running"})


class TestRebaseClean:
    def test_clean_rebase_success(self, tmp_path):
        repo, job_id = _make_repo(tmp_path)

        def run_side(cmd, check=True):
            if "test -d" in cmd or "test -f" in cmd:
                return _ok()
            if "rev-parse HEAD" in cmd:
                return _ok("abc123\n")
            if "rev-parse --verify" in cmd:
                return _ok("def456\n")
            if "fetch origin" in cmd:
                return _ok()
            if "git rebase" in cmd and "--continue" not in cmd:
                return _ok()
            return _ok()

        backend = MagicMock()
        backend.workspace = "~/ptq_workspace"
        backend.run = MagicMock(side_effect=run_side)

        with patch(
            "ptq.application.rebase_service.backend_for_job", return_value=backend
        ):
            result = rebase(repo, job_id)

        assert result.state == RebaseState.SUCCEEDED
        assert result.before_sha == "abc123"

        job = repo.get(job_id)
        assert job.rebase is not None
        assert job.rebase.state == RebaseState.SUCCEEDED

    def test_rebase_target_not_found(self, tmp_path):
        repo, job_id = _make_repo(tmp_path)

        def run_side(cmd, check=True):
            if "test -d" in cmd or "test -f" in cmd:
                return _ok()
            if "rev-parse HEAD" in cmd:
                return _ok("abc123\n")
            if "rev-parse --verify" in cmd:
                return _fail("not found")
            if "fetch origin" in cmd:
                return _ok()
            return _ok()

        backend = MagicMock()
        backend.workspace = "~/ptq_workspace"
        backend.run = MagicMock(side_effect=run_side)

        with (
            patch(
                "ptq.application.rebase_service.backend_for_job", return_value=backend
            ),
            pytest.raises(PtqError, match="Target ref not found"),
        ):
            rebase(repo, job_id)

        job = repo.get(job_id)
        assert job.rebase.state == RebaseState.FAILED

    def test_no_worktree_raises(self, tmp_path):
        repo, job_id = _make_repo(tmp_path)
        backend = MagicMock()
        backend.workspace = "~/ptq_workspace"
        backend.run.return_value = _fail()

        with (
            patch(
                "ptq.application.rebase_service.backend_for_job", return_value=backend
            ),
            pytest.raises(PtqError, match="No worktree"),
        ):
            rebase(repo, job_id)


class TestRebaseConflictResolution:
    def _setup(self, tmp_path):
        repo, job_id = _make_repo(tmp_path)
        state = {"agent_ran": False}

        def run_side(cmd, check=True):
            if "rebase-merge" in cmd or "rebase-apply" in cmd:
                if state["agent_ran"]:
                    return _fail()
                return _ok()
            if "test -d" in cmd or "test -f" in cmd:
                return _ok()
            if "rev-parse HEAD" in cmd:
                return _ok("abc123\n")
            if "rev-parse --verify" in cmd:
                return _ok("def456\n")
            if "fetch origin" in cmd:
                return _ok()
            if "rebase --continue" in cmd:
                return _ok()
            if "rebase" in cmd and "fetch" not in cmd:
                return _fail("CONFLICT")
            if "diff --name-only --diff-filter=U" in cmd:
                if not state["agent_ran"]:
                    return _ok("file.py\n")
                return _ok()
            return _ok()

        def fake_launch_bg(cmd, log_file):
            state["agent_ran"] = True
            return 12345

        backend = MagicMock()
        backend.workspace = "~/ptq_workspace"
        backend.run = MagicMock(side_effect=run_side)
        backend.is_pid_alive = MagicMock(return_value=False)
        backend.launch_background = MagicMock(side_effect=fake_launch_bg)
        backend.copy_to = MagicMock()

        return repo, job_id, backend, state

    def test_conflict_resolved_in_one_attempt(self, tmp_path):
        repo, job_id, backend, _ = self._setup(tmp_path)

        with patch(
            "ptq.application.rebase_service.backend_for_job", return_value=backend
        ):
            result = rebase(repo, job_id, max_attempts=3)

        assert result.state == RebaseState.SUCCEEDED
        assert result.attempts == 1
        assert backend.launch_background.call_count == 1

    def test_codex_rebase_uses_rebase_prompt_for_agents_file(self, tmp_path):
        repo, job_id, backend, _ = self._setup(tmp_path)

        with patch(
            "ptq.application.rebase_service.backend_for_job", return_value=backend
        ):
            rebase(repo, job_id, agent_name="codex", max_attempts=1)

        run_cmds = [
            call.args[0]
            for call in backend.run.call_args_list
            if isinstance(call.args[0], str)
        ]
        assert any(
            cmd == f"cp ~/ptq_workspace/jobs/{job_id}/rebase_prompt_1.md "
            f"~/ptq_workspace/jobs/{job_id}/pytorch/AGENTS.md"
            for cmd in run_cmds
        )

    def test_escalates_after_max_attempts(self, tmp_path):
        repo, job_id = _make_repo(tmp_path)

        def run_side(cmd, check=True):
            if "rebase-merge" in cmd or "rebase-apply" in cmd:
                return _ok()
            if "test -d" in cmd or "test -f" in cmd:
                return _ok()
            if "rev-parse HEAD" in cmd:
                return _ok("abc123\n")
            if "rev-parse --verify" in cmd:
                return _ok("def456\n")
            if "fetch origin" in cmd:
                return _ok()
            if "rebase --continue" in cmd:
                return _fail("conflicts remain")
            if "rebase" in cmd and "fetch" not in cmd:
                return _fail("CONFLICT")
            if "diff --name-only --diff-filter=U" in cmd:
                return _ok("file.py\n")
            return _ok()

        backend = MagicMock()
        backend.workspace = "~/ptq_workspace"
        backend.run = MagicMock(side_effect=run_side)
        backend.is_pid_alive = MagicMock(return_value=False)
        backend.launch_background = MagicMock(return_value=12345)
        backend.copy_to = MagicMock()

        with patch(
            "ptq.application.rebase_service.backend_for_job", return_value=backend
        ):
            result = rebase(repo, job_id, max_attempts=2)

        assert result.state == RebaseState.NEEDS_HUMAN
        assert result.attempts == 2
        assert "file.py" in result.error
        assert backend.launch_background.call_count == 2

        job = repo.get(job_id)
        assert job.rebase.state == RebaseState.NEEDS_HUMAN


class TestRebaseCLI:
    def test_rebase_command_help(self):
        from typer.testing import CliRunner

        from ptq.cli import app

        result = CliRunner().invoke(app, ["rebase", "--help"])
        assert result.exit_code == 0
        assert "--onto" in result.output
        assert "--max-attempts" in result.output

    def test_rebase_command_success(self, tmp_path):
        from typer.testing import CliRunner

        from ptq.cli import app

        repo = JobRepository(tmp_path / "jobs.json")
        repo.save(
            JobRecord(
                job_id="20260217-42",
                issue=42,
                machine="gpu-dev",
                workspace="~/ws",
            )
        )

        mock_result = RebaseInfo(
            state=RebaseState.SUCCEEDED,
            before_sha="aaa111",
            after_sha="bbb222",
        )

        with (
            patch("ptq.cli._repo", return_value=repo),
            patch("ptq.application.rebase_service.rebase", return_value=mock_result),
        ):
            result = CliRunner().invoke(app, ["rebase", "20260217-42"])

        assert result.exit_code == 0
        assert "Rebase complete" in result.output

    def test_rebase_command_needs_human(self, tmp_path):
        from typer.testing import CliRunner

        from ptq.cli import app

        repo = JobRepository(tmp_path / "jobs.json")
        repo.save(
            JobRecord(
                job_id="20260217-42",
                issue=42,
                machine="gpu-dev",
                workspace="~/ws",
            )
        )

        mock_result = RebaseInfo(
            state=RebaseState.NEEDS_HUMAN,
            error="file.py has unresolved conflicts",
        )

        with (
            patch("ptq.cli._repo", return_value=repo),
            patch("ptq.application.rebase_service.rebase", return_value=mock_result),
        ):
            result = CliRunner().invoke(app, ["rebase", "42"])

        assert result.exit_code == 0
        assert "human intervention" in result.output


class TestRebaseWeb:
    @pytest.fixture()
    def client(self, tmp_path):
        from fastapi.testclient import TestClient

        from ptq.config import AgentModels, Config
        from ptq.web.app import create_app

        repo = JobRepository(tmp_path / "jobs.json")
        repo.save(
            JobRecord(
                job_id="20260217-100001",
                issue=100001,
                runs=2,
                machine="gpu-dev",
                workspace="~/ptq_workspace",
                agent="claude",
                pid=12345,
            )
        )

        mock_backend = MagicMock()
        mock_backend.workspace = "~/ptq_workspace"
        mock_backend.is_pid_alive.return_value = False

        cfg = Config(
            default_agent="claude",
            default_model="opus",
            default_max_turns=100,
            machines=["gpu-dev"],
            agent_models={
                "claude": AgentModels(available=[], default="opus"),
            },
        )

        with (
            patch("ptq.web.routes._repo", return_value=repo),
            patch("ptq.web.deps.JobRepository", return_value=repo),
            patch("ptq.web.deps.backend_for_job", return_value=mock_backend),
            patch("ptq.web.routes.backend_for_job", return_value=mock_backend),
            patch("ptq.web.routes.load_config", return_value=cfg),
        ):
            test_client = TestClient(create_app())
            test_client.repo = repo
            yield test_client

    def test_job_detail_has_rebase_button(self, client):
        resp = client.get("/jobs/20260217-100001")
        assert resp.status_code == 200
        assert "Rebase" in resp.text

    def test_rebase_post_redirects(self, client):
        mock_result = RebaseInfo(state=RebaseState.SUCCEEDED, after_sha="abc")
        with patch("ptq.application.rebase_service.rebase", return_value=mock_result):
            resp = client.post(
                "/jobs/20260217-100001/rebase",
                data={"target_ref": "origin/main", "max_attempts": "2"},
                follow_redirects=False,
            )
        assert resp.status_code == 303
        assert "/jobs/rebasing/" in resp.headers["location"]

    def test_rebase_progress_missing(self, client):
        resp = client.get("/jobs/rebasing/doesnotexist/progress")
        assert resp.status_code == 200
        assert "not found" in resp.text

    def test_rebase_success_banner_clears_after_first_render(self, client):
        client.repo.save_rebase(
            "20260217-100001",
            RebaseInfo(
                state=RebaseState.SUCCEEDED,
                target_ref="origin/main",
                before_sha="4d5536d435",
                after_sha="977b4d4cb7",
            ).to_dict(),
        )

        first = client.get("/jobs/20260217-100001")
        assert first.status_code == 200
        assert (
            "Rebased onto <code>origin/main</code> — 4d5536d435 → 977b4d4cb7"
            in first.text
        )

        second = client.get("/jobs/20260217-100001")
        assert second.status_code == 200
        assert (
            "Rebased onto <code>origin/main</code> — 4d5536d435 → 977b4d4cb7"
            not in second.text
        )
