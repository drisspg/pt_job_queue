from __future__ import annotations

from subprocess import CompletedProcess
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from ptq.application.rebase_service import rebase
from ptq.cli import app
from ptq.domain.models import JobRecord, PtqError, RebaseInfo, RebaseState, StackTool
from ptq.infrastructure.job_repository import JobRepository


def _make_repo(tmp_path) -> tuple[JobRepository, str]:
    repo = JobRepository(tmp_path / "jobs.json")
    job_id = "20260217-42"
    repo.save(
        JobRecord(
            job_id=job_id,
            issue=42,
            workspace="~/ptq_workspace",
        )
    )
    return repo, job_id


def _ok(stdout: str = "") -> CompletedProcess[str]:
    return CompletedProcess("", 0, stdout, "")


def _fail(stderr: str = "") -> CompletedProcess[str]:
    return CompletedProcess("", 1, "", stderr)


class TestRebaseInfo:
    def test_idle_to_dict_empty(self):
        assert RebaseInfo().to_dict() == {}

    def test_roundtrip(self):
        info = RebaseInfo(
            state=RebaseState.SUCCEEDED,
            target_ref="origin/main",
            before_sha="aaa",
            after_sha="bbb",
        )
        assert RebaseInfo.from_dict(info.to_dict()) == info

    def test_job_roundtrip(self):
        info = RebaseInfo(
            state=RebaseState.NEEDS_HUMAN,
            target_ref="origin/main",
            error="conflicts remain",
        )
        restored = JobRecord.from_dict(
            "j", JobRecord(job_id="j", rebase=info).to_dict()
        )
        assert restored.rebase == info

    def test_repository_save_and_clear(self, tmp_path):
        repo = JobRepository(tmp_path / "jobs.json")
        repo.save(JobRecord(job_id="j1"))
        repo.save_rebase("j1", {"state": "running", "target_ref": "origin/main"})
        assert repo.get("j1").rebase_info.state == RebaseState.RUNNING
        repo.save_rebase("j1", {})
        assert repo.get("j1").rebase is None


class TestRebase:
    def test_clean_rebase(self, tmp_path):
        repo, job_id = _make_repo(tmp_path)

        def run_side(command, check=True):
            if "rev-parse HEAD" in command:
                return _ok("abc123\n")
            return _ok()

        backend = MagicMock(workspace="~/ptq_workspace")
        backend.run.side_effect = run_side
        with patch(
            "ptq.application.rebase_service.backend_for_job", return_value=backend
        ):
            result = rebase(repo, job_id)

        assert result.state == RebaseState.SUCCEEDED
        assert result.before_sha == "abc123"
        assert repo.get(job_id).rebase_info.state == RebaseState.SUCCEEDED

    @pytest.mark.parametrize(
        ("pr_state", "rebase_command", "stack_base", "base_pr_url"),
        [
            ("open", "rebase origin/ptq/42", "ptq/42", "https://x/pull/99"),
            ("merged", "rebase --onto origin/main origin/ptq/42", "main", None),
        ],
    )
    def test_stack_on_conventional_pr_follows_its_base(
        self, tmp_path, pr_state, rebase_command, stack_base, base_pr_url
    ):
        repo, job_id = _make_repo(tmp_path)
        job = repo.get(job_id)
        job.stack_base = "ptq/42"
        job.stack_base_pr_url = "https://x/pull/99"
        repo.save(job)

        def run_side(command, check=True):
            if "rev-parse HEAD" in command:
                return _ok("abc123\n")
            return _ok()

        backend = MagicMock(workspace="~/ptq_workspace")
        backend.run.side_effect = run_side
        with (
            patch(
                "ptq.application.rebase_service.backend_for_job", return_value=backend
            ),
            patch("ptq.application.rebase_service.get_pr_state", return_value=pr_state),
        ):
            result = rebase(repo, job_id)

        assert result.state == RebaseState.SUCCEEDED
        commands = [call.args[0] for call in backend.run.call_args_list]
        assert any(command.endswith(rebase_command) for command in commands)
        saved = repo.get(job_id)
        assert saved.stack_base == stack_base
        assert saved.stack_base_pr_url == base_pr_url

    def test_sp_stack_rebases_with_sp_sync(self, tmp_path):
        repo, job_id = _make_repo(tmp_path)
        job = repo.get(job_id)
        job.stack_tool = StackTool.SP
        repo.save(job)
        backend = MagicMock(workspace="~/ptq_workspace")
        backend.run.side_effect = lambda command, check=True: _ok("abc123\n")
        with patch(
            "ptq.application.rebase_service.backend_for_job", return_value=backend
        ):
            result = rebase(repo, job_id)

        assert result.state == RebaseState.SUCCEEDED
        commands = [call.args[0] for call in backend.run.call_args_list]
        assert any(command.endswith("&& sp sync -T main") for command in commands)
        assert not any(" rebase " in command for command in commands)

    def test_stack_on_closed_conventional_pr_needs_a_human(self, tmp_path):
        repo, job_id = _make_repo(tmp_path)
        job = repo.get(job_id)
        job.stack_base_pr_url = "https://x/pull/99"
        repo.save(job)
        backend = MagicMock(workspace="~/ptq_workspace")
        with (
            patch(
                "ptq.application.rebase_service.backend_for_job", return_value=backend
            ),
            patch("ptq.application.rebase_service.get_pr_state", return_value="closed"),
            pytest.raises(PtqError, match="is closed"),
        ):
            rebase(repo, job_id)

    def test_target_not_found(self, tmp_path):
        repo, job_id = _make_repo(tmp_path)

        def run_side(command, check=True):
            if "rev-parse HEAD" in command:
                return _ok("abc123\n")
            if "rev-parse --verify" in command:
                return _fail("not found")
            return _ok()

        backend = MagicMock(workspace="~/ptq_workspace")
        backend.run.side_effect = run_side
        with (
            patch(
                "ptq.application.rebase_service.backend_for_job", return_value=backend
            ),
            pytest.raises(PtqError, match="Target ref not found"),
        ):
            rebase(repo, job_id)
        assert repo.get(job_id).rebase_info.state == RebaseState.FAILED

    def test_no_worktree(self, tmp_path):
        repo, job_id = _make_repo(tmp_path)
        backend = MagicMock(workspace="~/ptq_workspace")
        backend.run.return_value = _fail()
        with (
            patch(
                "ptq.application.rebase_service.backend_for_job", return_value=backend
            ),
            pytest.raises(PtqError, match="No worktree"),
        ):
            rebase(repo, job_id)

    def test_leaves_conflicts_for_interactive_resolution(self, tmp_path):
        repo, job_id = _make_repo(tmp_path)

        def run_side(command, check=True):
            if "diff --name-only --diff-filter=U" in command:
                return _ok("file.py\n")
            if "rebase-merge" in command or "rebase-apply" in command:
                return _ok()
            if "rev-parse HEAD" in command:
                return _ok("abc123\n")
            if " rebase origin/main" in command:
                return _fail("CONFLICT")
            return _ok()

        backend = MagicMock(workspace="~/ptq_workspace")
        backend.run.side_effect = run_side
        with patch(
            "ptq.application.rebase_service.backend_for_job", return_value=backend
        ):
            result = rebase(repo, job_id)

        assert result.state == RebaseState.NEEDS_HUMAN
        assert "file.py" in result.error
        assert "Resolve interactively" in result.error

    def test_non_conflict_failure_raises(self, tmp_path):
        repo, job_id = _make_repo(tmp_path)

        def run_side(command, check=True):
            if "rebase-merge" in command or "rebase-apply" in command:
                return _fail()
            if "diff --name-only --diff-filter=U" in command:
                return _ok()
            if " rebase origin/main" in command:
                return _fail("fatal error")
            return _ok("abc123\n")

        backend = MagicMock(workspace="~/ptq_workspace")
        backend.run.side_effect = run_side
        with (
            patch(
                "ptq.application.rebase_service.backend_for_job", return_value=backend
            ),
            pytest.raises(PtqError, match="fatal error"),
        ):
            rebase(repo, job_id)
        assert any(
            "rev-parse --absolute-git-dir" in call.args[0]
            for call in backend.run.call_args_list
        )


class TestRebaseCLI:
    def test_help_has_no_agent_options(self):
        result = CliRunner().invoke(app, ["rebase", "--help"])
        assert result.exit_code == 0
        assert "--onto" in result.output
        assert "--agent" not in result.output
        assert "--max-attempts" not in result.output

    def test_success(self, tmp_path):
        repo, job_id = _make_repo(tmp_path)
        result_info = RebaseInfo(
            state=RebaseState.SUCCEEDED,
            before_sha="aaa111",
            after_sha="bbb222",
        )
        with (
            patch("ptq.cli._repo", return_value=repo),
            patch("ptq.application.rebase_service.rebase", return_value=result_info),
        ):
            result = CliRunner().invoke(app, ["rebase", job_id])
        assert result.exit_code == 0
        assert "Rebase complete" in result.output

    def test_conflict_points_to_takeover(self, tmp_path):
        repo, _ = _make_repo(tmp_path)
        result_info = RebaseInfo(
            state=RebaseState.NEEDS_HUMAN,
            error="Resolve interactively in the job workspace: file.py",
        )
        with (
            patch("ptq.cli._repo", return_value=repo),
            patch("ptq.application.rebase_service.rebase", return_value=result_info),
        ):
            result = CliRunner().invoke(app, ["rebase", "42"])
        assert result.exit_code == 0
        assert "human intervention" in result.output
        assert "ptq_workspace/jobs/20260217-42" in result.output
