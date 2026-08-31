from __future__ import annotations

from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import MagicMock, patch

import pytest

from ptq.domain.models import JobRecord, PtqError
from ptq.infrastructure.job_repository import JobRepository


def _make_repo(tmp_path: Path, records: list[JobRecord] | None = None) -> JobRepository:
    repo = JobRepository(tmp_path / "jobs.json")
    for r in records or []:
        repo.save(r)
    return repo


def _ok(*args, **kwargs) -> CompletedProcess[str]:
    return CompletedProcess(args="", returncode=0, stdout="", stderr="")


class TestJobContext:
    def test_writes_prime_and_auto_discovered_agents_file(self):
        from ptq.application.job_context import write_job_context

        backend = MagicMock()
        write_job_context(
            backend,
            job_id="job-1",
            workspace="/tmp/ws",
            name="example",
        )

        commands = [call.args[0] for call in backend.run.call_args_list]
        assert any("/job-1/prime.md" in command for command in commands)
        assert any(
            command == "cp /tmp/ws/jobs/job-1/prime.md /tmp/ws/jobs/job-1/AGENTS.md"
            for command in commands
        )
        cleanup = next(command for command in commands if command.startswith("rm -f"))
        assert "PTQ_CONTEXT.md" in cleanup
        assert "CLAUDE.md" in cleanup
        assert "/agent_space/prime.md" in cleanup


class TestNameResolution:
    def test_resolve_id_by_name(self, tmp_path):
        repo = _make_repo(
            tmp_path,
            [
                JobRecord(
                    job_id="20260217-adhoc-abc123",
                    name="flex-attn",
                    workspace="/tmp/ws",
                ),
            ],
        )
        assert repo.resolve_id("flex-attn") == "20260217-adhoc-abc123"

    def test_resolve_id_prefers_exact_job_id(self, tmp_path):
        repo = _make_repo(
            tmp_path,
            [
                JobRecord(
                    job_id="flex-attn",
                    workspace="/tmp/ws",
                ),
                JobRecord(
                    job_id="20260217-adhoc-xyz789",
                    name="flex-attn",
                    workspace="/tmp/ws",
                ),
            ],
        )
        assert repo.resolve_id("flex-attn") == "flex-attn"

    def test_find_by_name(self, tmp_path):
        repo = _make_repo(
            tmp_path,
            [
                JobRecord(
                    job_id="20260217-adhoc-abc123",
                    name="flex-attn",
                    workspace="/tmp/ws",
                ),
                JobRecord(
                    job_id="20260217-adhoc-xyz789",
                    name="other",
                    workspace="/tmp/ws",
                ),
            ],
        )
        assert repo.find_by_name("flex-attn") == "20260217-adhoc-abc123"
        assert repo.find_by_name("other") == "20260217-adhoc-xyz789"
        assert repo.find_by_name("nonexistent") is None


class TestEnsureJobWorktree:
    def test_creates_issue_job_without_launching_agent(self, tmp_path, frozen_date):
        from ptq.application.worktree_service import ensure_job_worktree

        repo = _make_repo(tmp_path)
        backend = MagicMock(workspace="/tmp/ws")
        backend.run.return_value = _ok()

        with (
            patch("ptq.application.worktree_service.deploy_scripts"),
            patch("ptq.application.worktree_service.provision_worktree") as provision,
            patch(
                "ptq.application.worktree_service.write_job_context"
            ) as write_context,
        ):
            job_id, created = ensure_job_worktree(
                repo,
                backend,
                issue_number=123,
            )

        assert created is True
        job = repo.get(job_id)
        assert job.issue == 123
        provision.assert_called_once()
        write_context.assert_called_once()

    def test_reuses_existing_issue_job(self, tmp_path):
        from ptq.application.worktree_service import ensure_job_worktree

        repo = _make_repo(
            tmp_path,
            [
                JobRecord(
                    job_id="existing-job",
                    issue=123,
                    workspace="/tmp/ws",
                )
            ],
        )
        backend = MagicMock(workspace="/tmp/ws")

        with patch("ptq.application.worktree_service.prepare_job_worktree") as prepare:
            job_id, created = ensure_job_worktree(
                repo,
                backend,
                issue_number=123,
            )

        assert (job_id, created) == ("existing-job", False)
        prepare.assert_called_once()
        args, kwargs = prepare.call_args
        assert args[0].workspace == "/tmp/ws"
        assert args[1] == "existing-job"
        assert kwargs == {
            "name": None,
            "verbose": False,
            "progress": None,
            "repo": "pytorch",
        }

    def test_rejects_reuse_from_another_workspace(self, tmp_path):
        from ptq.application.worktree_service import ensure_job_worktree

        repo = _make_repo(
            tmp_path,
            [JobRecord(job_id="existing-job", issue=123, workspace="/other/ws")],
        )
        with pytest.raises(PtqError, match="already exists in /other/ws"):
            ensure_job_worktree(
                repo,
                MagicMock(workspace="/tmp/ws"),
                issue_number=123,
                workspace_explicit=True,
            )

    def test_removes_record_when_new_job_provisioning_fails(
        self, tmp_path, frozen_date
    ):
        from ptq.application.worktree_service import ensure_job_worktree

        repo = _make_repo(tmp_path)
        backend = MagicMock(workspace="/tmp/ws")
        backend.run.return_value = _ok()
        with (
            patch(
                "ptq.application.worktree_service.prepare_job_worktree",
                side_effect=PtqError("provision failed"),
            ),
            pytest.raises(PtqError, match="provision failed"),
        ):
            ensure_job_worktree(repo, backend, name="example")
        assert repo.list_all() == {}


class TestProvisionWorktree:
    def test_creates_worktree_and_venv(self, frozen_date):
        from ptq.application.worktree_service import provision_worktree

        backend = MagicMock()
        backend.workspace = "/tmp/ws"

        call_order: list[str] = []

        def run_side(cmd: str, check: bool = True, **kw) -> CompletedProcess[str]:
            if "test -d" in cmd or "test -f" in cmd:
                return CompletedProcess(args="", returncode=1, stdout="", stderr="")
            if "create_worktree.py" in cmd:
                call_order.append("create_worktree")
            return _ok()

        backend.run = MagicMock(side_effect=run_side)

        with patch("ptq.config.load_config") as mock_cfg:
            mock_cfg.return_value.build_env_prefix.return_value = "USE_NINJA=1 "
            created = provision_worktree(backend, "test-job")

        assert created is True
        assert "create_worktree" in call_order
        mkdir_cmds = [
            call.args[0]
            for call in backend.run.call_args_list
            if isinstance(call.args[0], str) and "mkdir -p" in call.args[0]
        ]
        assert any("/tmp/ws/jobs/test-job" in c for c in mkdir_cmds)

    def test_reuses_existing_worktree(self):
        from ptq.application.worktree_service import provision_worktree

        backend = MagicMock()
        backend.workspace = "/tmp/ws"

        def run_side(cmd: str, check: bool = True, **kw) -> CompletedProcess[str]:
            if "test -d" in cmd or "test -f" in cmd:
                return CompletedProcess(args="", returncode=0, stdout="", stderr="")
            return _ok()

        backend.run = MagicMock(side_effect=run_side)
        created = provision_worktree(backend, "test-job")

        assert created is False
        run_cmds = [
            call.args[0]
            for call in backend.run.call_args_list
            if isinstance(call.args[0], str)
        ]
        assert not any("create_worktree.py" in c for c in run_cmds)
