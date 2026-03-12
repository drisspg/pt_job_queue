from __future__ import annotations

from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from ptq.application.run_service import launch
from ptq.cli import app
from ptq.domain.models import JobRecord, RunRequest
from ptq.infrastructure.job_repository import JobRepository
from ptq.ssh import LocalBackend

runner = CliRunner()


def _make_repo(tmp_path: Path, records: list[JobRecord] | None = None) -> JobRepository:
    repo = JobRepository(tmp_path / "jobs.json")
    for r in records or []:
        repo.save(r)
    return repo


def _ok(*args, **kwargs) -> CompletedProcess[str]:
    return CompletedProcess(args="", returncode=0, stdout="", stderr="")


def _mock_backend(backend: LocalBackend, *, worktree_exists: bool = False) -> None:
    def run_side_effect(cmd: str, check: bool = True, **kw) -> CompletedProcess[str]:
        if "test -d" in cmd or "test -f" in cmd:
            rc = 0 if worktree_exists else 1
            return CompletedProcess(args="", returncode=rc, stdout="", stderr="")
        return _ok()

    backend.run = MagicMock(side_effect=run_side_effect)
    backend.copy_to = MagicMock()
    backend.launch_background = MagicMock(return_value=12345)
    backend.tail_log = MagicMock()


class TestWorktreeCommand:
    def test_creates_job_record_with_name(self, tmp_path, frozen_date):
        repo = _make_repo(tmp_path)
        mock_backend = MagicMock()
        mock_backend.workspace = "/tmp/ws"
        mock_backend.run = MagicMock(return_value=_ok())

        with (
            patch("ptq.cli._repo", return_value=repo),
            patch(
                "ptq.infrastructure.backends.LocalBackend", return_value=mock_backend
            ),
            patch("ptq.config.load_config") as mock_cfg,
        ):
            mock_cfg.return_value.build_env_prefix.return_value = "USE_NINJA=1 "
            result = runner.invoke(app, ["worktree", "flex-attn"])

        assert result.exit_code == 0, result.output
        assert "flex-attn" in result.output
        assert "ready" in result.output.lower()

        all_jobs = repo.list_all()
        assert len(all_jobs) == 1
        job = list(all_jobs.values())[0]
        assert job.name == "flex-attn"
        assert job.runs == 0

    def test_defaults_to_local(self, tmp_path, frozen_date):
        repo = _make_repo(tmp_path)
        mock_backend = MagicMock()
        mock_backend.workspace = "/tmp/ws"
        mock_backend.run = MagicMock(return_value=_ok())

        with (
            patch("ptq.cli._repo", return_value=repo),
            patch(
                "ptq.infrastructure.backends.LocalBackend", return_value=mock_backend
            ),
            patch("ptq.config.load_config") as mock_cfg,
        ):
            mock_cfg.return_value.build_env_prefix.return_value = "USE_NINJA=1 "
            result = runner.invoke(app, ["worktree", "my-fix"])

        assert result.exit_code == 0, result.output
        job = list(repo.list_all().values())[0]
        assert job.local is True

    def test_rejects_duplicate_name(self, tmp_path, frozen_date):
        repo = _make_repo(
            tmp_path,
            [
                JobRecord(
                    job_id="20260217-adhoc-abc123",
                    name="flex-attn",
                    local=True,
                    workspace="/tmp/ws",
                ),
            ],
        )
        with patch("ptq.cli._repo", return_value=repo):
            result = runner.invoke(app, ["worktree", "flex-attn"])

        assert result.exit_code == 1
        assert "already exists" in result.output

    def test_prints_enter_command_local(self, tmp_path, frozen_date):
        repo = _make_repo(tmp_path)
        mock_backend = MagicMock()
        mock_backend.workspace = "/tmp/ws"
        mock_backend.run = MagicMock(return_value=_ok())

        with (
            patch("ptq.cli._repo", return_value=repo),
            patch(
                "ptq.infrastructure.backends.LocalBackend", return_value=mock_backend
            ),
            patch("ptq.config.load_config") as mock_cfg,
        ):
            mock_cfg.return_value.build_env_prefix.return_value = "USE_NINJA=1 "
            result = runner.invoke(app, ["worktree", "my-fix", "--local"])

        assert result.exit_code == 0, result.output
        assert "source ../.venv/bin/activate" in result.output
        assert "ptq run my-fix" in result.output

    def test_prints_ssh_command_remote(self, tmp_path, frozen_date):
        repo = _make_repo(tmp_path)
        mock_backend = MagicMock()
        mock_backend.workspace = "/tmp/ws"
        mock_backend.run = MagicMock(return_value=_ok())

        with (
            patch("ptq.cli._repo", return_value=repo),
            patch(
                "ptq.infrastructure.backends.RemoteBackend", return_value=mock_backend
            ),
            patch("ptq.config.load_config") as mock_cfg,
        ):
            mock_cfg.return_value.build_env_prefix.return_value = "USE_NINJA=1 "
            result = runner.invoke(app, ["worktree", "my-fix", "--machine", "gpu-dev"])

        assert result.exit_code == 0, result.output
        assert "ssh -t gpu-dev" in result.output

    def test_no_agent_launched(self, tmp_path, frozen_date):
        repo = _make_repo(tmp_path)
        mock_backend = MagicMock()
        mock_backend.workspace = "/tmp/ws"
        mock_backend.run = MagicMock(return_value=_ok())
        mock_backend.launch_background = MagicMock()

        with (
            patch("ptq.cli._repo", return_value=repo),
            patch(
                "ptq.infrastructure.backends.LocalBackend", return_value=mock_backend
            ),
            patch("ptq.config.load_config") as mock_cfg,
        ):
            mock_cfg.return_value.build_env_prefix.return_value = "USE_NINJA=1 "
            result = runner.invoke(app, ["worktree", "my-fix"])

        assert result.exit_code == 0, result.output
        mock_backend.launch_background.assert_not_called()

    def test_record_created_before_worktree(self, tmp_path, frozen_date):
        repo = _make_repo(tmp_path)
        mock_backend = MagicMock()
        mock_backend.workspace = "/tmp/ws"

        call_order: list[str] = []
        original_save = repo.save

        def tracked_save(record):
            call_order.append("save")
            original_save(record)

        def run_side(cmd: str, check: bool = True, **kw) -> CompletedProcess[str]:
            if "create_worktree.py" in cmd:
                call_order.append("create_worktree")
            if "pytorch/.git" in cmd and "jobs" not in cmd:
                return _ok()
            if "test -d" in cmd or "test -f" in cmd:
                return CompletedProcess(args="", returncode=1, stdout="", stderr="")
            return _ok()

        mock_backend.run = MagicMock(side_effect=run_side)

        with (
            patch("ptq.cli._repo", return_value=repo),
            patch.object(repo, "save", side_effect=tracked_save),
            patch(
                "ptq.infrastructure.backends.LocalBackend", return_value=mock_backend
            ),
            patch("ptq.config.load_config") as mock_cfg,
        ):
            mock_cfg.return_value.build_env_prefix.return_value = "USE_NINJA=1 "
            result = runner.invoke(app, ["worktree", "my-fix"])

        assert result.exit_code == 0, result.output
        assert "save" in call_order
        assert "create_worktree" in call_order
        assert call_order.index("save") < call_order.index("create_worktree")


class TestWorktreeReuse:
    @patch("ptq.application.run_service.deploy_scripts")
    def test_run_adopts_precreated_worktree(self, _deploy, repo, frozen_date):
        backend = LocalBackend(workspace="/tmp/ws")
        _mock_backend(backend, worktree_exists=True)

        repo.save(
            JobRecord(
                job_id="20260217-adhoc-aaaaaa",
                runs=0,
                agent="claude",
                model="",
                local=True,
                workspace="/tmp/ws",
                name="flex-attn",
            )
        )

        launch(
            repo,
            backend,
            RunRequest(
                message="optimize codegen",
                local=True,
                follow=False,
                existing_job_id="20260217-adhoc-aaaaaa",
            ),
        )

        job = repo.get("20260217-adhoc-aaaaaa")
        assert job.runs == 1
        assert job.name == "flex-attn"
        backend.launch_background.assert_called_once()

    @patch("ptq.application.run_service.deploy_scripts")
    def test_run_reuses_existing_worktree_no_rebuild(self, _deploy, repo, frozen_date):
        backend = LocalBackend(workspace="/tmp/ws")
        _mock_backend(backend, worktree_exists=True)

        repo.save(
            JobRecord(
                job_id="20260217-adhoc-bbbbbb",
                runs=0,
                local=True,
                workspace="/tmp/ws",
                name="my-fix",
            )
        )

        launch(
            repo,
            backend,
            RunRequest(
                message="do the thing",
                local=True,
                follow=False,
                existing_job_id="20260217-adhoc-bbbbbb",
            ),
        )

        run_cmds = [
            call.args[0]
            for call in backend.run.call_args_list
            if isinstance(call.args[0], str)
        ]
        assert not any("create_worktree.py" in c for c in run_cmds)


class TestNameResolution:
    def test_resolve_id_by_name(self, tmp_path):
        repo = _make_repo(
            tmp_path,
            [
                JobRecord(
                    job_id="20260217-adhoc-abc123",
                    name="flex-attn",
                    local=True,
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
                    local=True,
                    workspace="/tmp/ws",
                ),
                JobRecord(
                    job_id="20260217-adhoc-xyz789",
                    name="flex-attn",
                    local=True,
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
                    local=True,
                    workspace="/tmp/ws",
                ),
                JobRecord(
                    job_id="20260217-adhoc-xyz789",
                    name="other",
                    local=True,
                    workspace="/tmp/ws",
                ),
            ],
        )
        assert repo.find_by_name("flex-attn") == "20260217-adhoc-abc123"
        assert repo.find_by_name("other") == "20260217-adhoc-xyz789"
        assert repo.find_by_name("nonexistent") is None


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
