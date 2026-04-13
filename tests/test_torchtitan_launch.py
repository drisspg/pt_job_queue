"""Tests for torchtitan launch: worktree creation, venv setup, repo persistence."""

from __future__ import annotations

from subprocess import CompletedProcess
from unittest.mock import MagicMock, patch

from ptq.application.run_service import launch
from ptq.domain.models import RunRequest
from ptq.ssh import LocalBackend


def _ok(cmd="", **kwargs):
    return CompletedProcess(args="", returncode=0, stdout="", stderr="")


def _mock_backend(backend):
    def run_side_effect(cmd: str, check: bool = True, **kw):
        if "test -d" in cmd or "test -f" in cmd:
            return CompletedProcess(args="", returncode=1, stdout="", stderr="")
        return _ok()

    backend.run = MagicMock(side_effect=run_side_effect)
    backend.copy_to = MagicMock()
    backend.launch_background = MagicMock(return_value=12345)
    backend.tail_log = MagicMock()


class TestLaunchTorchtitan:
    @patch("ptq.application.run_service.deploy_scripts")
    def test_torchtitan_uses_git_worktree(self, _deploy, repo, frozen_date):
        """Torchtitan should use standard git worktree, not create_worktree.py."""
        backend = LocalBackend(workspace="/tmp/ws")
        _mock_backend(backend)

        launch(
            repo,
            backend,
            RunRequest(message="hello", local=True, follow=False, repo="torchtitan"),
        )

        run_cmds = [
            call.args[0]
            for call in backend.run.call_args_list
            if isinstance(call.args[0], str)
        ]
        assert not any("create_worktree.py" in c for c in run_cmds)
        assert any("git worktree add" in c for c in run_cmds)

    @patch("ptq.application.run_service.deploy_scripts")
    def test_torchtitan_repo_persisted(self, _deploy, repo, frozen_date):
        backend = LocalBackend(workspace="/tmp/ws")
        _mock_backend(backend)

        job_id = launch(
            repo,
            backend,
            RunRequest(message="hello", local=True, follow=False, repo="torchtitan"),
        )

        job = repo.get(job_id)
        assert job.repo == "torchtitan"

    @patch("ptq.application.run_service.deploy_scripts")
    def test_torchtitan_worktree_path(self, _deploy, repo, frozen_date):
        """Worktree should be under torchtitan/, not pytorch/."""
        backend = LocalBackend(workspace="/tmp/ws")
        _mock_backend(backend)

        job_id = launch(
            repo,
            backend,
            RunRequest(message="hello", local=True, follow=False, repo="torchtitan"),
        )

        run_cmds = [
            call.args[0]
            for call in backend.run.call_args_list
            if isinstance(call.args[0], str)
        ]
        worktree_cmds = [c for c in run_cmds if "git worktree add" in c]
        assert any(f"/jobs/{job_id}/torchtitan" in c for c in worktree_cmds)

    @patch("ptq.application.run_service.deploy_scripts")
    def test_pytorch_still_uses_create_worktree(self, _deploy, repo, frozen_date):
        """Pytorch should still use create_worktree.py."""
        backend = LocalBackend(workspace="/tmp/ws")
        _mock_backend(backend)

        launch(
            repo,
            backend,
            RunRequest(message="hello", local=True, follow=False, repo="pytorch"),
        )

        run_cmds = [
            call.args[0]
            for call in backend.run.call_args_list
            if isinstance(call.args[0], str)
        ]
        assert any("create_worktree.py" in c for c in run_cmds)
