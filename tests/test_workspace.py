from __future__ import annotations

from unittest.mock import MagicMock

from ptq.application.venv_service import (
    PYTORCH_TEST_REQUIREMENTS,
    TRANSFORMER_NUGGETS_REQUIREMENT,
)
from ptq.workspace import setup_workspace


def _setup_backend(workspace: str, existing_checkout: bool = True) -> MagicMock:
    backend = MagicMock()
    backend.workspace = workspace

    def run_side(cmd: str, check: bool = True, **kw):
        if cmd == f"test -d {workspace}/pytorch/.git":
            return MagicMock(returncode=0 if existing_checkout else 1, stdout="")
        return MagicMock(returncode=0, stdout="")

    backend.run = MagicMock(side_effect=run_side)
    return backend


class TestSetupWorkspace:
    def test_existing_checkout_is_not_reset_without_build(self, tmp_path):
        backend = _setup_backend(str(tmp_path))
        setup_workspace(backend, build=False)

        cmds = [call.args[0] for call in backend.run.call_args_list]
        assert not any("git reset --hard origin/main" in cmd for cmd in cmds)

    def test_existing_checkout_is_reset_with_build(self, tmp_path):
        backend = _setup_backend(str(tmp_path))
        setup_workspace(backend, build=True)

        cmds = [call.args[0] for call in backend.run.call_args_list]
        assert any("git -C" in cmd and "fetch origin" in cmd for cmd in cmds)
        assert any(
            "git -C" in cmd and "reset --hard origin/main" in cmd for cmd in cmds
        )

    def test_existing_checkout_can_reset_to_custom_ref(self, tmp_path):
        backend = _setup_backend(str(tmp_path))
        setup_workspace(backend, build=True, target_ref="upstream/viable/strict")

        cmds = [call.args[0] for call in backend.run.call_args_list]
        assert any("remote add upstream" in cmd for cmd in cmds)
        assert any("git -C" in cmd and "fetch upstream" in cmd for cmd in cmds)
        assert any(
            "git -C" in cmd and "reset --hard upstream/viable/strict" in cmd
            for cmd in cmds
        )

    def test_installs_test_requirements_in_base_venv(self, tmp_path):
        backend = _setup_backend(str(tmp_path))
        setup_workspace(backend)

        cmds = [call.args[0] for call in backend.run.call_args_list]
        assert any(PYTORCH_TEST_REQUIREMENTS in cmd for cmd in cmds)

    def test_installs_transformer_nuggets_when_torch_is_available(self, tmp_path):
        backend = _setup_backend(str(tmp_path))
        setup_workspace(backend, build=True)

        cmds = [call.args[0] for call in backend.run.call_args_list]
        assert any(TRANSFORMER_NUGGETS_REQUIREMENT in cmd for cmd in cmds)
        assert any(
            f"uv pip install --python {tmp_path}/.venv/bin/python" in cmd
            and "--reinstall-package transformer_nuggets" in cmd
            for cmd in cmds
        )
