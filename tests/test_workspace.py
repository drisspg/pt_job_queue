from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ptq.workspace import detect_cuda_version, setup_workspace


def _make_backend(stdout: str = "", returncode: int = 0) -> MagicMock:
    backend = MagicMock()
    result = MagicMock()
    result.stdout = stdout
    result.returncode = returncode
    backend.run.return_value = result
    return backend


def _setup_backend(workspace: str, existing_checkout: bool = True) -> MagicMock:
    backend = MagicMock()
    backend.workspace = workspace

    def run_side(cmd: str, check: bool = True, **kw):
        if cmd == f"test -d {workspace}/pytorch/.git":
            return MagicMock(returncode=0 if existing_checkout else 1, stdout="")
        return MagicMock(returncode=0, stdout="")

    backend.run = MagicMock(side_effect=run_side)
    return backend


class TestDetectCudaVersion:
    def test_cuda_13_0(self):
        backend = _make_backend(
            "NVIDIA-SMI 570.00  Driver Version: 570.00  CUDA Version: 13.0"
        )
        assert detect_cuda_version(backend) == "cu130"

    def test_cuda_12_6(self):
        backend = _make_backend(
            "NVIDIA-SMI 560.00  Driver Version: 560.00  CUDA Version: 12.6"
        )
        assert detect_cuda_version(backend) == "cu126"

    def test_cuda_12_7_rounds_down(self):
        backend = _make_backend(
            "NVIDIA-SMI 560.00  Driver Version: 560.00  CUDA Version: 12.7"
        )
        assert detect_cuda_version(backend) == "cu126"

    def test_nvidia_smi_fails(self):
        backend = _make_backend(returncode=1)
        with pytest.raises(SystemExit, match="nvidia-smi not found"):
            detect_cuda_version(backend)

    def test_unparseable_output(self):
        backend = _make_backend("some garbage output")
        with pytest.raises(SystemExit, match="Could not parse"):
            detect_cuda_version(backend)

    def test_version_too_old(self):
        backend = _make_backend(
            "NVIDIA-SMI 400.00  Driver Version: 400.00  CUDA Version: 10.0"
        )
        with pytest.raises(SystemExit, match="too old"):
            detect_cuda_version(backend)


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
        assert any("git reset --hard origin/main" in cmd for cmd in cmds)
