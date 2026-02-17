from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ptq.workspace import detect_cuda_version


def _make_backend(stdout: str = "", returncode: int = 0) -> MagicMock:
    backend = MagicMock()
    result = MagicMock()
    result.stdout = stdout
    result.returncode = returncode
    backend.run.return_value = result
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
