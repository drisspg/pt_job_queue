from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from ptq.results import fetch_results


def _make_job_entry(*, local: bool = True) -> dict:
    return {
        "issue": None,
        "runs": 1,
        "local": local,
        "workspace": "~/.ptq_workspace",
    }


class TestFetchResultsMissingArtifacts:
    def test_handles_missing_file_local_backend(self, jobs_db, tmp_path):
        jobs_db["j1"] = _make_job_entry()

        backend = MagicMock()
        backend.workspace = str(tmp_path / "workspace")

        call_count = 0

        def copy_side_effect(remote_path: str, local_path: Path) -> None:
            nonlocal call_count
            call_count += 1
            if "fix.diff" in remote_path:
                raise FileNotFoundError(remote_path)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_text("content")

        backend.copy_from.side_effect = copy_side_effect

        with patch("ptq.results.backend_for_job", return_value=backend):
            dest = fetch_results("j1", output_dir=tmp_path / "out")

        assert (dest / "report.md").exists()
        assert (dest / "worklog.md").exists()
        assert not (dest / "fix.diff").exists()

    def test_handles_called_process_error_remote_backend(self, jobs_db, tmp_path):
        import subprocess

        jobs_db["j1"] = _make_job_entry()

        backend = MagicMock()
        backend.workspace = str(tmp_path / "workspace")

        def copy_side_effect(remote_path: str, local_path: Path) -> None:
            if "fix.diff" in remote_path:
                raise subprocess.CalledProcessError(1, "scp")
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_text("content")

        backend.copy_from.side_effect = copy_side_effect

        with patch("ptq.results.backend_for_job", return_value=backend):
            dest = fetch_results("j1", output_dir=tmp_path / "out")

        assert not (dest / "fix.diff").exists()

    def test_all_artifacts_fetched(self, jobs_db, tmp_path):
        jobs_db["j1"] = _make_job_entry()

        backend = MagicMock()
        backend.workspace = str(tmp_path / "workspace")

        def copy_side_effect(remote_path: str, local_path: Path) -> None:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_text("content")

        backend.copy_from.side_effect = copy_side_effect

        with patch("ptq.results.backend_for_job", return_value=backend):
            dest = fetch_results("j1", output_dir=tmp_path / "out")

        assert (dest / "report.md").exists()
        assert (dest / "fix.diff").exists()
        assert (dest / "worklog.md").exists()
        assert (dest / "claude-1.log").exists()
