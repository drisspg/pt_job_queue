from __future__ import annotations

from unittest.mock import MagicMock, patch

from ptq.application.artifact_service import fetch_results, read_artifact
from ptq.domain.models import JobRecord
from ptq.infrastructure.job_repository import JobRepository


class TestFetchResults:
    def test_fetches_artifacts(self, tmp_path):
        repo = JobRepository(tmp_path / "jobs.json")
        repo.save(
            JobRecord(
                job_id="j1",
                issue=42,
                machine="gpu-dev",
                workspace="~/ws",
                agent="claude",
                runs=2,
            )
        )
        backend = MagicMock()
        backend.workspace = "~/ws"
        backend.copy_from = MagicMock()

        output = tmp_path / "results"
        with patch(
            "ptq.application.artifact_service.backend_for_job",
            return_value=backend,
        ):
            dest, fetched, missing = fetch_results(repo, "j1", output)

        assert dest == output
        assert backend.copy_from.call_count == 4

    def test_missing_artifacts_tracked(self, tmp_path):
        repo = JobRepository(tmp_path / "jobs.json")
        repo.save(JobRecord(job_id="j1", machine="gpu-dev", workspace="~/ws"))
        backend = MagicMock()
        backend.workspace = "~/ws"
        backend.copy_from.side_effect = FileNotFoundError

        with patch(
            "ptq.application.artifact_service.backend_for_job",
            return_value=backend,
        ):
            _, fetched, missing = fetch_results(repo, "j1", tmp_path / "out")

        assert len(fetched) == 0
        assert len(missing) == 4


class TestReadArtifact:
    def test_returns_content(self):
        backend = MagicMock()
        backend.run.return_value = MagicMock(returncode=0, stdout="hello\n")
        assert read_artifact(backend, "/path") == "hello\n"

    def test_returns_none_on_failure(self):
        backend = MagicMock()
        backend.run.return_value = MagicMock(returncode=1, stdout="")
        assert read_artifact(backend, "/path") is None

    def test_returns_none_on_empty(self):
        backend = MagicMock()
        backend.run.return_value = MagicMock(returncode=0, stdout="   \n")
        assert read_artifact(backend, "/path") is None
