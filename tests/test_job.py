from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import pytest

from ptq.application.job_service import clean_jobs, clean_single_job
from ptq.domain.models import JobNotFoundError, JobRecord, PtqError, SubmissionMode
from ptq.domain.policies import make_job_id
from ptq.infrastructure.backends import LocalBackend, backend_for_job, create_backend
from ptq.infrastructure.job_repository import JobRepository


def _save_job_after_barrier(
    path, barrier: threading.Barrier, job_id: str, issue: int | None = None
) -> None:
    barrier.wait()
    JobRepository(path).save(JobRecord(job_id=job_id, issue=issue))


class TestMakeJobId:
    def test_issue_id(self, frozen_date):
        assert make_job_id(issue_number=42) == "20260217-pytorch-42"

    def test_adhoc_id(self, frozen_date):
        result = make_job_id(message="hello")
        assert result.startswith("20260217-pytorch-adhoc-")
        assert len(result.split("-")) == 4

    def test_adhoc_ids_differ_by_message(self, frozen_date):
        assert make_job_id(message="a") != make_job_id(message="b")


class TestJobRecord:
    def test_roundtrip(self):
        record = JobRecord(
            job_id="20260217-42",
            issue=42,
            workspace="~/workspace",
            name="example",
        )
        assert JobRecord.from_dict("20260217-42", record.to_dict()) == record

    def test_ignores_legacy_execution_and_remote_fields(self):
        record = JobRecord.from_dict(
            "legacy",
            {
                "runs": 2,
                "agent": "pi",
                "model": "old-model",
                "thinking": "high",
                "pid": 123,
                "initializing": True,
                "local": True,
                "machine": "old-host",
            },
        )
        assert record.job_id == "legacy"
        assert record.workspace == "~/.ptq_workspace"
        assert record.to_dict() == {
            "issue": None,
            "workspace": "~/.ptq_workspace",
            "machine": "old-host",
        }

    def test_stack_mode_roundtrip(self):
        record = JobRecord(
            job_id="j",
            submission_mode=SubmissionMode.GHSTACK,
            stack_base="release/2.8",
            stack_pr_urls=[
                "https://github.com/pytorch/pytorch/pull/98",
                "https://github.com/pytorch/pytorch/pull/99",
            ],
        )
        restored = JobRecord.from_dict("j", record.to_dict())
        assert restored.submission_mode == SubmissionMode.GHSTACK
        assert restored.stack_base == "release/2.8"
        assert restored.stack_pr_urls == record.stack_pr_urls

    def test_default_optional_fields_are_omitted(self):
        data = JobRecord(job_id="j").to_dict()
        assert "pr_url" not in data
        assert "submission_mode" not in data
        assert "stack_base" not in data
        assert "repo" not in data


class TestJobRepository:
    def test_save_and_get(self, repo: JobRepository):
        repo.save(JobRecord(job_id="test-1", issue=42))
        assert repo.get("test-1").issue == 42

    def test_get_unknown_raises(self, repo: JobRepository):
        with pytest.raises(JobNotFoundError):
            repo.get("nonexistent")

    def test_delete(self, repo: JobRepository):
        repo.save(JobRecord(job_id="del-me"))
        repo.delete("del-me")
        with pytest.raises(JobNotFoundError):
            repo.get("del-me")

    def test_resolve_id_by_job_issue_and_name(self, repo: JobRepository):
        repo.save(JobRecord(job_id="job-42", issue=42, name="example"))
        assert repo.resolve_id("job-42") == "job-42"
        assert repo.resolve_id("42") == "job-42"
        assert repo.resolve_id("example") == "job-42"

    def test_find_by_issue_and_repo(self, repo: JobRepository):
        repo.save(JobRecord(job_id="pytorch", issue=42))
        repo.save(JobRecord(job_id="titan", issue=42, repo="torchtitan"))
        assert repo.find_by_issue(42) == "pytorch"
        assert repo.find_by_issue(42, repo="torchtitan") == "titan"

    def test_concurrent_writes_do_not_drop_jobs(self, tmp_path):
        path = tmp_path / "jobs.json"
        job_ids = [f"job-{index}" for index in range(16)]
        barrier = threading.Barrier(len(job_ids))
        with ThreadPoolExecutor(max_workers=len(job_ids)) as executor:
            futures = [
                executor.submit(_save_job_after_barrier, path, barrier, job_id, index)
                for index, job_id in enumerate(job_ids)
            ]
            for future in futures:
                future.result()
        assert set(JobRepository(path).list_all()) == set(job_ids)


class TestBackend:
    def test_default_workspace(self):
        backend = create_backend()
        assert isinstance(backend, LocalBackend)
        assert backend.workspace == "~/.ptq_workspace"

    def test_custom_workspace(self):
        assert create_backend(workspace="/custom/ws").workspace == "/custom/ws"

    def test_backend_for_job(self):
        record = JobRecord(job_id="j1", workspace="~/workspace")
        backend = backend_for_job(record)
        assert isinstance(backend, LocalBackend)
        assert backend.workspace == "~/workspace"

    def test_legacy_remote_job_is_rejected(self):
        record = JobRecord.from_dict(
            "remote", {"machine": "gpu-dev", "workspace": "~/workspace"}
        )
        with pytest.raises(JobNotFoundError, match="removed remote machine"):
            backend_for_job(record)


class TestClean:
    @staticmethod
    def backend(*, dirty: bool = False, remove_fails: bool = False):
        backend = MagicMock(workspace="~/ws")

        def run(command, check=True, **kwargs):
            if "status --porcelain" in command:
                return MagicMock(returncode=0, stdout=" M file.py\n" if dirty else "")
            if "rev-parse --absolute-git-dir" in command:
                return MagicMock(returncode=0, stdout="/tmp/git-dir\n")
            if "rebase-merge" in command or "rebase-apply" in command:
                return MagicMock(returncode=1, stdout="")
            if (
                "worktree remove" in command or "create_worktree.py remove" in command
            ) and remove_fails:
                return MagicMock(returncode=1, stdout="", stderr="remove failed")
            return MagicMock(returncode=0, stdout="", stderr="")

        backend.run.side_effect = run
        return backend

    def test_single_clean_job(self, repo: JobRepository):
        repo.save(JobRecord(job_id="j1", workspace="~/ws"))
        with patch(
            "ptq.application.job_service.backend_for_job",
            return_value=self.backend(),
        ):
            removed = clean_single_job(repo, "j1")
        assert removed.job_id == "j1"
        with pytest.raises(JobNotFoundError):
            repo.get("j1")

    def test_refuses_dirty_job(self, repo: JobRepository):
        repo.save(JobRecord(job_id="j1", workspace="~/ws"))
        with (
            patch(
                "ptq.application.job_service.backend_for_job",
                return_value=self.backend(dirty=True),
            ),
            pytest.raises(PtqError, match="uncommitted work"),
        ):
            clean_single_job(repo, "j1")
        assert repo.get("j1").job_id == "j1"

    def test_failed_removal_preserves_record(self, repo: JobRepository):
        repo.save(JobRecord(job_id="j1", workspace="~/ws"))
        with (
            patch(
                "ptq.application.job_service.backend_for_job",
                return_value=self.backend(remove_fails=True),
            ),
            pytest.raises(PtqError, match="Failed to remove worktree"),
        ):
            clean_single_job(repo, "j1")
        assert repo.get("j1").job_id == "j1"

    def test_force_removes_dirty_job(self, repo: JobRepository):
        repo.save(JobRecord(job_id="j1", workspace="~/ws"))
        with patch(
            "ptq.application.job_service.backend_for_job",
            return_value=self.backend(dirty=True),
        ):
            clean_single_job(repo, "j1", force=True)
        with pytest.raises(JobNotFoundError):
            repo.get("j1")

    def test_keep_preserves_newest(self, repo: JobRepository):
        for index in range(5):
            repo.save(JobRecord(job_id=f"job-{index}", workspace="~/ws"))
        backend = self.backend()
        with patch("ptq.application.job_service.backend_for_job", return_value=backend):
            removed = clean_jobs(repo, keep=2)
        assert removed == ["job-0", "job-1", "job-2"]
        assert set(repo.list_all()) == {"job-3", "job-4"}
