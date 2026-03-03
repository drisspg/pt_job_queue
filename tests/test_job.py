from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ptq.application.job_service import clean_machine, get_status, kill_job
from ptq.domain.models import JobNotFoundError, JobRecord, JobStatus
from ptq.domain.policies import make_job_id
from ptq.infrastructure.backends import backend_for_job, create_backend
from ptq.infrastructure.job_repository import JobRepository
from ptq.ssh import LocalBackend, RemoteBackend


class TestMakeJobId:
    def test_issue_id(self, frozen_date):
        assert make_job_id(issue_number=42) == "20260217-42"

    def test_adhoc_id(self, frozen_date):
        result = make_job_id(message="hello")
        assert result.startswith("20260217-adhoc-")
        assert len(result.split("-")) == 3

    def test_adhoc_ids_differ_by_message(self, frozen_date):
        assert make_job_id(message="a") != make_job_id(message="b")

    def test_adhoc_default(self, frozen_date):
        result = make_job_id()
        assert "adhoc" in result


class TestJobRecord:
    def test_roundtrip(self):
        record = JobRecord(
            job_id="20260217-42",
            issue=42,
            runs=2,
            agent="codex",
            model="o3",
            machine="gpu-dev",
            workspace="~/ptq_workspace",
            pid=12345,
        )
        d = record.to_dict()
        restored = JobRecord.from_dict("20260217-42", d)
        assert restored.job_id == record.job_id
        assert restored.issue == record.issue
        assert restored.runs == record.runs
        assert restored.agent == record.agent
        assert restored.model == record.model
        assert restored.machine == record.machine
        assert restored.pid == record.pid

    def test_local_roundtrip(self):
        record = JobRecord(
            job_id="20260217-adhoc-abc123",
            local=True,
            workspace="~/.ptq_workspace",
        )
        d = record.to_dict()
        assert d["local"] is True
        assert "machine" not in d
        restored = JobRecord.from_dict("20260217-adhoc-abc123", d)
        assert restored.local is True
        assert restored.workspace == "~/.ptq_workspace"

    def test_target_property(self):
        assert JobRecord(job_id="j", machine="gpu-dev").target == "gpu-dev"
        assert JobRecord(job_id="j", local=True).target == "local"

    def test_initializing_omitted_when_false(self):
        d = JobRecord(job_id="j").to_dict()
        assert "initializing" not in d

    def test_pid_omitted_when_none(self):
        d = JobRecord(job_id="j").to_dict()
        assert "pid" not in d


class TestJobRepository:
    def test_save_and_get(self, repo: JobRepository):
        record = JobRecord(job_id="test-1", issue=42, agent="claude")
        repo.save(record)
        restored = repo.get("test-1")
        assert restored.issue == 42
        assert restored.agent == "claude"

    def test_get_unknown_raises(self, repo: JobRepository):
        with pytest.raises(JobNotFoundError):
            repo.get("nonexistent")

    def test_delete(self, repo: JobRepository):
        repo.save(JobRecord(job_id="del-me"))
        repo.delete("del-me")
        with pytest.raises(JobNotFoundError):
            repo.get("del-me")

    def test_list_all(self, repo: JobRepository):
        repo.save(JobRecord(job_id="a", issue=1))
        repo.save(JobRecord(job_id="b", issue=2))
        all_jobs = repo.list_all()
        assert len(all_jobs) == 2
        assert "a" in all_jobs
        assert "b" in all_jobs

    def test_resolve_id_by_job_id(self, repo: JobRepository):
        repo.save(JobRecord(job_id="20260217-42", issue=42))
        assert repo.resolve_id("20260217-42") == "20260217-42"

    def test_resolve_id_by_issue(self, repo: JobRepository):
        repo.save(JobRecord(job_id="20260217-42", issue=42))
        assert repo.resolve_id("42") == "20260217-42"

    def test_resolve_id_unknown_raises(self, repo: JobRepository):
        with pytest.raises(JobNotFoundError):
            repo.resolve_id("nonexistent")

    def test_find_by_issue(self, repo: JobRepository):
        repo.save(JobRecord(job_id="j1", issue=42, machine="gpu-dev"))
        assert repo.find_by_issue(42, machine="gpu-dev") == "j1"
        assert repo.find_by_issue(42, machine="other") is None
        assert repo.find_by_issue(99, machine="gpu-dev") is None

    def test_find_by_issue_local(self, repo: JobRepository):
        repo.save(JobRecord(job_id="j1", issue=42, local=True))
        assert repo.find_by_issue(42, local=True) == "j1"
        assert repo.find_by_issue(42, machine="gpu-dev") is None

    def test_increment_run(self, repo: JobRepository):
        repo.save(JobRecord(job_id="j1", issue=42, runs=1))
        new_run = repo.increment_run("j1", agent_type="codex", model="o3")
        assert new_run == 2
        updated = repo.get("j1")
        assert updated.runs == 2
        assert updated.agent == "codex"
        assert updated.model == "o3"
        assert updated.initializing is True
        assert updated.pid is None

    def test_save_pid(self, repo: JobRepository):
        repo.save(JobRecord(job_id="j1", initializing=True))
        repo.save_pid("j1", 12345)
        updated = repo.get("j1")
        assert updated.pid == 12345
        assert updated.initializing is False

    def test_clear_pid(self, repo: JobRepository):
        repo.save(JobRecord(job_id="j1", pid=12345))
        repo.save_pid("j1", None)
        updated = repo.get("j1")
        assert updated.pid is None

    def test_clear_pid_also_clears_initializing(self, repo: JobRepository):
        repo.save(JobRecord(job_id="j1", pid=12345, initializing=True))
        repo.save_pid("j1", None)
        updated = repo.get("j1")
        assert updated.pid is None
        assert updated.initializing is False

    def test_save_pid_on_missing_job_is_noop(self, repo: JobRepository):
        repo.save_pid("nonexistent", 999)

    def test_from_dict_minimal(self):
        record = JobRecord.from_dict("j1", {})
        assert record.runs == 1
        assert record.agent == "claude"
        assert record.model == "opus"
        assert record.machine is None
        assert record.local is False

    def test_adhoc_none_and_empty_equivalent(self, frozen_date):
        assert make_job_id(message=None) == make_job_id(message="")

    def test_adhoc_deterministic(self, frozen_date):
        assert make_job_id(message="fix bug") == make_job_id(message="fix bug")


class TestCreateBackend:
    def test_local(self):
        b = create_backend(local=True)
        assert isinstance(b, LocalBackend)
        assert b.workspace == "~/.ptq_workspace"

    def test_local_custom_workspace(self):
        b = create_backend(local=True, workspace="/custom/ws")
        assert isinstance(b, LocalBackend)
        assert b.workspace == "/custom/ws"

    def test_remote(self):
        b = create_backend(machine="gpu-dev")
        assert isinstance(b, RemoteBackend)
        assert b.workspace == "~/ptq_workspace"

    def test_neither_raises(self):
        with pytest.raises(ValueError, match="Must specify"):
            create_backend()


class TestBackendForJob:
    def test_local_job(self):
        record = JobRecord(job_id="j1", local=True, workspace="~/.ptq_workspace")
        b = backend_for_job(record)
        assert isinstance(b, LocalBackend)

    def test_remote_job(self):
        record = JobRecord(job_id="j1", machine="gpu-dev", workspace="~/ws")
        b = backend_for_job(record)
        assert isinstance(b, RemoteBackend)

    def test_no_target_raises(self):
        record = JobRecord(job_id="j1")
        with pytest.raises(JobNotFoundError, match="has no target"):
            backend_for_job(record)


class TestGetStatus:
    def test_initializing(self):
        job = JobRecord(job_id="j1", initializing=True, pid=123)
        backend = MagicMock()
        assert get_status(job, backend) == JobStatus.INITIALIZING

    def test_no_pid(self):
        job = JobRecord(job_id="j1")
        backend = MagicMock()
        assert get_status(job, backend) == JobStatus.STOPPED

    def test_alive_pid(self):
        job = JobRecord(job_id="j1", pid=123)
        backend = MagicMock()
        backend.is_pid_alive.return_value = True
        assert get_status(job, backend) == JobStatus.RUNNING

    def test_dead_pid(self):
        job = JobRecord(job_id="j1", pid=123)
        backend = MagicMock()
        backend.is_pid_alive.return_value = False
        assert get_status(job, backend) == JobStatus.STOPPED

    def test_pid_zero_is_not_none(self):
        job = JobRecord(job_id="j1", pid=0)
        backend = MagicMock()
        backend.is_pid_alive.return_value = False
        assert get_status(job, backend) == JobStatus.STOPPED
        backend.is_pid_alive.assert_called_once_with(0)


class TestKillJob:
    def test_kills_running(self, repo: JobRepository):
        repo.save(JobRecord(job_id="j1", machine="gpu-dev", workspace="~/ws", pid=42))
        backend = MagicMock()
        backend.is_pid_alive.return_value = True
        with MagicMock() as mock_bfj:
            mock_bfj.return_value = backend
            from unittest.mock import patch

            with patch("ptq.application.job_service.backend_for_job", mock_bfj):
                killed = kill_job(repo, "j1")
        assert killed is True
        backend.kill_pid.assert_called_once_with(42)
        assert repo.get("j1").pid is None

    def test_already_stopped(self, repo: JobRepository):
        repo.save(JobRecord(job_id="j1", machine="gpu-dev", workspace="~/ws"))
        backend = MagicMock()
        from unittest.mock import patch

        with patch("ptq.application.job_service.backend_for_job", return_value=backend):
            killed = kill_job(repo, "j1")
        assert killed is False
        backend.kill_pid.assert_not_called()


class TestCleanMachine:
    def _setup_jobs(self, repo: JobRepository, count: int) -> list[str]:
        ids = []
        for i in range(count):
            jid = f"job-{i}"
            repo.save(JobRecord(job_id=jid, machine="gpu-dev", workspace="~/ws"))
            ids.append(jid)
        return ids

    def test_keep_preserves_newest(self, repo: JobRepository):
        ids = self._setup_jobs(repo, 5)
        backend = MagicMock()
        backend.workspace = "~/ws"
        backend.is_pid_alive.return_value = False
        removed, skipped = clean_machine(repo, backend, machine="gpu-dev", keep=2)
        assert len(removed) == 3
        assert set(removed) == {ids[0], ids[1], ids[2]}
        remaining = repo.list_all()
        assert ids[3] in remaining
        assert ids[4] in remaining

    def test_keep_equal_to_count_removes_none(self, repo: JobRepository):
        self._setup_jobs(repo, 2)
        backend = MagicMock()
        backend.workspace = "~/ws"
        backend.is_pid_alive.return_value = False
        removed, _ = clean_machine(repo, backend, machine="gpu-dev", keep=2)
        assert removed == []
        assert len(repo.list_all()) == 2

    def test_keep_greater_than_count_removes_none(self, repo: JobRepository):
        self._setup_jobs(repo, 2)
        backend = MagicMock()
        backend.workspace = "~/ws"
        backend.is_pid_alive.return_value = False
        removed, _ = clean_machine(repo, backend, machine="gpu-dev", keep=5)
        assert removed == []

    def test_skips_running_by_default(self, repo: JobRepository):
        repo.save(JobRecord(job_id="stopped", machine="gpu-dev", workspace="~/ws"))
        repo.save(
            JobRecord(job_id="running", machine="gpu-dev", workspace="~/ws", pid=42)
        )
        backend = MagicMock()
        backend.workspace = "~/ws"
        backend.is_pid_alive.side_effect = lambda pid: pid == 42
        removed, skipped = clean_machine(repo, backend, machine="gpu-dev")
        assert "stopped" in removed
        assert "running" not in removed
        assert skipped == 1

    def test_include_running(self, repo: JobRepository):
        repo.save(
            JobRecord(job_id="running", machine="gpu-dev", workspace="~/ws", pid=42)
        )
        backend = MagicMock()
        backend.workspace = "~/ws"
        backend.is_pid_alive.return_value = True
        removed, _ = clean_machine(
            repo, backend, machine="gpu-dev", include_running=True
        )
        assert "running" in removed
