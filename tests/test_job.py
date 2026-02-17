from __future__ import annotations

import pytest

from ptq.job import (
    find_existing_job,
    get_job,
    increment_run,
    make_job_id,
    register_job,
    resolve_job_id,
    save_pid,
)


class TestMakeJobId:
    def test_issue_number(self, jobs_db, frozen_date):
        assert make_job_id(issue_number=123) == "20260217-123"

    def test_message_produces_adhoc(self, jobs_db, frozen_date):
        jid = make_job_id(message="fix oom bug")
        assert jid.startswith("20260217-adhoc-")
        assert len(jid.split("-")[-1]) == 6

    def test_deterministic_for_same_message(self, jobs_db, frozen_date):
        assert make_job_id(message="hello") == make_job_id(message="hello")

    def test_different_messages_differ(self, jobs_db, frozen_date):
        assert make_job_id(message="hello") != make_job_id(message="world")


class TestResolveJobId:
    def test_exact_match(self, jobs_db):
        jobs_db["20260217-100"] = {"issue": 100, "runs": 1}
        assert resolve_job_id("20260217-100") == "20260217-100"

    def test_numeric_issue_lookup(self, jobs_db):
        jobs_db["20260217-42"] = {"issue": 42, "runs": 1}
        assert resolve_job_id("42") == "20260217-42"

    def test_picks_latest_when_multiple(self, jobs_db):
        jobs_db["20260210-5"] = {"issue": 5, "runs": 1}
        jobs_db["20260217-5"] = {"issue": 5, "runs": 2}
        assert resolve_job_id("5") == "20260217-5"

    def test_unknown_id_exits(self, jobs_db):
        with pytest.raises(SystemExit, match="Unknown job"):
            resolve_job_id("nonexistent")

    def test_unknown_issue_exits(self, jobs_db):
        with pytest.raises(SystemExit, match="No jobs found"):
            resolve_job_id("9999")


class TestGetJob:
    def test_found(self, jobs_db):
        jobs_db["j1"] = {"issue": 1, "runs": 1}
        assert get_job("j1") == {"issue": 1, "runs": 1}

    def test_missing_exits(self, jobs_db):
        with pytest.raises(SystemExit, match="Unknown job"):
            get_job("nope")


class TestRegisterJob:
    def test_local(self, jobs_db):
        register_job("j-local", issue_number=10, local=True)
        entry = jobs_db["j-local"]
        assert entry["local"] is True
        assert entry["workspace"] == "~/.ptq_workspace"
        assert entry["issue"] == 10

    def test_remote(self, jobs_db):
        register_job("j-remote", issue_number=20, machine="gpu-box")
        entry = jobs_db["j-remote"]
        assert entry["machine"] == "gpu-box"
        assert entry["workspace"] == "~/ptq_workspace"


class TestIncrementRun:
    def test_bumps_runs_and_clears_pid(self, jobs_db):
        jobs_db["j1"] = {"issue": 1, "runs": 1, "pid": 12345}
        result = increment_run("j1")
        assert result == 2
        assert jobs_db["j1"]["runs"] == 2
        assert "pid" not in jobs_db["j1"]


class TestSavePid:
    def test_sets_pid(self, jobs_db):
        jobs_db["j1"] = {"issue": 1, "runs": 1}
        save_pid("j1", 9999)
        assert jobs_db["j1"]["pid"] == 9999

    def test_clears_pid(self, jobs_db):
        jobs_db["j1"] = {"issue": 1, "runs": 1, "pid": 9999}
        save_pid("j1", None)
        assert "pid" not in jobs_db["j1"]

    def test_noop_for_unknown(self, jobs_db):
        save_pid("missing", 1234)


class TestFindExistingJob:
    def test_match_by_issue_and_local(self, jobs_db):
        jobs_db["j1"] = {"issue": 5, "local": True, "runs": 1}
        assert find_existing_job(5, local=True) == "j1"

    def test_match_by_issue_and_machine(self, jobs_db):
        jobs_db["j1"] = {"issue": 5, "machine": "gpu-box", "runs": 1}
        assert find_existing_job(5, machine="gpu-box") == "j1"

    def test_no_match(self, jobs_db):
        jobs_db["j1"] = {"issue": 5, "machine": "gpu-box", "runs": 1}
        assert find_existing_job(99, machine="gpu-box") is None
