"""Tests for TorchTitan repository job metadata."""

from __future__ import annotations

from ptq.domain.models import JobRecord


class TestJobRecordRepo:
    def test_default_repo_is_pytorch(self):
        assert JobRecord(job_id="j1").repo == "pytorch"

    def test_repo_roundtrip(self):
        record = JobRecord(job_id="j1", repo="torchtitan")
        restored = JobRecord.from_dict("j1", record.to_dict())
        assert restored.repo == "torchtitan"

    def test_default_repo_omitted_from_dict(self):
        assert "repo" not in JobRecord(job_id="j1").to_dict()

    def test_from_dict_missing_repo_defaults_to_pytorch(self):
        assert JobRecord.from_dict("j1", {}).repo == "pytorch"
