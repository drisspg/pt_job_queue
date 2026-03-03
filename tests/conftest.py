from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from ptq.infrastructure.job_repository import JobRepository


@pytest.fixture()
def repo(tmp_path: Path) -> JobRepository:
    return JobRepository(tmp_path / "jobs.json")


@pytest.fixture()
def frozen_date():
    fake_now = datetime(2026, 2, 17, 12, 0, 0)
    with patch("ptq.domain.policies.datetime") as mock_dt:
        mock_dt.now.return_value = fake_now
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        yield "20260217"
