from __future__ import annotations

from datetime import datetime
from unittest.mock import patch

import pytest


@pytest.fixture()
def jobs_db():
    db: dict = {}
    with (
        patch("ptq.ssh.load_jobs_db", return_value=db),
        patch("ptq.ssh.save_jobs_db"),
        patch("ptq.job.load_jobs_db", return_value=db),
        patch("ptq.job.save_jobs_db"),
    ):
        yield db


@pytest.fixture()
def frozen_date():
    fake_now = datetime(2026, 2, 17, 12, 0, 0)
    with patch("ptq.job.datetime") as mock_dt:
        mock_dt.now.return_value = fake_now
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        yield "20260217"
