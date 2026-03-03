from __future__ import annotations

import logging
from pathlib import Path

from fastapi.templating import Jinja2Templates

from ptq.application.job_service import get_status
from ptq.domain.models import JobRecord, JobStatus
from ptq.infrastructure.backends import backend_for_job
from ptq.infrastructure.job_repository import JobRepository

_WEB_DIR = Path(__file__).parent

templates = Jinja2Templates(directory=_WEB_DIR / "templates")

log = logging.getLogger("ptq.web")

_finalized_runs: set[tuple[str, int]] = set()


def get_job_status_with_finalize(job_id: str, job: JobRecord) -> str:
    """Get status and trigger finalization for newly-stopped jobs.

    The core get_status() in job_service remains pure/read-only.
    This web-specific wrapper adds finalization side effects.
    """
    backend = backend_for_job(job)
    status = get_status(job, backend)

    if status == JobStatus.STOPPED and job.pid is not None:
        run_key = (job_id, job.runs)
        if run_key not in _finalized_runs:
            _finalized_runs.add(run_key)
            JobRepository().save_pid(job_id, None)
            try:
                from ptq.application.run_service import finalize_run

                finalize_run(backend, job_id, job)
            except Exception:
                log.exception("failed to finalize run %s", job_id)

    return status.value
