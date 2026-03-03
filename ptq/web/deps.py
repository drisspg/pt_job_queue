from __future__ import annotations

import logging
from pathlib import Path

from fastapi.templating import Jinja2Templates

from ptq.ssh import Backend, backend_for_job

_WEB_DIR = Path(__file__).parent

templates = Jinja2Templates(directory=_WEB_DIR / "templates")

log = logging.getLogger("ptq.web")

_finalized_runs: set[tuple[str, int]] = set()


def get_job_status(job_id: str, entry: dict) -> str:
    if entry.get("initializing"):
        return "initializing"
    pid = entry.get("pid")
    if not pid:
        return "stopped"
    backend = backend_for_job(job_id)
    if backend.is_pid_alive(pid):
        return "running"

    from ptq.agent import finalize_run
    from ptq.job import save_pid

    run_number = entry.get("runs", 1)
    run_key = (job_id, run_number)
    if run_key not in _finalized_runs:
        _finalized_runs.add(run_key)
        save_pid(job_id, None)
        try:
            finalize_run(backend, job_id, entry)
        except Exception:
            log.exception("failed to finalize run %s", job_id)

    return "stopped"


def read_artifact(backend: Backend, path: str) -> str | None:
    result = backend.run(f"cat {path}", check=False)
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout
    return None
