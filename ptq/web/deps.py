from __future__ import annotations

from pathlib import Path

from fastapi.templating import Jinja2Templates

from ptq.ssh import Backend, backend_for_job

_WEB_DIR = Path(__file__).parent

templates = Jinja2Templates(directory=_WEB_DIR / "templates")


def get_job_status(job_id: str, entry: dict) -> str:
    pid = entry.get("pid")
    if not pid:
        return "stopped"
    backend = backend_for_job(job_id)
    return "running" if backend.is_pid_alive(pid) else "stopped"


def read_artifact(backend: Backend, path: str) -> str | None:
    result = backend.run(f"cat {path}", check=False)
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout
    return None
