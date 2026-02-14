from __future__ import annotations

import secrets
from datetime import datetime

from ptq.ssh import load_jobs_db, save_jobs_db


def make_job_id(issue_number: int) -> str:
    suffix = secrets.token_hex(2)
    return f"{datetime.now().strftime('%Y%m%d')}-{issue_number}-{suffix}"


def register_job(job_id: str, *, machine: str | None = None, local: bool = False, workspace: str | None = None) -> None:
    db = load_jobs_db()
    parts = job_id.split("-")
    entry: dict = {"issue": int(parts[1])}
    if local:
        entry["local"] = True
        entry["workspace"] = workspace or "~/.ptq_workspace"
    else:
        entry["machine"] = machine
        entry["workspace"] = workspace or "~/ptq_workspace"
    db[job_id] = entry
    save_jobs_db(db)


def get_job(job_id: str) -> dict:
    db = load_jobs_db()
    entry = db.get(job_id)
    if not entry:
        raise SystemExit(f"Unknown job: {job_id}")
    return entry
