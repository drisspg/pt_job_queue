from __future__ import annotations

import hashlib
from datetime import datetime

from ptq.ssh import load_jobs_db, save_jobs_db


def make_job_id(issue_number: int | None = None, message: str | None = None) -> str:
    date = datetime.now().strftime("%Y%m%d")
    if issue_number is not None:
        return f"{date}-{issue_number}"
    short = hashlib.md5((message or "adhoc").encode()).hexdigest()[:6]
    return f"{date}-adhoc-{short}"


def find_existing_job(
    issue_number: int, machine: str | None = None, local: bool = False
) -> str | None:
    db = load_jobs_db()
    for job_id, entry in sorted(db.items(), reverse=True):
        if entry.get("issue") != issue_number:
            continue
        if local and entry.get("local"):
            return job_id
        if machine and entry.get("machine") == machine:
            return job_id
    return None


def register_job(
    job_id: str,
    *,
    issue_number: int | None = None,
    machine: str | None = None,
    local: bool = False,
    workspace: str | None = None,
    run_number: int = 1,
) -> None:
    db = load_jobs_db()
    entry: dict = {"issue": issue_number, "runs": run_number}
    if local:
        entry["local"] = True
        entry["workspace"] = workspace or "~/.ptq_workspace"
    else:
        entry["machine"] = machine
        entry["workspace"] = workspace or "~/ptq_workspace"
    db[job_id] = entry
    save_jobs_db(db)


def increment_run(job_id: str) -> int:
    db = load_jobs_db()
    entry = db[job_id]
    run_number = entry.get("runs", 0) + 1
    entry["runs"] = run_number
    entry.pop("pid", None)
    save_jobs_db(db)
    return run_number


def save_pid(job_id: str, pid: int | None) -> None:
    db = load_jobs_db()
    if job_id not in db:
        return
    if pid is not None:
        db[job_id]["pid"] = pid
    else:
        db[job_id].pop("pid", None)
    save_jobs_db(db)


def resolve_job_id(job_id_or_issue: str) -> str:
    db = load_jobs_db()
    if job_id_or_issue in db:
        return job_id_or_issue
    if job_id_or_issue.isdigit():
        issue_num = int(job_id_or_issue)
        matches = [(k, v) for k, v in db.items() if v.get("issue") == issue_num]
        if matches:
            return sorted(matches, key=lambda x: x[0])[-1][0]
        raise SystemExit(f"No jobs found for issue #{issue_num}")
    raise SystemExit(f"Unknown job: {job_id_or_issue}")


def save_reservation_id(job_id: str, reservation_id: str) -> None:
    db = load_jobs_db()
    if job_id not in db:
        return
    db[job_id]["reservation_id"] = reservation_id
    save_jobs_db(db)


def get_reservation_id(job_id: str) -> str | None:
    db = load_jobs_db()
    entry = db.get(job_id)
    if not entry:
        return None
    return entry.get("reservation_id")


def get_job(job_id: str) -> dict:
    db = load_jobs_db()
    entry = db.get(job_id)
    if not entry:
        raise SystemExit(f"Unknown job: {job_id}")
    return entry
