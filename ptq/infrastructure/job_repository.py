from __future__ import annotations

import json
from pathlib import Path

from ptq.domain.models import JobNotFoundError, JobRecord


class JobRepository:
    def __init__(self, path: Path | None = None):
        self._path = path or (Path.home() / ".ptq" / "jobs.json")

    def _load_raw(self) -> dict:
        if self._path.exists():
            return json.loads(self._path.read_text())
        return {}

    def _save_raw(self, db: dict) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(db, indent=2))

    def list_all(self) -> dict[str, JobRecord]:
        return {
            jid: JobRecord.from_dict(jid, entry)
            for jid, entry in self._load_raw().items()
        }

    def get(self, job_id: str) -> JobRecord:
        db = self._load_raw()
        entry = db.get(job_id)
        if not entry:
            raise JobNotFoundError(f"Unknown job: {job_id}")
        return JobRecord.from_dict(job_id, entry)

    def save(self, record: JobRecord) -> None:
        db = self._load_raw()
        db[record.job_id] = record.to_dict()
        self._save_raw(db)

    def delete(self, job_id: str) -> None:
        db = self._load_raw()
        db.pop(job_id, None)
        self._save_raw(db)

    def resolve_id(self, job_id_or_issue: str) -> str:
        db = self._load_raw()
        if job_id_or_issue in db:
            return job_id_or_issue
        if job_id_or_issue.isdigit():
            issue_num = int(job_id_or_issue)
            matches = [(k, v) for k, v in db.items() if v.get("issue") == issue_num]
            if matches:
                return sorted(matches, key=lambda x: x[0])[-1][0]
            raise JobNotFoundError(f"No jobs found for issue #{issue_num}")
        raise JobNotFoundError(f"Unknown job: {job_id_or_issue}")

    def find_by_issue(
        self,
        issue_number: int,
        machine: str | None = None,
        local: bool = False,
    ) -> str | None:
        for job_id, entry in sorted(self._load_raw().items(), reverse=True):
            if entry.get("issue") != issue_number:
                continue
            if local and entry.get("local"):
                return job_id
            if machine and entry.get("machine") == machine:
                return job_id
        return None

    def increment_run(
        self, job_id: str, agent_type: str | None = None, model: str | None = None
    ) -> int:
        job = self.get(job_id)
        job.runs += 1
        job.pid = None
        job.initializing = True
        if agent_type:
            job.agent = agent_type
        if model:
            job.model = model
        self.save(job)
        return job.runs

    def save_rebase(self, job_id: str, rebase_data: dict) -> None:
        db = self._load_raw()
        if job_id not in db:
            return
        if rebase_data:
            db[job_id]["rebase"] = rebase_data
        else:
            db[job_id].pop("rebase", None)
        self._save_raw(db)

    def save_pid(self, job_id: str, pid: int | None) -> None:
        db = self._load_raw()
        if job_id not in db:
            return
        if pid is not None:
            db[job_id]["pid"] = pid
        else:
            db[job_id].pop("pid", None)
        db[job_id].pop("initializing", None)
        self._save_raw(db)
