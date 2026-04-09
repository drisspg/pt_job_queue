from __future__ import annotations

import fcntl
import json
import os
import tempfile
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path

from ptq.domain.models import JobNotFoundError, JobRecord


class JobRepository:
    def __init__(self, path: Path | None = None):
        self._path = path or (Path.home() / ".ptq" / "jobs.json")

    @property
    def _lock_path(self) -> Path:
        return self._path.with_suffix(f"{self._path.suffix}.lock")

    @contextmanager
    def _locked(self, *, exclusive: bool):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock_path.open("a", encoding="utf-8") as lock_file:
            lock_type = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
            fcntl.flock(lock_file.fileno(), lock_type)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _load_raw_unlocked(self) -> dict:
        if self._path.exists():
            return json.loads(self._path.read_text())
        return {}

    def _load_raw(self) -> dict:
        with self._locked(exclusive=False):
            return self._load_raw_unlocked()

    def _save_raw_unlocked(self, db: dict) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            dir=self._path.parent,
            prefix=f".{self._path.name}.",
            suffix=".tmp",
            text=True,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
                json.dump(db, tmp_file, indent=2)
                tmp_file.write("\n")
                tmp_file.flush()
                os.fsync(tmp_file.fileno())
            os.replace(tmp_name, self._path)
        finally:
            if os.path.exists(tmp_name):
                os.remove(tmp_name)

    def _save_raw(self, db: dict) -> None:
        with self._locked(exclusive=True):
            self._save_raw_unlocked(db)

    def _update_raw(self, update: Callable[[dict], None]) -> None:
        with self._locked(exclusive=True):
            db = self._load_raw_unlocked()
            update(db)
            self._save_raw_unlocked(db)

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
        self._update_raw(lambda db: db.__setitem__(record.job_id, record.to_dict()))

    def delete(self, job_id: str) -> None:
        self._update_raw(lambda db: db.pop(job_id, None))

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
        by_name = [(k, v) for k, v in db.items() if v.get("name") == job_id_or_issue]
        if by_name:
            return sorted(by_name, key=lambda x: x[0])[-1][0]
        raise JobNotFoundError(f"Unknown job: {job_id_or_issue}")

    def find_by_name(self, name: str) -> str | None:
        for job_id, entry in sorted(self._load_raw().items(), reverse=True):
            if entry.get("name") == name:
                return job_id
        return None

    def find_by_issue(
        self,
        issue_number: int,
        machine: str | None = None,
        local: bool = False,
        repo: str = "pytorch",
    ) -> str | None:
        for job_id, entry in sorted(self._load_raw().items(), reverse=True):
            if entry.get("issue") != issue_number:
                continue
            if entry.get("repo", "pytorch") != repo:
                continue
            if local and entry.get("local"):
                return job_id
            if machine and entry.get("machine") == machine:
                return job_id
        return None

    def increment_run(
        self, job_id: str, agent_type: str | None = None, model: str | None = None
    ) -> int:
        with self._locked(exclusive=True):
            db = self._load_raw_unlocked()
            entry = db.get(job_id)
            if not entry:
                raise JobNotFoundError(f"Unknown job: {job_id}")
            job = JobRecord.from_dict(job_id, entry)
            job.runs += 1
            job.pid = None
            job.initializing = True
            if agent_type:
                job.agent = agent_type
            if model:
                job.model = model
            db[job_id] = job.to_dict()
            self._save_raw_unlocked(db)
            return job.runs

    def save_rebase(self, job_id: str, rebase_data: dict) -> None:
        def update(db: dict) -> None:
            if job_id not in db:
                return
            if rebase_data:
                db[job_id]["rebase"] = rebase_data
            else:
                db[job_id].pop("rebase", None)

        self._update_raw(update)

    def save_name(self, job_id: str, name: str | None) -> None:
        def update(db: dict) -> None:
            if job_id not in db:
                return
            if name:
                db[job_id]["name"] = name
            else:
                db[job_id].pop("name", None)

        self._update_raw(update)

    def save_pid(self, job_id: str, pid: int | None) -> None:
        def update(db: dict) -> None:
            if job_id not in db:
                return
            if pid is not None:
                db[job_id]["pid"] = pid
            else:
                db[job_id].pop("pid", None)
            db[job_id].pop("initializing", None)

        self._update_raw(update)
