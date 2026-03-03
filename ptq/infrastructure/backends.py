from __future__ import annotations

from ptq.domain.models import JobNotFoundError, JobRecord
from ptq.ssh import Backend, LocalBackend, RemoteBackend


def create_backend(
    *, machine: str | None = None, local: bool = False, workspace: str | None = None
) -> Backend:
    if local:
        return LocalBackend(workspace=workspace or "~/.ptq_workspace")
    if machine:
        return RemoteBackend(machine=machine, workspace=workspace or "~/ptq_workspace")
    raise ValueError("Must specify machine or local=True")


def backend_for_job(record: JobRecord) -> Backend:
    if record.local:
        return LocalBackend(workspace=record.workspace)
    if record.machine:
        return RemoteBackend(machine=record.machine, workspace=record.workspace)
    raise JobNotFoundError(f"Job {record.job_id} has no target")
