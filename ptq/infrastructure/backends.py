from __future__ import annotations

import subprocess
from dataclasses import dataclass

from ptq.domain.models import JobNotFoundError, JobRecord


@dataclass
class LocalBackend:
    """Execute PTQ workspace operations on the local machine."""

    workspace: str = "~/.ptq_workspace"

    def run(
        self, command: str, check: bool = True, stream: bool = False
    ) -> subprocess.CompletedProcess[str]:
        kwargs: dict = {"text": True, "check": check}
        if not stream:
            kwargs["capture_output"] = True
        return subprocess.run(["zsh", "-c", command], **kwargs)


Backend = LocalBackend


def create_backend(*, workspace: str | None = None) -> LocalBackend:
    return LocalBackend(workspace=workspace or "~/.ptq_workspace")


def backend_for_job(record: JobRecord) -> LocalBackend:
    if record.legacy_machine:
        raise JobNotFoundError(
            f"Job {record.job_id} targets removed remote machine "
            f"{record.legacy_machine!r}; recreate it as a local workspace."
        )
    return LocalBackend(workspace=record.workspace)
