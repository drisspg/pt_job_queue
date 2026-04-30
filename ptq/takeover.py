from __future__ import annotations

import shlex

from ptq.domain.models import JobRecord
from ptq.repo_profiles import get_profile


def job_dir_path(workspace: str, job_id: str) -> str:
    return f"{workspace}/jobs/{job_id}"


def worktree_path(workspace: str, job_id: str, repo: str = "pytorch") -> str:
    profile = get_profile(repo)
    return f"{job_dir_path(workspace, job_id)}/{profile.dir_name}"


def _shell_path(path: str) -> str:
    if path in {"~", "~/"}:
        return "$HOME"
    if path.startswith("~/"):
        return f"$HOME/{shlex.quote(path[2:])}"
    return shlex.quote(path)


def shell_command(
    *,
    workspace: str,
    job_id: str,
    repo: str = "pytorch",
    local: bool,
    machine: str | None = None,
) -> str:
    job_dir = _shell_path(job_dir_path(workspace, job_id))
    if local:
        return f"cd {job_dir} && source .venv/bin/activate"
    remote_cmd = f"cd {job_dir} && source .venv/bin/activate && exec $SHELL"
    return f"ssh -t {shlex.quote(machine or '')} {shlex.quote(remote_cmd)}"


def for_job(job_id: str, job: JobRecord) -> str:
    return shell_command(
        workspace=job.workspace,
        job_id=job_id,
        repo=job.repo,
        local=job.local,
        machine=job.machine,
    )
