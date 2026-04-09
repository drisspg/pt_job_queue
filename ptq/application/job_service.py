from __future__ import annotations

from ptq.domain.models import JobRecord, JobStatus
from ptq.infrastructure.backends import backend_for_job
from ptq.infrastructure.job_repository import JobRepository
from ptq.repo_profiles import get_profile
from ptq.ssh import Backend


def get_status(job: JobRecord, backend: Backend) -> JobStatus:
    if job.initializing:
        return JobStatus.INITIALIZING
    if job.pid is None:
        return JobStatus.STOPPED
    if backend.is_pid_alive(job.pid):
        return JobStatus.RUNNING
    return JobStatus.STOPPED


def kill_job(repo: JobRepository, job_id: str) -> bool:
    job = repo.get(job_id)
    backend = backend_for_job(job)
    killed = False
    if job.pid is not None and backend.is_pid_alive(job.pid):
        backend.kill_pid(job.pid)
        killed = True
    repo.save_pid(job_id, None)
    return killed


def clean_single_job(repo: JobRepository, job_id: str) -> JobRecord:
    """Remove a job: kill agent, delete files, drop from DB. Returns the removed record."""
    job = repo.get(job_id)
    profile = get_profile(job.repo)
    backend = backend_for_job(job)
    ws = backend.workspace
    job_dir = f"{ws}/jobs/{job_id}"

    if job.pid is not None and backend.is_pid_alive(job.pid):
        backend.kill_pid(job.pid)

    if profile.uses_custom_worktree_tool:
        backend.run(
            f"cd {ws}/pytorch && {ws}/.venv/bin/python tools/create_worktree.py remove pytorch "
            f"--parent-dir {job_dir}",
            check=False,
        )
    else:
        worktree_path = f"{job_dir}/{profile.dir_name}"
        backend.run(f"git -C {ws}/{profile.dir_name} worktree remove --force {worktree_path}", check=False)

    backend.run(f"rm -rf {job_dir}", check=False)
    backend.run(f"cd {ws}/{profile.dir_name} && git worktree prune", check=False)
    repo.delete(job_id)
    return job


def clean_machine(
    repo: JobRepository,
    backend: Backend,
    *,
    machine: str | None = None,
    local: bool = False,
    keep: int = 0,
    include_running: bool = False,
) -> tuple[list[str], int]:
    """Bulk clean jobs on a machine. Returns (removed_ids, skipped_running_count)."""
    all_jobs = repo.list_all()
    matching = [
        (jid, job)
        for jid, job in sorted(all_jobs.items())
        if (local and job.local) or (machine and job.machine == machine)
    ]

    skipped_running = 0
    if not include_running:
        stopped = []
        for jid, job in matching:
            if job.pid is not None and backend.is_pid_alive(job.pid):
                skipped_running += 1
            else:
                stopped.append((jid, job))
        matching = stopped

    to_remove = matching[:-keep] if keep else matching
    if not to_remove:
        return [], skipped_running

    ws = backend.workspace
    # Prune worktrees for all repos that have jobs being cleaned
    repos_seen: set[str] = set()

    removed: list[str] = []
    for jid, job in to_remove:
        profile = get_profile(job.repo)
        repos_seen.add(job.repo)
        if job.pid is not None and backend.is_pid_alive(job.pid):
            backend.kill_pid(job.pid)
        job_dir = f"{ws}/jobs/{jid}"
        if profile.uses_custom_worktree_tool:
            backend.run(
                f"cd {ws}/pytorch && {ws}/.venv/bin/python tools/create_worktree.py remove pytorch "
                f"--parent-dir {job_dir}",
                check=False,
            )
        else:
            worktree_path = f"{job_dir}/{profile.dir_name}"
            backend.run(f"git -C {ws}/{profile.dir_name} worktree remove --force {worktree_path}", check=False)
        backend.run(f"rm -rf {job_dir}")
        repo.delete(jid)
        removed.append(jid)

    for repo_name in repos_seen:
        p = get_profile(repo_name)
        backend.run(f"cd {ws}/{p.dir_name} && git worktree prune", check=False)
    return removed, skipped_running
