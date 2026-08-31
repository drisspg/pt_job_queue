from __future__ import annotations

from ptq.domain.models import JobRecord, PtqError
from ptq.infrastructure.backends import Backend, backend_for_job
from ptq.infrastructure.job_repository import JobRepository
from ptq.repo_profiles import get_profile
from ptq.takeover import shell_path


def _job_paths(backend: Backend, job_id: str, job: JobRecord) -> tuple[str, str]:
    job_dir = f"{backend.workspace}/jobs/{job_id}"
    worktree = f"{job_dir}/{get_profile(job.repo).dir_name}"
    return job_dir, worktree


def _require_clean_worktree(backend: Backend, job_id: str, job: JobRecord) -> None:
    _, worktree = _job_paths(backend, job_id, job)
    quoted = shell_path(worktree)
    exists = backend.run(f"test -d {quoted}/.git || test -f {quoted}/.git", check=False)
    if exists.returncode != 0:
        return
    status = backend.run(
        f"git -C {quoted} status --porcelain --untracked-files=all", check=False
    )
    if status.returncode != 0 or status.stdout.strip():
        raise PtqError(
            f"Job {job_id} has uncommitted work; inspect it or rerun clean with --force."
        )
    git_dir = backend.run(
        f"git -C {quoted} rev-parse --absolute-git-dir", check=False
    ).stdout.strip()
    if git_dir:
        quoted_git_dir = shell_path(git_dir)
        in_progress = backend.run(
            f"test -d {quoted_git_dir}/rebase-merge -o "
            f"-d {quoted_git_dir}/rebase-apply",
            check=False,
        )
        if in_progress.returncode == 0:
            raise PtqError(
                f"Job {job_id} has a rebase in progress; resolve it or use --force."
            )


def _remove_job_files(
    backend: Backend, job_id: str, job: JobRecord, *, force: bool
) -> None:
    job_dir, worktree = _job_paths(backend, job_id, job)
    if not force:
        _require_clean_worktree(backend, job_id, job)

    profile = get_profile(job.repo)
    if profile.uses_custom_worktree_tool:
        result = backend.run(
            f"cd {backend.workspace}/pytorch && {backend.workspace}/.venv/bin/python "
            f"tools/create_worktree.py remove pytorch --parent-dir {job_dir}",
            check=False,
        )
    else:
        force_flag = " --force" if force else ""
        result = backend.run(
            f"git -C {backend.workspace}/{profile.dir_name} worktree remove"
            f"{force_flag} {worktree}",
            check=False,
        )
    if result.returncode != 0:
        raise PtqError(
            f"Failed to remove worktree for {job_id}: {result.stderr.strip()}"
        )

    backend.run(f"rm -rf {job_dir}")
    backend.run(
        f"cd {backend.workspace}/{profile.dir_name} && git worktree prune",
        check=False,
    )


def clean_single_job(
    repo: JobRepository, job_id: str, *, force: bool = False
) -> JobRecord:
    """Remove a clean job worktree and its repository record."""
    job = repo.get(job_id)
    backend = backend_for_job(job)
    _remove_job_files(backend, job_id, job, force=force)
    repo.delete(job_id)
    return job


def clean_jobs(repo: JobRepository, *, keep: int = 0, force: bool = False) -> list[str]:
    """Remove local jobs except the newest requested count."""
    jobs = [
        (job_id, job)
        for job_id, job in sorted(repo.list_all().items())
        if not job.legacy_machine
    ]
    to_remove = jobs[:-keep] if keep else jobs

    if not force:
        for job_id, job in to_remove:
            _require_clean_worktree(backend_for_job(job), job_id, job)

    for job_id, job in to_remove:
        backend = backend_for_job(job)
        _remove_job_files(backend, job_id, job, force=True)
        repo.delete(job_id)
    return [job_id for job_id, _ in to_remove]
