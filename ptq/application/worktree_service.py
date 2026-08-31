from __future__ import annotations

import time
from contextlib import contextmanager
from pathlib import Path

from ptq.application.job_context import write_job_context
from ptq.application.venv_service import (
    ProgressCallback,
    _noop_progress,
    _setup_job_venv,
)
from ptq.domain.models import JobRecord, PtqError
from ptq.domain.policies import make_job_id
from ptq.infrastructure.backends import Backend, backend_for_job
from ptq.infrastructure.job_repository import JobRepository
from ptq.repo_profiles import get_profile
from ptq.workspace import deploy_scripts


@contextmanager
def _timed(label: str, progress: ProgressCallback):
    t0 = time.monotonic()
    yield
    progress(f"  {label}: {time.monotonic() - t0:.1f}s")


def validate_workspace(backend: Backend, workspace: str, repo: str = "pytorch") -> None:
    profile = get_profile(repo)
    result = backend.run(f"test -d {workspace}/{profile.dir_name}/.git", check=False)
    if result.returncode != 0:
        raise PtqError(
            f"Workspace broken: {workspace}/{profile.dir_name}/.git missing. Re-run: ptq setup"
        )


def ensure_job_worktree(
    job_repo: JobRepository,
    backend: Backend,
    *,
    issue_number: int | None = None,
    name: str | None = None,
    verbose: bool = False,
    progress: ProgressCallback | None = None,
    repo: str = "pytorch",
    workspace_explicit: bool = False,
) -> tuple[str, bool]:
    """Create or reuse a PTQ job worktree without launching an agent."""
    if (issue_number is None) == (name is None):
        raise PtqError("Provide exactly one of issue_number or name.")

    existing = (
        job_repo.find_by_issue(issue_number, repo=repo)
        if issue_number is not None
        else job_repo.find_by_name(name or "", repo=repo)
    )
    if existing:
        job = job_repo.get(existing)
        if job.legacy_machine:
            raise PtqError(
                f"Job {existing} targets removed remote machine "
                f"{job.legacy_machine!r}; choose a new local name."
            )
        if (
            workspace_explicit
            and Path(job.workspace).expanduser() != Path(backend.workspace).expanduser()
        ):
            raise PtqError(
                f"Job {existing} already exists in {job.workspace}; "
                "open it by ID or choose a different name."
            )
        prepare_job_worktree(
            backend_for_job(job),
            existing,
            name=job.name,
            verbose=verbose,
            progress=progress,
            repo=job.repo,
        )
        return existing, False

    validate_workspace(backend, backend.workspace, repo=repo)
    job_id = make_job_id(issue_number=issue_number, message=name, repo=repo)
    job_repo.save(
        JobRecord(
            job_id=job_id,
            issue=issue_number,
            workspace=backend.workspace,
            name=name,
            repo=repo,
        )
    )
    try:
        prepare_job_worktree(
            backend,
            job_id,
            name=name,
            verbose=verbose,
            progress=progress,
            repo=repo,
        )
    except Exception:
        job_repo.delete(job_id)
        raise
    return job_id, True


def prepare_job_worktree(
    backend: Backend,
    job_id: str,
    *,
    name: str | None = None,
    verbose: bool = False,
    progress: ProgressCallback | None = None,
    repo: str = "pytorch",
) -> None:
    """Verify a job worktree, repair missing pieces, and refresh its context."""
    validate_workspace(backend, backend.workspace, repo=repo)
    deploy_scripts(backend)
    provision_worktree(
        backend,
        job_id,
        verbose=verbose,
        progress=progress,
        repo=repo,
    )
    write_job_context(
        backend,
        job_id=job_id,
        workspace=backend.workspace,
        repo=repo,
        name=name,
    )


def provision_worktree(
    backend: Backend,
    job_id: str,
    *,
    verbose: bool = False,
    progress: ProgressCallback | None = None,
    repo: str = "pytorch",
) -> bool:
    """Create a git worktree and per-worktree venv if they don't already exist.

    Returns True if a new worktree was created, False if reusing existing.
    """
    cb = progress or _noop_progress
    profile = get_profile(repo)
    workspace = backend.workspace
    job_dir = f"{workspace}/jobs/{job_id}"
    worktree_path = f"{job_dir}/{profile.dir_name}"

    backend.run(f"mkdir -p {job_dir}")

    worktree_exists = backend.run(
        f"test -d {worktree_path}/.git || test -f {worktree_path}/.git", check=False
    )
    venv_exists = backend.run(f"test -d {job_dir}/.venv/bin", check=False)
    if worktree_exists.returncode == 0 and venv_exists.returncode == 0:
        cb("Reusing existing worktree.")
        return False

    if worktree_exists.returncode != 0:
        if profile.uses_custom_worktree_tool:
            cb("Creating worktree with submodules...")
            with _timed("worktree creation", cb):
                backend.run(
                    f"cd {workspace}/pytorch && {workspace}/.venv/bin/python tools/create_worktree.py create pytorch "
                    f"--parent-dir {job_dir} --commit HEAD",
                    stream=verbose,
                )
        else:
            cb(f"Creating {profile.name} worktree...")
            with _timed("worktree creation", cb):
                branch = f"ptq-{job_id}"
                backend.run(
                    f"cd {workspace}/{profile.dir_name} && "
                    f"git worktree add -b {branch} {worktree_path} HEAD",
                    stream=verbose,
                )

    if venv_exists.returncode != 0:
        cb("Creating per-job venv...")
        from ptq.config import load_config

        _setup_job_venv(
            backend,
            job_dir,
            worktree_path,
            verbose=verbose,
            progress=cb,
            build_env_prefix=load_config().build_env_prefix(),
            repo=repo,
        )

    return worktree_exists.returncode != 0
