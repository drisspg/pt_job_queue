from __future__ import annotations

import time
from contextlib import contextmanager

from ptq.application.venv_service import (
    ProgressCallback,
    _noop_progress,
    _setup_job_venv,
)
from ptq.domain.models import PtqError
from ptq.repo_profiles import get_profile
from ptq.ssh import Backend


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
