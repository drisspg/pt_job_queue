from __future__ import annotations

from collections.abc import Callable

from ptq.domain.models import PtqError, RebaseInfo, RebaseState
from ptq.infrastructure.backends import Backend, backend_for_job
from ptq.infrastructure.job_repository import JobRepository
from ptq.repo_profiles import get_profile

ProgressCallback = Callable[[str], None]


def _noop_progress(_message: str) -> None:
    pass


def _get_sha(backend: Backend, worktree: str) -> str:
    result = backend.run(f"git -C {worktree} rev-parse HEAD", check=False)
    return result.stdout.strip() if result.returncode == 0 else ""


def _conflicted_files(backend: Backend, worktree: str) -> list[str]:
    result = backend.run(
        f"git -C {worktree} diff --name-only --diff-filter=U", check=False
    )
    if result.returncode != 0 or not result.stdout.strip():
        return []
    return [path.strip() for path in result.stdout.splitlines() if path.strip()]


def _is_rebase_in_progress(backend: Backend, worktree: str) -> bool:
    result = backend.run(
        f"git_dir=$(git -C {worktree} rev-parse --absolute-git-dir) && "
        'test -d "$git_dir/rebase-merge" -o -d "$git_dir/rebase-apply"',
        check=False,
    )
    return result.returncode == 0


def rebase(
    repo: JobRepository,
    job_id: str,
    *,
    target_ref: str = "origin/main",
    on_progress: ProgressCallback | None = None,
) -> RebaseInfo:
    """Rebase a job, leaving conflicts in place for interactive resolution."""
    progress = on_progress or _noop_progress
    job = repo.get(job_id)
    backend = backend_for_job(job)
    workspace = backend.workspace
    worktree = f"{workspace}/jobs/{job_id}/{get_profile(job.repo).dir_name}"

    worktree_check = backend.run(
        f"test -d {worktree}/.git || test -f {worktree}/.git", check=False
    )
    if worktree_check.returncode != 0:
        raise PtqError(f"No worktree found at {worktree}")

    info = RebaseInfo(
        state=RebaseState.RUNNING,
        target_ref=target_ref,
        before_sha=_get_sha(backend, worktree),
    )
    repo.save_rebase(job_id, info.to_dict())
    progress(f"Starting rebase onto {target_ref} (from {info.before_sha[:10]})")

    progress("Fetching latest from origin...")
    backend.run(f"git -C {worktree} fetch origin", check=False)
    target_exists = backend.run(
        f"git -C {worktree} rev-parse --verify {target_ref}", check=False
    )
    if target_exists.returncode != 0:
        info.state = RebaseState.FAILED
        info.error = f"Target ref not found: {target_ref}"
        repo.save_rebase(job_id, info.to_dict())
        raise PtqError(info.error)

    progress("Running git rebase...")
    result = backend.run(f"git -C {worktree} rebase {target_ref}", check=False)
    if result.returncode == 0:
        info.state = RebaseState.SUCCEEDED
        info.after_sha = _get_sha(backend, worktree)
        repo.save_rebase(job_id, info.to_dict())
        progress(f"Rebase clean — now at {info.after_sha[:10]}")
        return info

    conflicts = _conflicted_files(backend, worktree)
    if conflicts or _is_rebase_in_progress(backend, worktree):
        detail = ", ".join(conflicts) if conflicts else "rebase stopped"
        info.state = RebaseState.NEEDS_HUMAN
        info.error = f"Resolve interactively in the job workspace: {detail}"
        repo.save_rebase(job_id, info.to_dict())
        progress(info.error)
        return info

    info.state = RebaseState.FAILED
    info.error = result.stderr.strip() or "Rebase failed for unknown reason"
    repo.save_rebase(job_id, info.to_dict())
    raise PtqError(f"Rebase failed: {info.error}")
