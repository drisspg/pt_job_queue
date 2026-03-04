from __future__ import annotations

import tempfile
import time
from collections.abc import Callable
from pathlib import Path

from ptq.agents import RunContext, get_agent
from ptq.domain.models import PtqError, RebaseInfo, RebaseState
from ptq.infrastructure.backends import backend_for_job
from ptq.infrastructure.job_repository import JobRepository
from ptq.ssh import Backend, RemoteBackend

ProgressCallback = Callable[[str], None]

_REBASE_PROMPT_TEMPLATE = (
    Path(__file__).parent.parent.parent / "prompts" / "rebase_conflict.md"
).read_text()


def _noop_progress(_msg: str) -> None:
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
    return [f.strip() for f in result.stdout.strip().splitlines() if f.strip()]


def _is_rebase_in_progress(backend: Backend, worktree: str) -> bool:
    return (
        backend.run(
            f"test -d {worktree}/.git/rebase-merge -o -d {worktree}/.git/rebase-apply",
            check=False,
        ).returncode
        == 0
    )


def _build_conflict_prompt(
    job_id: str,
    workspace: str,
    worktree_path: str,
    target_ref: str,
    attempt: int,
    max_attempts: int,
    conflict_files: list[str],
) -> str:
    return _REBASE_PROMPT_TEMPLATE.format(
        job_id=job_id,
        workspace=workspace,
        worktree_path=worktree_path,
        target_ref=target_ref,
        attempt=attempt,
        max_attempts=max_attempts,
        conflict_files="\n".join(f"- `{f}`" for f in conflict_files),
    )


def _launch_conflict_agent(
    backend: Backend,
    job_id: str,
    workspace: str,
    worktree_path: str,
    job_dir: str,
    agent_name: str,
    model: str,
    prompt: str,
    attempt: int,
    progress: ProgressCallback,
) -> None:
    agent = get_agent(agent_name)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(prompt)
        prompt_tmp = Path(f.name)

    prompt_remote = f"{job_dir}/rebase_prompt_{attempt}.md"
    backend.copy_to(prompt_tmp, prompt_remote)
    prompt_tmp.unlink()

    agent.setup_workspace(backend, worktree_path, job_dir, workspace)

    log_file = f"{job_dir}/rebase-{attempt}.log"
    unbuffer = "stdbuf -oL " if isinstance(backend, RemoteBackend) else ""
    ctx = RunContext(
        worktree_path=worktree_path,
        job_dir=job_dir,
        message="Resolve the git rebase conflicts as described in your system prompt. Do not abort the rebase.",
        model=model,
        max_turns=50,
        system_prompt_file=prompt_remote,
        unbuffer_prefix=unbuffer,
    )
    agent_cmd = agent.build_cmd(ctx)

    backend.run(f"touch {log_file}")
    pid = backend.launch_background(agent_cmd, log_file)

    if pid is None:
        raise PtqError("Failed to launch rebase conflict agent")

    progress(f"Agent started (pid {pid}), waiting for completion...")
    while backend.is_pid_alive(pid):
        time.sleep(5)
    progress("Agent finished.")


def rebase(
    repo: JobRepository,
    job_id: str,
    *,
    target_ref: str = "origin/main",
    agent_name: str | None = None,
    model: str | None = None,
    max_attempts: int = 3,
    on_progress: ProgressCallback | None = None,
) -> RebaseInfo:
    progress = on_progress or _noop_progress
    job = repo.get(job_id)
    backend = backend_for_job(job)
    workspace = backend.workspace
    job_dir = f"{workspace}/jobs/{job_id}"
    worktree = f"{job_dir}/pytorch"

    agent_name = agent_name or job.agent
    model = model or job.model

    worktree_check = backend.run(
        f"test -d {worktree}/.git || test -f {worktree}/.git", check=False
    )
    if worktree_check.returncode != 0:
        raise PtqError(f"No worktree found at {worktree}")

    before_sha = _get_sha(backend, worktree)
    info = RebaseInfo(
        state=RebaseState.RUNNING,
        target_ref=target_ref,
        before_sha=before_sha,
    )
    repo.save_rebase(job_id, info.to_dict())
    progress(f"Starting rebase onto {target_ref} (from {before_sha[:10]})")

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
    rebase_result = backend.run(f"git -C {worktree} rebase {target_ref}", check=False)

    if rebase_result.returncode == 0:
        info.state = RebaseState.SUCCEEDED
        info.after_sha = _get_sha(backend, worktree)
        repo.save_rebase(job_id, info.to_dict())
        progress(f"Rebase clean — now at {info.after_sha[:10]}")
        return info

    conflicts = _conflicted_files(backend, worktree)
    if not conflicts and not _is_rebase_in_progress(backend, worktree):
        info.state = RebaseState.FAILED
        stderr = rebase_result.stderr.strip() if rebase_result.stderr else ""
        info.error = stderr or "Rebase failed for unknown reason"
        repo.save_rebase(job_id, info.to_dict())
        raise PtqError(f"Rebase failed: {info.error}")

    progress(f"Conflicts in {len(conflicts)} file(s): {', '.join(conflicts)}")

    for attempt in range(1, max_attempts + 1):
        info.attempts = attempt
        repo.save_rebase(job_id, info.to_dict())
        progress(f"Conflict resolution attempt {attempt}/{max_attempts}...")

        prompt = _build_conflict_prompt(
            job_id=job_id,
            workspace=workspace,
            worktree_path=worktree,
            target_ref=target_ref,
            attempt=attempt,
            max_attempts=max_attempts,
            conflict_files=conflicts,
        )

        _launch_conflict_agent(
            backend=backend,
            job_id=job_id,
            workspace=workspace,
            worktree_path=worktree,
            job_dir=job_dir,
            agent_name=agent_name,
            model=model,
            prompt=prompt,
            attempt=attempt,
            progress=progress,
        )

        if not _is_rebase_in_progress(backend, worktree):
            info.state = RebaseState.SUCCEEDED
            info.after_sha = _get_sha(backend, worktree)
            repo.save_rebase(job_id, info.to_dict())
            progress(
                f"Rebase completed on attempt {attempt} — now at {info.after_sha[:10]}"
            )
            return info

        conflicts = _conflicted_files(backend, worktree)
        if not conflicts:
            progress("No conflicts remain, continuing rebase...")
            cont = backend.run(f"git -C {worktree} rebase --continue", check=False)
            if cont.returncode == 0 and not _is_rebase_in_progress(backend, worktree):
                info.state = RebaseState.SUCCEEDED
                info.after_sha = _get_sha(backend, worktree)
                repo.save_rebase(job_id, info.to_dict())
                progress(
                    f"Rebase completed on attempt {attempt} — now at {info.after_sha[:10]}"
                )
                return info
            conflicts = _conflicted_files(backend, worktree)
            if conflicts:
                progress(f"New conflicts after continue: {', '.join(conflicts)}")

    info.state = RebaseState.NEEDS_HUMAN
    info.error = f"Unresolved after {max_attempts} attempt(s): {', '.join(conflicts)}"
    repo.save_rebase(job_id, info.to_dict())
    progress(f"Escalating — conflicts remain after {max_attempts} attempt(s)")
    return info
