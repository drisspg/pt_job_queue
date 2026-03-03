from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ptq.job import get_job
from ptq.ssh import backend_for_job

if TYPE_CHECKING:
    from ptq.ssh import Backend


@dataclass
class PRResult:
    url: str
    branch: str


def _read_file(backend: Backend, path: str) -> str:
    result = backend.run(f"cat {path}", check=False)
    if result.returncode == 0:
        return result.stdout.strip()
    return ""


def _build_pr_body(report: str, worklog: str, issue_number: int | None) -> str:
    parts: list[str] = []
    if report:
        parts.append(report)
    if issue_number is not None:
        parts.append(f"\n\nFixes #{issue_number}")
    if worklog:
        parts.append(
            f"\n\n<details>\n<summary>Worklog</summary>\n\n{worklog}\n\n</details>"
        )
    return "\n".join(parts) if parts else "Automated fix from ptq."


_HTTPS_TO_SSH = {
    "https://github.com/": "git@github.com:",
}


def _ensure_ssh_remote(
    backend: Backend, worktree: str, _log: Callable[[str], None]
) -> None:
    result = backend.run(f"cd {worktree} && git remote get-url origin", check=False)
    url = result.stdout.strip()
    for https_prefix, ssh_prefix in _HTTPS_TO_SSH.items():
        if url.startswith(https_prefix):
            ssh_url = url.replace(https_prefix, ssh_prefix)
            if not ssh_url.endswith(".git"):
                ssh_url += ".git"
            _log(f"Switching origin to SSH: {ssh_url}")
            backend.run(f"cd {worktree} && git remote set-url origin '{ssh_url}'")
            return


def create_pr(
    job_id: str,
    *,
    title: str | None = None,
    draft: bool = False,
    log: Callable[[str], None] | None = None,
) -> PRResult:
    _log = log or (lambda _: None)
    job = get_job(job_id)
    backend = backend_for_job(job_id)
    ws = backend.workspace
    job_dir = f"{ws}/jobs/{job_id}"
    worktree = f"{job_dir}/pytorch"
    issue_number = job.get("issue")

    branch = f"ptq/{issue_number}" if issue_number is not None else f"ptq/{job_id}"
    pr_title = title or (
        f"Fix #{issue_number}" if issue_number is not None else f"Fix from {job_id}"
    )

    _log(f"Branch: {branch}")
    _log(f"Title: {pr_title}")

    report = _read_file(backend, f"{job_dir}/report.md")
    worklog = _read_file(backend, f"{job_dir}/worklog.md")
    body = _build_pr_body(report, worklog, issue_number)
    _log(
        f"PR body: report.md {'found' if report else 'missing'}, worklog.md {'found' if worklog else 'missing'}"
    )

    _SCRUB_PATHS = ".claude/ .cursorrules AGENTS.md"

    _log("Staging changes...")
    backend.run(f"cd {worktree} && git add -A")

    commit_msg = pr_title.replace("'", "'\\''")
    _log(f"Creating branch {branch} (single commit)...")
    base = backend.run(
        f"cd {worktree} && git merge-base HEAD origin/main", check=False
    ).stdout.strip()
    backend.run(f"cd {worktree} && git checkout -B '{branch}'")
    if base:
        backend.run(f"cd {worktree} && git reset --soft {base}")
    backend.run(f"cd {worktree} && git reset HEAD -- {_SCRUB_PATHS}", check=False)
    backend.run(
        f"cd {worktree} && git commit -m '{commit_msg}' --allow-empty",
        check=False,
    )

    _ensure_ssh_remote(backend, worktree, _log)
    _log("Pushing and creating PR...")
    body_escaped = body.replace("'", "'\\''")
    result = backend.run(
        f"cd {worktree} && git push -u origin '{branch}' --force && "
        f"gh pr create "
        f"--title '{commit_msg}' "
        f"--body '{body_escaped}' "
        f"--head '{branch}'"
        f"{' --draft' if draft else ''}",
        check=False,
    )

    url = ""
    for line in result.stdout.strip().splitlines():
        if line.startswith("http"):
            url = line.strip()
            break

    if not url and result.returncode != 0:
        stderr = result.stderr.strip() if result.stderr else ""
        if "already exists" in stderr:
            list_result = backend.run(
                f"cd {worktree} && gh pr list --head '{branch}' --json url --jq '.[0].url'",
                check=False,
            )
            url = list_result.stdout.strip()
            if url:
                return PRResult(url=url, branch=branch)
        raise SystemExit(f"gh pr create failed: {stderr or result.stdout}")

    return PRResult(url=url, branch=branch)
