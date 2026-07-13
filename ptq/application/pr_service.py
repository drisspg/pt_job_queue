from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass

from ptq.domain.models import JobRecord, PRResult, PtqError
from ptq.infrastructure.backends import backend_for_job
from ptq.infrastructure.job_repository import JobRepository
from ptq.repo_profiles import get_profile
from ptq.ssh import Backend

_HTTPS_TO_SSH = {
    "https://github.com/": "git@github.com:",
}

_PR_STATE_TTL_SECONDS = 45.0
_pr_state_cache: dict[str, tuple[float, str]] = {}
_JELLYFISH_FIELD_LABEL_RE = re.compile(
    r"(?m)^([ \t]*(?:Approved by|Differential Revision|Pull Request resolved|"
    r"Reviewed By|Reviewed-by|Reviewers|Rollback Plan|Subscribers|Summary|Tags|"
    r"Task|Tasks|Test Plan|Tested By|Title)):"
)
_PR_TITLE_ARTIFACT = "pr_title.txt"
_MAX_PR_TITLE_CHARS = 200


@dataclass(frozen=True)
class PRDefaults:
    title: str
    human_note: str
    synced_from_github: bool = False
    human_note_synced_from_github: bool = False


@dataclass(frozen=True)
class CurrentPRMetadata:
    title: str | None = None
    human_note: str | None = None
    url: str = ""
    fetched: bool = False


def _read_file(backend: Backend, path: str) -> str:
    result = backend.run(f"cat {path}", check=False)
    if result.returncode == 0 and isinstance(result.stdout, str):
        return result.stdout.strip()
    return ""


def _fallback_pr_title(job: JobRecord) -> str:
    return f"Fix #{job.issue}" if job.issue is not None else f"Fix from {job.job_id}"


def _normalize_pr_title(title: str) -> str:
    """Extract a safe one-line PR title from human or agent-authored text."""
    for line in title.splitlines():
        candidate = line.strip()
        if not candidate:
            continue
        lower_candidate = candidate.lower()
        for prefix in ("pr title:", "title:"):
            if lower_candidate.startswith(prefix):
                candidate = candidate[len(prefix) :].strip()
                break
        if candidate:
            return candidate[:_MAX_PR_TITLE_CHARS]
    return ""


def _artifact_pr_title(backend: Backend, job_dir: str) -> str:
    return _normalize_pr_title(
        _read_file(backend, f"{job_dir}/{_PR_TITLE_ARTIFACT}")
    )


def _resolve_pr_title(
    job: JobRecord, backend: Backend, job_dir: str, title: str | None
) -> str:
    """Apply the PR title precedence shared by CLI, web, and PR creation."""
    explicit_title = _normalize_pr_title(title or "")
    if explicit_title:
        return explicit_title
    if job.pr_title:
        saved_title = _normalize_pr_title(job.pr_title)
        if saved_title:
            return saved_title
    artifact_title = _artifact_pr_title(backend, job_dir)
    if artifact_title:
        return artifact_title
    return _fallback_pr_title(job)


def _extract_human_note(body: str) -> str | None:
    """Extract the editable Human Note section from a PTQ-created PR body."""
    match = re.search(r"(?m)^## Human Note\s*$", body)
    if not match:
        return None
    rest = body[match.end() :].lstrip("\r\n")
    markers = [
        re.search(pattern, rest)
        for pattern in (
            r"(?m)^##\s+",
            r"(?m)^Fixes #\d+\s*$",
            r"(?m)^<details>\s*$",
            r"(?m)^---\s*$",
        )
    ]
    ends = [marker.start() for marker in markers if marker]
    if ends:
        rest = rest[: min(ends)]
    return rest.strip()


def _fetch_open_pr_metadata(
    job: JobRecord,
    backend: Backend,
    log: Callable[[str], None] | None = None,
) -> CurrentPRMetadata:
    """Read title and Human Note from GitHub when the saved PR is still open."""
    if not job.pr_url:
        return CurrentPRMetadata()

    _log = log or (lambda _: None)
    pr_state = get_pr_state(backend, job.pr_url, force_refresh=True)
    match pr_state:
        case "open":
            result = backend.run(
                f"gh pr view '{job.pr_url}' --json title,body",
                check=False,
            )
            if result.returncode != 0 or not result.stdout.strip():
                _log("Could not fetch current GitHub PR metadata.")
                return CurrentPRMetadata(url=job.pr_url)
            try:
                data = json.loads(result.stdout)
            except json.JSONDecodeError:
                _log("Could not parse current GitHub PR metadata.")
                return CurrentPRMetadata(url=job.pr_url)
            return CurrentPRMetadata(
                title=_normalize_pr_title(str(data.get("title") or "")) or None,
                human_note=_extract_human_note(str(data.get("body") or "")),
                url=job.pr_url,
                fetched=True,
            )
        case "closed" | "merged":
            _log(f"Stored PR is {pr_state}. Creating a new PR from current branch.")
        case _:
            _log("Stored PR state is unknown. Proceeding with create-or-find flow.")
    return CurrentPRMetadata()


def sync_pr_metadata(
    repo: JobRepository,
    job_id: str,
    *,
    job: JobRecord | None = None,
    backend: Backend | None = None,
    log: Callable[[str], None] | None = None,
) -> CurrentPRMetadata:
    """Persist current open GitHub PR metadata so reruns do not clobber edits."""
    job = job or repo.get(job_id)
    backend = backend or backend_for_job(job)
    metadata = _fetch_open_pr_metadata(job, backend, log)
    updated = False
    if metadata.title is not None:
        job.pr_title = metadata.title or None
        updated = True
    if metadata.fetched:
        job.human_note = metadata.human_note or None
        updated = True
    if updated:
        repo.save(job)
    return metadata


def pr_defaults(repo: JobRepository, job_id: str) -> PRDefaults:
    """Return PR title/note defaults after syncing any open GitHub PR edits."""
    job = repo.get(job_id)
    backend = backend_for_job(job)
    metadata = sync_pr_metadata(repo, job_id, job=job, backend=backend)
    job_dir = f"{backend.workspace}/jobs/{job_id}"
    return PRDefaults(
        title=_resolve_pr_title(job, backend, job_dir, None),
        human_note=job.human_note or "",
        synced_from_github=metadata.fetched,
        human_note_synced_from_github=metadata.human_note is not None,
    )


def suggest_pr_title(repo: JobRepository, job_id: str) -> str:
    """Return the PR title default humans see before creating or updating a PR."""
    return pr_defaults(repo, job_id).title


def _normalize_pr_state(raw_state: str, merged_at: str) -> str:
    match raw_state.upper():
        case "OPEN":
            return "open"
        case "MERGED":
            return "merged"
        case "CLOSED":
            return "merged" if merged_at else "closed"
        case _:
            return "unknown"


def _escape_jellyfish_fields(text: str) -> str:
    """Keep nested PR artifact prose from becoming structured Jellyfish fields."""
    return _JELLYFISH_FIELD_LABEL_RE.sub(r"\1&#58;", text)


def get_pr_state(
    backend: Backend,
    pr_url: str,
    *,
    force_refresh: bool = False,
    ttl_seconds: float = _PR_STATE_TTL_SECONDS,
) -> str:
    if not pr_url:
        return "unknown"

    now = time.monotonic()
    cached = _pr_state_cache.get(pr_url)
    if not force_refresh and cached and now - cached[0] < ttl_seconds:
        return cached[1]

    result = backend.run(
        f"gh pr view '{pr_url}' --json state,mergedAt --jq '[.state, (.mergedAt // \"\")] | @tsv'",
        check=False,
    )
    if result.returncode != 0:
        state = "unknown"
    else:
        raw_state, _, merged_at = result.stdout.strip().partition("\t")
        state = _normalize_pr_state(raw_state.strip(), merged_at.strip())
    _pr_state_cache[pr_url] = (now, state)
    return state


def _build_pr_body(
    report: str,
    worklog: str,
    repro: str,
    issue_number: int | None,
    human_note: str,
) -> str:
    parts: list[str] = [
        "## Human Note",
        _escape_jellyfish_fields(human_note),
    ]
    if report:
        parts.append(f"\n## Agent Report\n{_escape_jellyfish_fields(report)}")
    if issue_number is not None:
        parts.append(f"\n\nFixes #{issue_number}")
    if repro:
        parts.append(
            f"\n\n<details>\n<summary>Repro Script</summary>\n\n```python\n{repro}\n```\n\n</details>"
        )
    if worklog:
        parts.append(
            "\n\n<details>\n<summary>Agent Worklog</summary>\n\n"
            f"{_escape_jellyfish_fields(worklog)}\n\n</details>"
        )
    parts.append(
        "\n---\n*This PR was generated by [ptq](https://github.com/drisspg/pt_job_queue) "
        "with human review.*"
    )
    return "\n".join(parts)


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
    repo: JobRepository,
    job_id: str,
    *,
    human_note: str | None,
    title: str | None = None,
    draft: bool = False,
    log: Callable[[str], None] | None = None,
) -> PRResult:
    _log = log or (lambda _: None)
    job = repo.get(job_id)
    backend = backend_for_job(job)
    job_dir = f"{backend.workspace}/jobs/{job_id}"
    profile = get_profile(job.repo)
    worktree = f"{job_dir}/{profile.dir_name}"
    existing_pr = sync_pr_metadata(repo, job_id, job=job, backend=backend, log=_log)
    existing_open_pr_url = existing_pr.url
    if existing_open_pr_url:
        _log(f"Existing PR is open: {existing_open_pr_url}")

    resolved_human_note = human_note.strip() if human_note else ""
    if not resolved_human_note:
        resolved_human_note = job.human_note or ""
    if not resolved_human_note:
        raise PtqError(
            "A human note is required. Describe what this PR does, "
            "why you believe it's correct, and how the reviewer should approach it."
        )

    branch = f"ptq/{job.issue}" if job.issue is not None else f"ptq/{job_id}"
    pr_title = _resolve_pr_title(job, backend, job_dir, title)

    _log(f"Branch: {branch}")
    _log(f"Title: {pr_title}")

    report = _read_file(backend, f"{job_dir}/report.md")
    worklog = _read_file(backend, f"{job_dir}/worklog.md")
    repro = _read_file(backend, f"{job_dir}/repro.py")
    body = _build_pr_body(report, worklog, repro, job.issue, resolved_human_note)
    _log(
        f"PR body: report.md {'found' if report else 'missing'}, "
        f"worklog.md {'found' if worklog else 'missing'}, "
        f"repro.py {'found' if repro else 'missing'}"
    )

    commit_msg = pr_title.replace("'", "'\\''")
    _log(f"Checking out branch {branch}...")
    backend.run(f"cd {worktree} && git checkout -B '{branch}'")
    _log("Staging changes...")
    backend.run(f"cd {worktree} && git add -A")
    has_staged_changes = (
        backend.run(
            f"cd {worktree} && git diff --cached --quiet", check=False
        ).returncode
        != 0
    )
    if has_staged_changes:
        _log("Creating commit...")
        commit_result = backend.run(
            f"cd {worktree} && git commit -m '{commit_msg}'",
            check=False,
        )
        if commit_result.returncode != 0:
            stderr = commit_result.stderr.strip() if commit_result.stderr else ""
            stdout = commit_result.stdout.strip() if commit_result.stdout else ""
            raise PtqError(f"git commit failed: {stderr or stdout or 'unknown error'}")
    else:
        _log("No staged changes to commit.")

    _ensure_ssh_remote(backend, worktree, _log)
    _log("Pushing branch...")
    push_result = backend.run(
        f"cd {worktree} && git push -u origin '{branch}'",
        check=False,
    )
    if push_result.returncode != 0:
        stderr = push_result.stderr.strip() if push_result.stderr else ""
        stdout = push_result.stdout.strip() if push_result.stdout else ""
        raise PtqError(f"git push failed: {stderr or stdout or 'unknown error'}")
    body_escaped = body.replace("'", "'\\''")
    url = ""
    if existing_open_pr_url:
        _log("Updating existing PR...")
        edit_result = backend.run(
            f"cd {worktree} && "
            f"gh pr edit '{existing_open_pr_url}' "
            f"--title '{commit_msg}' "
            f"--body '{body_escaped}'",
            check=False,
        )
        if edit_result.returncode == 0:
            url = existing_open_pr_url
        else:
            _log("Could not update existing PR, falling back to create-or-find flow.")

    result = None
    if not url:
        _log("Creating PR...")
        result = backend.run(
            f"cd {worktree} && "
            f"gh pr create "
            f"--title '{commit_msg}' "
            f"--body '{body_escaped}' "
            f"--head '{branch}'"
            f"{' --draft' if draft else ''}",
            check=False,
        )
        for line in result.stdout.strip().splitlines():
            if line.startswith("http"):
                url = line.strip()
                break

    if not url and result and result.returncode != 0:
        stderr = result.stderr.strip() if result.stderr else ""
        if "already exists" in stderr:
            url = backend.run(
                f"cd {worktree} && gh pr list --head '{branch}' --state open --json url --jq '.[0].url'",
                check=False,
            ).stdout.strip()
            if url:
                _log("Updating existing PR body...")
                backend.run(
                    f"cd {worktree} && "
                    f"gh pr edit '{branch}' "
                    f"--title '{commit_msg}' "
                    f"--body '{body_escaped}'",
                    check=False,
                )
        if not url:
            raise PtqError(f"gh pr create failed: {stderr or result.stdout}")

    if url:
        job.pr_url = url
        job.human_note = resolved_human_note
        job.pr_title = pr_title
        repo.save(job)

    return PRResult(url=url, branch=branch)
