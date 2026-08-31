from __future__ import annotations

import configparser
import re
import shlex
from collections.abc import Callable
from dataclasses import dataclass
from subprocess import CompletedProcess

from ptq.domain.models import JobRecord, PtqError, SubmissionMode
from ptq.infrastructure.backends import backend_for_job
from ptq.infrastructure.job_repository import JobRepository
from ptq.ssh import Backend
from ptq.takeover import shell_path, worktree_path

_FIELD_SEPARATOR = "\x1f"
_RECORD_SEPARATOR = "\x1e"
_SOURCE_ID_RE = re.compile(r"(?m)^ghstack-source-id:\s*(\S+)\s*$")
_PULL_REQUEST_RE = re.compile(
    r"(?m)^(?:Pull-Request|Pull Request resolved|Pull-Request-resolved):"
    r"\s*(https://\S+)\s*$"
)
_STACK_CONTEXT_FILE = "STACK_CONTEXT.md"


@dataclass(frozen=True)
class StackCommit:
    """A local commit and its optional ghstack identity."""

    sha: str
    subject: str
    source_id: str = ""
    pr_url: str = ""


@dataclass(frozen=True)
class StackStatus:
    """Preflight state for the commits PTQ would submit as one stack."""

    branch: str
    base: str
    remote: str
    base_ref: str
    dirty: bool
    has_merges: bool
    commits: tuple[StackCommit, ...]


@dataclass(frozen=True)
class StackSubmitResult:
    """The submitted stack and ghstack's user-facing output."""

    status: StackStatus
    output: str


def run_git(backend: Backend, worktree: str, args: str) -> CompletedProcess[str]:
    """Run a git command in a job worktree without interpolating its path unsafely."""
    return backend.run(
        f"cd {shell_path(worktree)} && git {args}",
        check=False,
    )


def checked_git(backend: Backend, worktree: str, args: str) -> str:
    """Run a mandatory git query and surface failures as PTQ errors."""
    result = run_git(backend, worktree, args)
    if result.returncode != 0:
        raise PtqError(result.stderr.strip() or f"git {args} failed.")
    return result.stdout


def ghstack_remote(backend: Backend, worktree: str) -> str:
    """Read the remote from the same repository-or-user config ghstack selects."""
    command = (
        f"search={shell_path(worktree)}; "
        "while :; do "
        'if test -f "$search/.ghstackrc"; then printf %s "$search/.ghstackrc"; exit; fi; '
        'parent=$(dirname -- "$search"); test "$parent" = "$search" && break; '
        'search="$parent"; done; '
        'printf %s "${GHSTACKRC_PATH:-$HOME/.ghstackrc}"'
    )
    config_path = backend.run(command, check=False).stdout.strip()
    contents = backend.run(f"cat {shell_path(config_path)}", check=False)
    if contents.returncode != 0:
        return "origin"

    config = configparser.ConfigParser()
    try:
        config.read_string(contents.stdout)
    except configparser.Error as error:
        raise PtqError(
            f"Could not parse ghstack config {config_path}: {error}"
        ) from error
    return config.get("ghstack", "remote_name", fallback="origin")


def resolve_base_ref(backend: Backend, worktree: str, base: str, remote: str) -> str:
    """Require the exact remote-tracking base that ghstack itself will use."""
    remote_ref = f"{remote}/{base}"
    result = run_git(
        backend,
        worktree,
        f"rev-parse --verify --quiet {shlex.quote(remote_ref)}",
    )
    if result.returncode != 0:
        raise PtqError(f"Could not resolve ghstack base {remote_ref!r} in {worktree}.")
    return remote_ref


def parse_commits(raw: str) -> tuple[StackCommit, ...]:
    """Parse git's delimiter-separated log format without altering commit bodies."""
    commits: list[StackCommit] = []
    for record in raw.split(_RECORD_SEPARATOR):
        record = record.strip("\n")
        if not record:
            continue
        fields = record.split(_FIELD_SEPARATOR, 2)
        if len(fields) != 3:
            raise PtqError("Could not parse the local commit stack.")
        sha, subject, body = fields
        source_id = _SOURCE_ID_RE.search(body)
        pull_request = _PULL_REQUEST_RE.search(body)
        commits.append(
            StackCommit(
                sha=sha.strip(),
                subject=subject.strip(),
                source_id=source_id.group(1) if source_id else "",
                pr_url=pull_request.group(1) if pull_request else "",
            )
        )
    return tuple(commits)


def inspect_stack(
    repo: JobRepository,
    job_id: str,
    *,
    base: str | None = None,
) -> StackStatus:
    """Inspect the exact local commits and repository state ghstack would use."""
    job = repo.get(job_id)
    base = base or job.stack_base
    backend = backend_for_job(job)
    worktree = worktree_path(backend.workspace, job.job_id, job.repo)
    remote = ghstack_remote(backend, worktree)
    base_ref = resolve_base_ref(backend, worktree, base, remote)

    branch_result = run_git(backend, worktree, "symbolic-ref --short --quiet HEAD")
    if branch_result.returncode not in {0, 1}:
        raise PtqError(branch_result.stderr.strip() or "Could not inspect HEAD.")
    branch = branch_result.stdout.strip() if branch_result.returncode == 0 else ""
    dirty = bool(checked_git(backend, worktree, "status --porcelain").strip())
    revision_range = f"{shlex.quote(base_ref)}..HEAD"
    has_merges = bool(
        checked_git(backend, worktree, f"rev-list --merges {revision_range}").strip()
    )
    log_format = "%H%x1f%s%x1f%B%x1e"
    log_output = checked_git(
        backend,
        worktree,
        f"log --reverse --format={shlex.quote(log_format)} {revision_range}",
    )
    return StackStatus(
        branch=branch,
        base=base,
        remote=remote,
        base_ref=base_ref,
        dirty=dirty,
        has_merges=has_merges,
        commits=parse_commits(log_output),
    )


def validate_submit(status: StackStatus) -> None:
    """Reject states where ghstack could publish unintended or unstable history."""
    if status.dirty:
        raise PtqError(
            "The worktree has uncommitted changes. Commit each reviewable change "
            "before submitting the stack."
        )
    if not status.branch:
        raise PtqError(
            "The worktree is on a detached HEAD. Create or switch to a local branch "
            "before submitting the stack."
        )
    if not status.commits:
        raise PtqError(f"There are no commits above {status.base_ref} to submit.")
    if status.has_merges:
        raise PtqError(
            "ghstack submission requires linear history without merge commits."
        )


def persist_submitted_stack(
    repo: JobRepository,
    job_id: str,
    commits: tuple[StackCommit, ...],
    *,
    update_metadata: bool = False,
) -> None:
    """Record the top PR after verifying ghstack wrote every commit mapping."""
    missing = [
        commit.subject
        for commit in commits
        if not commit.source_id or not commit.pr_url
    ]
    if missing:
        raise PtqError(
            "ghstack succeeded but did not write stack trailers for: "
            + ", ".join(missing)
        )

    job = repo.get(job_id)
    top = commits[-1]
    top_pr_changed = job.pr_url != top.pr_url
    job.pr_url = top.pr_url
    if update_metadata or top_pr_changed or not job.pr_title:
        job.pr_title = top.subject
    repo.save(job)


def stack_context(job_id: str, branch: str, base: str) -> str:
    """Render the durable agent handoff for a job intended for ghstack."""
    return f"""# PTQ ghstack context

This job is configured for a ghstack submission targeting `{base}` from branch `{branch}`.

- Build a linear sequence of independently reviewable commits.
- Keep each commit buildable and tested; include a feature's tests in its owning commit.
- Treat each commit subject/body as that PR's initial title and description.
- Use `## Human Note`, `## Agent note`, and `## Test Plan` sections in each commit body.
- Preserve `ghstack-source-id`, `ghstack-comment-id`, and `Pull-Request` trailers during rewrites.
- Use `uv run ptq stack show {job_id}` for read-only preflight.
- Use `uv run ptq stack submit {job_id}` to create or update the stack.
- Do not use `uv run ptq pr {job_id}` for this job.
- Do not update existing PR metadata unless the user explicitly requests it.
"""


def persist_stack_intent(
    repo: JobRepository,
    job: JobRecord,
    backend: Backend,
    *,
    branch: str,
    base: str,
) -> None:
    """Record ghstack intent for both PTQ monitoring and job-local agents."""
    job_dir = f"{backend.workspace}/jobs/{job.job_id}"
    content = stack_context(job.job_id, branch, base)
    result = backend.run(
        f"printf %s {shlex.quote(content)} > {shell_path(f'{job_dir}/{_STACK_CONTEXT_FILE}')}",
        check=False,
    )
    if result.returncode != 0:
        raise PtqError(result.stderr.strip() or "Could not write ghstack job context.")

    job.submission_mode = SubmissionMode.GHSTACK
    job.stack_base = base
    repo.save(job)


def initialize_stack(
    repo: JobRepository,
    job_id: str,
    *,
    base: str | None = None,
) -> StackStatus:
    """Mark a job for ghstack and move detached/base HEAD onto its stack branch."""
    job = repo.get(job_id)
    if job.pr_url and job.submission_mode != SubmissionMode.GHSTACK:
        raise PtqError(
            f"Job {job_id} already has a conventional PR; create a separate stack job."
        )
    status = inspect_stack(repo, job_id, base=base)
    base = status.base
    backend = backend_for_job(job)
    worktree = worktree_path(backend.workspace, job.job_id, job.repo)
    needs_stack_branch = not status.branch or status.branch == base
    if needs_stack_branch:
        if status.dirty:
            raise PtqError(
                "The worktree is dirty and not on a stack branch. Commit, stash, "
                "or clean the changes before initializing ghstack."
            )
        branch = f"ptq-stack/{job.job_id}"
        branch_ref = f"refs/heads/{branch}"
        current_head = checked_git(backend, worktree, "rev-parse HEAD").strip()
        existing = run_git(
            backend,
            worktree,
            f"show-ref --verify --quiet {shlex.quote(branch_ref)}",
        )
        switch_args = f"switch {shlex.quote(branch)}"
        if existing.returncode == 0:
            branch_head = checked_git(
                backend, worktree, f"rev-parse {shlex.quote(branch_ref)}"
            ).strip()
            if branch_head != current_head:
                raise PtqError(
                    f"Stack branch {branch!r} already points to a different commit."
                )
        elif existing.returncode == 1:
            switch_args = f"switch -c {shlex.quote(branch)}"
        else:
            raise PtqError(existing.stderr.strip() or f"Could not inspect {branch!r}.")
        switched = run_git(backend, worktree, switch_args)
        if switched.returncode != 0:
            raise PtqError(
                switched.stderr.strip() or f"Could not switch to stack branch {branch}."
            )
        status = inspect_stack(repo, job_id, base=base)

    persist_stack_intent(
        repo,
        job,
        backend,
        branch=status.branch,
        base=base,
    )
    return status


def ghstack_executable(backend: Backend) -> str:
    """Find ghstack on PATH or in the shared development virtualenv."""
    result = backend.run(
        "command -v ghstack || "
        '{ test -x "$HOME/.venvs/dev/bin/ghstack" && '
        'printf %s "$HOME/.venvs/dev/bin/ghstack"; }',
        check=False,
    )
    executable = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
    if result.returncode != 0 or not executable:
        raise PtqError("ghstack is not installed on the job target.")
    return executable


def submit_stack(
    repo: JobRepository,
    job_id: str,
    *,
    draft: bool = False,
    update_metadata: bool = False,
    stream_output: bool = False,
    log: Callable[[str], None] | None = None,
) -> StackSubmitResult:
    """Submit an initialized, clean, linear PTQ job stack through ghstack."""
    _log = log or (lambda _: None)
    job = repo.get(job_id)
    if job.submission_mode != SubmissionMode.GHSTACK:
        raise PtqError(f"Run `ptq stack init {job_id}` before submitting this job.")

    backend = backend_for_job(job)
    worktree = worktree_path(backend.workspace, job.job_id, job.repo)
    _log("Resolving ghstack configuration...")
    executable = ghstack_executable(backend)
    remote = ghstack_remote(backend, worktree)
    _log(f"Fetching {remote}...")
    checked_git(backend, worktree, f"fetch --prune {shlex.quote(remote)}")
    status = inspect_stack(repo, job_id)
    validate_submit(status)
    _log(
        f"Submitting {len(status.commits)} commit(s) from {status.branch} "
        f"onto {status.base_ref}..."
    )

    args = [executable, "submit", "--stack", "-B", status.base]
    if draft:
        args.append("--draft")
    if update_metadata:
        args.append("--update-fields")
    args.append("HEAD")
    command = " ".join(shlex.quote(arg) for arg in args)
    submit_command = f"cd {shell_path(worktree)} && {command}"
    if stream_output:
        result = backend.run(submit_command, check=False, stream=True)
    else:
        result = backend.run(submit_command, check=False)
    stdout = result.stdout.strip() if isinstance(result.stdout, str) else ""
    stderr = result.stderr.strip() if isinstance(result.stderr, str) else ""
    if result.returncode != 0:
        raise PtqError(
            f"ghstack submit failed: {stderr or stdout or 'see output above'}"
        )

    _log("Refreshing submitted stack metadata...")
    submitted = inspect_stack(repo, job_id)
    persist_submitted_stack(
        repo,
        job_id,
        submitted.commits,
        update_metadata=update_metadata,
    )
    return StackSubmitResult(status=submitted, output=stdout)
