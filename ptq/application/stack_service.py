from __future__ import annotations

import configparser
import os
import re
import shlex
from collections.abc import Callable
from dataclasses import dataclass
from subprocess import CompletedProcess

from ptq.application.pr_service import conventional_pr_branch, get_pr_state
from ptq.domain.models import JobRecord, PtqError, StackTool, SubmissionMode
from ptq.infrastructure.backends import Backend, backend_for_job
from ptq.infrastructure.job_repository import JobRepository
from ptq.takeover import shell_path, worktree_path

_FIELD_SEPARATOR = "\x1f"
_RECORD_SEPARATOR = "\x1e"
_SOURCE_ID_RE = re.compile(r"(?m)^ghstack-source-id:\s*(\S+)\s*$")
_PULL_REQUEST_RE = re.compile(
    r"(?m)^(?:Pull-Request|Pull Request resolved|Pull-Request-resolved):"
    r"\s*(https://\S+)\s*$"
)
_STACK_INFO_RE = re.compile(
    r"(?m)^stack-info: PR: (https://\S+/pull/\d+), branch: \S+\s*$"
)
_STACK_CONTEXT_FILE = "STACK_CONTEXT.md"


@dataclass(frozen=True)
class StackCommit:
    """A local commit and its optional ghstack identity."""

    sha: str
    subject: str
    source_id: str = ""
    pr_url: str = ""
    stack_info: bool = False


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
        pull_request = _PULL_REQUEST_RE.search(body) or _STACK_INFO_RE.search(body)
        commits.append(
            StackCommit(
                sha=sha.strip(),
                subject=subject.strip(),
                source_id=source_id.group(1) if source_id else "",
                pr_url=pull_request.group(1) if pull_request else "",
                stack_info=bool(_STACK_INFO_RE.search(body)),
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


def sp_executable(backend: Backend) -> str:
    """Return the sp executable on the job target, or "" when it is not installed."""
    result = backend.run("command -v sp", check=False)
    return result.stdout.strip() if result.returncode == 0 else ""


def choose_stack_tool(backend: Backend) -> StackTool:
    """Prefer sp's native GitHub stacks when installed; PTQ_STACK_TOOL overrides."""
    configured = os.environ.get("PTQ_STACK_TOOL", "")
    if configured:
        try:
            return StackTool(configured)
        except ValueError:
            raise PtqError(
                f"PTQ_STACK_TOOL must be 'sp' or 'ghstack', not {configured!r}."
            ) from None
    return StackTool.SP if sp_executable(backend) else StackTool.GHSTACK


def submission_tool(job: JobRecord, commits: tuple[StackCommit, ...]) -> StackTool:
    """Keep a submitted stack on the tool whose trailers it carries."""
    ghstack = any(commit.source_id for commit in commits)
    native = any(commit.stack_info for commit in commits)
    if ghstack and native:
        raise PtqError(
            "The stack mixes ghstack and sp trailers; finish it with one tool."
        )
    if ghstack:
        return StackTool.GHSTACK
    if native:
        return StackTool.SP
    return job.stack_tool or StackTool.GHSTACK


def persist_submitted_stack(
    repo: JobRepository,
    job_id: str,
    commits: tuple[StackCommit, ...],
    *,
    update_metadata: bool = False,
    tool: StackTool = StackTool.GHSTACK,
) -> None:
    """Record the top PR after verifying the stack tool wrote every commit mapping."""
    missing = [
        commit.subject
        for commit in commits
        if not commit.pr_url or (tool == StackTool.GHSTACK and not commit.source_id)
    ]
    if missing:
        raise PtqError(
            f"{tool.value} succeeded but did not write stack trailers for: "
            + ", ".join(missing)
        )

    job = repo.get(job_id)
    top = commits[-1]
    top_pr_changed = job.pr_url != top.pr_url
    base_pr = [job.stack_base_pr_url] if job.stack_base_pr_url else []
    job.stack_pr_urls = base_pr + [commit.pr_url for commit in commits]
    job.stack_tool = tool
    job.pr_url = top.pr_url
    if update_metadata or top_pr_changed or not job.pr_title:
        job.pr_title = top.subject
    job.rebase = None
    repo.save(job)


def stack_context(
    job_id: str,
    branch: str,
    base: str,
    base_pr_url: str | None = None,
    tool: StackTool = StackTool.GHSTACK,
    adopted_pr_url: str | None = None,
) -> str:
    """Render the durable agent handoff for a stacked job."""
    if adopted_pr_url:
        base_pr_note = f"""
The bottom commit is the existing PR {adopted_pr_url}: its `stack-info` trailer maps it to that
PR's branch. Fold changes for that PR into the bottom commit, keep the trailer, and run
`uv run ptq stack submit {job_id}` to update every layer.
"""
    else:
        base_pr_note = (
            f"""
The stack sits on the existing conventional PR {base_pr_url} (branch `{base}`). Update that PR
with a plain `git push` of its branch, never `ptq pr`. Land it first; `uv run ptq rebase {job_id}`
then moves the stack onto `main` and `uv run ptq stack submit {job_id}` resubmits it.
"""
            if base_pr_url
            else ""
        )
    return f"""# PTQ {tool.value} stack context

This job is configured for a {tool.value} stack targeting `{base}` from branch `{branch}`.
{base_pr_note}
- Build a linear sequence of independently reviewable commits.
- Keep each commit buildable and tested; include a feature's tests in its owning commit.
- Treat each commit subject/body as that PR's initial title and description.
- Use `## Human Note`, `## Agent note`, and `## Test Plan` sections in each commit body.
- Preserve `stack-info`, `ghstack-source-id`, `ghstack-comment-id`, and `Pull-Request` trailers during rewrites.
- Use `uv run ptq stack show {job_id}` for read-only preflight.
- Use `uv run ptq stack submit {job_id}` to create or update the stack.
- Do not use `uv run ptq pr {job_id}` for this job.
- Do not update existing PR metadata unless the user explicitly requests it.
- For PyTorch, `@pytorchbot merge` lands the selected PR and every open PR below it.
- After landing only a lower prefix, follow the monitor's rebase and resubmit actions.
"""


def persist_stack_intent(
    repo: JobRepository,
    job: JobRecord,
    backend: Backend,
    *,
    branch: str,
    base: str,
    base_pr_url: str | None = None,
    tool: StackTool = StackTool.GHSTACK,
    adopted_pr_url: str | None = None,
) -> None:
    """Record stack intent for both PTQ monitoring and job-local agents."""
    job_dir = f"{backend.workspace}/jobs/{job.job_id}"
    content = stack_context(job.job_id, branch, base, base_pr_url, tool, adopted_pr_url)
    result = backend.run(
        f"printf %s {shlex.quote(content)} > {shell_path(f'{job_dir}/{_STACK_CONTEXT_FILE}')}",
        check=False,
    )
    if result.returncode != 0:
        raise PtqError(result.stderr.strip() or "Could not write ghstack job context.")

    job.submission_mode = SubmissionMode.GHSTACK
    job.stack_base = base
    job.stack_base_pr_url = base_pr_url
    job.stack_tool = tool
    repo.save(job)


def adopt_conventional_pr(
    repo: JobRepository, job: JobRecord, backend: Backend, worktree: str, trunk: str
) -> StackStatus:
    """Make a job's conventional PR the bottom sp layer instead of replacing it.

    sp maps one commit to one PR through a stack-info trailer naming the PR's head
    branch, so the PR's commits are squashed (same tree) into one trailered commit,
    which is pushed to the PR branch with a lease on its current head. Rerunning
    after a successful push resumes from the adopted commit.
    """
    branch = conventional_pr_branch(job)
    stack_branch = f"ptq-stack/{job.job_id}"
    on_trunk = inspect_stack(repo, job.job_id, base=trunk)
    if len(on_trunk.commits) == 1 and on_trunk.commits[0].pr_url == job.pr_url:
        if on_trunk.branch != stack_branch:
            checked_git(backend, worktree, f"switch -c {shlex.quote(stack_branch)}")
        return inspect_stack(repo, job.job_id, base=trunk)
    if get_pr_state(backend, job.pr_url or "", force_refresh=True) != "open":
        raise PtqError(f"{job.pr_url} is not open; it cannot become a stack layer.")
    on_pr = inspect_stack(repo, job.job_id, base=branch)
    head = checked_git(backend, worktree, "rev-parse HEAD").strip()
    pr_head = checked_git(
        backend, worktree, f"rev-parse {shlex.quote(on_pr.base_ref)}"
    ).strip()
    if on_pr.dirty or head != pr_head:
        raise PtqError(
            f"HEAD must be the clean, pushed head of {job.pr_url} ({on_pr.base_ref}) "
            "before initializing the stack."
        )
    trunk_ref = resolve_base_ref(backend, worktree, trunk, on_pr.remote)
    merge_base = checked_git(
        backend, worktree, f"merge-base {shlex.quote(trunk_ref)} HEAD"
    ).strip()
    pr_commits = checked_git(
        backend, worktree, f"rev-list --reverse {merge_base}..HEAD"
    ).split()
    if not pr_commits:
        raise PtqError(f"{on_pr.base_ref} has no commits above {trunk_ref}.")
    first = checked_git(backend, worktree, f"log -1 --format=%B {pr_commits[0]}")
    subject, _, body = first.strip().partition("\n")
    message = f"{job.pr_title or subject}\n{body}".rstrip()
    message += f"\n\nstack-info: PR: {job.pr_url}, branch: {branch}"
    adopted = checked_git(
        backend,
        worktree,
        f"commit-tree 'HEAD^{{tree}}' -p {merge_base} -m {shlex.quote(message)}",
    ).strip()
    # Create the stack branch first so an existing one fails before the PR is rewritten.
    checked_git(backend, worktree, f"branch {shlex.quote(stack_branch)} {adopted}")
    pushed = run_git(
        backend,
        worktree,
        f"push --force-with-lease={shlex.quote(f'refs/heads/{branch}:{head}')} "
        f"{shlex.quote(on_pr.remote)} {adopted}:{shlex.quote(f'refs/heads/{branch}')}",
    )
    if pushed.returncode != 0:
        run_git(backend, worktree, f"branch -D {shlex.quote(stack_branch)}")
        raise PtqError(pushed.stderr.strip() or f"Could not push {branch}.")
    checked_git(backend, worktree, f"switch {shlex.quote(stack_branch)}")
    return inspect_stack(repo, job.job_id, base=trunk)


def initialize_stack(
    repo: JobRepository,
    job_id: str,
    *,
    base: str | None = None,
) -> StackStatus:
    """Mark a job for stacking and move detached/base HEAD onto its stack branch.

    A job that already has a conventional PR keeps it. With sp, the PR becomes the
    stack's bottom layer. GitHub cannot move a PR onto a ghstack head branch, so a
    ghstack stack is based on the PR's branch instead and moves onto main once that
    PR lands.
    """
    job = repo.get(job_id)
    backend = backend_for_job(job)
    stacks_on_pr = bool(job.pr_url) and job.submission_mode != SubmissionMode.GHSTACK
    if job.submission_mode == SubmissionMode.GHSTACK:
        tool = job.stack_tool or StackTool.GHSTACK
    else:
        tool = choose_stack_tool(backend)
    if stacks_on_pr and tool == StackTool.SP:
        trunk = base or job.stack_base
        worktree = worktree_path(backend.workspace, job.job_id, job.repo)
        status = adopt_conventional_pr(repo, job, backend, worktree, trunk)
        persist_stack_intent(
            repo,
            job,
            backend,
            branch=status.branch,
            base=trunk,
            tool=tool,
            adopted_pr_url=job.pr_url,
        )
        return status

    base_pr_url = job.stack_base_pr_url
    if stacks_on_pr:
        base_pr_url = job.pr_url
        base = conventional_pr_branch(job)
    status = inspect_stack(repo, job_id, base=base)
    if stacks_on_pr and status.commits:
        raise PtqError(
            f"HEAD has commits that are not on {status.base_ref}; push them to "
            f"{base_pr_url} or move them off this branch before initializing the stack."
        )
    base = status.base
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
        base_pr_url=base_pr_url,
        tool=tool,
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
    """Submit an initialized, clean, linear PTQ job stack through ghstack or sp."""
    _log = log or (lambda _: None)
    job = repo.get(job_id)
    if job.submission_mode != SubmissionMode.GHSTACK:
        raise PtqError(f"Run `ptq stack init {job_id}` before submitting this job.")

    backend = backend_for_job(job)
    worktree = worktree_path(backend.workspace, job.job_id, job.repo)
    _log("Resolving stack configuration...")
    remote = ghstack_remote(backend, worktree)
    _log(f"Fetching {remote}...")
    checked_git(backend, worktree, f"fetch --prune {shlex.quote(remote)}")
    status = inspect_stack(repo, job_id)
    validate_submit(status)
    tool = submission_tool(job, status.commits)
    if tool == StackTool.SP:
        executable = sp_executable(backend)
        if not executable:
            raise PtqError("sp is not installed on the job target.")
        args = [executable, "submit", "-R", status.remote, "-T", status.base]
        if draft:
            args.append("-d")
        if update_metadata:
            args.append("--sync-metadata")
    else:
        args = [ghstack_executable(backend), "submit", "--stack", "-B", status.base]
        if draft:
            args.append("--draft")
        if update_metadata:
            args.append("--update-fields")
        args.append("HEAD")
    _log(
        f"Submitting {len(status.commits)} commit(s) from {status.branch} "
        f"onto {status.base_ref} with {tool.value}..."
    )

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
            f"{tool.value} submit failed: {stderr or stdout or 'see output above'}"
        )

    _log("Refreshing submitted stack metadata...")
    submitted = inspect_stack(repo, job_id)
    # sp hands stacks to ghstack under `git config sp.backend ghstack`; trust the trailers.
    persist_submitted_stack(
        repo,
        job_id,
        submitted.commits,
        update_metadata=update_metadata,
        tool=submission_tool(job, submitted.commits),
    )
    return StackSubmitResult(status=submitted, output=stdout)
