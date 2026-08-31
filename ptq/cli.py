from __future__ import annotations

import sys
import time
from typing import Annotated

import typer
from rich.console import Console, Group
from rich.markup import escape

from ptq.domain.models import PtqError, RebaseState

app = typer.Typer(
    name="ptq",
    help="Manage isolated PyTorch workspaces, pull requests, and CI follow-up.",
)
stack_app = typer.Typer(help="Inspect and submit ghstack pull request stacks.")
app.add_typer(stack_app, name="stack")
console = Console()


def _can_prompt_for_pr_metadata() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def _handle_error(e: PtqError) -> None:
    console.print(f"[red]{e}[/red]")
    raise typer.Exit(1)


def _repo():
    from ptq.infrastructure.job_repository import JobRepository

    return JobRepository()


def _rebase_list_label(state: RebaseState) -> str:
    match state:
        case RebaseState.IDLE:
            return "[dim]-[/dim]"
        case RebaseState.RUNNING:
            return "[blue]run[/blue]"
        case RebaseState.SUCCEEDED:
            return "[green]ok[/green]"
        case RebaseState.NEEDS_HUMAN:
            return "[yellow]human[/yellow]"
        case RebaseState.FAILED:
            return "[red]fail[/red]"


def _pr_list_label(pr_url: str | None, backend) -> str:
    from ptq.application.pr_service import get_pr_state

    if not pr_url:
        return "[dim]-[/dim]"

    match get_pr_state(backend, pr_url):
        case "open":
            return "[green]open[/green]"
        case "closed":
            return "[yellow]closed[/yellow]"
        case "merged":
            return "[cyan]merged[/cyan]"
        case _:
            return "[dim]saved[/dim]"


@app.command()
def setup(
    build: Annotated[
        bool, typer.Option("--build", help="Also compile PyTorch from source.")
    ] = False,
    with_re_cc: Annotated[
        int | None,
        typer.Option(
            "--with-re-cc", help="Use re-cc distributed compiler with N parallel jobs."
        ),
    ] = None,
    workspace: Annotated[
        str | None, typer.Option(help="Custom workspace path.")
    ] = None,
    onto: Annotated[
        str,
        typer.Option(
            "--onto",
            help="Target ref for resetting the seed PyTorch checkout.",
        ),
    ] = "origin/main",
    extras: Annotated[
        list[str] | None,
        typer.Option(
            "--extras", help="Additional repos to clone (e.g. --extras torchtitan)."
        ),
    ] = None,
) -> None:
    """One-time workspace setup: clone PyTorch with submodules, create venv, install build deps.

    Use --build to also compile PyTorch from source (needed for C++ edit support).
    Use --extras to also clone add-on repos (e.g. --extras torchtitan).
    """
    from ptq.config import load_config
    from ptq.infrastructure.backends import create_backend
    from ptq.workspace import setup_workspace

    backend = create_backend(workspace=workspace)
    setup_workspace(
        backend,
        build=build,
        re_cc_jobs=with_re_cc or 0,
        build_env_prefix=load_config().build_env_prefix(),
        extras=extras or [],
        target_ref=onto,
    )


@app.command()
def clean(
    target: Annotated[
        str, typer.Argument(help="Job ID, issue number, name, or 'local'.")
    ],
    keep: Annotated[int, typer.Option(help="Number of newest jobs to keep.")] = 0,
    force: Annotated[
        bool, typer.Option("--force", help="Remove jobs with uncommitted work.")
    ] = False,
) -> None:
    """Remove one job or clean all local jobs."""
    from ptq.application.job_service import clean_jobs, clean_single_job

    repo = _repo()
    if target != "local":
        try:
            resolved = repo.resolve_id(target)
            job = clean_single_job(repo, resolved, force=force)
        except PtqError as e:
            _handle_error(e)
        label = f"issue #{job.issue}" if job.issue is not None else "adhoc"
        console.print(f"  removed {resolved} ({label})")
        return

    removed = clean_jobs(repo, keep=keep, force=force)
    if not removed:
        console.print("Nothing to clean.")
        return
    console.print(f"Removing {len(removed)} job(s) (keeping {keep})...")
    for jid in removed:
        console.print(f"  removed {jid}")
    console.print("[bold green]Clean complete.[/bold green]")


@app.command(name="list")
def list_jobs() -> None:
    """List all tracked jobs."""
    from rich.table import Table

    from ptq.infrastructure.backends import backend_for_job

    repo = _repo()
    all_jobs = repo.list_all()
    if not all_jobs:
        console.print("No jobs.")
        return

    table = Table(
        show_header=True, header_style="bold", show_lines=False, pad_edge=False
    )
    table.add_column("Job ID")
    table.add_column("Name")
    table.add_column("Issue", style="cyan")
    table.add_column("PR", width=7)
    table.add_column("Rebase", width=8)

    for job_id, job in sorted(all_jobs.items()):
        issue_display = f"#{job.issue}" if job.issue is not None else "[dim]adhoc[/dim]"
        if job.legacy_machine:
            name_display = f"{job.name or '-'} [red](remote unsupported)[/red]"
            pr_display = "[dim]saved[/dim]" if job.pr_url else "[dim]-[/dim]"
        else:
            name_display = job.name or "[dim]-[/dim]"
            pr_display = _pr_list_label(job.pr_url, backend_for_job(job))
        rebase_display = _rebase_list_label(job.rebase_info.state)
        table.add_row(
            job_id,
            name_display,
            issue_display,
            pr_display,
            rebase_display,
        )

    console.print(table)
    console.print()
    console.print("[dim]Actions:[/dim]")
    console.print(
        "[dim]  ptq open --issue NUM                  # create/open local issue workspace[/dim]"
    )
    console.print(
        "[dim]  ptq open --name NAME                  # create/open local named workspace[/dim]"
    )
    console.print(
        "[dim]  ptq open JOB_ID                       # reopen existing Herdr workspace[/dim]"
    )
    console.print(
        "[dim]  ptq peek JOB_ID                       # show worklog/report[/dim]"
    )
    console.print(
        "[dim]  ptq pr JOB_ID                         # create GitHub PR[/dim]"
    )
    console.print(
        "[dim]  ptq stack init JOB_ID                 # configure a ghstack job[/dim]"
    )
    console.print(
        "[dim]  ptq stack show JOB_ID                 # inspect ghstack commits[/dim]"
    )
    console.print(
        "[dim]  ptq takeover JOB_ID                   # enter the worktree[/dim]"
    )
    console.print(
        "[dim]  ptq clean JOB_ID                      # remove job entirely[/dim]"
    )
    console.print(
        "[dim]  ptq clean local                       # remove all local jobs[/dim]"
    )
    console.print(
        "[dim]  ptq monitor                           # watch PR/CI jobs[/dim]"
    )


def _monitor_phase_style(phase: str) -> str:
    match phase:
        case (
            "ready to merge"
            | "ready for PR"
            | "ready for stack"
            | "ready to resubmit stack"
        ):
            return "green"
        case "waiting on CI" | "landing":
            return "cyan"
        case "unrelated CI":
            return "yellow"
        case (
            "needs fix"
            | "needs rebase"
            | "needs stack rebase"
            | "needs human review"
            | "needs CI review"
        ):
            return "orange3"
        case "merged/closed":
            return "cyan"
        case _:
            return "red"


def _monitor_text_attr(row, name: str) -> str:
    """Read monitor row strings without letting MagicMock placeholders become labels."""
    value = getattr(row, name, "")
    return value if isinstance(value, str) else ""


def _monitor_link_markup(label: str, style: str, url: str = "") -> str:
    """Build Rich markup that stays readable when OSC-8 links are unsupported."""
    escaped_label = escape(label)
    if not url:
        return f"[{style}]{escaped_label}[/]" if style else escaped_label
    if not style:
        return f"[link={url}]{escaped_label}[/]"
    return f"[{style} link={url}]{escaped_label}[/]"


def _github_url_number(url: str, kind: str) -> str:
    """Extract the issue or pull number from a GitHub URL for compact labels."""
    marker = f"/{kind}/"
    if marker not in url:
        return ""
    number = url.rsplit(marker, 1)[1].split("/", 1)[0]
    return number if number.isdigit() else ""


def _monitor_job_markup(row) -> str:
    """Show a job's display name while retaining its stable PTQ identifier."""
    job_id = escape(_monitor_text_attr(row, "job_id"))
    job_name = _monitor_text_attr(row, "job_name")
    if not job_name:
        return job_id
    return f"{escape(job_name)}\n[dim]{job_id}[/]"


def _monitor_issue_markup(row) -> str:
    """Render issue labels as terminal hyperlinks when an issue number is known."""
    issue = _monitor_text_attr(row, "issue")
    if issue.startswith("#") and issue[1:].isdigit():
        return _monitor_link_markup(
            issue,
            "cyan",
            f"https://github.com/pytorch/pytorch/issues/{issue[1:]}",
        )
    return escape(issue)


def _monitor_pr_markup(row) -> str:
    """Render PR state with approval-aware color and an optional hyperlink."""
    pr_state = _monitor_text_attr(row, "pr_state")
    pr_url = _monitor_text_attr(row, "pr_url")
    pr_number = _github_url_number(pr_url, "pull")
    label_prefix = f"#{pr_number} " if pr_number else ""

    if getattr(row, "pr_is_draft", False) is True and pr_state == "open":
        return _monitor_link_markup(f"{label_prefix}draft", "blue", pr_url)
    match pr_state:
        case "open":
            style = (
                "green"
                if _monitor_text_attr(row, "review_decision") == "APPROVED"
                else "yellow"
            )
            return _monitor_link_markup(f"{label_prefix}open", style, pr_url)
        case "merged":
            return _monitor_link_markup(f"{label_prefix}merged", "cyan", pr_url)
        case "closed":
            return _monitor_link_markup(f"{label_prefix}closed", "red", pr_url)
        case "-":
            return "-"
        case _:
            return _monitor_link_markup(f"{label_prefix}{pr_state}", "dim", pr_url)


def _render_monitor_table(rows) -> object:
    """Render the mergedog-style PR monitor table for PTQ jobs."""
    from rich.table import Table

    table = Table(
        title="PTQ PR Monitor",
        show_header=True,
        header_style="bold",
        show_lines=False,
        pad_edge=False,
    )
    table.add_column("Phase")
    table.add_column("Job")
    table.add_column("Issue", style="cyan")
    table.add_column("PR")
    table.add_column("CI")
    table.add_column("Next action")

    for row in rows:
        table.add_row(
            f"[{_monitor_phase_style(row.phase)}]{row.phase}[/]",
            _monitor_job_markup(row),
            _monitor_issue_markup(row),
            _monitor_pr_markup(row),
            row.ci.label,
            row.next_action,
        )
    return table


def _monitor_renderable(rows, *, include_all: bool) -> Group:
    """Build the monitor view as one live-updatable renderable region."""
    from datetime import datetime

    from rich.text import Text

    if not rows:
        lines = [
            Text("No PTQ PR jobs to monitor."),
            Text.from_markup(
                "Use [bold]uv run ptq pr JOB_ID[/bold] to create a PR first."
            ),
        ]
        if not include_all:
            lines.append(
                Text.from_markup("Pass [bold]--all[/bold] to include jobs without PRs.")
            )
        return Group(*lines)

    summary = Text.from_markup(
        f"[dim]Updated {datetime.now().strftime('%H:%M:%S')}[/dim]\n"
        "[dim]Use takeover commands as the source of truth for where Herdr job panes should start.[/dim]"
    )

    entry_commands = Text()
    entry_commands.append("Herdr workspace entry commands\n", style="bold")
    for row in rows:
        entry_commands.append("  ")
        entry_commands.append(row.job_id, style="cyan")
        entry_commands.append(f": {row.takeover_command}\n")
    entry_commands.rstrip()

    renderables = [_render_monitor_table(rows), summary, entry_commands]
    failing_rows = [
        row
        for row in rows
        if row.ci.failing and row.phase not in {"landing", "unrelated CI"}
    ]
    if failing_rows:
        triage_commands = Text()
        triage_commands.append("Failing CI review\n", style="bold")
        for row in failing_rows:
            triage_commands.append("  ")
            triage_commands.append(row.job_id, style="cyan")
            triage_commands.append(f": {row.ci_triage_command}\n")
        triage_commands.rstrip()
        renderables.append(triage_commands)

    merge_ignore_rows = [
        row for row in rows if row.phase == "unrelated CI" and row.can_merge_ignore
    ]
    if merge_ignore_rows:
        merge_ignore_commands = Text()
        merge_ignore_commands.append("PyTorchBot merge-ignore commands\n", style="bold")
        for row in merge_ignore_rows:
            merge_ignore_commands.append("  ")
            merge_ignore_commands.append(row.job_id, style="cyan")
            merge_ignore_commands.append(f": {row.merge_ignore_command}\n")
        merge_ignore_commands.rstrip()
        renderables.append(merge_ignore_commands)
    return Group(*renderables)


def _render_monitor_rows(rows, *, include_all: bool) -> None:
    """Render monitor rows plus the commands that drive Herdr and CI triage."""
    console.print(_monitor_renderable(rows, include_all=include_all))


@app.command()
def monitor(
    watch: Annotated[
        bool, typer.Option("--watch", "-w", help="Refresh continuously.")
    ] = False,
    interval: Annotated[
        float, typer.Option(help="Refresh interval in seconds when watching.")
    ] = 30.0,
    include_all: Annotated[
        bool,
        typer.Option("--all", help="Include jobs without a recorded PR URL."),
    ] = False,
    refresh: Annotated[
        bool, typer.Option(help="Bypass cached PR state for this render.")
    ] = False,
) -> None:
    """Show a mergedog-style monitor for PTQ PR jobs."""
    from ptq.application.monitor_service import collect_monitor_rows

    repo = _repo()

    if not watch:
        _render_monitor_rows(
            collect_monitor_rows(
                repo,
                include_without_pr=include_all,
                force_refresh=refresh,
            ),
            include_all=include_all,
        )
        return

    from rich.live import Live

    try:
        with Live(
            _monitor_renderable(
                collect_monitor_rows(
                    repo,
                    include_without_pr=include_all,
                    force_refresh=True,
                ),
                include_all=include_all,
            ),
            console=console,
            refresh_per_second=4,
            screen=True,
            transient=False,
            vertical_overflow="ellipsis",
        ) as live:
            while True:
                time.sleep(interval)
                live.update(
                    _monitor_renderable(
                        collect_monitor_rows(
                            repo,
                            include_without_pr=include_all,
                            force_refresh=True,
                        ),
                        include_all=include_all,
                    ),
                    refresh=True,
                )
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Stopped monitor.[/bold yellow]")


@app.command()
def peek(
    job_id: Annotated[str, typer.Argument(help="Job ID, name, or issue number.")],
) -> None:
    """Display a job's worklog and latest report."""
    from rich.markdown import Markdown

    from ptq.infrastructure.backends import backend_for_job

    repo = _repo()
    try:
        job_id = repo.resolve_id(job_id)
    except PtqError as e:
        _handle_error(e)
    job = repo.get(job_id)
    backend = backend_for_job(job)
    issue_label = f"issue #{job.issue}" if job.issue is not None else "adhoc"
    console.print(f"[bold]{job_id}[/bold]  {issue_label}")

    job_dir = f"{backend.workspace}/jobs/{job_id}"
    found = False
    for filename, label in (("worklog.md", "Worklog"), ("report.md", "Report")):
        result = backend.run(f"cat {job_dir}/{filename}", check=False)
        if result.returncode == 0 and result.stdout.strip():
            console.print(f"\n[bold]{label}[/bold]")
            console.print(Markdown(result.stdout))
            found = True
    if not found:
        console.print("[yellow]No worklog or report yet.[/yellow]")


@app.command()
def takeover(
    job_id: Annotated[str, typer.Argument(help="Job ID or issue number.")],
) -> None:
    """Print the shell command to drop into a job's worktree."""
    from ptq.takeover import for_job as takeover_for_job

    repo = _repo()
    try:
        job_id = repo.resolve_id(job_id)
    except PtqError as e:
        _handle_error(e)
    job = repo.get(job_id)
    console.print(takeover_for_job(job_id, job))


@app.command("open")
def open_job(
    job_id: Annotated[
        str | None, typer.Argument(help="Existing job ID, name, or issue number.")
    ] = None,
    issue: Annotated[
        int | None,
        typer.Option("--issue", help="Create or reuse an issue worktree."),
    ] = None,
    name: Annotated[
        str | None,
        typer.Option("--name", help="Create or reuse a named worktree."),
    ] = None,
    workspace: Annotated[
        str | None,
        typer.Option(help="Custom seed workspace path."),
    ] = None,
    repo_name: Annotated[
        str,
        typer.Option("--repo", help="Repository profile for a new worktree."),
    ] = "pytorch",
    no_focus: Annotated[
        bool,
        typer.Option("--no-focus", help="Create the workspace without focusing it."),
    ] = False,
) -> None:
    """Create or reuse a job and open its Herdr workspace."""
    from ptq.application.herdr_service import open_job_workspace
    from ptq.application.worktree_service import (
        ensure_job_worktree,
        prepare_job_worktree,
    )
    from ptq.infrastructure.backends import backend_for_job, create_backend
    from ptq.takeover import for_job as takeover_for_job

    selectors = sum(value is not None for value in (job_id, issue, name))
    if selectors != 1:
        console.print("[red]Provide exactly one JOB_ID, --issue, or --name.[/red]")
        raise typer.Exit(1)

    repo = _repo()
    try:
        if job_id is not None:
            job_id = repo.resolve_id(job_id)
            existing_job = repo.get(job_id)
            prepare_job_worktree(
                backend_for_job(existing_job),
                job_id,
                name=existing_job.name,
                progress=lambda message: console.print(f"  {message}"),
                repo=existing_job.repo,
            )
        else:
            backend = create_backend(workspace=workspace)
            job_id, created = ensure_job_worktree(
                repo,
                backend,
                issue_number=issue,
                name=name,
                progress=lambda message: console.print(f"  {message}"),
                repo=repo_name,
                workspace_explicit=workspace is not None,
            )
            if created:
                console.print(f"[bold green]Created PTQ job {job_id}.[/bold green]")
            else:
                console.print(f"[yellow]Reusing PTQ job {job_id}.[/yellow]")
    except PtqError as e:
        _handle_error(e)
    job = repo.get(job_id)
    label = (
        f"ptq #{job.issue}" if job.issue is not None else f"ptq {job.name or job_id}"
    )
    takeover_command = takeover_for_job(job_id, job)
    try:
        workspace = open_job_workspace(
            job_id,
            takeover_command,
            label=label,
            focus=not no_focus,
        )
    except RuntimeError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1) from e
    console.print("[bold green]Opened PTQ job Herdr workspace.[/bold green]")
    console.print(f"  job: {job_id}")
    console.print(f"  workspace: {workspace.workspace_id}")
    console.print(f"  pane: {workspace.pane_id}")
    console.print(f"  entry: {takeover_command}")
    console.print("  prime: @prime.md")


@stack_app.command("init")
def stack_init(
    job_id: Annotated[str, typer.Argument(help="Job ID, name, or issue number.")],
    base: Annotated[
        str | None,
        typer.Option("--base", "-B", help="Set the stack base branch."),
    ] = None,
) -> None:
    """Configure a PTQ job and its agent context for ghstack submissions."""
    from ptq.application.stack_service import initialize_stack

    repo = _repo()
    try:
        job_id = repo.resolve_id(job_id)
        status = initialize_stack(repo, job_id, base=base)
    except PtqError as e:
        _handle_error(e)

    console.print(f"[bold green]Configured {escape(job_id)} for ghstack.[/bold green]")
    console.print(f"  Branch: {escape(status.branch)}")
    console.print(f"  Base: {escape(status.base_ref)}")
    console.print("  Agent context: STACK_CONTEXT.md")
    console.print(f"  Next: uv run ptq stack show {escape(job_id)}")


@stack_app.command("show")
def stack_show(
    job_id: Annotated[str, typer.Argument(help="Job ID, name, or issue number.")],
) -> None:
    """Show the commits and repository state that ghstack would submit."""
    from ptq.application.stack_service import inspect_stack

    repo = _repo()
    try:
        job_id = repo.resolve_id(job_id)
        status = inspect_stack(repo, job_id)
    except PtqError as e:
        _handle_error(e)

    branch = status.branch or "detached HEAD"
    cleanliness = "dirty" if status.dirty else "clean"
    console.print(f"[bold]ghstack for {escape(job_id)}[/bold]")
    console.print(f"  Branch: {escape(branch)}")
    console.print(f"  Remote: {escape(status.remote)}")
    console.print(f"  Base: {escape(status.base_ref)}")
    console.print(f"  Worktree: {cleanliness}")
    if status.has_merges:
        console.print("  [yellow]History contains merge commits.[/yellow]")
    if not status.commits:
        console.print("  [dim]No commits to submit.[/dim]")
        return

    console.print("\n  Commits (base to top):")
    for position, commit in enumerate(status.commits, start=1):
        link = f" -> {commit.pr_url}" if commit.pr_url else ""
        console.print(
            f"  {position}. {commit.sha[:10]} {escape(commit.subject)}{escape(link)}"
        )


@stack_app.command("submit")
def stack_submit(
    job_id: Annotated[str, typer.Argument(help="Job ID, name, or issue number.")],
    draft: Annotated[
        bool, typer.Option(help="Create new pull requests as drafts.")
    ] = False,
    update_metadata: Annotated[
        bool,
        typer.Option(
            "--update-metadata",
            help="Replace existing PR titles and bodies from local commit messages.",
        ),
    ] = False,
) -> None:
    """Submit or update a clean linear stack through ghstack."""
    from ptq.application.stack_service import submit_stack

    repo = _repo()
    try:
        job_id = repo.resolve_id(job_id)
        console.print(f"[bold]Submitting ghstack for {escape(job_id)}[/bold]")
        result = submit_stack(
            repo,
            job_id,
            draft=draft,
            update_metadata=update_metadata,
            stream_output=True,
            log=lambda message: console.print(f"  [dim]{message}[/dim]"),
        )
    except PtqError as e:
        _handle_error(e)

    console.print(
        f"[bold green]Submitted {len(result.status.commits)} PR(s).[/bold green]"
    )
    for position, commit in enumerate(result.status.commits, start=1):
        console.print(
            f"  {position}. {escape(commit.subject)} -> {escape(commit.pr_url)}"
        )
    if result.output:
        console.print(f"\n[dim]{escape(result.output)}[/dim]")


@app.command()
def pr(
    job_id: Annotated[str, typer.Argument(help="Job ID or issue number.")],
    note: Annotated[
        str | None,
        typer.Option(
            "--note",
            "-n",
            help="Your description of the PR: what it does, why it's correct, "
            "and how the reviewer should approach it. Opens $EDITOR if omitted.",
        ),
    ] = None,
    title: Annotated[str | None, typer.Option(help="PR title override.")] = None,
    draft: Annotated[bool, typer.Option(help="Create as draft PR.")] = False,
) -> None:
    """Create a GitHub PR from a job's worktree changes.

    Requires a human note describing the change. This is embedded at the top
    of the PR body so reviewers see the author's own assessment first.
    """
    from ptq.application.pr_service import (
        create_pr,
        ensure_conventional_pr,
        pr_defaults,
    )

    repo = _repo()
    try:
        job_id = repo.resolve_id(job_id)
        ensure_conventional_pr(repo.get(job_id))
        defaults = pr_defaults(repo, job_id)
    except PtqError as e:
        _handle_error(e)

    title = title.strip() if title else None
    if title is None and _can_prompt_for_pr_metadata():
        prompted_title = typer.prompt("PR title", default=defaults.title)
        title = prompted_title.strip() or None

    if not note and defaults.human_note:
        note = defaults.human_note
        source = "GitHub" if defaults.human_note_synced_from_github else "saved"
        console.print(f"[dim]Reusing {source} human note.[/dim]")

    if not note:
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", prefix="ptq-pr-note-", delete=False
        ) as f:
            f.write(
                "# Describe this PR for the reviewer\n"
                "#\n"
                "# What does this change do?\n"
                "# Why do you believe it's correct?\n"
                "# How should the reviewer approach it? (e.g. trivial fix, RFC, etc.)\n"
                "#\n"
                "# Lines starting with # will be stripped.\n"
            )
            note_path = f.name
        editor = os.environ.get("EDITOR", "vim")
        os.system(f"{editor} {note_path}")
        with open(note_path) as f:
            raw = f.read()
        os.unlink(note_path)
        note = "\n".join(
            line for line in raw.splitlines() if not line.startswith("#")
        ).strip()

    if not note:
        console.print("[red]No note provided — PR creation aborted.[/red]")
        raise typer.Exit(1)

    console.print(f"[bold]Creating PR for {job_id}[/bold]")
    try:
        result = create_pr(
            repo,
            job_id,
            human_note=note,
            title=title,
            draft=draft,
            log=lambda msg: console.print(f"  [dim]{msg}[/dim]"),
        )
    except PtqError as e:
        _handle_error(e)
    console.print(f"\n[bold green]PR created:[/bold green] {result.url}")


@app.command()
def rename(
    job_id: Annotated[str, typer.Argument(help="Job ID or issue number.")],
    name: Annotated[str, typer.Argument(help="New display name for the job.")],
) -> None:
    """Set or change the display name of a job."""
    repo = _repo()
    try:
        job_id = repo.resolve_id(job_id)
    except PtqError as e:
        _handle_error(e)

    repo.save_name(job_id, name)
    console.print(f"[bold]{job_id}[/bold] → {name}")


@app.command()
def rebase(
    job_id: Annotated[str, typer.Argument(help="Job ID or issue number.")],
    onto: Annotated[
        str, typer.Option("--onto", help="Target ref to rebase onto.")
    ] = "origin/main",
) -> None:
    """Rebase a job, leaving conflicts for interactive resolution in Herdr."""
    from ptq.application.rebase_service import rebase as do_rebase

    repo = _repo()
    try:
        job_id = repo.resolve_id(job_id)
    except PtqError as e:
        _handle_error(e)

    console.print(f"[bold]Rebasing {job_id} onto {onto}[/bold]")
    try:
        result = do_rebase(
            repo,
            job_id,
            target_ref=onto,
            on_progress=lambda msg: console.print(f"  {msg}"),
        )
    except PtqError as e:
        _handle_error(e)

    match result.state:
        case RebaseState.SUCCEEDED:
            console.print(
                f"\n[bold green]Rebase complete.[/bold green] "
                f"{result.before_sha[:10]} → {result.after_sha[:10]}"
            )
        case RebaseState.NEEDS_HUMAN:
            console.print("\n[bold yellow]Needs human intervention.[/bold yellow]")
            console.print(f"  {result.error}")
            from ptq.takeover import for_job as takeover_for_job

            job = repo.get(job_id)
            console.print(f"\n  {takeover_for_job(job_id, job)}")
        case _:
            console.print(f"\n[red]Rebase failed: {result.error}[/red]")


if __name__ == "__main__":
    app()
