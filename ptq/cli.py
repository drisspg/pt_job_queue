from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from ptq.agent import _clean, _indent, _truncate
from ptq.agents import StreamEvent, get_agent
from ptq.domain.models import JobRecord, JobStatus, PtqError, RebaseState, RunRequest

app = typer.Typer(
    name="ptq", help="PyTorch Job Queue — dispatch AI agents to fix issues in PyTorch and add-on repos."
)
console = Console()


def _handle_error(e: PtqError) -> None:
    console.print(f"[red]{e}[/red]")
    raise typer.Exit(1)


def _render_event(ev: StreamEvent) -> None:
    match ev.kind:
        case "text":
            console.print(_clean(ev.text), end="", highlight=False)
        case "tool_use":
            console.print()
            inp = ev.tool_input
            match ev.tool_name:
                case "Bash":
                    console.print(
                        f"  [bold cyan]$[/bold cyan] [dim]{inp.get('command', '')}[/dim]"
                    )
                case "Read":
                    console.print(
                        f"  [cyan]read[/cyan] [dim]{inp.get('file_path', '') or inp.get('path', '')}[/dim]"
                    )
                case "Edit":
                    console.print(
                        f"  [yellow]edit[/yellow] [dim]{inp.get('file_path', '') or inp.get('path', '')}[/dim]"
                    )
                case "Write":
                    console.print(
                        f"  [green]write[/green] [dim]{inp.get('file_path', '') or inp.get('path', '')}[/dim]"
                    )
                case "Grep":
                    console.print(
                        f"  [cyan]grep[/cyan] [dim]{inp.get('pattern', '')}[/dim]"
                    )
                case "Glob":
                    console.print(
                        f"  [cyan]glob[/cyan] [dim]{inp.get('pattern', '')}[/dim]"
                    )
                case _:
                    console.print(f"  [dim]{ev.tool_name}[/dim]")
        case "tool_result":
            if ev.text.strip():
                console.print(f"[dim]{_indent(_truncate(_clean(ev.text)))}[/dim]")
            console.print()
        case "error":
            console.print(f"[red]{_indent(_truncate(_clean(ev.text)))}[/red]")
            console.print()


def _follow_logs(backend, log_file: str, agent, job_id: str) -> None:
    time.sleep(2)
    tail = backend.tail_log(log_file)
    try:
        for line in tail.stdout:
            stripped = _clean(line.strip())
            if stripped.startswith("{"):
                try:
                    for ev in agent.parse_stream_line(stripped):
                        _render_event(ev)
                except (json.JSONDecodeError, ValueError):
                    pass
    except KeyboardInterrupt:
        tail.terminate()
        tail.wait()
        console.print(
            "\n[bold yellow]Detached. Agent still running on remote.[/bold yellow]"
        )
        console.print(f"  ptq results {job_id}")
        return
    tail.wait()
    console.print("\n[bold]Agent finished.[/bold]")
    console.print(f"  ptq results {job_id}")


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
    machine: Annotated[
        str | None, typer.Argument(help="Remote machine to set up.")
    ] = None,
    local: Annotated[
        bool, typer.Option("--local", help="Set up local workspace instead.")
    ] = False,
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
) -> None:
    """One-time workspace setup: clone PyTorch with submodules, create venv, install build deps.

    Use --build to also compile PyTorch from source (needed for C++ edit support).
    """
    if not machine and not local:
        raise typer.BadParameter("Provide a machine name or use --local.")

    from ptq.config import load_config
    from ptq.infrastructure.backends import create_backend
    from ptq.workspace import setup_workspace

    backend = create_backend(machine=machine, local=local, workspace=workspace)
    setup_workspace(
        backend,
        build=build,
        re_cc_jobs=with_re_cc or 0,
        build_env_prefix=load_config().build_env_prefix(),
    )


@app.command()
def run(
    job_id: Annotated[
        str | None, typer.Argument(help="Job ID or issue number to re-run.")
    ] = None,
    issue: Annotated[int | None, typer.Option(help="GitHub issue number.")] = None,
    machine: Annotated[
        str | None, typer.Option(help="Remote machine to run on.")
    ] = None,
    local: Annotated[bool, typer.Option("--local", help="Run locally.")] = False,
    follow: Annotated[
        bool, typer.Option(help="Stream agent output to terminal.")
    ] = True,
    model: Annotated[str | None, typer.Option(help="Model to use.")] = None,
    max_turns: Annotated[int | None, typer.Option(help="Max agent turns.")] = None,
    agent: Annotated[
        str | None, typer.Option(help="Agent type: claude, codex, or cursor.")
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Stream build output and show timings."),
    ] = False,
    workspace: Annotated[
        str | None, typer.Option(help="Custom workspace path.")
    ] = None,
    message: Annotated[
        str | None,
        typer.Option("--message", "-m", help="Custom instruction for the agent."),
    ] = None,
    preset: Annotated[
        str | None,
        typer.Option(
            "--preset",
            "-p",
            help="Prompt preset key/title (combine with -m to append extra instructions).",
        ),
    ] = None,
    input_file: Annotated[
        Path | None,
        typer.Option("--input", "-i", help="Read task description from a file."),
    ] = None,
    name: Annotated[
        str | None,
        typer.Option("--name", "-n", help="Display name for this job."),
    ] = None,
    repo: Annotated[
        str,
        typer.Option("--repo", help="Repo the issue is filed in (default: pytorch)."),
    ] = "pytorch",
) -> None:
    """Launch an AI agent to investigate a GitHub issue or run an adhoc task.

    Provide --issue for GitHub issue investigation, or --message for a freeform task.
    Re-run an existing job by passing its JOB_ID (or issue number) as a positional arg.

    Examples:
        ptq run --issue 174923 --machine aws-gpu-dev
        ptq run --agent codex -m "investigate OOM" --machine gpu-dev
        ptq run -p diagnose_and_plan --issue 174923 --machine gpu-dev
        ptq run -p fix_and_verify -m "focus on stride handling" --issue 174923
        ptq run -i task.md --machine gpu-dev --agent cursor
        ptq run 20260214-174923 -m "look at flex_attention.py instead"
        ptq run 174923 -m "try a different approach"
    """
    if input_file is not None and message is not None:
        raise typer.BadParameter("--input and --message are mutually exclusive.")
    if input_file is not None:
        if not input_file.exists():
            raise typer.BadParameter(f"File not found: {input_file}")
        message = input_file.read_text()

    from ptq.application import run_service
    from ptq.config import load_config
    from ptq.infrastructure.backends import create_backend
    from ptq.issue import fetch_issue

    cfg = load_config()
    if preset:
        selected_preset = cfg.prompt_preset(preset)
        if selected_preset is None:
            choices = ", ".join(cfg.prompt_preset_choices())
            raise typer.BadParameter(
                f"Unknown preset '{preset}'. Available presets: {choices}"
            )
        if message:
            message = f"{selected_preset.body}\n\n{message.strip()}"
        else:
            message = selected_preset.body

    job_repo = _repo()

    resolved_job_id: str | None = None
    if job_id is not None:
        try:
            resolved_job_id = job_repo.resolve_id(job_id)
        except PtqError as e:
            _handle_error(e)
        job = job_repo.get(resolved_job_id)
        issue = issue or job.issue
        machine = machine or job.machine
        local = local or job.local
        workspace = workspace or job.workspace
        repo = job.repo
        if agent is None:
            agent = job.agent

    if issue is None and message is None and job_id is None:
        raise typer.BadParameter(
            "Provide --issue, --preset, --message, or a JOB_ID to re-run."
        )
    if not machine and not local:
        local = True
    agent = agent or cfg.default_agent
    model = cfg.effective_model(agent, model)
    max_turns = max_turns or cfg.default_max_turns

    from ptq.repo_profiles import get_profile

    profile = get_profile(repo)

    issue_data = None
    if issue is not None:
        console.print(f"Fetching {profile.github_repo}#{issue}...")
        issue_data = fetch_issue(issue, repo=profile.github_repo)
        console.print(f"[bold]{issue_data['title']}[/bold]")

    backend = create_backend(machine=machine, local=local, workspace=workspace)
    request = RunRequest(
        issue_data=issue_data,
        issue_number=issue,
        message=message,
        machine=machine,
        local=local,
        follow=follow,
        model=model,
        max_turns=max_turns,
        agent_type=agent,
        existing_job_id=resolved_job_id,
        verbose=verbose,
        name=name,
        repo=repo,
    )

    try:
        launched_id = run_service.launch(
            job_repo, backend, request, on_progress=lambda msg: console.print(msg)
        )
    except PtqError as e:
        _handle_error(e)

    if follow:
        job = job_repo.get(launched_id)
        agent_impl = get_agent(job.agent)
        log_file = f"{backend.workspace}/jobs/{launched_id}/{agent_impl.log_filename(job.runs)}"
        _follow_logs(backend, log_file, agent_impl, launched_id)


@app.command("presets")
def list_presets() -> None:
    """List available prompt presets (built-in + custom from config)."""
    from ptq.config import load_config

    cfg = load_config()
    console.print("[bold]Available presets[/bold]")
    for preset in cfg.prompt_presets:
        console.print(f"- [cyan]{preset.key}[/cyan] — {preset.title}")


@app.command()
def results(
    job_id: Annotated[str, typer.Argument(help="Job ID or issue number.")],
    output_dir: Annotated[
        Path | None, typer.Option(help="Custom output directory.")
    ] = None,
) -> None:
    """Fetch and display results from a completed job."""
    from rich.markdown import Markdown

    from ptq.application.artifact_service import fetch_results

    repo = _repo()
    try:
        job_id = repo.resolve_id(job_id)
    except PtqError as e:
        _handle_error(e)

    console.print(f"Fetching results for {job_id}...")
    results_dir, fetched, missing = fetch_results(repo, job_id, output_dir)
    for name in fetched:
        console.print(f"  fetched {name}")
    for name in missing:
        console.print(f"  {name} not found")

    repro_path = results_dir / "repro.py"
    if repro_path.exists():
        from rich.syntax import Syntax

        console.print()
        console.print("[bold]Repro Script[/bold]")
        console.print(Syntax(repro_path.read_text(), "python", theme="monokai"))

    for name, label in [("worklog.md", "Worklog"), ("report.md", "Report")]:
        path = results_dir / name
        if path.exists():
            console.print()
            console.print(f"[bold]{label}[/bold]")
            console.print(Markdown(path.read_text()))

    diff_path = results_dir / "fix.diff"
    if diff_path.exists():
        diff_text = diff_path.read_text()
        if diff_text.strip():
            console.print()
            console.print("[bold]Diff[/bold]")
            console.print(diff_text)
        else:
            console.print("[yellow]fix.diff is empty — no changes made.[/yellow]")
    else:
        console.print("[yellow]No fix.diff found.[/yellow]")

    console.print(f"\nArtifacts saved to: {results_dir}")


@app.command()
def apply(
    job_id: Annotated[str, typer.Argument(help="Job ID or issue number.")],
    pytorch_path: Annotated[
        Path, typer.Option(help="Path to local pytorch checkout.")
    ] = Path("~/meta/pytorch"),
) -> None:
    """Apply a job's diff to a local PyTorch checkout."""
    from ptq.application.artifact_service import apply_diff

    repo = _repo()
    try:
        job_id = repo.resolve_id(job_id)
        branch = apply_diff(repo, job_id, pytorch_path.expanduser())
    except PtqError as e:
        _handle_error(e)

    console.print(
        f"\n[bold green]Diff applied to {pytorch_path} on branch {branch}[/bold green]"
    )
    console.print(f"\nTo create a PR, run: [bold]ptq pr {job_id}[/bold]")


@app.command()
def clean(
    target: Annotated[
        str | None, typer.Argument(help="Job ID, issue number, or machine name.")
    ] = None,
    local: Annotated[
        bool, typer.Option("--local", help="Clean local workspace.")
    ] = False,
    workspace: Annotated[
        str | None, typer.Option(help="Custom workspace path.")
    ] = None,
    keep: Annotated[int, typer.Option(help="Number of most recent jobs to keep.")] = 0,
    all_jobs: Annotated[
        bool, typer.Option("--all", help="Include running jobs.")
    ] = False,
) -> None:
    """Remove jobs: kill agent, delete remote files, drop from tracking DB.

    Pass a JOB_ID to clean a single job, or a MACHINE name to bulk clean.

    Examples:
        ptq clean 20260214-174923          # clean one job
        ptq clean 174923                   # clean by issue number
        ptq clean aws-gpu-dev              # clean all stopped jobs on machine
        ptq clean aws-gpu-dev --keep 2     # keep 2 most recent
        ptq clean aws-gpu-dev --all        # include running jobs
    """
    from ptq.application.job_service import clean_machine, clean_single_job
    from ptq.infrastructure.backends import create_backend

    repo = _repo()

    if target is not None:
        all_jobs_db = repo.list_all()
        is_job = target in all_jobs_db or (
            target.isdigit()
            and any(j.issue == int(target) for j in all_jobs_db.values())
        )
        if is_job:
            try:
                resolved = repo.resolve_id(target)
                job = clean_single_job(repo, resolved)
            except PtqError as e:
                _handle_error(e)
            label = f"issue #{job.issue}" if job.issue is not None else "adhoc"
            console.print(f"  removed {resolved} ({label})")
            return

    if not target and not local:
        raise typer.BadParameter("Provide a job ID, machine name, or --local.")

    backend = create_backend(machine=target, local=local, workspace=workspace)
    removed, skipped = clean_machine(
        repo,
        backend,
        machine=target,
        local=local,
        keep=keep,
        include_running=all_jobs,
    )
    if skipped:
        console.print(f"Skipping {skipped} running job(s). Use --all to include them.")
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

    from ptq.application.job_service import get_status
    from ptq.infrastructure.backends import backend_for_job

    repo = _repo()
    all_jobs = repo.list_all()
    if not all_jobs:
        console.print("No jobs.")
        return

    table = Table(
        show_header=True, header_style="bold", show_lines=False, pad_edge=False
    )
    table.add_column("Status", width=9)
    table.add_column("Job ID")
    table.add_column("Name")
    table.add_column("Issue", style="cyan")
    table.add_column("Agent", width=7)
    table.add_column("Runs", justify="right")
    table.add_column("PR", width=7)
    table.add_column("Rebase", width=8)
    table.add_column("Target")

    for job_id, job in sorted(all_jobs.items()):
        issue_display = f"#{job.issue}" if job.issue is not None else "[dim]adhoc[/dim]"
        backend = backend_for_job(job)
        status = get_status(job, backend)
        status_str = (
            "[bold green]running[/bold green]"
            if status == JobStatus.RUNNING
            else "[dim]stopped[/dim]"
        )
        pr_display = _pr_list_label(job.pr_url, backend)
        rebase_display = _rebase_list_label(job.rebase_info.state)
        table.add_row(
            status_str,
            job_id,
            job.name or "[dim]-[/dim]",
            issue_display,
            job.agent,
            str(job.runs),
            pr_display,
            rebase_display,
            job.target,
        )

    console.print(table)
    console.print()
    console.print("[dim]Actions:[/dim]")
    console.print(
        "[dim]  ptq run --issue NUM --machine TARGET  # new run from issue[/dim]"
    )
    console.print("[dim]  ptq run -m 'task' --machine TARGET    # new adhoc run[/dim]")
    console.print(
        "[dim]  ptq run JOB_ID                        # re-run existing job[/dim]"
    )
    console.print(
        "[dim]  ptq run JOB_ID -m 'look at X instead' # re-run with steering[/dim]"
    )
    console.print("[dim]  ptq peek JOB_ID                       # check progress[/dim]")
    console.print("[dim]  ptq results JOB_ID                    # fetch results[/dim]")
    console.print(
        "[dim]  ptq pr JOB_ID                         # create GitHub PR[/dim]"
    )
    console.print(
        "[dim]  ptq takeover JOB_ID                   # drop into worktree[/dim]"
    )
    console.print("[dim]  ptq kill JOB_ID                       # stop agent[/dim]")
    console.print(
        "[dim]  ptq clean JOB_ID                      # remove job entirely[/dim]"
    )
    console.print(
        "[dim]  ptq clean MACHINE                     # bulk clean stopped jobs[/dim]"
    )
    console.print(
        "[dim]  ptq web                               # start web dashboard[/dim]"
    )


@app.command()
def peek(
    job_id: Annotated[str, typer.Argument(help="Job ID or issue number.")],
    log_lines: Annotated[
        int, typer.Option("--log", help="Number of log lines to show.")
    ] = 0,
) -> None:
    """Peek at an agent's progress (worklog + optional log tail)."""
    from rich.markdown import Markdown

    from ptq.application.job_service import get_status
    from ptq.infrastructure.backends import backend_for_job

    repo = _repo()
    try:
        job_id = repo.resolve_id(job_id)
    except PtqError as e:
        _handle_error(e)
    job = repo.get(job_id)
    backend = backend_for_job(job)
    ws = backend.workspace
    status = get_status(job, backend)
    status_str = (
        "[bold green]running[/bold green]"
        if status == JobStatus.RUNNING
        else "[dim]stopped[/dim]"
    )
    issue_label = f"issue #{job.issue}" if job.issue is not None else "adhoc"
    console.print(
        f"{status_str}  {job_id}  {issue_label}  (run {job.runs}, {job.target})"
    )
    console.print()

    worklog_path = f"{ws}/jobs/{job_id}/worklog.md"
    result = backend.run(f"cat {worklog_path}", check=False)
    if result.returncode == 0 and result.stdout.strip():
        console.print("[bold]Worklog[/bold]")
        console.print(Markdown(result.stdout))
    else:
        console.print("[yellow]No worklog yet.[/yellow]")

    if log_lines > 0:
        agent_impl = get_agent(job.agent)
        log_file = f"{ws}/jobs/{job_id}/{agent_impl.log_filename(job.runs)}"
        tail_result = backend.run(f"tail -{log_lines} {log_file}", check=False)
        if tail_result.stdout.strip():
            console.print()
            console.print(f"[bold]Last {log_lines} log lines[/bold]")
            for line in tail_result.stdout.strip().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    for ev in agent_impl.parse_stream_line(line):
                        match ev.kind:
                            case "text":
                                console.print(f"  [dim]{ev.text[:200]}[/dim]")
                            case "tool_use":
                                console.print(f"  [cyan]{ev.tool_name}[/cyan]")
                            case _:
                                pass
                except (json.JSONDecodeError, ValueError):
                    console.print(f"  [dim]{line[:200]}[/dim]")


@app.command()
def takeover(
    job_id: Annotated[str, typer.Argument(help="Job ID or issue number.")],
) -> None:
    """Print the shell command to drop into a job's worktree."""
    from ptq.infrastructure.backends import backend_for_job

    repo = _repo()
    try:
        job_id = repo.resolve_id(job_id)
    except PtqError as e:
        _handle_error(e)
    job = repo.get(job_id)
    backend = backend_for_job(job)
    ws = backend.workspace
    job_dir = f"{ws}/jobs/{job_id}"

    if job.local:
        cmd = f"cd {job_dir} && source .venv/bin/activate"
    else:
        cmd = f"ssh -t {job.machine} 'cd {job_dir} && source .venv/bin/activate && exec $SHELL'"

    console.print(cmd)


@app.command()
def status(
    job_id: Annotated[str, typer.Argument(help="Job ID or issue number.")],
) -> None:
    """Check if an agent is still running for a job."""
    from ptq.application.job_service import get_status as _get_status
    from ptq.infrastructure.backends import backend_for_job

    repo = _repo()
    try:
        job_id = repo.resolve_id(job_id)
    except PtqError as e:
        _handle_error(e)
    job = repo.get(job_id)
    backend = backend_for_job(job)
    ws = backend.workspace

    if _get_status(job, backend) == JobStatus.RUNNING:
        console.print(
            f"[bold green]running[/bold green]  {job_id}  (run {job.runs}, {job.target}, pid {job.pid})"
        )
        if job.issue is not None:
            console.print(
                f"  ptq run --issue {job.issue} --machine {job.target}  # to reattach"
            )
        else:
            console.print(f"  ptq run {job_id}  # to reattach")
    else:
        console.print(
            f"[bold dim]stopped[/bold dim]  {job_id}  (run {job.runs}, {job.target})"
        )
        console.print(f"  ptq results {job_id}")

    agent_impl = get_agent(job.agent)
    log_file = f"{ws}/jobs/{job_id}/{agent_impl.log_filename(job.runs)}"
    tail = backend.run(f"tail -1 {log_file}", check=False)
    if tail.stdout.strip():
        console.print(f"\n  last log: [dim]{tail.stdout.strip()[:120]}[/dim]")


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
    from ptq.application.pr_service import create_pr

    repo = _repo()
    try:
        job_id = repo.resolve_id(job_id)
    except PtqError as e:
        _handle_error(e)

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
def web(
    port: Annotated[int, typer.Option(help="Port to listen on.")] = 8000,
    host: Annotated[str, typer.Option(help="Host to bind to.")] = "127.0.0.1",
    debug: Annotated[bool, typer.Option(help="Enable debug logging.")] = False,
) -> None:
    """Start the web dashboard."""
    try:
        import uvicorn

        from ptq.web.app import create_app
    except ModuleNotFoundError:
        console.print(
            "[red]Missing web dependencies.[/red] Install with: [bold]pip install -e .[/bold]"
        )
        raise typer.Exit(1)  # noqa: B904

    console.print(f"Starting ptq web at http://{host}:{port}")
    if debug:
        # Use string import so uvicorn can enable auto-reload
        uvicorn.run(
            "ptq.web.app:create_debug_app",
            factory=True,
            host=host,
            port=port,
            log_level="debug",
            reload=True,
            reload_dirs=[str(Path(__file__).resolve().parent)],
        )
    else:
        uvicorn.run(
            create_app(debug=False),
            host=host,
            port=port,
            log_level="info",
        )


@app.command()
def kill(
    job_id: Annotated[str, typer.Argument(help="Job ID or issue number.")],
) -> None:
    """Kill a running agent for a job."""
    from ptq.application.job_service import kill_job

    repo = _repo()
    try:
        job_id = repo.resolve_id(job_id)
    except PtqError as e:
        _handle_error(e)

    job = repo.get(job_id)
    killed = kill_job(repo, job_id)
    if killed:
        console.print(f"[bold]Killed agent for {job_id} (pid {job.pid})[/bold]")
    else:
        console.print(f"[dim]Agent already stopped for {job_id}[/dim]")


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
def worktree(
    name: Annotated[str, typer.Argument(help="Display name for this worktree.")],
    machine: Annotated[
        str | None, typer.Option(help="Remote machine to create worktree on.")
    ] = None,
    local: Annotated[
        bool, typer.Option("--local", help="Create in local workspace.")
    ] = False,
    workspace: Annotated[
        str | None, typer.Option(help="Custom workspace path.")
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Stream build output and show timings."),
    ] = False,
    repo: Annotated[
        str,
        typer.Option("--repo", help="Repo to create a worktree for (default: pytorch)."),
    ] = "pytorch",
) -> None:
    """Create a named worktree with a ready-to-use venv.

    Sets up a git worktree and per-worktree venv without launching an agent.
    Use `ptq run <name>` later to launch an agent in this worktree.

    Examples:
        ptq worktree flex-attn --machine gpu-dev
        ptq worktree my-fix --local
        ptq run flex-attn -m "optimize the CPU codegen"
    """
    if not machine and not local:
        local = True

    from ptq.application.worktree_service import provision_worktree, validate_workspace
    from ptq.domain.policies import make_job_id
    from ptq.infrastructure.backends import create_backend
    from ptq.repo_profiles import get_profile
    from ptq.workspace import deploy_scripts

    profile = get_profile(repo)
    job_repo = _repo()
    existing = job_repo.find_by_name(name)
    if existing:
        console.print(
            f"[yellow]Worktree '{name}' already exists as {existing}[/yellow]"
        )
        raise typer.Exit(1)

    backend = create_backend(machine=machine, local=local, workspace=workspace)
    try:
        validate_workspace(backend, backend.workspace, repo=repo)
    except PtqError as e:
        _handle_error(e)

    job_id = make_job_id(message=name, repo=repo)
    job_repo.save(
        JobRecord(
            job_id=job_id,
            runs=0,
            agent="claude",
            model="",
            machine=machine,
            local=local,
            workspace=backend.workspace,
            name=name,
            repo=repo,
        )
    )

    deploy_scripts(backend)
    try:
        provision_worktree(
            backend,
            job_id,
            verbose=verbose,
            progress=lambda msg: console.print(msg),
            repo=repo,
        )
    except PtqError as e:
        _handle_error(e)

    ws = backend.workspace
    job_dir = f"{ws}/jobs/{job_id}"
    dir_name = profile.dir_name
    console.print()
    console.print(f"[bold green]Worktree '{name}' ready.[/bold green]")
    console.print(f"  Job ID:   {job_id}")
    console.print(f"  Worktree: {job_dir}/{dir_name}")
    if local:
        console.print(f"\n  cd {job_dir}/{dir_name} && source ../.venv/bin/activate")
    else:
        console.print(
            f"\n  ssh -t {machine} 'cd {job_dir}/{dir_name} && "
            f"source ../.venv/bin/activate && exec $SHELL'"
        )
    console.print(f"\n  To launch an agent: ptq run {name} -m 'your task'")


@app.command()
def rebase(
    job_id: Annotated[str, typer.Argument(help="Job ID or issue number.")],
    onto: Annotated[
        str, typer.Option("--onto", help="Target ref to rebase onto.")
    ] = "origin/main",
    agent: Annotated[str | None, typer.Option(help="Agent type override.")] = None,
    model: Annotated[str | None, typer.Option(help="Model override.")] = None,
    max_attempts: Annotated[
        int, typer.Option(help="Max conflict resolution attempts.")
    ] = 3,
) -> None:
    """Rebase a job's worktree onto a newer commit (default: origin/main).

    If conflicts arise, an agent is launched to resolve them automatically.
    Escalates to human takeover if conflicts remain after max attempts.

    Examples:
        ptq rebase 174923
        ptq rebase 20260214-174923 --onto origin/main
        ptq rebase 174923 --agent codex --model o3 --max-attempts 2
    """
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
            agent_name=agent,
            model=model,
            max_attempts=max_attempts,
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
            job = repo.get(job_id)
            if job.local:
                console.print(
                    f"\n  cd {job.workspace}/jobs/{job_id} && source .venv/bin/activate"
                )
            else:
                console.print(
                    f"\n  ssh -t {job.target} 'cd {job.workspace}/jobs/{job_id} "
                    f"&& source .venv/bin/activate && exec $SHELL'"
                )
        case _:
            console.print(f"\n[red]Rebase failed: {result.error}[/red]")


if __name__ == "__main__":
    app()
