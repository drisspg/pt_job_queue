from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

app = typer.Typer(
    name="ptq", help="PyTorch Job Queue — dispatch Claude agents to fix PyTorch issues."
)
console = Console()


@app.command()
def setup(
    machine: Annotated[
        str | None, typer.Argument(help="Remote machine to set up.")
    ] = None,
    local: Annotated[
        bool, typer.Option("--local", help="Set up local workspace instead.")
    ] = False,
    cuda: Annotated[
        str | None, typer.Option(help="CUDA tag override (cu124, cu126, cu128, cu130).")
    ] = None,
    cpu: Annotated[
        bool, typer.Option("--cpu", help="Use CPU-only PyTorch (for macOS/testing).")
    ] = False,
    workspace: Annotated[
        str | None, typer.Option(help="Custom workspace path.")
    ] = None,
) -> None:
    """One-time workspace setup on a remote machine or locally."""
    if not machine and not local:
        raise typer.BadParameter("Provide a machine name or use --local.")

    from ptq.ssh import LocalBackend, RemoteBackend
    from ptq.workspace import setup_workspace

    if local:
        backend = LocalBackend(workspace=workspace or "~/.ptq_workspace")
    else:
        assert machine is not None
        backend = RemoteBackend(
            machine=machine, workspace=workspace or "~/ptq_workspace"
        )

    setup_workspace(backend, cuda_tag=cuda, cpu=cpu)


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
    model: Annotated[str, typer.Option(help="Claude model to use.")] = "opus",
    max_turns: Annotated[int, typer.Option(help="Max agent turns.")] = 100,
    workspace: Annotated[
        str | None, typer.Option(help="Custom workspace path.")
    ] = None,
    message: Annotated[
        str | None,
        typer.Option("--message", "-m", help="Custom instruction for the agent."),
    ] = None,
) -> None:
    """Launch a Claude agent to investigate a PyTorch issue.

    Re-run an existing job by passing its JOB_ID (or issue number) as a positional arg.
    Machine/local settings are pulled from the previous run automatically.

    Examples:
        ptq run --issue 174923 --machine aws-gpu-dev
        ptq run 20260214-174923 -m "look at flex_attention.py instead"
        ptq run 174923 -m "try a different approach"
    """
    from ptq.agent import launch_agent
    from ptq.issue import fetch_issue
    from ptq.ssh import LocalBackend, RemoteBackend

    if job_id is not None:
        from ptq.job import get_job, resolve_job_id

        resolved = resolve_job_id(job_id)
        job = get_job(resolved)
        issue = issue or job["issue"]
        machine = machine or job.get("machine")
        local = local or job.get("local", False)
        workspace = workspace or job.get("workspace")

    if issue is None:
        raise typer.BadParameter("Provide --issue or a JOB_ID to re-run.")
    if not machine and not local:
        raise typer.BadParameter("Provide --machine or --local.")

    console.print(f"Fetching issue #{issue}...")
    issue_data = fetch_issue(issue)
    console.print(f"[bold]{issue_data['title']}[/bold]")

    if local:
        backend = LocalBackend(workspace=workspace or "~/.ptq_workspace")
    else:
        assert machine is not None
        backend = RemoteBackend(
            machine=machine, workspace=workspace or "~/ptq_workspace"
        )

    launch_agent(
        backend,
        issue_data,
        issue,
        machine=machine,
        local=local,
        follow=follow,
        model=model,
        max_turns=max_turns,
        message=message,
    )


@app.command()
def results(
    job_id: Annotated[str, typer.Argument(help="Job ID or issue number.")],
    output_dir: Annotated[
        Path | None, typer.Option(help="Custom output directory.")
    ] = None,
) -> None:
    """Fetch and display results from a completed job."""
    from ptq.job import resolve_job_id
    from ptq.results import display_results, fetch_results

    job_id = resolve_job_id(job_id)
    console.print(f"Fetching results for {job_id}...")
    results_dir = fetch_results(job_id, output_dir)
    display_results(results_dir)
    console.print(f"\nArtifacts saved to: {results_dir}")


@app.command()
def apply(
    job_id: Annotated[str, typer.Argument(help="Job ID or issue number.")],
    pytorch_path: Annotated[
        Path, typer.Option(help="Path to local pytorch checkout.")
    ] = Path("~/meta/pytorch"),
) -> None:
    """Apply a job's diff to a local PyTorch checkout."""
    from ptq.apply import apply_diff
    from ptq.job import resolve_job_id

    job_id = resolve_job_id(job_id)
    apply_diff(job_id, pytorch_path.expanduser())


def _clean_single_job(job_id: str) -> None:
    """Kill agent, remove remote files, drop from DB."""
    from ptq.job import get_job
    from ptq.ssh import backend_for_job, load_jobs_db, save_jobs_db

    job = get_job(job_id)
    backend = backend_for_job(job_id)
    ws = backend.workspace
    job_dir = f"{ws}/jobs/{job_id}"

    pid = job.get("pid")
    if pid and backend.is_pid_alive(pid):
        backend.kill_pid(pid)
        console.print(f"  killed agent (pid {pid})")

    backend.run(
        f"cd {ws}/pytorch && git worktree remove {job_dir}/pytorch --force",
        check=False,
    )
    backend.run(f"rm -rf {job_dir}", check=False)
    backend.run(f"cd {ws}/pytorch && git worktree prune", check=False)

    db = load_jobs_db()
    db.pop(job_id, None)
    save_jobs_db(db)
    console.print(f"  removed {job_id} (issue #{job.get('issue', '?')})")


def _clean_machine(
    machine: str | None,
    local: bool,
    workspace: str | None,
    keep: int,
    include_running: bool,
) -> None:
    """Bulk clean jobs on a machine."""
    from ptq.ssh import LocalBackend, RemoteBackend, load_jobs_db, save_jobs_db

    if local:
        backend = LocalBackend(workspace=workspace or "~/.ptq_workspace")
    else:
        assert machine is not None
        backend = RemoteBackend(
            machine=machine, workspace=workspace or "~/ptq_workspace"
        )

    db = load_jobs_db()
    matching = [
        (jid, entry)
        for jid, entry in sorted(db.items())
        if (local and entry.get("local"))
        or (machine and entry.get("machine") == machine)
    ]

    if not include_running:
        running = []
        stopped = []
        for jid, entry in matching:
            pid = entry.get("pid")
            if pid and backend.is_pid_alive(pid):
                running.append((jid, entry))
            else:
                stopped.append((jid, entry))
        if running:
            console.print(
                f"Skipping {len(running)} running job(s). Use --all to include them."
            )
        matching = stopped

    to_remove = matching[:-keep] if keep and len(matching) > keep else matching
    if not to_remove:
        console.print("Nothing to clean.")
        return

    ws = backend.workspace
    console.print(f"Removing {len(to_remove)} job(s) (keeping {keep})...")
    backend.run(f"cd {ws}/pytorch && git worktree prune", check=False)

    for jid, entry in to_remove:
        pid = entry.get("pid")
        if pid and backend.is_pid_alive(pid):
            backend.kill_pid(pid)

        job_dir = f"{ws}/jobs/{jid}"
        backend.run(
            f"cd {ws}/pytorch && git worktree remove {job_dir}/pytorch --force",
            check=False,
        )
        backend.run(f"rm -rf {job_dir}")
        db.pop(jid, None)
        console.print(f"  removed {jid}")

    save_jobs_db(db)
    backend.run(f"cd {ws}/pytorch && git worktree prune", check=False)
    console.print("[bold green]Clean complete.[/bold green]")


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
    from ptq.job import resolve_job_id
    from ptq.ssh import load_jobs_db

    if target is not None:
        db = load_jobs_db()
        is_job = target in db or (
            target.isdigit() and any(v.get("issue") == int(target) for v in db.values())
        )
        if is_job:
            resolved = resolve_job_id(target)
            _clean_single_job(resolved)
            return

    if not target and not local:
        raise typer.BadParameter("Provide a job ID, machine name, or --local.")

    _clean_machine(
        machine=target,
        local=local,
        workspace=workspace,
        keep=keep,
        include_running=all_jobs,
    )


@app.command(name="list")
def list_jobs() -> None:
    """List all tracked jobs."""
    from rich.table import Table

    from ptq.ssh import backend_for_job, load_jobs_db

    db = load_jobs_db()
    if not db:
        console.print("No jobs.")
        return

    table = Table(
        show_header=True, header_style="bold", show_lines=False, pad_edge=False
    )
    table.add_column("Status", width=9)
    table.add_column("Job ID")
    table.add_column("Issue", style="cyan")
    table.add_column("Runs", justify="right")
    table.add_column("Target")

    for job_id, entry in sorted(db.items()):
        issue = str(entry.get("issue", "?"))
        runs = str(entry.get("runs", 1))
        target = entry.get("machine") or "local"
        pid = entry.get("pid")
        if pid:
            backend = backend_for_job(job_id)
            alive = backend.is_pid_alive(pid)
        else:
            alive = False
        status = "[bold green]running[/bold green]" if alive else "[dim]stopped[/dim]"
        table.add_row(status, job_id, f"#{issue}", runs, target)

    console.print(table)
    console.print()
    console.print("[dim]Actions:[/dim]")
    console.print("[dim]  ptq run --issue NUM --machine TARGET  # new run[/dim]")
    console.print(
        "[dim]  ptq run JOB_ID                        # re-run existing job[/dim]"
    )
    console.print(
        "[dim]  ptq run JOB_ID -m 'look at X instead' # re-run with steering[/dim]"
    )
    console.print("[dim]  ptq peek JOB_ID                       # check progress[/dim]")
    console.print("[dim]  ptq results JOB_ID                    # fetch results[/dim]")
    console.print("[dim]  ptq kill JOB_ID                       # stop agent[/dim]")
    console.print(
        "[dim]  ptq clean JOB_ID                      # remove job entirely[/dim]"
    )
    console.print(
        "[dim]  ptq clean MACHINE                     # bulk clean stopped jobs[/dim]"
    )


@app.command()
def peek(
    job_id: Annotated[str, typer.Argument(help="Job ID or issue number.")],
    log_lines: Annotated[
        int, typer.Option("--log", help="Number of log lines to show.")
    ] = 0,
) -> None:
    """Peek at an agent's progress (worklog + optional log tail)."""
    import json

    from rich.markdown import Markdown

    from ptq.job import get_job, resolve_job_id
    from ptq.ssh import backend_for_job

    job_id = resolve_job_id(job_id)
    job = get_job(job_id)
    backend = backend_for_job(job_id)
    ws = backend.workspace
    runs = job.get("runs", 1)
    pid = job.get("pid")
    target = job.get("machine") or "local"

    alive = pid is not None and backend.is_pid_alive(pid)
    status_str = "[bold green]running[/bold green]" if alive else "[dim]stopped[/dim]"
    console.print(
        f"{status_str}  {job_id}  issue #{job.get('issue', '?')}  (run {runs}, {target})"
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
        log_file = f"{ws}/jobs/{job_id}/claude-{runs}.log"
        tail_result = backend.run(f"tail -{log_lines} {log_file}", check=False)
        if tail_result.stdout.strip():
            console.print()
            console.print(f"[bold]Last {log_lines} log lines[/bold]")
            for line in tail_result.stdout.strip().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                    match event.get("type"):
                        case "assistant":
                            for block in event.get("message", {}).get("content", []):
                                if block.get("type") == "text" and block.get("text"):
                                    console.print(f"  [dim]{block['text'][:200]}[/dim]")
                                elif block.get("type") == "tool_use":
                                    console.print(
                                        f"  [cyan]{block.get('name', '')}[/cyan]"
                                    )
                        case _:
                            pass
                except (json.JSONDecodeError, ValueError):
                    console.print(f"  [dim]{line[:200]}[/dim]")


@app.command()
def status(
    job_id: Annotated[str, typer.Argument(help="Job ID or issue number.")],
) -> None:
    """Check if an agent is still running for a job."""
    from ptq.job import get_job, resolve_job_id
    from ptq.ssh import backend_for_job

    job_id = resolve_job_id(job_id)
    job = get_job(job_id)
    backend = backend_for_job(job_id)
    ws = backend.workspace

    pid = job.get("pid")
    runs = job.get("runs", 1)
    target = job.get("machine") or "local"

    alive = pid is not None and backend.is_pid_alive(pid)

    if alive:
        console.print(
            f"[bold green]running[/bold green]  {job_id}  (run {runs}, {target}, pid {pid})"
        )
        console.print(
            f"  ptq run --issue {job['issue']} --machine {target}  # to reattach"
        )
    else:
        console.print(f"[bold dim]stopped[/bold dim]  {job_id}  (run {runs}, {target})")
        console.print(f"  ptq results {job_id}")

    log_file = f"{ws}/jobs/{job_id}/claude-{runs}.log"
    tail = backend.run(f"tail -1 {log_file}", check=False)
    if tail.stdout.strip():
        console.print(f"\n  last log: [dim]{tail.stdout.strip()[:120]}[/dim]")


@app.command()
def kill(
    job_id: Annotated[str, typer.Argument(help="Job ID or issue number.")],
) -> None:
    """Kill a running agent for a job."""
    from ptq.job import get_job, resolve_job_id, save_pid
    from ptq.ssh import backend_for_job

    job_id = resolve_job_id(job_id)
    job = get_job(job_id)
    backend = backend_for_job(job_id)

    pid = job.get("pid")
    if not pid:
        console.print(f"[dim]No PID tracked for {job_id}[/dim]")
        return

    if backend.is_pid_alive(pid):
        backend.kill_pid(pid)
        save_pid(job_id, None)
        console.print(f"[bold]Killed agent for {job_id} (pid {pid})[/bold]")
    else:
        save_pid(job_id, None)
        console.print(f"[dim]Agent already stopped for {job_id}[/dim]")


if __name__ == "__main__":
    app()
