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
    issue: Annotated[int, typer.Option(help="GitHub issue number.")],
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
) -> None:
    """Launch a Claude agent to investigate a PyTorch issue."""
    if not machine and not local:
        raise typer.BadParameter("Provide --machine or --local.")

    from ptq.agent import launch_agent
    from ptq.issue import fetch_issue
    from ptq.ssh import LocalBackend, RemoteBackend

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


@app.command()
def clean(
    machine: Annotated[
        str | None, typer.Argument(help="Remote machine to clean.")
    ] = None,
    local: Annotated[
        bool, typer.Option("--local", help="Clean local workspace.")
    ] = False,
    workspace: Annotated[
        str | None, typer.Option(help="Custom workspace path.")
    ] = None,
    keep: Annotated[int, typer.Option(help="Number of most recent jobs to keep.")] = 0,
) -> None:
    """Remove old job worktrees and artifacts."""
    if not machine and not local:
        raise typer.BadParameter("Provide a machine name or use --local.")

    from ptq.ssh import LocalBackend, RemoteBackend

    if local:
        backend = LocalBackend(workspace=workspace or "~/.ptq_workspace")
    else:
        assert machine is not None
        backend = RemoteBackend(
            machine=machine, workspace=workspace or "~/ptq_workspace"
        )

    ws = backend.workspace
    result = backend.run(f"ls -1dt {ws}/jobs/*/", check=False)
    job_dirs = [d.strip().rstrip("/") for d in result.stdout.splitlines() if d.strip()]

    to_remove = job_dirs[keep:] if keep else job_dirs
    if not to_remove:
        console.print("Nothing to clean.")
        return

    console.print(f"Removing {len(to_remove)} job(s) (keeping {keep})...")
    backend.run(f"cd {ws}/pytorch && git worktree prune", check=False)
    for job_dir in to_remove:
        name = job_dir.split("/")[-1]
        backend.run(
            f"cd {ws}/pytorch && git worktree remove {job_dir}/pytorch --force",
            check=False,
        )
        backend.run(f"rm -rf {job_dir}")
        console.print(f"  removed {name}")
    backend.run(f"cd {ws}/pytorch && git worktree prune", check=False)
    console.print("[bold green]Clean complete.[/bold green]")


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


@app.command()
def prune(
    machine: Annotated[
        str | None, typer.Argument(help="Remote machine to prune.")
    ] = None,
    local: Annotated[bool, typer.Option("--local", help="Prune local agents.")] = False,
    workspace: Annotated[
        str | None, typer.Option(help="Custom workspace path.")
    ] = None,
) -> None:
    """Kill all running agents (tracked and zombie) on a machine."""
    if not machine and not local:
        raise typer.BadParameter("Provide a machine name or use --local.")

    from ptq.job import save_pid
    from ptq.ssh import LocalBackend, RemoteBackend, load_jobs_db

    if local:
        backend = LocalBackend(workspace=workspace or "~/.ptq_workspace")
    else:
        assert machine is not None
        backend = RemoteBackend(
            machine=machine, workspace=workspace or "~/ptq_workspace"
        )

    db = load_jobs_db()
    tracked_pids: set[int] = set()
    killed = 0
    cleared = 0

    for job_id, entry in sorted(db.items()):
        if local and not entry.get("local"):
            continue
        if machine and entry.get("machine") != machine:
            continue

        pid = entry.get("pid")
        if not pid:
            continue
        tracked_pids.add(pid)

        if backend.is_pid_alive(pid):
            backend.kill_pid(pid)
            save_pid(job_id, None)
            console.print(f"  [bold]killed[/bold]  {job_id}  (pid {pid})")
            killed += 1
        else:
            save_pid(job_id, None)
            cleared += 1

    ws = backend.workspace
    result = backend.run(f"ps aux | grep '[c]laude.*{ws}' || true", check=False)
    zombie_killed = 0
    for line in result.stdout.strip().splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 2 or not parts[1].isdigit():
            continue
        zpid = int(parts[1])
        if zpid in tracked_pids:
            continue
        cmd_preview = " ".join(parts[10:])[:100]
        backend.kill_pid(zpid)
        console.print(f"  [bold red]zombie[/bold red]  pid {zpid}  {cmd_preview}")
        zombie_killed += 1

    target = "local" if local else machine
    if killed or zombie_killed:
        console.print(
            f"\n{target}: killed {killed} tracked + {zombie_killed} zombie agent(s)."
        )
    if cleared:
        console.print(f"{target}: cleared {cleared} stale PID(s).")
    if not killed and not zombie_killed and not cleared:
        console.print(f"{target}: no running agents found.")


if __name__ == "__main__":
    app()
