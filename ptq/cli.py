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


if __name__ == "__main__":
    app()
