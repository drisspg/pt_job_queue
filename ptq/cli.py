from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

app = typer.Typer(name="ptq", help="PyTorch Job Queue — dispatch Claude agents to fix PyTorch issues.")
console = Console()


@app.command()
def setup(
    machine: Annotated[str | None, typer.Argument(help="Remote machine to set up.")] = None,
    local: Annotated[bool, typer.Option("--local", help="Set up local workspace instead.")] = False,
    cuda: Annotated[str | None, typer.Option(help="CUDA tag override (cu124, cu126, cu128, cu130).")] = None,
    cpu: Annotated[bool, typer.Option("--cpu", help="Use CPU-only PyTorch (for macOS/testing).")] = False,
    workspace: Annotated[str | None, typer.Option(help="Custom workspace path.")] = None,
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
        backend = RemoteBackend(machine=machine, workspace=workspace or "~/ptq_workspace")

    setup_workspace(backend, cuda_tag=cuda, cpu=cpu)


@app.command()
def run(
    issue: Annotated[int, typer.Option(help="GitHub issue number.")],
    machine: Annotated[str | None, typer.Option(help="Remote machine to run on.")] = None,
    local: Annotated[bool, typer.Option("--local", help="Run locally.")] = False,
    follow: Annotated[bool, typer.Option(help="Stream agent output to terminal.")] = True,
    model: Annotated[str, typer.Option(help="Claude model to use.")] = "opus",
    max_turns: Annotated[int, typer.Option(help="Max agent turns.")] = 100,
    workspace: Annotated[str | None, typer.Option(help="Custom workspace path.")] = None,
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
        backend = RemoteBackend(machine=machine, workspace=workspace or "~/ptq_workspace")

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
    job_id: Annotated[str, typer.Argument(help="Job ID to fetch results for.")],
    output_dir: Annotated[Path | None, typer.Option(help="Custom output directory.")] = None,
) -> None:
    """Fetch and display results from a completed job."""
    from ptq.results import display_results, fetch_results

    console.print(f"Fetching results for {job_id}...")
    results_dir = fetch_results(job_id, output_dir)
    display_results(results_dir)
    console.print(f"\nArtifacts saved to: {results_dir}")


@app.command()
def apply(
    job_id: Annotated[str, typer.Argument(help="Job ID whose diff to apply.")],
    pytorch_path: Annotated[Path, typer.Option(help="Path to local pytorch checkout.")] = Path("~/meta/pytorch"),
) -> None:
    """Apply a job's diff to a local PyTorch checkout."""
    from ptq.apply import apply_diff

    apply_diff(job_id, pytorch_path.expanduser())


if __name__ == "__main__":
    app()
