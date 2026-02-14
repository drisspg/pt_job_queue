from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown

from ptq.ssh import backend_for_job

console = Console()

ARTIFACTS = ["report.md", "fix.diff", "claude.log"]


def fetch_results(job_id: str, output_dir: Path | None = None) -> Path:
    backend = backend_for_job(job_id)
    workspace = backend.workspace
    dest = output_dir or (Path.home() / ".ptq" / "results" / job_id)
    dest.mkdir(parents=True, exist_ok=True)

    for artifact in ARTIFACTS:
        remote_path = f"{workspace}/jobs/{job_id}/{artifact}"
        local_path = dest / artifact
        backend.copy_from(remote_path, local_path)
        if local_path.exists():
            console.print(f"  fetched {artifact}")
        else:
            console.print(f"  [yellow]{artifact} not found[/yellow]")

    return dest


def display_results(results_dir: Path) -> None:
    report_path = results_dir / "report.md"
    diff_path = results_dir / "fix.diff"

    if report_path.exists():
        console.print()
        console.print("[bold]Report[/bold]")
        console.print(Markdown(report_path.read_text()))
    else:
        console.print("[yellow]No report.md found.[/yellow]")

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
