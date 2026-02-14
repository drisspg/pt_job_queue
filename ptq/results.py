from __future__ import annotations

import subprocess
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown

from ptq.job import get_job
from ptq.ssh import backend_for_job

console = Console()

ARTIFACTS = ["report.md", "fix.diff", "worklog.md"]


def fetch_results(job_id: str, output_dir: Path | None = None) -> Path:
    backend = backend_for_job(job_id)
    workspace = backend.workspace
    job = get_job(job_id)
    runs = job.get("runs", 1)
    dest = output_dir or (Path.home() / ".ptq" / "results" / job_id)
    dest.mkdir(parents=True, exist_ok=True)

    artifacts = [*ARTIFACTS, f"claude-{runs}.log"]
    for artifact in artifacts:
        remote_path = f"{workspace}/jobs/{job_id}/{artifact}"
        local_path = dest / artifact
        try:
            backend.copy_from(remote_path, local_path)
            console.print(f"  fetched {artifact}")
        except subprocess.CalledProcessError:
            console.print(f"  [yellow]{artifact} not found[/yellow]")

    return dest


def display_results(results_dir: Path) -> None:
    worklog_path = results_dir / "worklog.md"
    report_path = results_dir / "report.md"
    diff_path = results_dir / "fix.diff"

    if worklog_path.exists():
        console.print()
        console.print("[bold]Worklog[/bold]")
        console.print(Markdown(worklog_path.read_text()))

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
