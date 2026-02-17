from __future__ import annotations

import subprocess
from pathlib import Path

from rich.console import Console

from ptq.job import get_job
from ptq.ssh import backend_for_job

console = Console()


def _branch_exists(pytorch_path: Path, branch: str) -> bool:
    result = subprocess.run(
        ["git", "rev-parse", "--verify", branch],
        cwd=pytorch_path,
        capture_output=True,
    )
    return result.returncode == 0


def _current_branch(pytorch_path: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=pytorch_path,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def apply_diff(job_id: str, pytorch_path: Path) -> None:
    if not pytorch_path.exists():
        raise SystemExit(f"PyTorch path does not exist: {pytorch_path}")
    if not (pytorch_path / ".git").exists():
        raise SystemExit(f"Not a git repo: {pytorch_path}")

    job_entry = get_job(job_id)
    issue_number = job_entry.get("issue")

    diff_local = Path.home() / ".ptq" / "results" / job_id / "fix.diff"
    if not diff_local.exists():
        console.print("Fetching diff from remote...")
        backend = backend_for_job(job_id)
        workspace = backend.workspace
        backend.copy_from(f"{workspace}/jobs/{job_id}/fix.diff", diff_local)

    if not diff_local.exists() or not diff_local.read_text().strip():
        raise SystemExit("No diff available to apply.")

    branch_name = f"ptq/{issue_number}" if issue_number is not None else f"ptq/{job_id}"

    if _branch_exists(pytorch_path, branch_name):
        current = _current_branch(pytorch_path)
        if current == branch_name:
            console.print(f"Already on branch {branch_name}, resetting changes...")
            subprocess.run(["git", "checkout", "."], cwd=pytorch_path, check=True)
        else:
            console.print(f"Branch {branch_name} exists, switching and resetting...")
            subprocess.run(
                ["git", "checkout", branch_name], cwd=pytorch_path, check=True
            )
            subprocess.run(["git", "checkout", "."], cwd=pytorch_path, check=True)
    else:
        console.print(f"Creating branch: {branch_name}")
        subprocess.run(
            ["git", "checkout", "-b", branch_name], cwd=pytorch_path, check=True
        )

    console.print("Running dry-run check...")
    check = subprocess.run(
        ["git", "apply", "--check", str(diff_local)],
        cwd=pytorch_path,
        capture_output=True,
        text=True,
    )
    if check.returncode != 0:
        console.print(f"[red]Diff does not apply cleanly:[/red]\n{check.stderr}")
        raise SystemExit("Diff check failed.")

    console.print("Applying diff...")
    subprocess.run(["git", "apply", str(diff_local)], cwd=pytorch_path, check=True)

    console.print(
        f"\n[bold green]Diff applied to {pytorch_path} on branch {branch_name}[/bold green]"
    )
    console.print("\n[bold]Copy & paste:[/bold]\n")
    if issue_number is not None:
        console.print(
            f"cd {pytorch_path} && git add -p && "
            f"git commit -m 'Fix #{issue_number}' && "
            f"gh pr create --title 'Fix #{issue_number}' "
            f"--body 'Fixes #{issue_number}'"
        )
    else:
        console.print(
            f"cd {pytorch_path} && git add -p && "
            f"git commit -m 'Fix from {job_id}' && "
            f"gh pr create --title 'Fix from {job_id}'"
        )
