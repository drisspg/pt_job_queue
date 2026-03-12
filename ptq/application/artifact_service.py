from __future__ import annotations

import contextlib
import subprocess
from pathlib import Path

from ptq.domain.models import PtqError
from ptq.infrastructure.backends import backend_for_job
from ptq.infrastructure.job_repository import JobRepository

ARTIFACTS = ["report.md", "fix.diff", "worklog.md", "repro.py"]


def fetch_results(
    repo: JobRepository,
    job_id: str,
    output_dir: Path | None = None,
) -> tuple[Path, list[str], list[str]]:
    """Fetch job artifacts. Returns (dest_dir, fetched_names, missing_names)."""
    job = repo.get(job_id)
    backend = backend_for_job(job)
    dest = output_dir or (Path.home() / ".ptq" / "results" / job_id)
    dest.mkdir(parents=True, exist_ok=True)

    artifacts = [*ARTIFACTS, f"agent_logs/{job.agent}-{job.runs}.log"]
    fetched: list[str] = []
    missing: list[str] = []
    for artifact in artifacts:
        remote_path = f"{backend.workspace}/jobs/{job_id}/{artifact}"
        local_path = dest / artifact
        try:
            backend.copy_from(remote_path, local_path)
            fetched.append(artifact)
        except (subprocess.CalledProcessError, FileNotFoundError):
            missing.append(artifact)

    return dest, fetched, missing


def read_artifact(backend, path: str) -> str | None:
    result = backend.run(f"cat {path}", check=False)
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout
    return None


def apply_diff(
    repo: JobRepository,
    job_id: str,
    pytorch_path: Path,
) -> str:
    """Apply a job's diff to a local pytorch checkout. Returns branch name."""
    if not pytorch_path.exists():
        raise PtqError(f"PyTorch path does not exist: {pytorch_path}")
    if not (pytorch_path / ".git").exists():
        raise PtqError(f"Not a git repo: {pytorch_path}")

    job = repo.get(job_id)

    diff_local = Path.home() / ".ptq" / "results" / job_id / "fix.diff"
    if not diff_local.exists():
        backend = backend_for_job(job)
        workspace = backend.workspace
        with contextlib.suppress(subprocess.CalledProcessError, FileNotFoundError):
            backend.copy_from(f"{workspace}/jobs/{job_id}/fix.diff", diff_local)

    if not diff_local.exists() or not diff_local.read_text().strip():
        raise PtqError("No diff available to apply.")

    branch_name = f"ptq/{job.issue}" if job.issue is not None else f"ptq/{job_id}"

    _setup_branch(pytorch_path, branch_name)

    check = subprocess.run(
        ["git", "apply", "--check", str(diff_local)],
        cwd=pytorch_path,
        capture_output=True,
        text=True,
    )
    if check.returncode != 0:
        raise PtqError(f"Diff does not apply cleanly:\n{check.stderr}")

    subprocess.run(["git", "apply", str(diff_local)], cwd=pytorch_path, check=True)
    return branch_name


def _setup_branch(pytorch_path: Path, branch_name: str) -> None:
    def branch_exists() -> bool:
        return (
            subprocess.run(
                ["git", "rev-parse", "--verify", branch_name],
                cwd=pytorch_path,
                capture_output=True,
            ).returncode
            == 0
        )

    def current_branch() -> str:
        return subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=pytorch_path,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    if branch_exists():
        if current_branch() != branch_name:
            subprocess.run(
                ["git", "checkout", branch_name], cwd=pytorch_path, check=True
            )
        subprocess.run(["git", "checkout", "."], cwd=pytorch_path, check=True)
    else:
        subprocess.run(
            ["git", "checkout", "-b", branch_name], cwd=pytorch_path, check=True
        )
