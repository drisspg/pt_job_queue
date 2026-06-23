from __future__ import annotations

import json
import shlex
from dataclasses import dataclass

from ptq.application.job_service import get_status
from ptq.application.pr_service import get_pr_state
from ptq.domain.models import JobRecord, JobStatus, RebaseState
from ptq.infrastructure.backends import backend_for_job
from ptq.infrastructure.job_repository import JobRepository
from ptq.takeover import for_job as takeover_for_job


@dataclass
class CheckSummary:
    label: str = "unknown"
    failing: int = 0
    pending: int = 0
    passing: int = 0
    total: int = 0


@dataclass
class MonitorRow:
    job_id: str
    issue: str
    title: str
    agent: str
    runs: int
    target: str
    job_status: JobStatus
    pr_state: str
    ci: CheckSummary
    phase: str
    next_action: str
    takeover_command: str
    ci_triage_command: str
    pr_url: str


def summarize_pr_checks(job: JobRecord) -> CheckSummary:
    """Summarize GitHub check buckets without treating CI logs as instructions."""
    if not job.pr_url:
        return CheckSummary(label="-")

    result = backend_for_job(job).run(
        "gh pr checks "
        f"{shlex.quote(job.pr_url)} "
        "--json bucket,state,name",
        check=False,
    )
    if not result.stdout.strip():
        return CheckSummary()

    try:
        checks = json.loads(result.stdout)
    except json.JSONDecodeError:
        return CheckSummary()

    failing = 0
    pending = 0
    passing = 0
    for check in checks:
        bucket = str(check.get("bucket") or "").lower()
        state = str(check.get("state") or "").lower()
        if bucket == "fail":
            failing += 1
        elif bucket == "pending" or state in {"queued", "in_progress", "pending"}:
            pending += 1
        elif bucket == "pass":
            passing += 1

    total = len(checks)
    if failing:
        label = f"fail {failing}"
    elif pending:
        label = f"pending {pending}"
    elif total:
        label = "pass"
    else:
        label = "none"
    return CheckSummary(
        label=label,
        failing=failing,
        pending=pending,
        passing=passing,
        total=total,
    )


def shell_path(path: str) -> str:
    """Quote paths for backend shell commands while preserving home expansion."""
    if path in {"~", "~/"}:
        return "$HOME"
    if path.startswith("~/"):
        return f"$HOME/{shlex.quote(path[2:])}"
    return shlex.quote(path)


def job_has_pr_artifacts(job_id: str, backend) -> bool:
    """Detect stopped PTQ jobs that have enough artifacts to review for PR creation."""
    job_dir = shell_path(f"{backend.workspace}/jobs/{job_id}")
    result = backend.run(
        f"test -s {job_dir}/report.md || test -s {job_dir}/fix.diff",
        check=False,
    )
    return result.returncode == 0


def monitor_phase(job: JobRecord, status: JobStatus, pr_state: str, ci: CheckSummary) -> str:
    """Classify the row by who can unblock it next."""
    rebase_state = job.rebase_info.state
    if pr_state == "merged":
        return "merged/closed"
    if pr_state == "closed":
        return "merged/closed"
    if rebase_state == RebaseState.RUNNING:
        return "needs rebase"
    if rebase_state in {RebaseState.NEEDS_HUMAN, RebaseState.FAILED}:
        return "needs human review"
    if status == JobStatus.RUNNING:
        return "agent working"
    if ci.failing:
        return "needs fix"
    if ci.pending:
        return "waiting on CI"
    if ci.label == "pass":
        return "ready to merge"
    if pr_state == "open":
        return "needs human review"
    return "halted"


def ci_triage_command(pr_url: str) -> str:
    """Return the local CI triage command used before asking an agent to fix CI."""
    if not pr_url:
        return "-"
    return f"~/dotfiles/scripts/github_ci_triage {shlex.quote(pr_url)}"


def next_action(job_id: str, phase: str) -> str:
    """Return the primary PTQ command for the monitor row."""
    match phase:
        case "needs fix":
            return "triage failing CI"
        case "needs rebase":
            return f"ptq rebase {job_id}"
        case "needs human review":
            return f"ptq open {job_id}"
        case "ready for PR":
            return f"ptq pr {job_id}"
        case "ready to merge":
            return "trigger merge"
        case "agent working" | "waiting on CI":
            return f"ptq peek {job_id}"
        case "merged/closed":
            return f"ptq clean {job_id}"
        case _:
            return f"ptq status {job_id}"


def collect_monitor_rows(
    repo: JobRepository,
    *,
    include_without_pr: bool = False,
    force_refresh: bool = False,
) -> list[MonitorRow]:
    """Collect PTQ jobs into monitor rows using PTQ state as the source of truth."""
    rows: list[MonitorRow] = []
    for job_id, job in sorted(repo.list_all().items()):
        backend = backend_for_job(job)
        status = get_status(job, backend)
        ready_for_pr = (
            not job.pr_url
            and status == JobStatus.STOPPED
            and job_has_pr_artifacts(job_id, backend)
        )
        if not include_without_pr and not job.pr_url and not ready_for_pr:
            continue
        pr_state = (
            get_pr_state(backend, job.pr_url, force_refresh=force_refresh)
            if job.pr_url
            else "-"
        )
        ci = (
            summarize_pr_checks(job)
            if job.pr_url and pr_state not in {"closed", "merged"}
            else CheckSummary(label="-")
        )
        phase = "ready for PR" if ready_for_pr else monitor_phase(job, status, pr_state, ci)
        triage_command = ci_triage_command(job.pr_url or "")
        rows.append(
            MonitorRow(
                job_id=job_id,
                issue=f"#{job.issue}" if job.issue is not None else "adhoc",
                title=job.pr_title or job.name or "-",
                agent=job.agent,
                runs=job.runs,
                target=job.target,
                job_status=status,
                pr_state=pr_state,
                ci=ci,
                phase=phase,
                next_action=next_action(job_id, phase),
                takeover_command=takeover_for_job(job_id, job),
                ci_triage_command=triage_command,
                pr_url=job.pr_url or "",
            )
        )
    return rows
