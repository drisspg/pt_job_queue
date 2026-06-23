from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from ptq.application.monitor_service import collect_monitor_rows
from ptq.cli import app
from ptq.domain.models import JobRecord, JobStatus
from ptq.infrastructure.job_repository import JobRepository

runner = CliRunner()


def _repo(tmp_path: Path, records: list[JobRecord]) -> JobRepository:
    repo = JobRepository(tmp_path / "jobs.json")
    for record in records:
        repo.save(record)
    return repo


def test_collect_monitor_rows_prioritizes_ci_triage_for_failing_pr(tmp_path):
    repo = _repo(
        tmp_path,
        [
            JobRecord(
                job_id="job-1",
                issue=123,
                local=True,
                workspace="/tmp/ws",
                agent="pi",
                pr_url="https://github.com/pytorch/pytorch/pull/123",
                pr_title="Fix thing",
            )
        ],
    )
    backend = MagicMock()
    backend.run.return_value = MagicMock(
        returncode=0,
        stdout='[{"bucket":"fail","name":"linux test","conclusion":"failure"}]',
    )

    with (
        patch("ptq.application.monitor_service.backend_for_job", return_value=backend),
        patch("ptq.application.monitor_service.get_status", return_value=JobStatus.STOPPED),
        patch("ptq.application.monitor_service.get_pr_state", return_value="open"),
    ):
        rows = collect_monitor_rows(repo)

    assert len(rows) == 1
    assert rows[0].phase == "needs fix"
    assert rows[0].next_action == "triage failing CI"
    assert rows[0].ci_triage_command == (
        "~/dotfiles/scripts/github_ci_triage "
        "https://github.com/pytorch/pytorch/pull/123"
    )
    assert rows[0].takeover_command == (
        "cd /tmp/ws/jobs/job-1 && source .venv/bin/activate"
    )


def test_collect_monitor_rows_parses_checks_from_nonzero_gh_exit(tmp_path):
    repo = _repo(
        tmp_path,
        [
            JobRecord(
                job_id="job-pending",
                issue=124,
                local=True,
                workspace="/tmp/ws",
                agent="pi",
                pr_url="https://github.com/pytorch/pytorch/pull/124",
            )
        ],
    )
    backend = MagicMock()
    backend.run.return_value = MagicMock(
        returncode=8,
        stdout='[{"bucket":"pending","name":"linux test","state":"PENDING"}]',
    )

    with (
        patch("ptq.application.monitor_service.backend_for_job", return_value=backend),
        patch("ptq.application.monitor_service.get_status", return_value=JobStatus.STOPPED),
        patch("ptq.application.monitor_service.get_pr_state", return_value="open"),
    ):
        rows = collect_monitor_rows(repo)

    assert len(rows) == 1
    assert rows[0].ci.label == "pending 1"
    assert rows[0].phase == "waiting on CI"


def test_collect_monitor_rows_includes_pr_ready_jobs_without_pr_url(tmp_path):
    repo = _repo(
        tmp_path,
        [
            JobRecord(
                job_id="job-ready",
                issue=456,
                local=True,
                workspace="/tmp/ws",
                agent="pi",
            )
        ],
    )
    backend = MagicMock()
    backend.run.return_value = MagicMock(returncode=0, stdout="", stderr="")

    with (
        patch("ptq.application.monitor_service.backend_for_job", return_value=backend),
        patch("ptq.application.monitor_service.get_status", return_value=JobStatus.STOPPED),
    ):
        rows = collect_monitor_rows(repo)

    assert len(rows) == 1
    assert rows[0].phase == "ready for PR"
    assert rows[0].next_action == "ptq pr job-ready"


def test_collect_monitor_rows_checks_pr_ready_artifacts_under_home_workspace(tmp_path):
    repo = _repo(
        tmp_path,
        [
            JobRecord(
                job_id="job-ready",
                issue=456,
                local=True,
                workspace="~/.ptq_workspace",
                agent="pi",
            )
        ],
    )
    backend = MagicMock()
    backend.workspace = "~/.ptq_workspace"
    backend.run.return_value = MagicMock(returncode=0, stdout="", stderr="")

    with (
        patch("ptq.application.monitor_service.backend_for_job", return_value=backend),
        patch("ptq.application.monitor_service.get_status", return_value=JobStatus.STOPPED),
    ):
        rows = collect_monitor_rows(repo)

    assert len(rows) == 1
    assert rows[0].phase == "ready for PR"
    backend.run.assert_called_with(
        "test -s $HOME/.ptq_workspace/jobs/job-ready/report.md || "
        "test -s $HOME/.ptq_workspace/jobs/job-ready/fix.diff",
        check=False,
    )


def test_monitor_cli_prints_failing_ci_triage_command(tmp_path):
    repo = _repo(
        tmp_path,
        [
            JobRecord(
                job_id="job-1",
                issue=123,
                local=True,
                workspace="/tmp/ws",
                agent="pi",
                pr_url="https://github.com/pytorch/pytorch/pull/123",
            )
        ],
    )
    row = MagicMock()
    row.phase = "needs fix"
    row.job_id = "job-1"
    row.issue = "#123"
    row.pr_state = "open"
    row.ci.label = "fail 1"
    row.agent = "pi"
    row.runs = 1
    row.target = "local"
    row.next_action = "triage failing CI"
    row.takeover_command = "cd /tmp/ws/jobs/job-1 && source .venv/bin/activate"
    row.ci_triage_command = "~/dotfiles/scripts/github_ci_triage https://github.com/pytorch/pytorch/pull/123"

    with (
        patch("ptq.cli._repo", return_value=repo),
        patch("ptq.application.monitor_service.collect_monitor_rows", return_value=[row]),
    ):
        result = runner.invoke(app, ["monitor"])

    assert result.exit_code == 0, result.output
    assert "PTQ PR Monitor" in result.output
    assert "Failing CI triage" in result.output
    assert "github_ci_triage" in result.output
    assert "https://github.com/pytorch/pytorch/pull/123" in result.output


def test_monitor_cli_handles_no_pr_jobs(tmp_path):
    repo = _repo(tmp_path, [])
    with patch("ptq.cli._repo", return_value=repo):
        result = runner.invoke(app, ["monitor"])

    assert result.exit_code == 0, result.output
    assert "No PTQ PR jobs to monitor" in result.output
