from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from ptq.application.monitor_service import collect_monitor_rows
from ptq.cli import app, _monitor_issue_markup, _monitor_pr_markup
from ptq.domain.models import JobRecord, JobStatus
from ptq.infrastructure.job_repository import JobRepository

runner = CliRunner()


def _repo(tmp_path: Path, records: list[JobRecord]) -> JobRepository:
    repo = JobRepository(tmp_path / "jobs.json")
    for record in records:
        repo.save(record)
    return repo


def test_collect_monitor_rows_marks_raw_failing_pr_for_ci_review(tmp_path):
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
    assert rows[0].phase == "needs CI review"
    assert rows[0].next_action == "ptq open job-1"
    assert rows[0].ci_triage_command == (
        "~/dotfiles/scripts/github_ci_triage "
        "https://github.com/pytorch/pytorch/pull/123"
    )
    assert rows[0].takeover_command == (
        "cd /tmp/ws/jobs/job-1 && source .venv/bin/activate"
    )


def test_collect_monitor_rows_marks_merging_pr_as_landing_even_with_red_ci(tmp_path):
    repo = _repo(
        tmp_path,
        [
            JobRecord(
                job_id="job-landing",
                issue=125,
                local=True,
                workspace="/tmp/ws",
                agent="pi",
                pr_url="https://github.com/pytorch/pytorch/pull/125",
            )
        ],
    )
    backend = MagicMock()

    def run(command, check=False):
        if "gh pr checks" in command:
            return MagicMock(
                returncode=8,
                stdout='[{"bucket":"fail","name":"linux test","state":"FAILURE"}]',
            )
        if "gh pr view" in command:
            return MagicMock(
                returncode=0,
                stdout=(
                    '{"labels":[{"name":"merging"}],'
                    '"mergeStateStatus":"BLOCKED",'
                    '"comments":[]}'
                ),
            )
        return MagicMock(returncode=0, stdout="")

    backend.run.side_effect = run

    with (
        patch("ptq.application.monitor_service.backend_for_job", return_value=backend),
        patch("ptq.application.monitor_service.get_status", return_value=JobStatus.STOPPED),
        patch("ptq.application.monitor_service.get_pr_state", return_value="open"),
    ):
        rows = collect_monitor_rows(repo)

    assert len(rows) == 1
    assert rows[0].phase == "landing"
    assert rows[0].ci.label == "fail 1"
    assert rows[0].next_action == "monitor merge"


def test_collect_monitor_rows_marks_nonlanding_unrelated_ci_without_merge_command(tmp_path):
    repo = _repo(
        tmp_path,
        [
            JobRecord(
                job_id="job-unrelated",
                issue=126,
                local=True,
                workspace="/tmp/ws",
                agent="pi",
                pr_url="https://github.com/pytorch/pytorch/pull/126",
            )
        ],
    )
    backend = MagicMock()
    drci_comment = (
        "<!-- drci-comment-start -->\n"
        "## :x: 2 Unrelated Failures\n"
        "<details><summary><b>FLAKY</b></summary></details>"
    )

    def run(command, check=False):
        if "gh pr checks" in command:
            return MagicMock(
                returncode=8,
                stdout='[{"bucket":"fail","name":"linux test","state":"FAILURE"}]',
            )
        if "gh pr view" in command:
            return MagicMock(
                returncode=0,
                stdout=json.dumps(
                    {
                        "labels": [],
                        "mergeStateStatus": "BLOCKED",
                        "comments": [
                            {
                                "author": {"login": "pytorch-bot"},
                                "body": drci_comment,
                            }
                        ],
                    }
                ),
            )
        return MagicMock(returncode=0, stdout="")

    backend.run.side_effect = run

    with (
        patch("ptq.application.monitor_service.backend_for_job", return_value=backend),
        patch("ptq.application.monitor_service.get_status", return_value=JobStatus.STOPPED),
        patch("ptq.application.monitor_service.get_pr_state", return_value="open"),
    ):
        rows = collect_monitor_rows(repo)

    assert len(rows) == 1
    assert rows[0].phase == "unrelated CI"
    assert rows[0].next_action == "ptq peek job-unrelated"
    assert rows[0].can_merge_ignore is False
    assert rows[0].merge_ignore_command == (
        "gh pr comment https://github.com/pytorch/pytorch/pull/126 "
        "--body '@pytorchbot merge -i'"
    )


def test_collect_monitor_rows_suggests_merge_ignore_after_stopped_landing(tmp_path):
    repo = _repo(
        tmp_path,
        [
            JobRecord(
                job_id="job-unrelated-landing",
                issue=126,
                local=True,
                workspace="/tmp/ws",
                agent="pi",
                pr_url="https://github.com/pytorch/pytorch/pull/126",
            )
        ],
    )
    backend = MagicMock()
    drci_comment = (
        "<!-- drci-comment-start -->\n"
        "## :x: 2 Unrelated Failures\n"
        "<details><summary><b>FLAKY</b></summary></details>"
    )

    def run(command, check=False):
        if "gh pr checks" in command:
            return MagicMock(
                returncode=8,
                stdout='[{"bucket":"fail","name":"linux test","state":"FAILURE"}]',
            )
        if "gh pr view" in command:
            return MagicMock(
                returncode=0,
                stdout=json.dumps(
                    {
                        "labels": [],
                        "mergeStateStatus": "BLOCKED",
                        "comments": [
                            {
                                "author": {"login": "pytorch-bot"},
                                "body": drci_comment,
                            },
                            {
                                "author": {"login": "pytorchmergebot"},
                                "body": "Merge failed because required checks failed.",
                            },
                        ],
                    }
                ),
            )
        return MagicMock(returncode=0, stdout="")

    backend.run.side_effect = run

    with (
        patch("ptq.application.monitor_service.backend_for_job", return_value=backend),
        patch("ptq.application.monitor_service.get_status", return_value=JobStatus.STOPPED),
        patch("ptq.application.monitor_service.get_pr_state", return_value="open"),
    ):
        rows = collect_monitor_rows(repo)

    assert len(rows) == 1
    assert rows[0].phase == "unrelated CI"
    assert rows[0].next_action == "comment @pytorchbot merge -i"
    assert rows[0].can_merge_ignore is True


def test_collect_monitor_rows_uses_ai_unrelated_verdict_text(tmp_path):
    repo = _repo(
        tmp_path,
        [
            JobRecord(
                job_id="job-ai-unrelated",
                issue=129,
                local=True,
                workspace="/tmp/ws",
                agent="pi",
                pr_url="https://github.com/pytorch/pytorch/pull/129",
            )
        ],
    )
    backend = MagicMock()
    drci_comment = (
        "<!-- drci-comment-start -->\n"
        "<details open><summary><b>NEW FAILURE</b> - The following job has failed:</summary>\n"
        "* inductor_timm\n"
        "  <details><summary>AI verdict: badge</summary><blockquote>\n"
        "  The failure already existed at the merge base with the same job, same error. "
        "The suspect commit only updates docs, causing no behavioral change.\n"
        "  </blockquote></details>\n"
        "</details>\n"
        "<!-- drci-comment-end -->"
    )

    def run(command, check=False):
        if "gh pr checks" in command:
            return MagicMock(
                returncode=8,
                stdout='[{"bucket":"fail","name":"linux test","state":"FAILURE"}]',
            )
        if "gh pr view" in command:
            return MagicMock(
                returncode=0,
                stdout=json.dumps(
                    {
                        "labels": [],
                        "mergeStateStatus": "BLOCKED",
                        "reviewDecision": "APPROVED",
                        "comments": [
                            {
                                "author": {"login": "pytorch-bot"},
                                "body": drci_comment,
                            }
                        ],
                    }
                ),
            )
        return MagicMock(returncode=0, stdout="")

    backend.run.side_effect = run

    with (
        patch("ptq.application.monitor_service.backend_for_job", return_value=backend),
        patch("ptq.application.monitor_service.get_status", return_value=JobStatus.STOPPED),
        patch("ptq.application.monitor_service.get_pr_state", return_value="open"),
    ):
        rows = collect_monitor_rows(repo)

    assert len(rows) == 1
    assert rows[0].phase == "unrelated CI"
    assert rows[0].next_action == "review merge readiness"


def test_collect_monitor_rows_sends_new_failures_to_ci_review(tmp_path):
    repo = _repo(
        tmp_path,
        [
            JobRecord(
                job_id="job-new-failures",
                issue=127,
                local=True,
                workspace="/tmp/ws",
                agent="pi",
                pr_url="https://github.com/pytorch/pytorch/pull/127",
            )
        ],
    )
    backend = MagicMock()
    drci_comment = (
        "<!-- drci-comment-start -->\n"
        "## :x: 11 New Failures, 27 Pending, 5 Unrelated Failures\n"
        "<details open><summary><b>NEW FAILURES</b></summary></details>\n"
        "<details><summary><b>FLAKY</b></summary></details>\n"
        "<details><summary><b>BROKEN TRUNK</b></summary></details>"
    )

    def run(command, check=False):
        if "gh pr checks" in command:
            return MagicMock(
                returncode=8,
                stdout='[{"bucket":"fail","name":"linux test","state":"FAILURE"}]',
            )
        if "gh pr view" in command:
            return MagicMock(
                returncode=0,
                stdout=json.dumps(
                    {
                        "labels": [],
                        "mergeStateStatus": "BLOCKED",
                        "comments": [
                            {
                                "author": {"login": "pytorch-bot"},
                                "body": drci_comment,
                            }
                        ],
                    }
                ),
            )
        return MagicMock(returncode=0, stdout="")

    backend.run.side_effect = run

    with (
        patch("ptq.application.monitor_service.backend_for_job", return_value=backend),
        patch("ptq.application.monitor_service.get_status", return_value=JobStatus.STOPPED),
        patch("ptq.application.monitor_service.get_pr_state", return_value="open"),
    ):
        rows = collect_monitor_rows(repo)

    assert len(rows) == 1
    assert rows[0].phase == "needs CI review"
    assert rows[0].next_action == "ptq open job-new-failures"


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


def test_collect_monitor_rows_keeps_passing_draft_out_of_ready_to_merge(tmp_path):
    repo = _repo(
        tmp_path,
        [
            JobRecord(
                job_id="job-draft",
                issue=128,
                local=True,
                workspace="/tmp/ws",
                agent="pi",
                pr_url="https://github.com/pytorch/pytorch/pull/128",
            )
        ],
    )
    backend = MagicMock()

    def run(command, check=False):
        if "gh pr checks" in command:
            return MagicMock(
                returncode=0,
                stdout='[{"bucket":"pass","name":"linux test","state":"SUCCESS"}]',
            )
        if "gh pr view" in command:
            return MagicMock(
                returncode=0,
                stdout=json.dumps(
                    {
                        "labels": [],
                        "mergeStateStatus": "CLEAN",
                        "comments": [],
                        "isDraft": True,
                    }
                ),
            )
        return MagicMock(returncode=0, stdout="")

    backend.run.side_effect = run

    with (
        patch("ptq.application.monitor_service.backend_for_job", return_value=backend),
        patch("ptq.application.monitor_service.get_status", return_value=JobStatus.STOPPED),
        patch("ptq.application.monitor_service.get_pr_state", return_value="open"),
    ):
        rows = collect_monitor_rows(repo)

    assert len(rows) == 1
    assert rows[0].pr_is_draft is True
    assert rows[0].phase == "needs human review"


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


def test_monitor_cli_marks_issue_and_pr_cells_as_terminal_links():
    row = MagicMock()
    row.issue = "#150321"
    row.pr_state = "open"
    row.pr_url = "https://github.com/pytorch/pytorch/pull/188178"
    row.pr_is_draft = False

    assert _monitor_issue_markup(row) == (
        "[cyan link=https://github.com/pytorch/pytorch/issues/150321]#150321[/]"
    )
    assert _monitor_pr_markup(row) == (
        "[green link=https://github.com/pytorch/pytorch/pull/188178]#188178 open[/]"
    )


def test_monitor_cli_prints_failing_ci_review_command(tmp_path):
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
    row.phase = "needs CI review"
    row.job_id = "job-1"
    row.issue = "#123"
    row.pr_state = "open"
    row.pr_is_draft = False
    row.ci.label = "fail 1"
    row.ci.failing = 1
    row.agent = "pi"
    row.runs = 1
    row.target = "local"
    row.next_action = "ptq open job-1"
    row.takeover_command = "cd /tmp/ws/jobs/job-1 && source .venv/bin/activate"
    row.ci_triage_command = "~/dotfiles/scripts/github_ci_triage https://github.com/pytorch/pytorch/pull/123"

    with (
        patch("ptq.cli._repo", return_value=repo),
        patch("ptq.application.monitor_service.collect_monitor_rows", return_value=[row]),
    ):
        result = runner.invoke(app, ["monitor"])

    assert result.exit_code == 0, result.output
    assert "PTQ PR Monitor" in result.output
    assert "Failing CI review" in result.output
    assert "github_ci_triage" in result.output
    assert "https://github.com/pytorch/pytorch/pull/123" in result.output


def test_monitor_cli_marks_draft_pr_and_omits_agent_target_columns(tmp_path):
    repo = _repo(
        tmp_path,
        [
            JobRecord(
                job_id="job-draft",
                issue=128,
                local=True,
                workspace="/tmp/ws",
                agent="pi",
                pr_url="https://github.com/pytorch/pytorch/pull/128",
            )
        ],
    )
    row = MagicMock()
    row.phase = "waiting on CI"
    row.job_id = "job-draft"
    row.issue = "#128"
    row.pr_state = "open"
    row.pr_is_draft = True
    row.ci.label = "pending 1"
    row.ci.failing = 0
    row.next_action = "ptq peek job-draft"
    row.takeover_command = "cd /tmp/ws/jobs/job-draft && source .venv/bin/activate"

    with (
        patch("ptq.cli._repo", return_value=repo),
        patch("ptq.application.monitor_service.collect_monitor_rows", return_value=[row]),
    ):
        result = runner.invoke(app, ["monitor"])

    assert result.exit_code == 0, result.output
    assert "draft" in result.output
    assert "Agent" not in result.output
    assert "Target" not in result.output


def test_monitor_cli_prints_merge_ignore_command_for_unrelated_ci(tmp_path):
    repo = _repo(
        tmp_path,
        [
            JobRecord(
                job_id="job-unrelated",
                issue=126,
                local=True,
                workspace="/tmp/ws",
                agent="pi",
                pr_url="https://github.com/pytorch/pytorch/pull/126",
            )
        ],
    )
    row = MagicMock()
    row.phase = "unrelated CI"
    row.job_id = "job-unrelated"
    row.issue = "#126"
    row.pr_state = "open"
    row.pr_is_draft = False
    row.ci.label = "fail 1"
    row.ci.failing = 1
    row.agent = "pi"
    row.runs = 1
    row.target = "local"
    row.next_action = "comment @pytorchbot merge -i"
    row.can_merge_ignore = True
    row.takeover_command = "cd /tmp/ws/jobs/job-unrelated && source .venv/bin/activate"
    row.merge_ignore_command = (
        "gh pr comment https://github.com/pytorch/pytorch/pull/126 "
        "--body '@pytorchbot merge -i'"
    )

    with (
        patch("ptq.cli._repo", return_value=repo),
        patch("ptq.application.monitor_service.collect_monitor_rows", return_value=[row]),
    ):
        result = runner.invoke(app, ["monitor"])

    assert result.exit_code == 0, result.output
    assert "PyTorchBot merge-ignore commands" in result.output
    assert "@pytorchbot merge -i" in result.output


def test_monitor_watch_uses_resize_safe_live_screen(tmp_path):
    repo = _repo(tmp_path, [])
    live_context = MagicMock()
    live_context.__enter__.return_value = MagicMock()
    live_context.__exit__.return_value = False

    with (
        patch("ptq.cli._repo", return_value=repo),
        patch("ptq.application.monitor_service.collect_monitor_rows", return_value=[]),
        patch("rich.live.Live", return_value=live_context) as live_cls,
        patch("ptq.cli.time.sleep", side_effect=KeyboardInterrupt),
    ):
        result = runner.invoke(app, ["monitor", "--watch"])

    assert result.exit_code == 0, result.output
    assert live_cls.call_args.kwargs["screen"] is True
    assert live_cls.call_args.kwargs["vertical_overflow"] == "ellipsis"


def test_monitor_cli_handles_no_pr_jobs(tmp_path):
    repo = _repo(tmp_path, [])
    with patch("ptq.cli._repo", return_value=repo):
        result = runner.invoke(app, ["monitor"])

    assert result.exit_code == 0, result.output
    assert "No PTQ PR jobs to monitor" in result.output
