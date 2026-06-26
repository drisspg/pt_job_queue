from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from ptq.application.monitor_service import CheckSummary, MonitorRow
from ptq.application.supervisor_service import (
    classify_failing_ci,
    run_ci_triage,
    supervise_row,
)
from ptq.domain.models import JobRecord, JobStatus


def row() -> MonitorRow:
    return MonitorRow(
        job_id="job-1",
        issue="#123",
        title="Fix SDPA",
        agent="pi",
        runs=1,
        target="local",
        job_status=JobStatus.STOPPED,
        pr_state="open",
        ci=CheckSummary(label="fail 1", failing=1, total=1),
        phase="needs CI review",
        next_action="ptq open job-1",
        takeover_command="cd /tmp/ws/jobs/job-1 && source .venv/bin/activate",
        ci_triage_command="~/dotfiles/scripts/github_ci_triage https://github.com/pytorch/pytorch/pull/123",
        merge_ignore_command="gh pr comment https://github.com/pytorch/pytorch/pull/123 --body '@pytorchbot merge -i'",
        pr_url="https://github.com/pytorch/pytorch/pull/123",
    )


def test_classify_nonlanding_unrelated_failure_as_unrelated_ci():
    drci = """
    <details open><summary><b>NEW FAILURE</b> - The following job has failed:</summary>
    tf_efficientnet_b0
    The failure already existed at the merge base before this commit. The suspect commit only modifies test/inductor/test_compiled_autograd.py, which is unrelated to this patch.
    </details>
    <details><summary><b>FLAKY</b></summary></details>
    """

    status, summary, action = classify_failing_ci(row(), drci, "")

    assert status == "unrelated CI"
    assert "unrelated" in summary
    assert action == "uv run ptq open job-1"


def test_classify_landing_stopped_unrelated_ci_as_merge_ignore_candidate():
    test_row = row()
    test_row.phase = "unrelated CI"
    test_row.can_merge_ignore = True

    status, summary, action = classify_failing_ci(test_row, "", "")

    assert status == "merge-ignore candidate"
    assert "was landing" in summary
    assert action == "gh pr comment https://github.com/pytorch/pytorch/pull/123 --body '@pytorchbot merge -i'"


def test_classify_new_failure_as_needs_fix_when_ai_verdict_says_pr_modified_test():
    drci = """
    <details open><summary><b>NEW FAILURES</b> - The following jobs have failed:</summary>
    test/inductor/test_compiled_autograd.py::TestCompiledAutograd::test_trace_run_with_rng_state
    The commit directly modifies test_trace_run_with_rng_state. This exact test is the one failing.
    </details>
    """

    status, summary, action = classify_failing_ci(row(), drci, "")

    assert status == "needs fix"
    assert "overlap" in summary
    assert action == "uv run ptq open job-1"


def test_classify_merge_base_evidence_as_unrelated_ci_when_not_landing():
    drci = """
    <details open><summary><b>NEW FAILURE</b> - The following job has failed:</summary>
    test/inductor/test_compiled_autograd.py::TestCompiledAutograd::test_trace_run_with_rng_state
    The failure already existed at the merge base before this commit.
    </details>
    """

    status, summary, action = classify_failing_ci(row(), drci, "")

    assert status == "unrelated CI"
    assert "unrelated" in summary
    assert action == "uv run ptq open job-1"


def test_classify_ambiguous_new_failure_as_ci_review_without_triage():
    drci = """
    <details open><summary><b>NEW FAILURE</b> - The following job has failed:</summary>
    mysterious_test_failure
    </details>
    """

    status, summary, action = classify_failing_ci(row(), drci, "")

    assert status == "needs CI review"
    assert "needs bounded triage" in summary
    assert action == "uv run ptq open job-1"


def test_classify_ambiguous_new_failure_ignores_following_flaky_bucket():
    drci = """
    <details open><summary><b>NEW FAILURE</b> - The following job has failed:</summary>
    mysterious_test_failure
    </details>
    <details open><summary><b>FLAKY</b></summary>
    This unrelated flaky bucket should not classify the new failure.
    </details>
    """

    status, summary, action = classify_failing_ci(row(), drci, "")

    assert status == "needs CI review"
    assert "needs bounded triage" in summary
    assert action == "uv run ptq open job-1"


def test_classify_ambiguous_new_failure_ignores_non_decisive_triage_words():
    drci = """
    <details open><summary><b>NEW FAILURE</b> - The following job has failed:</summary>
    mysterious_test_failure
    </details>
    """
    triage_output = "The log mentions a previous flaky retry, but no verdict was found."

    status, summary, action = classify_failing_ci(row(), drci, triage_output)

    assert status == "needs human review"
    assert "not confidently" in summary
    assert action == "uv run ptq open job-1"


def test_run_ci_triage_persists_missing_helper_as_evidence(tmp_path):
    with patch("ptq.application.supervisor_service.Path.home", return_value=tmp_path):
        output, path = run_ci_triage(
            "https://github.com/pytorch/pytorch/pull/123",
            tmp_path / "supervisor",
        )

    assert "github_ci_triage helper not found" in output
    assert Path(path).read_text() == output


def test_supervise_row_leaves_landing_pr_waiting_without_triage():
    test_row = row()
    test_row.phase = "landing"
    test_row.next_action = "monitor merge"
    job = JobRecord(
        job_id="job-1",
        issue=123,
        local=True,
        workspace="/tmp/ws",
        agent="pi",
        pr_url="https://github.com/pytorch/pytorch/pull/123",
    )

    with (
        patch("ptq.application.supervisor_service.fetch_drci_body") as fetch_drci,
        patch("ptq.application.supervisor_service.run_ci_triage") as triage,
    ):
        verdict = supervise_row(test_row, job)

    assert verdict.status == "waiting"
    assert verdict.summary == "landing"
    assert verdict.suggested_action == "monitor merge"
    fetch_drci.assert_not_called()
    triage.assert_not_called()


def test_supervise_row_returns_worker_prompt_and_persists_evidence_path(tmp_path):
    test_row = row()
    job = JobRecord(
        job_id="job-1",
        issue=123,
        local=True,
        workspace="/tmp/ws",
        agent="pi",
        pr_url="https://github.com/pytorch/pytorch/pull/123",
    )

    with (
        patch(
            "ptq.application.supervisor_service.fetch_drci_body",
            return_value="<details><summary><b>NEW FAILURE</b></summary>already existed at the merge base</details>",
        ),
        patch(
            "ptq.application.supervisor_service.run_ci_triage",
            return_value=("triage", str(tmp_path / "triage.md")),
        ),
    ):
        verdict = supervise_row(test_row, job, output_root=tmp_path)

    assert verdict.status == "unrelated CI"
    assert str(tmp_path / "triage.md") in verdict.evidence
    assert "github_ci_triage https://github.com/pytorch/pytorch/pull/123" in verdict.worker_prompt
    assert "needs human review" in verdict.worker_prompt
