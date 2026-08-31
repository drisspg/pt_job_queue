from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from ptq.application.pr_service import PRDefaults
from ptq.application.stack_service import StackCommit, StackStatus, StackSubmitResult
from ptq.cli import app
from ptq.domain.models import (
    JobRecord,
    PRResult,
    RebaseInfo,
    RebaseState,
    SubmissionMode,
)
from ptq.infrastructure.job_repository import JobRepository

runner = CliRunner()


def _make_repo(tmp_path: Path, records: list[JobRecord] | None = None) -> JobRepository:
    repo = JobRepository(tmp_path / "jobs.json")
    for r in records or []:
        repo.save(r)
    return repo


class TestStackCommand:
    def test_init_configures_job(self, tmp_path):
        repo = _make_repo(
            tmp_path,
            [JobRecord(job_id="stack-job", workspace="/tmp/ws")],
        )
        status = StackStatus(
            branch="ptq-stack/stack-job",
            base="main",
            remote="origin",
            base_ref="origin/main",
            dirty=False,
            has_merges=False,
            commits=(),
        )
        with (
            patch("ptq.cli._repo", return_value=repo),
            patch(
                "ptq.application.stack_service.initialize_stack",
                return_value=status,
            ) as initialize,
        ):
            result = runner.invoke(app, ["stack", "init", "stack-job"])

        assert result.exit_code == 0, result.output
        assert "Configured stack-job for ghstack" in result.output
        initialize.assert_called_once_with(repo, "stack-job", base=None)

    def test_show_renders_stack(self, tmp_path):
        repo = _make_repo(
            tmp_path,
            [JobRecord(job_id="stack-job", workspace="/tmp/ws")],
        )
        status = StackStatus(
            branch="feature-stack",
            base="main",
            remote="origin",
            base_ref="origin/main",
            dirty=False,
            has_merges=False,
            commits=(
                StackCommit(
                    sha="abc123456789",
                    subject="First change",
                    source_id="source-1",
                    pr_url="https://github.com/pytorch/pytorch/pull/99",
                ),
            ),
        )
        with (
            patch("ptq.cli._repo", return_value=repo),
            patch("ptq.application.stack_service.inspect_stack", return_value=status),
        ):
            result = runner.invoke(app, ["stack", "show", "stack-job"])

        assert result.exit_code == 0, result.output
        assert "feature-stack" in result.output
        assert "abc1234567 First change" in result.output
        assert "pull/99" in result.output

    def test_submit_passes_safe_options(self, tmp_path):
        repo = _make_repo(
            tmp_path,
            [
                JobRecord(
                    job_id="stack-job",
                    workspace="/tmp/ws",
                    submission_mode=SubmissionMode.GHSTACK,
                    stack_base="release/2.8",
                )
            ],
        )
        status = StackStatus(
            branch="feature-stack",
            base="release/2.8",
            remote="origin",
            base_ref="origin/release/2.8",
            dirty=False,
            has_merges=False,
            commits=(
                StackCommit(
                    sha="abc123",
                    subject="First change",
                    source_id="source-1",
                    pr_url="https://github.com/pytorch/pytorch/pull/99",
                ),
            ),
        )
        with (
            patch("ptq.cli._repo", return_value=repo),
            patch(
                "ptq.application.stack_service.submit_stack",
                return_value=StackSubmitResult(status=status, output="submitted"),
            ) as submit,
        ):
            result = runner.invoke(
                app,
                [
                    "stack",
                    "submit",
                    "stack-job",
                    "--draft",
                    "--update-metadata",
                ],
            )

        assert result.exit_code == 0, result.output
        assert "Submitting ghstack for stack-job" in result.output
        assert "Submitted 1 PR(s)" in result.output
        assert submit.call_args.kwargs["draft"] is True
        assert submit.call_args.kwargs["update_metadata"] is True
        assert submit.call_args.kwargs["stream_output"] is True
        assert callable(submit.call_args.kwargs["log"])


class TestPrCommand:
    def test_rejects_ghstack_job_before_prompting(self, tmp_path):
        repo = _make_repo(
            tmp_path,
            [
                JobRecord(
                    job_id="stack-job",
                    workspace="/tmp/ws",
                    submission_mode=SubmissionMode.GHSTACK,
                )
            ],
        )
        with (
            patch("ptq.cli._repo", return_value=repo),
            patch("ptq.cli.typer.prompt") as prompt,
            patch("ptq.application.pr_service.create_pr") as create_pr,
        ):
            result = runner.invoke(app, ["pr", "stack-job"])

        assert result.exit_code == 1
        assert "configured for ghstack" in result.output
        prompt.assert_not_called()
        create_pr.assert_not_called()

    def test_reuses_saved_human_note_when_note_omitted(self, tmp_path):
        repo = _make_repo(
            tmp_path,
            [
                JobRecord(
                    job_id="job-with-pr",
                    issue=187801,
                    workspace="/tmp/ws",
                    human_note="Saved reviewer note",
                )
            ],
        )

        with (
            patch("ptq.cli._repo", return_value=repo),
            patch("ptq.application.pr_service.create_pr") as mock_create_pr,
        ):
            mock_create_pr.return_value = PRResult(
                url="https://github.com/pytorch/pytorch/pull/187966",
                branch="ptq/187801",
            )
            result = runner.invoke(app, ["pr", "job-with-pr"])

        assert result.exit_code == 0, result.output
        assert "Reusing saved human note" in result.output
        assert mock_create_pr.call_args.kwargs["human_note"] == "Saved reviewer note"

    def test_note_option_overrides_saved_human_note(self, tmp_path):
        repo = _make_repo(
            tmp_path,
            [
                JobRecord(
                    job_id="job-with-pr",
                    issue=187801,
                    workspace="/tmp/ws",
                    human_note="Saved reviewer note",
                )
            ],
        )

        with (
            patch("ptq.cli._repo", return_value=repo),
            patch("ptq.application.pr_service.create_pr") as mock_create_pr,
        ):
            mock_create_pr.return_value = PRResult(
                url="https://github.com/pytorch/pytorch/pull/187966",
                branch="ptq/187801",
            )
            result = runner.invoke(app, ["pr", "job-with-pr", "-n", "Fresh note"])

        assert result.exit_code == 0, result.output
        assert "Reusing saved human note" not in result.output
        assert mock_create_pr.call_args.kwargs["human_note"] == "Fresh note"

    def test_prompts_for_title_when_interactive(self, tmp_path):
        repo = _make_repo(
            tmp_path,
            [
                JobRecord(
                    job_id="job-with-pr",
                    issue=187801,
                    workspace="/tmp/ws",
                )
            ],
        )

        with (
            patch("ptq.cli._repo", return_value=repo),
            patch("ptq.cli._can_prompt_for_pr_metadata", return_value=True),
            patch(
                "ptq.application.pr_service.pr_defaults",
                return_value=PRDefaults(title="Suggested Title", human_note=""),
            ),
            patch("ptq.cli.typer.prompt", return_value="Prompted Title") as prompt,
            patch("ptq.application.pr_service.create_pr") as mock_create_pr,
        ):
            mock_create_pr.return_value = PRResult(
                url="https://github.com/pytorch/pytorch/pull/187966",
                branch="ptq/187801",
            )
            result = runner.invoke(app, ["pr", "job-with-pr", "-n", "Fresh note"])

        assert result.exit_code == 0, result.output
        assert prompt.call_args.kwargs["default"] == "Suggested Title"
        assert mock_create_pr.call_args.kwargs["title"] == "Prompted Title"

    def test_uses_github_human_note_default(self, tmp_path):
        repo = _make_repo(
            tmp_path,
            [
                JobRecord(
                    job_id="job-with-pr",
                    issue=187801,
                    workspace="/tmp/ws",
                    human_note="Saved reviewer note",
                )
            ],
        )

        with (
            patch("ptq.cli._repo", return_value=repo),
            patch(
                "ptq.application.pr_service.pr_defaults",
                return_value=PRDefaults(
                    title="GitHub Title",
                    human_note="GitHub reviewer note",
                    synced_from_github=True,
                    human_note_synced_from_github=True,
                ),
            ),
            patch("ptq.application.pr_service.create_pr") as mock_create_pr,
        ):
            mock_create_pr.return_value = PRResult(
                url="https://github.com/pytorch/pytorch/pull/187966",
                branch="ptq/187801",
            )
            result = runner.invoke(app, ["pr", "job-with-pr"])

        assert result.exit_code == 0, result.output
        assert "Reusing GitHub human note" in result.output
        assert mock_create_pr.call_args.kwargs["human_note"] == "GitHub reviewer note"


def _make_clean_repo(tmp_path: Path) -> JobRepository:
    return _make_repo(
        tmp_path,
        [
            JobRecord(
                job_id="job-old",
                issue=100,
                workspace="/tmp/ws",
            ),
            JobRecord(
                job_id="job-new",
                issue=200,
                workspace="/tmp/ws",
            ),
        ],
    )


class TestCleanSingleJob:
    def test_removes_job_from_db(self, tmp_path):
        repo = _make_clean_repo(tmp_path)
        mock_backend = MagicMock()
        mock_backend.workspace = "/tmp/ws"
        mock_backend.run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        with (
            patch("ptq.cli._repo", return_value=repo),
            patch(
                "ptq.application.job_service.backend_for_job", return_value=mock_backend
            ),
        ):
            result = runner.invoke(app, ["clean", "job-old", "--force"])
        assert result.exit_code == 0, result.output
        assert "job-old" not in repo.list_all()
        assert "removed" in result.output

    def test_unknown_job_fails(self, tmp_path):
        repo = _make_clean_repo(tmp_path)
        with patch("ptq.cli._repo", return_value=repo):
            result = runner.invoke(app, ["clean", "nonexistent"])
        assert result.exit_code == 1
        assert "Unknown job" in result.output


class TestCleanMachine:
    def test_bulk_clean_removes_jobs(self, tmp_path):
        repo = _make_clean_repo(tmp_path)
        mock_backend = MagicMock()
        mock_backend.workspace = "/tmp/ws"
        mock_backend.run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        with (
            patch("ptq.cli._repo", return_value=repo),
            patch(
                "ptq.application.job_service.backend_for_job", return_value=mock_backend
            ),
        ):
            result = runner.invoke(app, ["clean", "local", "--force"])
        assert result.exit_code == 0, result.output
        remaining = repo.list_all()
        assert remaining == {}


class TestSetupCLI:
    def test_help_has_no_remote_target_options(self):
        result = runner.invoke(app, ["setup", "--help"])
        assert result.exit_code == 0
        assert "--machine" not in result.output
        assert "--local" not in result.output


class TestList:
    def test_list_shows_pr_and_rebase_state(self, tmp_path):
        repo = _make_repo(
            tmp_path,
            [
                JobRecord(
                    job_id="job-1",
                    issue=176093,
                    workspace="/tmp/ws",
                    pr_url="https://github.com/pytorch/pytorch/pull/176243",
                    rebase=RebaseInfo(state=RebaseState.NEEDS_HUMAN),
                )
            ],
        )
        mock_backend = MagicMock()
        mock_backend.workspace = "/tmp/ws"

        with (
            patch("ptq.cli._repo", return_value=repo),
            patch(
                "ptq.infrastructure.backends.backend_for_job", return_value=mock_backend
            ),
            patch("ptq.application.pr_service.get_pr_state", return_value="closed"),
        ):
            result = runner.invoke(app, ["list"])

        assert result.exit_code == 0, result.output
        assert "PR" in result.output
        assert "Rebase" in result.output
        assert "closed" in result.output
        assert "human" in result.output

    def test_list_shows_dashes_when_no_pr_or_rebase(self, tmp_path):
        repo = _make_repo(
            tmp_path,
            [
                JobRecord(
                    job_id="job-2",
                    issue=176094,
                    workspace="/tmp/ws",
                )
            ],
        )
        mock_backend = MagicMock()
        mock_backend.workspace = "/tmp/ws"

        with (
            patch("ptq.cli._repo", return_value=repo),
            patch(
                "ptq.infrastructure.backends.backend_for_job", return_value=mock_backend
            ),
        ):
            result = runner.invoke(app, ["list"])

        assert result.exit_code == 0, result.output
        assert "PR" in result.output
        assert "Rebase" in result.output
        assert "#176" in result.output
