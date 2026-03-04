from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from ptq.cli import app
from ptq.domain.models import JobRecord, RebaseInfo, RebaseState
from ptq.infrastructure.job_repository import JobRepository

runner = CliRunner()


def _make_repo(tmp_path: Path, records: list[JobRecord] | None = None) -> JobRepository:
    repo = JobRepository(tmp_path / "jobs.json")
    for r in records or []:
        repo.save(r)
    return repo


class TestRunValidation:
    def test_no_issue_no_message_no_job_id(self):
        result = runner.invoke(app, ["run", "--local"])
        assert result.exit_code != 0
        assert "Provide --issue, --message, or a JOB_ID" in result.output

    def test_defaults_to_local_when_no_machine(self, tmp_path):
        repo = _make_repo(tmp_path)

        def fake_launch(r, b, req, **kw):
            repo.save(JobRecord(job_id="test-job", local=True, workspace="/tmp/ws"))
            return "test-job"

        with (
            patch("ptq.cli._repo", return_value=repo),
            patch(
                "ptq.application.run_service.launch", side_effect=fake_launch
            ) as mock_launch,
        ):
            result = runner.invoke(app, ["run", "-m", "hello", "--no-follow"])

        assert result.exit_code == 0, result.output
        mock_launch.assert_called_once()
        request = mock_launch.call_args.args[2]
        assert request.local is True

    def test_input_and_message_mutually_exclusive(self):
        result = runner.invoke(app, ["run", "-i", "f.md", "-m", "hello", "--local"])
        assert result.exit_code != 0
        assert "mutually exclusive" in result.output

    def test_input_file_not_found(self):
        result = runner.invoke(app, ["run", "-i", "/nonexistent/path.md", "--local"])
        assert result.exit_code != 0
        assert "File not found" in result.output

    def test_input_file_reads_contents(self, tmp_path):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("do the thing")
            f.flush()
            tmp_file = f.name

        repo = _make_repo(tmp_path)

        def fake_launch(r, b, req, **kw):
            repo.save(JobRecord(job_id="test-job", local=True, workspace="/tmp/ws"))
            return "test-job"

        with (
            patch("ptq.cli._repo", return_value=repo),
            patch(
                "ptq.application.run_service.launch", side_effect=fake_launch
            ) as mock_launch,
        ):
            result = runner.invoke(
                app, ["run", "-i", tmp_file, "--local", "--no-follow"]
            )

        Path(tmp_file).unlink()
        assert result.exit_code == 0, result.output
        mock_launch.assert_called_once()
        request = mock_launch.call_args.args[2]
        assert request.message == "do the thing"

    def test_agent_type_passed_through(self, tmp_path):
        repo = _make_repo(tmp_path)

        def fake_launch(r, b, req, **kw):
            repo.save(
                JobRecord(
                    job_id="test-job", local=True, workspace="/tmp/ws", agent="codex"
                )
            )
            return "test-job"

        with (
            patch("ptq.cli._repo", return_value=repo),
            patch(
                "ptq.application.run_service.launch", side_effect=fake_launch
            ) as mock_launch,
        ):
            result = runner.invoke(
                app, ["run", "-m", "hello", "--agent", "codex", "--no-follow"]
            )

        assert result.exit_code == 0, result.output
        request = mock_launch.call_args.args[2]
        assert request.agent_type == "codex"

    def test_rerun_passes_existing_job_id(self, tmp_path):
        repo = _make_repo(
            tmp_path,
            [
                JobRecord(
                    job_id="20260217-adhoc-abc123",
                    runs=1,
                    local=True,
                    workspace="/tmp/ws",
                    agent="cursor",
                ),
            ],
        )
        with (
            patch("ptq.cli._repo", return_value=repo),
            patch("ptq.application.run_service.launch") as mock_launch,
        ):
            mock_launch.return_value = "20260217-adhoc-abc123"
            result = runner.invoke(
                app, ["run", "20260217-adhoc-abc123", "-m", "try again"]
            )

        assert result.exit_code == 0, result.output
        request = mock_launch.call_args.args[2]
        assert request.existing_job_id == "20260217-adhoc-abc123"
        assert request.agent_type == "cursor"


def _make_clean_repo(tmp_path: Path) -> JobRepository:
    return _make_repo(
        tmp_path,
        [
            JobRecord(
                job_id="job-stopped",
                issue=100,
                runs=1,
                local=True,
                workspace="/tmp/ws",
                agent="claude",
            ),
            JobRecord(
                job_id="job-running",
                issue=200,
                runs=2,
                local=True,
                workspace="/tmp/ws",
                agent="codex",
                pid=99999,
            ),
        ],
    )


RUNNING_PID = 99999


class TestCleanSingleJob:
    def test_removes_job_from_db(self, tmp_path):
        repo = _make_clean_repo(tmp_path)
        mock_backend = MagicMock()
        mock_backend.workspace = "/tmp/ws"
        mock_backend.is_pid_alive.return_value = False

        with (
            patch("ptq.cli._repo", return_value=repo),
            patch(
                "ptq.application.job_service.backend_for_job", return_value=mock_backend
            ),
        ):
            result = runner.invoke(app, ["clean", "job-stopped"])
        assert result.exit_code == 0, result.output
        assert "job-stopped" not in repo.list_all()
        assert "removed" in result.output

    def test_unknown_target_treated_as_machine(self, tmp_path):
        repo = _make_clean_repo(tmp_path)
        mock_backend = MagicMock()
        mock_backend.workspace = "/tmp/ws"
        mock_backend.is_pid_alive.return_value = False

        with (
            patch("ptq.cli._repo", return_value=repo),
            patch(
                "ptq.infrastructure.backends.RemoteBackend", return_value=mock_backend
            ),
        ):
            result = runner.invoke(app, ["clean", "nonexistent-machine"])
        assert result.exit_code == 0
        assert "Nothing to clean" in result.output


class TestCleanMachine:
    def test_bulk_clean_removes_stopped_jobs(self, tmp_path):
        repo = _make_clean_repo(tmp_path)
        mock_backend = MagicMock()
        mock_backend.workspace = "/tmp/ws"
        mock_backend.is_pid_alive.side_effect = lambda pid: pid == RUNNING_PID

        with (
            patch("ptq.cli._repo", return_value=repo),
            patch(
                "ptq.infrastructure.backends.LocalBackend", return_value=mock_backend
            ),
        ):
            result = runner.invoke(app, ["clean", "--local"])
        assert result.exit_code == 0, result.output
        remaining = repo.list_all()
        assert "job-stopped" not in remaining
        assert "job-running" in remaining


class TestSetupValidation:
    def test_no_machine_no_local(self):
        result = runner.invoke(app, ["setup"])
        assert result.exit_code != 0


class TestList:
    def test_list_shows_pr_and_rebase_state(self, tmp_path):
        repo = _make_repo(
            tmp_path,
            [
                JobRecord(
                    job_id="job-1",
                    issue=176093,
                    local=True,
                    workspace="/tmp/ws",
                    agent="cursor",
                    runs=46,
                    pr_url="https://github.com/pytorch/pytorch/pull/176243",
                    rebase=RebaseInfo(state=RebaseState.NEEDS_HUMAN),
                )
            ],
        )
        mock_backend = MagicMock()
        mock_backend.workspace = "/tmp/ws"
        mock_backend.is_pid_alive.return_value = False

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
                    local=True,
                    workspace="/tmp/ws",
                    agent="claude",
                )
            ],
        )
        mock_backend = MagicMock()
        mock_backend.workspace = "/tmp/ws"
        mock_backend.is_pid_alive.return_value = False

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
