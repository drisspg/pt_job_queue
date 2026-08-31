from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from ptq.application.herdr_service import cwd_from_takeover_command, open_job_workspace
from ptq.cli import app
from ptq.domain.models import JobRecord
from ptq.infrastructure.job_repository import JobRepository

runner = CliRunner()


def test_cwd_from_takeover_command_expands_home():
    command = "cd $HOME/.ptq_workspace/jobs/job-1 && source .venv/bin/activate"
    assert cwd_from_takeover_command(command).endswith("/.ptq_workspace/jobs/job-1")


def test_open_job_workspace_uses_takeover_command_as_pane_command():
    client = MagicMock()
    client.run.side_effect = [
        MagicMock(
            returncode=0,
            stdout=json.dumps(
                {
                    "result": {
                        "root_pane": {
                            "pane_id": "pane-1",
                            "workspace_id": "workspace-1",
                        }
                    }
                }
            ),
            stderr="",
        ),
        MagicMock(returncode=0, stdout="", stderr=""),
    ]
    command = "cd /tmp/ws/jobs/job-1 && source .venv/bin/activate"

    workspace = open_job_workspace("job-1", command, client=client, focus=False)

    assert workspace.workspace_id == "workspace-1"
    assert workspace.pane_id == "pane-1"
    assert client.run.call_args_list[0].args[0] == [
        "workspace",
        "create",
        "--cwd",
        "/tmp/ws/jobs/job-1",
        "--label",
        "ptq job-1",
        "--no-focus",
    ]
    assert client.run.call_args_list[1].args[0] == ["pane", "run", "pane-1", command]


def test_open_cli_resolves_job_and_opens_herdr_workspace(tmp_path: Path):
    repo = JobRepository(tmp_path / "jobs.json")
    repo.save(JobRecord(job_id="job-1", issue=123, workspace="/tmp/ws"))
    opened = MagicMock(workspace_id="workspace-1", pane_id="pane-1")

    with (
        patch("ptq.cli._repo", return_value=repo),
        patch("ptq.application.worktree_service.prepare_job_worktree"),
        patch(
            "ptq.application.herdr_service.open_job_workspace", return_value=opened
        ) as mock_open,
    ):
        result = runner.invoke(app, ["open", "123", "--no-focus"])

    assert result.exit_code == 0, result.output
    mock_open.assert_called_once_with(
        "job-1",
        "cd /tmp/ws/jobs/job-1 && source .venv/bin/activate",
        label="ptq #123",
        focus=False,
    )
    assert "Opened PTQ job Herdr workspace" in result.output


def test_open_cli_creates_issue_worktree_before_opening_herdr(tmp_path: Path):
    repo = JobRepository(tmp_path / "jobs.json")
    backend = MagicMock(workspace="/tmp/ws")
    opened = MagicMock(workspace_id="workspace-1", pane_id="pane-1")

    def ensure_job(job_repo, _backend, **kwargs):
        assert kwargs["issue_number"] == 123
        job_repo.save(
            JobRecord(
                job_id="job-1",
                issue=123,
                workspace="/tmp/ws",
            )
        )
        return "job-1", True

    with (
        patch("ptq.cli._repo", return_value=repo),
        patch("ptq.infrastructure.backends.create_backend", return_value=backend),
        patch(
            "ptq.application.worktree_service.ensure_job_worktree",
            side_effect=ensure_job,
        ),
        patch(
            "ptq.application.herdr_service.open_job_workspace",
            return_value=opened,
        ) as mock_open,
    ):
        result = runner.invoke(app, ["open", "--issue", "123", "--no-focus"])

    assert result.exit_code == 0, result.output
    assert "Created PTQ job job-1" in result.output
    mock_open.assert_called_once_with(
        "job-1",
        "cd /tmp/ws/jobs/job-1 && source .venv/bin/activate",
        label="ptq #123",
        focus=False,
    )


def test_open_cli_requires_one_selector():
    result = runner.invoke(app, ["open"])

    assert result.exit_code == 1
    assert "Provide exactly one JOB_ID, --issue, or --name" in result.output
