from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from ptq.application.herdr_service import (
    cwd_from_takeover_command,
    monitor_operator_bootstrap_command,
    open_job_workspace,
    open_monitor_workspace,
)
from ptq.cli import app
from ptq.domain.models import JobRecord
from ptq.infrastructure.job_repository import JobRepository

runner = CliRunner()


def test_cwd_from_takeover_command_expands_home():
    command = "cd $HOME/.ptq_workspace/jobs/job-1 && source .venv/bin/activate"
    assert cwd_from_takeover_command(command).endswith("/.ptq_workspace/jobs/job-1")


def test_monitor_operator_bootstrap_command_points_at_repo_skill():
    command = monitor_operator_bootstrap_command()

    assert "--skill .agents/skills/monitor" in command
    assert "/monitor" in command
    assert "uv run ptq open JOB_ID" in command


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


def test_open_monitor_workspace_checks_operator_pane_bootstrap_failure():
    client = MagicMock()
    client.run.side_effect = [
        MagicMock(
            returncode=0,
            stdout=json.dumps(
                {
                    "result": {
                        "root_pane": {
                            "pane_id": "visual-pane",
                            "workspace_id": "workspace-1",
                        }
                    }
                }
            ),
            stderr="",
        ),
        MagicMock(returncode=0, stdout="", stderr=""),
        MagicMock(
            returncode=0,
            stdout=json.dumps(
                {
                    "result": {
                        "pane": {
                            "pane_id": "operator-pane",
                            "workspace_id": "workspace-1",
                        }
                    }
                }
            ),
            stderr="",
        ),
        MagicMock(returncode=1, stdout="", stderr="operator failed"),
    ]

    try:
        open_monitor_workspace(cwd="/tmp/repo", visual_command="uv run ptq monitor", client=client)
    except RuntimeError as e:
        assert "operator failed" in str(e)
    else:
        raise AssertionError("expected operator pane failure")


def test_open_monitor_workspace_reports_malformed_herdr_json():
    client = MagicMock()
    client.run.return_value = MagicMock(returncode=0, stdout="not json", stderr="")

    try:
        open_monitor_workspace(cwd="/tmp/repo", visual_command="uv run ptq monitor", client=client)
    except RuntimeError as e:
        assert "invalid JSON" in str(e)
    else:
        raise AssertionError("expected malformed JSON failure")


def test_open_cli_resolves_job_and_opens_herdr_workspace(tmp_path: Path):
    repo = JobRepository(tmp_path / "jobs.json")
    repo.save(JobRecord(job_id="job-1", issue=123, local=True, workspace="/tmp/ws"))
    opened = MagicMock(workspace_id="workspace-1", pane_id="pane-1")

    with (
        patch("ptq.cli._repo", return_value=repo),
        patch("ptq.application.herdr_service.open_job_workspace", return_value=opened) as mock_open,
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
