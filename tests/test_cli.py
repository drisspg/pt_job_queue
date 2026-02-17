from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from ptq.cli import app

runner = CliRunner()


class TestRunValidation:
    def test_no_issue_no_message_no_job_id(self):
        result = runner.invoke(app, ["run", "--local"])
        assert result.exit_code != 0
        assert "Provide --issue, --message, or a JOB_ID" in result.output

    def test_no_machine_no_local(self):
        result = runner.invoke(app, ["run", "--issue", "123"])
        assert result.exit_code != 0
        assert "Provide --machine or --local" in result.output

    def test_input_and_message_mutually_exclusive(self):
        result = runner.invoke(app, ["run", "-i", "f.md", "-m", "hello", "--local"])
        assert result.exit_code != 0
        assert "mutually exclusive" in result.output

    def test_input_file_not_found(self):
        result = runner.invoke(app, ["run", "-i", "/nonexistent/path.md", "--local"])
        assert result.exit_code != 0
        assert "File not found" in result.output

    def test_input_file_reads_contents(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("do the thing")
            f.flush()
            tmp_path = f.name

        with (
            patch("ptq.agent.launch_agent") as mock_launch,
            patch("ptq.ssh.LocalBackend") as mock_backend_cls,
        ):
            mock_backend_cls.return_value = MagicMock()
            result = runner.invoke(app, ["run", "-i", tmp_path, "--local"])

        Path(tmp_path).unlink()
        assert result.exit_code == 0, result.output
        mock_launch.assert_called_once()
        assert mock_launch.call_args.kwargs["message"] == "do the thing"


class TestSetupValidation:
    def test_no_machine_no_local(self):
        result = runner.invoke(app, ["setup"])
        assert result.exit_code != 0
