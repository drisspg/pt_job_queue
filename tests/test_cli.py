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

    def test_defaults_to_local_when_no_machine(self):
        with patch("ptq.agent.launch_agent") as mock_launch:
            result = runner.invoke(app, ["run", "-m", "hello"])

        assert result.exit_code == 0, result.output
        mock_launch.assert_called_once()
        assert mock_launch.call_args.kwargs["local"] is True

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

    def test_agent_type_passed_through(self):
        with patch("ptq.agent.launch_agent") as mock_launch:
            result = runner.invoke(app, ["run", "-m", "hello", "--agent", "codex"])

        assert result.exit_code == 0, result.output
        assert mock_launch.call_args.kwargs["agent_type"] == "codex"

    def test_rerun_passes_existing_job_id(self):
        db = {
            "20260217-adhoc-abc123": {
                "issue": None,
                "runs": 1,
                "local": True,
                "workspace": "/tmp/ws",
                "agent": "cursor",
            }
        }
        with (
            patch("ptq.ssh.load_jobs_db", return_value=db),
            patch("ptq.ssh.save_jobs_db"),
            patch("ptq.job.load_jobs_db", return_value=db),
            patch("ptq.job.save_jobs_db"),
            patch("ptq.agent.launch_agent") as mock_launch,
        ):
            result = runner.invoke(
                app, ["run", "20260217-adhoc-abc123", "-m", "try again"]
            )

        assert result.exit_code == 0, result.output
        assert (
            mock_launch.call_args.kwargs["existing_job_id"] == "20260217-adhoc-abc123"
        )
        assert mock_launch.call_args.kwargs["agent_type"] == "cursor"


def _make_clean_db():
    return {
        "job-stopped": {
            "issue": 100,
            "runs": 1,
            "local": True,
            "workspace": "/tmp/ws",
            "agent": "claude",
        },
        "job-running": {
            "issue": 200,
            "runs": 2,
            "local": True,
            "workspace": "/tmp/ws",
            "agent": "codex",
            "pid": 99999,
        },
    }


RUNNING_PID = 99999


def _patch_clean(db: dict):
    mock_backend = MagicMock()
    mock_backend.workspace = "/tmp/ws"
    mock_backend.is_pid_alive.side_effect = lambda pid: pid == RUNNING_PID

    return (
        patch("ptq.ssh.load_jobs_db", return_value=db),
        patch("ptq.ssh.save_jobs_db"),
        patch("ptq.job.load_jobs_db", return_value=db),
        patch("ptq.job.save_jobs_db"),
        patch("ptq.ssh.backend_for_job", return_value=mock_backend),
        patch("ptq.ssh.LocalBackend", return_value=mock_backend),
    )


class TestCleanSingleJob:
    def test_removes_job_from_db(self):
        db = _make_clean_db()
        p1, p2, p3, p4, p5, p6 = _patch_clean(db)
        with p1, p2, p3, p4, p5, p6:
            result = runner.invoke(app, ["clean", "job-stopped"])
        assert result.exit_code == 0, result.output
        assert "job-stopped" not in db
        assert "removed" in result.output

    def test_runs_worktree_remove(self):
        db = _make_clean_db()
        p1, p2, p3, p4, p5, p6 = _patch_clean(db)
        with p1, p2, p3, p4, p5 as mock_backend_for_job, p6:
            backend = mock_backend_for_job.return_value
            runner.invoke(app, ["clean", "job-stopped"])

        run_cmds = [c.args[0] for c in backend.run.call_args_list]
        assert any("create_worktree.py remove" in c for c in run_cmds)
        assert any("rm -rf" in c for c in run_cmds)
        assert any("worktree prune" in c for c in run_cmds)

    def test_kills_running_agent(self):
        db = _make_clean_db()
        p1, p2, p3, p4, p5, p6 = _patch_clean(db)
        with p1, p2, p3, p4, p5 as mock_backend_for_job, p6:
            backend = mock_backend_for_job.return_value
            backend.is_pid_alive.return_value = True
            runner.invoke(app, ["clean", "job-running"])

        backend.kill_pid.assert_called_once_with(99999)

    def test_unknown_target_treated_as_machine(self):
        db = _make_clean_db()
        p1, p2, p3, p4, p5, p6 = _patch_clean(db)
        with p1, p2, p3, p4, p5, p6:
            result = runner.invoke(app, ["clean", "nonexistent-machine"])
        assert result.exit_code == 0
        assert "Nothing to clean" in result.output


class TestCleanMachine:
    def test_bulk_clean_removes_stopped_jobs(self):
        db = _make_clean_db()
        p1, p2, p3, p4, p5, p6 = _patch_clean(db)
        with p1, p2, p3, p4, p5, p6:
            result = runner.invoke(app, ["clean", "--local"])
        assert result.exit_code == 0, result.output
        assert "job-stopped" not in db
        assert "job-running" in db

    def test_bulk_clean_with_all_includes_running(self):
        db = _make_clean_db()
        p1, p2, p3, p4, p5, p6 = _patch_clean(db)
        with p1, p2, p3, p4, p5, p6:
            result = runner.invoke(app, ["clean", "--local", "--all"])
        assert result.exit_code == 0, result.output
        assert "job-stopped" not in db
        assert "job-running" not in db

    def test_keep_preserves_recent(self):
        db = _make_clean_db()
        p1, p2, p3, p4, p5, p6 = _patch_clean(db)
        with p1, p2, p3, p4, p5, p6:
            result = runner.invoke(app, ["clean", "--local", "--all", "--keep", "1"])
        assert result.exit_code == 0, result.output
        assert len(db) == 1

    def test_nothing_to_clean(self):
        db = {}
        p1, p2, p3, p4, p5, p6 = _patch_clean(db)
        with p1, p2, p3, p4, p5, p6:
            result = runner.invoke(app, ["clean", "--local"])
        assert result.exit_code == 0
        assert "Nothing to clean" in result.output

    def test_no_target_no_local_errors(self):
        result = runner.invoke(app, ["clean"])
        assert result.exit_code != 0


class TestSetupValidation:
    def test_no_machine_no_local(self):
        result = runner.invoke(app, ["setup"])
        assert result.exit_code != 0
