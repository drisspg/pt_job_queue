from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from ptq.application.herdr_service import (
    cwd_from_takeover_command,
    in_herdr_app,
    open_job_space,
    open_job_workspace,
)
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
        MagicMock(returncode=0, stdout="", stderr=""),
    ]
    command = "cd /tmp/ws/jobs/job-1 && source .venv/bin/activate"

    workspace = open_job_workspace(
        "job-1", command, pane_label="ptq #1 · fix", client=client, focus=False
    )

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
    assert client.run.call_args_list[1].args[0] == [
        "pane",
        "rename",
        "pane-1",
        "ptq #1 · fix",
    ]
    assert client.run.call_args_list[2].args[0] == ["pane", "run", "pane-1", command]


def test_open_cli_resolves_job_and_opens_herdr_workspace(tmp_path: Path):
    repo = JobRepository(tmp_path / "jobs.json")
    repo.save(JobRecord(job_id="job-1", issue=123, workspace="/tmp/ws"))
    opened = MagicMock(workspace_id="workspace-1", pane_id="pane-1")
    backend = MagicMock()
    backend.run.return_value = MagicMock(
        returncode=0,
        stdout="flex_attention returns incorrect output under Inductor\n",
    )

    with (
        patch("ptq.cli._repo", return_value=repo),
        patch("ptq.application.worktree_service.prepare_job_worktree"),
        patch("ptq.infrastructure.backends.backend_for_job", return_value=backend),
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
        pane_label="ptq #123 · flex_attention returns incorrect output…",
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
                name="flex-small-seq",
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
        pane_label="ptq #123 · flex-small-seq",
        focus=False,
    )


def test_open_cli_no_herdr_prints_entry_command_without_herdr(tmp_path: Path):
    repo = JobRepository(tmp_path / "jobs.json")
    backend = MagicMock(workspace="/tmp/ws")

    def ensure_job(job_repo, _backend, **kwargs):
        job_repo.save(JobRecord(job_id="job-1", name="box", workspace="/tmp/ws"))
        return "job-1", True

    with (
        patch("ptq.cli._repo", return_value=repo),
        patch("ptq.infrastructure.backends.create_backend", return_value=backend),
        patch(
            "ptq.application.worktree_service.ensure_job_worktree",
            side_effect=ensure_job,
        ),
        patch("ptq.application.herdr_service.open_job_workspace") as mock_open,
    ):
        result = runner.invoke(app, ["open", "--name", "box", "--no-herdr"])

    assert result.exit_code == 0, result.output
    assert result.output.splitlines()[-1] == (
        "cd /tmp/ws/jobs/job-1 && source .venv/bin/activate"
    )
    mock_open.assert_not_called()


def test_open_cli_requires_one_selector():
    result = runner.invoke(app, ["open"])

    assert result.exit_code == 1
    assert "Provide exactly one JOB_ID, --issue, or --name" in result.output


class FakeHerdrCore:
    """Minimal herdr-core: terminals with labels plus the app layout document."""

    def __init__(self, terminals=None, spaces=None, *, app_running=True, conflicts=0):
        self.terminals = terminals or {}
        self.spaces = spaces or []
        self.app_running = app_running
        self.conflicts = conflicts
        self.version = 1
        self.commands = []

    def adopt(self, address):
        space_id = self.terminals[address]["labels"]["herdr.space"]
        space = next((s for s in self.spaces if s["id"] == space_id), None)
        if space is None:
            space = {"id": space_id, "name": "", "folder": "", "tabs": []}
            self.spaces.append(space)
        root = {"type": "row", "children": [{"content": {"terminal": address}}]}
        space["tabs"].append({"id": f"tb-{address}", "name": "", "root": root})

    def __call__(self, args, stdin=None):
        self.commands.append(args)
        match args:
            case ["hc", "ls", "--json", "--label", selector]:
                key, value = selector.split("=", 1)
                out = {
                    "terminals": [
                        {"address": a, "state": "live"}
                        for a, t in self.terminals.items()
                        if t["labels"].get(key) == value
                    ]
                }
            case ["hc", "get", address, "--json"]:
                out = {"address": address, **self.terminals[address]}
            case ["hc", "run", *rest]:
                address = f"t{len(self.terminals) + 1}"
                labels = dict(
                    rest[i + 1].split("=", 1)
                    for i, arg in enumerate(rest)
                    if arg == "--label"
                )
                self.terminals[address] = {"labels": labels}
                if self.app_running:
                    self.adopt(address)
                out = {"address": address, "state": "live"}
            case ["hc", "send", *_]:
                out = {}
            case ["hc", "connect"]:
                request = json.loads(stdin)
                if request["method"] == "doc.get":
                    value = {"spaces": json.loads(json.dumps(self.spaces))}
                    out = {"result": {"value": value, "version": self.version}}
                elif self.conflicts:
                    self.conflicts -= 1
                    self.version += 1
                    out = {"error": {"message": "stale", "data": {"kind": "conflict"}}}
                else:
                    self.spaces = request["params"]["value"]["spaces"]
                    self.version += 1
                    out = {"result": {"version": self.version}}
            case ["herdr-app", *_]:
                raise FileNotFoundError("herdr-app")
            case _:
                raise AssertionError(f"unexpected command {args}")
        return subprocess.CompletedProcess(args, 0, json.dumps(out), "")


def test_in_herdr_app_distinguishes_desktop_app_from_tui():
    assert in_herdr_app({"HC_TERMINAL_ID": "t2"})
    assert not in_herdr_app({"HC_TERMINAL_ID": "t2", "HERDR_ENV": "1"})
    assert not in_herdr_app({})


def test_open_job_space_creates_named_app_space_running_takeover_command():
    core = FakeHerdrCore()
    command = "cd /tmp/ws/jobs/job-1 && source .venv/bin/activate"

    workspace = open_job_space(
        "job-1", command, label="ptq #123", tab_label="ptq #123 · bug", run=core
    )

    assert workspace.pane_id == "t1"
    assert core.terminals["t1"]["labels"] == {
        "herdr.space": workspace.workspace_id,
        "ptq.job": "job-1",
    }
    run_args = next(args for args in core.commands if args[:2] == ["hc", "run"])
    assert run_args[run_args.index("--cwd") + 1] == "/tmp/ws/jobs/job-1"
    assert ["hc", "send", "--json", "t1", command] in core.commands
    assert [(s["id"], s["name"]) for s in core.spaces] == [
        (workspace.workspace_id, "ptq #123")
    ]
    assert core.spaces[0]["tabs"][0]["name"] == "ptq #123 · bug"


def test_open_job_space_reuses_live_job_terminal():
    core = FakeHerdrCore(
        terminals={
            "t1": {"labels": {"herdr.space": "sp-other"}},
            "t2": {"labels": {"herdr.space": "sp-job", "ptq.job": "job-1"}},
        }
    )
    core.adopt("t1")
    core.adopt("t2")

    workspace = open_job_space("job-1", "cd /tmp/ws/jobs/job-1", run=core)

    assert (workspace.workspace_id, workspace.pane_id) == ("sp-job", "t2")
    assert not any(
        args[:2] in (["hc", "run"], ["hc", "send"]) for args in core.commands
    )


def test_open_job_space_retries_layout_conflicts_when_naming():
    core = FakeHerdrCore(conflicts=2)

    workspace = open_job_space("job-1", "cd /tmp/ws", label="ptq job-1", run=core)

    assert [(s["id"], s["name"]) for s in core.spaces] == [
        (workspace.workspace_id, "ptq job-1")
    ]


def test_open_job_space_fails_when_app_does_not_adopt_terminal(monkeypatch):
    monkeypatch.setattr("ptq.application.herdr_service.time.sleep", lambda _: None)
    core = FakeHerdrCore(app_running=False)

    with pytest.raises(RuntimeError, match="did not show terminal t1"):
        open_job_space("job-1", "cd /tmp/ws", run=core)


def test_open_job_workspace_routes_app_terminals_to_hc(monkeypatch):
    monkeypatch.setenv("HC_TERMINAL_ID", "t2")
    monkeypatch.delenv("HERDR_ENV", raising=False)
    opened = MagicMock()

    with patch(
        "ptq.application.herdr_service.open_job_space", return_value=opened
    ) as mock_space:
        assert open_job_workspace("job-1", "cd /tmp/ws", focus=False) is opened

    mock_space.assert_called_once_with(
        "job-1", "cd /tmp/ws", label=None, tab_label=None, focus=False
    )
