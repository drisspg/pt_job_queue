from __future__ import annotations

import json
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class HerdrWorkspace:
    workspace_id: str
    pane_id: str
    cwd: str
    takeover_command: str


@dataclass
class HerdrMonitorWorkspace:
    workspace_id: str
    visual_pane_id: str
    operator_pane_id: str
    cwd: str


def cwd_from_takeover_command(command: str) -> str:
    """Extract the local cwd from PTQ takeover commands when one is present."""
    parts = shlex.split(command)
    if len(parts) >= 2 and parts[0] == "cd":
        return os.path.expandvars(os.path.expanduser(parts[1]))
    return str(Path.cwd())


class HerdrClient:
    def run(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        """Run the Herdr CLI and capture JSON output for PTQ orchestration."""
        return subprocess.run(
            ["herdr", *args],
            check=False,
            capture_output=True,
            text=True,
        )


def monitor_operator_bootstrap_command() -> str:
    """Print the repo-local monitor skill entrypoint in the operator pane."""
    prompt = "Start by running uv run ptq monitor and summarize current state."
    pi_command = (
        "pi --model openai-codex/gpt-5.5 --thinking high "
        "--skill .agents/skills/monitor "
        f"{shlex.quote(prompt)}"
    )
    lines = [
        "PTQ monitor operator pane",
        "Start the monitor operator Pi with:",
        pi_command,
        "",
        "In interactive Pi, use /monitor to expand the repo prompt template.",
        "The monitor skill lives in .agents/skills/monitor/SKILL.md.",
        "Use uv run ptq open JOB_ID to open an interactive job workspace.",
    ]
    return "printf '%s\\n' " + " ".join(shlex.quote(line) for line in lines)


def herdr_error(action: str, result: subprocess.CompletedProcess[str]) -> RuntimeError:
    detail = result.stderr.strip() or result.stdout.strip() or "no output"
    return RuntimeError(f"herdr {action} failed: {detail}")


def herdr_json(action: str, result: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    """Parse Herdr JSON responses into RuntimeError-based CLI messages."""
    if result.returncode != 0:
        raise herdr_error(action, result)
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"herdr {action} returned invalid JSON: {result.stdout.strip()}") from e
    if not isinstance(data, dict):
        raise RuntimeError(f"herdr {action} returned unexpected JSON: {result.stdout.strip()}")
    return data


def pane_info(action: str, data: dict[str, Any], key: str) -> dict[str, Any]:
    """Extract required Herdr pane objects from known CLI response shapes."""
    try:
        pane = data["result"][key]
    except KeyError as e:
        raise RuntimeError(f"herdr {action} response missing result.{key}") from e
    if not isinstance(pane, dict):
        raise RuntimeError(f"herdr {action} response result.{key} was not an object")
    for field in ("pane_id", "workspace_id"):
        if not isinstance(pane.get(field), str):
            raise RuntimeError(f"herdr {action} response missing result.{key}.{field}")
    return pane


def open_job_workspace(
    job_id: str,
    takeover_command: str,
    *,
    label: str | None = None,
    focus: bool = True,
    client: HerdrClient | None = None,
) -> HerdrWorkspace:
    """Create a Herdr workspace for interactive PTQ job work."""
    herdr = client or HerdrClient()
    cwd = cwd_from_takeover_command(takeover_command)
    data = herdr_json(
        "workspace create",
        herdr.run(
            [
                "workspace",
                "create",
                "--cwd",
                cwd,
                "--label",
                label or f"ptq {job_id}",
                "--focus" if focus else "--no-focus",
            ]
        ),
    )
    root_pane = pane_info("workspace create", data, "root_pane")
    pane_id = root_pane["pane_id"]
    workspace_id = root_pane["workspace_id"]
    run_result = herdr.run(["pane", "run", pane_id, takeover_command])
    if run_result.returncode != 0:
        raise herdr_error("pane run", run_result)
    return HerdrWorkspace(
        workspace_id=workspace_id,
        pane_id=pane_id,
        cwd=cwd,
        takeover_command=takeover_command,
    )


def open_monitor_workspace(
    *,
    cwd: str,
    visual_command: str,
    label: str = "ptq monitor",
    client: HerdrClient | None = None,
) -> HerdrMonitorWorkspace:
    """Create a two-pane Herdr workspace for visual monitoring and operator work."""
    herdr = client or HerdrClient()
    data = herdr_json(
        "workspace create",
        herdr.run(["workspace", "create", "--cwd", cwd, "--label", label, "--focus"]),
    )
    root_pane = pane_info("workspace create", data, "root_pane")
    visual_pane_id = root_pane["pane_id"]
    workspace_id = root_pane["workspace_id"]

    run_result = herdr.run(["pane", "run", visual_pane_id, visual_command])
    if run_result.returncode != 0:
        raise herdr_error("pane run", run_result)

    split_data = herdr_json(
        "pane split",
        herdr.run(
            [
                "pane",
                "split",
                visual_pane_id,
                "--direction",
                "right",
                "--cwd",
                cwd,
                "--no-focus",
            ]
        ),
    )
    operator_pane_id = pane_info("pane split", split_data, "pane")["pane_id"]
    operator_result = herdr.run(
        ["pane", "run", operator_pane_id, monitor_operator_bootstrap_command()]
    )
    if operator_result.returncode != 0:
        raise herdr_error("operator pane run", operator_result)
    return HerdrMonitorWorkspace(
        workspace_id=workspace_id,
        visual_pane_id=visual_pane_id,
        operator_pane_id=operator_pane_id,
        cwd=cwd,
    )
