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
        raise RuntimeError(
            f"herdr {action} returned invalid JSON: {result.stdout.strip()}"
        ) from e
    if not isinstance(data, dict):
        raise RuntimeError(
            f"herdr {action} returned unexpected JSON: {result.stdout.strip()}"
        )
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
