from __future__ import annotations

import json
import os
import secrets
import shlex
import subprocess
import time
from collections.abc import Callable, Mapping
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


def in_herdr_app(env: Mapping[str, str] = os.environ) -> bool:
    """Whether PTQ runs inside a Herdr desktop app terminal rather than the Herdr TUI."""
    return env.get("HERDR_ENV") != "1" and bool(env.get("HC_TERMINAL_ID"))


def open_job_workspace(
    job_id: str,
    takeover_command: str,
    *,
    label: str | None = None,
    pane_label: str | None = None,
    focus: bool = True,
    client: HerdrClient | None = None,
) -> HerdrWorkspace:
    """Open a Herdr workspace (TUI) or space (desktop app) for interactive PTQ job work."""
    if client is None and in_herdr_app():
        return open_job_space(
            job_id, takeover_command, label=label, tab_label=pane_label, focus=focus
        )
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
    if pane_label:
        rename_result = herdr.run(["pane", "rename", pane_id, pane_label])
        if rename_result.returncode != 0:
            raise herdr_error("pane rename", rename_result)
    run_result = herdr.run(["pane", "run", pane_id, takeover_command])
    if run_result.returncode != 0:
        raise herdr_error("pane run", run_result)
    return HerdrWorkspace(
        workspace_id=workspace_id,
        pane_id=pane_id,
        cwd=cwd,
        takeover_command=takeover_command,
    )


# Herdr desktop app support. Its terminals live in herdr-core (`hc`) and its window
# layout lives in the core's `herdr-app` document, so this works on remote machines
# that have `hc` but no `herdr-app` CLI. The app adopts a terminal labelled
# `herdr.space=<id>` (without `herdr.tab`) as a new tab, creating the space if needed.

HERDR_APP_DOC = "herdr-app"
PTQ_JOB_LABEL = "ptq.job"

Runner = Callable[[list[str], str | None], subprocess.CompletedProcess[str]]


class HcError(RuntimeError):
    def __init__(self, message: str, kind: str | None = None):
        super().__init__(message)
        self.kind = kind


def run_command(
    args: list[str], stdin: str | None = None
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            args, input=stdin, check=False, capture_output=True, text=True, timeout=15
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"{args[0]} {args[1]} timed out after 15s") from e


def hc_json(action: str, result: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    """Parse `hc --json` output; hc reports failures as JSON on stdout."""
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        data = None
    if result.returncode != 0 or not isinstance(data, dict) or "error" in data:
        detail = result.stderr.strip() or result.stdout.strip() or "no output"
        raise RuntimeError(f"hc {action} failed: {detail}")
    return data


def hc_call(method: str, params: dict[str, Any], run: Runner) -> dict[str, Any]:
    """Make one JSON-RPC call against this machine's herdr-core."""
    request = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    result = run(["hc", "connect"], json.dumps(request) + "\n")
    lines = result.stdout.splitlines()
    try:
        response = json.loads(lines[0])
    except (IndexError, json.JSONDecodeError) as e:
        raise RuntimeError(f"hc {method} returned no response") from e
    if "error" in response:
        error = response["error"]
        raise HcError(
            f"hc {method} failed: {error.get('message', error)}",
            kind=(error.get("data") or {}).get("kind"),
        )
    return response["result"]


def find_job_terminal(job_id: str, run: Runner) -> dict[str, Any] | None:
    """Return a live, app-visible terminal previously opened for this job."""
    listed = hc_json(
        "ls", run(["hc", "ls", "--json", "--label", f"{PTQ_JOB_LABEL}={job_id}"], None)
    )
    for entry in listed["terminals"]:
        if entry.get("state") != "live":
            continue
        terminal = hc_json("get", run(["hc", "get", entry["address"], "--json"], None))
        if (terminal.get("labels") or {}).get("herdr.space"):
            return terminal
    return None


def layout_holds_terminal(node: dict[str, Any], terminal: str) -> bool:
    if (node.get("content") or {}).get("terminal") == terminal:
        return True
    return any(
        layout_holds_terminal(child, terminal) for child in node.get("children", [])
    )


def name_app_space(
    space_id: str,
    name: str,
    run: Runner,
    *,
    terminal: str | None = None,
    tab_name: str | None = None,
    attempts: int = 25,
    delay: float = 0.2,
) -> bool:
    """Name a space, and the tab holding `terminal`, once the app has adopted them."""
    for _ in range(attempts):
        doc = hc_call("doc.get", {"name": HERDR_APP_DOC}, run)
        value = doc.get("value") or {}
        space = next(
            (s for s in value.get("spaces", []) if s.get("id") == space_id), None
        )
        if space is None:
            time.sleep(delay)
            continue
        space["name"] = name
        if tab_name and terminal:
            for tab in space.get("tabs", []):
                if layout_holds_terminal(tab.get("root") or {}, terminal):
                    tab["name"] = tab_name
        try:
            hc_call(
                "doc.set",
                {"name": HERDR_APP_DOC, "value": value, "version": doc["version"]},
                run,
            )
            return True
        except HcError as e:
            if e.kind != "conflict":
                raise
    return False


def focus_app_space(space_id: str, terminal: str, run: Runner) -> None:
    """Focus the space when the herdr-app CLI is on this machine; remote hosts lack it."""
    try:
        run(["herdr-app", "do", "space.select", "--id", space_id], None)
        run(["herdr-app", "do", "pane.focus", "--terminal", terminal], None)
    except FileNotFoundError:
        pass


def open_job_space(
    job_id: str,
    takeover_command: str,
    *,
    label: str | None = None,
    tab_label: str | None = None,
    focus: bool = True,
    run: Runner = run_command,
) -> HerdrWorkspace:
    """Open, or reuse, a Herdr desktop app space with a shell in the job worktree."""
    cwd = cwd_from_takeover_command(takeover_command)
    existing = find_job_terminal(job_id, run)
    if existing is not None:
        space_id = existing["labels"]["herdr.space"]
        terminal = existing["address"]
    else:
        space_id = f"sp{secrets.token_hex(16)}"
        created = hc_json(
            "run",
            run(
                [
                    "hc",
                    "run",
                    "--bg",
                    "--json",
                    "--no-parent",
                    "--keep",
                    "--cwd",
                    cwd,
                    "--label",
                    f"herdr.space={space_id}",
                    "--label",
                    f"{PTQ_JOB_LABEL}={job_id}",
                ],
                None,
            ),
        )
        terminal = created["address"]
        hc_json("send", run(["hc", "send", "--json", terminal, takeover_command], None))
    named = name_app_space(
        space_id,
        label or f"ptq {job_id}",
        run,
        terminal=terminal,
        tab_name=tab_label,
    )
    if not named:
        raise RuntimeError(
            f"Herdr app did not show terminal {terminal} (space {space_id}); "
            "is the app connected to this machine? Inspect it with "
            f"`hc read {terminal}` or remove it with `hc kill {terminal}`."
        )
    if focus:
        focus_app_space(space_id, terminal, run)
    return HerdrWorkspace(
        workspace_id=space_id,
        pane_id=terminal,
        cwd=cwd,
        takeover_command=takeover_command,
    )
