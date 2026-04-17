from __future__ import annotations

import json
import shlex
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ptq.ssh import Backend


def _coerce_event_text(value: object) -> str:
    match value:
        case str():
            return value
        case None:
            return ""
        case list() as items:
            parts: list[str] = []
            for item in items:
                text = _coerce_event_text(item)
                if text:
                    parts.append(text)
            return "\n".join(parts)
        case {"text": str(text)}:
            return text
        case {"content": content}:
            return _coerce_event_text(content)
        case {"output": output}:
            return _coerce_event_text(output)
        case {"message": str(message)}:
            return message
        case dict():
            return json.dumps(value, ensure_ascii=False)
        case _:
            return str(value)


def _quote(value: str) -> str:
    return shlex.quote(value)


def _quote_path(value: str) -> str:
    if value.startswith("~"):
        return value
    return shlex.quote(value)


def _pi_tool_name(name: str) -> str:
    match name:
        case "bash":
            return "Bash"
        case "read":
            return "Read"
        case "edit":
            return "Edit"
        case "write":
            return "Write"
        case "grep":
            return "Grep"
        case "find":
            return "Find"
        case "ls":
            return "List"
        case _:
            return name


def _pi_text_from_content(content: object) -> str:
    match content:
        case list() as items:
            parts = [_pi_text_from_content(item) for item in items]
            return "".join(part for part in parts if part)
        case {"type": "text", "text": str(text)}:
            return text
        case {"content": inner}:
            return _pi_text_from_content(inner)
        case _:
            return _coerce_event_text(content)


@dataclass
class RunContext:
    worktree_path: str
    job_dir: str
    message: str
    model: str
    thinking: str | None
    max_turns: int
    system_prompt_file: str
    unbuffer_prefix: str = ""


@dataclass
class StreamEvent:
    kind: Literal["text", "tool_use", "tool_result", "error"]
    text: str = ""
    tool_name: str = ""
    tool_input: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.text = _coerce_event_text(self.text)


@runtime_checkable
class Agent(Protocol):
    name: str

    def build_cmd(self, ctx: RunContext) -> str: ...
    def parse_stream_line(self, line: str) -> list[StreamEvent]: ...
    def log_filename(self, run_number: int) -> str: ...
    def setup_workspace(
        self,
        backend: Backend,
        worktree_path: str,
        job_dir: str,
        workspace: str,
        prompt_file: str,
    ) -> None: ...
    def extract_summary(self, log_content: str) -> str | None: ...


# ---------------------------------------------------------------------------
# Claude
# ---------------------------------------------------------------------------


@dataclass
class ClaudeAgent:
    name: str = "claude"

    def build_cmd(self, ctx: RunContext) -> str:
        escaped = ctx.message.replace("'", "'\\''")
        return (
            f"cd {ctx.job_dir} && "
            f"{ctx.unbuffer_prefix}"
            f"claude -p '{escaped}' "
            f"--model {ctx.model} "
            f"{f'--effort {ctx.thinking} ' if ctx.thinking else ''}"
            f"--max-turns {ctx.max_turns} "
            f"--allowedTools 'Read,Edit,Write,Bash,Grep,Glob' "
            f"--dangerously-skip-permissions "
            f"--append-system-prompt-file {ctx.system_prompt_file} "
            f"--output-format stream-json "
            f"--verbose"
        )

    def parse_stream_line(self, line: str) -> list[StreamEvent]:
        event = json.loads(line)
        events: list[StreamEvent] = []
        match event.get("type"):
            case "assistant":
                for block in event.get("message", {}).get("content", []):
                    if block.get("type") == "text" and block.get("text"):
                        events.append(StreamEvent(kind="text", text=block["text"]))
                    elif block.get("type") == "tool_use":
                        events.append(
                            StreamEvent(
                                kind="tool_use",
                                tool_name=block.get("name", ""),
                                tool_input=block.get("input", {}),
                            )
                        )
            case "user":
                for block in event.get("message", {}).get("content", []):
                    if block.get("type") != "tool_result":
                        continue
                    content = block.get("content", "")
                    is_error = block.get("is_error", False)
                    if is_error:
                        events.append(StreamEvent(kind="error", text=content))
                    elif content:
                        result = event.get("tool_use_result", {})
                        stdout = (
                            result.get("stdout", "") if isinstance(result, dict) else ""
                        )
                        events.append(
                            StreamEvent(kind="tool_result", text=stdout or content)
                        )
        return events

    def extract_summary(self, log_content: str) -> str | None:
        last_text = None
        for line in log_content.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            if event.get("type") != "assistant":
                continue
            for block in event.get("message", {}).get("content", []):
                if block.get("type") == "text" and block.get("text"):
                    last_text = block["text"]
        return last_text

    def log_filename(self, run_number: int) -> str:
        return f"agent_logs/claude-{run_number}.log"

    def setup_workspace(
        self,
        backend: Backend,
        worktree_path: str,
        job_dir: str,
        workspace: str,
        prompt_file: str,
    ) -> None:
        backend.run(f"mkdir -p {job_dir}/.claude", check=False)
        settings = json.dumps(
            {
                "sandbox": {
                    "enabled": True,
                    "allowedDirectories": [job_dir, f"{workspace}/scripts"],
                },
                "env": {"CLAUDE_CODE_ATTRIBUTION_HEADER": "0"},
            }
        )
        backend.run(
            f"cat > {job_dir}/.claude/settings.json << 'SETTINGS_EOF'\n{settings}\nSETTINGS_EOF"
        )


# ---------------------------------------------------------------------------
# Codex
# ---------------------------------------------------------------------------


@dataclass
class CodexAgent:
    name: str = "codex"

    def build_cmd(self, ctx: RunContext) -> str:
        escaped = ctx.message.replace("'", "'\\''")
        thinking = (
            f"-c {_quote(f'reasoning_effort="{ctx.thinking}"')} "
            if ctx.thinking
            else ""
        )
        return (
            f"{ctx.unbuffer_prefix}"
            f"codex exec '{escaped}' "
            f"--model {ctx.model} "
            f"{thinking}"
            f"--dangerously-bypass-approvals-and-sandbox "
            f"-C {ctx.job_dir} "
            f"--json"
        )

    def parse_stream_line(self, line: str) -> list[StreamEvent]:
        event = json.loads(line)
        events: list[StreamEvent] = []
        match event.get("type"):
            case "item.completed":
                item = event.get("item", {})
                match item.get("type"):
                    case "agent_message":
                        events.append(
                            StreamEvent(kind="text", text=item.get("text", ""))
                        )
                    case "command_execution":
                        cmd = item.get("command", "")
                        output = item.get("aggregated_output", "")
                        exit_code = item.get("exit_code")
                        if exit_code is not None:
                            events.append(
                                StreamEvent(
                                    kind="tool_use",
                                    tool_name="Bash",
                                    tool_input={"command": cmd},
                                )
                            )
                            events.append(StreamEvent(kind="tool_result", text=output))
                    case "reasoning":
                        pass
            case "error":
                events.append(StreamEvent(kind="error", text=event.get("message", "")))
        return events

    def extract_summary(self, log_content: str) -> str | None:
        last_text = None
        for line in log_content.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            if event.get("type") != "item.completed":
                continue
            item = event.get("item", {})
            if item.get("type") == "agent_message" and item.get("text"):
                last_text = item["text"]
        return last_text

    def log_filename(self, run_number: int) -> str:
        return f"agent_logs/codex-{run_number}.log"

    def setup_workspace(
        self,
        backend: Backend,
        worktree_path: str,
        job_dir: str,
        workspace: str,
        prompt_file: str,
    ) -> None:
        backend.run(f"cp {prompt_file} {job_dir}/AGENTS.md", check=False)


# ---------------------------------------------------------------------------
# Cursor Agent
# ---------------------------------------------------------------------------

_CURSOR_TOOL_NAME_MAP = {
    "readToolCall": "Read",
    "shellToolCall": "Bash",
    "editToolCall": "Edit",
    "writeToolCall": "Write",
    "grepToolCall": "Grep",
    "globToolCall": "Glob",
    "listToolCall": "List",
}


@dataclass
class CursorAgent:
    name: str = "cursor"

    def build_cmd(self, ctx: RunContext) -> str:
        escaped = ctx.message.replace("'", "'\\''")
        return (
            f"{ctx.unbuffer_prefix}"
            f"agent -p '{escaped}' "
            f"--model {ctx.model} "
            f"--force "
            f"--workspace {ctx.job_dir} "
            f"--output-format stream-json"
        )

    def parse_stream_line(self, line: str) -> list[StreamEvent]:
        event = json.loads(line)
        events: list[StreamEvent] = []
        match event.get("type"):
            case "assistant":
                for block in event.get("message", {}).get("content", []):
                    if block.get("type") == "text" and block.get("text"):
                        events.append(StreamEvent(kind="text", text=block["text"]))
            case "tool_call":
                tool_call = event.get("tool_call", {})
                tool_key = next(iter(tool_call), "")
                tool_name = _CURSOR_TOOL_NAME_MAP.get(tool_key, tool_key)
                subtype = event.get("subtype", "")
                if subtype == "started":
                    args = tool_call.get(tool_key, {}).get("args", {})
                    events.append(
                        StreamEvent(
                            kind="tool_use", tool_name=tool_name, tool_input=args
                        )
                    )
                elif subtype == "completed":
                    inner = tool_call.get(tool_key, {})
                    result = inner.get("result", {})
                    success = result.get("success", {})
                    content = success.get("content", "") or success.get("output", "")
                    if content:
                        events.append(StreamEvent(kind="tool_result", text=content))
            case "result":
                text = event.get("result", "")
                if event.get("is_error"):
                    events.append(
                        StreamEvent(kind="error", text=text or "unknown error")
                    )
        return events

    def extract_summary(self, log_content: str) -> str | None:
        result_text = None
        last_assistant = None
        for line in log_content.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            match event.get("type"):
                case "result" if not event.get("is_error") and event.get("result"):
                    result_text = event["result"]
                case "assistant":
                    for block in event.get("message", {}).get("content", []):
                        if block.get("type") == "text" and block.get("text"):
                            last_assistant = block["text"]
        return result_text or last_assistant

    def log_filename(self, run_number: int) -> str:
        return f"agent_logs/cursor-{run_number}.log"

    def setup_workspace(
        self,
        backend: Backend,
        worktree_path: str,
        job_dir: str,
        workspace: str,
        prompt_file: str,
    ) -> None:
        backend.run(f"cp {prompt_file} {job_dir}/.cursorrules", check=False)


# ---------------------------------------------------------------------------
# Pi
# ---------------------------------------------------------------------------


@dataclass
class PiAgent:
    name: str = "pi"

    def build_cmd(self, ctx: RunContext) -> str:
        escaped = ctx.message.replace("'", "'\\''")
        return (
            f"cd {_quote_path(ctx.job_dir)} && "
            f"{ctx.unbuffer_prefix}"
            f"pi --print --mode json --no-session "
            f"--model {_quote(ctx.model)} "
            f"{f'--thinking {_quote(ctx.thinking)} ' if ctx.thinking else ''}"
            f"--append-system-prompt {_quote_path(ctx.system_prompt_file)} "
            f"--tools read,bash,edit,write "
            f"'{escaped}'"
        )

    def parse_stream_line(self, line: str) -> list[StreamEvent]:
        event = json.loads(line)
        events: list[StreamEvent] = []
        match event.get("type"):
            case "message_end":
                message = event.get("message", {})
                if message.get("role") == "assistant":
                    text = "".join(
                        block.get("text", "")
                        for block in message.get("content", [])
                        if isinstance(block, dict) and block.get("type") == "text"
                    )
                    if text:
                        events.append(StreamEvent(kind="text", text=text))
            case "tool_execution_start":
                tool_name = event.get("toolName", "")
                events.append(
                    StreamEvent(
                        kind="tool_use",
                        tool_name=_pi_tool_name(tool_name),
                        tool_input=event.get("args", {}),
                    )
                )
            case "tool_execution_end":
                text = _pi_text_from_content(event.get("result", {}))
                if event.get("isError"):
                    events.append(StreamEvent(kind="error", text=text))
                elif text:
                    events.append(StreamEvent(kind="tool_result", text=text))
        return events

    def extract_summary(self, log_content: str) -> str | None:
        last_text = None
        for line in log_content.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except (json.JSONDecodeError, ValueError):
                continue
            if event.get("type") != "message_end":
                continue
            message = event.get("message", {})
            if message.get("role") != "assistant":
                continue
            text = _pi_text_from_content(message.get("content", []))
            if text:
                last_text = text
        return last_text

    def log_filename(self, run_number: int) -> str:
        return f"agent_logs/pi-{run_number}.log"

    def setup_workspace(
        self,
        backend: Backend,
        worktree_path: str,
        job_dir: str,
        workspace: str,
        prompt_file: str,
    ) -> None:
        return None


# ---------------------------------------------------------------------------
# Registry
#
# Adding a new agent:
#
# 1. Run the CLI in JSON/streaming mode with a simple prompt to capture the
#    output format:
#        <binary> exec 'say hello' --json  (or equivalent flags)
#    Look at the JSONL lines to understand the event schema.
#
# 2. Create a @dataclass below that structurally satisfies the Agent protocol:
#
#        @dataclass
#        class MyAgent:
#            name: str = "myagent"
#
#            def build_cmd(self, ctx: RunContext) -> str:
#                # Return the full shell command. Use ctx for message, model,
#                # worktree_path, etc. Escape ctx.message for shell quoting.
#                ...
#
#            def parse_stream_line(self, line: str) -> list[StreamEvent]:
#                # Parse one JSONL line into normalized StreamEvent objects.
#                # Return [] for lines that should be ignored (init, heartbeat).
#                ...
#
#            def log_filename(self, run_number: int) -> str:
#                return f"agent_logs/myagent-{run_number}.log"
#
#            def setup_workspace(
#                self, backend: Backend, worktree_path: str,
#                job_dir: str, workspace: str, prompt_file: str,
#            ) -> None:
#                # Write agent-specific config files into the worktree.
#                # e.g. AGENTS.md for codex, .cursorrules for cursor,
#                #      .claude/settings.json for claude.
#                # prompt_file points at the uploaded prompt to stage.
#                ...
#
# 3. Add it to the AGENTS dict below.
#
# 4. Run: ptq run --agent myagent -m "hello" --local
#
# No changes needed in agent.py, cli.py, or anywhere else.
# ---------------------------------------------------------------------------

AGENTS: dict[
    str, type[ClaudeAgent] | type[CodexAgent] | type[CursorAgent] | type[PiAgent]
] = {
    "claude": ClaudeAgent,
    "codex": CodexAgent,
    "cursor": CursorAgent,
    "pi": PiAgent,
}


def get_agent(name: str) -> Agent:
    if name not in AGENTS:
        available = ", ".join(AGENTS)
        raise SystemExit(f"Unknown agent: {name}. Available: {available}")
    return AGENTS[name]()
