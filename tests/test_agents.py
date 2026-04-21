from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from ptq.agents import (
    Agent,
    ClaudeAgent,
    CodexAgent,
    CursorAgent,
    PiAgent,
    RunContext,
    StreamEvent,
    get_agent,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def ctx() -> RunContext:
    return RunContext(
        worktree_path="/tmp/wt",
        job_dir="/tmp/job",
        message="fix the bug",
        model="opus",
        thinking=None,
        max_turns=50,
        system_prompt_file="/tmp/job/system_prompt.md",
    )


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocol:
    @pytest.mark.parametrize("cls", [ClaudeAgent, CodexAgent, CursorAgent, PiAgent])
    def test_satisfies_agent_protocol(self, cls):
        assert isinstance(cls(), Agent)

    @pytest.mark.parametrize("name", ["claude", "codex", "cursor", "pi"])
    def test_registry_returns_agent(self, name):
        agent = get_agent(name)
        assert isinstance(agent, Agent)
        assert agent.name == name

    def test_unknown_agent_raises(self):
        with pytest.raises(SystemExit, match="Unknown agent"):
            get_agent("nonexistent")


# ---------------------------------------------------------------------------
# build_cmd
# ---------------------------------------------------------------------------


class TestBuildCmd:
    def test_claude_cmd_structure(self, ctx):
        cmd = ClaudeAgent().build_cmd(ctx)
        assert cmd.startswith("cd /tmp/job && ")
        assert "claude -p " in cmd
        assert "--model opus" in cmd
        assert "--max-turns 50" in cmd
        assert "--dangerously-skip-permissions" in cmd
        assert "--append-system-prompt-file /tmp/job/system_prompt.md" in cmd
        assert "--output-format stream-json" in cmd

    def test_codex_cmd_structure(self, ctx):
        cmd = CodexAgent().build_cmd(ctx)
        assert "codex exec " in cmd
        assert "--model opus" in cmd
        assert "--dangerously-bypass-approvals-and-sandbox" in cmd
        assert "-C /tmp/job" in cmd
        assert "--json" in cmd
        assert "--max-turns" not in cmd

    def test_cursor_cmd_structure(self, ctx):
        cmd = CursorAgent().build_cmd(ctx)
        assert "agent -p " in cmd
        assert "--model opus" in cmd
        assert "--force" in cmd
        assert "--workspace /tmp/job" in cmd
        assert "--output-format stream-json" in cmd
        assert "--max-turns" not in cmd

    def test_pi_cmd_structure(self, ctx):
        ctx.thinking = "high"
        cmd = PiAgent().build_cmd(ctx)
        assert cmd.startswith("cd /tmp/job && ")
        assert "pi --print --mode json --no-session" in cmd
        assert "--model opus" in cmd
        assert "--thinking high" in cmd
        assert "--append-system-prompt /tmp/job/system_prompt.md" in cmd
        assert "--tools read,bash,edit,write" in cmd
        assert "--max-turns" not in cmd

    def test_message_quoting(self, ctx):
        ctx.message = "it's broken"
        for cls in (ClaudeAgent, CodexAgent, CursorAgent, PiAgent):
            cmd = cls().build_cmd(ctx)
            assert "it'\\''s broken" in cmd

    def test_unbuffer_prefix(self, ctx):
        ctx.unbuffer_prefix = "stdbuf -oL "
        for cls in (ClaudeAgent, CodexAgent, CursorAgent, PiAgent):
            cmd = cls().build_cmd(ctx)
            assert "stdbuf -oL " in cmd

    def test_thinking_flags(self, ctx):
        ctx.thinking = "high"
        assert "--effort high" in ClaudeAgent().build_cmd(ctx)
        assert 'reasoning_effort="high"' in CodexAgent().build_cmd(ctx)
        assert "--thinking high" in PiAgent().build_cmd(ctx)
        assert "--thinking" not in CursorAgent().build_cmd(ctx)

    def test_tilde_paths_not_single_quoted(self, ctx):
        """Single-quoted ~ prevents shell tilde expansion, breaking `cd`."""
        ctx.job_dir = "~/.ptq_workspace/jobs/job1"
        ctx.worktree_path = "~/.ptq_workspace/jobs/job1/pytorch"
        ctx.system_prompt_file = "~/.ptq_workspace/jobs/job1/system_prompt.md"
        for cls in (ClaudeAgent, CodexAgent, CursorAgent, PiAgent):
            cmd = cls().build_cmd(ctx)
            assert "'~/.ptq_workspace" not in cmd, (
                f"{cls.__name__} single-quotes a ~ path (blocks tilde expansion): {cmd}"
            )


# ---------------------------------------------------------------------------
# log_filename
# ---------------------------------------------------------------------------


class TestLogFilename:
    @pytest.mark.parametrize(
        "cls,expected",
        [
            (ClaudeAgent, "agent_logs/claude-3.log"),
            (CodexAgent, "agent_logs/codex-3.log"),
            (CursorAgent, "agent_logs/cursor-3.log"),
            (PiAgent, "agent_logs/pi-3.log"),
        ],
    )
    def test_log_filename(self, cls, expected):
        assert cls().log_filename(3) == expected


# ---------------------------------------------------------------------------
# parse_stream_line — Claude
# ---------------------------------------------------------------------------


class TestClaudeParser:
    def test_text_event(self):
        line = json.dumps(
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "hello"}]},
            }
        )
        events = ClaudeAgent().parse_stream_line(line)
        assert len(events) == 1
        assert events[0] == StreamEvent(kind="text", text="hello")

    def test_tool_use_event(self):
        line = json.dumps(
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "Bash",
                            "input": {"command": "ls"},
                        },
                    ]
                },
            }
        )
        events = ClaudeAgent().parse_stream_line(line)
        assert len(events) == 1
        assert events[0].kind == "tool_use"
        assert events[0].tool_name == "Bash"
        assert events[0].tool_input == {"command": "ls"}

    def test_tool_result_event(self):
        line = json.dumps(
            {
                "type": "user",
                "message": {
                    "content": [
                        {
                            "type": "tool_result",
                            "content": "file.py\n",
                            "is_error": False,
                        },
                    ]
                },
            }
        )
        events = ClaudeAgent().parse_stream_line(line)
        assert len(events) == 1
        assert events[0].kind == "tool_result"

    def test_error_event(self):
        line = json.dumps(
            {
                "type": "user",
                "message": {
                    "content": [
                        {
                            "type": "tool_result",
                            "content": "command failed",
                            "is_error": True,
                        },
                    ]
                },
            }
        )
        events = ClaudeAgent().parse_stream_line(line)
        assert len(events) == 1
        assert events[0] == StreamEvent(kind="error", text="command failed")

    def test_mixed_content_blocks(self):
        line = json.dumps(
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {"type": "text", "text": "running..."},
                        {
                            "type": "tool_use",
                            "name": "Read",
                            "input": {"file_path": "a.py"},
                        },
                    ]
                },
            }
        )
        events = ClaudeAgent().parse_stream_line(line)
        assert len(events) == 2
        assert events[0].kind == "text"
        assert events[1].kind == "tool_use"

    def test_unknown_type_returns_empty(self):
        line = json.dumps({"type": "system", "data": "init"})
        assert ClaudeAgent().parse_stream_line(line) == []


# ---------------------------------------------------------------------------
# parse_stream_line — Codex
# ---------------------------------------------------------------------------


class TestCodexParser:
    def test_agent_message(self):
        line = json.dumps(
            {
                "type": "item.completed",
                "item": {"type": "agent_message", "text": "done"},
            }
        )
        events = CodexAgent().parse_stream_line(line)
        assert events == [StreamEvent(kind="text", text="done")]

    def test_command_execution(self):
        line = json.dumps(
            {
                "type": "item.completed",
                "item": {
                    "type": "command_execution",
                    "command": "/usr/bin/zsh -lc 'ls'",
                    "aggregated_output": "file.py\n",
                    "exit_code": 0,
                    "status": "completed",
                },
            }
        )
        events = CodexAgent().parse_stream_line(line)
        assert len(events) == 2
        assert events[0].kind == "tool_use"
        assert events[0].tool_name == "Bash"
        assert events[1].kind == "tool_result"
        assert events[1].text == "file.py\n"

    def test_command_in_progress_ignored(self):
        line = json.dumps(
            {
                "type": "item.started",
                "item": {
                    "type": "command_execution",
                    "command": "ls",
                    "aggregated_output": "",
                    "exit_code": None,
                    "status": "in_progress",
                },
            }
        )
        events = CodexAgent().parse_stream_line(line)
        assert events == []

    def test_reasoning_skipped(self):
        line = json.dumps(
            {
                "type": "item.completed",
                "item": {"type": "reasoning", "text": "thinking..."},
            }
        )
        assert CodexAgent().parse_stream_line(line) == []

    def test_error_event(self):
        line = json.dumps({"type": "error", "message": "model error"})
        events = CodexAgent().parse_stream_line(line)
        assert events == [StreamEvent(kind="error", text="model error")]

    def test_turn_completed_ignored(self):
        line = json.dumps(
            {
                "type": "turn.completed",
                "usage": {"input_tokens": 100, "output_tokens": 50},
            }
        )
        assert CodexAgent().parse_stream_line(line) == []


# ---------------------------------------------------------------------------
# parse_stream_line — Cursor Agent
# ---------------------------------------------------------------------------


class TestCursorParser:
    def test_assistant_text(self):
        line = json.dumps(
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "hi"}]},
            }
        )
        events = CursorAgent().parse_stream_line(line)
        assert events == [StreamEvent(kind="text", text="hi")]

    def test_tool_call_started(self):
        line = json.dumps(
            {
                "type": "tool_call",
                "subtype": "started",
                "tool_call": {"readToolCall": {"args": {"path": "/etc/hostname"}}},
            }
        )
        events = CursorAgent().parse_stream_line(line)
        assert len(events) == 1
        assert events[0].kind == "tool_use"
        assert events[0].tool_name == "Read"
        assert events[0].tool_input == {"path": "/etc/hostname"}

    def test_tool_call_completed(self):
        line = json.dumps(
            {
                "type": "tool_call",
                "subtype": "completed",
                "tool_call": {
                    "readToolCall": {
                        "args": {"path": "/etc/hostname"},
                        "result": {"success": {"content": "spark-670f\n"}},
                    }
                },
            }
        )
        events = CursorAgent().parse_stream_line(line)
        assert len(events) == 1
        assert events[0] == StreamEvent(kind="tool_result", text="spark-670f\n")

    def test_tool_call_completed_with_content_blocks(self):
        line = json.dumps(
            {
                "type": "tool_call",
                "subtype": "completed",
                "tool_call": {
                    "readToolCall": {
                        "args": {"path": "/etc/hostname"},
                        "result": {
                            "success": {
                                "content": [{"type": "text", "text": "spark-670f\n"}]
                            }
                        },
                    }
                },
            }
        )
        events = CursorAgent().parse_stream_line(line)
        assert len(events) == 1
        assert events[0] == StreamEvent(kind="tool_result", text="spark-670f\n")

    def test_shell_tool_mapped(self):
        line = json.dumps(
            {
                "type": "tool_call",
                "subtype": "started",
                "tool_call": {"shellToolCall": {"args": {"command": "ls"}}},
            }
        )
        events = CursorAgent().parse_stream_line(line)
        assert events[0].tool_name == "Bash"

    def test_unknown_tool_uses_raw_key(self):
        line = json.dumps(
            {
                "type": "tool_call",
                "subtype": "started",
                "tool_call": {"newFancyTool": {"args": {"x": 1}}},
            }
        )
        events = CursorAgent().parse_stream_line(line)
        assert events[0].tool_name == "newFancyTool"

    def test_system_init_ignored(self):
        line = json.dumps({"type": "system", "subtype": "init"})
        assert CursorAgent().parse_stream_line(line) == []

    def test_result_success_ignored(self):
        line = json.dumps(
            {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "result": "done",
            }
        )
        assert CursorAgent().parse_stream_line(line) == []

    def test_result_error(self):
        line = json.dumps(
            {
                "type": "result",
                "subtype": "error",
                "is_error": True,
                "result": "something broke",
            }
        )
        events = CursorAgent().parse_stream_line(line)
        assert events == [StreamEvent(kind="error", text="something broke")]


# ---------------------------------------------------------------------------
# parse_stream_line — Pi Agent
# ---------------------------------------------------------------------------


class TestPiParser:
    def test_text_delta_is_ignored(self):
        line = json.dumps(
            {
                "type": "message_update",
                "assistantMessageEvent": {"type": "text_delta", "delta": "hello"},
            }
        )
        assert PiAgent().parse_stream_line(line) == []

    def test_message_end_emits_full_assistant_text(self):
        line = json.dumps(
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Once upon a time, "},
                        {"type": "text", "text": "there was a process."},
                    ],
                },
            }
        )
        events = PiAgent().parse_stream_line(line)
        assert events == [
            StreamEvent(kind="text", text="Once upon a time, there was a process.")
        ]

    def test_message_end_non_assistant_ignored(self):
        line = json.dumps(
            {
                "type": "message_end",
                "message": {
                    "role": "user",
                    "content": [{"type": "text", "text": "hi"}],
                },
            }
        )
        assert PiAgent().parse_stream_line(line) == []

    def test_tool_execution_start(self):
        line = json.dumps(
            {
                "type": "tool_execution_start",
                "toolName": "bash",
                "args": {"command": "ls"},
            }
        )
        events = PiAgent().parse_stream_line(line)
        assert events == [
            StreamEvent(kind="tool_use", tool_name="Bash", tool_input={"command": "ls"})
        ]

    def test_tool_execution_update_emits_delta(self):
        agent = PiAgent()
        start = json.dumps(
            {
                "type": "tool_execution_start",
                "toolCallId": "call-1",
                "toolName": "bash",
                "args": {"command": "echo hi"},
            }
        )
        update1 = json.dumps(
            {
                "type": "tool_execution_update",
                "toolCallId": "call-1",
                "partialResult": {"content": [{"type": "text", "text": "hi\n"}]},
            }
        )
        update2 = json.dumps(
            {
                "type": "tool_execution_update",
                "toolCallId": "call-1",
                "partialResult": {"content": [{"type": "text", "text": "hi\nthere\n"}]},
            }
        )
        agent.parse_stream_line(start)
        assert agent.parse_stream_line(update1) == [
            StreamEvent(kind="tool_result", text="hi\n")
        ]
        assert agent.parse_stream_line(update2) == [
            StreamEvent(kind="tool_result", text="there\n")
        ]

    def test_tool_execution_end(self):
        line = json.dumps(
            {
                "type": "tool_execution_end",
                "toolName": "read",
                "result": {"content": [{"type": "text", "text": "hello\n"}]},
                "isError": False,
            }
        )
        events = PiAgent().parse_stream_line(line)
        assert events == [StreamEvent(kind="tool_result", text="hello\n")]

    def test_tool_execution_end_after_update_emits_remaining_delta(self):
        agent = PiAgent()
        start = json.dumps(
            {
                "type": "tool_execution_start",
                "toolCallId": "call-2",
                "toolName": "bash",
                "args": {"command": "echo hi"},
            }
        )
        update = json.dumps(
            {
                "type": "tool_execution_update",
                "toolCallId": "call-2",
                "partialResult": {"content": [{"type": "text", "text": "hi\n"}]},
            }
        )
        end = json.dumps(
            {
                "type": "tool_execution_end",
                "toolCallId": "call-2",
                "toolName": "bash",
                "result": {"content": [{"type": "text", "text": "hi\nthere\n"}]},
                "isError": False,
            }
        )
        agent.parse_stream_line(start)
        agent.parse_stream_line(update)
        assert agent.parse_stream_line(end) == [
            StreamEvent(kind="tool_result", text="there\n")
        ]

    def test_tool_execution_error(self):
        line = json.dumps(
            {
                "type": "tool_execution_end",
                "toolName": "bash",
                "result": {"content": [{"type": "text", "text": "boom"}]},
                "isError": True,
            }
        )
        events = PiAgent().parse_stream_line(line)
        assert events == [StreamEvent(kind="error", text="boom")]

    def test_non_stream_event_ignored(self):
        line = json.dumps({"type": "agent_start"})
        assert PiAgent().parse_stream_line(line) == []


# ---------------------------------------------------------------------------
# Live CLI integration tests — skipped if the binary is not installed
# ---------------------------------------------------------------------------


def _has_binary(name: str) -> bool:
    return shutil.which(name) is not None


@pytest.mark.skipif(not _has_binary("codex"), reason="codex CLI not installed")
class TestCodexLive:
    def test_simple_prompt(self):
        result = subprocess.run(
            [
                "codex",
                "exec",
                "respond with exactly: PONG",
                "--json",
                "--dangerously-bypass-approvals-and-sandbox",
                "-C",
                "/tmp",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0

        agent = CodexAgent()
        found_text = False
        for line in result.stdout.strip().splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            for ev in agent.parse_stream_line(line):
                if ev.kind == "text":
                    found_text = True
        assert found_text, "Expected at least one text event from codex"


@pytest.mark.skipif(not _has_binary("agent"), reason="cursor agent CLI not installed")
class TestCursorLive:
    def test_simple_prompt(self):
        result = subprocess.run(
            [
                "agent",
                "-p",
                "respond with exactly: PONG",
                "--force",
                "--workspace",
                "/tmp",
                "--output-format",
                "stream-json",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0

        agent = CursorAgent()
        found_text = False
        for line in result.stdout.strip().splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            for ev in agent.parse_stream_line(line):
                if ev.kind == "text":
                    found_text = True
        assert found_text, "Expected at least one text event from cursor agent"


@pytest.mark.skipif(not _has_binary("pi"), reason="pi CLI not installed")
class TestPiLive:
    def test_simple_prompt(self):
        result = subprocess.run(
            [
                "pi",
                "--no-session",
                "--mode",
                "json",
                "--print",
                "--model",
                "codex",
                "respond with exactly: PONG",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0 and "No API key found" in result.stderr:
            pytest.skip("pi is installed but not authenticated for live model tests")
        assert result.returncode == 0

        agent = PiAgent()
        found_text = False
        for line in result.stdout.strip().splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            for ev in agent.parse_stream_line(line):
                if ev.kind == "text":
                    found_text = True
        assert found_text, "Expected at least one text event from pi"
