from __future__ import annotations

from subprocess import CompletedProcess
from unittest.mock import MagicMock, patch

from ptq.agent import (
    _clean,
    _indent,
    _sanitize_for_api,
    _truncate,
    build_adhoc_prompt,
    build_system_prompt,
    launch_agent,
)
from ptq.ssh import LocalBackend, RemoteBackend


class TestSanitizeForApi:
    def test_redacts_anthropic_headers(self):
        assert "[redacted-header]" in _sanitize_for_api("key: x-anthropic-api-key")

    def test_leaves_normal_text(self):
        assert _sanitize_for_api("hello world") == "hello world"


class TestClean:
    def test_strips_ansi(self):
        assert _clean("\x1b[31mred\x1b[0m") == "red"

    def test_strips_carriage_return(self):
        assert _clean("line\r") == "line"

    def test_passthrough_plain(self):
        assert _clean("plain text") == "plain text"


class TestTruncate:
    def test_short_text_unchanged(self):
        assert _truncate("a\nb\nc") == "a\nb\nc"

    def test_long_text_truncated(self):
        text = "\n".join(f"line {i}" for i in range(100))
        result = _truncate(text, max_lines=5)
        assert result.count("\n") == 5
        assert "(95 more lines)" in result


class TestIndent:
    def test_default_prefix(self):
        assert _indent("a\nb") == "    a\n    b"

    def test_custom_prefix(self):
        assert _indent("a\nb", prefix=">> ") == ">> a\n>> b"


class TestBuildAdhocPrompt:
    def test_substitutes_placeholders(self):
        result = build_adhoc_prompt("fix oom", "j-123", "/ws")
        assert "fix oom" in result
        assert "j-123" in result
        assert "/ws" in result
        assert "{job_id}" not in result
        assert "{task_description}" not in result
        assert "{workspace}" not in result


class TestBuildSystemPrompt:
    def test_substitutes_values(self):
        issue_data = {
            "title": "Bug",
            "body": "desc",
            "labels": [],
            "comments": [],
        }
        result = build_system_prompt(issue_data, 42, "j-42", "/ws")
        assert "j-42" in result
        assert "#42" in result or "42" in result
        assert "/ws" in result
        assert "{job_id}" not in result
        assert "{workspace}" not in result

    def test_sanitizes_issue_content(self):
        issue_data = {
            "title": "Bug",
            "body": "header x-anthropic-secret leaks",
            "labels": [],
            "comments": [],
        }
        result = build_system_prompt(issue_data, 1, "j-1", "/ws")
        assert "x-anthropic-secret" not in result
        assert "[redacted-header]" in result


def _ok(*args, **kwargs) -> CompletedProcess[str]:
    return CompletedProcess(args="", returncode=0, stdout="", stderr="")


def _worktree_missing(*args, **kwargs) -> CompletedProcess[str]:
    return CompletedProcess(args="", returncode=1, stdout="", stderr="")


def _mock_backend(backend: LocalBackend | RemoteBackend) -> None:
    def run_side_effect(cmd: str, check: bool = True) -> CompletedProcess[str]:
        if "test -d" in cmd or "test -f" in cmd:
            return _worktree_missing()
        return _ok()

    backend.run = MagicMock(side_effect=run_side_effect)
    backend.copy_to = MagicMock()
    backend.launch_background = MagicMock(return_value=12345)
    backend.tail_log = MagicMock()


class TestLaunchAgentStdbuf:
    @patch("ptq.agent.deploy_scripts")
    def test_remote_backend_uses_stdbuf(self, _deploy, jobs_db, frozen_date):
        backend = RemoteBackend(machine="gpu-box", workspace="/tmp/ws")
        _mock_backend(backend)

        launch_agent(
            backend,
            message="hello",
            machine="gpu-box",
            follow=False,
        )

        cmd = backend.launch_background.call_args[0][0]
        assert "stdbuf -oL" in cmd

    @patch("ptq.agent.deploy_scripts")
    def test_local_backend_skips_stdbuf(self, _deploy, jobs_db, frozen_date):
        backend = LocalBackend(workspace="/tmp/ws")
        _mock_backend(backend)

        launch_agent(
            backend,
            message="hello",
            local=True,
            follow=False,
        )

        cmd = backend.launch_background.call_args[0][0]
        assert "stdbuf" not in cmd
