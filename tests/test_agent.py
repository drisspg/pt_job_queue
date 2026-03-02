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


def _worktree_exists(*args, **kwargs) -> CompletedProcess[str]:
    return CompletedProcess(args="", returncode=0, stdout="", stderr="")


def _mock_backend(
    backend: LocalBackend | RemoteBackend, *, worktree_exists: bool = False
) -> None:
    def run_side_effect(cmd: str, check: bool = True, **kw) -> CompletedProcess[str]:
        if "test -d" in cmd or "test -f" in cmd:
            return _worktree_exists() if worktree_exists else _worktree_missing()
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


class TestEarlyRegistration:
    @patch("ptq.agent.deploy_scripts")
    def test_job_registered_before_worktree_creation(
        self, _deploy, jobs_db, frozen_date
    ):
        """Job must be in DB before worktree/build so Ctrl-C leaves a cleanable entry."""
        backend = LocalBackend(workspace="/tmp/ws")

        call_order = []
        original_run = MagicMock(
            side_effect=lambda cmd, **kw: (
                call_order.append(("run", cmd)),
                CompletedProcess(
                    args="",
                    returncode=1 if ("test -d" in cmd or "test -f" in cmd) else 0,
                    stdout="",
                    stderr="",
                ),
            )[-1]
        )

        backend.run = original_run
        backend.copy_to = MagicMock()
        backend.launch_background = MagicMock(return_value=12345)
        backend.tail_log = MagicMock()

        launch_agent(backend, message="hello", local=True, follow=False)

        run_cmds = [cmd for _, cmd in call_order]
        mkdir_idx = next(i for i, c in enumerate(run_cmds) if "mkdir -p" in c)
        worktree_idx = next(
            i for i, c in enumerate(run_cmds) if "create_worktree.py" in c
        )

        registered_ids = list(jobs_db.keys())
        assert len(registered_ids) == 1

        assert mkdir_idx < worktree_idx
        assert registered_ids[0] in jobs_db

    @patch("ptq.agent.deploy_scripts")
    def test_job_in_db_even_if_build_not_started(self, _deploy, jobs_db, frozen_date):
        """Even before the agent launches, the job should be in the DB."""
        backend = LocalBackend(workspace="/tmp/ws")
        _mock_backend(backend, worktree_exists=True)

        job_id = launch_agent(backend, message="hello", local=True, follow=False)

        assert job_id in jobs_db
        assert jobs_db[job_id]["runs"] == 1

    @patch("ptq.agent.deploy_scripts")
    def test_rerun_does_not_re_register(self, _deploy, jobs_db, frozen_date):
        """Re-running with existing_job_id should not create a duplicate DB entry."""
        backend = LocalBackend(workspace="/tmp/ws")
        _mock_backend(backend, worktree_exists=True)

        first_id = launch_agent(backend, message="hello", local=True, follow=False)
        assert jobs_db[first_id]["runs"] == 1

        launch_agent(
            backend,
            message="try again",
            local=True,
            follow=False,
            existing_job_id=first_id,
        )

        assert len(jobs_db) == 1
        assert jobs_db[first_id]["runs"] == 2


class TestLaunchAgentRerun:
    @patch("ptq.agent.deploy_scripts")
    def test_adhoc_rerun_reuses_job_id(self, _deploy, jobs_db, frozen_date):
        """Re-running an adhoc job with existing_job_id should reuse the ID, not create a new one."""
        backend = LocalBackend(workspace="/tmp/ws")
        _mock_backend(backend, worktree_exists=True)

        first_id = launch_agent(backend, message="hello", local=True, follow=False)

        returned_id = launch_agent(
            backend,
            message="different message",
            local=True,
            follow=False,
            existing_job_id=first_id,
        )

        assert returned_id == first_id
        assert jobs_db[first_id]["runs"] == 2

    @patch("ptq.agent.deploy_scripts")
    def test_adhoc_rerun_loads_prior_context(self, _deploy, jobs_db, frozen_date):
        """Re-running an existing job should attempt to load prior worklog/report."""
        backend = LocalBackend(workspace="/tmp/ws")
        _mock_backend(backend, worktree_exists=True)

        first_id = launch_agent(backend, message="hello", local=True, follow=False)

        launch_agent(
            backend,
            message="try again",
            local=True,
            follow=False,
            existing_job_id=first_id,
        )

        run_cmds = [
            call.args[0]
            for call in backend.run.call_args_list
            if isinstance(call.args[0], str)
        ]
        cat_cmds = [c for c in run_cmds if "cat " in c and "worklog.md" in c]
        assert len(cat_cmds) >= 1

    @patch("ptq.agent.deploy_scripts")
    def test_new_adhoc_gets_unique_id(self, _deploy, jobs_db, frozen_date):
        """Two different adhoc messages should get different job IDs."""
        backend = LocalBackend(workspace="/tmp/ws")
        _mock_backend(backend)

        id1 = launch_agent(backend, message="task one", local=True, follow=False)
        id2 = launch_agent(backend, message="task two", local=True, follow=False)

        assert id1 != id2


class TestLaunchAgentType:
    @patch("ptq.agent.deploy_scripts")
    def test_agent_type_in_command(self, _deploy, jobs_db, frozen_date):
        """Each agent type should produce a distinct command binary."""
        backend = LocalBackend(workspace="/tmp/ws")
        _mock_backend(backend)

        for agent_type, expected_binary in [
            ("claude", "claude -p"),
            ("codex", "codex exec"),
            ("cursor", "agent -p"),
        ]:
            _mock_backend(backend)
            launch_agent(
                backend,
                message="hello",
                local=True,
                follow=False,
                agent_type=agent_type,
            )
            cmd = backend.launch_background.call_args[0][0]
            assert expected_binary in cmd, (
                f"{agent_type} cmd should contain {expected_binary!r}, got: {cmd}"
            )

    @patch("ptq.agent.deploy_scripts")
    def test_agent_type_persisted_in_db(self, _deploy, jobs_db, frozen_date):
        backend = LocalBackend(workspace="/tmp/ws")
        _mock_backend(backend)

        job_id = launch_agent(
            backend,
            message="hello",
            local=True,
            follow=False,
            agent_type="codex",
        )

        assert jobs_db[job_id]["agent"] == "codex"

    @patch("ptq.agent.deploy_scripts")
    def test_agent_log_filename(self, _deploy, jobs_db, frozen_date):
        """Log file should use the agent name, not hardcoded 'claude'."""
        backend = LocalBackend(workspace="/tmp/ws")
        _mock_backend(backend)

        for agent_type in ("claude", "codex", "cursor"):
            _mock_backend(backend)
            launch_agent(
                backend,
                message="hello",
                local=True,
                follow=False,
                agent_type=agent_type,
            )
            log_file = backend.launch_background.call_args[0][1]
            assert f"{agent_type}-1.log" in log_file

    @patch("ptq.agent.deploy_scripts")
    def test_codex_setup_copies_agents_md(self, _deploy, jobs_db, frozen_date):
        backend = LocalBackend(workspace="/tmp/ws")
        _mock_backend(backend)

        launch_agent(
            backend,
            message="hello",
            local=True,
            follow=False,
            agent_type="codex",
        )

        run_cmds = [
            call.args[0]
            for call in backend.run.call_args_list
            if isinstance(call.args[0], str)
        ]
        assert any("AGENTS.md" in c for c in run_cmds)

    @patch("ptq.agent.deploy_scripts")
    def test_cursor_setup_copies_cursorrules(self, _deploy, jobs_db, frozen_date):
        backend = LocalBackend(workspace="/tmp/ws")
        _mock_backend(backend)

        launch_agent(
            backend,
            message="hello",
            local=True,
            follow=False,
            agent_type="cursor",
        )

        run_cmds = [
            call.args[0]
            for call in backend.run.call_args_list
            if isinstance(call.args[0], str)
        ]
        assert any(".cursorrules" in c for c in run_cmds)

    @patch("ptq.agent.deploy_scripts")
    def test_claude_setup_writes_settings_json(self, _deploy, jobs_db, frozen_date):
        backend = LocalBackend(workspace="/tmp/ws")
        _mock_backend(backend)

        launch_agent(
            backend,
            message="hello",
            local=True,
            follow=False,
            agent_type="claude",
        )

        run_cmds = [
            call.args[0]
            for call in backend.run.call_args_list
            if isinstance(call.args[0], str)
        ]
        assert any(".claude/settings.json" in c for c in run_cmds)
