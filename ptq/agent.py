from __future__ import annotations

import json
import re
import tempfile
import time
from pathlib import Path

from rich.console import Console

from ptq.issue import extract_repro_script, format_issue_context
from ptq.job import (
    find_existing_job,
    increment_run,
    make_job_id,
    register_job,
    save_pid,
)
from ptq.ssh import Backend
from ptq.workspace import deploy_scripts

console = Console()

PROMPT_TEMPLATE = (
    Path(__file__).parent.parent / "prompts" / "investigate.md"
).read_text()


RESERVED_HEADER_RE = re.compile(r"x-anthropic-\S+", re.IGNORECASE)


def _sanitize_for_api(text: str) -> str:
    return RESERVED_HEADER_RE.sub("[redacted-header]", text)


def build_system_prompt(
    issue_data: dict, issue_number: int, job_id: str, workspace: str
) -> str:
    return _sanitize_for_api(
        PROMPT_TEMPLATE.format(
            job_id=job_id,
            issue_number=issue_number,
            issue_context=format_issue_context(issue_data, issue_number),
            workspace=workspace,
        )
    )


MAX_OUTPUT_LINES = 30

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]|\r")


def _clean(text: str) -> str:
    return ANSI_RE.sub("", text)


def _truncate(text: str, max_lines: int = MAX_OUTPUT_LINES) -> str:
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    return "\n".join(lines[:max_lines]) + f"\n... ({len(lines) - max_lines} more lines)"


def _indent(text: str, prefix: str = "    ") -> str:
    return "\n".join(prefix + line for line in text.splitlines())


def _print_stream_event(line: str) -> None:
    line = _clean(line.strip())
    if not line:
        return
    event = json.loads(line)
    match event.get("type"):
        case "assistant":
            for block in event.get("message", {}).get("content", []):
                if block.get("type") == "text" and block.get("text"):
                    console.print(_clean(block["text"]), end="", highlight=False)
                elif block.get("type") == "tool_use":
                    tool = block.get("name", "")
                    inp = block.get("input", {})
                    console.print()
                    match tool:
                        case "Bash":
                            console.print(
                                f"  [bold cyan]$[/bold cyan] [dim]{inp.get('command', '')}[/dim]"
                            )
                        case "Read":
                            console.print(
                                f"  [cyan]read[/cyan] [dim]{inp.get('file_path', '')}[/dim]"
                            )
                        case "Edit":
                            console.print(
                                f"  [yellow]edit[/yellow] [dim]{inp.get('file_path', '')}[/dim]"
                            )
                        case "Write":
                            console.print(
                                f"  [green]write[/green] [dim]{inp.get('file_path', '')}[/dim]"
                            )
                        case "Grep":
                            console.print(
                                f"  [cyan]grep[/cyan] [dim]{inp.get('pattern', '')}[/dim]"
                            )
                        case "Glob":
                            console.print(
                                f"  [cyan]glob[/cyan] [dim]{inp.get('pattern', '')}[/dim]"
                            )
                        case _:
                            console.print(f"  [dim]{tool}[/dim]")
        case "user":
            for block in event.get("message", {}).get("content", []):
                if block.get("type") != "tool_result":
                    continue
                content = block.get("content", "")
                is_error = block.get("is_error", False)
                if is_error:
                    console.print(f"[red]{_indent(_truncate(_clean(content)))}[/red]")
                elif content:
                    result = event.get("tool_use_result", {})
                    stdout = (
                        result.get("stdout", "") if isinstance(result, dict) else ""
                    )
                    if stdout.strip():
                        console.print(
                            f"[dim]{_indent(_truncate(_clean(stdout)))}[/dim]"
                        )
                console.print()


DEFAULT_MESSAGE = (
    "Investigate and fix the PyTorch issue described in your system prompt."
)


def _build_prior_context(backend: Backend, job_dir: str) -> str:
    worklog = backend.run(f"cat {job_dir}/worklog.md", check=False)
    report = backend.run(f"cat {job_dir}/report.md", check=False)

    worklog_content = worklog.stdout.strip() if worklog.returncode == 0 else ""
    report_content = report.stdout.strip() if report.returncode == 0 else ""

    if not worklog_content and not report_content:
        return ""

    sections = [
        "\n\n## Prior Run Context\n",
        "The following is from a previous investigation attempt on this issue. "
        "Use it to avoid repeating work and to build on what was already found.\n",
    ]
    if worklog_content:
        sections.append(f"### Previous Worklog\n{worklog_content}\n")
    if report_content:
        sections.append(f"### Previous Report\n{report_content}\n")

    return "\n".join(sections)


def launch_agent(
    backend: Backend,
    issue_data: dict,
    issue_number: int,
    *,
    machine: str | None = None,
    local: bool = False,
    follow: bool = True,
    model: str = "opus",
    max_turns: int = 100,
    message: str | None = None,
) -> str:
    workspace = backend.workspace
    existing = find_existing_job(issue_number, machine=machine, local=local)

    if existing:
        job_id = existing
        run_number = increment_run(job_id)
        console.print(
            f"[bold]Job {job_id}[/bold] — issue #{issue_number} (run {run_number})"
        )
    else:
        job_id = make_job_id(issue_number)
        run_number = 1
        console.print(f"[bold]Job {job_id}[/bold] — issue #{issue_number} (run 1)")

    job_dir = f"{workspace}/jobs/{job_id}"
    worktree_path = f"{job_dir}/pytorch"

    backend.run(f"mkdir -p {job_dir}")
    deploy_scripts(backend)

    worktree_exists = backend.run(
        f"test -d {worktree_path}/.git || test -f {worktree_path}/.git", check=False
    )
    if worktree_exists.returncode != 0:
        console.print("Creating git worktree...")
        backend.run(
            f"cd {workspace}/pytorch && git worktree add {worktree_path} HEAD",
        )
    else:
        console.print("Reusing existing worktree.")

    backend.run(f"mkdir -p {worktree_path}/.claude", check=False)
    sandbox_settings = json.dumps(
        {
            "sandbox": {
                "enabled": True,
                "allowedDirectories": [
                    job_dir,
                    f"{workspace}/.venv",
                    f"{workspace}/scripts",
                ],
            }
        }
    )
    backend.run(
        f"cat > {worktree_path}/.claude/settings.json << 'SETTINGS_EOF'\n{sandbox_settings}\nSETTINGS_EOF"
    )

    system_prompt = build_system_prompt(issue_data, issue_number, job_id, workspace)

    if existing:
        prior_context = _build_prior_context(backend, job_dir)
        if prior_context:
            system_prompt += prior_context
            console.print("Loaded prior run context (worklog/report).")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(system_prompt)
        prompt_tmp = Path(f.name)

    backend.copy_to(prompt_tmp, f"{job_dir}/system_prompt.md")
    prompt_tmp.unlink()

    repro = extract_repro_script(issue_data)
    if repro:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(repro)
            repro_tmp = Path(f.name)
        backend.copy_to(repro_tmp, f"{job_dir}/repro.py")
        repro_tmp.unlink()
        console.print("Extracted and uploaded repro script.")
    else:
        console.print(
            "[yellow]No repro script found in issue — agent will write one.[/yellow]"
        )

    agent_message = message or DEFAULT_MESSAGE
    log_file = f"{job_dir}/claude-{run_number}.log"
    escaped_message = agent_message.replace("'", "'\\''")
    claude_cmd = (
        f"cd {worktree_path} && "
        f"stdbuf -oL "
        f"claude -p '{escaped_message}' "
        f"--model {model} "
        f"--max-turns {max_turns} "
        f"--allowedTools 'Read,Edit,Write,Bash,Grep,Glob' "
        f"--dangerously-skip-permissions "
        f"--append-system-prompt-file {job_dir}/system_prompt.md "
        f"--output-format stream-json "
        f"--verbose"
    )

    if not existing:
        register_job(
            job_id,
            machine=machine,
            local=local,
            workspace=workspace,
            run_number=run_number,
        )

    console.print(f"Launching agent ({'local' if local else machine})...")
    backend.run(f"touch {log_file}")
    pid = backend.launch_background(claude_cmd, log_file)
    save_pid(job_id, pid)

    if not follow:
        console.print("[bold]Agent launched in background.[/bold]")
        console.print(f"  ptq results {job_id}")
        return job_id

    time.sleep(2)
    tail = backend.tail_log(log_file)

    try:
        for line in tail.stdout:
            if line.strip().startswith("{"):
                _print_stream_event(line)
    except KeyboardInterrupt:
        tail.terminate()
        tail.wait()
        console.print(
            "\n[bold yellow]Detached. Agent still running on remote.[/bold yellow]"
        )
        console.print(f"  ptq results {job_id}")
        return job_id

    tail.wait()
    console.print("\n[bold]Agent finished.[/bold]")
    console.print(f"  ptq results {job_id}")

    return job_id
