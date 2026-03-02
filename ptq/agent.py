from __future__ import annotations

import json
import re
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

from rich.console import Console

from ptq.agents import RunContext, StreamEvent, get_agent
from ptq.issue import extract_repro_script, format_issue_context
from ptq.job import (
    find_existing_job,
    increment_run,
    make_job_id,
    register_job,
    save_pid,
)
from ptq.ssh import Backend, RemoteBackend
from ptq.workspace import deploy_scripts

console = Console()


@contextmanager
def _timed(label: str):
    t0 = time.monotonic()
    yield
    elapsed = time.monotonic() - t0
    console.print(f"[dim]  {label}: {elapsed:.1f}s[/dim]")


PROMPT_TEMPLATE = (
    Path(__file__).parent.parent / "prompts" / "investigate.md"
).read_text()

ADHOC_PROMPT_TEMPLATE = (
    Path(__file__).parent.parent / "prompts" / "adhoc.md"
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


def build_adhoc_prompt(message: str, job_id: str, workspace: str) -> str:
    return _sanitize_for_api(
        ADHOC_PROMPT_TEMPLATE.format(
            job_id=job_id,
            task_description=message,
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


def _render_event(ev: StreamEvent) -> None:
    match ev.kind:
        case "text":
            console.print(_clean(ev.text), end="", highlight=False)
        case "tool_use":
            console.print()
            inp = ev.tool_input
            match ev.tool_name:
                case "Bash":
                    console.print(
                        f"  [bold cyan]$[/bold cyan] [dim]{inp.get('command', '')}[/dim]"
                    )
                case "Read":
                    console.print(
                        f"  [cyan]read[/cyan] [dim]{inp.get('file_path', '') or inp.get('path', '')}[/dim]"
                    )
                case "Edit":
                    console.print(
                        f"  [yellow]edit[/yellow] [dim]{inp.get('file_path', '') or inp.get('path', '')}[/dim]"
                    )
                case "Write":
                    console.print(
                        f"  [green]write[/green] [dim]{inp.get('file_path', '') or inp.get('path', '')}[/dim]"
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
                    console.print(f"  [dim]{ev.tool_name}[/dim]")
        case "tool_result":
            if ev.text.strip():
                console.print(f"[dim]{_indent(_truncate(_clean(ev.text)))}[/dim]")
            console.print()
        case "error":
            console.print(f"[red]{_indent(_truncate(_clean(ev.text)))}[/red]")
            console.print()


DEFAULT_MESSAGE = (
    "Investigate and fix the PyTorch issue described in your system prompt."
)


def _build_prior_context(backend: Backend, job_dir: str, run_number: int) -> str:
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

    sections.append(
        f"\n## Continuation Instructions\n"
        f"This is **run {run_number}**. A `## Run {run_number}` section (with the "
        f"user's steering message) has already been appended to the worklog. You MUST:\n"
        f"1. Append your findings, analysis, or changes under that section before "
        f"you finish — even if the user's message was a question rather than a fix request.\n"
        f"2. If you made any code changes, regenerate `fix.diff` and update `report.md`.\n"
        f"3. If the user asked an analytical question, update `report.md` with your "
        f"findings as a new section.\n"
        f"\nEvery run must leave a trace in the worklog and artifacts.\n"
    )
    return "\n".join(sections)


def _setup_job_venv(
    backend: Backend, job_dir: str, worktree_path: str, *, verbose: bool = False
) -> None:
    console.print("Creating per-job venv...")
    with _timed("venv creation"):
        backend.run(f"cd {job_dir} && uv venv --python 3.12")

    job_python = f"{job_dir}/.venv/bin/python"
    console.print("Installing build deps...")
    with _timed("build deps"):
        backend.run(
            f"cd {worktree_path} && "
            f"uv pip install --python {job_python} -r requirements-build.txt",
            stream=verbose,
        )
    pip_verbose = " -v" if verbose else ""
    console.print("Editable install (pytorch)...")
    with _timed("editable install"):
        result = backend.run(
            f"cd {worktree_path} && CCACHE_NOHASHDIR=true USE_NINJA=1 "
            f"uv pip install --python {job_python} --no-build-isolation{pip_verbose} -e .",
            check=False,
            stream=verbose,
        )
    if result.returncode != 0:
        console.print(f"[red]Editable install failed:[/red]\n{result.stderr}")
        console.print("[yellow]Agent will need to build manually.[/yellow]")
    else:
        console.print("[green]Editable install complete.[/green]")


def _validate_workspace(backend: Backend, workspace: str) -> None:
    result = backend.run(f"test -d {workspace}/pytorch/.git", check=False)
    if result.returncode != 0:
        raise SystemExit(
            f"Workspace broken: {workspace}/pytorch/.git missing. Re-run: ptq setup"
        )


def _stamp_worklog_header(
    backend: Backend, job_dir: str, run_number: int, message: str | None
) -> None:
    lines = ["", "", f"## Run {run_number}", ""]
    if message:
        lines.append(f"> **User:** {message}")
        lines.append("")
    header = "\n".join(lines)
    backend.run(
        f"cat >> {job_dir}/worklog.md << 'WORKLOG_STAMP_EOF'\n{header}\nWORKLOG_STAMP_EOF"
    )


def launch_agent(
    backend: Backend,
    *,
    issue_data: dict | None = None,
    issue_number: int | None = None,
    machine: str | None = None,
    local: bool = False,
    follow: bool = True,
    model: str = "opus",
    max_turns: int = 100,
    message: str | None = None,
    agent_type: str = "claude",
    existing_job_id: str | None = None,
    verbose: bool = False,
) -> str:
    agent = get_agent(agent_type)
    workspace = backend.workspace
    is_adhoc = issue_number is None

    if existing_job_id:
        job_id = existing_job_id
        run_number = increment_run(job_id)
        label = f"issue #{issue_number}" if issue_number else "adhoc"
        console.print(f"[bold]Job {job_id}[/bold] — {label} (run {run_number})")
        existing = job_id
    elif is_adhoc:
        existing = None
        job_id = make_job_id(message=message)
        run_number = 1
        console.print(f"[bold]Job {job_id}[/bold] — adhoc (run 1)")
    else:
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

    if existing:
        _validate_workspace(backend, workspace)

    backend.run(f"mkdir -p {job_dir}")

    if not existing:
        register_job(
            job_id,
            issue_number=issue_number,
            machine=machine,
            local=local,
            workspace=workspace,
            run_number=run_number,
            agent_type=agent_type,
        )

    deploy_scripts(backend)

    worktree_exists = backend.run(
        f"test -d {worktree_path}/.git || test -f {worktree_path}/.git", check=False
    )
    if worktree_exists.returncode != 0:
        console.print("Creating worktree with submodules...")
        with _timed("worktree creation"):
            backend.run(
                f"cd {workspace}/pytorch && python tools/create_worktree.py create pytorch "
                f"--parent-dir {job_dir} --commit HEAD",
                stream=verbose,
            )
        _setup_job_venv(backend, job_dir, worktree_path, verbose=verbose)
    else:
        console.print("Reusing existing worktree.")

    with _timed("agent workspace setup"):
        agent.setup_workspace(backend, worktree_path, job_dir, workspace)

    if is_adhoc:
        system_prompt = build_adhoc_prompt(message, job_id, workspace)
    else:
        system_prompt = build_system_prompt(issue_data, issue_number, job_id, workspace)

    if existing:
        prior_context = _build_prior_context(backend, job_dir, run_number)
        if prior_context:
            system_prompt += prior_context
            console.print("Loaded prior run context (worklog/report).")

    _stamp_worklog_header(backend, job_dir, run_number, message)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(system_prompt)
        prompt_tmp = Path(f.name)

    backend.copy_to(prompt_tmp, f"{job_dir}/system_prompt.md")
    prompt_tmp.unlink()

    if not is_adhoc:
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

    if is_adhoc:
        agent_message = message
    elif existing:
        agent_message = message or DEFAULT_MESSAGE
    elif message:
        agent_message = f"{DEFAULT_MESSAGE}\n\nAdditional context: {message}"
    else:
        agent_message = DEFAULT_MESSAGE

    log_file = f"{job_dir}/{agent.log_filename(run_number)}"
    unbuffer = "stdbuf -oL " if isinstance(backend, RemoteBackend) else ""
    ctx = RunContext(
        worktree_path=worktree_path,
        job_dir=job_dir,
        message=agent_message,
        model=model,
        max_turns=max_turns,
        system_prompt_file=f"{job_dir}/system_prompt.md",
        unbuffer_prefix=unbuffer,
    )
    agent_cmd = agent.build_cmd(ctx)

    console.print(f"Launching {agent.name} agent ({'local' if local else machine})...")
    backend.run(f"touch {log_file}")
    pid = backend.launch_background(agent_cmd, log_file)
    save_pid(job_id, pid)

    if not follow:
        console.print("[bold]Agent launched in background.[/bold]")
        console.print(f"  ptq results {job_id}")
        return job_id

    time.sleep(2)
    tail = backend.tail_log(log_file)

    try:
        for line in tail.stdout:
            stripped = _clean(line.strip())
            if stripped.startswith("{"):
                try:
                    for ev in agent.parse_stream_line(stripped):
                        _render_event(ev)
                except (json.JSONDecodeError, ValueError):
                    pass
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
