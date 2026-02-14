from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from rich.console import Console

from ptq.issue import extract_repro_script, format_issue_context
from ptq.job import make_job_id, register_job
from ptq.ssh import Backend
from ptq.workspace import deploy_scripts

console = Console()

PROMPT_TEMPLATE = (Path(__file__).parent.parent / "prompts" / "investigate.md").read_text()


def build_system_prompt(issue_data: dict, issue_number: int, job_id: str, workspace: str) -> str:
    return PROMPT_TEMPLATE.format(
        job_id=job_id,
        issue_number=issue_number,
        issue_context=format_issue_context(issue_data, issue_number),
        workspace=workspace,
    )


def _print_stream_event(line: str) -> None:
    line = line.strip()
    if not line:
        return
    event = json.loads(line)
    match event.get("type"):
        case "assistant":
            for block in event.get("message", {}).get("content", []):
                if block.get("type") == "text" and block.get("text"):
                    sys.stdout.write(block["text"])
                    sys.stdout.flush()
                elif block.get("type") == "tool_use":
                    tool = block.get("name", "")
                    inp = block.get("input", {})
                    match tool:
                        case "Bash":
                            console.print(f"\n[dim]$ {inp.get('command', '')}[/dim]")
                        case "Read":
                            console.print(f"\n[dim]reading {inp.get('file_path', '')}[/dim]")
                        case "Edit":
                            console.print(f"\n[dim]editing {inp.get('file_path', '')}[/dim]")
                        case "Write":
                            console.print(f"\n[dim]writing {inp.get('file_path', '')}[/dim]")
                        case "Grep":
                            console.print(f"\n[dim]grep {inp.get('pattern', '')}[/dim]")
                        case "Glob":
                            console.print(f"\n[dim]glob {inp.get('pattern', '')}[/dim]")
                        case _:
                            console.print(f"\n[dim]{tool}[/dim]")


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
) -> str:
    job_id = make_job_id(issue_number)
    workspace = backend.workspace
    job_dir = f"{workspace}/jobs/{job_id}"
    worktree_path = f"{job_dir}/pytorch"

    console.print(f"[bold]Job {job_id}[/bold] — issue #{issue_number}")

    backend.run(f"mkdir -p {job_dir}")

    deploy_scripts(backend)

    console.print("Creating git worktree...")
    backend.run(
        f"cd {workspace}/pytorch && git worktree add {worktree_path} HEAD",
    )

    system_prompt = build_system_prompt(issue_data, issue_number, job_id, workspace)
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
        console.print("[yellow]No repro script found in issue — agent will write one.[/yellow]")

    claude_cmd = (
        f"cd {worktree_path} && "
        f"claude -p 'Investigate and fix the PyTorch issue described in your system prompt.' "
        f"--model {model} "
        f"--max-turns {max_turns} "
        f"--allowedTools 'Read,Edit,Write,Bash,Grep,Glob' "
        f"--dangerously-skip-permissions "
        f"--append-system-prompt-file {job_dir}/system_prompt.md "
        f"--output-format stream-json "
        f"--verbose "
        f"2>&1 | tee {job_dir}/claude.log"
    )

    register_job(job_id, machine=machine, local=local, workspace=workspace)
    console.print(f"Launching agent ({'local' if local else machine})...")

    proc = backend.run_streaming(claude_cmd, follow=follow)

    if follow and hasattr(proc, "stdout") and proc.stdout:
        try:
            for line in proc.stdout:
                if line.strip().startswith("{"):
                    _print_stream_event(line)
            proc.wait()
            console.print(f"\n[bold]Agent finished.[/bold]")
            console.print(f"  ptq results {job_id}")
        except KeyboardInterrupt:
            proc.terminate()
            proc.wait()
            console.print(f"\n[bold yellow]Interrupted.[/bold yellow]")
            console.print(f"  ptq results {job_id}")
    else:
        console.print(f"[bold]Agent launched in background.[/bold]")
        console.print(f"  ptq results {job_id}")

    return job_id
