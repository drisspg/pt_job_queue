from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

app = typer.Typer(
    name="ptq", help="PyTorch Job Queue — dispatch Claude agents to fix PyTorch issues."
)
console = Console()


@app.command()
def setup(
    machine: Annotated[
        str | None, typer.Argument(help="Remote machine to set up.")
    ] = None,
    local: Annotated[
        bool, typer.Option("--local", help="Set up local workspace instead.")
    ] = False,
    cuda: Annotated[
        str | None, typer.Option(help="CUDA tag override (cu124, cu126, cu128, cu130).")
    ] = None,
    cpu: Annotated[
        bool, typer.Option("--cpu", help="Use CPU-only PyTorch (for macOS/testing).")
    ] = False,
    workspace: Annotated[
        str | None, typer.Option(help="Custom workspace path.")
    ] = None,
) -> None:
    """One-time workspace setup on a remote machine or locally."""
    if not machine and not local:
        raise typer.BadParameter("Provide a machine name or use --local.")

    from ptq.ssh import LocalBackend, RemoteBackend
    from ptq.workspace import setup_workspace

    if local:
        backend = LocalBackend(workspace=workspace or "~/.ptq_workspace")
    else:
        assert machine is not None
        backend = RemoteBackend(
            machine=machine, workspace=workspace or "~/ptq_workspace"
        )

    setup_workspace(backend, cuda_tag=cuda, cpu=cpu)


@app.command()
def run(
    job_id: Annotated[
        str | None, typer.Argument(help="Job ID or issue number to re-run.")
    ] = None,
    issue: Annotated[int | None, typer.Option(help="GitHub issue number.")] = None,
    machine: Annotated[
        str | None, typer.Option(help="Remote machine to run on.")
    ] = None,
    local: Annotated[bool, typer.Option("--local", help="Run locally.")] = False,
    follow: Annotated[
        bool, typer.Option(help="Stream agent output to terminal.")
    ] = True,
    model: Annotated[str, typer.Option(help="Claude model to use.")] = "opus",
    max_turns: Annotated[int, typer.Option(help="Max agent turns.")] = 100,
    workspace: Annotated[
        str | None, typer.Option(help="Custom workspace path.")
    ] = None,
    message: Annotated[
        str | None,
        typer.Option("--message", "-m", help="Custom instruction for the agent."),
    ] = None,
    input_file: Annotated[
        Path | None,
        typer.Option("--input", "-i", help="Read task description from a file."),
    ] = None,
) -> None:
    """Launch a Claude agent to investigate a PyTorch issue or run an adhoc task.

    Provide --issue for GitHub issue investigation, or --message for a freeform task.
    Re-run an existing job by passing its JOB_ID (or issue number) as a positional arg.

    Examples:
        ptq run --issue 174923 --machine aws-gpu-dev
        ptq run -m "investigate flex_attention OOM on H100" --machine gpu-dev
        ptq run -i task.md --machine gpu-dev
        ptq run 20260214-174923 -m "look at flex_attention.py instead"
        ptq run 174923 -m "try a different approach"
    """
    if input_file is not None and message is not None:
        raise typer.BadParameter("--input and --message are mutually exclusive.")
    if input_file is not None:
        if not input_file.exists():
            raise typer.BadParameter(f"File not found: {input_file}")
        message = input_file.read_text()

    from ptq.agent import launch_agent
    from ptq.issue import fetch_issue
    from ptq.ssh import LocalBackend, RemoteBackend

    if job_id is not None:
        from ptq.job import get_job, resolve_job_id

        resolved = resolve_job_id(job_id)
        job = get_job(resolved)
        issue = issue or job["issue"]
        machine = machine or job.get("machine")
        local = local or job.get("local", False)
        workspace = workspace or job.get("workspace")

    if issue is None and message is None and job_id is None:
        raise typer.BadParameter("Provide --issue, --message, or a JOB_ID to re-run.")
    if not machine and not local:
        raise typer.BadParameter("Provide --machine or --local.")

    if issue is not None:
        console.print(f"Fetching issue #{issue}...")
        issue_data = fetch_issue(issue)
        console.print(f"[bold]{issue_data['title']}[/bold]")
    else:
        issue_data = None

    if local:
        backend = LocalBackend(workspace=workspace or "~/.ptq_workspace")
    else:
        assert machine is not None
        backend = RemoteBackend(
            machine=machine, workspace=workspace or "~/ptq_workspace"
        )

    launch_agent(
        backend,
        issue_data=issue_data,
        issue_number=issue,
        machine=machine,
        local=local,
        follow=follow,
        model=model,
        max_turns=max_turns,
        message=message,
    )


@app.command()
def auto(
    issue: Annotated[int | None, typer.Option(help="GitHub issue number.")] = None,
    message: Annotated[
        str | None,
        typer.Option("--message", "-m", help="Custom instruction for the agent."),
    ] = None,
    input_file: Annotated[
        Path | None,
        typer.Option("--input", "-i", help="Read task description from a file."),
    ] = None,
    gpu_type: Annotated[str, typer.Option(help="GPU type.")] = "h100",
    gpus: Annotated[int, typer.Option(help="Number of GPUs.")] = 1,
    hours: Annotated[float, typer.Option(help="Max reservation hours.")] = 4.0,
    dockerfile: Annotated[
        Path | None, typer.Option(help="Custom Dockerfile to use.")
    ] = None,
    no_pr: Annotated[
        bool, typer.Option("--no-pr", help="Skip PR creation.")
    ] = False,
    prompt: Annotated[
        str | None,
        typer.Option("--prompt", "-p", help="Extra guidance for the agent."),
    ] = None,
    model: Annotated[str, typer.Option(help="Claude model to use.")] = "opus",
    max_turns: Annotated[int, typer.Option(help="Max agent turns.")] = 100,
    follow: Annotated[
        bool, typer.Option(help="Stream agent output to terminal.")
    ] = True,
) -> None:
    """Full auto: reserve GPU, setup, run agent, fetch results, cancel reservation.

    Handles the complete lifecycle: gpu-dev reservation -> workspace setup ->
    agent launch -> results fetch -> reservation cancellation. The reservation
    auto-cancels when the agent finishes (or on Ctrl+C / crash).

    Examples:
        ptq auto --issue 149002
        ptq auto --issue 149002 --gpu-type a100 --hours 2
        ptq auto --issue 149002 --no-pr
        ptq auto --issue 149002 -p "focus on the nan_assert codepath"
        ptq auto -m "investigate flex_attention OOM" --hours 6
    """
    if input_file is not None and message is not None:
        raise typer.BadParameter("--input and --message are mutually exclusive.")
    if input_file is not None:
        if not input_file.exists():
            raise typer.BadParameter(f"File not found: {input_file}")
        message = input_file.read_text()
    if issue is None and message is None:
        raise typer.BadParameter("Provide --issue or --message/-i.")

    from ptq.agent import launch_agent
    from ptq.issue import fetch_issue
    from ptq.job import save_reservation_id
    from ptq.provision import cancel_reservation, create_reservation, ensure_ssh_config
    from ptq.results import fetch_results
    from ptq.ssh import RemoteBackend
    from ptq.workspace import setup_workspace

    # 1. Fetch issue data (fail fast before reserving GPU)
    if issue is not None:
        console.print(f"Fetching issue #{issue}...")
        issue_data = fetch_issue(issue)
        console.print(f"[bold]{issue_data['title']}[/bold]")
    else:
        issue_data = None

    # 2. Build Docker context with skills (only if custom dockerfile provided)
    docker_context = None
    if dockerfile:
        from ptq.docker import build_context_tarball

        console.print("Building Docker context...")
        docker_context = build_context_tarball(
            extra_dockerfile=dockerfile,
        )

    # 3. Verify SSH config
    if not ensure_ssh_config():
        console.print(
            "[yellow]Warning: ~/.ssh/config may not include gpu-dev SSH config. "
            "Run gpu-dev setup if SSH connection fails.[/yellow]"
        )

    # 4. Create reservation
    reservation_name = f"ptq-{issue}" if issue else "ptq-adhoc"
    console.print(
        f"Reserving {gpus}x {gpu_type} for up to {hours}h..."
    )

    reservation_id = None
    job_id = None
    try:
        reservation_id, pod_name, conn_info = create_reservation(
            gpu_type=gpu_type,
            gpu_count=gpus,
            duration_hours=hours,
            dockerfile=docker_context,
            name=reservation_name,
        )
        console.print(
            f"[bold green]Reservation active[/bold green]: {reservation_id[:8]}... "
            f"(pod: {pod_name})"
        )

        # 5. Setup workspace (with git identity from local machine)
        import subprocess as _sp

        _gh_result = _sp.run(
            ["gh", "auth", "token"], capture_output=True, text=True, check=False
        )
        gh_token = _gh_result.stdout.strip() if _gh_result.returncode == 0 else None

        _name_result = _sp.run(
            ["git", "config", "user.name"], capture_output=True, text=True, check=False
        )
        git_name = _name_result.stdout.strip() or None

        _email_result = _sp.run(
            ["git", "config", "user.email"], capture_output=True, text=True, check=False
        )
        git_email = _email_result.stdout.strip() or None

        backend = RemoteBackend(machine=pod_name)
        console.print("Setting up workspace...")
        setup_workspace(backend, git_name=git_name, git_email=git_email)

        # Configure GitHub auth + SSH on pod
        if gh_token:
            console.print("Configuring GitHub auth on pod...")
            hosts_yml = (
                "github.com:\n"
                f"    oauth_token: {gh_token}\n"
                "    git_protocol: https\n"
            )
            backend.run("mkdir -p ~/.config/gh")
            backend.run(
                f"cat > ~/.config/gh/hosts.yml << 'GH_HOSTS_EOF'\n{hosts_yml}GH_HOSTS_EOF"
            )
            # Configure git to use gh as credential helper (for git push over HTTPS)
            backend.run("gh auth setup-git", check=False)
        else:
            console.print(
                "[yellow]No GH_TOKEN found locally — PR creation will be skipped.[/yellow]"
            )

        # Add GitHub SSH host keys so git push over SSH doesn't fail
        backend.run(
            "mkdir -p ~/.ssh && ssh-keyscan -t ed25519 github.com >> ~/.ssh/known_hosts 2>/dev/null",
            check=False,
        )

        # 6. Build agent message with PR instruction
        pr_instruction = ""
        if not no_pr and issue is not None:
            pr_instruction = (
                "\n\nAfter fixing the issue, use the verify-fix skill to run "
                "tests and perf checks, then use the make-pr skill to create "
                "a draft PR."
            )

        # 7. Launch agent
        agent_message = message or ""
        if prompt:
            agent_message = f"{agent_message}\n\n{prompt}" if agent_message else prompt
        if pr_instruction:
            agent_message = f"{agent_message}{pr_instruction}"
        job_id = launch_agent(
            backend,
            issue_data=issue_data,
            issue_number=issue,
            machine=pod_name,
            follow=follow,
            model=model,
            max_turns=max_turns,
            message=agent_message or None,
        )

        # Track reservation with job
        if job_id:
            save_reservation_id(job_id, reservation_id)

        # 8. Fetch results
        if job_id:
            console.print("Fetching results...")
            try:
                results_dir = fetch_results(job_id)
                console.print(f"Artifacts saved to: {results_dir}")
            except Exception as e:
                console.print(f"[yellow]Could not fetch results: {e}[/yellow]")

    finally:
        # 9. Cancel reservation on exit — but not in no-follow mode
        # (agent is still running on the remote)
        if reservation_id and follow:
            console.print(f"Cancelling reservation {reservation_id[:8]}...")
            if cancel_reservation(reservation_id):
                console.print("[bold]Reservation cancelled.[/bold]")
            else:
                console.print(
                    f"[yellow]Could not cancel reservation {reservation_id}. "
                    f"Cancel manually: gpu-dev cancel {reservation_id}[/yellow]"
                )
        elif reservation_id and not follow:
            console.print(
                f"Reservation {reservation_id[:8]}... still active "
                f"(auto-expires in {hours}h)."
            )
            console.print(
                f"  gpu-dev cancel {reservation_id}  # to cancel early"
            )


@app.command()
def results(
    job_id: Annotated[str, typer.Argument(help="Job ID or issue number.")],
    output_dir: Annotated[
        Path | None, typer.Option(help="Custom output directory.")
    ] = None,
) -> None:
    """Fetch and display results from a completed job."""
    from ptq.job import resolve_job_id
    from ptq.results import display_results, fetch_results

    job_id = resolve_job_id(job_id)
    console.print(f"Fetching results for {job_id}...")
    results_dir = fetch_results(job_id, output_dir)
    display_results(results_dir)
    console.print(f"\nArtifacts saved to: {results_dir}")


@app.command()
def apply(
    job_id: Annotated[str, typer.Argument(help="Job ID or issue number.")],
    pytorch_path: Annotated[
        Path, typer.Option(help="Path to local pytorch checkout.")
    ] = Path("~/meta/pytorch"),
) -> None:
    """Apply a job's diff to a local PyTorch checkout."""
    from ptq.apply import apply_diff
    from ptq.job import resolve_job_id

    job_id = resolve_job_id(job_id)
    apply_diff(job_id, pytorch_path.expanduser())


def _clean_single_job(job_id: str) -> None:
    """Kill agent, cancel reservation, remove remote files, drop from DB."""
    from ptq.job import get_job, get_reservation_id
    from ptq.ssh import backend_for_job, load_jobs_db, save_jobs_db

    job = get_job(job_id)
    backend = backend_for_job(job_id)
    ws = backend.workspace
    job_dir = f"{ws}/jobs/{job_id}"

    pid = job.get("pid")
    if pid and backend.is_pid_alive(pid):
        backend.kill_pid(pid)
        console.print(f"  killed agent (pid {pid})")

    res_id = get_reservation_id(job_id)
    if res_id:
        from ptq.provision import cancel_reservation

        if cancel_reservation(res_id):
            console.print(f"  cancelled reservation {res_id[:8]}...")
        else:
            console.print(f"  [yellow]could not cancel reservation {res_id}[/yellow]")

    backend.run(
        f"cd {ws}/pytorch && git worktree remove {job_dir}/pytorch --force",
        check=False,
    )
    backend.run(f"rm -rf {job_dir}", check=False)
    backend.run(f"cd {ws}/pytorch && git worktree prune", check=False)

    db = load_jobs_db()
    db.pop(job_id, None)
    save_jobs_db(db)
    issue_val = job.get("issue")
    label = f"issue #{issue_val}" if issue_val is not None else "adhoc"
    console.print(f"  removed {job_id} ({label})")


def _clean_machine(
    machine: str | None,
    local: bool,
    workspace: str | None,
    keep: int,
    include_running: bool,
) -> None:
    """Bulk clean jobs on a machine."""
    from ptq.ssh import LocalBackend, RemoteBackend, load_jobs_db, save_jobs_db

    if local:
        backend = LocalBackend(workspace=workspace or "~/.ptq_workspace")
    else:
        assert machine is not None
        backend = RemoteBackend(
            machine=machine, workspace=workspace or "~/ptq_workspace"
        )

    db = load_jobs_db()
    matching = [
        (jid, entry)
        for jid, entry in sorted(db.items())
        if (local and entry.get("local"))
        or (machine and entry.get("machine") == machine)
    ]

    if not include_running:
        running = []
        stopped = []
        for jid, entry in matching:
            pid = entry.get("pid")
            if pid and backend.is_pid_alive(pid):
                running.append((jid, entry))
            else:
                stopped.append((jid, entry))
        if running:
            console.print(
                f"Skipping {len(running)} running job(s). Use --all to include them."
            )
        matching = stopped

    to_remove = matching[:-keep] if keep and len(matching) > keep else matching
    if not to_remove:
        console.print("Nothing to clean.")
        return

    ws = backend.workspace
    console.print(f"Removing {len(to_remove)} job(s) (keeping {keep})...")
    backend.run(f"cd {ws}/pytorch && git worktree prune", check=False)

    for jid, entry in to_remove:
        pid = entry.get("pid")
        if pid and backend.is_pid_alive(pid):
            backend.kill_pid(pid)

        job_dir = f"{ws}/jobs/{jid}"
        backend.run(
            f"cd {ws}/pytorch && git worktree remove {job_dir}/pytorch --force",
            check=False,
        )
        backend.run(f"rm -rf {job_dir}")
        db.pop(jid, None)
        console.print(f"  removed {jid}")

    save_jobs_db(db)
    backend.run(f"cd {ws}/pytorch && git worktree prune", check=False)
    console.print("[bold green]Clean complete.[/bold green]")


@app.command()
def clean(
    target: Annotated[
        str | None, typer.Argument(help="Job ID, issue number, or machine name.")
    ] = None,
    local: Annotated[
        bool, typer.Option("--local", help="Clean local workspace.")
    ] = False,
    workspace: Annotated[
        str | None, typer.Option(help="Custom workspace path.")
    ] = None,
    keep: Annotated[int, typer.Option(help="Number of most recent jobs to keep.")] = 0,
    all_jobs: Annotated[
        bool, typer.Option("--all", help="Include running jobs.")
    ] = False,
) -> None:
    """Remove jobs: kill agent, delete remote files, drop from tracking DB.

    Pass a JOB_ID to clean a single job, or a MACHINE name to bulk clean.

    Examples:
        ptq clean 20260214-174923          # clean one job
        ptq clean 174923                   # clean by issue number
        ptq clean aws-gpu-dev              # clean all stopped jobs on machine
        ptq clean aws-gpu-dev --keep 2     # keep 2 most recent
        ptq clean aws-gpu-dev --all        # include running jobs
    """
    from ptq.job import resolve_job_id
    from ptq.ssh import load_jobs_db

    if target is not None:
        db = load_jobs_db()
        is_job = target in db or (
            target.isdigit() and any(v.get("issue") == int(target) for v in db.values())
        )
        if is_job:
            resolved = resolve_job_id(target)
            _clean_single_job(resolved)
            return

    if not target and not local:
        raise typer.BadParameter("Provide a job ID, machine name, or --local.")

    _clean_machine(
        machine=target,
        local=local,
        workspace=workspace,
        keep=keep,
        include_running=all_jobs,
    )


@app.command(name="list")
def list_jobs() -> None:
    """List all tracked jobs."""
    from rich.table import Table

    from ptq.ssh import backend_for_job, load_jobs_db

    db = load_jobs_db()
    if not db:
        console.print("No jobs.")
        return

    has_reservations = any(
        entry.get("reservation_id") for entry in db.values()
    )

    table = Table(
        show_header=True, header_style="bold", show_lines=False, pad_edge=False
    )
    table.add_column("Status", width=9)
    table.add_column("Job ID")
    table.add_column("Issue", style="cyan")
    table.add_column("Runs", justify="right")
    table.add_column("Target")
    if has_reservations:
        table.add_column("Reservation", style="dim")

    for job_id, entry in sorted(db.items()):
        issue_val = entry.get("issue")
        issue_display = f"#{issue_val}" if issue_val is not None else "[dim]adhoc[/dim]"
        runs = str(entry.get("runs", 1))
        target = entry.get("machine") or "local"
        pid = entry.get("pid")
        if pid:
            backend = backend_for_job(job_id)
            alive = backend.is_pid_alive(pid)
        else:
            alive = False
        status = "[bold green]running[/bold green]" if alive else "[dim]stopped[/dim]"
        row = [status, job_id, issue_display, runs, target]
        if has_reservations:
            res_id = entry.get("reservation_id", "")
            row.append(res_id[:8] if res_id else "")
        table.add_row(*row)

    console.print(table)
    console.print()
    console.print("[dim]Actions:[/dim]")
    console.print(
        "[dim]  ptq run --issue NUM --machine TARGET  # new run from issue[/dim]"
    )
    console.print("[dim]  ptq run -m 'task' --machine TARGET    # new adhoc run[/dim]")
    console.print(
        "[dim]  ptq run JOB_ID                        # re-run existing job[/dim]"
    )
    console.print(
        "[dim]  ptq run JOB_ID -m 'look at X instead' # re-run with steering[/dim]"
    )
    console.print("[dim]  ptq peek JOB_ID                       # check progress[/dim]")
    console.print("[dim]  ptq results JOB_ID                    # fetch results[/dim]")
    console.print("[dim]  ptq kill JOB_ID                       # stop agent[/dim]")
    console.print(
        "[dim]  ptq clean JOB_ID                      # remove job entirely[/dim]"
    )
    console.print(
        "[dim]  ptq clean MACHINE                     # bulk clean stopped jobs[/dim]"
    )


@app.command()
def peek(
    job_id: Annotated[str, typer.Argument(help="Job ID or issue number.")],
    log_lines: Annotated[
        int, typer.Option("--log", help="Number of log lines to show.")
    ] = 0,
) -> None:
    """Peek at an agent's progress (worklog + optional log tail)."""
    import json

    from rich.markdown import Markdown

    from ptq.job import get_job, resolve_job_id
    from ptq.ssh import backend_for_job

    job_id = resolve_job_id(job_id)
    job = get_job(job_id)
    backend = backend_for_job(job_id)
    ws = backend.workspace
    runs = job.get("runs", 1)
    pid = job.get("pid")
    target = job.get("machine") or "local"

    alive = pid is not None and backend.is_pid_alive(pid)
    status_str = "[bold green]running[/bold green]" if alive else "[dim]stopped[/dim]"
    issue_val = job.get("issue")
    issue_label = f"issue #{issue_val}" if issue_val is not None else "adhoc"
    console.print(f"{status_str}  {job_id}  {issue_label}  (run {runs}, {target})")
    console.print()

    worklog_path = f"{ws}/jobs/{job_id}/worklog.md"
    result = backend.run(f"cat {worklog_path}", check=False)
    if result.returncode == 0 and result.stdout.strip():
        console.print("[bold]Worklog[/bold]")
        console.print(Markdown(result.stdout))
    else:
        console.print("[yellow]No worklog yet.[/yellow]")

    if log_lines > 0:
        log_file = f"{ws}/jobs/{job_id}/claude-{runs}.log"
        tail_result = backend.run(f"tail -{log_lines} {log_file}", check=False)
        if tail_result.stdout.strip():
            console.print()
            console.print(f"[bold]Last {log_lines} log lines[/bold]")
            for line in tail_result.stdout.strip().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                    match event.get("type"):
                        case "assistant":
                            for block in event.get("message", {}).get("content", []):
                                if block.get("type") == "text" and block.get("text"):
                                    console.print(f"  [dim]{block['text'][:200]}[/dim]")
                                elif block.get("type") == "tool_use":
                                    console.print(
                                        f"  [cyan]{block.get('name', '')}[/cyan]"
                                    )
                        case _:
                            pass
                except (json.JSONDecodeError, ValueError):
                    console.print(f"  [dim]{line[:200]}[/dim]")


@app.command()
def status(
    job_id: Annotated[str, typer.Argument(help="Job ID or issue number.")],
) -> None:
    """Check if an agent is still running for a job."""
    from ptq.job import get_job, resolve_job_id
    from ptq.ssh import backend_for_job

    job_id = resolve_job_id(job_id)
    job = get_job(job_id)
    backend = backend_for_job(job_id)
    ws = backend.workspace

    pid = job.get("pid")
    runs = job.get("runs", 1)
    target = job.get("machine") or "local"

    alive = pid is not None and backend.is_pid_alive(pid)

    issue_val = job.get("issue")

    if alive:
        console.print(
            f"[bold green]running[/bold green]  {job_id}  (run {runs}, {target}, pid {pid})"
        )
        if issue_val is not None:
            console.print(
                f"  ptq run --issue {issue_val} --machine {target}  # to reattach"
            )
        else:
            console.print(f"  ptq run {job_id}  # to reattach")
    else:
        console.print(f"[bold dim]stopped[/bold dim]  {job_id}  (run {runs}, {target})")
        console.print(f"  ptq results {job_id}")

    log_file = f"{ws}/jobs/{job_id}/claude-{runs}.log"
    tail = backend.run(f"tail -1 {log_file}", check=False)
    if tail.stdout.strip():
        console.print(f"\n  last log: [dim]{tail.stdout.strip()[:120]}[/dim]")


@app.command()
def kill(
    job_id: Annotated[str, typer.Argument(help="Job ID or issue number.")],
) -> None:
    """Kill a running agent for a job."""
    from ptq.job import get_job, resolve_job_id, save_pid
    from ptq.ssh import backend_for_job

    job_id = resolve_job_id(job_id)
    job = get_job(job_id)
    backend = backend_for_job(job_id)

    pid = job.get("pid")
    if not pid:
        console.print(f"[dim]No PID tracked for {job_id}[/dim]")
        return

    if backend.is_pid_alive(pid):
        backend.kill_pid(pid)
        save_pid(job_id, None)
        console.print(f"[bold]Killed agent for {job_id} (pid {pid})[/bold]")
    else:
        save_pid(job_id, None)
        console.print(f"[dim]Agent already stopped for {job_id}[/dim]")


if __name__ == "__main__":
    app()
