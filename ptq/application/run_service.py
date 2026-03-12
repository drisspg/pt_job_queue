from __future__ import annotations

import logging
import tempfile
import time
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path

from ptq.agent import (
    DEFAULT_MESSAGE,
    build_adhoc_prompt,
    build_system_prompt,
)
from ptq.agents import RunContext, get_agent
from ptq.domain.models import JobRecord, PtqError, RunRequest
from ptq.domain.policies import make_job_id
from ptq.infrastructure.job_repository import JobRepository
from ptq.issue import extract_repro_script
from ptq.ssh import Backend, RemoteBackend
from ptq.workspace import deploy_scripts

log = logging.getLogger("ptq.run")

ProgressCallback = Callable[[str], None]


def _noop_progress(_msg: str) -> None:
    pass


@contextmanager
def _timed(label: str, progress: ProgressCallback):
    t0 = time.monotonic()
    yield
    progress(f"  {label}: {time.monotonic() - t0:.1f}s")


def _validate_workspace(backend: Backend, workspace: str) -> None:
    result = backend.run(f"test -d {workspace}/pytorch/.git", check=False)
    if result.returncode != 0:
        raise PtqError(
            f"Workspace broken: {workspace}/pytorch/.git missing. Re-run: ptq setup"
        )


def _try_clone_base_venv(
    backend: Backend,
    job_dir: str,
    worktree_path: str,
    *,
    verbose: bool = False,
    progress: ProgressCallback = _noop_progress,
) -> bool:
    """Clone base workspace venv + source artifacts instead of rebuilding.

    Copies the venv (hardlinks), rsyncs gitignored source artifacts
    (.so, generated .py) into the worktree, and rewrites editable-install
    paths.  Skips the cmake build dir — if the agent needs to rebuild C++,
    ccache handles it (~3 min with warm cache).
    Falls back to the slow editable-install if anything fails.
    """
    workspace = backend.workspace
    base_venv = f"{workspace}/.venv"

    def _last_line(cmd: str) -> str:
        lines = backend.run(cmd, check=False).stdout.strip().splitlines()
        return lines[-1] if lines else ""

    with _timed("path resolution", progress):
        old_src = _last_line(f"realpath {workspace}/pytorch")
        new_src = _last_line(f"realpath {worktree_path}")
    if not old_src or not new_src or old_src == new_src:
        log.info("fast-path skipped: path resolution failed or same path")
        return False

    with _timed("base torch check", progress):
        has_torch = backend.run(
            f"cd /tmp && {base_venv}/bin/python -c 'import torch' 2>/dev/null",
            check=False,
        ).returncode
    if has_torch != 0:
        log.info("fast-path skipped: base venv has no torch")
        return False

    log.info("fast-path clone: %s -> %s", old_src, new_src)
    progress("Cloning base venv (fast path)...")
    with _timed("venv clone", progress):
        for cp_flags in ("-al", "-a"):
            if (
                backend.run(
                    f"cp {cp_flags} {base_venv} {job_dir}/.venv", check=False
                ).returncode
                == 0
            ):
                break
            backend.run(f"rm -rf {job_dir}/.venv", check=False)
        else:
            return False

    def _bail() -> bool:
        backend.run(f"rm -rf {job_dir}/.venv", check=False)
        backend.run(f"cd {new_src} && git clean -fdx 2>/dev/null", check=False)
        return False

    progress("Syncing build artifacts into worktree...")
    with _timed("artifact sync", progress):
        r = backend.run(
            f"rsync -a --ignore-existing --exclude='.git' --exclude='build' "
            f"--exclude='__pycache__' --link-dest={old_src} {old_src}/ {new_src}/",
            check=False,
        )
        if r.returncode not in (0, 23):
            progress(f"fast-path bail: rsync failed (rc={r.returncode})")
            return _bail()

    # rsync --link-dest hardlinks .so files to the base workspace.  A later
    # rebuild.sh (pip install -e .) may skip overwriting them if setuptools
    # sees matching size/mtime, leaving the job on stale native code.
    # Break hardlinks now so rebuilds always produce independent copies.
    backend.run(
        f"find {new_src}/torch -name '*.so' -links +1 "
        f"-exec cp --remove-destination {{}} {{}}.tmp \\; "
        f"-exec mv -f {{}}.tmp {{}} \\;",
        check=False,
    )

    job_python = f"{job_dir}/.venv/bin/python"
    sp_dir = _last_line(
        f"{job_python} -c 'import sysconfig; print(sysconfig.get_path(\"purelib\"))'"
    )
    if not sp_dir:
        progress("fast-path bail: could not resolve site-packages dir")
        return _bail()

    progress("Rewriting venv paths...")
    with _timed("path rewrite", progress):
        job_venv = f"{job_dir}/.venv"
        resolved_venv = _last_line(f"realpath {job_venv}") or job_venv
        backend.run(
            f'sed -i "s|{base_venv}|{resolved_venv}|g" {job_venv}/bin/activate {job_venv}/bin/activate.csh {job_venv}/bin/activate.fish {job_venv}/bin/activate.nu 2>/dev/null',
            check=False,
        )
        backend.run(
            f"""sed -i "s|^VIRTUAL_ENV=.*|VIRTUAL_ENV='{resolved_venv}'|" {job_venv}/bin/activate 2>/dev/null""",
            check=False,
        )
        backend.run(
            f"""sed -i 's|^setenv VIRTUAL_ENV .*|setenv VIRTUAL_ENV "{resolved_venv}"|' {job_venv}/bin/activate.csh 2>/dev/null""",
            check=False,
        )
        backend.run(
            f"""sed -i 's|^set -gx VIRTUAL_ENV .*|set -gx VIRTUAL_ENV "{resolved_venv}"|' {job_venv}/bin/activate.fish 2>/dev/null""",
            check=False,
        )
        backend.run(
            f'sed -i "1s|#!{base_venv}/bin/python[0-9.]*|#!{resolved_venv}/bin/python|" {job_venv}/bin/* 2>/dev/null',
            check=False,
        )

        backend.run(
            f"for f in {sp_dir}/__editable__*torch* {sp_dir}/torch*.dist-info/direct_url.json; do "
            f'[ -f "$f" ] && sed -i "s|{old_src}|{new_src}|g" "$f"; done',
            check=False,
        )
        backend.run(f"rm -f {sp_dir}/__pycache__/__editable__*torch*.pyc", check=False)

    progress("Installing dev deps (build + test)...")
    with _timed("dev deps", progress):
        r = backend.run(
            f"cd {worktree_path} && "
            f"uv pip install --python {job_python} -r requirements.txt pytest",
            check=False,
            stream=verbose,
        )
        if r.returncode != 0:
            progress(f"fast-path bail: dev deps install failed (rc={r.returncode})")
            return _bail()

    progress("Verifying torch import...")
    with _timed("smoke test", progress):
        smoke = backend.run(
            f"cd /tmp && {job_python} -c "
            f"'import torch; print(torch.__file__, torch.__version__, torch.cuda.is_available())'",
            check=False,
        )
    if smoke.returncode != 0 or new_src not in smoke.stdout:
        reason = (
            f"rc={smoke.returncode} got={smoke.stdout.strip()!r} "
            f"stderr={smoke.stderr.strip()!r} expected={new_src}"
        )
        progress(f"fast-path bail: smoke test failed: {reason}")
        progress(f"Clone verification failed: {reason}")
        progress("Falling back to full install.")
        return _bail()

    log.info("fast-path complete: %s", smoke.stdout.strip())
    progress(f"Editable install complete (cloned) — {smoke.stdout.strip()}")
    return True


def _install_triton(
    backend: Backend,
    job_dir: str,
    worktree_path: str,
    *,
    verbose: bool = False,
    progress: ProgressCallback = _noop_progress,
) -> None:
    progress("Installing Triton...")
    with _timed("triton install", progress):
        r = backend.run(
            f"cd {worktree_path} && PATH={job_dir}/.venv/bin:$PATH make triton",
            check=False,
            stream=verbose,
        )
    if r.returncode != 0:
        progress("Triton install failed (non-fatal) — agent can install manually.")
    else:
        progress("Triton installed.")


def _setup_job_venv(
    backend: Backend,
    job_dir: str,
    worktree_path: str,
    *,
    verbose: bool = False,
    progress: ProgressCallback = _noop_progress,
    build_env_prefix: str = "USE_NINJA=1 ",
) -> None:
    if not _try_clone_base_venv(
        backend, job_dir, worktree_path, verbose=verbose, progress=progress
    ):
        log.info("slow-path: full editable install for %s", job_dir)
        with _timed("venv creation", progress):
            backend.run(f"cd {job_dir} && uv venv --python 3.12")

        job_python = f"{job_dir}/.venv/bin/python"
        progress("Installing dev deps (build + test)...")
        with _timed("dev deps", progress):
            backend.run(
                f"cd {worktree_path} && "
                f"uv pip install --python {job_python} -r requirements.txt pytest",
                stream=verbose,
            )
        pip_verbose = " -v" if verbose else ""
        pip_cmd = f"uv pip install --python {job_python} --no-build-isolation{pip_verbose} -e ."
        re_cc_cfg = f"{backend.workspace}/.re-cc-config"
        re_cc_check = backend.run(f"cat {re_cc_cfg}", check=False)
        if re_cc_check.returncode == 0 and re_cc_check.stdout.strip().isdigit():
            re_cc_jobs = re_cc_check.stdout.strip()
            pip_cmd = f"re-cc -- {pip_cmd}"
            build_env_prefix = f"MAX_JOBS={re_cc_jobs} {build_env_prefix}"
            progress(f"Using re-cc with MAX_JOBS={re_cc_jobs}")
        progress("Editable install (pytorch)... this takes a few minutes")
        with _timed("editable install", progress):
            result = backend.run(
                f"cd {worktree_path} && {build_env_prefix}{pip_cmd}",
                check=False,
                stream=verbose,
            )
        if result.returncode != 0:
            progress("Editable install failed — agent will need to build manually.")
        else:
            progress("Editable install complete.")

    _install_triton(backend, job_dir, worktree_path, verbose=verbose, progress=progress)


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


def launch(
    repo: JobRepository,
    backend: Backend,
    request: RunRequest,
    *,
    on_progress: ProgressCallback | None = None,
) -> str:
    """Launch an agent job. Returns job_id."""
    progress = on_progress or _noop_progress
    agent = get_agent(request.agent_type)
    workspace = backend.workspace
    is_adhoc = request.issue_number is None

    if request.existing_job_id:
        job_id = request.existing_job_id
        run_number = repo.increment_run(
            job_id, agent_type=request.agent_type, model=request.model
        )
        label = f"issue #{request.issue_number}" if request.issue_number else "adhoc"
        progress(f"Job {job_id} — {label} (run {run_number})")
        existing = job_id
    elif is_adhoc:
        existing = None
        job_id = make_job_id(message=request.message)
        run_number = 1
        progress(f"Job {job_id} — adhoc (run 1)")
    else:
        existing = repo.find_by_issue(
            request.issue_number, machine=request.machine, local=request.local
        )
        if existing:
            job_id = existing
            run_number = repo.increment_run(
                job_id, agent_type=request.agent_type, model=request.model
            )
            progress(f"Job {job_id} — issue #{request.issue_number} (run {run_number})")
        else:
            job_id = make_job_id(request.issue_number)
            run_number = 1
            progress(f"Job {job_id} — issue #{request.issue_number} (run 1)")

    job_dir = f"{workspace}/jobs/{job_id}"
    worktree_path = f"{job_dir}/pytorch"

    if existing:
        _validate_workspace(backend, workspace)

    backend.run(f"mkdir -p {job_dir}")

    if not existing:
        repo.save(
            JobRecord(
                job_id=job_id,
                issue=request.issue_number,
                runs=run_number,
                agent=request.agent_type,
                model=request.model,
                machine=request.machine,
                local=request.local,
                workspace=workspace,
                initializing=True,
                name=request.name,
            )
        )
    elif request.name:
        repo.save_name(job_id, request.name)

    deploy_scripts(backend)

    worktree_exists = backend.run(
        f"test -d {worktree_path}/.git || test -f {worktree_path}/.git", check=False
    )
    if worktree_exists.returncode != 0:
        progress("Creating worktree with submodules...")
        with _timed("worktree creation", progress):
            backend.run(
                f"cd {workspace}/pytorch && {workspace}/.venv/bin/python tools/create_worktree.py create pytorch "
                f"--parent-dir {job_dir} --commit HEAD",
                stream=request.verbose,
            )
        progress("Creating per-job venv...")
        from ptq.config import load_config

        _setup_job_venv(
            backend,
            job_dir,
            worktree_path,
            verbose=request.verbose,
            progress=progress,
            build_env_prefix=load_config().build_env_prefix(),
        )
    else:
        progress("Reusing existing worktree.")

    if is_adhoc:
        system_prompt = build_adhoc_prompt(request.message, job_id, workspace)
    else:
        system_prompt = build_system_prompt(
            request.issue_data, request.issue_number, job_id, workspace
        )

    if existing:
        prior_context = _build_prior_context(backend, job_dir, run_number)
        if prior_context:
            system_prompt += prior_context
            progress("Loaded prior run context (worklog/report).")

    _stamp_worklog_header(backend, job_dir, run_number, request.message)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(system_prompt)
        prompt_tmp = Path(f.name)

    prompt_remote = f"{job_dir}/system_prompt.md"
    backend.copy_to(prompt_tmp, prompt_remote)
    prompt_tmp.unlink()

    progress("Configuring agent workspace...")
    agent.setup_workspace(backend, worktree_path, job_dir, workspace, prompt_remote)

    if not is_adhoc:
        repro = extract_repro_script(request.issue_data)
        if repro:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(repro)
                repro_tmp = Path(f.name)
            backend.copy_to(repro_tmp, f"{job_dir}/repro.py")
            repro_tmp.unlink()
            progress("Extracted and uploaded repro script.")
        else:
            progress("No repro script found in issue — agent will write one.")

    if is_adhoc:
        agent_message = request.message
    elif existing:
        agent_message = request.message or DEFAULT_MESSAGE
    elif request.message:
        agent_message = f"{DEFAULT_MESSAGE}\n\nAdditional context: {request.message}"
    else:
        agent_message = DEFAULT_MESSAGE

    log_file = f"{job_dir}/{agent.log_filename(run_number)}"
    unbuffer = "stdbuf -oL " if isinstance(backend, RemoteBackend) else ""
    ctx = RunContext(
        worktree_path=worktree_path,
        job_dir=job_dir,
        message=agent_message,
        model=request.model,
        max_turns=request.max_turns,
        system_prompt_file=prompt_remote,
        unbuffer_prefix=unbuffer,
    )
    agent_cmd = agent.build_cmd(ctx)

    progress(
        f"Launching {agent.name} agent ({'local' if request.local else request.machine})..."
    )
    backend.run(f"mkdir -p {job_dir}/agent_logs && touch {log_file}")
    pid = backend.launch_background(agent_cmd, log_file)
    repo.save_pid(job_id, pid)

    return job_id


def finalize_run(backend: Backend, job_id: str, job: JobRecord) -> None:
    """Extract agent summary from log and append to worklog if the agent didn't."""
    ws = backend.workspace
    job_dir = f"{ws}/jobs/{job_id}"
    run_number = job.runs
    agent = get_agent(job.agent)
    log_file = f"{job_dir}/{agent.log_filename(run_number)}"

    worklog_result = backend.run(f"cat {job_dir}/worklog.md", check=False)
    if worklog_result.returncode != 0:
        return

    run_header = f"## Run {run_number}"
    worklog_text = worklog_result.stdout
    header_pos = worklog_text.rfind(run_header)
    if header_pos == -1:
        return

    next_header = worklog_text.find("\n## Run ", header_pos + len(run_header))
    section = (
        worklog_text[header_pos:next_header]
        if next_header != -1
        else worklog_text[header_pos:]
    )

    for line in section.splitlines()[1:]:
        stripped = line.strip()
        if stripped and not stripped.startswith("> **User:**"):
            return

    log_result = backend.run(f"cat {log_file}", check=False)
    if log_result.returncode != 0 or not log_result.stdout.strip():
        return

    summary = agent.extract_summary(log_result.stdout)
    if not summary:
        return

    entry_text = f"\n### Agent Summary (auto-extracted)\n{summary}\n"
    backend.run(
        f"cat >> {job_dir}/worklog.md << 'WORKLOG_AUTO_EOF'\n{entry_text}\nWORKLOG_AUTO_EOF"
    )
