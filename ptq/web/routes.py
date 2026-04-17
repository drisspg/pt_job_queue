from __future__ import annotations

import asyncio
import html
import logging
import re
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field

import markdown as md_lib
from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse
from sse_starlette.sse import EventSourceResponse

from ptq.agents import AGENTS, get_agent
from ptq.application.artifact_service import read_artifact
from ptq.config import (
    Config,
    cached_models,
    discover_models,
    discover_ssh_hosts,
    load_config,
)
from ptq.domain.models import PtqError, RebaseState, RunRequest
from ptq.infrastructure.backends import backend_for_job
from ptq.infrastructure.job_repository import JobRepository
from ptq.repo_profiles import available_repos, get_profile
from ptq.web.deps import get_job_status_with_finalize, templates

log = logging.getLogger("ptq.web")

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]|\r")
MAX_RESULT_LINES = 20
_PENDING_LAUNCH_TTL_SECONDS = 600.0
THINKING_LEVELS = ["", "off", "minimal", "low", "medium", "high", "xhigh"]


@dataclass
class PendingLaunch:
    params: dict[str, str | int]
    progress: list[str] = field(default_factory=list)
    done_url: str | None = None
    error: str | None = None
    task: asyncio.Task | None = None
    updated_at: float = field(default_factory=time.monotonic)
    finished_at: float | None = None
    wake_event: asyncio.Event = field(default_factory=asyncio.Event)


_pending_launches: dict[str, PendingLaunch] = {}


def _touch_launch(launch: PendingLaunch) -> None:
    launch.updated_at = time.monotonic()
    launch.wake_event.set()


def _prune_pending_launches() -> None:
    now = time.monotonic()
    expired = [
        launch_id
        for launch_id, launch in _pending_launches.items()
        if launch.finished_at is not None
        and now - launch.finished_at > _PENDING_LAUNCH_TTL_SECONDS
    ]
    for launch_id in expired:
        _pending_launches.pop(launch_id, None)


def _render_md(text: str) -> str:
    return md_lib.markdown(text, extensions=["fenced_code", "tables", "nl2br"])


def _repo() -> JobRepository:
    return JobRepository()


async def _run_pending_launch(launch: PendingLaunch) -> None:
    from ptq.application import run_service
    from ptq.infrastructure.backends import create_backend

    params = launch.params
    is_local = params["target_type"] == "local"
    machine_name = str(params["machine"]) or None
    issue_raw = str(params["issue"])
    issue_number = int(issue_raw) if issue_raw else None
    message_text = str(params["message"]) or None
    job_name = str(params.get("name", "")) or None

    backend = create_backend(machine=machine_name, local=is_local)
    loop = asyncio.get_running_loop()

    def on_progress(msg: str) -> None:
        loop.call_soon_threadsafe(_append_progress, msg)

    def _append_progress(msg: str) -> None:
        launch.progress.append(msg)
        _touch_launch(launch)

    try:
        issue_data = None
        repo_name = str(params.get("repo", "pytorch"))

        if issue_number is not None:
            from ptq.issue import fetch_issue

            profile = get_profile(repo_name)
            issue_data = await asyncio.to_thread(
                fetch_issue, issue_number, repo=profile.github_repo
            )

        run_request = RunRequest(
            issue_data=issue_data,
            issue_number=issue_number,
            message=message_text,
            machine=machine_name,
            local=is_local,
            follow=False,
            model=str(params["model"]),
            thinking=str(params.get("thinking", "")).strip() or None,
            max_turns=int(params["max_turns"]),
            agent_type=str(params["agent"]),
            name=job_name,
            repo=repo_name,
        )

        log.info(
            "creating job: agent=%s model=%s thinking=%s target=%s",
            params["agent"],
            params["model"],
            params.get("thinking", ""),
            "local" if is_local else machine_name,
        )
        job_id = await asyncio.to_thread(
            run_service.launch,
            _repo(),
            backend,
            run_request,
            on_progress=on_progress,
        )
    except Exception as exc:
        log.exception("launch failed: %s", exc)
        launch.error = str(exc)
        launch.finished_at = time.monotonic()
        _touch_launch(launch)
        return

    launch.done_url = f"/jobs/{job_id}"
    launch.finished_at = time.monotonic()
    _touch_launch(launch)
    _prune_pending_launches()


@dataclass
class PendingRebase:
    job_id: str
    target_ref: str
    agent_name: str | None = None
    model: str | None = None
    max_attempts: int = 3
    progress: list[str] = field(default_factory=list)
    done_url: str | None = None
    error: str | None = None
    task: asyncio.Task | None = None
    finished_at: float | None = None
    wake_event: asyncio.Event = field(default_factory=asyncio.Event)


_pending_rebases: dict[str, PendingRebase] = {}


def _touch_rebase(op: PendingRebase) -> None:
    op.wake_event.set()


def _prune_pending_rebases() -> None:
    now = time.monotonic()
    expired = [
        op_id
        for op_id, op in _pending_rebases.items()
        if op.finished_at is not None
        and now - op.finished_at > _PENDING_LAUNCH_TTL_SECONDS
    ]
    for op_id in expired:
        _pending_rebases.pop(op_id, None)


async def _run_pending_rebase(op: PendingRebase) -> None:
    from ptq.application.rebase_service import rebase as do_rebase

    loop = asyncio.get_running_loop()

    def on_progress(msg: str) -> None:
        loop.call_soon_threadsafe(_append_rebase_progress, msg)

    def _append_rebase_progress(msg: str) -> None:
        op.progress.append(msg)
        _touch_rebase(op)

    try:
        result = await asyncio.to_thread(
            do_rebase,
            _repo(),
            op.job_id,
            target_ref=op.target_ref,
            agent_name=op.agent_name,
            model=op.model,
            max_attempts=op.max_attempts,
            on_progress=on_progress,
        )
    except Exception as exc:
        log.exception("rebase failed: %s", exc)
        op.error = str(exc)
        op.finished_at = time.monotonic()
        _touch_rebase(op)
        return

    from ptq.domain.models import RebaseState

    if result.state == RebaseState.NEEDS_HUMAN:
        op.error = result.error
    op.done_url = f"/jobs/{op.job_id}"
    op.finished_at = time.monotonic()
    _touch_rebase(op)
    _prune_pending_rebases()


router = APIRouter()


@contextmanager
def _catch_error():
    try:
        yield
    except PtqError as e:
        raise HTTPException(status_code=404, detail=str(e)) from None


@router.get("/", response_class=RedirectResponse)
async def root():
    return RedirectResponse(url="/jobs", status_code=302)


@router.get("/jobs", response_class=HTMLResponse)
async def job_list(request: Request, status_filter: str = "all"):
    repo = _repo()
    all_jobs = repo.list_all()
    jobs: list[dict] = []

    for job_id, job in sorted(all_jobs.items()):
        status = get_job_status_with_finalize(job_id, job)
        if status_filter != "all" and status != status_filter:
            continue
        jobs.append(
            {
                "id": job_id,
                "issue": job.issue,
                "repo": job.repo,
                "agent": job.agent,
                "target": job.target,
                "runs": job.runs,
                "status": status,
                "pr_url": job.pr_url,
                "name": job.name,
            }
        )

    return templates.TemplateResponse(
        request,
        "job_list.html",
        {"jobs": jobs, "status_filter": status_filter},
    )


def _form_context(error: str | None = None) -> dict:
    cfg = load_config()
    agent_models = {}
    for name, am in cfg.agent_models.items():
        available = am.available or cached_models(name)
        agent_models[name] = {
            "available": available,
            "default": am.default,
            "thinking": am.thinking,
        }
    machines = list(dict.fromkeys(cfg.machines + discover_ssh_hosts()))

    return {
        "agents": list(AGENTS.keys()),
        "machines": machines,
        "agent_models": agent_models,
        "prompt_presets": _prompt_presets(cfg),
        "repos": [get_profile(r) for r in available_repos()],
        "thinking_levels": THINKING_LEVELS,
        "defaults": {
            "agent": cfg.default_agent,
            "model": cfg.default_model,
            "thinking": cfg.effective_thinking(cfg.default_agent) or "",
            "max_turns": cfg.default_max_turns,
        },
        "error": error,
    }


def _prompt_presets(cfg: Config) -> list[dict[str, str]]:
    return [
        {"key": preset.key, "title": preset.title, "body": preset.body}
        for preset in cfg.prompt_presets
    ]


@router.get("/jobs/new", response_class=HTMLResponse)
async def job_new(request: Request):
    return templates.TemplateResponse(request, "job_new.html", _form_context())


def _model_picker_html(models: list[str], default: str, agent_name: str) -> str:
    if models:
        options = "".join(
            f'<option value="{m}" {"selected" if m == default else ""}>{m}</option>'
            for m in models
        )
        select = f'<select id="model" name="model">{options}</select>'
    else:
        select = f'<input type="text" id="model" name="model" value="{default}" placeholder="e.g. opus">'
    refresh = (
        f'<button type="button" class="refresh-models" title="Refresh models" '
        f"onclick=\"refreshModels('{agent_name}')\">\u27f3</button>"
    )
    return f'<span class="model-picker-row">{select}{refresh}</span>'


@router.get("/api/models/{agent_name}", response_class=HTMLResponse)
async def agent_models_options(agent_name: str):
    cfg = load_config()
    am = cfg.models_for(agent_name)
    default_val = am.default or cfg.default_model
    models = am.available or cached_models(agent_name)
    return HTMLResponse(_model_picker_html(models, default_val, agent_name))


@router.post("/api/models/{agent_name}/refresh", response_class=HTMLResponse)
async def agent_models_refresh(agent_name: str):
    cfg = load_config()
    am = cfg.models_for(agent_name)
    default_val = am.default or cfg.default_model
    models = am.available or await asyncio.to_thread(discover_models, agent_name)
    return HTMLResponse(_model_picker_html(models, default_val, agent_name))


@router.post("/jobs", response_class=HTMLResponse)
async def job_create(
    request: Request,
    task_type: str = Form(...),
    issue: str = Form(""),
    message: str = Form(""),
    target_type: str = Form(...),
    machine: str = Form(""),
    agent: str = Form("claude"),
    model: str = Form("opus"),
    thinking: str = Form(""),
    max_turns: int = Form(100),
    name: str = Form(""),
    repo: str = Form("pytorch"),
):
    if task_type == "issue" and not issue.strip():
        return templates.TemplateResponse(
            request,
            "job_new.html",
            _form_context("Issue number is required."),
            status_code=422,
        )
    if task_type == "adhoc" and not message.strip():
        return templates.TemplateResponse(
            request,
            "job_new.html",
            _form_context("Message is required for ad-hoc tasks (or select an issue)."),
            status_code=422,
        )
    if target_type == "machine" and not machine.strip():
        return templates.TemplateResponse(
            request,
            "job_new.html",
            _form_context("Machine name is required."),
            status_code=422,
        )

    _prune_pending_launches()
    launch_id = uuid.uuid4().hex[:12]
    _pending_launches[launch_id] = PendingLaunch(
        params={
            "task_type": task_type,
            "issue": issue.strip(),
            "message": message.strip(),
            "target_type": target_type,
            "machine": machine.strip(),
            "agent": agent,
            "model": model,
            "thinking": thinking,
            "max_turns": max_turns,
            "name": name.strip(),
            "repo": repo.strip(),
        }
    )
    return RedirectResponse(url=f"/jobs/launching/{launch_id}", status_code=303)


@router.get("/jobs/launching/{launch_id}", response_class=HTMLResponse)
async def job_launching(request: Request, launch_id: str):
    _prune_pending_launches()
    if launch_id not in _pending_launches:
        raise HTTPException(status_code=404, detail="Launch not found")
    return templates.TemplateResponse(
        request, "job_launching.html", {"launch_id": launch_id}
    )


@router.get("/jobs/launching/{launch_id}/progress")
async def job_launch_progress(launch_id: str):
    _prune_pending_launches()
    launch = _pending_launches.get(launch_id)
    if launch is None:

        async def _not_found():
            yield {"event": "error", "data": "Launch not found."}

        return EventSourceResponse(_not_found())

    if launch.task is None:
        launch.task = asyncio.create_task(_run_pending_launch(launch))

    async def event_generator():
        next_index = 0
        while True:
            while next_index < len(launch.progress):
                msg = launch.progress[next_index]
                next_index += 1
                yield {"event": "progress", "data": html.escape(msg)}

            if launch.error is not None:
                yield {"event": "error", "data": html.escape(launch.error)}
                return
            if launch.done_url is not None:
                yield {"event": "done", "data": launch.done_url}
                return

            try:
                await asyncio.wait_for(launch.wake_event.wait(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            launch.wake_event.clear()

    return EventSourceResponse(event_generator())


@router.get("/jobs/{job_id}", response_class=HTMLResponse)
async def job_detail(request: Request, job_id: str):
    repo = _repo()
    with _catch_error():
        job_id = repo.resolve_id(job_id)
        job = repo.get(job_id)

    status = get_job_status_with_finalize(job_id, job)
    cfg = load_config()
    default_model = job.model or cfg.effective_model(job.agent)
    default_thinking = job.thinking or cfg.effective_thinking(job.agent) or ""
    am = cfg.models_for(job.agent)
    available_models = am.available or cached_models(job.agent)
    agent_models = {
        name: {
            "available": am.available or cached_models(name),
            "default": am.default,
            "thinking": am.thinking,
        }
        for name, am in cfg.agent_models.items()
    }
    pr_state = "unknown"
    if job.pr_url:
        from ptq.application.pr_service import get_pr_state

        pr_state = await asyncio.to_thread(
            get_pr_state, backend_for_job(job), job.pr_url
        )

    profile = get_profile(job.repo)

    rb = job.rebase_info
    if rb.state == RebaseState.SUCCEEDED:
        repo.save_rebase(job_id, {})
    return templates.TemplateResponse(
        request,
        "job_detail.html",
        {
            "job_id": job_id,
            "job_name": job.name or "",
            "status": status,
            "target": job.target,
            "issue": job.issue,
            "agent_name": job.agent,
            "default_model": default_model,
            "default_thinking": default_thinking,
            "available_models": available_models,
            "thinking_levels": THINKING_LEVELS,
            "agent_models": agent_models,
            "runs": job.runs,
            "agents": list(AGENTS.keys()),
            "pr_url": job.pr_url,
            "pr_state": pr_state,
            "human_note": job.human_note or "",
            "pr_title": job.pr_title or "",
            "workspace": job.workspace,
            "is_local": job.local,
            "rebase_state": rb.state.value,
            "rebase_target": rb.target_ref,
            "rebase_before": rb.before_sha,
            "rebase_after": rb.after_sha,
            "rebase_attempts": rb.attempts,
            "rebase_error": rb.error,
            "prompt_presets": _prompt_presets(cfg),
            "github_repo": profile.github_repo,
            "repo_name": job.repo,
            "dir_name": profile.dir_name,
        },
    )


@router.post("/jobs/{job_id}/rename", response_class=HTMLResponse)
async def job_rename(
    job_id: str,
    name: str = Form(""),
):
    repo = _repo()
    with _catch_error():
        job_id = repo.resolve_id(job_id)
    repo.save_name(job_id, name.strip() or None)
    return RedirectResponse(url=f"/jobs/{job_id}", status_code=303)


@router.post("/jobs/{job_id}/rerun", response_class=HTMLResponse)
async def job_rerun(
    request: Request,
    job_id: str,
    message: str = Form(""),
    agent_type: str = Form(""),
    model: str = Form(""),
    thinking: str = Form(""),
):
    repo = _repo()
    with _catch_error():
        job_id = repo.resolve_id(job_id)
        job = repo.get(job_id)

    from ptq.application import run_service

    cfg = load_config()
    backend = backend_for_job(job)
    agent_type = agent_type.strip() or job.agent
    model = model.strip() or cfg.effective_model(agent_type)
    thinking = cfg.effective_thinking(agent_type, thinking.strip() or None)

    log.info(
        "follow-up on %s: agent=%s model=%s thinking=%s message=%r",
        job_id,
        agent_type,
        model,
        thinking,
        message.strip()[:80],
    )

    issue_data = None
    if job.issue is not None:
        from ptq.issue import fetch_issue

        profile = get_profile(job.repo)
        issue_data = await asyncio.to_thread(
            fetch_issue, job.issue, repo=profile.github_repo
        )

    run_request = RunRequest(
        issue_data=issue_data,
        issue_number=job.issue,
        message=message.strip() or None,
        machine=job.machine,
        local=job.local,
        follow=False,
        model=model,
        thinking=thinking,
        max_turns=cfg.default_max_turns,
        agent_type=agent_type,
        existing_job_id=job_id,
        repo=job.repo,
    )

    await asyncio.to_thread(run_service.launch, repo, backend, run_request)

    log.info("follow-up launched for %s", job_id)
    return RedirectResponse(url=f"/jobs/{job_id}", status_code=303)


@router.post("/jobs/{job_id}/pr", response_class=HTMLResponse)
async def job_create_pr(
    request: Request,
    job_id: str,
    human_note: str = Form(""),
    title: str = Form(""),
    draft: str = Form(""),
):
    repo = _repo()
    with _catch_error():
        job_id = repo.resolve_id(job_id)

    if not human_note.strip():
        raise HTTPException(
            status_code=422,
            detail="A human note is required before creating a PR.",
        )

    from ptq.application.pr_service import create_pr

    log.info("creating PR for %s", job_id)
    result = await asyncio.to_thread(
        create_pr,
        repo,
        job_id,
        human_note=human_note,
        title=title.strip() or None,
        draft=bool(draft),
    )
    log.info("PR created for %s: %s", job_id, result.url)
    return RedirectResponse(url=f"/jobs/{job_id}", status_code=303)


@router.post("/jobs/{job_id}/kill", response_class=HTMLResponse)
async def job_kill(request: Request, job_id: str):
    repo = _repo()
    with _catch_error():
        job_id = repo.resolve_id(job_id)

    from ptq.application.job_service import kill_job

    kill_job(repo, job_id)
    log.info("killed %s", job_id)
    return RedirectResponse(url=f"/jobs/{job_id}", status_code=303)


@router.delete("/jobs/{job_id}", response_class=HTMLResponse)
async def job_delete(job_id: str):
    repo = _repo()
    with _catch_error():
        job_id = repo.resolve_id(job_id)

    from ptq.application.job_service import clean_single_job

    log.info("cleaning %s", job_id)
    await asyncio.to_thread(clean_single_job, repo, job_id)
    return HTMLResponse("", headers={"HX-Redirect": "/jobs"})


@router.post("/jobs/{job_id}/clean", response_class=HTMLResponse)
async def job_clean(job_id: str):
    repo = _repo()
    with _catch_error():
        job_id = repo.resolve_id(job_id)

    from ptq.application.job_service import clean_single_job

    log.info("cleaning %s", job_id)
    await asyncio.to_thread(clean_single_job, repo, job_id)
    return RedirectResponse(url="/jobs", status_code=303)


@router.post("/jobs/{job_id}/rebase", response_class=HTMLResponse)
async def job_rebase(
    job_id: str,
    target_ref: str = Form("origin/main"),
    agent_type: str = Form(""),
    model: str = Form(""),
    max_attempts: int = Form(3),
):
    repo = _repo()
    with _catch_error():
        job_id = repo.resolve_id(job_id)

    _prune_pending_rebases()
    op_id = uuid.uuid4().hex[:12]
    _pending_rebases[op_id] = PendingRebase(
        job_id=job_id,
        target_ref=target_ref.strip() or "origin/main",
        agent_name=agent_type.strip() or None,
        model=model.strip() or None,
        max_attempts=max_attempts,
    )
    log.info("rebase requested for %s onto %s (op=%s)", job_id, target_ref, op_id)
    return RedirectResponse(url=f"/jobs/rebasing/{op_id}", status_code=303)


@router.get("/jobs/rebasing/{op_id}", response_class=HTMLResponse)
async def job_rebasing_page(request: Request, op_id: str):
    _prune_pending_rebases()
    if op_id not in _pending_rebases:
        raise HTTPException(status_code=404, detail="Rebase operation not found")
    op = _pending_rebases[op_id]
    return templates.TemplateResponse(
        request, "job_rebasing.html", {"op_id": op_id, "job_id": op.job_id}
    )


@router.get("/jobs/rebasing/{op_id}/progress")
async def job_rebase_progress(op_id: str):
    _prune_pending_rebases()
    op = _pending_rebases.get(op_id)
    if op is None:

        async def _not_found():
            yield {"event": "error", "data": "Rebase operation not found."}

        return EventSourceResponse(_not_found())

    if op.task is None:
        op.task = asyncio.create_task(_run_pending_rebase(op))

    async def event_generator():
        next_index = 0
        while True:
            while next_index < len(op.progress):
                msg = op.progress[next_index]
                next_index += 1
                yield {"event": "progress", "data": html.escape(msg)}

            if op.error is not None and op.done_url is not None:
                yield {"event": "warning", "data": html.escape(op.error)}
                yield {"event": "done", "data": op.done_url}
                return
            if op.error is not None:
                yield {"event": "error", "data": html.escape(op.error)}
                return
            if op.done_url is not None:
                yield {"event": "done", "data": op.done_url}
                return

            try:
                await asyncio.wait_for(op.wake_event.wait(), timeout=0.5)
            except asyncio.TimeoutError:
                continue
            op.wake_event.clear()

    return EventSourceResponse(event_generator())


@router.get("/jobs/{job_id}/status", response_class=HTMLResponse)
async def job_status_badge(request: Request, job_id: str):
    repo = _repo()
    with _catch_error():
        job = repo.get(job_id)
    status = get_job_status_with_finalize(job_id, job)
    return templates.TemplateResponse(
        request,
        "partials/status_badge.html",
        {"status": status, "job_id": job_id},
    )


@router.get("/jobs/{job_id}/report", response_class=HTMLResponse)
async def job_report(job_id: str):
    repo = _repo()
    with _catch_error():
        job = repo.get(job_id)
    backend = backend_for_job(job)
    content = read_artifact(backend, f"{backend.workspace}/jobs/{job_id}/report.md")
    return HTMLResponse(
        f'<div class="rendered-md">{_render_md(content)}</div>'
        if content
        else '<p class="muted">No report yet.</p>'
    )


@router.get("/jobs/{job_id}/diff")
async def job_diff(job_id: str):
    repo = _repo()
    with _catch_error():
        job = repo.get(job_id)
    backend = backend_for_job(job)
    profile = get_profile(job.repo)
    worktree = f"{backend.workspace}/jobs/{job_id}/{profile.dir_name}"

    # Show all changes: committed (branch vs merge-base with main) + uncommitted
    # Use a single shell command to avoid multiple SSH round-trips
    result = backend.run(
        f"cd {worktree} && "
        f"mb=$(git merge-base HEAD origin/main 2>/dev/null) && "
        f"git -c color.diff=never diff --no-color --no-ext-diff $mb",
        check=False,
    )
    content = result.stdout.strip() if result.returncode == 0 else None
    if not content:
        # merge-base failed or no committed changes — try plain diff (uncommitted only)
        result = backend.run(
            f"cd {worktree} && git -c color.diff=never diff --no-color --no-ext-diff",
            check=False,
        )
        content = result.stdout.strip() if result.returncode == 0 else None
    if not content:
        log.warning(
            "Empty diff for job %s (worktree=%s, stderr=%s)",
            job_id,
            worktree,
            result.stderr.strip() if result.stderr else "",
        )
        return PlainTextResponse("")
    return PlainTextResponse(content)


@router.get("/jobs/{job_id}/repro", response_class=HTMLResponse)
async def job_repro(job_id: str):
    repo = _repo()
    with _catch_error():
        job = repo.get(job_id)
    backend = backend_for_job(job)
    content = read_artifact(backend, f"{backend.workspace}/jobs/{job_id}/repro.py")
    if not content:
        return HTMLResponse('<p class="muted">No repro script found.</p>')
    escaped = html.escape(content)
    return HTMLResponse(f'<pre class="log-viewer"><code>{escaped}</code></pre>')


@router.get("/jobs/{job_id}/worklog", response_class=HTMLResponse)
async def job_worklog(job_id: str):
    repo = _repo()
    with _catch_error():
        job = repo.get(job_id)
    backend = backend_for_job(job)
    content = read_artifact(backend, f"{backend.workspace}/jobs/{job_id}/worklog.md")
    return HTMLResponse(
        f'<div class="rendered-md">{_render_md(content)}</div>'
        if content
        else '<p class="muted">No worklog yet.</p>'
    )


_TOOL_DISPLAY = {
    "Bash": ("$", "command", ""),
    "Read": ("read", "file_path|path", ""),
    "Edit": ("edit", "file_path|path", " edit"),
    "Write": ("write", "file_path|path", " write"),
    "Grep": ("grep", "pattern", ""),
    "Glob": ("glob", "pattern", ""),
}


def _render_event_html(ev) -> str:
    match ev.kind:
        case "text":
            text = ANSI_RE.sub("", ev.text).strip()
            if not text:
                return ""
            return f'<span class="log-text">{html.escape(text)}</span>\n'
        case "tool_use":
            if ev.tool_name in _TOOL_DISPLAY:
                label, keys, css_extra = _TOOL_DISPLAY[ev.tool_name]
                detail = ""
                for key in keys.split("|"):
                    detail = ev.tool_input.get(key, "")
                    if detail:
                        break
                return f'<span class="log-tool"><span class="log-tool-name{css_extra}">{label}</span> {html.escape(detail)}</span>\n'
            return f'<span class="log-tool"><span class="log-tool-name">{html.escape(ev.tool_name)}</span></span>\n'
        case "tool_result":
            text = ANSI_RE.sub("", ev.text).strip()
            if not text:
                return ""
            lines = text.splitlines()
            if len(lines) > MAX_RESULT_LINES:
                text = (
                    "\n".join(lines[:MAX_RESULT_LINES])
                    + f"\n… ({len(lines) - MAX_RESULT_LINES} more lines)"
                )
            return f'<span class="log-result">{html.escape(text)}</span>\n'
        case "error":
            return f'<span class="log-error">{html.escape(ev.text.strip())}</span>\n'
    return ""


def _format_log_line(line: str, agent_parser) -> str:
    stripped = ANSI_RE.sub("", line.strip())
    if not stripped:
        return ""
    if stripped.startswith("{"):
        try:
            events = agent_parser.parse_stream_line(stripped)
            return "".join(_render_event_html(ev) for ev in events)
        except (ValueError, KeyError):
            pass
    return f"<span>{html.escape(stripped)}</span>\n"


@router.get("/jobs/{job_id}/logs/stream")
async def stream_logs(job_id: str):
    repo = _repo()
    with _catch_error():
        job_id = repo.resolve_id(job_id)
        job = repo.get(job_id)

    backend = backend_for_job(job)
    ws = backend.workspace
    agent = get_agent(job.agent)
    log_file = f"{ws}/jobs/{job_id}/{agent.log_filename(job.runs)}"
    is_alive = job.pid is not None and backend.is_pid_alive(job.pid)
    log.debug(
        "stream_logs %s: log_file=%s pid=%s alive=%s",
        job_id,
        log_file,
        job.pid,
        is_alive,
    )

    async def _file_exists() -> bool:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: backend.run(f"test -f {log_file}", check=False).returncode == 0,
        )

    async def event_generator():
        loop = asyncio.get_event_loop()
        if is_alive:
            for _ in range(30):
                if await _file_exists():
                    break
                await asyncio.sleep(1)
            else:
                if not await _file_exists():
                    log.warning("log file never appeared: %s", log_file)
                    yield {"event": "done", "data": "<em>Log file not found.</em>"}
                    return

            proc = backend.tail_log(log_file)
            try:
                while True:
                    line = await loop.run_in_executor(None, proc.stdout.readline)
                    if not line:
                        break
                    if line.rstrip().startswith("tail:"):
                        continue
                    rendered = _format_log_line(line, agent)
                    if rendered:
                        yield {"event": "log", "data": rendered}
            finally:
                proc.terminate()
                proc.wait()
        else:
            if not await _file_exists():
                log.debug("no log file for stopped job %s: %s", job_id, log_file)
                yield {"event": "done", "data": "<em>No logs available.</em>"}
                return
            result = await loop.run_in_executor(
                None, lambda: backend.run(f"cat {log_file}", check=False)
            )
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    rendered = _format_log_line(line, agent)
                    if rendered:
                        yield {"event": "log", "data": rendered}
            yield {"event": "done", "data": "<em>— end of log —</em>"}

    return EventSourceResponse(event_generator())
