from __future__ import annotations

import asyncio
import html
import logging
import re
import uuid
from contextlib import contextmanager

import markdown as md_lib
from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sse_starlette.sse import EventSourceResponse

from ptq.agents import AGENTS, get_agent
from ptq.application.artifact_service import read_artifact
from ptq.config import cached_models, discover_models, discover_ssh_hosts, load_config
from ptq.domain.models import PtqError, RunRequest
from ptq.infrastructure.backends import backend_for_job
from ptq.infrastructure.job_repository import JobRepository
from ptq.web.deps import get_job_status_with_finalize, templates

log = logging.getLogger("ptq.web")

_pending_launches: dict[str, dict] = {}

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]|\r")
MAX_RESULT_LINES = 20


def _render_md(text: str) -> str:
    return md_lib.markdown(text, extensions=["fenced_code", "tables", "nl2br"])


def _repo() -> JobRepository:
    return JobRepository()


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


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    repo = _repo()
    all_jobs = repo.list_all()
    jobs: list[dict] = []
    running_count = 0
    stopped_count = 0
    machines: dict[str, dict] = {}

    for job_id, job in sorted(all_jobs.items()):
        status = get_job_status_with_finalize(job_id, job)
        if status in ("running", "initializing"):
            running_count += 1
        else:
            stopped_count += 1

        machines.setdefault(job.target, {"running": 0, "stopped": 0, "total": 0})
        machines[job.target][
            "running" if status in ("running", "initializing") else "stopped"
        ] += 1
        machines[job.target]["total"] += 1

        jobs.append(
            {
                "id": job_id,
                "issue": job.issue,
                "agent": job.agent,
                "target": job.target,
                "runs": job.runs,
                "status": status,
            }
        )

    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {
            "jobs": jobs,
            "total": len(all_jobs),
            "running": running_count,
            "stopped": stopped_count,
            "machines": machines,
        },
    )


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
                "agent": job.agent,
                "target": job.target,
                "runs": job.runs,
                "status": status,
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
        agent_models[name] = {"available": available, "default": am.default}
    machines = list(dict.fromkeys(cfg.machines + discover_ssh_hosts()))
    return {
        "agents": list(AGENTS.keys()),
        "machines": machines,
        "agent_models": agent_models,
        "defaults": {
            "agent": cfg.default_agent,
            "model": cfg.default_model,
            "max_turns": cfg.default_max_turns,
        },
        "error": error,
    }


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
    max_turns: int = Form(100),
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

    launch_id = uuid.uuid4().hex[:12]
    _pending_launches[launch_id] = {
        "task_type": task_type,
        "issue": issue.strip(),
        "message": message.strip(),
        "target_type": target_type,
        "machine": machine.strip(),
        "agent": agent,
        "model": model,
        "max_turns": max_turns,
    }
    return RedirectResponse(url=f"/jobs/launching/{launch_id}", status_code=303)


@router.get("/jobs/launching/{launch_id}", response_class=HTMLResponse)
async def job_launching(request: Request, launch_id: str):
    if launch_id not in _pending_launches:
        raise HTTPException(status_code=404, detail="Launch not found")
    return templates.TemplateResponse(
        request, "job_launching.html", {"launch_id": launch_id}
    )


@router.get("/jobs/launching/{launch_id}/progress")
async def job_launch_progress(launch_id: str):
    params = _pending_launches.pop(launch_id, None)
    if params is None:

        async def _not_found():
            yield {"event": "error", "data": "Launch not found."}

        return EventSourceResponse(_not_found())

    progress_queue: asyncio.Queue[str] = asyncio.Queue()

    def on_progress(msg: str):
        progress_queue.put_nowait(msg)

    async def run_launch():
        from ptq.application import run_service
        from ptq.infrastructure.backends import create_backend

        is_local = params["target_type"] == "local"
        machine_name = params["machine"] or None
        issue_number = int(params["issue"]) if params["issue"] else None
        message_text = params["message"] or None

        backend = create_backend(machine=machine_name, local=is_local)

        issue_data = None
        if issue_number is not None:
            from ptq.issue import fetch_issue

            issue_data = await asyncio.to_thread(fetch_issue, issue_number)

        request = RunRequest(
            issue_data=issue_data,
            issue_number=issue_number,
            message=message_text,
            machine=machine_name,
            local=is_local,
            follow=False,
            model=params["model"],
            max_turns=params["max_turns"],
            agent_type=params["agent"],
        )

        log.info(
            "creating job: agent=%s model=%s target=%s",
            params["agent"],
            params["model"],
            "local" if is_local else machine_name,
        )
        return await asyncio.to_thread(
            run_service.launch,
            _repo(),
            backend,
            request,
            on_progress=on_progress,
        )

    async def event_generator():
        task = asyncio.create_task(run_launch())
        while not task.done():
            try:
                msg = await asyncio.wait_for(progress_queue.get(), timeout=0.5)
                yield {"event": "progress", "data": html.escape(msg)}
            except asyncio.TimeoutError:
                continue
        while not progress_queue.empty():
            msg = progress_queue.get_nowait()
            yield {"event": "progress", "data": html.escape(msg)}

        exc = task.exception()
        if exc:
            log.exception("launch failed: %s", exc)
            yield {"event": "error", "data": html.escape(str(exc))}
        else:
            job_id = task.result()
            yield {"event": "done", "data": f"/jobs/{job_id}"}

    return EventSourceResponse(event_generator())


@router.get("/jobs/{job_id}", response_class=HTMLResponse)
async def job_detail(request: Request, job_id: str, pr_url: str | None = None):
    repo = _repo()
    with _catch_error():
        job_id = repo.resolve_id(job_id)
        job = repo.get(job_id)

    status = get_job_status_with_finalize(job_id, job)
    cfg = load_config()
    default_model = job.model or cfg.effective_model(job.agent)
    am = cfg.models_for(job.agent)
    available_models = am.available or cached_models(job.agent)

    return templates.TemplateResponse(
        request,
        "job_detail.html",
        {
            "job_id": job_id,
            "job": job.to_dict(),
            "status": status,
            "target": job.target,
            "issue": job.issue,
            "agent_name": job.agent,
            "default_model": default_model,
            "available_models": available_models,
            "runs": job.runs,
            "agents": list(AGENTS.keys()),
            "pr_url": pr_url,
            "workspace": job.workspace,
            "is_local": job.local,
        },
    )


@router.post("/jobs/{job_id}/rerun", response_class=HTMLResponse)
async def job_rerun(
    request: Request,
    job_id: str,
    message: str = Form(""),
    agent_type: str = Form(""),
    model: str = Form(""),
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

    log.info(
        "follow-up on %s: agent=%s model=%s message=%r",
        job_id,
        agent_type,
        model,
        message.strip()[:80],
    )

    issue_data = None
    if job.issue is not None:
        from ptq.issue import fetch_issue

        issue_data = await asyncio.to_thread(fetch_issue, job.issue)

    run_request = RunRequest(
        issue_data=issue_data,
        issue_number=job.issue,
        message=message.strip() or None,
        machine=job.machine,
        local=job.local,
        follow=False,
        model=model,
        max_turns=cfg.default_max_turns,
        agent_type=agent_type,
        existing_job_id=job_id,
    )

    await asyncio.to_thread(run_service.launch, repo, backend, run_request)

    log.info("follow-up launched for %s", job_id)
    return RedirectResponse(url=f"/jobs/{job_id}", status_code=303)


@router.post("/jobs/{job_id}/pr", response_class=HTMLResponse)
async def job_create_pr(
    request: Request,
    job_id: str,
    human_note: str = Form(""),
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
        create_pr, repo, job_id, human_note=human_note, draft=bool(draft)
    )
    log.info("PR created for %s: %s", job_id, result.url)
    return RedirectResponse(url=f"/jobs/{job_id}?pr_url={result.url}", status_code=303)


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
    return HTMLResponse("")


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


@router.get("/jobs/{job_id}/diff", response_class=HTMLResponse)
async def job_diff(job_id: str):
    repo = _repo()
    with _catch_error():
        job = repo.get(job_id)
    backend = backend_for_job(job)
    content = read_artifact(backend, f"{backend.workspace}/jobs/{job_id}/fix.diff")
    if not content:
        return HTMLResponse('<p class="muted">No diff yet.</p>')
    escaped = html.escape(content)
    styled: list[str] = []
    for line in escaped.splitlines():
        if line.startswith("+"):
            styled.append(f'<span class="diff-add">{line}</span>')
        elif line.startswith("-"):
            styled.append(f'<span class="diff-del">{line}</span>')
        elif line.startswith("@@"):
            styled.append(f'<span class="diff-hunk">{line}</span>')
        else:
            styled.append(line)
    return HTMLResponse(f'<pre class="log-viewer">{chr(10).join(styled)}</pre>')


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
