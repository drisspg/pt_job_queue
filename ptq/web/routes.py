from __future__ import annotations

import asyncio
import html
import logging
import re
from contextlib import contextmanager

import markdown as md_lib
from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from sse_starlette.sse import EventSourceResponse

from ptq.agents import AGENTS, get_agent
from ptq.config import load_config
from ptq.job import get_job, resolve_job_id, save_pid
from ptq.ssh import (
    LocalBackend,
    RemoteBackend,
    backend_for_job,
    load_jobs_db,
)
from ptq.web.deps import get_job_status, read_artifact, templates

log = logging.getLogger("ptq.web")

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]|\r")
MAX_RESULT_LINES = 20


def _render_md(text: str) -> str:
    return md_lib.markdown(text, extensions=["fenced_code", "tables", "nl2br"])


router = APIRouter()


@contextmanager
def _catch_exit():
    try:
        yield
    except SystemExit as e:
        raise HTTPException(status_code=404, detail=str(e)) from None


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------


@router.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    db = load_jobs_db()
    jobs: list[dict] = []
    running_count = 0
    stopped_count = 0
    machines: dict[str, dict] = {}

    for job_id, entry in sorted(db.items()):
        status = get_job_status(job_id, entry)
        if status == "running":
            running_count += 1
        else:
            stopped_count += 1

        target = entry.get("machine") or "local"
        machines.setdefault(target, {"running": 0, "stopped": 0, "total": 0})
        machines[target][status] += 1
        machines[target]["total"] += 1

        jobs.append(
            {
                "id": job_id,
                "issue": entry.get("issue"),
                "agent": entry.get("agent", "claude"),
                "target": target,
                "runs": entry.get("runs", 1),
                "status": status,
            }
        )

    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {
            "jobs": jobs,
            "total": len(db),
            "running": running_count,
            "stopped": stopped_count,
            "machines": machines,
        },
    )


# ---------------------------------------------------------------------------
# Job list
# ---------------------------------------------------------------------------


@router.get("/jobs", response_class=HTMLResponse)
async def job_list(request: Request, status_filter: str = "all"):
    db = load_jobs_db()
    jobs: list[dict] = []

    for job_id, entry in sorted(db.items()):
        status = get_job_status(job_id, entry)
        if status_filter != "all" and status != status_filter:
            continue
        jobs.append(
            {
                "id": job_id,
                "issue": entry.get("issue"),
                "agent": entry.get("agent", "claude"),
                "target": entry.get("machine") or "local",
                "runs": entry.get("runs", 1),
                "status": status,
            }
        )

    return templates.TemplateResponse(
        request,
        "job_list.html",
        {"jobs": jobs, "status_filter": status_filter},
    )


# ---------------------------------------------------------------------------
# New job form
# ---------------------------------------------------------------------------


def _form_context(error: str | None = None) -> dict:
    cfg = load_config()
    return {
        "agents": list(AGENTS.keys()),
        "machines": cfg.machines,
        "agent_models": {
            name: {"available": am.available, "default": am.default}
            for name, am in cfg.agent_models.items()
        },
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


@router.get("/api/models/{agent_name}", response_class=HTMLResponse)
async def agent_models_options(agent_name: str):
    cfg = load_config()
    am = cfg.models_for(agent_name)
    if am.available:
        options = "".join(
            f'<option value="{m}" {"selected" if m == am.default else ""}>{m}</option>'
            for m in am.available
        )
        return HTMLResponse(f'<select id="model" name="model">{options}</select>')
    default_val = am.default or cfg.default_model
    return HTMLResponse(
        f'<input type="text" id="model" name="model" value="{default_val}" placeholder="e.g. opus">'
    )


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
            _form_context("Message is required for adhoc tasks."),
            status_code=422,
        )
    if target_type == "machine" and not machine.strip():
        return templates.TemplateResponse(
            request,
            "job_new.html",
            _form_context("Machine name is required."),
            status_code=422,
        )

    issue_number = int(issue) if issue.strip() else None
    message_text = message.strip() or None
    is_local = target_type == "local"
    machine_name = machine.strip() or None

    backend = LocalBackend() if is_local else RemoteBackend(machine=machine_name)

    issue_data = None
    if issue_number is not None:
        from ptq.issue import fetch_issue

        issue_data = await asyncio.to_thread(fetch_issue, issue_number)

    from ptq.agent import launch_agent

    log.info(
        "creating job: agent=%s model=%s target=%s",
        agent,
        model,
        "local" if is_local else machine_name,
    )
    job_id = await asyncio.to_thread(
        launch_agent,
        backend,
        issue_data=issue_data,
        issue_number=issue_number,
        machine=machine_name,
        local=is_local,
        follow=False,
        model=model,
        max_turns=max_turns,
        message=message_text,
        agent_type=agent,
    )

    return RedirectResponse(url=f"/jobs/{job_id}", status_code=303)


# ---------------------------------------------------------------------------
# Job detail
# ---------------------------------------------------------------------------


@router.get("/jobs/{job_id}", response_class=HTMLResponse)
async def job_detail(request: Request, job_id: str):
    with _catch_exit():
        job_id = resolve_job_id(job_id)
        job = get_job(job_id)

    status = get_job_status(job_id, job)
    target = job.get("machine") or "local"
    issue_val = job.get("issue")

    cfg = load_config()
    agent_name = job.get("agent", "claude")
    default_model = cfg.effective_model(agent_name)

    return templates.TemplateResponse(
        request,
        "job_detail.html",
        {
            "job_id": job_id,
            "job": job,
            "status": status,
            "target": target,
            "issue": issue_val,
            "agent_name": agent_name,
            "default_model": default_model,
            "runs": job.get("runs", 1),
            "agents": list(AGENTS.keys()),
        },
    )


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------


@router.post("/jobs/{job_id}/rerun", response_class=HTMLResponse)
async def job_rerun(
    request: Request,
    job_id: str,
    message: str = Form(""),
    agent_type: str = Form(""),
    model: str = Form(""),
):
    with _catch_exit():
        job_id = resolve_job_id(job_id)
        job = get_job(job_id)

    from ptq.agent import launch_agent

    cfg = load_config()
    backend = backend_for_job(job_id)
    agent_type = agent_type.strip() or job.get("agent", cfg.default_agent)
    model = model.strip() or cfg.effective_model(agent_type)
    issue_number = job.get("issue")

    log.info(
        "follow-up on %s: agent=%s model=%s message=%r",
        job_id,
        agent_type,
        model,
        message.strip()[:80],
    )

    issue_data = None
    if issue_number is not None:
        from ptq.issue import fetch_issue

        issue_data = await asyncio.to_thread(fetch_issue, issue_number)

    await asyncio.to_thread(
        launch_agent,
        backend,
        issue_data=issue_data,
        issue_number=issue_number,
        machine=job.get("machine"),
        local=job.get("local", False),
        follow=False,
        model=model,
        max_turns=cfg.default_max_turns,
        message=message.strip() or None,
        agent_type=agent_type,
        existing_job_id=job_id,
    )

    log.info("follow-up launched for %s", job_id)
    return RedirectResponse(url=f"/jobs/{job_id}", status_code=303)


@router.post("/jobs/{job_id}/kill", response_class=HTMLResponse)
async def job_kill(request: Request, job_id: str):
    with _catch_exit():
        job_id = resolve_job_id(job_id)
        job = get_job(job_id)

    backend = backend_for_job(job_id)
    pid = job.get("pid")
    if pid and backend.is_pid_alive(pid):
        backend.kill_pid(pid)
        save_pid(job_id, None)
        log.info("killed %s (pid %s)", job_id, pid)

    return RedirectResponse(url=f"/jobs/{job_id}", status_code=303)


@router.delete("/jobs/{job_id}", response_class=HTMLResponse)
async def job_delete(job_id: str):
    with _catch_exit():
        job_id = resolve_job_id(job_id)
        job = get_job(job_id)

    from ptq.cli import clean_job

    log.info("cleaning %s", job_id)
    backend = backend_for_job(job_id)
    await asyncio.to_thread(clean_job, job_id, job, backend)
    return HTMLResponse("")


# ---------------------------------------------------------------------------
# Partials (htmx targets)
# ---------------------------------------------------------------------------


@router.get("/jobs/{job_id}/status", response_class=HTMLResponse)
async def job_status_badge(request: Request, job_id: str):
    with _catch_exit():
        job = get_job(job_id)
    status = get_job_status(job_id, job)
    return templates.TemplateResponse(
        request,
        "partials/status_badge.html",
        {"status": status, "job_id": job_id},
    )


@router.get("/jobs/{job_id}/report", response_class=HTMLResponse)
async def job_report(job_id: str):
    with _catch_exit():
        get_job(job_id)
    backend = backend_for_job(job_id)
    ws = backend.workspace
    content = read_artifact(backend, f"{ws}/jobs/{job_id}/report.md")
    return HTMLResponse(
        f'<div class="rendered-md">{_render_md(content)}</div>'
        if content
        else '<p class="muted">No report yet.</p>'
    )


@router.get("/jobs/{job_id}/diff", response_class=HTMLResponse)
async def job_diff(job_id: str):
    with _catch_exit():
        get_job(job_id)
    backend = backend_for_job(job_id)
    ws = backend.workspace
    content = read_artifact(backend, f"{ws}/jobs/{job_id}/fix.diff")
    if not content:
        return HTMLResponse('<p class="muted">No diff yet.</p>')
    escaped = html.escape(content)
    lines = escaped.splitlines()
    styled: list[str] = []
    for line in lines:
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
    with _catch_exit():
        get_job(job_id)
    backend = backend_for_job(job_id)
    ws = backend.workspace
    content = read_artifact(backend, f"{ws}/jobs/{job_id}/worklog.md")
    return HTMLResponse(
        f'<div class="rendered-md">{_render_md(content)}</div>'
        if content
        else '<p class="muted">No worklog yet.</p>'
    )


# ---------------------------------------------------------------------------
# SSE log streaming
# ---------------------------------------------------------------------------

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
    with _catch_exit():
        job_id = resolve_job_id(job_id)
        job = get_job(job_id)

    backend = backend_for_job(job_id)
    ws = backend.workspace
    runs = job.get("runs", 1)
    agent = get_agent(job.get("agent", "claude"))
    log_file = f"{ws}/jobs/{job_id}/{agent.log_filename(runs)}"
    pid = job.get("pid")
    is_alive = pid is not None and backend.is_pid_alive(pid)
    log.debug(
        "stream_logs %s: log_file=%s pid=%s alive=%s", job_id, log_file, pid, is_alive
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
