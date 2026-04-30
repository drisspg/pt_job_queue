from __future__ import annotations

from ptq.repo_profiles import get_profile
from ptq.ssh import Backend


def render_job_context(
    *,
    job_id: str,
    workspace: str,
    repo: str = "pytorch",
    name: str | None = None,
) -> str:
    profile = get_profile(repo)
    job_dir = f"{workspace}/jobs/{job_id}"
    worktree_path = f"{job_dir}/{profile.dir_name}"
    venv_path = f"{job_dir}/.venv"
    title = name or job_id

    return f"""# PTQ Job Context

This directory is a PTQ-managed job home for `{title}`.

## Paths

- Job ID: `{job_id}`
- Job directory: `{job_dir}`
- Source worktree: `{worktree_path}`
- Python/venv: `{venv_path}/bin/python`
- Artifacts: `{job_dir}`

## Enter the PTQ job home

```bash
cd {job_dir} && source .venv/bin/activate
```

The PyTorch source worktree is in `pytorch/` from there.

## Source and environment rules

- Edit source in `{worktree_path}`.
- Use `{venv_path}/bin/python` for Python commands.
- Write scratch files and reports under `{job_dir}` or `{worktree_path}/agent_space`.
- For PyTorch C++ changes, rebuild with:

```bash
bash {workspace}/scripts/rebuild.sh {worktree_path}
```

## PTQ commands

Run these from the PTQ repo, usually `~/meta/pt_job_queue`.

```bash
uv run ptq list
uv run ptq takeover {job_id}
uv run ptq run {job_id} -m 'follow-up instructions here' --agent pi
uv run ptq peek {job_id}
uv run ptq results {job_id}
uv run ptq clean {job_id}
```

If this job has a name, you can also launch by name:

```bash
uv run ptq run {title} -m 'task instructions here' --agent pi
```

## Agent context

PTQ-launched agents receive a rendered system prompt at `{job_dir}/system_prompt.md`.
Follow-up runs also receive prior context from `{job_dir}/worklog.md` and `{job_dir}/report.md`.

Manual agents launched from this job directory should read this file first, then follow repo-local instructions in `{worktree_path}/AGENTS.md` when editing source.
"""


def write_job_context(
    backend: Backend,
    *,
    job_id: str,
    workspace: str,
    repo: str = "pytorch",
    name: str | None = None,
) -> None:
    profile = get_profile(repo)
    job_dir = f"{workspace}/jobs/{job_id}"
    worktree_path = f"{job_dir}/{profile.dir_name}"
    content = render_job_context(
        job_id=job_id,
        workspace=workspace,
        repo=repo,
        name=name,
    )
    backend.run(
        f"cat > {job_dir}/PTQ_CONTEXT.md << 'PTQ_CONTEXT_EOF'\n{content}\nPTQ_CONTEXT_EOF"
    )
    backend.run(f"mkdir -p {worktree_path}/agent_space", check=False)
    backend.run(
        f"cp {job_dir}/PTQ_CONTEXT.md {worktree_path}/agent_space/PTQ_CONTEXT.md",
        check=False,
    )
    backend.run(f"cp {job_dir}/PTQ_CONTEXT.md {job_dir}/AGENTS.md", check=False)
    backend.run(f"cp {job_dir}/PTQ_CONTEXT.md {job_dir}/CLAUDE.md", check=False)
