# Project Instructions

- For this repo, run the project CLI with `uv run ptq ...` rather than calling `python -m ptq.cli` or a bare `ptq`
- Prefer the command forms documented in `README.md` for repo tasks such as `run`, `clean`, `list`, `results`, and `web`
- When invoking tests for this repo, follow the README convention and use `uv run --extra dev pytest ...`

# Common PTQ Commands

- List jobs and named worktrees: `uv run ptq list`
- Create a local named PyTorch worktree with a ready venv: `uv run ptq worktree NAME --local -v`
- Create one on a remote machine: `uv run ptq worktree NAME --machine MACHINE -v`
- Launch an agent in a named worktree: `uv run ptq run NAME -m 'task message' --agent pi`
- Follow up on an existing job with more instructions: `uv run ptq run JOB_ID -m 'follow-up message' --agent pi`
- Peek progress/worklog: `uv run ptq peek JOB_ID`
- Fetch final results/artifacts: `uv run ptq results JOB_ID`
- Enter a worktree with the job venv activated: `uv run ptq takeover JOB_ID`, then run the printed command if needed
- Stop an agent: `uv run ptq kill JOB_ID`
- Remove a job and prune its worktree: `uv run ptq clean JOB_ID`
- Bulk clean stopped jobs for a target: `uv run ptq clean local` or `uv run ptq clean MACHINE`
- Start the dashboard: `uv run ptq web`

# Worktree Layout Notes

- PTQ-managed worktrees live under `~/.ptq_workspace/jobs/<job-id>/<repo-dir>` and have a per-job venv at `~/.ptq_workspace/jobs/<job-id>/.venv`.
- Raw PyTorch worktrees directly under `~/.ptq_workspace/<name>` are not PTQ-managed unless they are also registered in `~/.ptq/jobs.json`.
- Before deleting or recreating any worktree, check for uncommitted work with `git -C PATH status --short`.
