# Project Instructions

- For this repo, run the project CLI with `uv run ptq ...` rather than calling `python -m ptq.cli` or a bare `ptq`
- Prefer the command forms documented in `README.md` for repo tasks such as `run`, `clean`, `list`, `results`, and `web`
- When a user asks to create, make, or set up a workspace for a PyTorch GitHub issue URL/number, prefer the fast local PTQ issue-job path unless they explicitly ask for a remote machine, fresh seed checkout, isolated workspace, or rebuild.
- Fast local issue workflow: run `uv run ptq list`; if the issue already has a job, use `uv run ptq takeover JOB_ID`; otherwise run `ISSUE=123456; uv run ptq run --issue "$ISSUE" --local --agent pi --no-follow; uv run ptq takeover "$ISSUE"`.
- Do not run `ptq setup --build` per issue for a fast workspace. The fast path uses the existing local seed workspace, creates `~/.ptq_workspace/jobs/<job-id>/pytorch`, and clones the seed venv/build artifacts.
- Use an issue-scoped `uv run ptq setup --local --workspace "$WS" --build` only when a separate built seed checkout is intentionally desired; it is slow and not the default response to “make a workspace for this issue.”
- When invoking tests for this repo, follow the README convention and use `uv run --extra dev pytest ...`

# Common PTQ Commands

- List jobs and named worktrees before creating anything new: `uv run ptq list`
- For a new fast local PyTorch issue job, use the shared seed workspace and then take it over: `ISSUE=123456; uv run ptq run --issue "$ISSUE" --local --agent pi --no-follow; uv run ptq takeover "$ISSUE"`
- Create a local named PyTorch worktree with a ready venv: `uv run ptq worktree NAME --local`
- Create one on a remote machine: `uv run ptq worktree NAME --machine MACHINE`
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

- PTQ-managed worktrees live under `<workspace>/jobs/<job-id>/<repo-dir>` and have a per-job venv at `<workspace>/jobs/<job-id>/.venv`.
- Fast local issue jobs live under the shared seed workspace `~/.ptq_workspace/jobs/<job-id>/pytorch`; do not create `~/.ptq_workspaces/pytorch-<issue>` unless you intentionally want a slow, separate built seed workspace.
- Reuse existing issue jobs with `uv run ptq takeover JOB_ID`; do not create a second checkout just to inspect or continue work.
- Do not pass `-v/--verbose` to routine `ptq worktree` commands. It only streams provisioning output and can hide an accidental fallback build behind a wall of logs.
- Raw PyTorch worktrees directly under a workspace root are not PTQ-managed unless they are also registered in `~/.ptq/jobs.json`.
- Before deleting or recreating any worktree, check for uncommitted work with `git -C PATH status --short`.
