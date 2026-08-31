# ptq — PyTorch Workspace Queue

PTQ creates isolated PyTorch worktrees, opens them in Herdr, tracks their pull requests, and monitors CI. Development happens interactively in Pi from each job workspace.

## Install

No installation step is required:

```bash
cd pt_job_queue
uv run ptq --help
```

For development:

```bash
uv run --extra dev pytest
```

## Set up a seed workspace

Create one local seed checkout and build it once:

```bash
uv run ptq setup --build
```

Use a different base when needed:

```bash
uv run ptq setup --build --onto upstream/viable/strict
```

Setup creates the source checkout, uv-managed environment, build dependencies, and helper scripts used to provision job worktrees. Add build environment settings to `~/.ptq/config.toml`:

```toml
[build.env]
USE_SYSTEM_NCCL = "1"
```

## Open a workspace

Create or reuse a local issue workspace and open it in Herdr:

```bash
uv run ptq open --issue 143260
```

Create or reuse named work:

```bash
uv run ptq open --name scaled-addmm-api
```

Open an existing job by ID, issue number, or name:

```bash
uv run ptq open JOB_ID
```

For an intentionally isolated seed checkout:

```bash
ISSUE=143260
WS="$HOME/.ptq_workspaces/pytorch-$ISSUE"
uv run ptq setup --workspace "$WS" --build
uv run ptq open --issue "$ISSUE" --workspace "$WS"
```

`open` does not launch a background agent. In the created Herdr workspace, start Pi from the job directory and load:

```text
@prime.md
```

`prime.md` contains the paths and workflow rules. The same context is copied to job-root `AGENTS.md` for agents that discover it automatically.

Use `takeover` when only the shell-entry command is needed:

```bash
uv run ptq takeover JOB_ID
```

## Inspect jobs

```bash
uv run ptq list
uv run ptq peek JOB_ID
```

`peek` displays the current `worklog.md` and `report.md` directly from the job workspace.

## Monitor pull requests

```bash
uv run ptq monitor
uv run ptq monitor --watch
uv run ptq monitor --all
```

The monitor shows PR state, CI state, rebase state, and the next action. Issue and PR cells are terminal hyperlinks when OSC-8 is supported.

For a `needs CI review` row, the Driver skill gathers evidence with:

```bash
~/dotfiles/scripts/github_ci_triage PR_URL
```

If a stopped landing attempt has only unrelated, flaky, or broken-trunk failures, the driver may propose:

```bash
gh pr comment PR_URL --body '@pytorchbot merge -i'
```

The driver never posts comments, reruns CI, pushes, merges, or cleans jobs without user approval.

The unified driver workflow lives at `.agents/skills/driver/SKILL.md`; `.pi/prompts/driver.md` provides `/driver` in interactive Pi.

## Submit a pull request

```bash
uv run ptq pr JOB_ID
```

PTQ submits the current worktree directly. It uses `pr_title.txt`, `pr_labels.txt`, `report.md`, and `worklog.md` when present. Existing GitHub titles and human notes remain the source of truth during updates.

## Ghstack workflow

Initialize stack mode before implementation:

```bash
uv run ptq open --name MY_STACK
uv run ptq stack init MY_STACK
```

Create a linear sequence of independently buildable and tested commits, then inspect and submit it:

```bash
uv run ptq stack show MY_STACK
uv run ptq stack submit MY_STACK --draft
```

Ordinary updates preserve GitHub titles and bodies. Use `--update-metadata` only when intentionally replacing them from commit messages.

For PyTorch, `@pytorchbot merge` on a ghstack PR lands that PR and every open PR below it. Use the bottom PR for one layer or the top PR for the whole stack; do not use the GitHub merge button.

After a lower layer lands:

```bash
uv run ptq rebase MY_STACK
uv run ptq stack submit MY_STACK
```

## Rebase

```bash
uv run ptq rebase JOB_ID
uv run ptq rebase JOB_ID --onto origin/main
```

A clean rebase completes automatically. If conflicts occur, PTQ leaves the rebase in progress and prints the job entry command so they can be resolved interactively in Herdr.

## Clean up

```bash
uv run ptq clean JOB_ID
uv run ptq clean local
uv run ptq clean local --keep 3
```

Before deleting or recreating a worktree, check it for uncommitted work.

## Add a repository

Add a section to `~/.ptq/config.toml`:

```toml
[repos.example]
github_repo = "org/example"
clone_url = "https://github.com/org/example.git"
dir_name = "example"
smoke_test_import = "example"
```

Optional profile fields:

| Field | Default | Purpose |
|---|---:|---|
| `uses_custom_worktree_tool` | `false` | Use PyTorch's `tools/create_worktree.py` |
| `needs_cpp_build` | `false` | Build native code during environment provisioning |
| `lint_cmd` | unset | Repository lint command recorded in context |

## Project layout

```text
pt_job_queue/
├── ptq/
│   ├── cli.py
│   ├── config.py
│   ├── repo_profiles.py
│   ├── workspace.py
│   ├── domain/
│   │   ├── models.py
│   │   └── policies.py
│   ├── infrastructure/
│   │   ├── backends.py
│   │   └── job_repository.py
│   └── application/
│       ├── herdr_service.py
│       ├── job_context.py
│       ├── job_service.py
│       ├── monitor_service.py
│       ├── pr_service.py
│       ├── rebase_service.py
│       ├── stack_service.py
│       ├── venv_service.py
│       └── worktree_service.py
├── tests/
└── scripts/
    └── rebuild.sh
```

A job directory contains:

```text
<workspace>/jobs/<job-id>/
├── .venv/
├── <repo>/
├── prime.md
├── AGENTS.md
├── worklog.md
├── report.md
├── pr_title.txt
├── pr_labels.txt
└── STACK_CONTEXT.md  # ghstack jobs only
```
