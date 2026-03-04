# ptq — PyTorch Job Queue

CLI tool that takes a GitHub issue number, SSHs into a remote GPU machine, and launches a Claude agent to autonomously investigate and fix the bug. The agent produces a report and a diff that you can review and turn into a PR.

## Install

```bash
cd pt_job_queue
uv pip install -e .
```

## Usage

### 1. Set up a machine

```bash
# Remote GPU machine (auto-detects CUDA version)
ptq setup my-gpu-box

# Remote with explicit CUDA version
ptq setup my-gpu-box --cuda cu130

# Local (for testing/development)
ptq setup --local --cpu
```

This creates a workspace with:
- A `uv`-managed venv with PyTorch nightly
- A pytorch source clone at the matching nightly commit
- Helper scripts for applying fixes to site-packages

### 2. Launch an investigation

```bash
# On a remote machine
ptq run --issue 174923 --machine my-gpu-box

# Locally
ptq run --issue 174923 --local

# Run in background (don't stream output)
ptq run --issue 174923 --machine my-gpu-box --no-follow

# Ad-hoc task (no issue, just a message)
ptq run --machine my-gpu-box -m "Optimize the flex attention CPU codegen"

# Issue + extra context
ptq run --issue 174923 --machine my-gpu-box -m "Focus on the stride logic"

# Use a different agent
ptq run --issue 174923 --machine my-gpu-box --agent cursor --model gpt-5.3-codex-xhigh-fast
```

The agent will:
1. Reproduce the bug using a repro script extracted from the issue
2. Read pytorch source to find the root cause
3. Apply a minimal Python-only fix
4. Test the fix by copying edits to site-packages and re-running the repro
5. Write `report.md` and `fix.diff`

Re-running the same issue reuses the existing worktree and preserves prior edits. Each run gets its own log (`claude-1.log`, `claude-2.log`, ...). Different issues run concurrently via separate git worktrees.

### 3. Web dashboard

```bash
ptq web
# or on a custom port
ptq web --port 9000
```

The web UI lets you:
- Launch jobs (issue-based or ad-hoc) with agent/model/machine selection
- Monitor live logs via streaming
- View reports, diffs, and worklogs
- Follow up on stopped jobs with steering messages
- **Take Over** — copies an SSH command that drops you into the job's worktree with the venv activated
- Create PRs directly from the UI

### 4. Monitor progress (CLI)

```bash
# Peek at the agent's worklog
ptq peek 174923

# Peek with recent log activity
ptq peek 174923 --log 30

# List all jobs with running/stopped status
ptq list
```

The agent maintains a `worklog.md` with entries after each significant step, so you can check progress without streaming the full output.

### 5. View results

```bash
# By issue number (uses most recent job)
ptq results 174923

# By full job ID
ptq results 20260214-174923
```

Fetches `report.md`, `fix.diff`, `worklog.md`, and the run log from the remote.

### 6. Apply the fix

```bash
ptq apply 174923 --pytorch-path ~/meta/pytorch
```

Creates a branch `ptq/{issue_number}`, applies the diff, and prints next steps for creating a PR.

### 7. Manage agents

```bash
# Check status of a specific job
ptq status 174923

# Kill a specific agent
ptq kill 174923

# Kill all agents on a machine (tracked + zombie processes)
ptq prune my-gpu-box

# Kill all local agents
ptq prune --local
```

### 8. Clean up

```bash
# Remove all jobs on a machine
ptq clean my-gpu-box

# Keep the 3 most recent
ptq clean my-gpu-box --keep 3

# Clean local workspace
ptq clean --local
```

Removes job directories and prunes git worktrees.

## Options

| Flag | Command | Default | Description |
|------|---------|---------|-------------|
| `--cuda` | setup | auto-detect | CUDA tag (`cu124`, `cu126`, `cu128`, `cu130`) |
| `--cpu` | setup | | Use CPU-only PyTorch (macOS/testing) |
| `--machine` | run | | Remote machine hostname |
| `--local` | setup, run, clean, prune | | Use local workspace instead of SSH |
| `--follow/--no-follow` | run | follow | Stream agent output to terminal |
| `--agent` | run | claude | Agent (`claude`, `codex`, `cursor`) |
| `--model` | run | opus | Model name (agent-specific) |
| `--max-turns` | run | 100 | Max agent turns |
| `-m/--message` | run | | Ad-hoc task or extra context for an issue |
| `--workspace` | setup, run, prune | `~/ptq_workspace` | Custom workspace path |
| `--keep` | clean | 0 | Number of recent jobs to keep |
| `--log` | peek | 0 | Number of log lines to show |

## Project layout

```
pt_job_queue/
├── pyproject.toml
├── ptq/
│   ├── cli.py                          # Thin Typer CLI adapter
│   ├── ssh.py                          # SSH/SCP + local subprocess backends
│   ├── issue.py                        # GitHub issue fetching via gh
│   ├── agent.py                        # Prompt construction + text utilities
│   ├── agents.py                       # Agent protocol + claude/codex/cursor
│   ├── config.py                       # Config loading (~/.ptq/config.toml)
│   ├── workspace.py                    # Remote workspace setup
│   ├── domain/
│   │   ├── models.py                   # JobRecord, RunRequest, JobStatus, errors
│   │   └── policies.py                 # Job ID generation
│   ├── infrastructure/
│   │   ├── job_repository.py           # JSON persistence (~/.ptq/jobs.json)
│   │   └── backends.py                 # Backend factory functions
│   ├── application/
│   │   ├── run_service.py              # Launch/rerun orchestration
│   │   ├── job_service.py              # Status/kill/clean/list
│   │   ├── artifact_service.py         # Results fetching + diff apply
│   │   └── pr_service.py              # PR creation workflow
│   └── web/
│       ├── app.py                      # FastAPI app factory
│       ├── deps.py                     # Template + status helpers
│       ├── routes.py                   # Thin web route adapter
│       ├── static/style.css            # Dark-theme styles
│       └── templates/                  # Jinja2 templates (Pico CSS + htmx)
├── prompts/
│   ├── investigate.md                  # Issue investigation prompt
│   └── adhoc.md                        # Freeform task prompt
└── scripts/
    └── rebuild.sh
```

## Workspace layout (on remote/local)

```
~/ptq_workspace/
├── .venv/                          # uv-managed, PyTorch nightly
├── pytorch/                        # Source clone at nightly commit
├── scripts/apply_to_site_pkgs.sh   # Copies edits to site-packages
└── jobs/
    └── 20260214-174923/            # Per-issue job directory
        ├── pytorch/                # git worktree (isolated)
        ├── system_prompt.md
        ├── repro.py
        ├── claude-1.log            # Per-run logs
        ├── claude-2.log
        ├── worklog.md              # Agent progress log
        ├── report.md
        └── fix.diff
```
