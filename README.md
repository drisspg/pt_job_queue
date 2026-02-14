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
```

The agent will:
1. Reproduce the bug using a repro script extracted from the issue
2. Read pytorch source to find the root cause
3. Apply a minimal Python-only fix
4. Test the fix by copying edits to site-packages and re-running the repro
5. Write `report.md` and `fix.diff`

Re-running the same issue reuses the existing worktree and preserves prior edits. Each run gets its own log (`claude-1.log`, `claude-2.log`, ...). Different issues run concurrently via separate git worktrees.

### 3. View results

```bash
# By issue number (uses most recent job)
ptq results 174923

# By full job ID
ptq results 20260214-174923
```

Fetches `report.md`, `fix.diff`, and `claude.log` from the remote and displays the report.

### 4. Apply the fix

```bash
ptq apply 174923 --pytorch-path ~/meta/pytorch
```

Creates a branch `ptq/{issue_number}`, applies the diff, and prints next steps for creating a PR.

### 5. Manage running agents

```bash
# List all jobs with running/stopped status
ptq list

# Check status of a specific job
ptq status 174923

# Kill a specific agent
ptq kill 174923

# Kill all agents on a machine (tracked + zombie processes)
ptq prune my-gpu-box

# Kill all local agents
ptq prune --local
```

### 6. Clean up

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
| `--local` | setup, run, clean | | Use local workspace instead of SSH |
| `--follow/--no-follow` | run | follow | Stream agent output to terminal |
| `--model` | run | opus | Claude model |
| `--max-turns` | run | 100 | Max agent turns |
| `--workspace` | setup, run, prune | `~/ptq_workspace` | Custom workspace path |
| `--keep` | clean | 0 | Number of recent jobs to keep |

## Project layout

```
pt_job_queue/
├── pyproject.toml
├── ptq/
│   ├── cli.py              # Typer CLI
│   ├── ssh.py              # SSH/SCP + local subprocess backends
│   ├── issue.py            # GitHub issue fetching via gh
│   ├── job.py              # Job ID generation + local state
│   ├── workspace.py        # Remote workspace setup
│   ├── agent.py            # Agent prompt construction + launch
│   ├── results.py          # Fetch + display results
│   └── apply.py            # Apply diff to local pytorch checkout
├── prompts/
│   └── investigate.md      # Agent system prompt template
└── scripts/
    └── apply_to_site_pkgs.sh
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
        ├── report.md
        └── fix.diff
```
