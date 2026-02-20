# ptq — PyTorch Job Queue

CLI tool that dispatches Claude agents to autonomously investigate and fix PyTorch GitHub issues on GPU machines. The agent reproduces the bug, finds the root cause, applies a fix, and produces a report and diff.

## Install

```bash
cd pt_job_queue
uv pip install -e .

# With GPU auto-provisioning support (optional)
uv pip install -e '.[gpu]'
```

## Quick start

### Fully automatic (recommended)

```bash
# Reserve a GPU, set up workspace, run agent, fetch results — all in one command
ptq auto --issue 149002

# With options
ptq auto --issue 149002 --gpu-type a100 --hours 2
ptq auto --issue 149002 -p "focus on the inductor nan_assert codepath"
ptq auto --issue 149002 --no-pr --no-follow
```

`ptq auto` handles the full lifecycle:
1. Fetches the GitHub issue (fails fast before reserving a GPU)
2. Reserves a GPU pod via gpu-dev
3. Sets up the workspace (uv, PyTorch nightly, source clone)
4. Launches a Claude agent to investigate and fix the issue
5. Fetches results when the agent finishes
6. Cancels the reservation (in `--follow` mode; in `--no-follow`, the reservation auto-expires)

The `--hours` flag (default: 4) is a max cap — the reservation is cancelled as soon as the agent finishes.

### Manual workflow

If you already have a GPU machine provisioned:

```bash
# 1. One-time setup
ptq setup my-gpu-box

# 2. Run an investigation
ptq run --issue 174923 --machine my-gpu-box

# 3. View results
ptq results 174923

# 4. Apply the fix locally
ptq apply 174923 --pytorch-path ~/pytorch
```

## Commands

### `ptq auto` — Full auto mode

Reserves a GPU, runs the full pipeline, and cleans up.

```bash
ptq auto --issue 149002                              # investigate a GitHub issue
ptq auto --issue 149002 --gpu-type a100 --hours 2    # custom GPU and duration
ptq auto --issue 149002 -p "check flex_attention.py"  # extra guidance for agent
ptq auto --issue 149002 --no-pr                       # skip PR creation
ptq auto -m "benchmark torch.compile on H100" -h 6   # freeform task
ptq auto --issue 149002 --no-follow                   # launch in background
```

| Option | Default | Description |
|--------|---------|-------------|
| `--issue` | | GitHub issue number |
| `--message`, `-m` | | Freeform task (instead of issue) |
| `--input`, `-i` | | Read task from file |
| `--prompt`, `-p` | | Extra guidance passed to the agent |
| `--gpu-type` | `h100` | GPU type (`h100`, `a100`, `a10g`, `t4`, etc.) |
| `--gpus` | `1` | Number of GPUs |
| `--hours` | `4.0` | Max reservation hours (auto-cancels when done) |
| `--no-pr` | | Skip PR creation after fix |
| `--model` | `opus` | Claude model |
| `--max-turns` | `100` | Max agent turns |
| `--follow/--no-follow` | `follow` | Stream output or run in background |
| `--dockerfile` | | Custom Dockerfile for the GPU pod |

### `ptq setup` — Workspace setup

One-time setup on a remote machine or locally.

```bash
ptq setup my-gpu-box               # remote (auto-detects CUDA)
ptq setup my-gpu-box --cuda cu130   # explicit CUDA version
ptq setup --local --cpu             # local (for testing)
```

### `ptq run` — Launch agent

Launch a Claude agent on a pre-provisioned machine.

```bash
ptq run --issue 174923 --machine my-gpu-box
ptq run -m "investigate OOM in flex_attention" --machine gpu-dev
ptq run -i task.md --machine gpu-dev
ptq run 174923 -m "try a different approach"          # re-run with steering
```

### `ptq results` — Fetch results

```bash
ptq results 174923
ptq results 20260214-174923
```

Fetches `report.md`, `fix.diff`, `worklog.md`, and the agent log.

### `ptq apply` — Apply fix locally

```bash
ptq apply 174923 --pytorch-path ~/pytorch
```

Creates branch `ptq/{issue_number}`, applies the diff, prints next steps for PR.

### `ptq list` — List all jobs

```bash
ptq list
```

Shows status (running/stopped), job ID, issue number, run count, target machine, and reservation ID (if provisioned via `ptq auto`).

### `ptq peek` — Check progress

```bash
ptq peek 174923              # show worklog
ptq peek 174923 --log 30     # also show last 30 log lines
```

### `ptq status` — Check if agent is running

```bash
ptq status 174923
```

### `ptq kill` — Stop an agent

```bash
ptq kill 174923
```

### `ptq clean` — Remove jobs

```bash
ptq clean 174923                    # single job
ptq clean my-gpu-box                # all stopped jobs on machine
ptq clean my-gpu-box --keep 2       # keep 2 most recent
ptq clean my-gpu-box --all          # include running jobs
ptq clean --local                   # local workspace
```

When cleaning a job created by `ptq auto`, the associated GPU reservation is also cancelled.

## How the agent works

The agent receives a system prompt with the issue context, a workspace with PyTorch nightly + source, and a repro script extracted from the issue. It then:

1. **Reproduces** the bug using the repro script
2. **Investigates** by reading PyTorch source in an isolated git worktree
3. **Fixes** with minimal Python-only edits
4. **Tests** by syncing edits to site-packages and re-running the repro
5. **Outputs** `report.md` and `fix.diff`
6. **Verifies & creates PR** (if instructed) using built-in skills

### Skills

The agent has access to two skills (baked into the workspace):

- **verify-fix**: Runs the repro to confirm the fix, executes related regression tests, and does a before/after performance comparison.
- **make-pr**: Creates a `ptq/{issue}` branch, commits the fix, pushes, and creates a draft PR via `gh pr create --draft`.

## GPU provisioning (gpu-dev integration)

`ptq auto` uses [gpu-dev-cli](https://github.com/wdvr/osdc) to manage GPU reservations. Requirements:

1. `gpu-dev-cli` installed (`pip install 'ptq[gpu]'`)
2. AWS credentials configured for gpu-dev
3. GitHub username set: `gpu-dev config set github_user <username>`
4. SSH config includes gpu-dev: the CLI handles this automatically

The reservation uses `no_persistent_disk=True` since results are fetched before cancellation.

## Project layout

```
pt_job_queue/
├── pyproject.toml
├── ptq/
│   ├── cli.py              # Typer CLI (setup, run, auto, results, etc.)
│   ├── ssh.py              # SSH + local subprocess backends
│   ├── issue.py            # GitHub issue fetching via gh
│   ├── job.py              # Job ID generation + local state tracking
│   ├── workspace.py        # Remote workspace setup (uv, PyTorch, git)
│   ├── agent.py            # Agent prompt construction + Claude Code launch
│   ├── results.py          # Fetch + display results
│   ├── apply.py            # Apply diff to local pytorch checkout
│   ├── provision.py        # GPU reservation lifecycle (gpu-dev wrapper)
│   └── docker.py           # Dockerfile generation + build context
├── prompts/
│   ├── investigate.md      # System prompt for issue investigations
│   └── adhoc.md            # System prompt for freeform tasks
├── skills/
│   ├── verify-fix/SKILL.md # Testing + perf verification skill
│   └── make-pr/SKILL.md    # Draft PR creation skill
├── scripts/
│   └── apply_to_site_pkgs.sh
└── tests/
    ├── test_agent.py
    ├── test_cli.py
    ├── test_docker.py
    ├── test_job.py
    ├── test_provision.py
    ├── test_issue.py
    ├── test_results.py
    └── test_workspace.py
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
