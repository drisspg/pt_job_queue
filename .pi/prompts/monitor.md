---
description: Start the PTQ monitor operator for Herdr PR/job monitoring
argument-hint: "[focus or instructions]"
---
Load and follow the repo-local monitor skill at `.agents/skills/monitor/SKILL.md`.

User monitor request: $ARGUMENTS

If running inside Herdr, rename the current Herdr workspace/namespace to `ptq monitor` first:

```bash
if [ -n "${HERDR_PANE_ID:-}" ]; then
  WORKSPACE_ID="$(herdr pane get "$HERDR_PANE_ID" | python -c 'import json, sys; print(json.load(sys.stdin)["result"]["pane"]["workspace_id"])')"
  herdr workspace rename "$WORKSPACE_ID" "ptq monitor"
fi
```

Then run:

```bash
uv run ptq monitor
```

Then summarize actionable PR/job states. For failing CI, run the printed `~/dotfiles/scripts/github_ci_triage PR_URL` command first, and use `hud` checks only when available or when the triage output includes job ids worth cross-checking. Do not make code changes from the monitor pane; use `uv run ptq open JOB_ID` for interactive job workspaces.
