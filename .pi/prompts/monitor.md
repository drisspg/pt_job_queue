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

If failing CI rows are present or the user asks for an action decision, run the read-only supervisor layer:

```bash
uv run ptq supervise --prompts
```

Then summarize actionable PR/job states from the monitor table plus supervisor verdicts. Use the saved `agent_space/supervisor/JOB_ID/` triage transcript as evidence, and use `hud` checks only when available or when the triage output includes job ids worth cross-checking. Do not make code changes from the monitor pane; use `uv run ptq open JOB_ID` for interactive job workspaces.
