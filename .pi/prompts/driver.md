---
description: Load PTQ driver context for Herdr workspace guidance
argument-hint: "[request]"
---
Load and follow the repo-local driver skill at `.agents/skills/driver/SKILL.md`.

User driver request: $ARGUMENTS

Treat this prompt as PTQ driver context setup only except for the load-time Herdr workspace rename specified by the driver skill.

Default behavior:
- Rename the current Herdr workspace to `ptq driver` when `HERDR_PANE_ID` is available.
- Explain the relevant PTQ/Herdr workflow.
- Suggest copy/paste-ready commands.
- After the load-time rename, only run commands when the user explicitly asks you to run, check, open, create, focus, rename, triage, inspect, or otherwise act.
- Leave PR/CI monitoring and triage behavior to the `/monitor` skill unless the user explicitly asks the driver to run those commands.

Load-time rename command:

```bash
if [ -n "${HERDR_PANE_ID:-}" ]; then
  WORKSPACE_ID="$(herdr pane get "$HERDR_PANE_ID" | python -c 'import json, sys; print(json.load(sys.stdin)["result"]["pane"]["workspace_id"])')"
  herdr workspace rename "$WORKSPACE_ID" "ptq driver"
fi
```

Useful commands to suggest or run only when explicitly requested:

```bash
# Reconstruct state
uv run ptq list
uv run ptq monitor
herdr status
herdr pane list

# Open/focus the monitor workspace
uv run ptq monitor --herdr

# Open/focus a job workspace
uv run ptq open JOB_ID
```
