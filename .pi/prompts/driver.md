---
description: Load PTQ driver context for Herdr workspace guidance
argument-hint: "[request]"
---
Load and follow the repo-local driver skill at `.agents/skills/driver/SKILL.md`.

User driver request: $ARGUMENTS

Treat this prompt as PTQ driver context setup only. Do not run commands just because the prompt was invoked.

Default behavior:
- Explain the relevant PTQ/Herdr workflow.
- Suggest copy/paste-ready commands.
- Only run commands when the user explicitly asks you to run, check, open, create, focus, rename, triage, inspect, or otherwise act.
- Leave PR/CI monitoring and triage behavior to the `/monitor` skill unless the user explicitly asks the driver to run those commands.

Useful commands to suggest or run only when explicitly requested:

```bash
# Rename current Herdr workspace to the driver name
if [ -n "${HERDR_PANE_ID:-}" ]; then
  WORKSPACE_ID="$(herdr pane get "$HERDR_PANE_ID" | python -c 'import json, sys; print(json.load(sys.stdin)["result"]["pane"]["workspace_id"])')"
  herdr workspace rename "$WORKSPACE_ID" "ptq driver"
fi

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
