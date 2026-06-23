---
description: Start the PTQ main driver for Herdr workspace orchestration
argument-hint: "[request]"
---
Load and follow the repo-local driver skill at `.agents/skills/driver/SKILL.md`.

User driver request: $ARGUMENTS

Start by reconstructing state with:

```bash
uv run ptq list
uv run ptq monitor
herdr status
herdr pane list
```

Act as the main PTQ Herdr driver: coordinate with the monitor workspace, use `uv run ptq open JOB_ID` to create/focus interactive job workspaces, and keep actual code investigation/fixes inside those job workspaces rather than in the driver pane.
