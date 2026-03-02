#!/bin/bash
WORKTREE="${1:?Usage: rebuild.sh <pytorch-worktree-path>}"
VENV="$(dirname "$WORKTREE")/.venv/bin/python"
cd "$WORKTREE" && USE_NINJA=1 "$VENV" setup.py develop 2>&1
