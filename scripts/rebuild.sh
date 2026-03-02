#!/bin/bash
WORKTREE="${1:?Usage: rebuild.sh <pytorch-worktree-path>}"
JOB_DIR="$(dirname "$WORKTREE")"
JOB_PYTHON="$JOB_DIR/.venv/bin/python"
cd "$WORKTREE" && CCACHE_NOHASHDIR=true USE_NINJA=1 \
    uv pip install --python "$JOB_PYTHON" --no-build-isolation -e . 2>&1
