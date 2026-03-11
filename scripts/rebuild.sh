#!/bin/bash
WORKTREE="${1:?Usage: rebuild.sh <pytorch-worktree-path>}"
JOB_DIR="$(dirname "$WORKTREE")"
JOB_PYTHON="$JOB_DIR/.venv/bin/python"
WORKSPACE="$(dirname "$(dirname "$WORKTREE")")"
RE_CC_CONFIG="$WORKSPACE/.re-cc-config"

if [ -f "$RE_CC_CONFIG" ]; then
    MAX_JOBS=$(cat "$RE_CC_CONFIG")
    echo "Using re-cc with MAX_JOBS=$MAX_JOBS"
    cd "$WORKTREE" && MAX_JOBS="$MAX_JOBS" CCACHE_NOHASHDIR=true USE_NINJA=1 \
        re-cc -- uv pip install --python "$JOB_PYTHON" --no-build-isolation -e . 2>&1
else
    cd "$WORKTREE" && CCACHE_NOHASHDIR=true USE_NINJA=1 \
        uv pip install --python "$JOB_PYTHON" --no-build-isolation -e . 2>&1
fi
