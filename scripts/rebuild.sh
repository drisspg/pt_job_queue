#!/bin/bash
WORKTREE="${1:?Usage: rebuild.sh <pytorch-worktree-path>}"
JOB_DIR="$(dirname "$WORKTREE")"
JOB_PYTHON="$JOB_DIR/.venv/bin/python"
WORKSPACE="$(dirname "$(dirname "$JOB_DIR")")"
RE_CC_CONFIG="$WORKSPACE/.re-cc-config"
BUILD_ENV="$WORKSPACE/scripts/.build-env"

[ -f "$BUILD_ENV" ] && source "$BUILD_ENV"

# Scope env to the job venv so CMake subprojects (e.g. NNPACK's PeachPy
# codegen) resolve `python` from this venv instead of whatever env
# launched rebuild.sh (uv run, conda, etc.).
unset VIRTUAL_ENV
export PATH="$JOB_DIR/.venv/bin:$PATH"

export CCACHE_NOHASHDIR=true USE_NINJA="${USE_NINJA:-1}" BUILD_TEST="${BUILD_TEST:-0}"

# Fast-path cloned worktrees may still have hardlinked .so files that
# setuptools won't overwrite.  Break them before the editable install.
find "$WORKTREE/torch" -name '*.so' -links +1 \
    -exec cp --remove-destination {} {}.tmp \; \
    -exec mv -f {}.tmp {} \; 2>/dev/null

if [ -f "$RE_CC_CONFIG" ]; then
    MAX_JOBS=$(cat "$RE_CC_CONFIG")
    echo "Using re-cc with MAX_JOBS=$MAX_JOBS"
    cd "$WORKTREE" && MAX_JOBS="$MAX_JOBS" \
        re-cc -- uv pip install --python "$JOB_PYTHON" --no-build-isolation -e . 2>&1
else
    cd "$WORKTREE" && \
        uv pip install --python "$JOB_PYTHON" --no-build-isolation -e . 2>&1
fi
