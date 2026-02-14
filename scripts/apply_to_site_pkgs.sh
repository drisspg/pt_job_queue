#!/bin/bash
PYTORCH_SRC="${1:-~/ptq_workspace/pytorch}"
JOB_DIR="$(dirname "$PYTORCH_SRC")"
WORKSPACE="$(dirname "$(dirname "$JOB_DIR")")"
SITE_PKGS=$("$WORKSPACE/.venv/bin/python" -c "import torch, pathlib; print(pathlib.Path(torch.__file__).parent)")
cd "$PYTORCH_SRC"
for f in $(git diff --name-only --diff-filter=M | grep '\.py$'); do
    [[ "$f" == torch/* ]] || continue
    DEST="$SITE_PKGS/${f#torch/}"
    [ -f "$DEST" ] && cp "$f" "$DEST" && echo "  $f -> $DEST"
done
