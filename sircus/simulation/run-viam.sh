#!/bin/sh
# This launcher is invoked by viam-server with its managed module arguments.
set -eu
SIRCUS_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SIRCUS_ROOT"
export SIRCUS_VIAM_CONFIG="$SIRCUS_ROOT/viam-config.local.json"
export SIRCUS_SCENE_CONFIG="$SIRCUS_ROOT/viam-scene.json"
exec "$SIRCUS_ROOT/.venv/bin/python" "$SIRCUS_ROOT/viam_module.py" "$@"
