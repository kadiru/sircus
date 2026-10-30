#!/bin/sh
set -eu
SIRCUS_TWO_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SIRCUS_TWO_ROOT"
exec "$SIRCUS_TWO_ROOT/../simulation/.venv/bin/python" "$SIRCUS_TWO_ROOT/viam_module.py" "$@"
