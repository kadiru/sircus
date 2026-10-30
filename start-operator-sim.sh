#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
TASK_PYTHON=${SIRCUS_PYTHON:-$TASK_ROOT/sircus/simulation/.venv/bin/python}
if ! command -v "$TASK_PYTHON" >/dev/null; then
  echo "Set up this checkout with ./setup-ubuntu.sh or set SIRCUS_PYTHON to its Python 3.12 environment." >&2
  exit 1
fi
# Dedicated loopback process: never starts a Viam cloud part or a physical adapter.
exec "$TASK_PYTHON" "$TASK_ROOT/sircus/simulation/server.py" --offline --physical-mode disabled --operations --port "${SIRCUS_OPERATOR_PORT:-18766}" "$@"
