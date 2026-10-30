#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
TASK_PYTHON=${SIRCUS_PYTHON:-$TASK_ROOT/sircus/simulation/.venv/bin/python}
TASK_MODE=dry_run
if [[ ${1:-} == --hardware ]]; then
  TASK_MODE=hardware
  shift
fi
if ! command -v "$TASK_PYTHON" >/dev/null; then
  echo 'Set SIRCUS_PYTHON to the project Python 3.12 environment or run setup-ubuntu.sh.' >&2
  exit 1
fi
if [[ "$TASK_MODE" == hardware ]]; then
  TASK_DATA=${SIRCUS_PHYSICAL_DATA_ROOT:-$TASK_ROOT/sircus/physical_trials/data/hardware}
else
  TASK_DATA=$TASK_ROOT/.runtime/show/dry_run
fi
# Source synchronization never executes this script. Hardware requires the
# deliberate --hardware argument; numerical workflows stay unavailable there.
exec "$TASK_PYTHON" "$TASK_ROOT/sircus/simulation/server.py" \
  --offline --show --physical-mode "$TASK_MODE" --port "${SIRCUS_SHOW_PORT:-18769}" \
  --physical-data-root "$TASK_DATA" --workflow-data-root "$TASK_ROOT/.runtime/show/workflow" "$@"
