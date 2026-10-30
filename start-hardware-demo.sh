#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$TASK_ROOT"
# Stdlib-only preflight. The validated profile chooses the deployment interpreter.
exec python3 -m sircus.physical_trials.run_hardware_demo "$@"
