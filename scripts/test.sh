#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$TASK_ROOT"
TEST_PYTHON=${SIRCUS_PYTHON:-$TASK_ROOT/sircus/simulation/.venv/bin/python}
command -v "$TEST_PYTHON" >/dev/null || TEST_PYTHON=python3
"$TEST_PYTHON" -m unittest discover -s sircus/physical_trials -p 'test_*.py' -q
cd sircus/two-arm
for test in check_bucket_physics.py check_physics.py check_bucket_native.py check_training_data.py check_guard.py check_orientation_height.py; do
 "$TEST_PYTHON" "$test"
done
