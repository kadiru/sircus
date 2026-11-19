#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$TASK_ROOT"
TEST_PYTHON=${SIRCUS_PYTHON:-$TASK_ROOT/sircus/simulation/.venv/bin/python}
command -v "$TEST_PYTHON" >/dev/null || TEST_PYTHON=python3
"$TEST_PYTHON" -m unittest discover -s sircus/physical_trials -p 'test_*.py' -q
"$TEST_PYTHON" -m unittest discover -s sircus/simulation/tests -p 'test_*.py' -q
"$TEST_PYTHON" -m unittest discover -s sircus/isaac/tests -p 'test_*.py' -q
"$TEST_PYTHON" -m unittest discover -s sircus/mjlab/tests -p 'test_*.py' -q
node --test sircus/simulation/tests/test_study_panel.mjs sircus/isaac/tests/test_preview_diagnostic.mjs sircus/isaac/tests/test_plane_panel.mjs sircus/isaac/tests/test_catalogue_panel.mjs sircus/isaac/tests/test_distance_catalogue_panel.mjs sircus/isaac/tests/test_bucket_grid_panel.mjs sircus/isaac/tests/test_parallel_panel.mjs sircus/isaac/tests/test_feasibility_panel.mjs sircus/mjlab/tests/test_interface_ui.mjs sircus/mjlab/tests/test_expanded_ui.mjs sircus/mjlab/tests/test_stable_ui.mjs sircus/mjlab/tests/test_contact_ui.mjs sircus/mjlab/tests/test_corrected_ui.mjs sircus/mjlab/tests/test_window_ui.mjs
node --test sircus/mjlab/tests/test_neighborhood_ui.mjs
node --test sircus/mjlab/tests/test_distance_ui.mjs
cd sircus/two-arm
for test in check_bucket_physics.py check_physics.py check_bucket_native.py check_training_data.py check_guard.py check_capture_scoring.py check_orientation_height.py check_scene_study.py check_published_study.py; do
 "$TEST_PYTHON" "$test"
done
