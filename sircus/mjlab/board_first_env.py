"""Explicit board-first scoring adapter; importing never starts a simulator.

The original held-endpoint physics executes once. Its dense evidence is scored
by the new primary and independent oracle; original labels remain in each row.
No old entrypoint, active policy, contact model, or native timestep is changed.
"""
from __future__ import annotations

import copy
import math
import time

import numpy as np

VERSION = "sircus-board-first-held-endpoint-v4"


def _same(left, right, path="score"):
    if isinstance(left, dict):
        if not isinstance(right, dict) or set(left) != set(right):
            raise ValueError("Independent v4 score keys disagree: " + path)
        for key in left:
            _same(left[key], right[key], path + "/" + key)
    elif isinstance(left, (float, np.floating)):
        if not isinstance(right, (int, float, np.number)) or not math.isclose(left, right, abs_tol=1e-9, rel_tol=0):
            raise ValueError("Independent v4 score disagrees: " + path)
    elif left != right:
        raise ValueError("Independent v4 score disagrees: " + path)


def rescore_native_rows(scene, calibration, rows, arrays):
    """Return copied native rows with old labels preserved and v4 audited scores."""
    from . import calibrated_score as v2
    from . import table_assisted_score as v3
    from . import board_first_score as v4
    from . import board_first_score_oracle as oracle

    positions = np.asarray(arrays["position_m"])
    if positions.ndim != 3 or positions.shape[1:] != (len(rows), 3):
        raise ValueError("Dense native rows and batch count disagree")
    if "record_ids" in arrays and list(arrays["record_ids"]) != [row["id"] for row in rows]:
        raise ValueError("Dense native record identities disagree")
    result = []
    for index, original in enumerate(rows):
        if original.get("native_release") is not True or original.get("scoring_version") not in (v2.VERSION, v3.VERSION):
            raise ValueError("The v4 adapter requires original v2 or v3 native rows")
        if not np.array_equal(positions[0, index], original["native_initial_ball_center_m"]):
            raise ValueError("Native row and recorded release center disagree")
        trace = {key: np.asarray(arrays[key])[:, index] for key in ("position_m", "velocity_m_s", "forces_N")}
        if "contact_distance_m" in arrays:
            trace["contact_distance_m"] = np.asarray(arrays["contact_distance_m"])[:, index]
        for key in ("time_s", "dt_s"):
            if key in arrays:
                trace[key] = np.asarray(arrays[key])
        score = v4.score_trace(scene, original["bucket_xy_m"], original["board_normal"], calibration, trace)
        independent = oracle.score_trace(scene, original["bucket_xy_m"], original["board_normal"], calibration, trace)
        _same(score, independent)
        for key in ("success", "board_bounce", "whole_ball_entry", "max_retention_s", "physics_steps", "invalid_physics"):
            _same(original[key], score[key], "original_capture/" + key)
        row = copy.deepcopy(original)
        row["historical_pre_board_first_score"] = {key: copy.deepcopy(original[key]) for key in original if key in score}
        row.update(score, score_adapter_version=VERSION, scalar_score_audit_passed=True)
        result.append(row)
    return result


class BoardFirstScoringMixin:
    def run(self, *args, **kwargs):
        original, timing = super().run(*args, **kwargs)
        started = time.monotonic()
        rows = rescore_native_rows(self.scene, self.calibration, original, self.last_arrays)
        elapsed = time.monotonic() - started
        updated = dict(timing)
        for key in ("seconds", "score_and_oracle_s"):
            updated[key] = float(updated.get(key, 0.)) + elapsed
        updated["board_first_rescore_s"] = elapsed
        updated["scoring_adapter_version"] = VERSION
        return rows, updated


def make_environment(*args, **kwargs):
    """Construct only on explicit invocation; inherits the 4090-only guard."""
    from .calibrated_env import CalibratedEnvironment

    class BoardFirstEnvironment(BoardFirstScoringMixin, CalibratedEnvironment):
        pass

    return BoardFirstEnvironment(*args, **kwargs)
