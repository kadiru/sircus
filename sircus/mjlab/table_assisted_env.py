"""Explicit future-training adapter; importing this module never starts a GPU.

The existing held-endpoint rollout runs once, unchanged. This opt-in mixin then
scores its dense observations with the v3 primary scorer and independent oracle.
Old training entrypoints and frozen contracts continue to use v2 until a new
experiment explicitly selects this adapter and the v3 reward function.
"""
from __future__ import annotations

import copy
import math
import time

import numpy as np

VERSION = "sircus-table-assisted-held-endpoint-v3"


def _same(left, right, path="score"):
    if isinstance(left, dict):
        if not isinstance(right, dict) or set(left) != set(right):
            raise ValueError("Independent v3 score keys disagree: " + path)
        for key in left:
            _same(left[key], right[key], path + "/" + key)
    elif isinstance(left, (float, np.floating)):
        if not isinstance(right, (int, float, np.number)) or not math.isclose(left, right, abs_tol=1e-9, rel_tol=0):
            raise ValueError("Independent v3 score disagrees: " + path)
    elif left != right:
        raise ValueError("Independent v3 score disagrees: " + path)


def rescore_native_rows(scene, calibration, rows, arrays):
    """Copy scored rows, retaining their original labels and native identities."""
    from . import calibrated_score as v2
    from . import table_assisted_score as v3
    from . import table_assisted_score_oracle as oracle

    count = len(rows)
    positions = np.asarray(arrays["position_m"])
    if positions.ndim != 3 or positions.shape[1:] != (count, 3):
        raise ValueError("Dense native rows and batch count disagree")
    result = []
    for index, original in enumerate(rows):
        if original.get("native_release") is not True or original.get("scoring_version") != v2.VERSION:
            raise ValueError("The v3 adapter requires original v2 native rows")
        if not np.array_equal(positions[0, index], original["native_initial_ball_center_m"]):
            raise ValueError("Native row and recorded release center disagree")
        trace = {key: (value[:, index] if value.ndim >= 2 and key != "time_s" else value)
                 for key, value in arrays.items()}
        scored = v3.score_trace(scene, original["bucket_xy_m"], original["board_normal"], calibration, trace)
        independent = oracle.score_trace(scene, original["bucket_xy_m"], original["board_normal"], calibration, trace)
        _same(scored, independent)
        # These are unchanged capture/validity facts, not the route/reward choice.
        for key in ("success", "board_bounce", "whole_ball_entry", "max_retention_s", "physics_steps", "invalid_physics"):
            _same(original[key], scored[key], "original_capture/" + key)
        old_keys = ("scoring_version", "route_version", "reward_version", "success", "complete_success",
                    "route_success", "route_quality", "single_bounce_success", "external_contact_steps", "reward",
                    "invalid", "invalid_physics", "physics_steps")
        row = copy.deepcopy(original)
        row["historical_v2_score"] = {key: copy.deepcopy(original[key]) for key in old_keys}
        row.update(scored, score_adapter_version=VERSION, scalar_score_audit_passed=True)
        result.append(row)
    return result


class TableAssistedScoringMixin:
    def run(self, *args, **kwargs):
        original, timing = super().run(*args, **kwargs)
        started = time.monotonic()
        rows = rescore_native_rows(self.scene, self.calibration, original, self.last_arrays)
        elapsed = time.monotonic() - started
        updated = dict(timing)
        for key in ("seconds", "score_and_oracle_s"):
            updated[key] = float(updated.get(key, 0.)) + elapsed
        updated["table_assisted_rescore_s"] = elapsed
        updated["scoring_adapter_version"] = VERSION
        return rows, updated


def make_environment(*args, **kwargs):
    """Construct only on explicit invocation; inherits the 4090-only guard."""
    from .calibrated_env import CalibratedEnvironment

    class TableAssistedEnvironment(TableAssistedScoringMixin, CalibratedEnvironment):
        pass

    return TableAssistedEnvironment(*args, **kwargs)
