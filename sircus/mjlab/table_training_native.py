"""Explicit v3 requested-action bridge; historical v2 engine stays untouched.

Only scoring changes. Geometry resolution, exact requested action/branch proofs,
native options and 64-substep held-endpoint physics keep their original guards.
The caller owns the shared 4090 lease and finite experiment budget.
"""
import copy
import math
from pathlib import Path

from .calibrated_adapter import CalibratedAdapter
from .calibrated_admission import source_geometry
from .expanded_actions import WIDE_BOUNDS
from .policy64_native import Engine as HistoricalEngine, _vector
from .table_assisted_env import make_environment, VERSION as ADAPTER_VERSION
from . import table_assisted_score as score

VERSION = "sircus-table-assisted-training-native-v1"
SCORER_VERSION = score.VERSION


class Engine(HistoricalEngine):
    def __init__(self, ref, assets, directory, *, prior, calibration, home, batch=32):
        if type(batch) is not int or not 1 <= batch <= 32:
            raise ValueError("Bounded native batch1..32 required")
        if not isinstance(prior, dict) or not isinstance(prior.get("records"), list) or not prior["records"]:
            raise ValueError("Nonempty frozen shared prior required")
        self.sources = [source_geometry(r) for r in copy.deepcopy(prior["records"])]
        if len({r["id"] for r in self.sources}) != len(self.sources):
            raise ValueError("Shared source IDs must be unique")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        config = copy.deepcopy(calibration.get("score_config", calibration))
        initial = copy.deepcopy(home.get("home", home))
        # The factory is supplied at construction. No v2 environment is first
        # created or invisibly swapped after an engine identity was established.
        self.adapter = CalibratedAdapter(
            ref, assets, self.directory, prior_records=self.sources,
            calibration=config, home=initial, batch=batch,
            environment_factory=lambda: make_environment(
                ref["scene"], assets, config, n=batch, bounds=WIDE_BOUNDS, substeps=64,
            ),
        )
        self._anchors = {}
        self._effective_options_sha256 = None

    def evaluate(self, requests, phase):
        if not isinstance(phase, str) or not phase:
            raise ValueError("Nonempty execution phase required")
        normalized = []
        for request in requests:
            value = copy.deepcopy(request)
            value["bucket_xy_m"] = _vector(value.get("bucket_xy_m"), 2, "bucket XY")
            value["values"] = _vector(value.get("values"), 5, "action values")
            normalized.append(value)
        self._freeze_effective_options()
        rows, timing = self.adapter.evaluate(normalized, phase)
        if len(rows) != len(normalized):
            raise RuntimeError("Adapter changed requested trial count")
        effective = self._freeze_effective_options(required=any(r.get("native_release") for r in rows))
        result = []
        for measured, request in zip(rows, normalized):
            if (measured.get("id") != request.get("id")
                or measured.get("bucket_xy_m") != request["bucket_xy_m"]
                or measured.get("values") != request["values"]):
                raise RuntimeError("Adapter replaced or reordered an exact requested action")
            if measured.get("action_substituted") is not False:
                raise RuntimeError("Action substitution is forbidden")
            if measured.get("native_release"):
                if (measured.get("contact_protocol", {}).get("substeps") != 64
                    or measured.get("scoring_version") != score.VERSION
                    or measured.get("route_version") != score.ROUTE_VERSION
                    or measured.get("reward_version") != score.REWARD_VERSION
                    or measured.get("score_adapter_version") != ADAPTER_VERSION
                    or measured.get("scalar_score_audit_passed") is not True
                    or measured.get("historical_v2_score", {}).get("scoring_version") != score.CONTAINMENT_VERSION):
                    raise RuntimeError("Native release differs from pinned64 explicit v3 scoring")
                # Reject mixed reward/route semantics even if version strings
                # were accidentally copied onto an older native row.
                expected_reward = score.reward(measured)
                recorded_reward = measured.get("reward")
                if (not isinstance(recorded_reward, (int, float))
                    or not math.isfinite(expected_reward) or not math.isfinite(recorded_reward)
                    or abs(expected_reward - recorded_reward) > 1e-9):
                    raise RuntimeError("Native v3 reward disagrees with route evidence")
            row = dict(request)
            row.update(copy.deepcopy(measured))
            row.update(adapter_directory="native", training_engine_version=VERSION)
            if measured.get("native_release"):
                row.update(effective)
            result.append(row)
        return result, timing
