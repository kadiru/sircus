"""Opt-in rejection short circuit for the unchanged sampled path auditor.

Accepted paths retain every forward/reverse segment, sample, and certificate
field. A rejected path stops after the first failed segment: later segments
cannot turn that same path into an accepted one. This does not change path
ordering, clearance tolerances, joint interpolation, or the sampled scope.
"""
from __future__ import annotations

from types import MethodType

import numpy as np

from . import calibrated_admission as admission


VERSION = "sircus-arm-path-rejection-short-circuit-v1"


def _audit_path_short_circuit(self, points):
    forward, reverse = [], []
    total_segments = 2 * max(0, len(points) - 1)

    def certificate(passed):
        checked = forward + reverse
        result = dict(
            passed=passed,
            waypoints=[np.asarray(q).tolist() for q in points],
            edges=forward,
            reverse_edges=reverse,
            scope=admission.SCOPE,
            minimum_clearance_m=min(
                (edge.get("minimum_clearance_m", 0.) for edge in checked),
                default=0.,
            ),
            max_joint_sample_step_rad=admission.SAMPLE_STEP_RAD,
        )
        if not passed:
            # The partial rejection evidence must never look like a complete
            # audit. Accepted certificates have exactly the historical fields.
            result.update(
                audit_complete=len(checked) == total_segments,
                stopped_after_failure=True,
                unchecked_segment_count=total_segments - len(checked),
            )
        return result

    for sequence, destination in ((points, forward), (points[::-1], reverse)):
        for first, second in zip(sequence, sequence[1:]):
            edge = admission.audit_segment(
                self.gate, first, second, max_step_rad=admission.SAMPLE_STEP_RAD,
            )
            destination.append(edge)
            if not edge["passed"]:
                return certificate(False)
    return certificate(True)


def install_path_short_circuit(planner):
    """Install on one unmodified SerializedPlanner instance; return that instance.

    No class/global patch is made. Repeated installation is harmless; an
    unrelated custom auditor is rejected rather than silently overwritten.
    """
    if not isinstance(planner, admission.SerializedPlanner):
        raise TypeError("A SerializedPlanner instance is required")
    current = getattr(planner._audit_path, "__func__", None)
    if current is _audit_path_short_circuit:
        return planner
    if current is not admission.SerializedPlanner._audit_path:
        raise ValueError("Refusing to replace a custom path auditor")
    planner._audit_path = MethodType(_audit_path_short_circuit, planner)
    return planner
