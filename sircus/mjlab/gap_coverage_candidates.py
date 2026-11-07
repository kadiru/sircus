"""Two prespecified gap-endpoint proposals; no new fit or native search.

The original bank samples the BC gap at plus/minus 36 mm. This ablation changes
only those two slots to the existing anchor-relative domain endpoints (plus/
minus 120 mm). Fourteen original proposals and every board pose stay identical.
Unsupported nonpositive endpoints remain literal proposals for downstream
ranking/admission; this module does not clamp, reject or relabel them.
"""
from __future__ import annotations

import copy
from collections.abc import Mapping

import numpy as np

VERSION = "sircus-gap-coverage-candidates-v1"
COUNT = 16
GAP_HALF_WIDTH_M = .12
RESIDUAL_SCALE = np.asarray([.04, .04, .05, 8., .12])
REPLACEMENTS = ((8, -1., "bc_axis_4_-0.3", "lower"),
                (9, 1., "bc_axis_4_0.3", "upper"))


def _vector(value, label):
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("Finite five-dimensional " + label + " required") from error
    if result.shape != (5,) or not np.isfinite(result).all():
        raise ValueError("Finite five-dimensional " + label + " required")
    return result


def candidates(original_items, anchor):
    """Return a copied literal 16-bank with only slots 08/09 gaps replaced."""
    if not isinstance(anchor, Mapping):
        raise ValueError("Original anchor mapping required")
    center = _vector(anchor.get("values"), "anchor")
    if not isinstance(original_items, (list, tuple)) or len(original_items) != COUNT:
        raise ValueError("Exactly sixteen original candidates required")
    for index, row in enumerate(original_items):
        if not isinstance(row, Mapping) or row.get("id") != f"candidate-{index:02d}":
            raise ValueError("Original candidate identities and order required")
        values = _vector(row.get("values"), "candidate values")
        residual = _vector(row.get("normalized_residual"), "normalized residual")
        if (np.abs(residual) > 1.).any():
            raise ValueError("Original anchor-relative residual domain required")
        if not np.allclose(values, center + residual*RESIDUAL_SCALE, atol=1e-12, rtol=0):
            raise ValueError("Original values and anchor-relative residual disagree")
        if "gap_coverage" in row:
            raise ValueError("Original bank must not already contain this ablation")
    for index, sign, source_kind, bound in REPLACEMENTS:
        if original_items[index].get("source_kind") != source_kind:
            raise ValueError("Expected original BC gap-axis slots 08 and 09")
    result = copy.deepcopy(list(original_items))
    for index, sign, _, bound in REPLACEMENTS:
        row = result[index]
        row["values"][4] = float(center[4] + sign*GAP_HALF_WIDTH_M)
        row["normalized_residual"][4] = sign
        row["source_kind"] = "bc_pose_anchor_gap_" + bound + "_endpoint"
        row["gap_coverage"] = dict(version=VERSION,
            original_candidate=copy.deepcopy(original_items[index]),
            anchor_gap_m=float(center[4]), domain_half_width_m=GAP_HALF_WIDTH_M,
            bound=bound, normalized_gap=sign,
            scope="Only gap changed; board XYZ/tilt and original slot identity retained")
    return result
