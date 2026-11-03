"""CPU endpoint admission near several geometry-only continuation anchors.

The nearest height/tilt anchor is selected before any release outcomes. Every
context keeps one exact board branch; release-gap queries never change it.
No native dynamics, transit or actuator tracking is established here.
"""

import copy
from pathlib import Path

import numpy as np

from .expand_geometry import validate_bounds
from .learning_contract import digest
from .window_contract import BOUNDS
from .window_geometry import WindowGeometry

VERSION = "sircus-multi-anchor-height-tilt-geometry-v1"
LOCAL_VERSION = "sircus-nominal-branch-local-variation-v1"
ANCHOR_SCALES = (.04, 4.)
MAX_LOCAL_JOINT_DELTA_RAD = .10
GEOMETRY_KEYS = ("values", "bucket_xy_m", "q10", "q14")
OUTCOME_KEYS = frozenset(("route_success", "success", "complete_success", "reward",
    "learning_reward", "label", "outcome", "captures", "whole_ball_entry"))


def _geometry(value):
    result = {}
    for name, shape in (("values", (5,)), ("bucket_xy_m", (2,)), ("q10", (6,)), ("q14", (6,))):
        array = np.asarray(value.get(name, []), float)
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError("Finite complete geometry required: " + name)
        result[name] = array.tolist()
    return result


def _context(raw):
    if OUTCOME_KEYS.intersection(raw):
        raise ValueError("Contexts cannot contain outcome labels")
    name = raw.get("id")
    if not isinstance(name, str) or not name:
        raise ValueError("A nonempty context ID is required")
    point = np.asarray([raw.get("board_height_m"), raw.get("tilt_deg")], float)
    if not np.isfinite(point).all():
        raise ValueError("Finite context height and tilt required")
    return dict(id=name, board_height_m=float(point[0]), tilt_deg=float(point[1]))


def nearest_anchor_id(height, tilt, anchors):
    """Geometry-only, normalized squared distance with stable ID tie-breaking."""
    point = np.asarray([height, tilt], float)
    if point.shape != (2,) or not np.isfinite(point).all() or not anchors:
        raise ValueError("Finite height/tilt and at least one anchor required")
    return min(anchors, key=lambda a: (
        float(np.sum(((point - np.asarray(a["values"])[2:4]) / ANCHOR_SCALES) ** 2)),
        a["id"],
    ))["id"]


class NeighborhoodGeometry:
    """Reuse one CPU ``WindowGeometry`` instance per immutable source anchor.

    ``anchors`` accepts exactly ``id, values, bucket_xy_m, q10, q14``. Callers
    must discard historical labels/provenance before construction. Metadata on
    contexts is ignored, while explicit outcome labels are rejected.
    ``geometry_factory`` is an injection point for lightweight protocol tests.
    """

    def __init__(self, scene, assets, anchors, bounds=None, *, geometry_factory=None):
        self.scene, self.assets = copy.deepcopy(scene), Path(assets)
        self.bounds = copy.deepcopy(BOUNDS if bounds is None else bounds)
        low, high = validate_bounds(self.bounds)
        self.anchors = []
        for raw in anchors:
            if set(raw) != {"id", *GEOMETRY_KEYS} or not isinstance(raw["id"], str) or not raw["id"]:
                raise ValueError("Strict geometry-only anchors with nonempty IDs required")
            anchor = dict(id=raw["id"], **_geometry(raw))
            if not ((np.asarray(anchor["values"]) >= low).all() and (np.asarray(anchor["values"]) <= high).all()):
                raise ValueError("Continuation anchor outside declared bounds")
            self.anchors.append(anchor)
        if not self.anchors or len({a["id"] for a in self.anchors}) != len(self.anchors):
            raise ValueError("Distinct source anchors are required")
        self.anchors.sort(key=lambda a: a["id"])
        if len({tuple(a["values"][2:4]) for a in self.anchors}) != len(self.anchors):
            raise ValueError("Height/tilt anchors must be distinct")
        first = self.anchors[0]
        if any(not np.array_equal(a[k], first[k]) for a in self.anchors for k in ("bucket_xy_m",)) or any(
                not np.array_equal(a["values"][:2], first["values"][:2]) for a in self.anchors):
            raise ValueError("All anchors must share exact board and bucket XY")
        self._anchor = {a["id"]: a for a in self.anchors}
        self._factory = WindowGeometry if geometry_factory is None else geometry_factory
        self._pools, self._contexts, self._assignments = {}, {}, {}

    @property
    def contexts(self):
        return copy.deepcopy(list(self._contexts.values()))

    @property
    def assignments(self):
        return copy.deepcopy(self._assignments)

    def _pool(self, anchor_id):
        if anchor_id not in self._pools:
            self._pools[anchor_id] = self._factory(self.scene, self.assets,
                copy.deepcopy(self._anchor[anchor_id]), bounds=copy.deepcopy(self.bounds))
        return self._pools[anchor_id]

    def _decorate(self, row, anchor_id):
        return dict(copy.deepcopy(row), source_anchor_id=anchor_id,
            source_anchor_geometry_sha256=digest(self._anchor[anchor_id]), wrapper_version=VERSION)

    def prepare_contexts(self, contexts, progress=None):
        declared = [_context(raw) for raw in contexts]
        if len({c["id"] for c in declared}) != len(declared):
            raise ValueError("Duplicate context ID")
        # Validate the entire request before assigning/reusing any child context.
        for c in declared:
            if c["id"] in self._contexts and any(c[k] != self._contexts[c["id"]][k]
                    for k in ("board_height_m", "tilt_deg")):
                raise ValueError("A prepared context cannot change geometry")
        groups = {}
        for c in declared:
            anchor_id = nearest_anchor_id(c["board_height_m"], c["tilt_deg"], self.anchors)
            groups.setdefault(anchor_id, []).append(c)
        for anchor_id in sorted(groups):
            group = groups[anchor_id]
            child = self._pool(anchor_id).prepare_contexts(group)
            if len(child) != len(group) or [r["id"] for r in child] != [r["id"] for r in group]:
                raise RuntimeError("Child geometry changed context association")
            for c, row in zip(group, child):
                anchor = self._anchor[anchor_id]
                exact = [c["board_height_m"], c["tilt_deg"]] == anchor["values"][2:4]
                if row.get("board_admitted") and exact and not np.array_equal(row.get("q14"), anchor["q14"]):
                    raise RuntimeError("Exact anchor board branch was substituted")
                decorated = self._decorate(row, anchor_id)
                self._contexts[c["id"]], self._assignments[c["id"]] = decorated, anchor_id
                if progress:
                    progress(copy.deepcopy(decorated))
        return copy.deepcopy([self._contexts[c["id"]] for c in declared])

    def audit(self, context_ids, gaps, progress=None):
        names, gaps = list(context_ids), np.asarray(gaps, float)
        if gaps.shape != (len(names),) or not np.isfinite(gaps).all():
            raise ValueError("One finite release gap per context ID required")
        if any(name not in self._contexts for name in names):
            raise ValueError("Prepare every context before auditing gaps")
        groups = {}
        for name, gap in zip(names, gaps):
            # Resolve each exact request once, including duplicates in one batch.
            key = (name, float(gap).hex())
            groups.setdefault(self._assignments[name], {})[key] = float(gap)
        result = {}
        for anchor_id in sorted(groups):
            requests = groups[anchor_id]
            keys = list(requests)
            child = self._pool(anchor_id).audit([k[0] for k in keys], list(requests.values()))
            if len(child) != len(keys):
                raise RuntimeError("Child geometry changed query count")
            for key, row in zip(keys, child):
                context = self._contexts[key[0]]
                expected = self._anchor[anchor_id]["values"][:2] + [context["board_height_m"], context["tilt_deg"], requests[key]]
                if row.get("context_id") != key[0] or not np.array_equal(row.get("values"), expected):
                    raise RuntimeError("Child geometry changed query identity")
                if row.get("eligible") and not np.array_equal(row.get("q14"), context["q14"]):
                    raise RuntimeError("Release gap changed frozen board branch")
                anchor = self._anchor[anchor_id]
                if row.get("eligible") and np.array_equal(expected, anchor["values"]) and any(
                        not np.array_equal(row.get(k), anchor[k]) for k in ("q10", "q14")):
                    raise RuntimeError("Exact anchor joint branch was substituted")
                result[key] = self._decorate(row, anchor_id)
        rows = [copy.deepcopy(result[(name, float(gap).hex())]) for name, gap in zip(names, gaps)]
        if progress:
            for row in rows:
                progress(copy.deepcopy(row))
        return rows


def audit_local_variations(scene, assets, nominal_row, requests, bounds=None, *, geometry_factory=None):
    """Audit local H/tilt/gap perturbations using only nominal joint geometry.

    Requests need ``context_id`` (or ``id``) and either finite ``values[5]`` or
    ``board_height_m, tilt_deg, gap_m``. Shared IDs require identical H/tilt.
    Passed endpoints farther than .10rad in either arm are retained as rejected
    requests. Raw joint deltas are used, so a 2pi branch jump cannot be hidden.
    This continuity gate is not a path or actuator certificate.
    """
    nominal = _geometry(nominal_row)
    if nominal_row.get("eligible") is False or nominal_row.get("invalid"):
        raise ValueError("An admitted nominal geometry is required")
    parsed, contexts = [], {}
    for raw in requests:
        if OUTCOME_KEYS.intersection(raw):
            raise ValueError("Variation requests cannot contain outcome labels")
        name = raw.get("context_id", raw.get("id"))
        values = np.asarray(raw.get("values", nominal["values"][:2] + [raw.get("board_height_m"), raw.get("tilt_deg"), raw.get("gap_m")]), float)
        if not isinstance(name, str) or not name or values.shape != (5,) or not np.isfinite(values).all():
            raise ValueError("Finite complete variation request and context ID required")
        if not np.array_equal(values[:2], nominal["values"][:2]):
            raise ValueError("Local variations must preserve board XY")
        if any(k in raw and float(raw[k]) != values[i] for k, i in (("board_height_m", 2), ("tilt_deg", 3), ("gap_m", 4))):
            raise ValueError("Conflicting variation coordinates")
        c = dict(id=name, board_height_m=float(values[2]), tilt_deg=float(values[3]))
        if name in contexts and contexts[name] != c:
            raise ValueError("A variation context cannot change height or tilt")
        contexts[name] = c; parsed.append((name, float(values[4])))
    if not parsed:
        return []
    factory = WindowGeometry if geometry_factory is None else geometry_factory
    geometry = factory(copy.deepcopy(scene), Path(assets), nominal,
        bounds=copy.deepcopy(BOUNDS if bounds is None else bounds))
    geometry.prepare_contexts(list(contexts.values()))
    # Unique-query resolution prevents duplicates from acquiring another branch.
    unique = dict.fromkeys(parsed)
    child = geometry.audit([k[0] for k in unique], [k[1] for k in unique])
    if len(child) != len(unique):
        raise RuntimeError("Local geometry changed query count")
    rows = {}
    for (name, gap), original in zip(unique, child):
        row = copy.deepcopy(original)
        expected = nominal["values"][:2] + [contexts[name]["board_height_m"], contexts[name]["tilt_deg"], gap]
        if row.get("context_id") != name or not np.array_equal(row.get("values"), expected):
            raise RuntimeError("Local geometry changed requested coordinates")
        before = bool(row.get("eligible")); delta = {k: None for k in ("q10", "q14")}
        if before:
            for k in delta:
                q = np.asarray(row.get(k, []), float)
                if q.shape != (6,) or not np.isfinite(q).all():
                    raise RuntimeError("Admitted local geometry lacks finite joint arrays")
                delta[k] = float(np.max(np.abs(q - nominal[k])))
            if np.array_equal(expected, nominal["values"]) and any(delta[k] != 0 for k in delta):
                row.update(eligible=False, native_release=False, reason="nominal_branch_substituted")
            elif any(value > MAX_LOCAL_JOINT_DELTA_RAD for value in delta.values()):
                row.update(eligible=False, native_release=False, reason="branch_discontinuity")
        row.update(wrapper_version=LOCAL_VERSION, nominal_geometry_sha256=digest(nominal),
            endpoint_eligible_before_continuity_gate=before,
            local_variation_gate=dict(version=LOCAL_VERSION, max_joint_delta_rad=MAX_LOCAL_JOINT_DELTA_RAD,
                arm10_max_abs_joint_delta_rad=delta["q10"], arm14_max_abs_joint_delta_rad=delta["q14"],
                passed=bool(row.get("eligible"))))
        rows[(name, gap)] = row
    return [copy.deepcopy(rows[key]) for key in parsed]
