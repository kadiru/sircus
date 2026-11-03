"""CPU-only admission for bucket-distance development requests.

The complete outer bucket footprint is checked before constructing an IK model.
Moving a bucket along its original bearing retains an exact source board branch
at the same height and tilt. A changed bearing uses those joints as a numerical
hint only: a new full-pose, strictly checked board branch must be frozen before
release-gap queries. Neither endpoint admission nor tabletop fit proves a shot,
swept path, actuator tracking, or generalization.
"""

import copy
from pathlib import Path

import numpy as np

from .expand_geometry import validate_bounds
from .learning_contract import digest
from .learning_ik import shot_targets
from .neighborhood_geometry import GEOMETRY_KEYS, OUTCOME_KEYS, _geometry
from .neighborhood_geometry import audit_local_variations as _audit_local_variations
from .window_contract import BOUNDS
from .window_geometry import WindowGeometry, _ordered, _pool

VERSION = "sircus-bucket-distance-frozen-branch-geometry-v1"
LOCAL_VERSION = "sircus-bucket-distance-local-variation-v1"
FOOTPRINT_VERSION = "sircus-full-outer-bucket-table-footprint-v1"
EDGE_MARGIN_M = .005
BEARING_TOLERANCE = 1e-12
FOOTPRINT_TOLERANCE_M = 1e-12
SCOPE = "Strict CPU endpoints and tabletop footprint; no ball-flight or swept-path guarantee"


def table_footprint(scene, bucket_xy, edge_margin_m=EDGE_MARGIN_M):
    """Signed full-radius clearance against every tabletop XY edge, no clipping.

    The bucket is upright. Its largest outer diameter, not the inner opening or
    bottom radius, determines robot/table clearance. A zero signed margin means
    the outer surface touches an edge; ``passed`` requires the additional margin.
    """
    point = np.asarray(bucket_xy, float)
    center = np.asarray(scene["table"]["center_m"][:2], float)
    size = np.asarray(scene["table"]["size_m"][:2], float)
    diameters = np.asarray([scene["bucket"]["outer_top_diameter_m"],
        scene["bucket"]["outer_bottom_diameter_m"]], float)
    margin = float(edge_margin_m)
    if (point.shape != (2,) or center.shape != (2,) or size.shape != (2,)
            or not np.isfinite(np.r_[point, center, size, diameters, margin]).all()
            or (size <= 0).any() or (diameters <= 0).any() or margin < 0):
        raise ValueError("Finite positive table/bucket geometry and nonnegative margin required")
    radius = float(max(diameters) / 2)
    low, high = center - size / 2, center + size / 2
    edges = np.r_[point - low - radius, high - point - radius]
    clearance = float(min(edges))
    return dict(version=FOOTPRINT_VERSION, bucket_xy_m=point.tolist(),
        outer_radius_m=radius, table_min_xy_m=low.tolist(), table_max_xy_m=high.tolist(),
        required_edge_margin_m=margin, edge_clearances_m=edges.tolist(),
        minimum_edge_clearance_m=clearance, numerical_tolerance_m=FOOTPRINT_TOLERANCE_M,
        passed=bool(clearance >= margin - FOOTPRINT_TOLERANCE_M))


def ray_limits(scene, board_xy, ray_unit, edge_margin_m=EDGE_MARGIN_M):
    """Closed nonnegative distance interval whose full bucket fits the table.

    This is a footprint calculation only; the returned distance interval does
    not establish robot reachability or that any release will score.
    """
    board, ray = np.asarray(board_xy, float), np.asarray(ray_unit, float)
    if (board.shape != (2,) or ray.shape != (2,) or not np.isfinite(np.r_[board, ray]).all()
            or abs(np.linalg.norm(ray) - 1) > 1e-12):
        raise ValueError("Finite board XY and an explicitly unit ray required")
    footprint = table_footprint(scene, board, edge_margin_m)
    padding = footprint["outer_radius_m"] + footprint["required_edge_margin_m"]
    low = np.asarray(footprint["table_min_xy_m"]) + padding
    high = np.asarray(footprint["table_max_xy_m"]) - padding
    lower, upper = 0., float("inf")
    for point, direction, lo, hi in zip(board, ray, low, high):
        if abs(direction) < 1e-15:
            if not lo <= point <= hi:
                return dict(minimum_distance_m=None, maximum_distance_m=None, feasible=False,
                    allowed_bucket_center_min_xy_m=low.tolist(), allowed_bucket_center_max_xy_m=high.tolist(),
                    footprint_version=FOOTPRINT_VERSION, required_edge_margin_m=float(edge_margin_m))
            continue
        first, last = sorted(((lo - point) / direction, (hi - point) / direction))
        lower, upper = max(lower, float(first)), min(upper, float(last))
    valid = bool(lower <= upper and upper >= 0 and (low <= high).all())
    return dict(minimum_distance_m=lower if valid else None,
        maximum_distance_m=upper if valid else None, feasible=valid,
        allowed_bucket_center_min_xy_m=low.tolist(), allowed_bucket_center_max_xy_m=high.tolist(),
        footprint_version=FOOTPRINT_VERSION, required_edge_margin_m=float(edge_margin_m))


def same_bearing(board_xy, first_bucket, second_bucket):
    """Positive collinearity with an explicit numerical tolerance, not a ray fit."""
    a, b = np.asarray(first_bucket, float) - board_xy, np.asarray(second_bucket, float) - board_xy
    norms = np.linalg.norm(a) * np.linalg.norm(b)
    if not np.isfinite(norms) or norms <= 0:
        return False
    return bool(np.dot(a, b) > 0 and abs(a[0] * b[1] - a[1] * b[0]) <= BEARING_TOLERANCE * norms)


def _context(raw, anchors):
    if OUTCOME_KEYS.intersection(raw):
        raise ValueError("Distance contexts cannot contain outcome labels")
    name, source = raw.get("id"), raw.get("source_anchor_id")
    if (not isinstance(name, str) or not name or not isinstance(source, str)
            or source not in anchors):
        raise ValueError("A context ID and a declared source_anchor_id are required")
    xy = np.asarray(raw.get("bucket_xy_m", []), float)
    point = np.asarray([raw.get("board_height_m"), raw.get("tilt_deg")], float)
    if xy.shape != (2,) or not np.isfinite(np.r_[xy, point]).all():
        raise ValueError("Finite bucket XY, board height and tilt required")
    if "board_xy_m" in raw and not np.array_equal(raw["board_xy_m"], anchors[source]["values"][:2]):
        raise ValueError("Distance contexts must retain exact source board XY")
    result = dict(id=name, source_anchor_id=source, bucket_xy_m=xy.tolist(),
        board_height_m=float(point[0]), tilt_deg=float(point[1]))
    for key in ("series_id", "distance_id", "ray_id"):
        if key in raw:
            if not isinstance(raw[key], str) or not raw[key]:
                raise ValueError("Nonempty " + key + " required")
            result[key] = raw[key]
    if "distance_m" in raw:
        distance = float(raw["distance_m"])
        measured = float(np.linalg.norm(xy - anchors[source]["values"][:2]))
        if not np.isfinite(distance) or distance <= 0 or abs(distance - measured) > 1e-12:
            raise ValueError("Declared distance must match the exact requested bucket XY")
        result["distance_m"] = distance
    return result


def _board_workspace_prior(scene, bucket_xy, values):
    """Only board reach is context-fixed; source gap cannot reject other gaps."""
    point = np.asarray(values, float)
    direction = np.asarray(bucket_xy, float) - point[:2]
    distance = np.linalg.norm(direction)
    normal = np.r_[direction / distance * np.sin(np.deg2rad(point[3])),
        np.cos(np.deg2rad(point[3]))]
    wrist = point[:3] - (float(scene["board"]["tool_offset_m"]) + .09) * normal
    return bool(np.linalg.norm(wrist - np.asarray([scene["arms"]["base_spacing_m"], 0., .364])) <= .842)


def _derive_rotated_anchor(geometry, anchor, context):
    """A new pose is proved before it becomes an exact numerical control.

    ``WindowGeometry`` intentionally preserves an exact H/T anchor branch. A
    changed heading is a different target even when H/T is unchanged, so simply
    putting old joints beside a new bucket would reject or mislabel that branch.
    Derive the new target explicitly with preservation disabled, then verify it
    using the same common-park clearance and independent full-pose comparison.
    """
    values = np.asarray(anchor["values"], float).copy()
    values[2:4] = [context["board_height_m"], context["tilt_deg"]]
    xy = np.asarray(context["bucket_xy_m"], float)
    targets = shot_targets(geometry.scene, xy[None], values[None, :3], values[None, 3], values[None, 4])
    candidates = _pool(geometry.ik.board.solve_candidates(targets["board_flange_local"],
        geometry.initials["board_joint_rad"], known=np.asarray(anchor["q14"]),
        starts=16, iterations=180, preserve_known=np.asarray([False])))
    current = {k: candidates[k][0] for k in ("q", "valid", "residual")}
    proof = dict(version=VERSION, source_anchor_geometry_sha256=digest(anchor),
        bucket_xy_m=xy.tolist(), board_world=targets["board_world"][0].tolist(),
        board_query_sha256=candidates["query_sha256"][0], preserve_known=False,
        old_joints_role="numerical_continuation_only", branch_checks=[], admitted=False)
    for branch in _ordered(current, np.asarray(anchor["q14"])):
        q14 = current["q"][branch]
        checked = geometry._check_board(q14, targets["board_world"][0])
        proof["branch_checks"].append(dict(branch=branch, **checked))
        if checked.get("passed"):
            derived = copy.deepcopy(anchor)
            derived.update(values=values.tolist(), bucket_xy_m=xy.tolist(), q14=q14.tolist())
            proof.update(admitted=True, q14=q14.tolist(), board_branch_id=branch,
                derived_anchor_geometry_sha256=digest(derived))
            return derived, proof
    return None, proof


class DistanceGeometry:
    """Freeze each source/bucket/H/T context before reading any shot outcomes.

    Source anchors are strict geometry-only dictionaries. Context fields are
    ``id, source_anchor_id, bucket_xy_m, board_height_m, tilt_deg`` plus optional
    ``series_id, distance_id, ray_id, distance_m``. No source geometry is replaced in place. Context IDs, exact
    requests and continuation hints are stable under reordered queries.
    """

    def __init__(self, scene, assets, anchors, bounds=None, *, geometry_factory=None,
                 rotated_anchor_builder=None):
        self.scene, self.assets = copy.deepcopy(scene), Path(assets)
        self.bounds = copy.deepcopy(BOUNDS if bounds is None else bounds)
        self.low, self.high = validate_bounds(self.bounds)
        self._anchors = {}
        for raw in anchors:
            if (set(raw) != {"id", *GEOMETRY_KEYS} or not isinstance(raw["id"], str)
                    or not raw["id"] or raw["id"] in self._anchors):
                raise ValueError("Distinct strict geometry-only source anchors required")
            anchor = dict(id=raw["id"], **_geometry(raw))
            values = np.asarray(anchor["values"])
            if not ((values >= self.low).all() and (values <= self.high).all()):
                raise ValueError("Source anchor outside action bounds")
            self._anchors[raw["id"]] = anchor
        if not self._anchors:
            raise ValueError("At least one source anchor required")
        first = next(iter(self._anchors.values()))
        self.board_xy = np.asarray(first["values"][:2], float)
        if any(not np.array_equal(a["values"][:2], self.board_xy)
               or not np.array_equal(a["bucket_xy_m"], first["bucket_xy_m"])
               for a in self._anchors.values()):
            raise ValueError("Source anchors must share exact board and original bucket XY")
        self._factory = WindowGeometry if geometry_factory is None else geometry_factory
        self._rotated_builder = _derive_rotated_anchor if rotated_anchor_builder is None else rotated_anchor_builder
        self._contexts, self._declared, self._pools, self._cache = {}, {}, {}, {}
        self._shared = {}

    @property
    def contexts(self):
        return copy.deepcopy(list(self._contexts.values()))

    def _make_pool(self, anchor):
        kwargs = dict(bounds=copy.deepcopy(self.bounds))
        # CPU kinematics/collision models are reusable; every check explicitly
        # supplies the actual bucket. Requests are processed sequentially.
        if self._factory is WindowGeometry:
            kwargs.update(self._shared)
        pool = self._factory(self.scene, self.assets, copy.deepcopy(anchor), **kwargs)
        if self._factory is WindowGeometry and not self._shared:
            self._shared = dict(ik=pool.ik, gate=pool.gate, initials=pool.initials)
        return pool

    def _decorate(self, row, context):
        source = self._anchors[context["source_anchor_id"]]
        same = same_bearing(self.board_xy, source["bucket_xy_m"], context["bucket_xy_m"])
        exact_board = same and [context["board_height_m"], context["tilt_deg"]] == source["values"][2:4]
        output = dict(copy.deepcopy(row), source_anchor_id=source["id"],
            source_anchor_geometry_sha256=digest(source), wrapper_version=VERSION,
            bucket_footprint=table_footprint(self.scene, context["bucket_xy_m"]),
            same_source_bearing=same,
            source_board_hint_role="exact_source_branch_required" if exact_board else "numerical_continuation_only",
            source_board_branch_preserved=bool(exact_board and row.get("board_branch_sha256")
                and np.array_equal(row.get("q14"), source["q14"])),
            bearing_comparison_tolerance=BEARING_TOLERANCE)
        for key in ("series_id", "distance_id", "ray_id", "distance_m"):
            if key in context:
                output[key] = context[key]
        return output

    def prepare_contexts(self, contexts, progress=None):
        declared = [_context(raw, self._anchors) for raw in contexts]
        if len({c["id"] for c in declared}) != len(declared):
            raise ValueError("Duplicate context ID")
        for context in declared:
            if context["id"] in self._declared and context != self._declared[context["id"]]:
                raise ValueError("A prepared context cannot change geometry or series")
        for context in sorted(declared, key=lambda c: c["id"]):
            name = context["id"]
            if name in self._contexts:
                continue
            source = self._anchors[context["source_anchor_id"]]
            values = source["values"][:2] + [context["board_height_m"], context["tilt_deg"], source["values"][4]]
            footprint = table_footprint(self.scene, context["bucket_xy_m"])
            within = bool((np.asarray(values[:4]) >= self.low[:4]).all()
                and (np.asarray(values[:4]) <= self.high[:4]).all())
            distance = float(np.linalg.norm(np.asarray(context["bucket_xy_m"]) - self.board_xy))
            reason = ("bucket_outer_footprint_off_table" if not footprint["passed"] else
                "context_bounds_rejected" if not within else "bucket_distance_prior_reject" if distance <= .15 else None)
            base = dict(id=name, board_height_m=context["board_height_m"], tilt_deg=context["tilt_deg"],
                bucket_xy_m=context["bucket_xy_m"], prepared=True, board_admitted=False,
                q14=None, board_branch_sha256=None, reason=reason, scope=SCOPE)
            if reason is None:
                # Reject impossible coarse workspace requests before model/IK
                # allocation; keep their requested positions in the evidence.
                if not _board_workspace_prior(self.scene, context["bucket_xy_m"], values):
                    base["reason"] = "workspace_prior_reject"
                else:
                    anchor = copy.deepcopy(source)
                    anchor["bucket_xy_m"] = copy.deepcopy(context["bucket_xy_m"])
                    pool = self._make_pool(anchor)
                    if not same_bearing(self.board_xy, source["bucket_xy_m"], context["bucket_xy_m"]):
                        derived, derivation = self._rotated_builder(pool, copy.deepcopy(source), copy.deepcopy(context))
                        base["rotated_board_derivation"] = derivation
                        if derived is None:
                            base["reason"] = "rotated_board_ik_unresolved"
                        else:
                            # The supplied dropper hint still describes the
                            # source H/G. A rotated derivation may not silently
                            # change H/G and turn it into an exact old control.
                            if not np.array_equal(derived["values"], source["values"]):
                                if not np.array_equal(derived["values"][:2], source["values"][:2]):
                                    raise RuntimeError("Rotated branch changed board XY")
                                # Retain source values as the dropper continuation
                                # identity; the verified board branch is safe as
                                # an ordinary hint when H/T is changed below.
                                derived["values"] = copy.deepcopy(source["values"])
                            pool = self._make_pool(derived)
                    if base["reason"] is None:
                        rows = pool.prepare_contexts([dict(id=name, board_height_m=context["board_height_m"],
                            tilt_deg=context["tilt_deg"])])
                        if len(rows) != 1 or rows[0].get("id") != name:
                            raise RuntimeError("Child geometry changed context identity")
                        extra = {k: v for k, v in base.items() if k == "rotated_board_derivation"}
                        base = dict(rows[0], bucket_xy_m=context["bucket_xy_m"], **extra)
                        exact = [context["board_height_m"], context["tilt_deg"]] == source["values"][2:4]
                        if (base.get("board_admitted") and exact
                                and same_bearing(self.board_xy, source["bucket_xy_m"], context["bucket_xy_m"])
                                and not np.array_equal(base.get("q14"), source["q14"])):
                            raise RuntimeError("Same-bearing source board branch was substituted")
                        self._pools[name] = pool
            self._declared[name] = copy.deepcopy(context)
            self._contexts[name] = self._decorate(base, context)
            if progress:
                progress(copy.deepcopy(self._contexts[name]))
        return copy.deepcopy([self._contexts[c["id"]] for c in declared])

    def audit(self, context_ids, gaps, progress=None):
        names, gaps = list(context_ids), np.asarray(gaps, float)
        if gaps.shape != (len(names),) or not np.isfinite(gaps).all():
            raise ValueError("One finite gap per context ID required")
        if any(name not in self._contexts for name in names):
            raise ValueError("Prepare every context before auditing gaps")
        for name, gap in sorted(set(zip(names, map(float, gaps))), key=lambda x: (x[0], x[1].hex())):
            key = (name, gap.hex())
            if key in self._cache:
                continue
            context, prepared = self._declared[name], self._contexts[name]
            source = self._anchors[context["source_anchor_id"]]
            values = source["values"][:2] + [context["board_height_m"], context["tilt_deg"], gap]
            if name not in self._pools:
                row = dict(context_id=name, eligible=False, reason=prepared["reason"], values=values,
                    bucket_xy_m=context["bucket_xy_m"], q10=[0.] * 6, q14=[0.] * 6,
                    board_branch_sha256=None, minimum_clearance_lower_bound_m=None,
                    geometry_version=VERSION, scope=SCOPE, native_release=False, path_admitted=None)
            else:
                child = self._pools[name].audit([name], [gap])
                if (len(child) != 1 or child[0].get("context_id") != name
                        or not np.array_equal(child[0].get("values"), values)
                        or not np.array_equal(child[0].get("bucket_xy_m"), context["bucket_xy_m"])):
                    raise RuntimeError("Child geometry changed requested shot")
                row = child[0]
                if row.get("eligible") and not np.array_equal(row.get("q14"), prepared["q14"]):
                    raise RuntimeError("A gap changed the frozen board branch")
                if (row.get("eligible") and np.array_equal(values, source["values"])
                        and not np.array_equal(row.get("q10"), source["q10"])):
                    raise RuntimeError("Exact source dropper branch was substituted")
            row["same_source_bearing"] = prepared["same_source_bearing"]
            self._cache[key] = self._decorate(row, context)
        rows = [copy.deepcopy(self._cache[(name, float(gap).hex())]) for name, gap in zip(names, gaps)]
        if progress:
            for row in rows:
                progress(copy.deepcopy(row))
        return rows


def audit_distance_variations(scene, assets, nominal_row, requests, bounds=None, *, geometry_factory=None):
    """Joint H/T/G repeats at the actual nominal bucket with the existing .10rad gate."""
    footprint = table_footprint(scene, nominal_row["bucket_xy_m"])
    if not footprint["passed"]:
        raise ValueError("An admitted on-table nominal bucket is required")
    output = _audit_local_variations(scene, assets, nominal_row, requests, bounds,
        geometry_factory=geometry_factory)
    for row in output:
        row.update(distance_wrapper_version=LOCAL_VERSION, bucket_footprint=copy.deepcopy(footprint))
    return output
