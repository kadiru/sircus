"""Geometry-only union of continuous shot regions, without outcome selection.

Every map anchor passing the unchanged canonical-home path proxy is retained.
Regions have fixed, prespecified widths. Finite endpoint/path tests do not
certify a region interior: the native runner must solve and gate every request.
"""

import concurrent.futures
import copy
import hashlib
import json
import math
import re
import time
from pathlib import Path

import numpy as np

from continuous_feasibility import (
    ACTION_NAMES, HIGH, LOW, approximate_path_check, canonical_hash, decode_action,
)
from feasibility_guided_support import _home, _links, _values, anchor_score, require, signed

SCHEMA_V1 = "sircus-diverse-shot-support-v1"
SCHEMA = "sircus-diverse-shot-support-v2"
HALF_WIDTH = np.array([math.radians(7.5), .040, .040, .080, math.radians(8)])
GLOBAL_QUALIFICATION_SCOPE = "Full original action domain is unqualified. First geometry anchor supplies only an IK warm start; every requested action still needs fresh IK/path/settlement gates."
SCOPE = (
    "Geometry-only union of all canonical-home-path-qualified map anchors and fixed "
    "continuous residual boxes plus one full-domain exploration region per supported cell. "
    "The full domain is unqualified; its first local anchor is a warm start only. "
    "No capture labels, reward selection, projection, "
    "resampling or probe-based shrinkage. Geometry assists known held-out coordinates; "
    "not globally unseen-context inference or collision-certified region interiors."
)


def _coverage(regions):
    if not regions:
        return None
    return dict(low=np.min([r["low"] for r in regions], axis=0).tolist(),
                high=np.max([r["high"] for r in regions], axis=0).tolist(),
                anchor_low=np.min([_values(r["anchor_action"]) for r in regions], axis=0).tolist(),
                anchor_high=np.max([_values(r["anchor_action"]) for r in regions], axis=0).tolist(),
                scope="Coordinate-wise envelope of separate boxes, not a filled feasible box")


def _cell_support(job):
    scene, cell, index, options, assets = job
    links, home = _links(assets), _home(assets)
    checks, retained = [], []
    for option in options:
        original = option["action"]
        center = _values(original)
        action = decode_action(cell, center, original["id"])
        require(all(abs(original[k]-action[k]) < 1e-9
                    for k in ("board_x_m", "board_y_m", "tilt_deg")),
                "Map anchor geometry disagrees with action")
        path = approximate_path_check(scene, links, home,
            {name: option["ik"][name]["q"] for name in home}, cell=cell, samples=16)
        score = anchor_score(cell, action, option["ik"], home)
        check = dict(proposal=option["proposal"], action_id=action["id"], score=score, path=path)
        checks.append(check)
        if path["accepted"]:
            retained.append((score, int(option["proposal"]), option, path))
    regions = []
    for region_id, (score, proposal, option, path) in enumerate(sorted(retained, key=lambda v: v[:2])):
        center = _values(option["action"])
        regions.append(dict(id=region_id, kind="local_geometry", proposal=proposal, interior_qualified=False,
            anchor_action=decode_action(cell, center, option["action"]["id"]),
            anchor_ik={name: dict(q=copy.deepcopy(option["ik"][name]["q"])) for name in home},
            anchor_score=score, anchor_path=path,
            low=np.maximum(LOW, center-HALF_WIDTH).tolist(),
            high=np.minimum(HIGH, center+HALF_WIDTH).tolist(),
            qualification_scope="Anchor only: map endpoint IK/FK and canonical-home path proxy; interior unqualified"))
    local_count = len(regions)
    local_coverage = _coverage(regions)
    if regions:
        first = regions[0]
        regions.append(dict(id=local_count, kind="global_exploration", proposal=-1, interior_qualified=False,
            anchor_action=copy.deepcopy(first["anchor_action"]), anchor_ik=copy.deepcopy(first["anchor_ik"]),
            low=LOW.tolist(), high=HIGH.tolist(),
            qualification_scope=GLOBAL_QUALIFICATION_SCOPE))
    return dict(cell_id=cell["id"], cell_index=index, bucket_xy_m=copy.deepcopy(cell["bucket_xy_m"]),
        status="supported" if regions else "unresolved", map_candidates=len(options),
        region_count=len(regions), local_region_count=local_count, global_region_count=int(bool(local_count)),
        regions=regions, anchor_checks=checks, local_coverage=local_coverage,
        reason=None if regions else "no_map_anchor_with_home_path_proxy", coverage=_coverage(regions))


def build_support(feasibility, source_spec, assets, workers=8, progress=None):
    started = time.monotonic()
    signed(feasibility, "Geometry map")
    signed(source_spec, "Source geometry")
    require(feasibility.get("source_fingerprint") == source_spec["fingerprint"], "Map/source geometry mismatch")
    require(feasibility.get("ik_protocol") == "full_robust_v2", "Require full robust map anchors")
    cells = source_spec["cells"]
    require(len(cells) == 35 and len({c["id"] for c in cells}) == 35
            and len({tuple(c["bucket_xy_m"]) for c in cells}) == 35, "Expected 35 distinct cells")
    require(type(workers) is int and 1 <= workers <= 16, "Choose 1..16 CPU workers")
    assets = Path(assets)
    for filename, expected in feasibility["asset_sha256"].items():
        require(Path(filename).name == filename
                and hashlib.sha256((assets/filename).read_bytes()).hexdigest() == expected,
                "Geometry assets changed")
    by_cell = {c["id"]: [] for c in cells}
    for record in feasibility["records"]:
        require(record["cell_id"] in by_cell, "Unexpected map cell")
        if record["accepted"]:
            # Never copy capture/outcome fields, even when a caller adds them.
            by_cell[record["cell_id"]].append({k: copy.deepcopy(record[k])
                                              for k in ("action", "ik", "proposal")})
    jobs = [(source_spec["scene"], cell, i, by_cell[cell["id"]], str(assets))
            for i, cell in enumerate(cells)]
    pool = concurrent.futures.ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    rows = []
    try:
        for row in (pool.map(_cell_support, jobs) if pool else map(_cell_support, jobs)):
            rows.append(row)
            if progress:
                progress(dict(completed_cells=len(rows), regions=sum(r["region_count"] for r in rows),
                              supported_cells=sum(r["status"] == "supported" for r in rows),
                              elapsed_s=time.monotonic()-started))
    finally:
        if pool:
            pool.shutdown(wait=True, cancel_futures=True)
    regions = [region for row in rows for region in row["regions"]]
    result = dict(schema=SCHEMA, source_fingerprint=source_spec["fingerprint"],
        feasibility_fingerprint=feasibility["fingerprint"], policy_geometry_only=True,
        asset_sha256=copy.deepcopy(feasibility["asset_sha256"]),
        canonical_home_joint_rad={k: v.tolist() for k, v in _home(assets).items()},
        action_names=list(ACTION_NAMES), global_low=LOW.tolist(), global_high=HIGH.tolist(),
        fixed_half_width=HALF_WIDTH.tolist(), anchor_path_samples=16,
        selection="Every map candidate passing unchanged canonical-home path proxy plus one unqualified original-domain escape region per supported context; no outcome-dependent pruning",
        region_order="Geometry-prior score, then map proposal index; no reward/outcome input",
        seed_source="Selected geometry anchor is a warm start only; fresh native solves mandatory",
        cells=rows, region_count=len(regions), local_region_count=sum(r["local_region_count"] for r in rows),
        global_region_count=sum(r["global_region_count"] for r in rows),
        supported_cells=sum(r["status"] == "supported" for r in rows),
        max_regions_per_cell=max((r["region_count"] for r in rows), default=0),
        coverage=_coverage(regions), local_coverage=_coverage([r for r in regions if r["kind"] == "local_geometry"]),
        elapsed_s=time.monotonic()-started, scope=SCOPE)
    result["fingerprint"] = canonical_hash(result)
    return validate_support(result, cells)


def validate_support(support, cells=None):
    signed(support, "Diverse support")
    require(support.get("schema") in (SCHEMA, SCHEMA_V1) and support.get("policy_geometry_only") is True,
            "Unsupported diverse proposal contract")
    v2 = support["schema"] == SCHEMA
    for key in ("source_fingerprint", "feasibility_fingerprint"):
        require(isinstance(support.get(key), str) and re.fullmatch(r"[0-9a-f]{64}", support[key]),
                "Invalid diverse source identity")
    require(support.get("action_names") == list(ACTION_NAMES)
            and support.get("global_low") == LOW.tolist() and support.get("global_high") == HIGH.tolist()
            and support.get("fixed_half_width") == HALF_WIDTH.tolist(), "Diverse action domain changed")
    rows = support["cells"]
    require(len(rows) == 35 and len({r["cell_id"] for r in rows}) == 35
            and len({tuple(r["bucket_xy_m"]) for r in rows}) == 35, "Diverse support cells changed")
    if cells is not None:
        require([(r["cell_id"], r["bucket_xy_m"]) for r in rows]
                == [(c["id"], c["bucket_xy_m"]) for c in cells], "Diverse support bucket coordinates/order mismatch")
    for row in rows:
        regions = row["regions"]
        require(row["status"] in ("supported", "unresolved")
                and (row["status"] == "supported") == bool(regions)
                and row["region_count"] == len(regions), "Invalid diverse support status/count")
        require([r["id"] for r in regions] == list(range(len(regions))), "Noncanonical diverse region IDs")
        require(len({r["proposal"] for r in regions}) == len(regions), "Duplicate geometry region")
        locals_ = regions[:-1] if v2 and regions else regions
        if v2:
            require(row.get("local_region_count") == len(locals_)
                    and row.get("global_region_count") == int(bool(regions))
                    and row.get("local_coverage") == _coverage(locals_), "Diverse local/global counts changed")
            if regions:
                require(bool(locals_) and all(r.get("kind") == "local_geometry" and r["proposal"] >= 0 for r in locals_)
                        and regions[-1].get("kind") == "global_exploration" and regions[-1]["proposal"] == -1
                        and regions[-1]["anchor_action"] == locals_[0]["anchor_action"]
                        and regions[-1]["anchor_ik"] == locals_[0]["anchor_ik"]
                        and regions[-1].get("qualification_scope") == GLOBAL_QUALIFICATION_SCOPE,
                        "Global exploration escape region changed")
            require(all(r.get("interior_qualified") is False for r in regions), "Exploration interior cannot be labeled qualified")
        for region in regions:
            low, high = np.asarray(region["low"]), np.asarray(region["high"])
            center = _values(region["anchor_action"])
            require(low.shape == high.shape == (5,) and np.isfinite(low).all() and np.isfinite(high).all()
                    and (high > low).all() and (low >= LOW).all() and (high <= HIGH).all(), "Invalid diverse box")
            is_global = v2 and region.get("kind") == "global_exploration"
            require(np.array_equal(low, LOW if is_global else np.maximum(LOW, center-HALF_WIDTH))
                    and np.array_equal(high, HIGH if is_global else np.minimum(HIGH, center+HALF_WIDTH)), "Diverse box widths changed")
            require(isinstance(region.get("anchor_ik"), dict)
                    and set(region["anchor_ik"]) == {"Arm10", "Arm14"}, "Region missing IK warm starts")
            for name in ("Arm10", "Arm14"):
                require(set(region["anchor_ik"][name]) == {"q"}, "Unexpected anchor seed fields")
                q = np.asarray(region["anchor_ik"][name]["q"])
                require(q.shape == (6,) and np.isfinite(q).all(), "Invalid anchor joint warm start")
            expected = decode_action(dict(bucket_xy_m=row["bucket_xy_m"]), center, region["anchor_action"]["id"])
            require(set(expected) == set(region["anchor_action"]), "Unexpected anchor action fields")
            require(all(abs(expected[k]-region["anchor_action"][k]) < 1e-9
                        for k in ("board_x_m", "board_y_m", "tilt_deg")), "Region action/context mismatch")
        require(row["coverage"] == _coverage(regions), "Diverse cell coverage changed")
    regions = [r for row in rows for r in row["regions"]]
    require(support["region_count"] == len(regions)
            and support["supported_cells"] == sum(bool(r["regions"]) for r in rows)
            and support["max_regions_per_cell"] == max((len(r["regions"]) for r in rows), default=0)
            and support["coverage"] == _coverage(regions), "Diverse aggregate coverage changed")
    if v2:
        require(support.get("local_region_count") == sum(r["local_region_count"] for r in rows)
                and support.get("global_region_count") == support["supported_cells"]
                and support.get("local_coverage") == _coverage([r for r in regions if r["kind"] == "local_geometry"]),
                "Diverse aggregate local/global coverage changed")
    return support


def selected_rows(support, cells):
    validate_support(support)
    by_id = {r["cell_id"]: r for r in support["cells"]}
    selected = []
    for cell in cells:
        require(cell["id"] in by_id and cell["bucket_xy_m"] == by_id[cell["id"]]["bucket_xy_m"],
                "No diverse geometry support for this bucket coordinate")
        selected.append(by_id[cell["id"]])
    return selected


def _indices(values, count):
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    raw = np.asarray(values)
    require(raw.shape == (count,) and np.isfinite(raw).all()
            and (raw == np.floor(raw)).all() and (raw >= 0).all(), "Expected nonnegative integer region indices")
    return raw.astype(np.int64)


def selected_regions(support, cells, region_indices, allow_unresolved=False):
    rows = selected_rows(support, cells)
    indices = _indices(region_indices, len(rows))
    result = []
    for row, index in zip(rows, indices):
        if not row["regions"]:
            require(allow_unresolved and index == 0, "Unresolved support requires explicit dummy region zero")
            result.append(None)
        else:
            require(index < len(row["regions"]), "Unavailable diverse region index")
            result.append(row["regions"][int(index)])
    return result


def physical_actions(normalized, support, cells, region_indices, allow_unresolved=False):
    if hasattr(normalized, "detach"):
        normalized = normalized.detach().cpu().numpy()
    values = np.asarray(normalized, dtype=float)
    require(values.shape == (len(cells), 5) and np.isfinite(values).all()
            and (np.abs(values) <= 1).all(), "Expected finite normalized residual actions")
    regions = selected_regions(support, cells, region_indices, allow_unresolved)
    low = np.asarray([r["low"] if r else LOW for r in regions])
    high = np.asarray([r["high"] if r else HIGH for r in regions])
    return low+(values+1)*(high-low)/2


def ik_seeds(support, cells, region_indices, allow_unresolved=False):
    return [copy.deepcopy(r["anchor_ik"]) if r else None
            for r in selected_regions(support, cells, region_indices, allow_unresolved)]


def support_missing(support, cells):
    return [not bool(row["regions"]) for row in selected_rows(support, cells)]


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", type=Path, required=True)
    parser.add_argument("--source-spec", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new support directory")
    args.output.mkdir(parents=True)
    def progress(row):
        print(json.dumps(row), flush=True)
        (args.output/"progress.json").write_text(json.dumps(row, allow_nan=False))
    result = build_support(json.loads(args.map.read_text()), json.loads(args.source_spec.read_text()),
                           args.assets, args.workers, progress)
    (args.output/"support.json").write_text(json.dumps(result, allow_nan=False))
    print(json.dumps({k: result[k] for k in ("fingerprint", "supported_cells", "region_count", "elapsed_s")}))


if __name__ == "__main__":
    main()
