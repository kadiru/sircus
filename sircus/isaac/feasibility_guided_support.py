"""Geometry-only local continuous proposal domains, never successful-shot menus.

A deterministic feasible anchor at each known bucket coordinate supplies a local
box and IK warm starts. Finite probe acceptance does not guarantee every action
in that box is reachable. Every native request must still pass fresh gates.
"""

import concurrent.futures
import copy
import hashlib
import json
import math
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.stats import qmc

from continuous_feasibility import (
    ACTION_NAMES, HIGH, LOW, approximate_path_check, canonical_hash, decode_action,
    qualify_action,
)
from prepare_assets import chain

SCHEMA = "sircus-feasibility-guided-support-v1"
BASE_HALF_WIDTH = np.array([math.radians(5), .025, .025, .060, math.radians(6)])
WIDTH_SCALES = (1., .5, .25, .125)
PROBE_COUNT = 16
PROBE_SEED = 20261113
MIN_ACCEPTANCE = .75
SCOPE = (
    "Geometry-assisted local continuous residual domains at inherited bucket coordinates. "
    "No capture labels, reward search, action projection or online resampling. Finite "
    "probe acceptance is not complete feasibility or mesh/path certification. Held-out "
    "bucket geometry assists decoding but held-out outcomes never choose support."
)


def require(value, message):
    if not value:
        raise ValueError(message)


def signed(value, label):
    require(isinstance(value, dict), "Invalid " + label)
    unsigned = dict(value)
    fingerprint = unsigned.pop("fingerprint", None)
    require(fingerprint == canonical_hash(unsigned), label + " fingerprint mismatch")
    return fingerprint


def _values(action):
    values = np.asarray([action[k] for k in ACTION_NAMES], dtype=float)
    require(values.shape == (5,) and np.isfinite(values).all()
            and (values >= LOW).all() and (values <= HIGH).all(), "Invalid anchor action")
    return values


def prior_values(cell):
    distance = float(np.linalg.norm(np.array([.67, -.10])-cell["bucket_xy_m"]))
    return np.array([0., np.clip(distance, .29, 1.19), .35, .24, math.radians(29)])


def anchor_score(cell, action, fits, home):
    """Rank only geometry and joint travel, without taking outcome fields."""
    values, prior = _values(action), prior_values(cell)
    normalized_distance = float(np.square((values-prior)/(HIGH-LOW)).sum())
    travel = float(np.mean([np.square(np.asarray(fits[n]["q"])-home[n]).sum()
                            for n in ("Arm10", "Arm14")]))
    return normalized_distance + .01*travel


def _links(assets):
    assets = Path(assets)
    return {name: chain(ET.parse(assets/file).getroot())
            for name, file in (("Arm10", "dropper.urdf"), ("Arm14", "board_arm.urdf"))}


def _home(assets):
    poses = json.loads((Path(assets)/"poses.json").read_text())
    return {"Arm10": np.asarray(poses["dropper_joint_rad"], dtype=float),
            "Arm14": np.asarray(poses["board_joint_rad"], dtype=float)}


def _cell_support(job):
    from robust_ik import solve

    scene, cell, index, options, asset_path = job
    links, home = _links(asset_path), _home(asset_path)
    ranked, anchor_checks = [], []
    for option in options:
        action = copy.deepcopy(option["action"])
        values = _values(action)
        expected = decode_action(cell, values, action["id"])
        require(all(abs(action[k]-expected[k]) < 1e-9 for k in ("board_x_m", "board_y_m", "tilt_deg")),
                "Anchor geometry disagrees with action")
        path = approximate_path_check(scene, links, home,
                                     {n: option["ik"][n]["q"] for n in home}, cell=cell, samples=16)
        score = anchor_score(cell, action, option["ik"], home)
        anchor_checks.append(dict(proposal=option["proposal"], action_id=action["id"], score=score, path=path))
        if path["accepted"]:
            ranked.append((score, int(option["proposal"]), option))
    row = dict(cell_id=cell["id"], cell_index=index, bucket_xy_m=cell["bucket_xy_m"],
               map_candidates=len(options), anchor_checks=anchor_checks,
               status="unresolved", reason="no_map_anchor_with_home_path_proxy",
               low=LOW.tolist(), high=HIGH.tolist(), anchor_action=None, anchor_ik=None,
               calibration=[])
    if not ranked:
        return row
    score, _, anchor = min(ranked, key=lambda value: value[:2])
    center = _values(anchor["action"])
    seeds = {n: anchor["ik"][n]["q"] for n in home}
    unit = qmc.Sobol(d=5, scramble=True, seed=PROBE_SEED+index).random_base2(4)
    selected = None
    for scale in WIDTH_SCALES:
        low = np.maximum(LOW, center-BASE_HALF_WIDTH*scale)
        high = np.minimum(HIGH, center+BASE_HALF_WIDTH*scale)
        records = []
        for j, values in enumerate(low+unit*(high-low)):
            action = decode_action(cell, values, f"support-{index:02d}-{scale}-{j:02d}")
            fitted = qualify_action(scene, cell, action, links, seeds, solver=solve)
            path = None
            if fitted["accepted"]:
                path = approximate_path_check(scene, links, home,
                    {n: fitted["ik"][n]["q"] for n in home}, cell=cell, samples=16)
            records.append(dict(probe=j, action=action, qualification=fitted, path=path,
                                accepted=bool(fitted["accepted"] and path and path["accepted"])))
        count = sum(r["accepted"] for r in records)
        calibration = dict(scale=scale, low=low.tolist(), high=high.tolist(), requests=len(records),
                           accepted=count, acceptance=count/len(records), records=records)
        row["calibration"].append(calibration)
        selected = calibration
        if calibration["acceptance"] >= MIN_ACCEPTANCE:
            break
    row.update(status="supported", reason=None, anchor_proposal=anchor["proposal"],
               anchor_action=copy.deepcopy(anchor["action"]), anchor_ik=copy.deepcopy(anchor["ik"]),
               anchor_score=score, low=selected["low"], high=selected["high"],
               selected_width_scale=selected["scale"], probe_acceptance=selected["acceptance"],
               qualification_target_met=selected["acceptance"] >= MIN_ACCEPTANCE)
    # Even a small-box fallback below the target stays explicit. Native admission
    # remains mandatory, and no failed probe is relabeled or removed.
    if not row["qualification_target_met"]:
        row["reason"] = "finite_probe_acceptance_below_target"
    return row


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
    for name, expected in feasibility["asset_sha256"].items():
        require(Path(name).name == name and hashlib.sha256((assets/name).read_bytes()).hexdigest() == expected,
                "Geometry assets changed")
    by_cell = {c["id"]: [] for c in cells}
    for record in feasibility["records"]:
        require(record["cell_id"] in by_cell, "Unexpected map cell")
        if record["accepted"]:
            # Explicit allowlist drops any added outcome/reward fields. Selected
            # anchor values depend only on this geometry record subset.
            by_cell[record["cell_id"]].append({k: copy.deepcopy(record[k])
                                              for k in ("action", "ik", "proposal")})
    jobs = [(source_spec["scene"], c, i, by_cell[c["id"]], str(assets)) for i, c in enumerate(cells)]
    pool = concurrent.futures.ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    rows = []
    try:
        for row in (pool.map(_cell_support, jobs) if pool else map(_cell_support, jobs)):
            rows.append(row)
            if progress:
                progress(dict(completed_cells=len(rows), supported=sum(r["status"] == "supported" for r in rows),
                              probe_requests=sum(sum(x["requests"] for x in r["calibration"]) for r in rows),
                              elapsed_s=time.monotonic()-started))
    finally:
        if pool:
            pool.shutdown(wait=True, cancel_futures=True)
    result = dict(schema=SCHEMA, source_fingerprint=source_spec["fingerprint"],
                  feasibility_fingerprint=feasibility["fingerprint"], policy_geometry_only=True,
                  asset_sha256=copy.deepcopy(feasibility["asset_sha256"]),
                  canonical_home_joint_rad={name: values.tolist() for name, values in _home(assets).items()},
                  action_names=list(ACTION_NAMES), global_low=LOW.tolist(), global_high=HIGH.tolist(),
                  base_half_width=BASE_HALF_WIDTH.tolist(), width_scales=list(WIDTH_SCALES),
                  probe_count=PROBE_COUNT, probe_seed=PROBE_SEED, minimum_probe_acceptance=MIN_ACCEPTANCE,
                  anchor_rank="Normalized squared distance to declared prior + .01 mean squared canonical-home joint travel; proposal index breaks ties",
                  anchor_path_samples=16, probe_solver="unchanged full robust_ik.solve v2 seeded by chosen geometry anchor",
                  seed_source="poses.json canonical episode joint defaults; fresh native action solves still required",
                  cells=rows, elapsed_s=time.monotonic()-started,
                  probe_requests=sum(sum(x["requests"] for x in r["calibration"]) for r in rows),
                  probe_accepted=sum(sum(x["accepted"] for x in r["calibration"]) for r in rows),
                  supported_cells=sum(r["status"] == "supported" for r in rows), scope=SCOPE)
    result["fingerprint"] = canonical_hash(result)
    validate_support(result, cells)
    return result


make_support = build_support


def validate_support(support, cells=None):
    signed(support, "Guided support")
    require(support.get("schema") == SCHEMA and support.get("policy_geometry_only") is True,
            "Unsupported guided proposal contract")
    require(support.get("action_names") == list(ACTION_NAMES)
            and support.get("global_low") == LOW.tolist() and support.get("global_high") == HIGH.tolist(),
            "Guided global action domain changed")
    rows = support["cells"]
    require(len(rows) == 35 and len({r["cell_id"] for r in rows}) == 35, "Guided support cells changed")
    if cells is not None:
        require([(r["cell_id"], r["bucket_xy_m"]) for r in rows] == [(c["id"], c["bucket_xy_m"]) for c in cells],
                "Guided support bucket coordinates/order mismatch")
    for row in rows:
        low, high = np.asarray(row["low"]), np.asarray(row["high"])
        require(low.shape == high.shape == (5,) and np.isfinite(low).all() and np.isfinite(high).all()
                and (low >= LOW).all() and (high <= HIGH).all() and (high > low).all(), "Invalid local guided box")
        require(row["status"] in ("supported", "unresolved"), "Unknown guided support status")
        if row["status"] == "supported":
            require(row.get("anchor_action") is not None and row.get("anchor_ik") is not None,
                    "Supported box missing anchor")
            require((_values(row["anchor_action"]) >= low).all() and (_values(row["anchor_action"]) <= high).all(),
                    "Anchor outside local box")
        else:
            require(row.get("anchor_action") is None and row.get("anchor_ik") is None,
                    "Unresolved box cannot carry an anchor")
    return support


def _selected_rows(support, cells):
    validate_support(support)
    by_id = {r["cell_id"]: r for r in support["cells"]}
    selected = []
    for cell in cells:
        require(cell["id"] in by_id and cell["bucket_xy_m"] == by_id[cell["id"]]["bucket_xy_m"],
                "No guided geometry support for this bucket coordinate")
        selected.append(by_id[cell["id"]])
    return selected


def physical_actions(normalized, support, cells):
    """Per-context affine residual map only; no lookup of successful actions."""
    if hasattr(normalized, "detach"):
        normalized = normalized.detach().cpu().numpy()
    values = np.asarray(normalized, dtype=float)
    require(values.shape == (len(cells), 5) and np.isfinite(values).all() and (np.abs(values) <= 1).all(),
            "Expected finite normalized residual actions")
    rows = _selected_rows(support, cells)
    low = np.asarray([r["low"] for r in rows])
    high = np.asarray([r["high"] for r in rows])
    return low+(values+1)*(high-low)/2


def ik_seeds(support, cells):
    """Return anchor fits or None; never reuse them as unverified endpoints."""
    return [copy.deepcopy(r["anchor_ik"]) for r in _selected_rows(support, cells)]


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
    print(json.dumps({k: result[k] for k in ("fingerprint", "supported_cells", "probe_requests", "probe_accepted", "elapsed_s")}))


if __name__ == "__main__":
    main()
