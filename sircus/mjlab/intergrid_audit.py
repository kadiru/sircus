"""Reconstruct a completed inter-grid comparison from its frozen CPU-readable evidence."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

import numpy as np

from . import replay650_verify as replay
from .benchmark_verify import Checks, contained, digest, fingerprint, read, sha
from .calibrated_admission import verify_connection, verify_resolution
from .cell1050_rebound_audit import verify_heading
from .cell1050_rebound_study import control_requests, parent_inputs
from .intergrid_dataset import _grid, load_dataset
from .intergrid_layouts import MIN_SEPARATION_M, _placement_geometry
from .intergrid_ranker import features, load_checkpoint, predict_scores
from .intergrid_study import METHODS, VERSION, proposals_at, stable_captures
from .parallel_capture_grid_audit import observed_rows, rescore_packet, verify_controls, verify_release
from .policy64_verify import audit_native_options
from .sweep_worker import atomic
from .table_assisted_score import ROUTE_VERSION, VERSION as SCORING_VERSION


AUDIT_VERSION = "sircus-intergrid-independent-audit-v1"
PHASE_ORDER = ("controls_pre", "test", "controls_post")


def verify_training(study, spec, report, dataset_inputs, checks):
    rows = list(observed_rows(study / "dataset.jsonl"))
    provenance = read(study / "dataset-provenance.json")
    checks.equal(provenance["new_search_releases"], 13000, "Training evidence includes all 13,000 new releases")
    checks.equal(len(rows), provenance["total_unique_commands"], "All distinct training commands present")
    checks.equal(len({row["id"] for row in rows}), len(rows), "Unique training record IDs")
    if dataset_inputs:
        history, grid_study, grid_audit = dataset_inputs
        checks.equal(sha(grid_audit), spec["grid_audit_sha256"], "Same independent grid audit")
        original_rows, original_provenance = load_dataset(history, grid_study, grid_audit)
        checks.equal(rows, original_rows, "Exact reconstruction from original and audited new ledgers")
        checks.equal(provenance, original_provenance, "Exact reconstructed dataset provenance")

    grouped = defaultdict(list)
    excluded = 0
    canonical = _grid()
    for row in sorted(rows, key=lambda item: item["id"]):
        xy = tuple(round(float(value), 10) for value in row["bucket_xy_m"])
        checks.equal((row["context_id"], row["bucket_xy_m"]), canonical[xy], row["id"] + "/canonical bucket")
        checks.equal((row["native_release"], row["invalid"], row["invalid_physics"], row["route_version"],
                      row["scoring_version"]), (True, None, False, ROUTE_VERSION, SCORING_VERSION),
                     row["id"] + "/valid native64 v3 training record")
        checks.require(type(row["route_success"]) is bool, row["id"] + "/binary training outcome")
        checks.equal(row["route_success"], row["standard_capture"], row["id"] + "/standard capture target")
        if row["label_conflict"]:
            excluded += 1
            continue
        grouped[xy].append((row["id"], features(row).tolist(), int(row["route_success"])))

    checks.equal(set(grouped), set(canonical), "All 33 canonical buckets represented in model fit")
    checkpoint = load_checkpoint(study / "ranker.json")
    weighted, fitted, groups = [], [], []
    for xy, examples in sorted(grouped.items()):
        groups.append(dict(bucket_xy_m=list(xy), count=len(examples),
                           captures=sum(example[2] for example in examples), loss_weight=1 / 33))
        fitted.extend(examples)
        weighted.extend([1 / (33 * len(examples))] * len(examples))
    checks.equal(checkpoint["bucket_groups"], groups, "Each bucket has exactly 1/33 total training loss weight")
    checks.equal(checkpoint["dataset_sha256"], digest(fitted), "Frozen checkpoint binds exact fitted features and labels")
    checks.equal((checkpoint["training_rows"], checkpoint["excluded_conflicts"]), (len(fitted), excluded),
                 "Training sample and conflict counts")
    checks.equal((checkpoint["seed"], checkpoint["steps"], checkpoint["device"]),
                 (spec["seed"], spec["training_steps"], "cuda:0"), "Frozen training settings")
    checks.equal(report["training"], {key: checkpoint[key] for key in (
        "training_rows", "excluded_conflicts", "steps", "device", "curve",
    )}, "Reported frozen training metrics")
    weights = np.asarray(weighted)
    x = np.asarray([row[1] for row in fitted])
    center = weights @ x
    scale = np.maximum(np.sqrt(weights @ (x - center) ** 2), 1e-6)
    checks.numeric(checkpoint["center"], center, "Reconstructed equal-bucket feature means", 1e-12)
    checks.numeric(checkpoint["scale"], scale, "Reconstructed equal-bucket feature scales", 1e-12)
    w1, b1, w2, b2 = map(np.asarray, checkpoint["parameters"])
    logit = np.tanh(((x - center) / scale) @ w1 + b1) @ w2 + b2
    labels = np.asarray([row[2] for row in fitted])
    loss = float((np.logaddexp(0, logit) - labels * logit) @ weights)
    curve = checkpoint["curve"]
    checks.equal([entry["step"] for entry in curve], sorted({0, checkpoint["steps"], *range(20, checkpoint["steps"], 20)}),
                 "Exact recorded training loss cadence")
    checks.numeric(curve[-1]["weighted_bce"], loss, "Frozen model independently reproduces final weighted loss", 1e-10)
    return rows, checkpoint


def verify_layouts(study, scene, assets, checks):
    artifact = read(study / "test-layouts.json")
    layouts, provenance = artifact["rows"], artifact["provenance"]
    canonical = {context: xy for context, xy in _grid().values()}
    checks.equal(provenance["usable_grid"], canonical, "Held-out generator used the exact 33 training buckets")
    checks.equal(digest(layouts), provenance["layouts_sha256"], "Exact frozen test placements")
    checks.equal(provenance["minimum_separation_m"], MIN_SEPARATION_M, "Frozen five millimeter novelty distance")
    excluded = [tuple(xy) for xy in provenance["excluded_coordinates_m"]]
    checks.equal(excluded, sorted(set(excluded)), "Unique ordered historical exclusions")
    checks.equal(digest(excluded), provenance["excluded_coordinates_sha256"], "Exact historical coordinate exclusions")
    checks.require(set(map(tuple, canonical.values())).issubset(excluded), "Exclusions contain every training position")
    checks.equal(provenance["geometry"], _placement_geometry(scene, assets), "CPU-reconstructed placement geometry")
    checks.equal(len(layouts), 50, "Fifty unique held-out test placements")
    geometry = provenance["geometry"]
    low, high = np.asarray(geometry["tabletop_xy_bounds_m"])
    xs, ys = [np.unique(np.asarray(list(canonical.values()))[:, i]) for i in (0, 1)]
    checks.equal(provenance["grid_bounds_m"], [[xs[0], ys[0]], [xs[-1], ys[-1]]], "Frozen interpolation bounds")
    occupied = set(excluded)
    strata, rectangles = Counter(), Counter()
    for index, layout in enumerate(layouts):
        identity = f"intergrid-test-{index:02d}"
        checks.equal((layout["id"], layout["context_id"], layout["split"]), (identity, identity, "test"),
                     identity + "/frozen test identity")
        xy = tuple(layout["bucket_xy_m"])
        checks.require(len(xy) == 2 and all(math.isfinite(value) for value in xy), identity + "/finite XY")
        checks.require(all(math.dist(xy, other) >= MIN_SEPARATION_M for other in occupied),
                       identity + "/disjoint from training, archive, and other test positions")
        occupied.add(xy)
        radius = geometry["bucket_radius_m"]
        checks.require(np.minimum(np.asarray(xy) - low, high - np.asarray(xy)).min() >= radius,
                       identity + "/bucket footprint fits on tabletop")
        nearest = min(canonical, key=lambda key: math.dist(canonical[key], xy))
        checks.equal(layout["nearest_grid_id"], nearest, identity + "/independently closest training position")
        checks.numeric(layout["nearest_grid_distance_m"], math.dist(canonical[nearest], xy), identity + "/nearest-grid distance")
        clearance = min(math.dist(bound["center_xy_m"], xy) - (bound["radius_m"] + radius)
                        for bound in geometry["fixed_arm_bounds"].values())
        checks.require(clearance >= MIN_SEPARATION_M, identity + "/outside fixed base geometry")
        checks.numeric(layout["fixed_base_clearance_lower_bound_m"], clearance, identity + "/fixed base clearance")
        ix, iy = np.searchsorted(xs, xy[0]) - 1, np.searchsorted(ys, xy[1]) - 1
        checks.require(0 <= ix < len(xs) - 1 and 0 <= iy < len(ys) - 1, identity + "/inside interpolation grid")
        checks.equal(layout["grid_cell"], [ix, iy], identity + "/exact grid rectangle")
        stratum = layout["stratum"]
        strata[stratum] += 1
        if stratum != "near_fixed_base":
            rectangles[ix, iy] += 1
    checks.equal(dict(strata), dict(rectangle_center=24, rectangle_interior=8, grid_perimeter=16, near_fixed_base=2),
                 "Predeclared geometry-only test strata")
    checks.equal(dict(rectangles), {(ix, iy): 2 for ix in range(6) for iy in range(4)},
                 "Two test placements in each rectangle plus two near fixed bases")
    return layouts, strata


def verify_preparations(study, layouts, training, checkpoint, home, checks):
    library = stable_captures(training)
    menu = read(study / "candidate-menu.json")
    checks.equal(menu, [proposals_at(layout, library) for layout in layouts], "Shared frozen candidate menu derived from training captures")
    prior = read(study / "training-prior.json")
    checks.equal(prior, dict(records=[dict(
        {key: recipes[0][key] for key in ("id", "bucket_xy_m", "values", "q10", "q14")},
        source_record_sha256=recipes[0]["source_row_sha256"],
    ) for recipes in library.values()]), "Geometry preparation anchors come only from the 33 training positions")
    folder = study / "preparations"
    checks.equal({p.stem for p in folder.glob("*.json")}, {row["id"] for row in layouts}, "One preparation ledger per test bucket")
    selections, resolutions = [], {}
    directory = study / "native-heading"
    for layout, proposals in zip(layouts, menu, strict=True):
        identity = layout["id"]
        prepared = read(folder / (identity + ".json"))
        checks.equal(prepared["proposals"], proposals, identity + "/one shared menu unchanged")
        nearest_xy = min(library, key=lambda xy: (math.dist(xy, layout["bucket_xy_m"]), xy))
        checks.equal((proposals[0]["training_bucket_xy_m"], proposals[0]["training_recipe_id"]),
                     (list(nearest_xy), library[nearest_xy][0]["id"]), identity + "/strict nearest known shot")
        admitted = prepared["admitted"]
        checks.equal(len(prepared["records"]), len(proposals), identity + "/one preparation record per candidate")
        admitted_by_id = {item["id"]: item for item in admitted}
        checks.equal(len(admitted_by_id), len(admitted), identity + "/unique admitted candidates")
        recorded_ids = []
        for proposal, record in zip(proposals, prepared["records"], strict=True):
            label = proposal["id"]
            checks.equal((record["proposal_id"], record["proposal_sha256"]), (label, digest(proposal)),
                         label + "/exact frozen candidate")
            ids = record["admitted_request_ids"]
            checks.equal(record["admitted"], len(ids), label + "/admission count")
            checks.require(len(ids) <= 1, label + "/only one admitted branch per proposal")
            for identifier in ids:
                request = admitted_by_id[identifier]
                checks.equal(identifier, label + "-b0", label + "/admitted branch ID")
                inherited = {key: request[key] for key in proposal if key != "id"}
                checks.equal(inherited, {key: value for key, value in proposal.items() if key != "id"},
                             label + "/admission retains exact shared shot parameters")
                checks.equal((request["proposal_id"], request["native_prepared"], request["command_values"]),
                             (label, True, proposal["values"]), label + "/exact command admitted")
                name = request["context_resolution_file"]
                if name not in resolutions:
                    path = contained(directory, name)
                    anchor = dict(read(path), file=name, sha256=sha(path))
                    verify_resolution(anchor, prior["records"], home)
                    resolutions[name] = anchor
                anchor = resolutions[name]
                checks.equal((anchor["bucket_xy_m"], anchor["sha256"]),
                             (proposal["bucket_xy_m"], request["context_resolution_sha256"]), label + "/exact prepared bucket and anchor")
                path = contained(directory, request["prepared_connection_file"])
                proof = dict(read(path), file=request["prepared_connection_file"], sha256=sha(path))
                verify_connection(proof, anchor, home, request["values"])
                checks.equal((proof["passed"], proof["sha256"], proof["q10"], proof["q14"], proof["heading_key"],
                              proof["board_heading_offset_deg"], proof["board_twist_deg"]),
                             (True, request["prepared_connection_sha256"], request["selected_q10"], request["selected_q14"],
                              request["heading_key"], request["board_heading_offset_deg"], request["board_twist_deg"]),
                             label + "/verified exact endpoint and arm paths")
                recorded_ids.append(identifier)
        checks.equal(recorded_ids, list(admitted_by_id), identity + "/complete exact admitted menu")
        scores = predict_scores(checkpoint, admitted).tolist()
        checks.numeric(prepared["scores"], scores, identity + "/CPU reproduction of frozen scores", 1e-12)
        nearest = next((index for index, row in enumerate(admitted) if row["proposal_id"] == proposals[0]["id"]), None)
        learned = max(range(len(scores)), key=scores.__getitem__, default=None)
        for method, selected in zip(METHODS, (nearest, learned), strict=True):
            slot_id = f"{identity}-{method}"
            selected_request = None if selected is None else dict(admitted[selected], id=slot_id, method=method,
                                                                 stratum=layout["stratum"])
            selections.append(dict(id=slot_id, method=method, context_id=identity, bucket_xy_m=layout["bucket_xy_m"],
                                   stratum=layout["stratum"], request=selected_request,
                                   score=None if selected is None else prepared["scores"][selected],
                                   first_known_admitted=nearest is not None))
    choices = read(study / "choices.json")
    checks.equal(choices["checkpoint_sha256"], sha(study / "ranker.json"), "Choices bind frozen checkpoint bytes")
    checks.equal(choices["slots"], selections, "One strict nearest or highest frozen model choice per method and position")
    return selections, prior


def verify_measurements(study, parent, assets, spec, report, manifest, selections, prior, home, checks):
    calibration = replay.check_calibration(study, spec["identity"]["calibration"], checks)
    directory = study / "native-heading"
    for namespace in ("native-heading", "native-controls"):
        if namespace == "native-heading" and not any(slot["request"] for slot in selections):
            continue
        audit_native_options(study / namespace, spec["scene"], calibration, assets, checks)
    ledger = observed_rows(study / "rows.jsonl")
    timings = iter(observed_rows(study / "timing.jsonl"))
    controls, resolutions, observations = [], {}, {}
    releases = [slot["request"] for slot in selections if slot["request"] is not None]
    sources = read(parent / "control-replays.json")
    phases = []
    archives = set()
    released = 0
    for packet_number, path in enumerate(sorted((study / "requests").glob("*.json"))):
        label = f"requests/batch-{packet_number:05d}.json"
        checks.equal(path.relative_to(study).as_posix(), label, "Contiguous chronological native request packets")
        packet = read(path)
        phase, requests = packet["phase"], packet["rows"]
        phases.append(PHASE_ORDER.index(phase))
        checks.require(0 < len(requests) <= 32, label + "/bounded native packet")
        timing = next(timings, None)
        checks.require(timing is not None, label + "/saved timing")
        checks.equal((timing["packet"], timing["phase"]), (label, phase), label + "/matching timing")
        if phase == "test":
            checks.equal(requests, releases[released:released + len(requests)], label + "/exact preselected test requests")
            released += len(requests)
        else:
            checks.equal(requests, control_requests(sources, phase), label + "/exact frozen controls")
        rows = []
        for request in requests:
            row = next(ledger, None)
            checks.require(row is not None, label + "/native observation for each request")
            verify_release(row, request, label, manifest[label], phase, checks)
            if phase == "test":
                checks.equal((row["adapter_directory"], row["proposal_id"], row["method"]),
                             ("native-heading", request["proposal_id"], request["method"]), row["id"] + "/frozen test choice provenance")
                verify_heading(row, request, directory, spec["scene"], home, prior, resolutions, checks)
            else:
                checks.equal(row["adapter_directory"], "native-controls", row["id"] + "/control archive")
            rows.append(row)
        if phase == "test":
            classes = rescore_packet(directory, rows, spec["scene"], calibration, manifest, archives, checks)
            for row, route in zip(rows, classes, strict=True):
                checks.require(row["id"] not in observations, row["id"] + "/one test release per method and position")
                observations[row["id"]] = "invalid_physics" if row["invalid_physics"] else route
        else:
            controls.extend(rows)
    checks.equal(phases, [0] + [1] * (len(phases) - 2) + [2], "Exact pre-controls, one-shot test, post-controls chronology")
    checks.require(next(ledger, None) is None and next(timings, None) is None, "Every saved measurement and timing belongs to a packet")
    checks.equal(set(observations), {row["id"] for row in releases}, "All and only frozen admitted choices released")
    checks.equal(len(observations), report["measured_test_releases"], "Reported measured test releases")
    checks.equal(archives, {path.relative_to(directory).as_posix() for path in directory.rglob("*.npz")},
                 "Complete native test archive inventory")
    verify_controls(study, parent, spec, report, controls, calibration, assets, checks)
    return observations, len(controls)


def summarize(selections, observations):
    methods = {method: Counter() for method in METHODS}
    strata = defaultdict(lambda: {method: Counter() for method in METHODS})
    by_context = defaultdict(dict)
    for slot in selections:
        method = slot["method"]
        category = observations.get(slot["id"], "setup_rejected")
        methods[method][category] += 1
        strata[slot["stratum"]][method][category] += 1
        by_context[slot["context_id"]][method] = (slot, category)
    pairing = {frozenset(): "neither", frozenset(METHODS): "both",
               **{frozenset([method]): method + "_only" for method in METHODS}}
    paired, both_released = Counter(), Counter()
    changed = 0
    for pair in by_context.values():
        result = pairing[frozenset(method for method, (_, category) in pair.items() if category == "standard_capture")]
        paired[result] += 1
        if all(slot["id"] in observations for slot, _ in pair.values()):
            both_released[result] += 1
            changed += len({slot["request"]["proposal_id"] for slot, _ in pair.values()}) == 2
    return dict(methods={method: dict(counts) for method, counts in methods.items()},
                strata={key: {method: dict(counts) for method, counts in counters.items()}
                        for key, counters in strata.items()}, paired=dict(paired),
                both_released=dict(both_released), different_settings_when_both_released=changed)


def audit(study, parent, parent_audit, assets, *, dataset_inputs=None):
    checks = Checks()
    spec, report = read(study / "contract.json"), read(study / "report.json")
    parent_spec, parent_binding, _ = parent_inputs(parent, parent_audit)
    checks.equal((spec["version"], report["status"], report["phase"]), (VERSION, "completed", "completed"),
                 "Completed versioned inter-grid comparison")
    fingerprint(spec, checks, "Frozen comparison contract")
    checks.equal((spec["identity"], spec["scene"], spec["parent"]),
                 (parent_spec["identity"], parent_spec["scene"], parent_binding), "Exact audited native64 parent")
    checks.equal((spec["train_positions"], spec["test_positions"], spec["methods"],
                  spec["attempts_per_method_per_position"], spec["physics_substeps"]),
                 (33, 50, list(METHODS), 1, 64), "Frozen 33 training buckets and 50 paired one-attempt test placements")
    manifest = replay.check_manifest(study, checks)
    checks.equal(report["evidence_manifest_sha256"], sha(study / "evidence-manifest.json"), "Report binds complete evidence bytes")
    required = ("dataset.jsonl", "dataset-provenance.json", "test-layouts.json", "candidate-menu.json")
    checks.equal(spec["frozen_inputs"], {name: manifest[name] for name in required}, "All pre-training inputs match frozen contract")
    checks.equal(report["choices_sha256"], manifest["choices.json"], "Report binds frozen choices")
    training, checkpoint = verify_training(study, spec, report, dataset_inputs, checks)
    layouts, strata = verify_layouts(study, spec["scene"], assets, checks)
    home = read(parent / "setup-home.json")
    selections, prior = verify_preparations(study, layouts, training, checkpoint, home, checks)
    observations, controls = verify_measurements(study, parent, assets, spec, report, manifest,
                                                 selections, prior, home, checks)
    comparison = summarize(selections, observations)
    checks.equal(comparison, report["comparison"], "Independent paired comparison from rescored native outcomes")
    nearest_admitted = sum(slot["request"] is not None for slot in selections if slot["method"] == "nearest")
    checks.equal(report["nearest_first_choice_admitted"], nearest_admitted, "Strict nearest admission count")
    checks.equal(report["prepared_positions"], len(layouts), "All frozen test positions prepared")
    return dict(version=AUDIT_VERSION, status="passed", checks_count=checks.count,
                train_positions=33, test_positions=50, held_out_strata=dict(strata),
                train_source_reconstructed=dataset_inputs is not None,
                training_rows=checkpoint["training_rows"], equal_bucket_loss_weight=1 / 33,
                comparison=comparison, measured_test_releases=len(observations), control_releases=controls,
                report_sha256=sha(study / "report.json"), evidence_manifest_sha256=report["evidence_manifest_sha256"],
                checkpoint_sha256=manifest["ranker.json"], choices_sha256=manifest["choices.json"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "parent-study", "parent-audit", "assets", "audit"):
        parser.add_argument("--" + name, required=True, type=Path)
    for name in ("history-root", "grid-study", "grid-audit"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args()
    if args.audit.exists() or args.audit.resolve().is_relative_to(args.study.resolve()):
        parser.error("Write a fresh audit outside the frozen evidence directory")
    dataset_inputs = (args.history_root, args.grid_study, args.grid_audit)
    if any(dataset_inputs) and not all(dataset_inputs):
        parser.error("Provide all three source paths to reconstruct the training dataset")
    try:
        result = audit(args.study, args.parent_study, args.parent_audit, args.assets,
                       dataset_inputs=dataset_inputs if all(dataset_inputs) else None)
    except Exception as error:
        result = dict(version=AUDIT_VERSION, status="failed", error=f"{type(error).__name__}: {error}")
    atomic(args.audit, result)
    print(json.dumps(result), flush=True)
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
