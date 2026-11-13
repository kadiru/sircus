"""Audit frozen inter-grid proposals and rescored native64 trials on CPU."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

from . import replay650_verify as replay
from .benchmark_verify import Checks, contained, digest, fingerprint, read, sha
from .calibrated_admission import verify_connection, verify_resolution
from .cell1050_followup_plan import geometry_key
from .cell1050_rebound_audit import verify_heading
from .cell1050_rebound_study import control_requests, parent_inputs
from .intergrid_dataset import _grid
from .intergrid_layouts import MIN_SEPARATION_M, _placement_geometry
from .intergrid_proposer import fit_proposer, load_checkpoint
from .intergrid_proposer_plan import SEED, collection_proposals, sample_splits
from .intergrid_proposer_study import (
    METHODS, VERSION as STUDY_VERSION, expanded_library, geometry_prior, load_prior,
    test_proposals, training_rows,
)
from .intergrid_study import stable_captures
from .parallel_capture_grid_audit import observed_rows, rescore_packet, verify_controls, verify_release
from .policy64_verify import audit_native_options
from .sweep_worker import atomic


VERSION = "sircus-intergrid-proposer-independent-audit-v1"
PHASES = ("controls_pre", "train", "test", "controls_post")


def audit_splits(study, parent, previous, scene, assets, checks):
    artifact = read(study / "splits.json")
    split, provenance = artifact["splits"], artifact["provenance"]
    checks.equal(set(split), {"train", "test"}, "Only TRAIN and TEST placements")
    checks.equal((len(split["train"]), len(split["test"])), (24, 50), "24 new TRAIN and 50 untouched TEST layouts")
    checks.equal(provenance["seed"], SEED, "Prespecified spatial split seed")
    checks.equal(provenance["minimum_separation_m"], MIN_SEPARATION_M, "Prespecified spatial exclusion threshold")
    geometry = _placement_geometry(scene, assets)
    checks.equal(provenance["geometry"], geometry, "CPU-reconstructed physical placement geometry")
    exclusions = [tuple(xy) for xy in provenance["excluded_coordinates_m"]]
    checks.equal(exclusions, sorted(set(exclusions)), "Sorted unique historical exclusion coordinates")
    checks.equal(digest(exclusions), provenance["excluded_coordinates_sha256"], "Frozen exclusion coordinates")
    checks.equal(digest(split), provenance["splits_sha256"], "Frozen placements")
    prior_xy = sorted({tuple(row["bucket_xy_m"]) for row in previous})
    checks.equal(provenance["declared_previous_coordinates_m"], [list(xy) for xy in prior_xy], "Exact previous TEST layout exclusions")
    checks.require(set(prior_xy).issubset(exclusions), "Previously tested coordinates excluded from new TRAIN and TEST")
    grid = read(parent / "grid.json")
    checks.require({tuple(row["bucket_xy_m"]) for row in grid}.issubset(exclusions), "All original grid coordinates excluded")
    checks.equal(split, sample_splits(exclusions, geometry, grid, seed=SEED), "Reproduce frozen placements from geometry, seed and exclusion inventory")
    placements = {}
    coordinates = []
    for name in ("train", "test"):
        rows = split[name]
        checks.require(bool(rows), name + "/nonempty layout split")
        for layout in rows:
            label = layout["id"]
            xy = tuple(layout["bucket_xy_m"])
            checks.require(len(xy) == 2 and all(math.isfinite(value) for value in xy), label + "/finite XY")
            checks.equal(layout["split"], name, label + "/role declared before collection")
            checks.require(label not in placements, label + "/unique layout ID")
            checks.require(all(math.dist(xy, other) >= MIN_SEPARATION_M for other in coordinates),
                           label + "/at least 5 mm from other frozen placements")
            coordinates.append(xy)
            placements[label] = layout
    for label, xy in _grid().values():
        checks.require(all(math.dist(xy, row["bucket_xy_m"]) >= MIN_SEPARATION_M for row in split["test"]),
                       label + "/canonical TRAIN geometry disjoint from all TEST layouts")
    return split


def audit_preparation(artifact, layout, phase, directory, prior, home, resolutions, checks):
    identity = layout["id"]
    proposals, admitted, records = (artifact[key] for key in ("proposals", "admitted", "records"))
    checks.equal(len(records), len(proposals), identity + "/one setup record per proposal")
    checks.equal(len({item["id"] for item in proposals}), len(proposals), identity + "/unique proposal IDs")
    branches = {item["id"]: item for item in admitted}
    checks.equal(len(branches), len(admitted), identity + "/unique admitted branch IDs")
    ordered = []
    for proposal, record in zip(proposals, records, strict=True):
        label = proposal["id"]
        checks.equal((proposal["context_id"], proposal["bucket_xy_m"], proposal["phase"]),
                     (identity, layout["bucket_xy_m"], phase), label + "/exact frozen layout and role")
        checks.equal((record["proposal_id"], record["proposal_sha256"]), (label, digest(proposal)),
                     label + "/frozen proposal geometry")
        ids = record["admitted_request_ids"]
        checks.equal(len(ids), record["admitted"], label + "/exact admission count")
        checks.require(len(ids) <= 1, label + "/at most one admitted arm branch")
        for branch_id in ids:
            request = branches[branch_id]
            checks.equal(branch_id, label + "-b0", label + "/admitted branch identity")
            checks.equal({key: request[key] for key in proposal if key != "id"},
                         {key: value for key, value in proposal.items() if key != "id"},
                         label + "/admitted request retains exact proposal")
            checks.equal((request["proposal_id"], request["native_prepared"], request["command_values"]),
                         (label, True, proposal["values"]), label + "/exact admission command")
            name = request["context_resolution_file"]
            if name not in resolutions:
                path = contained(directory, name)
                anchor = dict(read(path), file=name, sha256=sha(path))
                verify_resolution(anchor, prior["records"], home)
                resolutions[name] = anchor
            anchor = resolutions[name]
            checks.equal((anchor["bucket_xy_m"], anchor["sha256"]),
                         (layout["bucket_xy_m"], request["context_resolution_sha256"]), label + "/exact anchor")
            path = contained(directory, request["prepared_connection_file"])
            proof = dict(read(path), file=request["prepared_connection_file"], sha256=sha(path))
            verify_connection(proof, anchor, home, request["values"])
            checks.equal((proof["passed"], proof["sha256"], proof["q10"], proof["q14"], proof["heading_key"],
                          proof["board_heading_offset_deg"], proof["board_twist_deg"]),
                         (True, request["prepared_connection_sha256"], request["selected_q10"], request["selected_q14"],
                          request["heading_key"], request["board_heading_offset_deg"], request["board_twist_deg"]),
                         label + "/verified exact arm paths and endpoint")
            ordered.append(branch_id)
    checks.equal(ordered, list(branches), identity + "/complete ordered admissions")
    return proposals, admitted


def audit_packets(study, parent, assets, spec, report, manifest, expected, prior, home, checks):
    calibration = replay.check_calibration(study, spec["identity"]["calibration"], checks)
    native = study / "native-heading"
    for namespace in ("native-heading", "native-controls"):
        if namespace == "native-heading" and not any(expected.values()):
            continue
        audit_native_options(study / namespace, spec["scene"], calibration, assets, checks)
    ledger = observed_rows(study / "rows.jsonl")
    timings = observed_rows(study / "timing.jsonl")
    sources = read(parent / "control-replays.json")
    used_archives, observed_ids, controls, phases = set(), set(), [], []
    observed = {name: {} for name in expected}
    resolutions = {}
    for index, path in enumerate(sorted((study / "requests").glob("*.json"))):
        relative = f"requests/batch-{index:05d}.json"
        checks.equal(path.relative_to(study).as_posix(), relative, "Contiguous native request packets")
        packet = read(path)
        phase, requests = packet["phase"], packet["rows"]
        phases.append(PHASES.index(phase))
        checks.require(0 < len(requests) <= 32, relative + "/bounded native packet")
        timing = next(timings, None)
        checks.require(timing is not None, relative + "/saved timing")
        checks.equal((timing["packet"], timing["phase"]), (relative, phase), relative + "/matching timing")
        if phase in expected:
            start = len(observed[phase])
            checks.equal(requests, expected[phase][start:start + len(requests)], relative + "/frozen admitted requests")
        else:
            checks.equal(requests, control_requests(sources, phase), relative + "/exact parent controls")
        rows = []
        for request in requests:
            row = next(ledger, None)
            checks.require(row is not None, relative + "/native observation exists")
            verify_release(row, request, relative, manifest[relative], phase, checks)
            identity = row["id"]
            checks.require(identity not in observed_ids, identity + "/unique native observation")
            observed_ids.add(identity)
            if phase in expected:
                checks.equal((row["adapter_directory"], row["proposal_id"], row["method"], row["role"]),
                             ("native-heading", request["proposal_id"], request["method"], phase),
                             identity + "/exact chosen proposal and split")
                verify_heading(row, request, native, spec["scene"], home, prior, resolutions, checks)
            else:
                checks.equal(row["adapter_directory"], "native-controls", identity + "/control archive")
            rows.append(row)
        if phase in expected:
            outcomes = rescore_packet(native, rows, spec["scene"], calibration, manifest, used_archives, checks)
            for row, outcome in zip(rows, outcomes, strict=True):
                observed[phase][row["id"]] = (row, "invalid_physics" if row["invalid_physics"] else outcome)
        else:
            controls.extend(rows)
    checks.equal(phases, sorted(phases), "Pre-controls, collection, frozen one-shot test and post-controls ordered")
    checks.equal((phases.count(0), phases.count(3)), (1, 1), "One pre-control and one post-control packet")
    for phase, requests in expected.items():
        checks.equal(list(observed[phase]), [row["id"] for row in requests], phase + "/all and only frozen requests released")
        checks.equal(len(observed[phase]), report[phase + "_releases"], phase + "/exact report count")
    checks.require(next(ledger, None) is None and next(timings, None) is None, "Every observation and timing belongs to a packet")
    checks.equal(used_archives, {path.relative_to(native).as_posix() for path in native.rglob("*.npz")},
                 "Complete native TRAIN and TEST archive inventory")
    verify_controls(study, parent, spec, report, controls, calibration, assets, checks)
    return observed, len(controls)


def audit_checkpoint(study, rows, spec, checks):
    checkpoint = load_checkpoint(study / "proposer.json")
    checks.equal((checkpoint["seed"], checkpoint["device"], checkpoint["base_positions"]),
                 (spec["seed"], "cuda:0", 33), "Fit used the authorized device with the frozen seed and 33 base positions")
    reproduced = fit_proposer(rows, seed=checkpoint["seed"], device="cpu")
    for key, value in reproduced.items():
        if key in {"coefficients", "normalized_training_rmse"}:
            checks.numeric(checkpoint[key], value, "CPU reconstruction of model " + key, 1e-9)
        elif key not in {"device", "fingerprint"}:
            checks.equal(checkpoint[key], value, "Exact frozen training and model " + key)
    groups = checkpoint["bucket_groups"]
    checks.require(all(group["loss_weight"] == 1 / len(groups) for group in groups),
                   "Equal contribution from every target training position used in fit")
    return checkpoint


def audit_training_split(rows, split, checks):
    canonical = {identity: xy for identity, xy in _grid().values()}
    new_train = {layout["id"]: layout["bucket_xy_m"] for layout in split["train"]}
    checks.require(not set(canonical) & set(new_train), "New TRAIN IDs do not collide with canonical buckets")
    train = canonical | new_train
    checks.equal(len({row["id"] for row in rows}), len(rows), "Distinct training sample IDs")
    for row in rows:
        label = row["id"]
        checks.equal(row["bucket_xy_m"], train[row["context_id"]], label + "/belongs to frozen TRAIN layout")
        checks.require(all(math.dist(row["bucket_xy_m"], layout["bucket_xy_m"]) >= MIN_SEPARATION_M
                           for layout in split["test"]), label + "/does not include a held-out test position")
        checks.require(row["native_release"] is True and not row["invalid"] and row["invalid_physics"] is False,
                       label + "/only valid native release evidence enters model")


def audit_proposal_geometry(proposals, known, saved_geometry, checkpoint, checks):
    for proposal in proposals:
        label = proposal["id"]
        source = known[proposal["training_recipe_id"]]
        checks.require(source["route_success"] is True and not source.get("label_conflict"),
                       label + "/valid known capture as source")
        checks.equal((proposal["training_bucket_xy_m"], proposal["seed_q10"], proposal["seed_q14"]),
                     (source["bucket_xy_m"], source["q10"], source["q14"]), label + "/exact training source and arm seeds")
        if proposal["method"] == "learned":
            checks.require(geometry_key(proposal) not in saved_geometry, label + "/predicts new numerical shot settings")
            checks.equal(proposal["proposer_fingerprint"], checkpoint["fingerprint"], label + "/frozen model provenance")
        else:
            checks.equal(proposal["method"], "nearest_feasible", label + "/paired comparison method")
            checks.equal(geometry_key(proposal), geometry_key(source), label + "/known shot geometry unchanged")


def audit_comparison(selections, outcomes):
    methods = {method: Counter() for method in METHODS}
    strata = defaultdict(lambda: {method: Counter() for method in METHODS})
    per_context = defaultdict(dict)
    for slot in selections:
        method = slot["method"]
        status = outcomes.get(slot["id"], "setup_rejected")
        methods[method][status] += 1
        strata[slot["stratum"]][method][status] += 1
        per_context[slot["context_id"]][method] = (slot, status)
    labels = {frozenset(): "neither", frozenset(METHODS): "both",
              frozenset([METHODS[0]]): "nearest_only", frozenset([METHODS[1]]): "learned_only"}
    paired, both_released = Counter(), Counter()
    for pair in per_context.values():
        category = labels[frozenset(method for method, (_, status) in pair.items() if status == "standard_capture")]
        paired[category] += 1
        if all(slot["id"] in outcomes for slot, _ in pair.values()):
            both_released[category] += 1
    return dict(methods={method: dict(counts) for method, counts in methods.items()},
                strata={key: {method: dict(counts) for method, counts in counters.items()}
                        for key, counters in strata.items()}, paired=dict(paired), both_released=dict(both_released))


def audit_collection(study, layouts, library, prior, home, resolutions, checks):
    menus = read(study / "collection-menu.json")
    checks.equal(menus, [collection_proposals(layout, library) for layout in layouts],
                 "Collection proposals derive only from the canonical training dataset")
    folder = study / "collection"
    checks.equal({p.stem for p in folder.glob("*.json")}, {layout["id"] for layout in layouts},
                 "Exactly one preparation record per collection layout")
    requests = []
    for layout, menu in zip(layouts, menus, strict=True):
        artifact = read(folder / (layout["id"] + ".json"))
        checks.equal(artifact["proposals"], menu, layout["id"] + "/frozen collection menu")
        _, admitted = audit_preparation(artifact, layout, "train", study / "native-heading", prior, home, resolutions, checks)
        requests.extend(admitted)
    return requests


def audit_choices(study, layouts, library, training, checkpoint, prior, home, resolutions, checks):
    known = {row["id"]: row for row in training}
    saved_geometry = {geometry_key(row) for row in training}
    folder = study / "test"
    checks.equal({p.stem for p in folder.glob("*.json")}, {layout["id"] for layout in layouts},
                 "Exactly one preparation record per TEST layout")
    selections = []
    for layout in layouts:
        identity = layout["id"]
        artifact = read(folder / (identity + ".json"))
        checks.equal(artifact["proposals"], test_proposals(layout, library, checkpoint, saved_geometry),
                     identity + "/frozen checkpoint reproduces all proposed settings and source order")
        proposals, admitted = audit_preparation(artifact, layout, "test", study / "native-heading", prior, home,
                                                resolutions, checks)
        audit_proposal_geometry(proposals, known, saved_geometry, checkpoint, checks)
        for method in METHODS:
            chosen = next((row for row in admitted if row["method"] == method), None)
            identifier = f"{identity}-{method}"
            selections.append(dict(id=identifier, method=method, context_id=identity, bucket_xy_m=layout["bucket_xy_m"],
                                   stratum=layout["stratum"], request=None if chosen is None else dict(
                                       chosen, id=identifier, stratum=layout["stratum"],
                                   )))
    choices = read(study / "choices.json")
    checks.equal(choices["checkpoint_sha256"], sha(study / "proposer.json"), "Frozen choices bind checkpoint bytes")
    checks.equal(choices["slots"], selections, "Exactly one paired slot and first admitted command for each method")
    return selections


def audit(study, parent, parent_audit, prior_study, prior_audit, assets):
    checks = Checks()
    spec, report = read(study / "contract.json"), read(study / "report.json")
    parent_spec, parent_binding, _ = parent_inputs(parent, parent_audit)
    checks.equal((spec["version"], report["status"], report["phase"]), (STUDY_VERSION, "completed", "completed"),
                 "Completed versioned continuous proposer comparison")
    fingerprint(spec, checks, "Frozen comparison contract")
    checks.equal((spec["identity"], spec["scene"], spec["parent"]),
                 (parent_spec["identity"], parent_spec["scene"], parent_binding), "Exact audited native64 parent")
    checks.equal((spec["canonical_positions"], spec["collection_positions"], spec["test_positions"],
                  spec["methods"], spec["attempts_per_method_per_position"], spec["physics_substeps"]),
                 (33, 24, 50, list(METHODS), 1, 64), "Frozen collection size, baseline and one paired native64 attempt")
    manifest = replay.check_manifest(study, checks)
    checks.equal(report["evidence_manifest_sha256"], sha(study / "evidence-manifest.json"), "Report binds complete evidence bytes")
    frozen = ("splits.json", "collection-menu.json", "training-prior.json")
    checks.equal(spec["frozen_inputs"], {name: manifest[name] for name in frozen}, "All pre-collection inputs match frozen contract")
    checks.equal(report["choices_sha256"], manifest["choices.json"], "Report binds frozen choices")
    canonical, prior_binding = load_prior(prior_study, prior_audit)
    checks.equal(spec["prior_comparison"], prior_binding, "Exact previous comparison and audited canonical dataset")
    previous_path = prior_study / "test-layouts.json"
    checks.equal(sha(previous_path), read(prior_study / "evidence-manifest.json")["test-layouts.json"],
                 "Previous holdout layouts authenticated")
    split = audit_splits(study, parent, read(previous_path)["rows"], spec["scene"], assets, checks)
    library = stable_captures(canonical)
    prior, home = read(study / "training-prior.json"), read(parent / "setup-home.json")
    checks.equal(prior, geometry_prior(library), "Geometry-only motion anchors derive from canonical captures")
    resolutions = {}
    train_requests = audit_collection(study, split["train"], library, prior, home, resolutions, checks)
    fresh, exclusions = training_rows(study, split["train"], manifest)
    training = list(observed_rows(study / "training-dataset.jsonl"))
    checks.equal(training, canonical + fresh, "Dataset contains exact canonical and valid new TRAIN releases only")
    audit_training_split(training, split, checks)
    checkpoint = audit_checkpoint(study, training, spec, checks)
    expanded = expanded_library(canonical, fresh)
    new_positions = len(expanded) - len(library)
    checks.require(new_positions > 0, "At least one newly sampled TRAIN position produced a capture")
    checks.equal(report["training"], dict(canonical_commands=len(canonical), new_valid_commands=len(fresh),
                 new_standard_captures=sum(row["route_success"] for row in fresh), new_positions_with_capture=new_positions,
                 exclusions=exclusions, checkpoint_sha256=manifest["proposer.json"]), "Exact training evidence and checkpoint report")
    selections = audit_choices(study, split["test"], expanded, training, checkpoint, prior, home, resolutions, checks)
    expected = dict(train=train_requests, test=[slot["request"] for slot in selections if slot["request"] is not None])
    observations, controls = audit_packets(study, parent, assets, spec, report, manifest, expected, prior, home, checks)
    comparison = audit_comparison(selections, {key: score for key, (_, score) in observations["test"].items()})
    checks.equal(report["comparison"], comparison, "Exact paired results from independently rescored native evidence")
    checks.equal((report["collection_prepared"], report["test_prepared"]), (24, 50), "All frozen placements prepared")
    return dict(version=VERSION, status="passed", checks_count=checks.count,
                train_releases=len(observations["train"]), test_releases=len(observations["test"]), control_releases=controls,
                training=report["training"], comparison=comparison, report_sha256=sha(study / "report.json"),
                evidence_manifest_sha256=report["evidence_manifest_sha256"], checkpoint_sha256=manifest["proposer.json"],
                choices_sha256=manifest["choices.json"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "parent-study", "parent-audit", "prior-study", "prior-audit", "assets", "audit"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    if args.audit.exists() or args.audit.resolve().is_relative_to(args.study.resolve()):
        parser.error("Write a fresh audit outside the frozen evidence directory")
    try:
        result = audit(args.study, args.parent_study, args.parent_audit, args.prior_study, args.prior_audit, args.assets)
    except Exception as error:
        result = dict(version=VERSION, status="failed", error=f"{type(error).__name__}: {error}")
    atomic(args.audit, result)
    print(json.dumps(result), flush=True)
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
