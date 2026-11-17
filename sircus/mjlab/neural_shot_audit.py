"""Read-only independent scoring and accounting for the neural shot selector."""
from __future__ import annotations

from collections import Counter
import copy
from itertools import combinations
import hashlib
import json
import math
from pathlib import Path
from statistics import median

from .benchmark_verify import Checks, contained, digest, fingerprint, read, sha

VERSION = "sircus-neural-shot-independent-audit-v1"
STUDY_VERSION = "sircus-neural-shot-study-v1"
METHODS = ("current_planner", "clearance_expansion", "neural_selector")
SPLITS = ("train", "validation", "test")
ADAPTERS = {"train": "native-collection", "validation": "native-collection",
            "test_baseline": "native-test-baseline", "test_neural": "native-test-neural"}
STANDARD_CAPTURES = {"direct", "table", "unclassified_capture"}


def _finite_seconds(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("Planning latency must be finite and nonnegative")
    return value


def _latency(values):
    values = sorted(_finite_seconds(value) for value in values)
    if not values:
        raise ValueError("Planning latency requires every test position")
    return dict(positions=len(values), median_s=median(values),
                p90_s=values[math.ceil(.9 * len(values)) - 1], max_s=values[-1],
                aggregate_s=math.fsum(values))


def summarize(layouts, slots, observations, baseline_sites, neural_sites):
    """Keep refused setups, invalid physics and robot-assisted shots visible."""
    from .intergrid_candidate_replay_plan import command_key
    from .intergrid_route_selector_plan import _account

    ids, performance, attempts = _account(layouts, slots, observations, METHODS)
    if ([site["layout"] for site in baseline_sites] != layouts
            or [site["layout"] for site in neural_sites] != layouts):
        raise ValueError("Every method needs the exact ordered frozen test positions")
    by_layout = {layout["id"]: layout for layout in layouts}
    for slot in slots:
        layout = by_layout[slot["context_id"]]
        if slot["bucket_xy_m"] != layout["bucket_xy_m"]:
            raise ValueError("Slot bucket differs from its frozen position")
        request = slot["request"]
        if request is not None and (request["id"] != slot["id"]
                                   or request["context_id"] != layout["id"]
                                   or request["bucket_xy_m"] != layout["bucket_xy_m"]
                                   or request["method"] != slot["method"]):
            raise ValueError("Frozen request identity or bucket differs from its slot")
    for method, counts in performance.items():
        rows = [observations[slot["id"]] for slot in slots
                if slot["method"] == method and slot["request"] is not None]
        counts.update(positions=len(layouts), admitted=len(rows), released=len(rows), unresolved=len(layouts) - len(rows),
                      flight_misses=counts["outcomes"].get("flight_miss", 0),
                      robot_contacts=sum(row.get("experimental_route", {}).get("robot_contact_steps", 0) > 0
                                         for row in rows))
    paired, same = {}, {}
    for left, right in combinations(METHODS, 2):
        label = left + "_vs_" + right
        counts = dict(both=0, left_only=0, right_only=0, neither=0)
        identical = dict(positions=0, capture_disagreements=0, outcome_disagreements=0)
        for identity in sorted(ids):
            (lslot, lout), (rslot, rout) = (attempts[identity, method] for method in (left, right))
            lhit, rhit = lout in STANDARD_CAPTURES, rout in STANDARD_CAPTURES
            counts["both" if lhit and rhit else "left_only" if lhit else "right_only" if rhit else "neither"] += 1
            if lslot["request"] is not None and rslot["request"] is not None and (
                    command_key(lslot["request"]) == command_key(rslot["request"])):
                identical["positions"] += 1
                identical["capture_disagreements"] += int(lhit != rhit)
                identical["outcome_disagreements"] += int(lout != rout)
        paired[label] = dict(counts, gained=counts["right_only"], lost=counts["left_only"])
        same[label] = identical
    positions = [dict(id=layout["id"], bucket_xy_m=layout["bucket_xy_m"],
                      outcomes={method: attempts[layout["id"], method][1] for method in METHODS})
                 for layout in layouts]
    latency = {method: _latency([site["latency_s"][method] for site in
                                (neural_sites if method == METHODS[-1] else baseline_sites)])
               for method in METHODS}
    return dict(methods=performance, paired=paired, same_command=same,
                positions=positions, latency=latency)


def audit_splits(archive, counts, checks, *, minimum_separation_m=.005):
    """Verify split membership and spatial disjointness independently of labels."""
    checks.equal(set(archive), set(SPLITS), "Exactly TRAIN, validation and TEST archives")
    checks.equal(set(counts), set(SPLITS), "Counts declared separately for all three splits")
    seen_ids, seen_xy = set(), []
    for split in SPLITS:
        rows = archive[split]["layouts"]
        checks.require(type(counts[split]) is int and counts[split] > 0, split + "/positive declared count")
        checks.equal(len(rows), counts[split], split + "/exact frozen position count")
        provenance = archive[split].get("provenance", {})
        if "layouts_sha256" in archive[split]:
            checks.equal(digest(rows), archive[split]["layouts_sha256"], split + "/frozen layout bytes")
        exclusions = provenance.get("excluded_coordinates_m", [])
        for layout in rows:
            identity, xy = layout["id"], layout["bucket_xy_m"]
            checks.require(isinstance(identity, str) and bool(identity) and identity not in seen_ids,
                           split + "/globally unique position identity")
            checks.equal((layout["context_id"], layout["split"]), (identity, split),
                         identity + "/unambiguous split and context")
            checks.require(len(xy) == 2 and all(type(v) in (int, float) and math.isfinite(v) for v in xy),
                           identity + "/finite XY")
            checks.require(all(math.dist(xy, other) >= minimum_separation_m for other in seen_xy),
                           identity + "/separate TRAIN, validation and TEST positions")
            checks.require(all(math.dist(xy, other) >= minimum_separation_m for other in exclusions),
                           identity + "/excluded historical positions remain excluded")
            seen_ids.add(identity)
            seen_xy.append(xy)
    return {name: archive[name]["layouts"] for name in SPLITS}


def verify_bindings(study, bindings, checks, label):
    checks.require(isinstance(bindings, dict) and bool(bindings), label + "/nonempty immutable file binding")
    for name, expected in bindings.items():
        checks.equal(sha(contained(study, name)), expected, label + "/" + name)


def verify_test_choices(study, layouts, manifest, checks):
    """Require checkpoint and development labels to be bound before TEST."""
    artifact = read(study / "choices-test.json")
    bindings = artifact["bindings"]
    required = {"neural-checkpoint.json", "fit.json", "training-dataset.jsonl", "validation-dataset.jsonl",
                "plans-test-baseline.json", "plans-test-neural.json"}
    checks.require(required <= set(bindings), "Test choices bind model, development labels and all plans")
    verify_bindings(study, bindings, checks, "Pre-release test selection")
    checks.require(all(manifest[name] == expected for name, expected in bindings.items()),
                   "Final evidence retains every pre-release selection dependency")
    baseline = read(study / "plans-test-baseline.json")
    neural = read(study / "plans-test-neural.json")
    checks.equal([row["layout"] for row in baseline], layouts, "Baseline prepares exact untouched test positions")
    checks.equal([row["layout"] for row in neural], layouts, "Neural prepares exact untouched test positions")
    slots = [choice for old, new in zip(baseline, neural, strict=True)
             for choice in [*old["choices"], *new["choices"]]]
    checks.equal(artifact["slots"], slots, "Exact three frozen choices per site")
    checks.equal([(slot["context_id"], slot["method"]) for slot in slots],
                 [(layout["id"], method) for layout in layouts for method in METHODS],
                 "Exactly one requested attempt per method at every test position")
    checks.equal(len({slot["id"] for slot in slots}), len(slots), "Globally unique requested test IDs")
    return slots, baseline, neural


def _collection_requests(study, splits, prior, home, checks):
    from .intergrid_proposer_audit import audit_preparation

    result, sites = {}, []
    resolutions = {}
    for phase in ("train", "validation"):
        folder = study / "preparation" / phase
        checks.equal({path.stem for path in folder.glob("*.json")}, {row["id"] for row in splits[phase]},
                     phase + "/complete preparation inventory")
        requests = []
        for layout in splits[phase]:
            site = read(folder / (layout["id"] + ".json"))
            checks.equal(site["layout"], layout, layout["id"] + "/exact development position")
            proposal_phase = site["proposals"][0]["phase"] if site["proposals"] else phase
            checks.require(proposal_phase == phase or (phase == "validation" and proposal_phase == "train"),
                           layout["id"] + "/explicit validation adapter phase")
            _, admitted = audit_preparation(site, layout, proposal_phase, study / "native-collection",
                                            prior, home, resolutions, checks)
            by_id = {row["id"]: row for row in admitted}
            selected = site["requests"]
            checks.require(type(site["max_releases"]) is int and 0 <= len(selected) <= site["max_releases"],
                           layout["id"] + "/bounded development releases")
            checks.equal(len({row["id"] for row in selected}), len(selected), layout["id"] + "/unique development releases")
            for row in selected:
                checks.equal(row, by_id.get(row["id"]), row["id"] + "/exact admitted development command")
            requests.extend(selected)
            sites.append(site)
        result[phase] = requests
    return result, sites


def _check_receipt(study, stage, requests, manifest, checks):
    receipt = read(study / ("native-started-" + stage + ".json"))
    checks.equal(receipt["stage"], stage, stage + "/exact native stage")
    checks.equal(receipt["requests_sha256"], digest(requests), stage + "/commands frozen before release")
    verify_bindings(study, receipt["bindings"], checks, stage + "/immutable start receipt")
    checks.require(all(manifest[name] == expected for name, expected in receipt["bindings"].items()),
                   stage + "/start bindings remain in final evidence")
    if stage == "test":
        checks.require({"choices-test.json", "neural-checkpoint.json", "fit.json", "training-dataset.jsonl",
                        "validation-dataset.jsonl"}
                       <= receipt["bindings"].keys(), stage + "/model and labels frozen before TEST")


def _model_digest(value):
    """Match the published checkpoint serialization, not the runner hash format."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _training_outcome(record):
    """Independent literal label interpretation; never use model-side labels."""
    setup, row = record["setup_ok"], record["observation"]
    if type(setup) is not bool:
        raise ValueError("Dataset setup labels must be Boolean")
    if not setup:
        if row is not None:
            raise ValueError("Rejected setup cannot have a native label")
        return "setup_rejected"
    if row is None:
        return "setup_only"
    if (row.get("native_release") is not True or type(row.get("invalid_physics")) is not bool
            or type(row.get("route_success")) is not bool):
        raise ValueError("Dataset needs its exact scored native observation")
    if row["invalid_physics"]:
        return "invalid_physics"
    assisted = row["experimental_route"]["robot_assisted_capture"]
    if type(assisted) is not bool:
        raise ValueError("Robot-contact capture classification must be explicit")
    if row["route_success"]:
        if assisted or row.get("shot_route") not in {"direct", "table", None}:
            raise ValueError("Standard and robot-assisted capture labels conflict")
        return row.get("shot_route") or "unclassified_capture"
    return "arm_assisted_capture" if assisted else "flight_miss"


def verify_learning(study, sites, observed, splits, checks):
    """Reconstruct every learning row from pre-IK proposals and verified trials."""
    from .parallel_capture_grid_audit import observed_rows
    from .neural_shot_model import _validate_checkpoint

    datasets = {}
    for split, filename in (("train", "training-dataset.jsonl"), ("validation", "validation-dataset.jsonl")):
        expected = []
        for site in sites:
            if site["layout"]["split"] != split:
                continue
            by_proposal = {row["proposal_id"]: row for row in site["admitted"]}
            for proposal in site["proposals"]:
                admitted = by_proposal.get(proposal["id"])
                observation = observed[split].get(admitted["id"]) if admitted else None
                expected.append(dict(request=proposal, setup_ok=admitted is not None,
                                     observation=observation, split=split))
        actual = list(observed_rows(study / filename))
        checks.equal(actual, expected, split + "/exact original proposals, admissions and independently verified labels")
        checks.equal(len({row["request"]["id"] for row in actual}), len(actual),
                     split + "/every proposal appears once")
        positions = {tuple(row["request"]["bucket_xy_m"]) for row in actual}
        checks.equal(positions, {tuple(layout["bucket_xy_m"]) for layout in splits[split]},
                     split + "/every frozen development position contributes setup evidence")
        datasets[split] = actual
    checkpoint, fit = read(study / "neural-checkpoint.json"), read(study / "fit.json")
    _validate_checkpoint(checkpoint)
    checks.equal(checkpoint["fingerprint"],
                 _model_digest({key: value for key, value in checkpoint.items() if key != "fingerprint"}),
                 "Frozen network coefficient fingerprint")
    checks.equal(checkpoint["device"], "cpu", "Small neural fit does not allocate a second GPU")
    for split, prefix in (("train", "training"), ("validation", "validation")):
        rows = datasets[split]
        checks.equal(checkpoint[prefix + "_sha256"], _model_digest(rows),
                     split + "/checkpoint binds exact development data")
        checks.equal(checkpoint[prefix + "_rows"], len(rows), split + "/exact model sample count")
        checks.equal(checkpoint[prefix + "_positions"], len(splits[split]), split + "/exact model position count")
        checks.equal(checkpoint[prefix + "_outcomes"], dict(sorted(Counter(map(_training_outcome, rows)).items())),
                     split + "/invalid physics and unreleased setups never become native misses")
    validation = fit["validation"]
    history = validation["history"]
    checks.equal([row["epoch"] for row in history], checkpoint["checkpoint_epochs"],
                 "Only declared epoch candidates evaluated for validation selection")
    checks.require(bool(history) and all(type(row["loss"]) in (int, float) and math.isfinite(row["loss"])
                                        for row in history), "Finite held-out validation loss")
    chosen = min(history, key=lambda row: (row["loss"], row["epoch"]))
    checks.equal((checkpoint["epoch"], validation["selected_epoch"]), (chosen["epoch"], chosen["epoch"]),
                 "Frozen epoch selected solely by validation loss")
    checks.equal(validation["checkpoint_sha256"], _model_digest(checkpoint), "Validation binds final network")
    checks.equal(validation["selected"],
                 {key: value for key, value in chosen.items() if key not in {"epoch", "train_loss"}},
                 "Recorded selected metrics match the chosen validation epoch")
    return checkpoint


def verify_test_plans(study, spec, assets, baseline, neural, checkpoint, prior, home, checks):
    """Reconstruct selection from frozen pre-release information, never outcomes."""
    from .clearance_expansion_audit import audit_branches
    from .clearance_expansion_plan import (
        STAGES, _all_admitted, _empty, branch_proposals, current_selection,
        expansion_proposals, needs_expansion, select_requests,
    )
    from .intergrid_flight_ranker import FlightRanker
    from .intergrid_library_plan import libraries
    from .intergrid_proposer_audit import audit_preparation
    from .intergrid_system_plan import base_proposals, fallback_proposals
    from .neural_shot_model import order, predict
    from .neural_shot_plan import CHECK_BATCH, CHECK_LIMIT, _choice, menu

    source_rows = read(study / "shot-library-rows.json")
    shot_libraries = libraries(source_rows["original"], source_rows["additions"])
    source_checkpoint = read(study / "route-models.json")
    history = read(study / "route-history.json")
    calibration = read(study / "calibration.json")["score_config"]
    ranker = FlightRanker(spec["scene"], calibration, assets, spec["flight_ranking"]["rule"])
    for sites, directory in ((baseline, study / "native-test-baseline"), (neural, study / "native-test-neural")):
        resolutions = {}
        for site in sites:
            layout, plan = site["layout"], site["plan"]
            identity = layout["id"]
            folder = "test-baseline" if directory.name == "native-test-baseline" else "test-neural"
            checks.equal(site, read(study / "preparation" / folder / (identity + ".json")),
                         identity + "/" + folder + "/per-position plan preserved")
            if folder == "test-baseline":
                checks.equal(set(plan), set(STAGES), identity + "/unchanged baseline preparation stages")
                checks.equal(plan["base"]["proposals"], base_proposals(layout, shot_libraries, source_checkpoint),
                             identity + "/exact frozen current-planner menu")
                base, admitted = audit_preparation(plan["base"], layout, "test", directory, prior, home, resolutions, checks)
                checks.equal(plan["fallback"]["proposals"], fallback_proposals(layout, shot_libraries, base, admitted),
                             identity + "/exact setup-only fallback")
                audit_preparation(plan["fallback"], layout, "test", directory, prior, home, resolutions, checks)
                original = current_selection(layout, plan, history, ranker)
                trigger = needs_expansion(original)
                checks.equal(site["trigger"], trigger, identity + "/pre-release clearance expansion trigger")
                checks.equal(plan["expansion"]["proposals"],
                             expansion_proposals(layout, shot_libraries, plan) if trigger else [],
                             identity + "/exact additional clearance geometry")
                audit_preparation(plan["expansion"], layout, "test", directory, prior, home, resolutions, checks)
                checks.equal(plan["branches"]["proposals"], branch_proposals(layout, plan, ranker) if trigger else [],
                             identity + "/unchanged baseline branch proposal rule")
                originals = {row["id"]: row for row in _all_admitted(plan, STAGES[:3])}
                audit_branches(plan["branches"], layout, originals, directory, prior, home, resolutions, checks)
                checks.equal(site["predictions"], [dict(id=row["id"], **ranker.predict(row)) for row in _all_admitted(plan)],
                             identity + "/pre-release physics predictions reconstructed")
                checks.equal(site["choices"], select_requests(layout, plan, history, ranker),
                             identity + "/current and wider-search commands reconstructed")
                continue
            proposals = menu(layout, shot_libraries, source_checkpoint)
            checks.equal(plan["proposals"], proposals, identity + "/neural candidates generated without IK or outcomes")
            ordered = order(checkpoint, proposals)
            checked = plan["checked"]
            checks.require(0 < len(checked) <= min(CHECK_LIMIT, len(ordered)), identity + "/bounded pre-IK shortlist")
            checks.equal(checked, ordered[:len(checked)], identity + "/check only a prefix of frozen neural ranking")
            checks.require(len(checked) % CHECK_BATCH == 0 or len(checked) == len(ordered),
                           identity + "/complete bounded check batches")
            checked_ids = {row["id"] for row in checked}
            initial = [row for row in plan["admitted"] if row["proposal_id"] in checked_ids]
            stage = dict(proposals=checked, admitted=initial, records=plan["records"])
            audit_preparation(stage, layout, "test", directory, prior, home, resolutions, checks)
            ranking_ids = [row["id"] for row in ordered]
            for stop in range(CHECK_BATCH, len(checked), CHECK_BATCH):
                prefix = {row["id"] for row in checked[:stop]}
                choice = _choice(layout, [row for row in initial if row["proposal_id"] in prefix],
                                 ranker, stop, ranking_ids)
                checks.require(needs_expansion(choice), identity + "/no unnecessary checks after positive clearance")
            choice = _choice(layout, initial, ranker, len(checked), ranking_ids)
            trigger = needs_expansion(choice) and bool(initial)
            temporary = dict(base=stage, fallback=_empty(), expansion=_empty(), branches=_empty())
            checks.equal(plan["branches"]["proposals"], branch_proposals(layout, temporary, ranker) if trigger else [],
                         identity + "/only checked geometry can seed alternative joints")
            audit_branches(plan["branches"], layout, {row["id"]: row for row in initial}, directory,
                           prior, home, resolutions, checks)
            from .intergrid_candidate_replay_plan import command_key
            all_admitted = list({command_key(row): row for row in initial + plan["branches"]["admitted"]}.values())
            checks.equal(plan["admitted"], all_admitted, identity + "/no unchecked geometry added to selector")
            choice = _choice(layout, all_admitted, ranker, len(checked), ranking_ids)
            choice["predicted_flight_risk"] = needs_expansion(choice)
            checks.equal(site["choices"], [choice], identity + "/neural one-drop command reconstructed")
            checks.equal(site["predictions"], [dict(id=row["id"], **predict(checkpoint, row)) for row in proposals],
                         identity + "/rank probabilities use only original pre-IK proposals")


def audit_packets(study, parent, assets, spec, report, manifest, expected, prior, home, checks):
    """Rescore every native archive with independent scalar and contact scoring."""
    from . import replay650_verify as replay
    from .cell1050_rebound_audit import verify_heading
    from .cell1050_rebound_study import control_requests
    from .intergrid_route_labels import annotate_routes
    from .parallel_capture_grid_audit import observed_rows, rescore_packet, verify_release
    from .policy64_verify import audit_native_options
    from .table_training_verify import control_gate, rescore_rows

    calibration = replay.check_calibration(study, spec["identity"]["calibration"], checks)
    sources = read(parent / "control-replays.json")
    ledger, timings = observed_rows(study / "rows.jsonl"), observed_rows(study / "timing.jsonl")
    observed = {name: {} for name in expected}
    used_archives = {name: set() for name in set(ADAPTERS.values())}
    resolutions = {name: {} for name in used_archives}
    controls, identifiers, control_counts = {}, set(), Counter()
    stage_order = []
    for index, path in enumerate(sorted((study / "requests").glob("*.json"))):
        relative = f"requests/batch-{index:05d}.json"
        checks.equal(path.relative_to(study).as_posix(), relative, "Contiguous native packet ledger")
        packet = read(path)
        phase, requests = packet["phase"], packet["rows"]
        stage = phase
        control = phase.endswith("_controls_pre") or phase.endswith("_controls_post")
        if control:
            stage = phase.rsplit("_controls_", 1)[0]
            directory = "native-controls-" + stage
        else:
            checks.require(stage in expected, relative + "/declared native phase")
            directory = ADAPTERS[stage]
        checks.equal(packet.get("adapter_directory", directory), directory, relative + "/fixed native namespace")
        checks.equal(packet["stage"], ("collection" if stage in {"train", "validation"} else
                                       "test" if stage.startswith("test_") else stage),
                     relative + "/declared study lease")
        checks.equal(packet["control"], control, relative + "/control marker")
        checks.require(0 < len(requests) <= 32, relative + "/bounded native batch")
        timing = next(timings, None)
        checks.require(timing is not None, relative + "/saved timing exists")
        checks.equal((timing["packet"], timing["phase"]), (relative, phase), relative + "/timing belongs to batch")
        if control:
            checks.equal(requests, control_requests(sources, phase), relative + "/unchanged matched controls")
            control_counts[phase] += 1
        else:
            start = len(observed[stage])
            checks.equal(requests, expected[stage][start:start + len(requests)],
                         relative + "/exact ordered commands frozen before release")
        stage_order.append((stage, "pre" if phase.endswith("_controls_pre") else
                            "post" if phase.endswith("_controls_post") else "data"))
        rows = []
        for request in requests:
            row = next(ledger, None)
            checks.require(row is not None, relative + "/observation exists")
            expected_request_phase = ("train" if phase in {"train", "validation"} else
                                      "test" if phase.startswith("test_") and not control else phase)
            checks.equal(request["phase"], expected_request_phase, request["id"] + "/original adapter proposal phase")
            verify_release(row, dict(request, phase=phase), relative, manifest[relative], phase, checks)
            checks.require(row["id"] not in identifiers, row["id"] + "/unique native observation")
            identifiers.add(row["id"])
            checks.equal(row["adapter_directory"], directory, row["id"] + "/fixed adapter namespace")
            if not control:
                checks.equal((row["proposal_id"], row["method"]), (request["proposal_id"], request["method"]),
                             row["id"] + "/same selected preparation and method")
                verify_heading(row, request, study / directory, spec["scene"], home, prior,
                               resolutions[directory], checks)
            rows.append(row)
        if control:
            controls[phase] = rows
        else:
            # The existing scalar checker has a fixed manifest namespace.
            # Remap lookup keys only, retaining original bytes and checksums.
            alias = {"native-heading/" + name[len(directory) + 1:]: value
                     for name, value in manifest.items() if name.startswith(directory + "/")}
            classes = list(rescore_packet(study / directory, rows, spec["scene"], calibration,
                                          alias, used_archives[directory], checks))
            relabeled = annotate_routes(copy.deepcopy(rows), study / directory)
            for row, actual, route_class in zip(rows, relabeled, classes, strict=True):
                checks.equal((row["shot_route"], row["table_rebounds_before_entry"]),
                             (actual["shot_route"], actual["table_rebounds_before_entry"]),
                             row["id"] + "/independent observed direct/table route")
                observed[stage][row["id"]] = row
    checks.require(next(ledger, None) is None and next(timings, None) is None,
                   "No observations or timings outside frozen native packets")
    for stage, requests in expected.items():
        checks.equal(list(observed[stage]), [row["id"] for row in requests], stage + "/all frozen releases accounted for")
    for directory, used in used_archives.items():
        if used:
            audit_native_options(study / directory, spec["scene"], calibration, assets, checks)
        checks.equal(used, {path.relative_to(study / directory).as_posix()
                           for path in (study / directory).rglob("*.npz")}, directory + "/complete native archive inventory")
    all_controls = [row for rows in controls.values() for row in rows]
    for stage in ("collection", "test"):
        directory = study / ("native-controls-" + stage)
        stage_rows = [row for row in all_controls if row["phase"].startswith(stage + "_controls_")]
        audit_native_options(directory, spec["scene"], calibration, assets, checks)
        rescore_rows(directory, stage_rows, spec["scene"], calibration,
                     spec["identity"]["trace_schema"], assets, checks)
        checks.equal({row["native_cache_file"] for row in stage_rows},
                     {path.relative_to(directory).as_posix() for path in directory.rglob("*.npz")},
                     stage + "/complete control archive inventory")
    checks.require(bool(controls), "Matched control blocks are required")
    for phase, rows in controls.items():
        checks.equal(control_counts[phase], 1, phase + "/exactly one control block")
        gate = control_gate(rows)
        checks.equal(gate["status"], "passed", phase + "/matched control outcomes")
        checks.equal(report["controls"][phase], gate, phase + "/reported matched control gate")
    checks.equal(set(report["controls"]), set(controls), "Every reported control block has native evidence")
    native_stages = [("collection" if stage in {"train", "validation"} else
                     "test" if stage.startswith("test_") else stage, kind)
                     for stage, kind in stage_order]
    expected_groups = ("collection", "test")
    checks.equal([group for group, kind in native_stages if kind == "pre"], list(expected_groups),
                 "One pre-control gate before collection and TEST")
    checks.equal([group for group, kind in native_stages if kind == "post"], list(expected_groups),
                 "One post-control gate after collection and TEST")
    ranks = {group: i for i, group in enumerate(expected_groups)}
    order = [ranks[group] * 3 + {"pre": 0, "data": 1, "post": 2}[kind] for group, kind in native_stages]
    checks.equal(order, sorted(order), "Controls bracket each native stage in chronological order")
    return observed, len(all_controls)


def audit(args):
    """Verify completed native64 evidence without simulation or policy fitting."""
    from . import replay650_verify as replay
    from .cell1050_rebound_study import parent_inputs
    from .intergrid_layouts import _placement_geometry
    from .neural_shot_study import split_layouts
    from .intergrid_route_labels import LABEL_VERSION

    study, parent, checks = Path(args.output), Path(args.parent_study), Checks()
    spec, report = read(study / "contract.json"), read(study / "report.json")
    fingerprint(spec, checks, "Frozen neural shot study")
    checks.equal((spec["version"], report["status"]), (STUDY_VERSION, "completed"), "Completed declared study")
    checks.equal((spec["physics_substeps"], spec["methods"], spec["label_version"]),
                 (64, list(METHODS), LABEL_VERSION), "Fixed64 scoring and exact three compared methods")
    checks.equal((spec["split_counts"], spec["attempts_per_method_per_position"]),
                 (dict(train=160, validation=40, test=200), 1), "Prespecified positions and single-drop test budget")
    parent_spec, parent_binding, _ = parent_inputs(parent, args.parent_audit)
    checks.equal((spec["identity"], spec["scene"], spec["parent"]),
                 (parent_spec["identity"], parent_spec["scene"], parent_binding), "Exact audited native physics")
    checks.require(0 <= _finite_seconds(report["gpu_lease_s"]) <= spec["gpu_cap_s"] <= 7 * 3600,
                   "Cumulative native allocation stays within seven hours")
    manifest = replay.check_manifest(study, checks)
    checks.equal(sha(study / "evidence-manifest.json"), report["evidence_manifest_sha256"], "Complete final evidence manifest")
    verify_bindings(study, spec["frozen_inputs"], checks, "Frozen original study inputs")
    source = Path(__file__).resolve().parents[2]
    for name, expected in spec["source_sha256"].items():
        checks.equal(sha(contained(source, name)), expected, "Exact producer source/" + name)
    archive = read(study / "layouts.json")
    splits = audit_splits(archive, spec["split_counts"], checks)
    provenance = read(study / "layout-provenance.json")
    geometry = _placement_geometry(spec["scene"], args.assets)
    checks.equal(provenance["geometry"], geometry, "Independently reconstructed fixed table and robot bases")
    checks.equal(archive, split_layouts(provenance["excluded_coordinates_m"], geometry,
                                      read(parent / "grid.json"), seed=spec["seed"]),
                 "All 400 positions reconstructed from frozen geometry, exclusions and seed")
    prior, home = read(study / "training-prior.json"), read(parent / "setup-home.json")
    collection, sites = _collection_requests(study, splits, prior, home, checks)
    slots, baseline, neural = verify_test_choices(study, splits["test"], manifest, checks)
    expected = dict(collection)
    expected.update(test_baseline=[slot["request"] for slot in slots if slot["method"] in METHODS[:2]
                                  and slot["request"] is not None],
                    test_neural=[slot["request"] for slot in slots if slot["method"] == METHODS[-1]
                                and slot["request"] is not None])
    for stage, requests in (("collection", {key: expected[key] for key in ("train", "validation")}),
                           ("test", {key: expected[key] for key in ("test_baseline", "test_neural")})):
        _check_receipt(study, stage, requests, manifest, checks)
    observed, controls = audit_packets(study, parent, args.assets, spec, report, manifest, expected, prior, home, checks)
    for split in ("train", "validation", "test"):
        releases = (len(observed["test_baseline"]) + len(observed["test_neural"]) if split == "test"
                    else len(observed[split]))
        progress = report["progress"][split]
        checks.equal((progress["prepared_positions"], progress["completed_releases"], progress["planned_releases"]),
                     (spec["split_counts"][split], releases, releases), split + "/complete reported preparation and drops")
    checkpoint = verify_learning(study, sites, observed, splits, checks)
    verify_test_plans(study, spec, args.assets, baseline, neural, checkpoint, prior, home, checks)
    test = observed["test_baseline"] | observed["test_neural"]
    comparison = summarize(splits["test"], slots, test, baseline, neural)
    checks.equal(report["comparison"], comparison, "Final report exactly matches all frozen test outcomes")
    return dict(version=VERSION, status="passed", checks_count=checks.count, comparison=comparison,
                controls=controls, releases={phase: len(rows) for phase, rows in observed.items()},
                contract_fingerprint=spec["fingerprint"], report_sha256=sha(study / "report.json"),
                evidence_manifest_sha256=sha(study / "evidence-manifest.json"))
