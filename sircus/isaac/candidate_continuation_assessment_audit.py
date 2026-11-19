"""Independent final-only assessment verification and paired reporting.

No policy update, simulator, native dispatch or process operation is performed.
The caller supplies newly recomputed frozen model readouts from independently
verified checkpoint bytes; producer-selected actions are never accepted alone.
"""

from collections import Counter
import copy
from dataclasses import asdict
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from candidate_continuation_audit import _chain
from candidate_continuation_journal import read_events
from candidate_continuation_ledger import (ASSESSMENT_COHORTS, Ledger, METHODS,
    Outcome, PARENT_CHECKPOINT_SHA256, canonical_hash)
from candidate_continuation_native import (PACKET_SCHEMA as TRAIN_PACKET_SCHEMA,
    SCHEMA as NATIVE_BINDING_SCHEMA, _frozen_identity, guard_sources, source_fingerprint)
from candidate_continuation_native_audit import _digest, _file, _input_bytes, _json, _packet
from candidate_continuation_native_guards import verify_guards
from candidate_continuation_native_score import verify_score
from candidate_continuation_policy import FrozenCandidateSupport
from candidate_native_audit import verify_controls
from continuous_feasibility import ACTION_NAMES, LOW, HIGH, canonical_hash as geometry_hash
from prepare_assets import chain
from settled_transit import protocol as setup_protocol

SCHEMA = "sircus-candidate-continuation-assessment-audit-v1"
BOOTSTRAP_SEED = 20261128
BOOTSTRAP_REPLICATES = 10000
SPLITS = ("train", "validation", "test")
SPLIT_COUNTS = dict(train=35, validation=35, test=70)
TIE_TOLERANCE = 1e-12
RESULT_FIELDS = {"request_id", "released", "decline_reason", "source_sha256",
                 "return", "captured", "clean"}
PLAN_SCHEMA = "sircus-candidate-continuation-assessment-plan-v1"
PACKET_SCHEMA = "sircus-candidate-continuation-assessment-native-trial-v1"
RESULT_SCHEMA = "sircus-candidate-continuation-assessment-result-v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _cells(cells_by_split):
    require(type(cells_by_split) is dict and set(cells_by_split) == set(SPLITS),
            "Assessment requires the exact TRAIN/validation/TEST cell sets")
    lookup, contexts = {}, {}
    for split in SPLITS:
        cells = cells_by_split[split]
        require(type(cells) in (list, tuple) and len(cells) == SPLIT_COUNTS[split],
                "Assessment split requested-position count changed")
        contexts[split] = []
        for cell in cells:
            require(type(cell) is dict and type(cell.get("id")) is str and cell["id"]
                and cell["id"] not in lookup and cell.get("split") == split,
                "Assessment split/context association changed")
            xy = np.asarray(cell.get("bucket_xy_m"), dtype=float)
            require(xy.shape == (2,) and np.isfinite(xy).all(), "Invalid assessment bucket coordinate")
            lookup[cell["id"]] = cell
            contexts[split].append(cell["id"])
    return lookup, contexts


def _support(value):
    if value is None:
        return None
    value = FrozenCandidateSupport.from_dict(value) if type(value) is dict else value
    require(type(value) is FrozenCandidateSupport, "Expected immutable assessment support")
    return value


def independent_preference(distribution, support):
    """Rebuild scores and the fixed parent/lexical tie rule without Torch."""
    support = _support(support)
    require(support is not None, "No finite preference exists for empty support")
    mean = np.asarray(distribution["latent_mean"], dtype=float)
    scale = np.asarray(distribution["latent_scale"], dtype=float)
    require(mean.shape == scale.shape == (5,) and np.isfinite(mean).all()
        and np.isfinite(scale).all() and (scale > 0).all(), "Invalid independently recomputed frozen Gaussian")
    latent = np.asarray(support.latents, dtype=float)
    scores = -.5*np.square((latent-mean)/scale).sum(axis=1)-np.log(scale).sum()-2.5*math.log(2*math.pi)
    require(np.isfinite(scores).all(), "Nonfinite final categorical preference")
    candidates = [i for i, score in enumerate(scores) if score >= float(scores.max())-TIE_TOLERANCE]
    index = (support.parent_index if support.parent_index in candidates else
             min(candidates, key=lambda i:support.candidate_ids[i]))
    normalizer = float(scores.max()+np.log(np.exp(scores-scores.max()).sum()))
    return dict(index=index, candidate_id=support.candidate_ids[index],
        support_fingerprint=support.support_fingerprint, action=support.action(index),
        latent=list(support.latents[index]), score=float(scores[index]),
        log_probability=float(scores[index]-normalizer))


def _independent_plan(train_ledger, final_checkpoint_sha256, cells_by_split, supports,
                      prepared_records, parent_distributions, continued_distributions,
                      model_hashes, bindings):
    """Reconstruct the producer plan without calling its planner/selector."""
    require(type(train_ledger) is Ledger and train_ledger.phase == "train"
        and train_ledger.pending is None and train_ledger.stopped_reason is None
        and train_ledger.next_batch() is None, "Assessment preceded the completed TRAIN cap")
    by_id, contexts = _cells(cells_by_split)
    require(tuple(contexts["train"]) == train_ledger.train_context_ids
        and tuple(contexts["validation"]) == train_ledger.validation_context_ids
        and tuple(contexts["test"]) == train_ledger.test_context_ids,
        "Final assessment split/context order changed")
    require(set(supports) == set(parent_distributions) == set(continued_distributions) == set(by_id),
            "Frozen model/support readouts do not cover exactly the 140 fixed contexts")
    supports = {key:_support(value) for key,value in supports.items()}
    for key,value in supports.items():
        require(value is None or value.context_id == key, "Assessment support association changed")
        for distribution in (parent_distributions[key],continued_distributions[key]):
            mean, scale = np.asarray(distribution["latent_mean"],float), np.asarray(distribution["latent_scale"],float)
            require(mean.shape == scale.shape == (5,) and np.isfinite(mean).all()
                and np.isfinite(scale).all() and (scale > 0).all(), "Invalid frozen model forward readout")
    assessment = train_ledger.begin_assessment(final_checkpoint_sha256)
    requests = assessment._requests("assessment",0,420)
    choices, cohorts = [], []
    for cohort,(split,offset,method) in enumerate(ASSESSMENT_COHORTS):
        local = []
        for request in requests[cohort*35:(cohort+1)*35]:
            key, support = request.context_id, supports[request.context_id]
            parent = copy.deepcopy(parent_distributions[key]["exact_action"])
            require(all(type(parent.get(k)) in (int,float) and math.isfinite(parent[k]) for k in ACTION_NAMES),
                    "Original frozen direct-parent action changed")
            vector = np.asarray([parent[k] for k in ACTION_NAMES])
            require((vector >= LOW).all() and (vector <= HIGH).all(), "Direct-parent action outside original action domain")
            candidate_id,index,seed,scores = None,None,None,None
            enabled,decline,action = True,None,parent
            if method != "parent":
                if support is None:
                    enabled,decline = False,"no_qualified_candidate_support"
                else:
                    distribution = parent_distributions[key] if method == "candidate_bank" else continued_distributions[key]
                    chosen = independent_preference(distribution,support)
                    index = chosen["index"]
                    candidate_id, action = chosen["candidate_id"], chosen["action"]
                    mean, scale = np.asarray(distribution["latent_mean"],float), np.asarray(distribution["latent_scale"],float)
                    scores = (-.5*np.square((np.asarray(support.latents)-mean)/scale).sum(axis=1)
                              -np.log(scale).sum()-2.5*math.log(2*math.pi)).tolist()
                    record = prepared_records[candidate_id]
                    decision = record["decision"]
                    require(record["cell_id"] == key and record["candidate_id"] == candidate_id
                        and record["action"] == action and record["accepted"] is True
                        and decision["accepted"] is True and decision["release_authorized"] is False
                        and decision.get("action") == action and decision.get("binding",{}).get("context",{}).get("id") == key
                        and decision["fingerprint"] == geometry_hash({k:v for k,v in decision.items() if k != "fingerprint"})
                        and decision["qualification"].get("accepted") is True
                        and decision.get("path",{}).get("accepted") is True
                        and set(decision["qualification"]["ik"]) == {"Arm10","Arm14"}
                        and all(fit["solved"] is True and len(fit["q"]) == 6 and np.isfinite(fit["q"]).all()
                                for fit in decision["qualification"]["ik"].values()),
                        "Selected assessment action lacks bound prepared qualification")
                    seed = None if index == support.parent_index else copy.deepcopy(decision["qualification"]["ik"])
            choice = dict(request_id=request.request_id,request_index=request.request_index,
                context_id=key,split=split,method=method,cohort=cohort,action=action,
                candidate_id=candidate_id,index=index,support_fingerprint=None if support is None else support.support_fingerprint,
                latent=None if index is None else list(support.latents[index]),scores=scores,
                enabled=enabled,decline_reason=decline,ik_seeds=seed)
            choices.append(choice);local.append(choice)
        cohorts.append(dict(index=cohort,split=split,offset=offset,method=method,
            request_ids=[c["request_id"] for c in local],choice_sha256=canonical_hash(local)))
    result = dict(schema=PLAN_SCHEMA,train_ledger_fingerprint=train_ledger.to_dict()["fingerprint"],
        initial_assessment_ledger_fingerprint=assessment.to_dict()["fingerprint"],
        final_checkpoint_sha256=final_checkpoint_sha256,parent_checkpoint_sha256=PARENT_CHECKPOINT_SHA256,
        model_hashes=copy.deepcopy(model_hashes),bindings=copy.deepcopy(bindings),
        cells_by_split=copy.deepcopy(cells_by_split),choices=choices,cohorts=cohorts,
        supports_sha256=canonical_hash({k:None if v is None else v.to_dict() for k,v in supports.items()}),
        prepared_records_sha256=canonical_hash(prepared_records),
        parent_distributions_sha256=canonical_hash(parent_distributions),
        continued_distributions_sha256=canonical_hash(continued_distributions),
        requests=420,optimizer_updates=0,held_out_outcomes_observed=0,
        execution_authorized=False,release_authorized=False)
    result["fingerprint"] = canonical_hash(result)
    return result


def verify_assessment_packet(native_directory, packet, *, scene, cells, links, choices):
    """Original guards/scoring on strict in-memory accounting adaptations only."""
    require(packet.get("schema") == PACKET_SCHEMA and packet.get("mode") == "assessment"
        and packet.get("cells") == cells and packet.get("choices") == choices
        and type(packet.get("optimizer_updates")) is int and packet["optimizer_updates"] == 0,
        "Assessment packet changed schema, choices or update count")
    require(len(choices) == 35 and all(type(packet.get(k)) is list and len(packet[k]) == 35
        for k in ("rows","actions","enabled","decline_reasons","proposal_ik_seeds","rewards","reward_features")),
        "Assessment must retain all 35 paid cohort requests")
    require(packet["enabled"] == [c["enabled"] for c in choices]
        and all(type(v) is bool for v in packet["enabled"])
        and packet["actions"] == [c["action"] for c in choices]
        and packet["request_ids"] == [c["request_id"] for c in choices]
        and packet["decline_reasons"] == [c["decline_reason"] for c in choices]
        and packet["proposal_ik_seeds"] == [c["ik_seeds"] for c in choices],
        "Assessment action/slot/decline/seed mapping changed")
    original = copy.deepcopy(packet)
    guarded, scored = copy.deepcopy(packet), copy.deepcopy(packet)
    for i,(row,choice,cell) in enumerate(zip(packet["rows"],choices,cells)):
        require(row.get("split") == cell["split"] == choice["split"], "Assessment split was relabelled")
        if choice["enabled"]:
            require(row.get("native_invalid") == row.get("invalid"), "Native guard decline was changed for an enabled request")
        else:
            require(choice["method"] != "parent" and choice["decline_reason"] == row.get("invalid") == "no_qualified_candidate_support"
                and row.get("native_invalid") == "inactive_split"
                and row.get("qualification") == dict(accepted=False,reason="inactive_split")
                and packet["preflight_records"][i] is None and row.get("path_proxy") is None
                and row.get("path_rejection") is None and row.get("success") is False
                and row.get("route_success") is False and row.get("route_quality") == 0
                and packet["rewards"][i] == -1. and packet["reward_features"][i] is None,
                "Empty-support request was replaced, tried, scored or removed")
        guarded["rows"][i]["invalid"] = row["native_invalid"]
        # The legacy score wrapper requires TRAIN labels and an active paid
        # prefix. Every assessment slot is paid, including a parked empty bank.
        scored["cells"][i]["split"] = scored["rows"][i]["split"] = "train"
        scored["choices"][i]["split"] = "train"
    scored.update(schema=TRAIN_PACKET_SCHEMA,mode="train",enabled=[True]*35)
    guards = verify_guards(guarded,scene,cells,links,packet["proposal_ik_seeds"])
    scores = verify_score(native_directory,scored,scene)
    require(packet == original, "Assessment audit mutated the saved source packet")
    return dict(guards=guards, scores=scores)


def comparison_report(outcomes, cells_by_split, *, historical_capture_ids=None):
    """Describe the fixed paired grid, including declines in every denominator.

    The interval resamples positions, not training seeds. Historical TRAIN
    witnesses are separate from the newly observed paired controls and never
    rank or select the final artifact.
    """
    _, contexts = _cells(cells_by_split)
    require(type(outcomes) in (list, tuple) and len(outcomes) == 420,
            "Complete assessment reporting requires exactly 420 paid requests")
    rows = {}
    for row in outcomes:
        split, method, context = row["split"], row["method"], row["context_id"]
        key = split, method, context
        require(split in SPLITS and method in METHODS and context in contexts[split] and key not in rows,
                "Repeated, missing or foreign assessment result")
        require(all(type(row[k]) is bool for k in ("released", "captured", "clean"))
            and (not row["clean"] or row["captured"]) and (not row["captured"] or row["released"]),
            "Assessment capture flags contradict release/clean")
        require((row["released"] and row["decline_reason"] is None)
            or (not row["released"] and type(row["decline_reason"]) is str and row["decline_reason"]),
            "Assessment rejection reason changed")
        rows[key] = row
    expected = {(s,m,c) for s in SPLITS for m in METHODS for c in contexts[s]}
    require(set(rows) == expected, "Incomplete assessment grid; no final comparison")
    methods = {}
    for method in METHODS:
        methods[method] = {}
        for split in SPLITS:
            cohort = [rows[split,method,c] for c in contexts[split]]
            requested = SPLIT_COUNTS[split]
            released = sum(r["released"] for r in cohort)
            captures, clean = sum(r["captured"] for r in cohort), sum(r["clean"] for r in cohort)
            methods[method][split] = dict(requested=requested, released=released,
                rejected=requested-released, captures=captures, clean_captures=clean,
                capture_rate=captures/requested, clean_capture_rate=clean/requested,
                capture_given_release=captures/released if released else None,
                rejection_reasons=dict(sorted(Counter(r["decline_reason"] for r in cohort if not r["released"]).items())),
                per_position=[dict(context_id=c, **{k:rows[split,method,c][k] for k in
                    ("released","captured","clean","decline_reason")}) for c in contexts[split]])
    indices = np.random.default_rng(BOOTSTRAP_SEED).integers(0, 70, size=(BOOTSTRAP_REPLICATES,70))
    pairs = {}
    for control in ("candidate_bank", "parent"):
        pair = {}
        for field in ("captured", "clean"):
            first = np.asarray([rows["test","continued",c][field] for c in contexts["test"]], dtype=int)
            second = np.asarray([rows["test",control,c][field] for c in contexts["test"]], dtype=int)
            delta = first-second
            bootstrap = delta[indices].mean(axis=1)
            interval = np.quantile(bootstrap,[.025,.975]).tolist()
            pair[field] = dict(gains=int((delta == 1).sum()), losses=int((delta == -1).sum()),
                ties=int((delta == 0).sum()), difference_count=int(delta.sum()),
                difference_rate=float(delta.mean()), paired_bootstrap_95_percent_interval=interval,
                gain_context_ids=[c for c,d in zip(contexts["test"],delta) if d == 1],
                loss_context_ids=[c for c,d in zip(contexts["test"],delta) if d == -1])
        pairs["continued_minus_"+control] = pair
    retention = dict(historical_reference_capture_counts=dict(parent=13,candidate_bank=16),
        historical_ids_available=historical_capture_ids is not None,
        scope="Historical 13/35 and 16/35 TRAIN captures are prior references; fresh paired controls above determine this assessment.")
    if historical_capture_ids is not None:
        require(type(historical_capture_ids) is dict and set(historical_capture_ids) == {"parent","candidate_bank"},
                "Expected both historical TRAIN reference methods")
        retention["historical_witnesses"] = {}
        for method,count in (("parent",13),("candidate_bank",16)):
            names = historical_capture_ids[method]
            require(type(names) in (list,tuple) and len(names) == len(set(names)) == count
                and set(names).issubset(contexts["train"]), "Historical TRAIN witness identity/count changed")
            retention["historical_witnesses"][method] = dict(requested_reference=count,
                continued_retained=sum(rows["train","continued",c]["captured"] for c in names),
                continued_loss_context_ids=[c for c in names if not rows["train","continued",c]["captured"]],
                fresh_control_recaptured=sum(rows["train",method,c]["captured"] for c in names),
                fresh_control_loss_context_ids=[c for c in names if not rows["train",method,c]["captured"]])
    return dict(methods=methods, primary_comparison="continued_minus_candidate_bank",
        primary_metric="Final frozen TEST captures divided by 70 requested positions; rejections and empty supports are failures",
        test_pairs=pairs, train_retention=retention,
        bootstrap=dict(replicates=BOOTSTRAP_REPLICATES,seed=BOOTSTRAP_SEED,positions=70,
            scope="Position-wise paired bootstrap on one fixed shared TEST grid and one continuation; not independent training-seed uncertainty"),
        model_promoted=False, inference_searches=0, assessment_optimizer_updates=0,
        geometry_assisted_evaluation=True,
        scope="Final frozen direct predictions on shared geometry; no inference-time flight search, reruns, checkpoint selection or automatic promotion")


def audit_assessment(directory, *, expected_head_sha256, expected_bindings, train_ledger,
                     expected_native_binding, scene, cells_by_split, supports, prepared_records,
                     parent_distributions, continued_distributions, model_hashes, assets_paths,
                     poses_path, camera_path, parent_checkpoint_path, final_checkpoint_path,
                     startup_directory=None, expected_result_sha256=None, historical_capture_ids=None):
    """Independently certify the complete frozen 420-request assessment.

    Readouts and model tensor hashes must be freshly recomputed by the caller
    from the verified frozen checkpoints, with the same pinned CPU forward
    arithmetic used before outcomes. This audit independently rebuilds the
    selector math, schedule, native guards, physical score and terminal joins.
    It never takes a producer's claimed selected action as proof of inference.
    """
    directory = Path(directory)
    native, journal = directory/"native", directory/"journal"
    require(native.is_dir() and not native.is_symlink() and journal.is_dir() and not journal.is_symlink()
        and all(p.is_file() and not p.is_symlink() for p in journal.iterdir()),
        "Missing or linked assessment evidence")
    events = read_events(journal)
    _chain(events,expected_head_sha256)
    require(events[0]["kind"] == "start" and events[-1]["kind"] == "final"
        and type(events[0]["payload"].get("synthetic")) is bool,
        "Incomplete final assessment history; no promotion or retries")
    synthetic = events[0]["payload"]["synthetic"]
    require(synthetic or train_ledger.bindings.complete, "Real assessment requires complete frozen preparation/source bindings")
    result_path = _file(directory,"assessment-result.json")
    result_sha = _digest(result_path)
    require(expected_result_sha256 is None or expected_result_sha256 == result_sha,
            "Externally bound assessment result bytes changed")
    result = _json(result_path.read_bytes())
    require(result.get("schema") == RESULT_SCHEMA and result.get("status") == "assessment_complete_audit_pending"
        and result.get("synthetic") is synthetic and result.get("journal_head_sha256") == expected_head_sha256
        and result.get("fingerprint") == canonical_hash({k:v for k,v in result.items() if k != "fingerprint"})
        and result.get("bindings") == expected_bindings and result.get("requests") == 420
        and type(result.get("optimizer_updates")) is int and result["optimizer_updates"] == 0
        and result.get("execution_authorized") is result.get("release_authorized") is result.get("model_promoted") is False,
        "Assessment result identity, update count or authorization changed")
    final_sha = _digest(Path(final_checkpoint_path))
    parent_sha = _digest(Path(parent_checkpoint_path))
    require(final_sha == result.get("final_checkpoint_sha256"), "Final frozen checkpoint bytes changed")
    require(synthetic or parent_sha == PARENT_CHECKPOINT_SHA256,
            "Original frozen parent checkpoint bytes changed")
    require(type(model_hashes) is dict and set(model_hashes) == {"parent_weight_hash","continued_weight_hash"}
        and all(type(v) is str and len(v) == 64 and set(v).issubset("0123456789abcdef") for v in model_hashes.values()),
        "Invalid independently recomputed model tensor identities")
    require(synthetic or model_hashes["parent_weight_hash"] == "ea9a69ec063e3d71cda2b92f1da64939f644244d4b4dbd70b500be132632e549",
            "Real assessment parent tensor identity changed")
    b = expected_native_binding
    require(b.get("schema") == NATIVE_BINDING_SCHEMA
        and b.get("parent_checkpoint_sha256") == PARENT_CHECKPOINT_SHA256
        and b.get("fingerprint") == canonical_hash({k:v for k,v in b.items() if k != "fingerprint"})
        and b.get("source_sha256") == expected_bindings.get("source_sha256") == source_fingerprint()
        and b.get("guard_source_sha256") == guard_sources()
        and b.get("scene_sha256") == geometry_hash(scene) and b.get("setup_controller") == setup_protocol()
        and b.get("execution_authorized") is b.get("release_authorized") is False
        and expected_bindings.get("native_binding_sha256") == b["fingerprint"],
        "Assessment source/guard/scene/native binding changed")
    if not synthetic:
        require(all(b.get(k) == v for k,v in _frozen_identity().items()),
                "Original assessment geometry/assets/camera identity changed")
    require(set(assets_paths) == set(b["assets_sha256"]) == {"Arm10","Arm14"}, "Assessment arm assets changed")
    links = {name:chain(ET.fromstring(_input_bytes(path,b["assets_sha256"][name],name))) for name,path in assets_paths.items()}
    poses = _json(_input_bytes(poses_path,b["poses_sha256"],"poses"))
    _input_bytes(camera_path,b["camera_sha256"],"camera")
    require(b["canonical_home_rad"] == {"Arm10":poses["dropper_joint_rad"],"Arm14":poses["board_joint_rad"]},
            "Assessment canonical home changed")
    train_ids = set(train_ledger.train_context_ids)
    train_supports = [_support(supports[key]) for key in train_ledger.train_context_ids]
    train_records = {k:v for k,v in prepared_records.items() if v["cell_id"] in train_ids}
    require(all(s is not None for s in train_supports)
        and b["cells_sha256"] == canonical_hash(cells_by_split["train"])
        and b["support_sha256"] == canonical_hash([s.to_dict() for s in train_supports])
        and b["prepared_records_sha256"] == canonical_hash(train_records)
        and train_ledger.bindings.train_support_sha256 == b["support_sha256"],
        "Assessment no longer shares the frozen TRAIN bank/native binding")
    require(synthetic or all(expected_bindings.get(k) == v for k,v in asdict(train_ledger.bindings).items()
        if k not in ("parent_checkpoint_sha256","train_support_sha256")),
        "Real assessment preparation/source identities differ from completed TRAIN")
    controls = verify_controls(Path(startup_directory) if startup_directory is not None else directory.parent,
                               scene,b["startup_control_sha256"])
    expected_plan = _independent_plan(train_ledger,final_sha,cells_by_split,supports,prepared_records,
        parent_distributions,continued_distributions,model_hashes,expected_bindings)
    plan = _json(_file(directory,"assessment-plan.json").read_bytes())
    require(plan == expected_plan, "Assessment plan differs from independently recomputed frozen inference or fixed schedule")
    initial = train_ledger.begin_assessment(final_sha)
    start = events[0]["payload"]
    require(start == dict(plan_fingerprint=plan["fingerprint"],bindings=expected_bindings,synthetic=synthetic,
        ledger=initial.to_dict(),model_hashes=model_hashes,parent_checkpoint_sha256=parent_sha,
        final_checkpoint_sha256=final_sha), "Assessment did not freeze the exact final model before its first reservation")
    require(result.get("plan_fingerprint") == plan["fingerprint"] and result.get("model_hashes") == model_hashes,
            "Assessment result differs from frozen methods")
    require(synthetic or historical_capture_ids is not None, "Historical TRAIN witness IDs required for the retention audit")
    if historical_capture_ids is not None:
        require(expected_bindings.get("historical_capture_ids_sha256") == canonical_hash(historical_capture_ids),
                "Historical TRAIN witness reference binding changed")
    ledger, known, seen, batches, expected_files = initial, {}, set(), [], set()
    by_id, _ = _cells(cells_by_split)
    costs = dict(wall_s=0.,process_cpu_s=0.)
    for event in events[1:]:
        kind, payload = event["kind"], event["payload"]
        if kind == "reserve":
            batch = ledger.next_batch()
            require(batch is not None and len(batch.requests) == 35, "Assessment reservation after cap or incomplete paid cohort")
            ledger = ledger.reserve_batch(batch)
            cohort = batch.requests[0].request_index//35
            choices = expected_plan["choices"][cohort*35:(cohort+1)*35]
            require(payload == dict(ledger=ledger.to_dict(),choices=choices,
                plan_fingerprint=plan["fingerprint"],model_hashes=model_hashes),
                "Assessment reservation order/action/model changed")
            stem = f"assessment-{cohort:06d}"
            packet, packet_sha = _packet(native,stem)
            expected_files.update(stem+s for s in (".dispatch.json",".json.gz",".receipt.json"))
            claim = _json(_file(native,stem+".dispatch.json").read_bytes())
            require(claim == dict(reservation_sha256=event["fingerprint"],ledger_fingerprint=ledger.to_dict()["fingerprint"],
                request_ids=[c["request_id"] for c in choices],assessment_plan_fingerprint=plan["fingerprint"],
                native_binding_fingerprint=b["fingerprint"],status="claimed_before_native_call"),
                "Assessment dispatch differs from exact durable reservation")
            require(packet.get("reservation_sha256") == event["fingerprint"]
                and packet.get("assessment_plan_fingerprint") == plan["fingerprint"]
                and packet.get("native_binding") == b and packet.get("model_hashes") == model_hashes
                and packet.get("final_checkpoint_sha256") == final_sha and packet.get("synthetic") is synthetic
                and type(packet.get("batch")) is int and packet["batch"] == batch.batch_index
                and type(packet.get("cohort")) is int and packet["cohort"] == cohort
                and packet.get("split") == choices[0]["split"] and packet.get("method") == choices[0]["method"]
                and packet.get("setup_controller") == setup_protocol() and packet.get("startup_controls") == controls,
                "Assessment packet method/checkpoint/source/cohort changed")
            cells = [dict(copy.deepcopy(cells_by_split["train"][i]),**copy.deepcopy(by_id[c["context_id"]]))
                     for i,c in enumerate(choices)]
            physics = packet.get("physics")
            if physics is not None:
                require(physics.get("file") == stem+".npz"
                    and _digest(_file(native,physics["file"])) == physics["sha256"], "Assessment physics bytes or basename changed")
                expected_files.add(physics["file"])
            verified = verify_assessment_packet(native,packet,scene=scene,cells=cells,links=links,choices=choices)
            terminals = verified["scores"]["terminal_results"]
            require(len(terminals) == 35, "Assessment scorer omitted a paid request")
            for row,choice in zip(terminals,choices):
                require(row["request_id"] == choice["request_id"] and row["request_id"] not in known,
                        "Assessment request was replaced or replayed")
                known[row["request_id"]] = dict(row,source_sha256=packet_sha)
            for field in costs:
                value = packet[field]
                require(type(value) in (int,float) and math.isfinite(value) and value >= 0,
                        "Invalid assessment native process cost")
                costs[field] += value
            batches.append(dict(cohort=cohort,split=choices[0]["split"],method=choices[0]["method"],
                packet_sha256=packet_sha,guards=verified["guards"],
                scores={k:v for k,v in verified["scores"].items() if k != "terminal_results"}))
        elif kind == "outcomes":
            rows = payload["results"]
            require(type(rows) is list and bool(rows), "Empty assessment terminal prefix")
            outcomes = []
            for row in rows:
                require(type(row) is dict and set(row) == RESULT_FIELDS
                    and row["request_id"] in known and row["request_id"] not in seen
                    and row == known[row["request_id"]], "Assessment terminal labels/reward/source differ from independent physics audit")
                outcomes.append(Outcome(row["request_id"],row["released"],row["decline_reason"],row["source_sha256"]))
                seen.add(row["request_id"])
            ledger = ledger.record_outcomes(tuple(outcomes))
            require(payload["ledger"] == ledger.to_dict(), "Assessment terminal ledger changed")
        elif kind == "final":
            require(event is events[-1] and payload == dict(ledger=ledger.to_dict(),
                plan_fingerprint=plan["fingerprint"],model_hashes=model_hashes,final_checkpoint_sha256=final_sha),
                "Assessment final boundary changed")
        else:
            raise ValueError("Unexpected update/failure after final model freeze; no completed assessment")
    require(ledger.pending is None and ledger.summary()["assessment_complete"]
        and len(known) == len(seen) == 420 and set(known) == seen and len(batches) == 12,
        "Incomplete assessment budget/coverage; no final comparison")
    require(result["ledger"] == ledger.to_dict() and {p.name for p in native.iterdir()} == expected_files
        and all(p.is_file() and not p.is_symlink() for p in native.iterdir()), "Unlisted/repeated/partial assessment evidence")
    for field in costs:
        require(type(result.get(field)) in (int,float) and math.isfinite(result[field])
            and result[field] >= costs[field]-1e-6, "Assessment phase cost cannot cover its native batches")
    require(_digest(Path(final_checkpoint_path)) == final_sha and _digest(Path(parent_checkpoint_path)) == parent_sha,
            "Frozen checkpoint bytes changed during independent assessment audit")
    rows = [dict(known[c["request_id"]],context_id=c["context_id"],split=c["split"],method=c["method"])
            for c in expected_plan["choices"]]
    report = comparison_report(rows,cells_by_split,historical_capture_ids=historical_capture_ids)
    value = dict(schema=SCHEMA,status="passed",synthetic=synthetic,
        source_result_sha256=result_sha,journal_head_sha256=expected_head_sha256,
        plan_fingerprint=plan["fingerprint"],native_binding_fingerprint=b["fingerprint"],
        parent_checkpoint_sha256=parent_sha,final_checkpoint_sha256=final_sha,model_hashes=model_hashes,
        requests=420,releases=sum(r["released"] for r in rows),captures=sum(r["captured"] for r in rows),
        clean_captures=sum(r["clean"] for r in rows),rejections=sum(not r["released"] for r in rows),
        ledger=ledger.summary(),batches=batches,comparison=report,
        costs=dict(assessment_phase=dict(wall_s=result["wall_s"],process_cpu_s=result["process_cpu_s"]),
            native_batch_sum=costs,scope="Native batch timers are contained in assessment phase timers; do not add overlapping scopes. Startup, TRAIN, learning, audit and device memory/occupancy require separate receipts."),
        frozen_forward_readouts_externally_verified=True,assessment_optimizer_updates=0,
        inference_searches=0,model_promoted=False,execution_authorized=False,release_authorized=False,
        scope="Final frozen 420-request ledger, exact deterministic inference, original saved guard and physics/reward audit. Geometry-assisted fixed-grid evaluation; no optimizer arithmetic, mesh collision or physical validation certificate.")
    value["fingerprint"] = canonical_hash(value)
    return value
