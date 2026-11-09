"""Read-only final continuation publication, using only the standard library.

Call only after closure. Externally trusted application/audit/launch/manifest
hashes bind already-audited evidence; this publisher never loads a model, replays
physics, starts a process, writes a report or grants execution permission.
"""

from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import re

SCHEMA = "sircus-candidate-continuation-published-v1"
PARENT_SHA256 = "57d7658e9ed2f72ad09cf4e8523ab497ccbe0708364f1b8ec4a4c5c1aef1fe9f"
METHODS = ("parent", "candidate_bank", "continued")
SPLITS = {"train":35, "validation":35, "test":70}
PHASES = ("initialization", "imitation", "train", "assessment", "audit")
COHORTS = (("train",0,"parent"), ("train",0,"candidate_bank"), ("train",0,"continued"),
    ("validation",0,"candidate_bank"), ("validation",0,"continued"), ("validation",0,"parent"),
    ("test",0,"continued"), ("test",0,"parent"), ("test",0,"candidate_bank"),
    ("test",35,"parent"), ("test",35,"candidate_bank"), ("test",35,"continued"))


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sha(value, label):
    require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value), "Invalid SHA256 " + label)
    return value


def count(value, label, maximum=None):
    require(type(value) is int and value >= 0 and (maximum is None or value <= maximum), "Invalid count " + label)
    return value


def finite(value, label):
    require(type(value) in (int,float) and math.isfinite(value) and value >= 0, "Invalid measurement " + label)


def fingerprint(value, label):
    require(type(value) is dict and value.get("fingerprint") ==
            canonical_hash({k:v for k,v in value.items() if k != "fingerprint"}), "Changed fingerprint " + label)


def read_json(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "Missing or linked receipt: " + path.name)
    raw = path.read_bytes()
    require(len(raw) <= 32*1024**2, "Receipt exceeds bounded publication size")
    def pairs(items):
        value = {}
        for key, child in items:
            require(key not in value, "Duplicate receipt JSON key")
            value[key] = child
        return value
    value = json.loads(raw, object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite receipt JSON")))
    require(type(value) is dict, "Expected receipt object")
    return value, hashlib.sha256(raw).hexdigest()


def local_file(root, name):
    path = root/name
    require(path.resolve().is_relative_to(root) and not any(p.is_symlink() for p in
        (path,*path.parents) if p.is_relative_to(root)), "Linked or escaped publication input")
    return path


def _ledger(value, *, phase, counters, final_sha):
    """Independently recount the final fixed schedule, not a producer summary."""
    fingerprint(value, "ledger")
    require(value.get("schema") == "sircus-candidate-continuation-request-ledger-v2"
        and value.get("phase") == phase and value.get("stopped_reason") is None
        and value.get("execution_authorized") is False and value.get("release_authorized") is False
        and value.get("source_bindings_complete") is True, "Incomplete final ledger")
    ids = {s:value[s+"_context_ids"] for s in SPLITS}
    require(all(type(ids[s]) is list and len(ids[s]) == n and len(set(ids[s])) == n
        and all(type(x) is str and x for x in ids[s]) for s,n in SPLITS.items())
        and len(set(sum(ids.values(), []))) == 140, "Final ledger split identities changed")
    bindings = value["bindings"]
    require(type(bindings) is dict and bindings.get("parent_checkpoint_sha256") == PARENT_SHA256,
            "Ledger parent changed")
    for k,v in bindings.items(): sha(v,k)
    require(value["source_binding_sha256"] == canonical_hash(bindings), "Ledger source binding changed")
    protocol = sha(value["protocol_sha256"], "accounting protocol")
    rng, order = random.Random(20261125), []
    while len(order) < 1750:
        round_ids = list(ids["train"]); rng.shuffle(round_ids); order.extend(round_ids)
    order = order[:1750]
    schedule = canonical_hash(dict(protocol_sha256=protocol, train_context_ids=ids["train"],
        validation_context_ids=ids["validation"], test_context_ids=ids["test"],
        train_request_order_sha256=canonical_hash(order)))
    require(value["schedule_sha256"] == schedule, "Ledger schedule hash changed")
    totals = {s:[0,0] for s in ("train","assessment")}
    exposure = {"train":dict.fromkeys(ids["train"],0), "assessment":dict.fromkeys(sum(ids.values(),[]),0)}
    saw_assessment = False
    for index,batch in enumerate(value["batches"]):
        require(batch["batch_index"] == index and type(batch["batch_index"]) is int, "Ledger batch order changed")
        role = batch["phase"]
        require(role in totals and (role == "assessment" or not saw_assessment), "TRAIN resumed after assessment")
        if role == "assessment":
            saw_assessment = True
            require(phase == "assessment" and (totals["train"][0] == 1750 or totals["train"][1] == 1400),
                    "Assessment precedes TRAIN cap")
        requested,released = totals[role]
        rc,lc = (1750,1400) if role == "train" else (420,420)
        active = min(35,rc-requested,lc-released,140-requested%140 if role == "train" else 35)
        requests,outcomes = batch["requests"],batch["outcomes"]
        require(type(requests) is list and type(outcomes) is list and 0 < len(requests) == len(outcomes) == active,
                "Final ledger has unresolved, padded or misbudgeted requests")
        for offset,(request,outcome) in enumerate(zip(requests,outcomes)):
            position = requested+offset
            split,base,method = ("train",0,"continued") if role == "train" else COHORTS[position//35]
            context = order[position] if role == "train" else ids[split][base+position%35]
            identifier = f"{role}-{position:06d}"
            require(request == dict(request_id=identifier,request_index=position,phase=role,split=split,
                method=method,context_id=context,protocol_sha256=protocol,schedule_sha256=schedule,
                source_binding_sha256=value["source_binding_sha256"],
                final_checkpoint_sha256=None if role == "train" else final_sha), "Final request schedule/identity changed")
            require(outcome["request_id"] == identifier and type(outcome["released"]) is bool
                and (outcome["decline_reason"] is None if outcome["released"] else
                     type(outcome["decline_reason"]) is str and bool(outcome["decline_reason"])), "Final terminal outcome changed")
            sha(outcome["source_sha256"], "terminal source")
            exposure[role][context] += 1
        totals[role][0] += active; totals[role][1] += sum(o["released"] for o in outcomes)
    require(totals["train"] == [counters["train_requests"],counters["train_releases"]]
        and totals["assessment"] == ([0,0] if phase == "train" else
            [counters["assessment_requests"],counters["assessment_releases"]]), "Final ledger counts differ")
    require(value["final_checkpoint_sha256"] == (None if phase == "train" else final_sha), "Ledger final model changed")
    summary = {}
    for role,(requested,released) in totals.items():
        rc,lc = (1750,1400) if role == "train" else (420,420)
        summary[role] = dict(request_limit=rc,release_limit=lc,requested=requested,terminal=requested,unresolved=0,
            released=released,declined=requested-released,remaining_requests=rc-requested,remaining_releases=lc-released,
            request_limit_reached=requested == rc,release_limit_reached=released == lc,
            request_cap_release_target_shortfall=lc-released if requested == rc else None,context_exposure=exposure[role])
    summary.update(total_new_task_requests=sum(x[0] for x in totals.values()),
        total_new_task_releases=sum(x[1] for x in totals.values()),assessment_complete=phase == "assessment")
    require(value["counters"] == summary, "Derived ledger counters changed")
    return ids, summary


def _comparison(value, ids):
    require(value.get("primary_comparison") == "continued_minus_candidate_bank"
        and value.get("model_promoted") is False and value.get("geometry_assisted_evaluation") is True
        and value.get("inference_searches") == value.get("assessment_optimizer_updates") == 0
        and set(value["methods"]) == set(METHODS), "Final comparison contract changed")
    positions, releases = {}, 0
    for method in METHODS:
        require(set(value["methods"][method]) == set(SPLITS), "Comparison split coverage changed")
        for split,n in SPLITS.items():
            row = value["methods"][method][split]
            require(row["requested"] == n and type(row["requested"]) is int, "Comparison denominator changed")
            for key in ("released","rejected","captures","clean_captures"): count(row[key],key,n)
            require(row["released"]+row["rejected"] == n and row["clean_captures"] <= row["captures"] <= row["released"],
                    "Comparison counts are inconsistent")
            actual = row["per_position"]
            require(type(actual) is list and [p["context_id"] for p in actual] == ids[split], "Comparison position order changed")
            for p in actual:
                require(all(type(p[k]) is bool for k in ("released","captured","clean"))
                    and (not p["clean"] or p["captured"]) and (not p["captured"] or p["released"])
                    and (p["decline_reason"] is None if p["released"] else
                         type(p["decline_reason"]) is str and bool(p["decline_reason"])), "Comparison position flags changed")
            require(sum(p["released"] for p in actual) == row["released"]
                and sum(p["captured"] for p in actual) == row["captures"]
                and sum(p["clean"] for p in actual) == row["clean_captures"]
                and dict(Counter(p["decline_reason"] for p in actual if not p["released"])) == row["rejection_reasons"],
                "Comparison aggregate differs from audited positions")
            require(row["capture_rate"] == row["captures"]/n and row["clean_capture_rate"] == row["clean_captures"]/n
                and row["capture_given_release"] == (row["captures"]/row["released"] if row["released"] else None),
                "Comparison rate or conditional denominator changed")
            positions[method,split] = {p["context_id"]:p for p in actual}; releases += row["released"]
    require(set(value["test_pairs"]) == {"continued_minus_parent","continued_minus_candidate_bank"}, "Paired controls changed")
    for control in ("parent","candidate_bank"):
        for field in ("captured","clean"):
            pair = value["test_pairs"]["continued_minus_"+control][field]
            gains = [c for c in ids["test"] if positions["continued","test"][c][field] and not positions[control,"test"][c][field]]
            losses = [c for c in ids["test"] if positions[control,"test"][c][field] and not positions["continued","test"][c][field]]
            require(pair["gains"] == len(gains) and pair["losses"] == len(losses)
                and pair["ties"] == 70-len(gains)-len(losses) and pair["difference_count"] == len(gains)-len(losses)
                and pair["difference_rate"] == (len(gains)-len(losses))/70
                and pair["gain_context_ids"] == gains and pair["loss_context_ids"] == losses, "Paired outcomes changed")
            interval = pair["paired_bootstrap_95_percent_interval"]
            require(type(interval) is list and len(interval) == 2 and all(type(x) in (int,float)
                and math.isfinite(x) and -1 <= x <= 1 for x in interval) and interval[0] <= interval[1], "Invalid paired interval")
    require(value["bootstrap"]["replicates"] == 10000 and value["bootstrap"]["seed"] == 20261128
        and value["bootstrap"]["positions"] == 70, "Paired descriptive bootstrap protocol changed")
    retention = value["train_retention"]
    require(retention.get("historical_reference_capture_counts") == dict(parent=13,candidate_bank=16)
        and retention.get("historical_ids_available") is True, "Historical TRAIN reference changed")
    for method,n in (("parent",13),("candidate_bank",16)):
        witness = retention["historical_witnesses"][method]
        require(witness["requested_reference"] == n, "Historical witness denominator changed")
        for prefix,who in (("continued","continued"),("fresh_control",method)):
            losses = witness[prefix+"_loss_context_ids"]
            retained = witness["continued_retained" if prefix == "continued" else "fresh_control_recaptured"]
            count(retained,"historical retention",n)
            require(type(losses) is list and len(losses) == len(set(losses)) == n-retained
                and all(c in positions[who,"train"] and not positions[who,"train"][c]["captured"] for c in losses),
                "Historical retention outcome changed")
    return releases


def _monitoring(root, learned, native):
    directory = local_file(root,"learning/journal")
    require(directory.is_dir(), "Missing completed TRAIN journal")
    files = sorted(directory.iterdir()); previous = "0"*64
    totals = dict(terminal=0,captures=0,clean_captures=0,released=0,rejected=0)
    curve, prefix, final = [], None, None
    for index,path in enumerate(files):
        require(path.name == f"event-{index:06d}.json", "Unexpected or partial TRAIN journal")
        event,_ = read_json(local_file(root,"learning/journal/"+path.name)); fingerprint(event,"TRAIN event")
        require(event.get("schema") == "sircus-candidate-continuation-journal-v1" and event.get("seq") == index
            and event.get("previous_sha256") == previous, "TRAIN journal chain changed")
        previous = event["fingerprint"]; final = event["kind"]
        if event["kind"] != "outcomes": continue
        rows = event["payload"]["results"]
        require(type(rows) is list and rows, "Empty TRAIN terminal prefix")
        prefix = dict(terminal=len(rows),captures=sum(r["captured"] for r in rows),
            clean_captures=sum(r["clean"] for r in rows),released=sum(r["released"] for r in rows),
            rejected=sum(not r["released"] for r in rows))
        for key in totals: totals[key] += prefix[key]
        curve.append(dict(totals))
    require(final == "final" and previous == learned["journal_head_sha256"] == native["journal_head_sha256"]
        and totals["terminal"] == native["requests"] and totals["captures"] == native["captures"]
        and totals["clean_captures"] == native["clean_captures"] and totals["released"] == native["releases"]
        and totals["rejected"] == native["rejections"], "Complete TRAIN curve differs from audited evidence")
    totals.update(requested=native["requests"],unresolved=0)
    return dict(audited=False,totals=totals,curve=curve,latest_terminal_prefix=prefix,
        source_journal_head_sha256=previous,
        scope="Exploratory TRAIN outcomes only, including every rejection; not frozen held-out performance. Complete counts are joined to the final audited TRAIN journal.")


def _build_child(root, *, manifest_path, expected_application_sha256, expected_audit_sha256,
                expected_launch_sha256, expected_manifest_sha256, expected_session, expected_source_sha256):
    """Build a compact final child from a closed run; no file is written."""
    require(re.fullmatch(r"[0-9a-f]{32}",expected_session or "") is not None, "Invalid expected session")
    sha(expected_source_sha256,"expected producer source")
    root = Path(root)
    require(root.is_dir() and not root.is_symlink(), "Expected closed nonsymlink run directory")
    root = root.resolve()
    values, hashes = {}, {}
    for name, expected in (("application-result.json",expected_application_sha256),
        ("combined-audit.json",expected_audit_sha256),("launch-result.json",expected_launch_sha256)):
        sha(expected,name); values[name],hashes[name] = read_json(local_file(root,name))
        require(hashes[name] == expected, "Externally bound "+name+" changed")
    manifest,manifest_sha = read_json(manifest_path); sha(expected_manifest_sha256,"manifest")
    require(manifest_sha == expected_manifest_sha256 and manifest.get("schema") == "sircus-candidate-continuation-production-inputs-v1"
        and manifest.get("source_sha256") == expected_source_sha256
        and manifest.get("execution_authorized") is False and manifest.get("release_authorized") is False,
        "Pinned production manifest changed")
    app,audit,launch = (values[n] for n in ("application-result.json","combined-audit.json","launch-result.json"))
    fingerprint(app,"application"); fingerprint(audit,"combined audit")
    require(app.get("schema") == "sircus-candidate-continuation-application-v1" and app.get("status") == "completed_audited"
        and app.get("synthetic") is False and app.get("audit") == audit
        and audit.get("schema") == "sircus-candidate-continuation-combined-audit-v1" and audit.get("status") == "passed"
        and audit.get("all_components_passed") is True and audit.get("synthetic") is False
        and audit.get("source_sha256") == expected_source_sha256, "Final independent audit has not passed")
    require(launch.get("schema") == "sircus-candidate-continuation-bounded-launch-v2"
        and launch.get("status") == "native_complete_independent_audit_required" and launch.get("mock_only") is False
        and launch.get("native_application_starts") == app.get("native_application_starts") == 1
        and all(x.get("session") == expected_session for x in (app,audit,launch)), "Production run is not closed in the expected session")
    for value in (app,audit,launch): require(value.get("model_promoted") is False, "Unexpected model promotion")
    for value in (app,audit):
        require(value.get("execution_authorized") is False and value.get("release_authorized") is False, "Publication cannot grant execution")
    for name in ("learning","native_train","assessment"):
        component = audit[name]; fingerprint(component,name+" audit")
        expected_schema = {"learning":"sircus-candidate-continuation-learning-audit-v1",
            "native_train":"sircus-candidate-continuation-native-audit-v1",
            "assessment":"sircus-candidate-continuation-assessment-audit-v1"}[name]
        require(component.get("schema") == expected_schema and component.get("status") == "passed"
            and component.get("synthetic") is False, "Component audit incomplete or synthetic")
    startup = audit["startup"]
    require(startup.get("status") == "passed" and startup.get("synthetic",False) is False, "Startup audit failed")
    for name in ("audit-request.json","learning/result.json","assessment/assessment-result.json","native-binding.json","native-setup/startup/receipt.json"):
        values[name],hashes[name] = read_json(local_file(root,name))
    request,learned,assessed,binding = (values[n] for n in
        ("audit-request.json","learning/result.json","assessment/assessment-result.json","native-binding.json"))
    for value,label in ((learned,"learning result"),(assessed,"assessment result"),(binding,"native binding")): fingerprint(value,label)
    require(request.get("session") == expected_session and request.get("source_sha256") == expected_source_sha256
        and audit["audit_request_sha256"] == hashes["audit-request.json"], "Audit request/session/source changed")
    for name,key in (("learning/result.json","learning_result_sha256"), ("assessment/assessment-result.json","assessment_result_sha256"),
        ("native-binding.json","native_binding_sha256"),("native-setup/startup/receipt.json","startup_receipt_sha256")):
        require(request[key] == hashes[name], "Final audit input bytes changed: "+name)
    require(learned.get("status") == "learning_complete_audit_pending" and assessed.get("status") == "assessment_complete_audit_pending"
        and learned.get("schema") == "sircus-candidate-continuation-run-v1"
        and assessed.get("schema") == "sircus-candidate-continuation-assessment-result-v1"
        and learned.get("synthetic") is assessed.get("synthetic") is False
        and audit["learning"]["result_sha256"] == hashes["learning/result.json"]
        and audit["assessment"]["source_result_sha256"] == hashes["assessment/assessment-result.json"] == app["assessment_result_sha256"]
        and startup["receipt_sha256"] == hashes["native-setup/startup/receipt.json"]
        and startup["startup_control_sha256"] == binding["startup_control_sha256"]
        and audit["native_train"]["startup_controls"] == binding["startup_control_sha256"]
        and audit["assessment"]["journal_head_sha256"] == assessed["journal_head_sha256"]
        and binding["source_sha256"] == expected_source_sha256, "Audited result/startup joins changed")
    final_sha = sha(app["final_checkpoint_sha256"],"final checkpoint")
    model_path = local_file(root,"learning/final.pt")
    require(model_path.is_file() and hashlib.sha256(model_path.read_bytes()).hexdigest() == final_sha
        and final_sha == audit["final_checkpoint_sha256"] == learned["checkpoint_sha256"]
            == assessed["final_checkpoint_sha256"] == audit["assessment"]["final_checkpoint_sha256"]
            == audit["learning"]["checkpoint_sha256"], "Final checkpoint bytes/identity changed")
    require(audit["learning"].get("checkpoint_bytes_verified") is True
        and audit["learning"].get("checkpoint_ledger_journal_bound") is True
        and audit["native_train"].get("raw_assets_poses_camera_verified") is True
        and audit["native_train"].get("parent_checkpoint_bytes_verified") is True
        and audit["native_train"]["native_binding_fingerprint"] == binding["fingerprint"]
            == audit["assessment"]["native_binding_fingerprint"] == assessed["bindings"]["native_binding_sha256"],
        "Independent checkpoint/native proof or binding is incomplete")
    require(learned["bindings"] == audit["learning"]["bindings"]
        and learned["bindings"]["source_sha256"] == expected_source_sha256
        and learned["bindings"]["parent_checkpoint_sha256"] == manifest["files"]["parent_checkpoint"]["sha256"] == PARENT_SHA256
        and learned["bindings"]["input_sha256"] == manifest["files"]["train_inputs"]["sha256"]
        and learned["bindings"]["train_bank_sha256"] == manifest["files"]["train_bank"]["sha256"]
        and learned["bindings"]["fresh_split_manifest_sha256"] == manifest["files"]["fresh_split_manifest"]["sha256"]
        and assessed["model_hashes"] == audit["assessment"]["model_hashes"]
        and all(learned["bindings"].get(k) == v == assessed["bindings"].get(k)
                for k,v in learned["ledger"]["bindings"].items())
        and assessed["ledger"]["bindings"] == learned["ledger"]["bindings"]
        and assessed["bindings"]["source_sha256"] == expected_source_sha256, "Pinned input/model bindings changed")
    counters = app["counters"]
    require(set(counters) == {"train_requests","train_releases","assessment_requests","assessment_releases"}, "Final counter fields changed")
    for key,limit in (("train_requests",1750),("train_releases",1400),("assessment_requests",420),("assessment_releases",420)):
        count(counters[key],key,limit)
    require(counters == launch["counters"] and counters["train_releases"] <= counters["train_requests"]
        and (counters["train_requests"] == 1750 or counters["train_releases"] == 1400)
        and counters["assessment_requests"] == 420, "Incomplete or misbudgeted final run")
    ids,train_summary = _ledger(learned["ledger"],phase="train",counters=counters,final_sha=final_sha)
    assessment_ids,assessment_summary = _ledger(assessed["ledger"],phase="assessment",counters=counters,final_sha=final_sha)
    require(ids == assessment_ids and audit["native_train"]["ledger"] == train_summary
        and audit["assessment"]["ledger"] == assessment_summary
        and audit["native_train"]["requests"] == counters["train_requests"]
        and audit["native_train"]["releases"] == counters["train_releases"]
        and audit["assessment"]["requests"] == 420 and audit["assessment"]["releases"] == counters["assessment_releases"], "Audited budget/ledger join changed")
    learning = learned["counters"]
    require(learning == audit["learning"]["counters"] == dict(imitation_steps=240,imitation_presentations=30680,
        ppo_samples=counters["train_requests"],ppo_updates=math.ceil(counters["train_requests"]/140),
        ppo_steps=4*math.ceil(counters["train_requests"]/140)), "Bounded learning counters changed")
    comparison = audit["assessment"]["comparison"]
    require(_comparison(comparison,ids) == counters["assessment_releases"], "Comparison release sum changed")
    resources = app["resources"]
    require(type(resources) is dict and set(resources["phases"]) == set(PHASES), "Missing phase-separated resources")
    for phase,row in resources["phases"].items():
        for key in ("wall_s","aggregate_cpu_s","lifetime_peak_rss_mib","largest_child_lifetime_peak_rss_mib"): finite(row[key],phase+" "+key)
        if row["device"] is not None:
            require(type(row["device"]) is dict, "Invalid device measurement")
            for key,value in row["device"].items(): finite(value,phase+" "+key)
    require(audit["audit_cost"].get("device") == "cpu" and audit["audit_cost"].get("cuda_visible_devices") == "",
            "Independent audit was not CPU-only")
    for key in ("wall_s","process_cpu_s","lifetime_peak_rss_mib"): finite(audit["audit_cost"][key],"audit "+key)
    diagnostics = launch["diagnostics"]
    require(diagnostics.get("complete") is True and diagnostics.get("task_requests") == 0
        and 0 < count(diagnostics["cohorts"],"diagnostic cohorts",14)
        and diagnostics["initialized_trajectories"] == diagnostics["cohorts"]*35 <= 490
        and count(diagnostics["physics_steps"],"diagnostic steps",1824)
        and all(startup[k] == diagnostics[k] for k in ("cohorts","initialized_trajectories","physics_steps")),
        "Startup diagnostics differ from closed launch")
    monitor = _monitoring(root,learned,audit["native_train"])
    child = dict(schema=SCHEMA,status="completed_audited",session=expected_session,phase="audit",model_promoted=False,
        counters=copy.deepcopy(counters),learning=dict(learning,optimizer_steps=learning["imitation_steps"]+learning["ppo_steps"]),
        comparison=copy.deepcopy(comparison),resources=copy.deepcopy(resources),train_monitoring=monitor,
        final_checkpoint_sha256=final_sha,audit=dict(status="passed",all_components_passed=True,synthetic=False,
            sha256=expected_audit_sha256,component_fingerprints={k:audit[k].get("fingerprint") for k in ("learning","native_train","assessment")}),
        evidence=dict(application_result_sha256=expected_application_sha256,combined_audit_sha256=expected_audit_sha256,
            launch_result_sha256=expected_launch_sha256,manifest_sha256=expected_manifest_sha256,
            source_sha256=expected_source_sha256,**{k.replace("/","_").replace(".json", "")+"_sha256":v for k,v in hashes.items()
                if k not in ("application-result.json","combined-audit.json","launch-result.json")}),
        experience=dict(inherited_parent_requests=10000,inherited_parent_updates=73,prior_frozen_selector_comparison_requests=70),
        resource_verification="Producer-measured phase costs are byte-bound to the closed application result; this publisher does not independently remeasure time or memory. Audit-child cost overlaps the parent audit phase.",
        audit_reader_cost=copy.deepcopy(audit["audit_cost"]),new_trial_videos=False,
        startup={k:diagnostics[k] for k in ("cohorts","initialized_trajectories","physics_steps")},
        scope="Final frozen, geometry-assisted shared-grid assessment of one extra-budget continuation. No automatic promotion, inference-time flight search, independent-seed uncertainty or physical validation.",
        execution_authorized=False,release_authorized=False)
    child["fingerprint"] = canonical_hash(child)
    return child


def build_child(root, *, manifest_path, expected_application_sha256, expected_audit_sha256,
                expected_launch_sha256, expected_manifest_sha256, expected_session, expected_source_sha256):
    """Read-only publication; malformed or missing evidence fails closed."""
    try:
        return _build_child(root,manifest_path=manifest_path,expected_application_sha256=expected_application_sha256,
            expected_audit_sha256=expected_audit_sha256,expected_launch_sha256=expected_launch_sha256,
            expected_manifest_sha256=expected_manifest_sha256,expected_session=expected_session,
            expected_source_sha256=expected_source_sha256)
    except (KeyError,TypeError,IndexError) as error:
        raise ValueError("Incomplete or malformed final publication evidence") from error


def add_continuation(historical_report, child):
    """Pure additive update: preserve every historical root field and prior attempt."""
    require(type(historical_report) is dict and historical_report.get("schema") == "sircus-candidate-native-comparison-report-v1",
            "Expected historical frozen-comparison root")
    fingerprint(child,"published continuation")
    require(child.get("schema") == SCHEMA and child.get("status") == "completed_audited", "Expected verified final continuation child")
    result, current = copy.deepcopy(historical_report), historical_report.get("continuation")
    replacement = copy.deepcopy(child)
    closed = ("failed","incomplete_no_retry","completed_audited")
    require(type(current) is not dict or current.get("session") == child["session"]
        or current.get("status") in closed, "A different-session active continuation must be preserved")
    prior = copy.deepcopy(current.get("prior_attempts",[])) if type(current) is dict else []
    require(type(prior) is list, "Malformed prior-attempt history")
    if type(current) is dict and current.get("status") in closed \
            and current.get("session") != child["session"]:
        displaced = {k:copy.deepcopy(v) for k,v in current.items() if k != "prior_attempts"}
        if canonical_hash(displaced) not in {canonical_hash(p) for p in prior}: prior.append(displaced)
    if prior: replacement["prior_attempts"] = prior
    replacement.pop("fingerprint",None); replacement["fingerprint"] = canonical_hash(replacement)
    result["continuation"] = replacement
    return result
