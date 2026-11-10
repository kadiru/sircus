"""Independent saved-evidence reader for the separate 210-request V4 TEST.

No app, simulator, IK search, training or new shot is constructed. Real success
requires independent frozen model reconstruction, startup checks and every
original guard/scorer replay. Fabricated dependencies are accepted only with
explicit synthetic provenance and can never produce a real passed receipt.
"""

from collections import Counter
import copy
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import xml.etree.ElementTree as ET

from candidate_v4_evaluation import validate_prepared
from candidate_v4_evaluation_audit import _plan, _journal
from candidate_v4_model_check import canonical_hash,validate_model_receipt,RUNTIME

SCHEMA="sircus-v4-test-independent-native-audit-v1"
RESULT_SCHEMA="sircus-v4-test-evaluation-result-v1"
ENVELOPE_SCHEMA="sircus-v4-test-native-envelope-v1"
METHODS=("parent","candidate_bank","continued")
FIELDS={"request_id","released","decline_reason","captured","clean","source_sha256","return"}


def require(value,message):
    if not value:raise ValueError(message)


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(raw):
    def pairs(items):
        result={}
        for key,value in items:
            require(key not in result,"Duplicate independent-audit JSON key");result[key]=value
        return result
    return json.loads(raw,object_pairs_hook=pairs,
        parse_constant=lambda _:(_ for _ in ()).throw(ValueError("Nonfinite independent-audit JSON")))


def _file(directory,name):
    require(type(name) is str and Path(name).name==name,"Native evidence filename escaped its directory")
    path=directory/name
    require(path.is_file() and not path.is_symlink() and path.resolve().parent==directory.resolve(),"Missing or linked native evidence: "+name)
    return path


def _receipt(path,expected):
    path=Path(path)
    require(path.is_file() and not path.is_symlink() and digest(path)==expected,"Externally bound receipt bytes changed")
    return _json(path.read_bytes())


def _packet(native,stem):
    target=_file(native,stem+".packet.json.gz")
    receipt=_json(_file(native,stem+".receipt.json").read_bytes())
    require(set(receipt)=={"file","sha256","bytes","uncompressed_sha256"}
        and receipt["file"]==target.name and type(receipt["bytes"]) is int
        and receipt["bytes"]==target.stat().st_size and receipt["sha256"]==digest(target),"Native compressed packet receipt changed")
    from candidate_continuation_native_audit import MAX_PACKET_BYTES
    with gzip.open(target,"rb") as stream:raw=stream.read(MAX_PACKET_BYTES+1)
    require(len(raw)<=MAX_PACKET_BYTES and hashlib.sha256(raw).hexdigest()==receipt["uncompressed_sha256"],"Native uncompressed packet bytes changed")
    return _json(raw),receipt["sha256"]


def _binding(binding,plan,model,scene,cells,assets_paths,poses_path,camera_path,synthetic):
    require(type(binding) is dict and binding.get("schema")=="sircus-candidate-continuation-native-binding-v1"
        and binding.get("fingerprint")==canonical_hash({k:v for k,v in binding.items() if k!="fingerprint"})
        and binding.get("source_sha256")==plan["bindings"]["evaluation_source_sha256"]
        and binding.get("guard_source_sha256")==model["original_guard_source_sha256"]
        and binding.get("scene_sha256")==canonical_hash(scene)
        and binding.get("cells_sha256")==canonical_hash(cells)
        and binding.get("execution_authorized") is binding.get("release_authorized") is False,
        "Native TEST source, guard, scene or model binding changed")
    old=model["original_native_binding"]
    mutable={"source_sha256","startup_control_sha256","fingerprint"}
    require({k:v for k,v in binding.items() if k not in mutable}=={k:v for k,v in old.items() if k not in mutable},
        "Evaluation changed original bank, geometry, assets, home, guards or scorer")
    if synthetic:return {"Arm10":None,"Arm14":None}
    from candidate_continuation_native import source_fingerprint,guard_sources,_frozen_identity
    from prepare_assets import chain
    require(source_fingerprint()==binding["source_sha256"] and guard_sources()==binding["guard_source_sha256"]
        and all(binding.get(k)==v for k,v in _frozen_identity().items()),"Executed guard or frozen scene identity changed")
    require(type(assets_paths) is dict and set(assets_paths)=={"Arm10","Arm14"},"Original arm assets required")
    links={}
    for name,path in assets_paths.items():
        p=Path(path);require(p.is_file() and not p.is_symlink() and digest(p)==binding["assets_sha256"][name],"Native arm asset bytes changed")
        links[name]=chain(ET.fromstring(p.read_bytes()))
    for path,key in ((poses_path,"poses_sha256"),(camera_path,"camera_sha256")):
        p=Path(path);require(p.is_file() and not p.is_symlink() and digest(p)==binding[key],"Original pose/camera bytes changed")
    poses=_json(Path(poses_path).read_bytes())
    require(binding["canonical_home_rad"]==dict(Arm10=poses["dropper_joint_rad"],Arm14=poses["board_joint_rad"]),"Canonical home differs from original poses")
    return links


def _outcome(row,choice):
    require(type(row) is dict and set(row)==FIELDS and row["request_id"]==choice["request_id"]
        and all(type(row[k]) is bool for k in ("released","captured","clean"))
        and (not row["clean"] or row["captured"]) and (not row["captured"] or row["released"]),"Invalid native TEST terminal identity or flags")
    require((row["released"] and row["decline_reason"] is None) or (not row["released"] and type(row["decline_reason"]) is str and row["decline_reason"]),
        "Native TEST rejection reason changed")
    value=row["return"]
    require(type(value) in (int,float) and math.isfinite(value)
        and ((not row["released"] and value==-1.) or (row["captured"] and value==(1.6 if row["clean"] else 1.5))
            or (row["released"] and not row["captured"] and -.8<=value<=0.)),"Native TEST reward changed")
    require(choice["enabled"] or (not row["released"] and row["decline_reason"]=="no_qualified_candidate_support"),
        "An empty TEST support was released or replaced")


def _comparison(plan,rows,synthetic):
    by_pair={(choice["method"],choice["context_id"]):row for choice,row in zip(plan["choices"],rows)}
    contexts=[cell["id"] for cell in plan["original_plan"]["cells_by_split"]["test"]]
    methods={}
    for method in METHODS:
        local=[by_pair[method,key] for key in contexts]
        methods[method]=dict(requested=70,released=sum(r["released"] for r in local),rejected=sum(not r["released"] for r in local),
            captures=sum(r["captured"] for r in local),clean_captures=sum(r["clean"] for r in local),
            rejection_reasons=dict(sorted(Counter(r["decline_reason"] for r in local if not r["released"]).items())))
    pairs={}
    for control in ("candidate_bank","parent"):
        pair={}
        for field in ("captured","clean"):
            gains=[key for key in contexts if by_pair["continued",key][field] and not by_pair[control,key][field]]
            losses=[key for key in contexts if by_pair[control,key][field] and not by_pair["continued",key][field]]
            pair[field]=dict(gains=len(gains),losses=len(losses),ties=70-len(gains)-len(losses),
                difference_count=len(gains)-len(losses),difference_rate=(len(gains)-len(losses))/70.,gain_context_ids=gains,loss_context_ids=losses)
        pairs["continued_minus_"+control]=pair
    return dict(methods=methods,test_pairs=pairs,synthetic=synthetic,primary_comparison="continued_minus_candidate_bank",
        primary_metric="Captures divided by all 70 requested TEST positions, including declines",
        scope="Separate paired TEST-only result of one saved continuation. Not equal-budget multi-seed uncertainty or completion of V4's old assessment.")


def audit_native_evaluation(directory,*,plan,summary,expected_head_sha256,expected_result_sha256,
        model_receipt_path,expected_model_receipt_sha256,native_binding,scene,original_cells,
        assets_paths=None,poses_path=None,camera_path=None,parent_checkpoint_path=None,final_checkpoint_path=None,
        startup_directory,startup_cells,synthetic=False,checks=None):
    """Read only closed new evidence; injected dependencies are synthetic-only."""
    plan=validate_prepared(plan);_plan(plan)
    require(type(synthetic) is bool and plan["synthetic"] is synthetic,"Independent native provenance differs from plan")
    require(checks is None or synthetic,"Synthetic dependencies cannot certify real native results")
    require(synthetic or os.environ.get("CUDA_VISIBLE_DEVICES")=="","Independent native audit must hide CUDA")
    if not synthetic:
        from candidate_continuation_production import verify_dependencies
        verify_dependencies(RUNTIME)
    if checks is None:
        from candidate_continuation_assessment_audit import verify_assessment_packet
        from candidate_continuation_scene import audit_startup
        from candidate_native_audit import verify_controls
        checks=dict(packet=verify_assessment_packet,startup=audit_startup,controls=verify_controls)
    require(type(checks) is dict and set(checks)=={"packet","startup","controls"} and all(callable(f) for f in checks.values()),"Missing independent native checks")
    directory=Path(directory)
    require(directory.is_absolute() and directory.resolve()==directory and directory.is_dir()
        and not any(p.is_symlink() for p in (directory,*directory.parents)),"Linked or missing independent audit root")
    require(not any((directory/name).exists() for name in ("application-failure.json","evaluation-failure.json")),"Failed or interrupted TEST evidence cannot pass")
    stored=_receipt(directory/"evaluation-result.json",expected_result_sha256)
    require(stored==summary and summary.get("schema")==RESULT_SCHEMA
        and summary.get("status")==("synthetic_complete_audit_pending" if synthetic else "native_complete_audit_pending")
        and summary.get("fingerprint")==canonical_hash({k:v for k,v in summary.items() if k!="fingerprint"}),"Closed native TEST result changed")
    require(_json(_file(directory,"evaluation-plan.json").read_bytes())==plan,"Stored TEST plan changed")
    model=validate_model_receipt(_receipt(model_receipt_path,expected_model_receipt_sha256),plan)
    links=_binding(native_binding,plan,model,scene,original_cells,assets_paths,poses_path,camera_path,synthetic)
    model_files={}
    if not synthetic:
        for path,key in ((parent_checkpoint_path,"parent_checkpoint_sha256"),(final_checkpoint_path,"final_checkpoint_sha256")):
            p=Path(path);require(p.is_file() and not p.is_symlink() and digest(p)==plan[key],"Frozen checkpoint changed after model reconstruction")
            model_files[str(p)]=digest(p)
    from camera_options import preview_origins
    startup=checks["startup"](Path(startup_directory)/"startup",scene=scene,cells=startup_cells,origins=preview_origins())
    require(startup.get("status")=="passed" and startup.get("startup_control_sha256")==native_binding["startup_control_sha256"],"Independent startup evidence did not pass or bind current controls")
    controls=checks["controls"](Path(startup_directory),scene,native_binding["startup_control_sha256"])
    events=_journal(directory/"journal")
    require(events[-1]["fingerprint"]==expected_head_sha256==summary.get("journal_head_sha256"),"Native TEST journal head changed")
    frozen=dict(final_checkpoint_sha256=plan["final_checkpoint_sha256"],parent_checkpoint_sha256=plan["parent_checkpoint_sha256"],
        bindings=plan["bindings"],model_hashes=plan["model_hashes"])
    require(events[0]["kind"]=="start" and events[0]["payload"]==dict(plan_fingerprint=plan["fingerprint"],
        synthetic=synthetic,mock_only=synthetic,frozen=frozen),"Native TEST start did not freeze the same models and sources")
    native=directory/"native";require(native.is_dir() and not native.is_symlink(),"Missing native TEST evidence")
    rows=[];pending=None;next_cohort=0;expected_files=set();batches=[];final=None
    costs=dict(wall_s=0.,process_cpu_s=0.)
    from settled_transit import protocol as setup_protocol
    test_by_id={c["id"]:c for c in plan["original_plan"]["cells_by_split"]["test"]}
    for event in events[1:]:
        require(final is None,"Native TEST journal continued after final freeze")
        kind,payload=event["kind"],event["payload"]
        if kind=="reserve":
            require(pending is None and next_cohort<6,"Repeated or unresolved native TEST reservation")
            cohort=next_cohort;choices=plan["cohorts"][cohort]["choices"];old=plan["original_plan"]["choices"][(cohort+6)*35:(cohort+7)*35]
            require(payload==dict(cohort=cohort,original_cohort=cohort+6,choices=choices,requested=(cohort+1)*35,
                terminal=len(rows),plan_fingerprint=plan["fingerprint"]),"Native TEST reservation order or accounting changed")
            stem=f"evaluation-{cohort:06d}";packet,packet_sha=_packet(native,stem)
            envelope_path=_file(native,stem+".envelope.json");envelope=_json(envelope_path.read_bytes())
            expected=dict(schema=ENVELOPE_SCHEMA,plan_fingerprint=plan["fingerprint"],reservation_sha256=event["fingerprint"],
                new_cohort=cohort,original_cohort=cohort+6,new_choices=choices,original_choices=old,
                packet=dict(file=stem+".packet.json.gz",sha256=packet_sha),native_binding_fingerprint=native_binding["fingerprint"],
                final_checkpoint_sha256=plan["final_checkpoint_sha256"],synthetic=synthetic)
            expected["fingerprint"]=canonical_hash(expected);require(envelope==expected,"Outer TEST envelope or inner original request association changed")
            dispatch=_json(_file(native,stem+".dispatch.json").read_bytes())
            require(dispatch==dict(schema="sircus-v4-test-native-dispatch-v1",plan_fingerprint=plan["fingerprint"],
                reservation_sha256=event["fingerprint"],new_cohort=cohort,original_cohort=cohort+6,
                new_request_ids=[c["request_id"] for c in choices],original_request_ids=[c["request_id"] for c in old],
                native_binding_fingerprint=native_binding["fingerprint"],status="claimed_before_native_call"),"Native TEST lacks its exact pre-call dispatch")
            require(packet.get("cohort")==packet.get("batch")==cohort+6 and packet.get("synthetic") is synthetic
                and packet.get("reservation_sha256")==event["fingerprint"] and packet.get("assessment_plan_fingerprint")==plan["original_plan"]["fingerprint"]
                and packet.get("native_binding")==native_binding and packet.get("model_hashes")==plan["model_hashes"]
                and packet.get("final_checkpoint_sha256")==plan["final_checkpoint_sha256"] and packet.get("split")=="test"
                and packet.get("method")==choices[0]["method"] and packet.get("startup_controls")==controls
                and packet.get("setup_controller")==setup_protocol(),
                "Native TEST inner packet changed model, source, method or startup control")
            expected_files.update(stem+suffix for suffix in (".dispatch.json",".packet.json.gz",".receipt.json",".envelope.json"))
            physics=packet.get("physics")
            if physics is not None:
                require(type(physics) is dict and physics.get("file")==stem+".npz" and digest(_file(native,physics["file"]))==physics["sha256"],"Native TEST physics bytes changed")
                expected_files.add(physics["file"])
            cells=[dict(copy.deepcopy(original_cells[i]),**copy.deepcopy(test_by_id[c["context_id"]])) for i,c in enumerate(old)]
            before_packet=canonical_hash(packet)
            proof=checks["packet"](native,packet,scene=scene,cells=cells,links=links,choices=old)
            require(canonical_hash(packet)==before_packet,"Independent TEST reader mutated saved packet data")
            terminals=proof["scores"]["terminal_results"]
            require(type(terminals) is list and len(terminals)==35,"Independent scorer omitted paid TEST slots")
            verified=[]
            for choice,old_choice,row in zip(choices,old,terminals):
                require(row.get("request_id")==old_choice["request_id"],"Independent scorer changed original slot identity")
                terminal=dict(copy.deepcopy(row),request_id=choice["request_id"],source_sha256=digest(envelope_path))
                _outcome(terminal,choice);verified.append(terminal)
            pending=dict(cohort=cohort,rows=verified,done=0);next_cohort+=1
            for field in costs:
                value=packet[field];require(type(value) in (int,float) and math.isfinite(value) and value>=0,"Invalid native TEST batch cost")
                costs[field]+=value
            batches.append(dict(cohort=cohort,original_cohort=cohort+6,packet_sha256=packet_sha,envelope_sha256=digest(envelope_path),
                guards=proof["guards"],scores={k:v for k,v in proof["scores"].items() if k!="terminal_results"}))
        elif kind=="outcomes":
            require(pending is not None and type(payload.get("results")) is list and payload["results"],"Native TEST outcomes lack a reservation")
            count=len(payload["results"]);begin=pending["done"]
            require(payload==dict(cohort=pending["cohort"],results=pending["rows"][begin:begin+count],
                requested=next_cohort*35,terminal=len(rows)+count) and begin+count<=35,"Terminal TEST receipts differ from independent guards/physics scoring")
            rows.extend(payload["results"]);pending["done"]+=count
            if pending["done"]==35:pending=None
        elif kind=="final":
            require(pending is None and next_cohort==6 and len(rows)==210,"Incomplete native TEST cannot pass")
            final=payload
        else:raise ValueError("Unexpected or failed native TEST history; no retry or partial pass")
    counts=dict(requested=210,terminal=210,released=sum(r["released"] for r in rows),rejected=sum(not r["released"] for r in rows),
        captures=sum(r["captured"] for r in rows),clean_captures=sum(r["clean"] for r in rows),unresolved=0)
    expected_final=dict(plan_fingerprint=plan["fingerprint"],frozen=frozen,counts=counts,synthetic=synthetic,mock_only=synthetic,
        training_requests=0,optimizer_steps=0,native_requests_executed=0 if synthetic else 210)
    require(type(final) is dict and all(type(final.get(k)) is int and type(summary.get(k)) is int
        for k in ("training_requests","optimizer_steps","native_requests_executed"))
        and type(summary.get("counts")) is dict and all(type(summary["counts"].get(k)) is int for k in counts)
        and type(final.get("counts")) is dict and all(type(final["counts"].get(k)) is int for k in counts)
        and final.get("synthetic") is summary.get("synthetic") is synthetic
        and final.get("mock_only") is summary.get("mock_only") is synthetic
        and final==expected_final and all(summary.get(k)==v for k,v in expected_final.items())
        and all(summary.get(k) is False for k in ("original_assessment_completed","model_promoted","execution_authorized","release_authorized")),
        "Native TEST final counts, immutable models or zero-learning boundary changed")
    require(set(p.name for p in native.iterdir())==expected_files and all(p.is_file() and not p.is_symlink() for p in native.iterdir()),
        "Unlisted, repeated, partial or linked native TEST evidence")
    require(all(digest(Path(p))==sha for p,sha in model_files.items()) and digest(Path(model_receipt_path))==expected_model_receipt_sha256,
        "Frozen model or independent receipt bytes changed during native audit")
    if not synthetic:_binding(native_binding,plan,model,scene,original_cells,assets_paths,poses_path,camera_path,False)
    result=dict(schema=SCHEMA,status="passed_synthetic" if synthetic else "passed",synthetic=synthetic,
        all_components_passed=True,full_native_audit=not synthetic,plan_fingerprint=plan["fingerprint"],journal_head_sha256=expected_head_sha256,
        evaluation_result_sha256=expected_result_sha256,model_receipt_sha256=expected_model_receipt_sha256,
        final_checkpoint_sha256=plan["final_checkpoint_sha256"],parent_checkpoint_sha256=plan["parent_checkpoint_sha256"],
        model_hashes=plan["model_hashes"],model=model,startup=startup,batches=batches,counts=counts,comparison=_comparison(plan,rows,synthetic),
        native_batch_costs=costs,current_checkpoint_bytes_verified=not synthetic,optimizer_steps=0,
        original_assessment_status="incomplete_no_retry",original_assessment_completed=False,model_promoted=False,
        execution_authorized=False,release_authorized=False,
        scope="Independent separate TEST-only model/startup/guard/scorer proof. Synthetic hooks certify software only; old V4 assessment and frozen benchmark remain unchanged. Batch cost sums are contained in application evaluation timers, not additive.")
    result["fingerprint"]=canonical_hash(result)
    return result
