"""Strict read-only inputs for a new evaluation, never training continuation.

The original producer's source and checkpoint bindings remain unchanged.
Execution source is a separate identity. File inspection does not start a model,
optimizer, simulator, service or process. Real inputs require the fixed V4 bytes.
"""

import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re

from candidate_continuation_application import load_manifest, read, sha, require
from candidate_continuation_ledger import SourceBindings, canonical_hash
from candidate_continuation_native import guard_sources, _frozen_identity
from candidate_continuation_production import ROLES, verify_original_identity, _parent_sha
from continuous_feasibility import canonical_hash as geometry_hash
from candidate_v4_evaluation import (validate_prepared, V4_PRODUCER_SHA256,
    V4_CHECKPOINT_SHA256, V4_PLAN_RAW_SHA256, V4_COMPONENT_SHA256,
    PARENT_CHECKPOINT_SHA256)

SOURCE=Path(__file__).resolve().parent
SCHEMA="sircus-v4-evaluation-production-inputs-v1"
ORIGINAL_MANIFEST_SHA256="16c7c5edca352b192794505d5433643211a7bf16aca0cc631ff6409183340c47"
LEARNING_RESULT_SHA256="0ab677a89142be6c0cf423194f2720493f5c18939e7ba5fb9a3f93872e408a86"
ORIGINAL_NATIVE_BINDING_SHA256="2c9d8431feaf79a1ba37de743c3a556b7a1ba973fdc49b3698afd4f1dea5d8ca"
MANDATORY=frozenset(("prepared_plan","parent_checkpoint","final_checkpoint","original_manifest",
    "original_learning_result","original_native_binding","old_assessment_plan",
    "old_assessment_inventory","old_component_summary"))
MODEL_FILES=("continuous_policy.py","continuous_feasibility.py","candidate_continuation_checkpoint.py",
    "candidate_continuation_policy.py","candidate_continuation_inputs.py","candidate_continuation_contract.py",
    "candidate_continuation_scene.py","presentation_recording.py","camera_options.py","film_cylinder_farm.py")
PINNED_RUNTIME=dict(python="3.11.13",numpy="1.26.0",scipy="1.15.3",torch="2.7.0+cu128")


def _source(directory):
    directory=Path(directory)
    require(directory.is_dir() and not directory.is_symlink(),"Missing or linked immutable source")
    members=sorted(directory.glob("*.py"))
    require(members and all(p.is_file() and not p.is_symlink() for p in members),"Invalid source member")
    return canonical_hash({p.name:sha(p) for p in members})


def _mapper(host,container):
    host,container=Path(host),Path(container)
    require(host.is_absolute() and ".." not in host.parts and container.is_absolute(),"Invalid workspace roots")
    def mapped(value):
        p=Path(value)
        require(p.is_absolute() and ".." not in p.parts and p.is_relative_to(host),"Input escaped workspace")
        target=container/p.relative_to(host)
        require(target.resolve().is_relative_to(container.resolve()) and not any(
            q.is_symlink() for q in (target,*target.parents) if q.is_relative_to(container)),"Linked or escaped input")
        return target
    return mapped


def load_evaluation_manifest(path,expected_sha256,*,container_root=Path("/workspace")):
    path=Path(path)
    require(path.is_file() and not path.is_symlink() and sha(path)==expected_sha256,"Evaluation manifest bytes changed")
    m=read(path)
    require(m.get("schema")==SCHEMA and type(m.get("synthetic")) is bool
        and m.get("execution_authorized") is m.get("release_authorized") is False,
        "Evaluation manifest cannot grant execution permission")
    mapped=_mapper(m["workspace_host"],container_root)
    require(type(m.get("files")) is dict and set(m["files"])==MANDATORY,"Evaluation input role coverage changed")
    paths={}
    for role,row in m["files"].items():
        require(type(row) is dict and set(row)=={"path","sha256"}
            and type(row["sha256"]) is str and re.fullmatch(r"[0-9a-f]{64}",row["sha256"]),"Malformed input binding")
        p=mapped(row["path"])
        require(p.is_file() and not p.is_symlink() and sha(p)==row["sha256"],"Evaluation input bytes changed: "+role)
        paths[role]=p
    executed=mapped(m["source_host"])/"sircus/isaac"
    original=mapped(m["original_source_host"])/"sircus/isaac"
    require(executed.resolve()==SOURCE.resolve() and _source(executed)==m["source_sha256"],"Executed evaluation source differs")
    require(_source(original)==m["original_source_sha256"],"Immutable original source differs")
    require(m.get("expected_runtime")==PINNED_RUNTIME,"Pinned producer runtime changed")
    plan=validate_prepared(read(paths["prepared_plan"]))
    require(plan["synthetic"] is m["synthetic"] and plan["fingerprint"]==m["prepared_plan_fingerprint"]
        and plan["bindings"]["evaluation_source_sha256"]==m["source_sha256"]
        and plan["bindings"]["producer_source_sha256"]==m["original_source_sha256"],"Prepared/source lineage differs")
    require(read(paths["old_assessment_plan"])==plan["original_plan"]
        and read(paths["old_assessment_inventory"])==plan["prior_evidence"]["inventory"],"Original plan/inventory changed")
    if not m["synthetic"]:
        expected=dict(parent_checkpoint=PARENT_CHECKPOINT_SHA256,final_checkpoint=V4_CHECKPOINT_SHA256,
            original_manifest=ORIGINAL_MANIFEST_SHA256,original_learning_result=LEARNING_RESULT_SHA256,
            original_native_binding=ORIGINAL_NATIVE_BINDING_SHA256,old_assessment_plan=V4_PLAN_RAW_SHA256,
            old_component_summary=V4_COMPONENT_SHA256)
        require(m["original_source_sha256"]==V4_PRODUCER_SHA256 and all(m["files"][k]["sha256"]==v for k,v in expected.items()),
            "Fixed closed V4 source/checkpoint/proof bytes changed")
    protected=m.get("protected_roots")
    require(type(protected) is list and protected and len(set(protected))==len(protected),"Explicit protected roots required")
    protected_paths=[mapped(p) for p in protected]
    require(all(any(p.resolve().is_relative_to(q.resolve()) for q in protected_paths) for p in paths.values())
        and any(original.resolve().is_relative_to(q.resolve()) for q in protected_paths)
        and any(executed.resolve().is_relative_to(q.resolve()) for q in protected_paths),"Sources and all evidence must be protected")
    assessment=paths["old_assessment_plan"].parent
    inventory={}
    for p in sorted(assessment.rglob("*")):
        require(not p.is_symlink(),"Linked prior assessment evidence")
        if p.is_file():inventory[str(p.relative_to(assessment))]=dict(sha256=sha(p),bytes=p.stat().st_size)
    require(inventory==plan["prior_evidence"]["inventory"],"Partial assessment changed or a TEST request appeared")
    return m,paths,mapped,plan,protected_paths


def _geometry_inputs(manifest_path,expected_sha256,*,container_root=Path("/workspace")):
    """Reconstruct original training input bindings, distinct from execution source."""
    from candidate_continuation_inputs import validate_inputs
    from candidate_continuation_policy import FrozenCandidateSupport
    from candidate_continuation_contract import PROTOCOL_SHA256
    from candidate_native_audit import SCORE_FILES
    from candidate_fresh_splits import cells_from_splits
    from settled_transit import protocol
    from continuous_policy import build_observations
    manifest,paths,mapped=load_manifest(manifest_path,expected_sha256,container_root=container_root)
    require(set(ROLES).issubset(paths) and all(k in ROLES or k.startswith("asset:") for k in paths),
            "Production preparation input coverage changed")
    old_source=mapped(manifest["source_host"])/"sircus/isaac"
    require(_source(old_source)==manifest["source_sha256"],"Original producer source changed")
    from candidate_continuation_native import GUARD_FILES
    for name in set(GUARD_FILES) | set(SCORE_FILES) | set(MODEL_FILES):
        require(sha(SOURCE/name)==sha(old_source/name),"Original physics/model arithmetic changed: "+name)
    values={role:read(path) for role,path in paths.items() if role not in
            ("parent_checkpoint","dropper_urdf","board_urdf") and role in ROLES}
    identity=_frozen_identity()
    verify_original_identity(manifest,values["scene"])
    require(values["source_contract"]["fingerprint"]==identity["geometry_fingerprint"]
        and values["source_contract"]["scene"]==values["scene"]
        and sha(paths["scene"])==sha(SOURCE/"scene.json"),
        "Executed source scene/geometry differs from verified original inputs")
    source_run=mapped(manifest["source_run"])
    assets_dir=mapped(manifest["assets_dir"])
    require(all(paths[role]==assets_dir/name for role,name in
        (("dropper_urdf","dropper.urdf"),("board_urdf","board_arm.urdf"),("poses","poses.json"))),
        "Native asset directory differs from verified inputs")
    asset_inventory={}
    for path in assets_dir.rglob("*"):
        require(not path.is_symlink(),"Linked generated simulation asset")
        if path.is_file():asset_inventory[str(path.relative_to(assets_dir))]=sha(path)
    bound_assets={str(paths[r].relative_to(assets_dir)):manifest["files"][r]["sha256"]
                  for r in ("dropper_urdf","board_urdf","poses")}
    for role,path in paths.items():
        if role.startswith("asset:"):
            require(path.is_relative_to(assets_dir) and role=="asset:"+str(path.relative_to(assets_dir)),
                    "Generated asset path/role mismatch")
            bound_assets[str(path.relative_to(assets_dir))]=manifest["files"][role]["sha256"]
    require(asset_inventory==bound_assets,"Unbound or changed generated simulation assets")
    for role,name in (("source_contract","contract.json"),("source_result","result.json"),
        ("source_menu","expanded-menu-qualified.json"),("source_camera_binding","camera-binding.json"),
        ("source_probes","probes.json"),("source_rim_rays","rim-rays.json"),("source_selected_camera","selected-camera.json")):
        require(paths[role]==source_run/name,"Source metadata path mapping changed")
    train=values["train_inputs"];train_supports=validate_inputs(train)
    qualification=values["train_qualification"]
    require(values["train_audit"]["status"]=="passed" and values["train_bank"]["status"]=="passed",
            "TRAIN bank independent audit must pass")
    for name,role in (("plan.json","train_plan"),("qualification.json","train_qualification"),("audit.json","train_audit")):
        require(values["train_bank"]["artifacts_sha256"][name]==manifest["files"][role]["sha256"],"TRAIN bank byte lineage changed")
    require(train["input_bindings"]["train_evidence_sha256"]==manifest["files"]["train_evidence"]["sha256"],
            "Original successful TRAIN demonstration bytes changed")
    fresh=values["fresh_summary"]
    require(set(manifest["expected_runtime"])=={"python","numpy","scipy","torch"}
        and all(type(v) is str and v for v in manifest["expected_runtime"].values()),
        "Exact reviewed producer runtime versions must be bound before execution")
    require(fresh["status"]==values["fresh_audit"]["status"]==values["fresh_split_audit"]["status"]=="passed",
            "Fresh split/support independent audits must pass")
    for name,role in (("split-manifest.json","fresh_split_manifest"),("split-audit.json","fresh_split_audit"),
                      ("parent-predictions.json","fresh_parent"),("plan.json","fresh_plan"),
                      ("qualification.json","fresh_qualification"),("audit.json","fresh_audit")):
        require(fresh["artifact_sha256"][name]==manifest["files"][role]["sha256"],"Fresh preparation byte lineage changed")
    split=values["fresh_split_manifest"];fresh_cells=cells_from_splits(split)
    cells_by_split=dict(train=copy.deepcopy(train["cells"]),
        validation=[c for c in fresh_cells if c["split"]=="validation"],
        test=[c for c in fresh_cells if c["split"]=="test"])
    supports={s.context_id:s for s in train_supports}
    for sheet in values["fresh_qualification"]["supports"]:
        supports[sheet["cell_id"]]=None if sheet["support"] is None else FrozenCandidateSupport.from_dict(sheet["support"])
    records={r["candidate_id"]:r for r in qualification["records"]}
    records.update({r["candidate_id"]:r for r in values["fresh_qualification"]["records"]})
    train_ids={c["id"] for c in cells_by_split["train"]}
    train_records={key:r for key,r in records.items() if r["cell_id"] in train_ids}
    parent=copy.deepcopy(train["parent_distributions"])
    for cell in fresh_cells:
        key=cell["id"];pred=values["fresh_parent"]
        parent[key]=dict(exact_action=pred["parent_actions"][key],latent_mean=pred["parent_latent_means"][key],
                         latent_scale=pred["parent_latent_scales"][key])
    reference=values["historical_reference"]
    require(reference["status"]==reference["audit"]["status"]=="passed"
        and reference["source_binding"]["checkpoint_sha256"]==_parent_sha()
        and reference["source_binding"]["candidate_bank_sha256"]==manifest["files"]["train_qualification"]["sha256"],
        "Historical TRAIN reference audit changed")
    require(len(reference["cases"])==35 and {c["cell_id"] for c in reference["cases"]}==train_ids,
            "Historical TRAIN reference positions changed")
    xy={c["id"]:c["bucket_xy_m"] for c in train["cells"]}
    require(all(c["bucket_xy_m"]==xy[c["cell_id"]] for c in reference["cases"]),"Historical retention coordinates changed")
    historical={method:[c["cell_id"] for c in reference["cases"] if c[key]["capture"]]
                for method,key in (("parent","parent"),("candidate_bank","bank"))}
    require(len(historical["parent"])==13 and len(historical["candidate_bank"])==16,
            "Historical TRAIN retention witnesses changed")
    common={"Arm10":values["poses"]["dropper_joint_rad"],"Arm14":values["poses"]["board_joint_rad"]}
    native_contract=dict(scene_sha256=geometry_hash(values["scene"]),canonical_home_rad=common,
        generated_assets_sha256=canonical_hash(bound_assets),
        cells_sha256=canonical_hash(train["cells"]),support_sha256=canonical_hash([s.to_dict() for s in train_supports]),
        prepared_records_sha256=canonical_hash(train_records),guard_source_sha256=guard_sources(),
        setup_controller=protocol(),source_sha256=manifest["source_sha256"],
        **{key:value for key,value in _frozen_identity().items() if key!="scene_sha256"})
    require(manifest["native_contract"]==native_contract,"Pre-start native contract changed")
    observations=build_observations(train["cells"])
    sources=dict(parent_checkpoint_sha256=_parent_sha(),train_bank_sha256=manifest["files"]["train_bank"]["sha256"],
        train_evidence_sha256=manifest["files"]["train_evidence"]["sha256"],train_support_sha256=native_contract["support_sha256"],
        fresh_split_manifest_sha256=manifest["files"]["fresh_split_manifest"]["sha256"],
        fresh_support_sha256=canonical_hash(values["fresh_qualification"]["supports"]),
        guard_protocol_sha256=canonical_hash(dict(sources=guard_sources(),controller=protocol())),
        scorer_sha256=canonical_hash({n:sha(Path(__file__).with_name(n)) for n in SCORE_FILES}),
        producer_source_sha256=manifest["source_sha256"])
    bindings=dict(sources,learning_protocol_sha256=PROTOCOL_SHA256,
        parent_weight_hash="ea9a69ec063e3d71cda2b92f1da64939f644244d4b4dbd70b500be132632e549",
        source_sha256=manifest["source_sha256"],support_sha256=native_contract["support_sha256"],
        demonstrations_sha256=canonical_hash(train["demonstrations"]),observations_sha256=canonical_hash(observations.tolist()),
        input_sha256=manifest["files"]["train_inputs"]["sha256"])
    return dict(manifest=manifest,paths=paths,mapped=mapped,values=values,train=train,train_supports=train_supports,
        supports=supports,records=records,train_records=train_records,cells_by_split=cells_by_split,
        parent_distributions=parent,historical_capture_ids=historical,canonical_home=common,
        source_bindings=SourceBindings(**sources),bindings=bindings,observations=observations)

def production_inputs(path,expected_sha256,*,container_root=Path("/workspace")):
    """Load identities/geometry only; model forward checking is a separate child."""
    m,p,mapped,plan,protected=load_evaluation_manifest(path,expected_sha256,container_root=container_root)
    original=read(p["original_manifest"])
    require(mapped(original["source_host"])/"sircus/isaac"==mapped(m["original_source_host"])/"sircus/isaac"
        and original["source_sha256"]==m["original_source_sha256"],"Original manifest source mapping differs")
    i=_geometry_inputs(p["original_manifest"],m["files"]["original_manifest"]["sha256"],container_root=container_root)
    old_roots=original.get("protected_roots")
    require(type(old_roots) is list and old_roots,"Original protected evidence roots required")
    inherited=[mapped(raw) for raw in old_roots]
    inherited.extend(mapped(original[key]) for key in ("source_host","source_run","assets_dir","frozen_experiment_root"))
    inherited.extend(i["paths"].values())
    require(all(any(path.resolve().is_relative_to(root.resolve()) for root in protected) for path in inherited),
        "New protection omitted original evidence, asset or source-run roots")
    learned=read(p["original_learning_result"])
    require(learned.get("status")=="learning_complete_audit_pending" and learned.get("synthetic") is m["synthetic"]
        and learned.get("bindings")==i["bindings"] and learned.get("checkpoint_sha256")==V4_CHECKPOINT_SHA256,
        "Original trained result/checkpoint bindings changed")
    require(i["paths"]["parent_checkpoint"]==p["parent_checkpoint"],"Parent checkpoint path association changed")
    old_binding=read(p["original_native_binding"])
    require(plan["original_plan"]["bindings"]["native_binding_sha256"]==old_binding.get("fingerprint"),
        "Original model reconstruction native binding changed")
    summary=read(p["old_component_summary"])
    require(summary.get("status")=="component_checks_passed_full_run_incomplete"
        and summary.get("full_combined_audit_status")=="not_run_incomplete"
        and all(summary.get("components",{}).get(k,{}).get("status")=="passed" for k in ("learning","native_train","startup"))
        and summary.get("closed_input_sha256",{}).get("final_checkpoint")==V4_CHECKPOINT_SHA256,
        "Completed TRAIN component proof changed or old failure was promoted")
    i.update(plan=plan,evaluation_manifest=m,evaluation_manifest_path=Path(path),
        evaluation_manifest_sha256=expected_sha256,evaluation_paths=p,protected_roots=protected,
        container_root=Path(container_root),
        execution=dict(source_sha256=m["source_sha256"],original_source_sha256=m["original_source_sha256"],
            plan_fingerprint=plan["fingerprint"]),original_learning_result=learned,original_native_binding=old_binding)
    i["model_request_base"]=dict(schema="sircus-v4-model-check-request-v1",synthetic=m["synthetic"],
        files={k:dict(path=str(p[k]),sha256=m["files"][k]["sha256"]) for k in
            ("prepared_plan","original_manifest","original_learning_result","original_native_binding", "old_assessment_plan","parent_checkpoint","final_checkpoint")},
        original_source_dir=str(mapped(m["original_source_host"])/"sircus/isaac"),
        original_source_sha256=m["original_source_sha256"],evaluation_source_sha256=m["source_sha256"],
        protected_roots=[str(q) for q in protected],expected_runtime=copy.deepcopy(PINNED_RUNTIME))
    return i


def model_request(inputs,receipt_path):
    """Bind a new private CPU-reader receipt, without executing the reader."""
    v=copy.deepcopy(inputs["model_request_base"]);v["receipt_path"]=str(Path(receipt_path))
    v["fingerprint"]=canonical_hash(v)
    return v


def frozen_verifier(inputs):
    """Return a byte-checking guard, with no model or native operation."""
    def verify():
        # Re-run the full immutable manifest and original partial-inventory
        # checks; this is intentionally separate from the saved model binding.
        m,p,_,plan,_=load_evaluation_manifest(inputs["evaluation_manifest_path"],
            inputs["evaluation_manifest_sha256"],container_root=inputs["container_root"])
        load_manifest(p["original_manifest"],m["files"]["original_manifest"]["sha256"],
            container_root=inputs["container_root"])
        require(plan==inputs["plan"] and m==inputs["evaluation_manifest"],"Evaluation inputs changed after freeze")
        return dict(final_checkpoint_sha256=plan["final_checkpoint_sha256"],
            parent_checkpoint_sha256=plan["parent_checkpoint_sha256"],
            bindings=copy.deepcopy(plan["bindings"]),model_hashes=copy.deepcopy(plan["model_hashes"]))
    return verify
