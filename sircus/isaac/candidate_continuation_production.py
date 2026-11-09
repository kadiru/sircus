"""Explicit production components for the one-start continuation application.

No component is constructed or executed on import. The command entry requires
separate execution authorization. This module is mock-tested, not live-validated.
"""

import copy
from dataclasses import asdict
import hashlib
import json
import os
import platform
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np

from candidate_continuation_application import load_manifest,read,sha,require
from candidate_continuation_journal import write_new_json
from candidate_continuation_ledger import Ledger,SourceBindings,canonical_hash
from candidate_continuation_native import make_binding,guard_sources,source_fingerprint,_frozen_identity,validate_transit_limits
from continuous_feasibility import canonical_hash as geometry_hash

ROLES=("scene","poses","camera","dropper_urdf","board_urdf","parent_checkpoint",
    "train_inputs","train_plan","train_qualification","train_audit","train_bank","train_evidence",
    "fresh_split_manifest","fresh_split_audit","fresh_plan","fresh_qualification","fresh_audit",
    "fresh_summary","fresh_parent","historical_reference","source_contract","source_result",
    "source_menu","source_camera_binding","source_probes","source_rim_rays","source_selected_camera")


def verify_original_identity(manifest,scene):
    identity=_frozen_identity()
    require(manifest["files"]["parent_checkpoint"]["sha256"]==_parent_sha()
        and geometry_hash(scene)==identity["scene_sha256"]
        and manifest["files"]["camera"]["sha256"]==identity["camera_sha256"]
        and manifest["files"]["poses"]["sha256"]==identity["poses_sha256"]
        and manifest["files"]["dropper_urdf"]["sha256"]==identity["assets_sha256"]["Arm10"]
        and manifest["files"]["board_urdf"]["sha256"]==identity["assets_sha256"]["Arm14"],
        "Original frozen physics/assets/parent/camera identity changed before app startup")


def production_inputs(manifest_path,expected_sha256,*,container_root=Path("/workspace")):
    """Read verified preparation only; no qualification, inference or native work."""
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
    require(source_fingerprint()==manifest["source_sha256"],"Executed source differs from manifest")
    values={role:read(path) for role,path in paths.items() if role not in
            ("parent_checkpoint","dropper_urdf","board_urdf") and role in ROLES}
    identity=_frozen_identity()
    verify_original_identity(manifest,values["scene"])
    require(values["source_contract"]["fingerprint"]==identity["geometry_fingerprint"]
        and values["source_contract"]["scene"]==values["scene"]
        and paths["scene"]==Path(__file__).resolve().with_name("scene.json"),
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
        setup_controller=protocol(),source_sha256=source_fingerprint(),
        **{key:value for key,value in _frozen_identity().items() if key!="scene_sha256"})
    require(manifest["native_contract"]==native_contract,"Pre-start native contract changed")
    observations=build_observations(train["cells"])
    sources=dict(parent_checkpoint_sha256=_parent_sha(),train_bank_sha256=manifest["files"]["train_bank"]["sha256"],
        train_evidence_sha256=manifest["files"]["train_evidence"]["sha256"],train_support_sha256=native_contract["support_sha256"],
        fresh_split_manifest_sha256=manifest["files"]["fresh_split_manifest"]["sha256"],
        fresh_support_sha256=canonical_hash(values["fresh_qualification"]["supports"]),
        guard_protocol_sha256=canonical_hash(dict(sources=guard_sources(),controller=protocol())),
        scorer_sha256=canonical_hash({n:sha(Path(__file__).with_name(n)) for n in SCORE_FILES}),
        producer_source_sha256=source_fingerprint())
    bindings=dict(sources,learning_protocol_sha256=PROTOCOL_SHA256,
        parent_weight_hash="ea9a69ec063e3d71cda2b92f1da64939f644244d4b4dbd70b500be132632e549",
        source_sha256=source_fingerprint(),support_sha256=native_contract["support_sha256"],
        demonstrations_sha256=canonical_hash(train["demonstrations"]),observations_sha256=canonical_hash(observations.tolist()),
        input_sha256=manifest["files"]["train_inputs"]["sha256"])
    return dict(manifest=manifest,paths=paths,mapped=mapped,values=values,train=train,train_supports=train_supports,
        supports=supports,records=records,train_records=train_records,cells_by_split=cells_by_split,
        parent_distributions=parent,historical_capture_ids=historical,canonical_home=common,
        source_bindings=SourceBindings(**sources),bindings=bindings,observations=observations)


def _parent_sha():
    from candidate_continuation_ledger import PARENT_CHECKPOINT_SHA256
    return PARENT_CHECKPOINT_SHA256


def verify_dependencies(expected):
    """Fail before app startup; never install or change an environment."""
    import scipy
    from continuous_policy import torch
    require(torch is not None,"The pinned production Torch runtime is required")
    actual=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,torch=torch.__version__)
    require(type(expected) is dict and expected==actual,"Production dependency versions differ from the bound runtime")
    return actual


def _parent(path):
    from candidate_continuation_inputs import STUDY_SHA,GEOMETRY_SHA,DOMAIN_SHA
    from continuous_policy import load_checkpoint
    require(sha(path)==_parent_sha(),"Original seed-22 checkpoint bytes changed")
    model,saved=load_checkpoint(path,study_fingerprint=STUDY_SHA,geometry_fingerprint=GEOMETRY_SHA,
        domain_fingerprint=DOMAIN_SHA,expected_updates=73,expected_attempted_shots=10000,device="cpu")
    require(saved["seed"]==20261122,"Wrong frozen parent seed")
    return model


def assessment_context(inputs,train_ledger,trained,final_path,native_binding):
    """Same pinned CPU forward path for planning and independent audit."""
    from candidate_continuation_assessment import predict_distributions
    from candidate_continuation_checkpoint import load_continuation
    from continuous_policy import weight_hash
    from continuous_policy import torch
    if torch is not None:torch.set_num_threads(2)
    parent=_parent(inputs["paths"]["parent_checkpoint"])
    final,_=load_continuation(final_path,expected_sha256=trained["checkpoint_sha256"],expected_bindings=inputs["bindings"],device="cpu")
    all_cells=sum((inputs["cells_by_split"][s] for s in ("train","validation","test")),[])
    parent_predictions=predict_distributions(parent,all_cells)
    # Prepared parent export tolerances preserve exact original actions and latents.
    # Both planning/auditing use its immutable recorded choices, after CPU verification.
    for key,prepared in inputs["parent_distributions"].items():
        actual=parent_predictions[key]
        require(np.allclose(actual["latent_mean"],prepared["latent_mean"],atol=3e-6,rtol=0)
            and np.allclose(actual["latent_scale"],prepared["latent_scale"],atol=3e-6,rtol=0),
            "Frozen parent CPU/prepared distribution changed")
        from continuous_feasibility import ACTION_NAMES
        require(np.allclose([actual["exact_action"][k] for k in ACTION_NAMES],
                            [prepared["exact_action"][k] for k in ACTION_NAMES],atol=3e-6,rtol=0),
                "Prepared direct-parent action differs from frozen CPU inference")
        support=inputs["supports"][key]
        if support is not None:
            from candidate_continuation_assessment import deterministic_index
            require(deterministic_index(support,actual)[0]==deterministic_index(support,prepared)[0],
                    "Frozen parent CPU/prepared candidate ranking changed")
    continued=predict_distributions(final,all_cells)
    hashes=dict(parent_weight_hash=weight_hash(parent),continued_weight_hash=weight_hash(final))
    bindings=dict(asdict(inputs["source_bindings"]),source_sha256=source_fingerprint(),native_binding_sha256=native_binding["fingerprint"],
        historical_capture_ids_sha256=canonical_hash(inputs["historical_capture_ids"]))
    return dict(parent=parent,continued=final,parent_distributions=inputs["parent_distributions"],
        continued_distributions=continued,model_hashes=hashes,bindings=bindings)


class ProductionComponents:
    synthetic=False
    def __init__(self,manifest_path,expected_sha256,output,session,*,device="cuda:0"):
        self.manifest_path,self.expected_sha256=Path(manifest_path),expected_sha256
        self.output,self.session,self.device=Path(output),session,device
        self.app=self.runner=self.assessment=None;self.closed=False;self.device_ready=False

    def initialize(self,publish_startup):
        self.inputs=production_inputs(self.manifest_path,self.expected_sha256)
        i=self.inputs;p=i["paths"];m=i["manifest"]
        verify_dependencies(m["expected_runtime"])
        from presentation_recording import load_source
        load_source(i["mapped"](m["source_run"]),p["camera"])
        # Exercise the original implicit-dtype tensor representation on CPU
        # before the expensive app startup; the actual GPU construction below
        # remains unchanged and is checked again by the native producer.
        from continuous_policy import torch
        cap_urdf_bytes=p["dropper_urdf"].read_bytes()
        cap_urdf=ET.fromstring(cap_urdf_bytes)
        cpu_caps=torch.tensor([float(cap_urdf.find(f".//joint[@name='joint{k+1}']/limit").get("effort"))
            for k in range(6)],device="cpu")
        validate_transit_limits(cpu_caps,cap_urdf_bytes)
        # All source/input checks precede the sole AppLauncher construction.
        from isaaclab.app import AppLauncher
        import argparse
        native_parser=argparse.ArgumentParser()
        AppLauncher.add_app_launcher_args(native_parser)
        native_args=native_parser.parse_args(["--headless","--enable_cameras","--device",self.device])
        self.app=AppLauncher(native_args).app
        from candidate_continuation_scene import construct_scene,StartupRecorder
        self.env,self.bundle=construct_scene(output=self.output/"native-setup",assets=i["mapped"](m["assets_dir"]),
            source_run=i["mapped"](m["source_run"]),camera=p["camera"],device=self.device)
        from prepare_assets import chain
        from continuous_policy import torch,build_observations
        from candidate_continuation_updates import LearningEngine
        from candidate_continuation_runner import ContinuationRunner
        from candidate_continuation_native import GuardedNativeProducer
        self.device_ready=True
        self.env.continuous_links={n:chain(ET.fromstring(p[r].read_bytes())) for n,r in (("Arm10","dropper_urdf"),("Arm14","board_urdf"))}
        urdf=ET.fromstring(p["dropper_urdf"].read_bytes())
        self.env.transit_limits=torch.tensor([float(urdf.find(f".//joint[@name='joint{k+1}']/limit").get("effort")) for k in range(6)],device=self.env.sim.device)
        self.startup=StartupRecorder(self.env,self.env.dest/"startup",publish=publish_startup).run()
        # Startup uses the unchanged original farm. Paid packets use the same
        # minimal context templates reconstructed independently by both audits.
        self.env.base_cells=copy.deepcopy(i["train"]["cells"])
        self.env.spec["cells"]=copy.deepcopy(i["train"]["cells"])
        controls={name:sha(self.env.dest/name) for name in ("probes.json","rim-rays.json")}
        self.native_binding=make_binding(scene=i["values"]["scene"],geometry_fingerprint=_frozen_identity()["geometry_fingerprint"],
            camera_sha256=sha(p["camera"]),assets_sha256={"Arm10":sha(p["dropper_urdf"]),"Arm14":sha(p["board_urdf"])},
            poses_sha256=sha(p["poses"]),cells=i["train"]["cells"],canonical_home_rad=i["canonical_home"],
            supports=i["train_supports"],prepared_records=i["train_records"],startup_control_sha256=controls)
        write_new_json(self.output/"native-binding.json",self.native_binding)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision("highest")
        policy=_parent(p["parent_checkpoint"])
        # The tiny updater stays on CPU with its existing dedicated CPU RNG;
        # native PhysX/renderer remain exclusively on the declared 4090.
        engine=LearningEngine(policy,torch.tensor(build_observations(i["train"]["cells"])),
            i["train_supports"],i["train"]["demonstrations"],i["train"]["successes_by_context"])
        ledger=Ledger.new(tuple(c["id"] for c in i["cells_by_split"]["train"]),
            tuple(c["id"] for c in i["cells_by_split"]["validation"]),tuple(c["id"] for c in i["cells_by_split"]["test"]),
            bindings=i["source_bindings"])
        self.runner=ContinuationRunner(self.output/"learning",engine,ledger,i["bindings"],synthetic=False)
        for name in ("probes.json","rim-rays.json"):
            with (self.runner.directory/name).open("xb") as stream:stream.write((self.env.dest/name).read_bytes())
        self.env.dest=self.runner.directory
        self.producer=GuardedNativeProducer(self.env,self.native_binding,i["train"]["cells"],i["train_supports"],i["train_records"],
            parent_checkpoint_path=p["parent_checkpoint"],camera_path=p["camera"],poses_path=p["poses"],
            assets_paths={"Arm10":p["dropper_urdf"],"Arm14":p["board_urdf"]},
            frozen_experiment_root=i["mapped"](m["frozen_experiment_root"]))
        return self.runner,self.producer,self.startup

    def begin_assessment(self,runner,trained):
        from candidate_continuation_assessment import build_assessment_plan,AssessmentRunner
        from candidate_continuation_assessment_native import GuardedAssessmentProducer
        from continuous_policy import weight_hash
        i=self.inputs
        self.context=assessment_context(i,runner.ledger,trained,runner.directory/"final.pt",self.native_binding)
        c=self.context
        plan=build_assessment_plan(runner.ledger,final_checkpoint_sha256=trained["checkpoint_sha256"],
            cells_by_split=i["cells_by_split"],supports=i["supports"],prepared_records=i["records"],
            parent_distributions=c["parent_distributions"],continued_distributions=c["continued_distributions"],
            model_hashes=c["model_hashes"],bindings=c["bindings"])
        self.assessment=AssessmentRunner(self.output/"assessment",runner.ledger,plan,
            parent_checkpoint_path=i["paths"]["parent_checkpoint"],final_checkpoint_path=runner.directory/"final.pt",
            state_hashes=lambda:dict(parent_weight_hash=weight_hash(c["parent"]),continued_weight_hash=weight_hash(c["continued"])),synthetic=False)
        return self.assessment,GuardedAssessmentProducer(self.producer,plan)

    def preparation_evidence(self):
        i=self.inputs
        return dict(reused_train_bank=dict(candidates=sum(len(s.candidate_ids) for s in i["train_supports"]),
            demonstrations=len(i["train"]["demonstrations"]),summary_sha256=i["manifest"]["files"]["train_bank"]["sha256"],
            scope="Previously prepared TRAIN bank; not new preparation cost or additional native experience."),
            fresh=dict(counts=copy.deepcopy(i["values"]["fresh_summary"]["counts"]),
                cost=copy.deepcopy(i["values"]["fresh_summary"]["cost"]),
                runtime=copy.deepcopy(i["values"]["fresh_summary"]["runtime"]),
                summary_sha256=i["manifest"]["files"]["fresh_summary"]["sha256"],
                scope="Previously completed CPU geometry-only preparation, separate from this application's phases."))

    def device_snapshot(self,reset):
        if not self.device_ready:return None
        from continuous_policy import torch
        torch.cuda.synchronize()
        if reset:torch.cuda.reset_peak_memory_stats();return None
        free,total=torch.cuda.mem_get_info()
        return dict(torch_peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
                    torch_peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2,
                    sampled_whole_device_occupied_mib=(total-free)/1024**2)

    def close_native(self):
        if self.app is not None and not self.closed:
            self.app.close(skip_cleanup=True);self.closed=True

    def abort(self,error):
        for component in (self.assessment,self.runner):
            if component is not None and not component.finished and not component.failed:component._fail(error)

    def audit(self,trained,assessed):
        request=dict(session=self.session,source_sha256=source_fingerprint(),
            learning_result_sha256=sha(self.output/"learning/result.json"),
            assessment_result_sha256=sha(self.output/"assessment/assessment-result.json"),
            native_binding_sha256=sha(self.output/"native-binding.json"),
            startup_receipt_sha256=sha(self.output/"native-setup/startup/receipt.json"))
        write_new_json(self.output/"audit-request.json",request)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES="",PYTHONDONTWRITEBYTECODE="1")
        command=[sys.executable,str(Path(__file__).with_name("candidate_continuation_entry.py")),
            "--audit-only","--manifest",str(self.manifest_path),"--expected-manifest-sha256",self.expected_sha256,
            "--output",str(self.output),"--session",self.session,
            "--expected-audit-request-sha256",sha(self.output/"audit-request.json")]
        subprocess.run(command,check=True,timeout=600,env=env)
        return read(self.output/"combined-audit.json")
