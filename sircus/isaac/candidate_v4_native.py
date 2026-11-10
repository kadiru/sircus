"""Evaluation-only native adapter. Importing this module starts nothing.

The existing measured controller, scene, shot guards and scorer are reused.
There is no learning engine or optimizer. A separately authorized application
must provide a new output, validated inputs and its durable reservation.
"""

import copy
import json
import shutil
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import numpy as np

from candidate_continuation_journal import read_events, write_new_json, _sync_directory
from candidate_continuation_ledger import canonical_hash
from candidate_continuation_native import _sync_file, _raw_incomplete
from candidate_v4_evaluation import validate_prepared

ENVELOPE_SCHEMA = "sircus-v4-test-native-envelope-v1"
PACKET_SCHEMA = "sircus-candidate-continuation-assessment-native-trial-v1"


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class GuardedTestProducer:
    """One paid native call per committed TEST reservation; no retry path."""

    def __init__(self, holder, plan, directory, *, frozen_guard, synthetic=False, free_bytes=None):
        self.plan = validate_prepared(plan)
        require(type(synthetic) is bool and self.plan["synthetic"] is synthetic,
                "Native TEST provenance differs from prepared plan")
        if not synthetic:
            from candidate_continuation_native import GuardedNativeProducer
            require(type(holder) is GuardedNativeProducer, "Expected original guarded native holder")
        self.holder, self.env = holder, holder.env
        self.binding = copy.deepcopy(holder.binding)
        self.directory = Path(directory)
        require(self.directory.is_absolute() and self.directory == self.directory.resolve()
            and self.directory.is_dir() and not any(p.is_symlink() for p in (self.directory,*self.directory.parents)),
            "Native TEST output must be an existing canonical nonsymlink directory")
        require(free_bytes is None or synthetic, "Production storage checks cannot be replaced")
        self.free_bytes = free_bytes or (lambda:shutil.disk_usage(self.directory).free)
        self.controls_directory = Path(self.env.dest)
        self.frozen_guard, self.synthetic = frozen_guard, synthetic
        self.original_cells = copy.deepcopy(holder.original_cells)
        self.claimed = set()
        self.plan_hash=canonical_hash(self.plan);self.binding_hash=canonical_hash(self.binding)
        self.original_cells_hash=canonical_hash(self.original_cells)
        require(len(self.original_cells) == 35 and callable(frozen_guard), "Missing original farm or frozen-input guard")
        require(self.binding["source_sha256"] == self.plan["bindings"]["evaluation_source_sha256"],
                "New evaluation source differs from native binding")
        require(not self.directory.is_relative_to(holder.frozen_root) and not holder.frozen_root.is_relative_to(self.directory),
            "New TEST output overlaps frozen benchmark")
        self._guard()

    def _guard(self):
        require(canonical_hash(self.plan)==self.plan_hash and canonical_hash(self.binding)==self.binding_hash
            and self.binding==self.holder.binding and canonical_hash(self.original_cells)==self.original_cells_hash,
            "Frozen native TEST plan, binding or original farm templates changed")
        free=self.free_bytes()
        require(type(free) is int and free>=6*1024**3, "Original 6 GiB storage floor reached before native TEST call")
        self.frozen_guard()
        self.holder._validate_binding()

    def execute_reserved(self, cohort, choices, reservation, *, journal_directory):
        self._guard()
        index = cohort["index"]
        require(type(index) is int and 0 <= index < 6 and cohort == self.plan["cohorts"][index]
            and choices == cohort["choices"] and len(choices) == 35,
            "TEST choices changed from the frozen six-cohort schedule")
        events = read_events(journal_directory)
        expected = dict(cohort=index, original_cohort=cohort["original_cohort"], choices=choices,
            requested=(index+1)*35, terminal=index*35, plan_fingerprint=self.plan["fingerprint"])
        require(events and events[-1]["kind"] == "reserve" and events[-1]["payload"] == expected
            and events[-1]["fingerprint"] == reservation.get("journal_head_sha256")
            and reservation.get("cohort") == index and reservation.get("original_cohort") == index+6
            and reservation.get("requested") == (index+1)*35
            and reservation.get("request_ids") == [c["request_id"] for c in choices],
            "Native TEST dispatch lacks its exact durable charged reservation")
        head = events[-1]["fingerprint"]
        require(head not in self.claimed, "Repeated native TEST reservation")
        original = self.plan["original_plan"]["choices"][(index+6)*35:(index+7)*35]
        native = self.directory/"native"
        if not native.exists():
            native.mkdir(); _sync_directory(self.directory)
        require(native.is_dir() and not native.is_symlink(), "Linked native output")
        stem = f"evaluation-{index:06d}"
        target, physics_path = native/(stem+".packet.json.gz"), native/(stem+".npz")
        require(not any(native.glob(stem+".*")), "Native TEST cohort already claimed; no retry")
        from candidate_native_audit import verify_controls
        controls = verify_controls(self.controls_directory,self.env.cfg,self.binding["startup_control_sha256"])
        dispatch = dict(schema="sircus-v4-test-native-dispatch-v1", plan_fingerprint=self.plan["fingerprint"],
            reservation_sha256=head,new_cohort=index,original_cohort=index+6,
            new_request_ids=[c["request_id"] for c in choices],original_request_ids=[c["request_id"] for c in original],
            native_binding_fingerprint=self.binding["fingerprint"],status="claimed_before_native_call")
        write_new_json(native/(stem+".dispatch.json"),dispatch)
        self.claimed.add(head)
        env = self.env
        cells_by_id = {c["id"]:c for c in self.plan["original_plan"]["cells_by_split"]["test"]}
        cells = [dict(copy.deepcopy(self.original_cells[i]),**copy.deepcopy(cells_by_id[c["context_id"]]))
            for i,c in enumerate(original)]
        actions=[copy.deepcopy(c["action"]) for c in original]
        enabled=[c["enabled"] for c in original]
        seeds=[copy.deepcopy(c["ik_seeds"]) for c in original]
        began,cpu = time.monotonic(),time.process_time()
        try:
            # Recheck immediately before home; a prepared fit is not a permit.
            self._guard()
            env.dest=self.directory
            env.capture_trial=env.film_trial=False;env.phase=None
            env.base_cells=copy.deepcopy(cells);env.spec["cells"]=copy.deepcopy(cells)
            env.bucket_xy=np.asarray([c["bucket_xy_m"] for c in cells])
            env.home_readback=env.home()
            env.active_mask=np.asarray(enabled,dtype=bool);env.support_missing=[False]*35
            env.proposal_ik_seeds=copy.deepcopy(seeds)
            env.trial_mode,env.trial_batch=original[0]["method"],index+6
            env.frames,env.physics_samples=[],[]
            env.capture_trial,env.film_trial,env.phase=True,False,"setup"
            env.trial_start_step=env.steps
            rows,raw=env.shot_batch(copy.deepcopy(actions),record_video=False,enabled=env.active_mask)
            env.capture_trial=env.film_trial=False;env.phase=None
            require(len(rows)==35,"Native TEST lost a paid slot")
            from presentation_recording import save_physics
            from shot_benchmark_reward import build_features,reward
            from settled_transit import release_ready,protocol as setup_protocol
            physics,features=None,[None]*35
            if env.physics_samples:
                save_physics(physics_path,env.physics_samples);_sync_file(physics_path)
                physics=dict(file=physics_path.name,sha256=digest(physics_path),samples=len(env.physics_samples))
                position=np.asarray([s["position_m"] for s in env.physics_samples])-env.origins[None]
                velocity=np.asarray([s["velocity_m_s"] for s in env.physics_samples])
                forces=np.asarray([s["forces_N"] for s in env.physics_samples])
                for i,row in enumerate(rows):
                    if not row["invalid"]:
                        features[i]=build_features(env.cfg,cells[i]["bucket_xy_m"],position[:,i],velocity[:,i],forces[:,i],env.dt,env.scoring_normals[i])
            returns=[]
            for i,(row,choice) in enumerate(zip(rows,original)):
                row["native_invalid"]=row["invalid"]
                if not choice["enabled"]:
                    require(row["invalid"]=="no_selected_setting" and env.preflight_records[i] is None
                        and env.qualifications[i]==dict(accepted=False,reason="inactive_split")
                        and env.path_checks[i] is None and env.path_reasons[i] is None
                        and row["route_success"] is False and row["success"] is False
                        and row["route_quality"] == 0 and features[i] is None,
                        "Empty TEST support was executed or fabricated a guard decline")
                    row["invalid"]=choice["decline_reason"]
                if not row["invalid"]:
                    require(choice["enabled"] and env.preflight_records[i]["accepted"] is True
                        and env.qualifications[i]["accepted"] is True and env.path_checks[i]["accepted"] is True
                        and env.path_reasons[i] is None and release_ready(row["settled"]),
                        "TEST release bypassed the unchanged measured guard gates")
                require(type(row["route_success"]) is bool and type(row["route_quality"]) is int
                    and row["route_quality"] in ((1,2) if row["route_success"] else (0,))
                    and (not row["invalid"] or not row["route_success"]),"Contradictory native TEST outcome")
                row.update(split="test",qualification=env.qualifications[i],path_rejection=env.path_reasons[i],
                    path_proxy=env.path_checks[i],scoring_board_normal=env.scoring_normals[i].tolist())
                returns.append(reward(row,features[i]))
            require(physics is not None or all(r["invalid"] for r in rows),"Released TEST request lacks physics evidence")
            packet=dict(schema=PACKET_SCHEMA,mode="assessment",batch=index+6,cohort=index+6,
                split="test",method=original[0]["method"],synthetic=self.synthetic,reservation_sha256=head,
                assessment_plan_fingerprint=self.plan["original_plan"]["fingerprint"],native_binding=self.binding,
                model_hashes=self.plan["model_hashes"],final_checkpoint_sha256=self.plan["final_checkpoint_sha256"],
                cells=cells,actions=actions,choices=copy.deepcopy(original),request_ids=[c["request_id"] for c in original],
                enabled=enabled,decline_reasons=[c["decline_reason"] for c in original],proposal_ik_seeds=seeds,
                rows=rows,rewards=returns,reward_features=features,home=env.home_readback,
                preflight_records=copy.deepcopy(env.preflight_records),release_step=env.release_step,score_end_step=env.steps,
                physics=physics,origins_m=env.origins.tolist(),setup_controller=setup_protocol(),startup_controls=controls,
                settling_history=raw["settling"],transit_history=env.transit_history,correction_history=env.correction_history,
                trajectory_m=raw["trajectory_m"],wall_s=time.monotonic()-began,process_cpu_s=time.process_time()-cpu,
                optimizer_updates=0,scope="Untouched original V4 TEST commands; no learning, search, repeat or retry.")
            from shot_benchmark_native import compressed_json
            receipt=compressed_json(target,packet);_sync_file(target)
            write_new_json(native/(stem+".receipt.json"),receipt)
            envelope=dict(schema=ENVELOPE_SCHEMA,plan_fingerprint=self.plan["fingerprint"],reservation_sha256=head,
                new_cohort=index,original_cohort=index+6,new_choices=copy.deepcopy(choices),original_choices=copy.deepcopy(original),
                packet=dict(file=target.name,sha256=digest(target)),native_binding_fingerprint=self.binding["fingerprint"],
                final_checkpoint_sha256=self.plan["final_checkpoint_sha256"],synthetic=self.synthetic)
            envelope["fingerprint"]=canonical_hash(envelope)
            envelope_path=native/(stem+".envelope.json");write_new_json(envelope_path,envelope)
            self._guard()
            return [dict(request_id=choice["request_id"],released=not bool(row["invalid"]),
                decline_reason=row["invalid"] or None,source_sha256=digest(envelope_path),captured=bool(row["route_success"]),
                clean=row["route_quality"]==2,**{"return":returns[i]}) for i,(choice,row) in enumerate(zip(choices,rows))]
        except BaseException as error:
            env.capture_trial=env.film_trial=False;env.phase=None
            partial=None
            if env.physics_samples and not physics_path.exists():
                try:
                    from presentation_recording import save_physics
                    save_physics(physics_path,env.physics_samples);_sync_file(physics_path)
                except Exception as telemetry_error:
                    p=native/(stem+".incomplete-physics.json")
                    write_new_json(p,dict(status="invalid_unscored_raw_evidence",error=repr(telemetry_error),samples=_raw_incomplete(env.physics_samples)))
                    partial=dict(file=p.name,sha256=digest(p))
            write_new_json(native/(stem+".failure.json"),dict(status="incomplete_no_retry",reservation_sha256=head,
                error=repr(error),physics_sha256=digest(physics_path) if physics_path.exists() else None,partial_physics=partial))
            raise


class ProductionComponents:
    """Explicit evaluation-only components, with model check before sole App."""

    def __init__(self,inputs,output,session,*,approved=False,synthetic=False,hooks=None):
        self.inputs,self.output,self.session=inputs,Path(output),session
        self.plan=validate_prepared(inputs["plan"])
        require(type(approved) is bool and type(synthetic) is bool and self.plan["synthetic"] is synthetic,
                "Explicit production provenance required")
        require(hooks is None or synthetic,"Production hooks cannot replace genuine native/model checks")
        self.approved,self.synthetic,self.hooks=approved,synthetic,hooks or {}
        self.app=self.env=self.producer=None;self.closed=False;self.started=False;self.device_ready=False

    def verify_frozen(self):
        if hasattr(self,"model_receipt_sha256"):
            require(digest(self.output/"model-check.json")==self.model_receipt_sha256,
                    "Frozen independent model receipt bytes changed")
        return self.inputs["verify_frozen"]()

    def initialize(self,publish_startup):
        require(not self.started and (self.approved or self.synthetic),"One separately approved initialization required; no retry")
        self.started=True;self.verify_frozen()
        from candidate_v4_model_check import check_models
        checker=self.hooks.get("model_check",check_models)
        request=self.inputs["model_request"]
        require(Path(request["receipt_path"]).resolve()==(self.output/"model-check.json").resolve(),"Model check receipt must be new output-owned")
        self.model_receipt=checker(request)
        from candidate_v4_model_check import validate_model_receipt
        validate_model_receipt(self.model_receipt,self.plan)
        if not self.synthetic:
            from candidate_continuation_native import guard_sources
            require(guard_sources()==self.model_receipt["original_guard_source_sha256"],
                    "Original shot guard/scorer source differs before native initialization")
        require(json.loads((self.output/"model-check.json").read_bytes())==self.model_receipt,
                "Independent model receipt differs from durable bytes")
        self.model_receipt_sha256=digest(self.output/"model-check.json")
        self.inputs["model_receipt_sha256"]=self.model_receipt_sha256
        self.verify_frozen()
        initializer=self.hooks.get("native_initialize",self._native_initialize)
        startup=initializer(publish_startup)
        self.verify_frozen()
        return startup

    def _native_initialize(self,publish_startup):
        i,p,m=self.inputs,self.inputs["paths"],self.inputs["manifest"]
        from candidate_continuation_production import verify_dependencies
        verify_dependencies(m["expected_runtime"])
        from presentation_recording import load_source
        load_source(i["mapped"](m["source_run"]),p["camera"])
        from continuous_policy import torch
        from candidate_continuation_native import validate_transit_limits
        cap_bytes=p["dropper_urdf"].read_bytes();urdf=ET.fromstring(cap_bytes)
        caps=torch.tensor([float(urdf.find(f".//joint[@name='joint{k+1}']/limit").get("effort")) for k in range(6)],device="cpu")
        validate_transit_limits(caps,cap_bytes)
        self.verify_frozen()
        from isaaclab.app import AppLauncher
        import argparse
        parser=argparse.ArgumentParser();AppLauncher.add_app_launcher_args(parser)
        args=parser.parse_args(["--headless","--enable_cameras","--device","cuda:0"])
        self.app=AppLauncher(args).app
        from candidate_continuation_scene import construct_scene,StartupRecorder
        self.env,self.bundle=construct_scene(output=self.output/"native-setup",assets=i["mapped"](m["assets_dir"]),
            source_run=i["mapped"](m["source_run"]),camera=p["camera"],device="cuda:0")
        self.device_ready=True
        from prepare_assets import chain
        self.env.continuous_links={n:chain(ET.fromstring(p[r].read_bytes())) for n,r in (("Arm10","dropper_urdf"),("Arm14","board_urdf"))}
        self.env.transit_limits=torch.tensor([float(urdf.find(f".//joint[@name='joint{k+1}']/limit").get("effort")) for k in range(6)],device=self.env.sim.device)
        startup=StartupRecorder(self.env,self.env.dest/"startup",publish=publish_startup).run()
        self.env.base_cells=copy.deepcopy(i["train"]["cells"]);self.env.spec["cells"]=copy.deepcopy(i["train"]["cells"])
        from candidate_continuation_native import make_binding,_frozen_identity,GuardedNativeProducer
        controls={name:digest(self.env.dest/name) for name in ("probes.json","rim-rays.json")}
        self.native_binding=make_binding(scene=i["values"]["scene"],geometry_fingerprint=_frozen_identity()["geometry_fingerprint"],
            camera_sha256=digest(p["camera"]),assets_sha256={"Arm10":digest(p["dropper_urdf"]),"Arm14":digest(p["board_urdf"])},
            poses_sha256=digest(p["poses"]),cells=i["train"]["cells"],canonical_home_rad=i["canonical_home"],
            supports=i["train_supports"],prepared_records=i["train_records"],startup_control_sha256=controls)
        write_new_json(self.output/"native-binding.json",self.native_binding)
        self.holder=GuardedNativeProducer(self.env,self.native_binding,i["train"]["cells"],i["train_supports"],i["train_records"],
            parent_checkpoint_path=p["parent_checkpoint"],camera_path=p["camera"],poses_path=p["poses"],
            assets_paths={"Arm10":p["dropper_urdf"],"Arm14":p["board_urdf"]},frozen_experiment_root=i["mapped"](m["frozen_experiment_root"]))
        self.producer=GuardedTestProducer(self.holder,self.plan,self.output,frozen_guard=self.verify_frozen,synthetic=False)
        return startup

    def execute_reserved(self,cohort,choices,reservation,*,journal_directory):
        require(self.started and not self.closed and self.producer is not None,"Native TEST adapter not initialized")
        return self.producer.execute_reserved(cohort,choices,reservation,journal_directory=journal_directory)

    def device_snapshot(self,reset):
        if not self.device_ready:return None
        from continuous_policy import torch
        torch.cuda.synchronize()
        if reset:torch.cuda.reset_peak_memory_stats();return None
        free,total=torch.cuda.mem_get_info()
        return dict(torch_peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
            torch_peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2,sampled_whole_device_occupied_mib=(total-free)/1024**2)

    def close_native(self):
        if self.closed:return
        self.closed=True
        if self.app is not None:self.app.close(skip_cleanup=True)

    def audit(self,summary):
        require(self.closed,"Native app must close before independent CPU audit")
        return self.inputs["audit"](self.output,self.plan,summary)
