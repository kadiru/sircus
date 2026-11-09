"""Synthetic all-rejected receipts test auditing, never native performance."""

import copy
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
import candidate_native_audit as audit
from continuous_feasibility import decode_action
from continuous_native_audit import expected_empty
from expanded_pose_menu import endpoint_targets
from joint_feasibility import preflight
from prepare_assets import chain


def signed(value):
    value = copy.deepcopy(value)
    value.pop("fingerprint", None)
    value["fingerprint"] = audit.canonical_hash(value)
    return value


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))


def compressed(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, separators=(",", ":"), allow_nan=False).encode()
    path.write_bytes(gzip.compress(raw, mtime=0))
    receipt = dict(file=path.name, sha256=audit.digest(path), bytes=path.stat().st_size,
                   uncompressed_sha256=hashlib.sha256(raw).hexdigest())
    save(str(path)+".receipt.json", receipt)
    return receipt


class PathReadbackArithmeticTests(unittest.TestCase):
    def setUp(self):
        self.path = dict(accepted=True,reasons=[],bucket_proxy_checked=True,
            scope="Finite interpolation proxy; not collision certification",
            checks=[dict(fraction=float(t),reasons=[],min_inter_arm_clearance_m=.05,
                         min_arm_table_clearance_m=.1,board_bottom_z_m=.25)
                    for t in np.linspace(0,1,16)])

    def test_only_tiny_numeric_metric_roundoff_is_accepted(self):
        saved=copy.deepcopy(self.path)
        saved["checks"][4]["min_inter_arm_clearance_m"]+=5e-8
        saved["checks"][10]["board_bottom_z_m"]-=4e-8
        self.assertTrue(audit.same_path(self.path,saved))
        saved["checks"][4]["min_inter_arm_clearance_m"]+=2e-7
        self.assertFalse(audit.same_path(self.path,saved))

    def test_flags_reasons_fraction_clock_and_sample_count_remain_exact(self):
        mutations=(lambda p:p.update(accepted=False),lambda p:p.update(accepted=1),
            lambda p:p.update(bucket_proxy_checked=False),lambda p:p.update(reasons=["new_rejection"]),
            lambda p:p["checks"][2].update(reasons=["new_rejection"]),
            lambda p:p["checks"][2].update(fraction=p["checks"][2]["fraction"]+1e-12),
            lambda p:p["checks"].pop(),lambda p:p.update(scope="Different scope"),
            lambda p:p["checks"][1].update(min_inter_arm_clearance_m=True))
        for mutate in mutations:
            saved=copy.deepcopy(self.path);mutate(saved)
            with self.subTest(mutate=mutate):
                self.assertFalse(audit.same_path(self.path,saved))

    def test_short_circuited_path_identity_cannot_be_fabricated(self):
        self.assertTrue(audit.same_path(None,None))
        self.assertFalse(audit.same_path(None,self.path))
        self.assertFalse(audit.same_path(self.path,None))


class NativeCandidateAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root/"run"
        self.run.mkdir()
        self.source = self.root/"source"
        self.source.mkdir()
        for name in (*audit.SCORE_FILES, "candidate_native_audit.py"):
            shutil.copyfile(SOURCE/name, self.source/name)
        self.patch = patch.object(audit, "SOURCE", self.source)
        self.patch.start(); self.addCleanup(self.patch.stop)
        self.hashes = {p.name:audit.digest(p) for p in self.source.glob("*.py")}
        self.scene = json.loads((SOURCE/"scene.json").read_text())
        self.cells = [dict(id=f"train-{i:03d}", split="train", bucket_xy_m=[.27+i*.001, -.1]) for i in range(35)]
        self.actions = [decode_action(c, [0.,.4,.35,.24,.5], "frozen-"+c["id"]) for c in self.cells]
        self.assets = self.root/"assets"; self.assets.mkdir()
        xml = '<robot name="synthetic">' + ''.join(
            f'<joint name="joint{i}" type="revolute"><origin xyz="0 0 {0.3 if i == 1 else 0}" rpy="0 0 0"/>'
            '<axis xyz="0 0 1"/><limit lower="-6.28" upper="6.28"/></joint>' for i in range(1,7)) + '</robot>'
        for filename in ("dropper.urdf", "board_arm.urdf"):
            (self.assets/filename).write_text(xml)
        self.links = {n:chain(ET.fromstring(xml)) for n in audit.ARMS}
        self.starts = {n:[0.]*6 for n in audit.ARMS}
        self.checkpoint = self.root/"parent.pt"
        self.checkpoint.write_bytes(b"synthetic-not-a-model")
        self.cp_sha = audit.digest(self.checkpoint)
        save(self.run/"selected-camera.json", dict(eye_m=[1,2,3],target_m=[0,0,0]))
        self.spec = signed(dict(scene=self.scene, cells=self.cells))
        save(self.run/"contract.json", self.spec)
        self.original = dict(commit="a"*40, archive_sha256="b"*64,
            geometry_fingerprint=self.spec["fingerprint"], camera_sha256=audit.digest(self.run/"selected-camera.json"),
            scoring_version="synthetic-audit-only")
        cohorts = [dict(method=m,request_count=35,actions=copy.deepcopy(self.actions),native_enabled=[True]*35,
                         decline_reasons=[None]*35,ik_seeds=[None]*35) for m in audit.METHODS]
        requests = [dict(request_index=i*35+j,cohort=i,method=m,slot=j,cell_id=c["id"],
                         action=copy.deepcopy(self.actions[j]),predeclined=False,decline_reason=None,candidate_id=None)
                    for i,m in enumerate(audit.METHODS) for j,c in enumerate(self.cells)]
        self.plan = signed(dict(schema=audit.PLAN_SCHEMA,seed=20261122,cells=self.cells,cohorts=cohorts,
            requests=requests,request_count=70,cohort_order=list(audit.METHODS),optimizer_updates=0,
            native_shots_observed=0,held_out_shot_files_read=0,release_authorized=False,
            input_bindings=dict(original=dict(checkpoint_sha256=self.cp_sha,split_manifest_sha256="c"*64),
                                assets_sha256={n:audit.digest(self.assets/f) for n,f in (("Arm10","dropper.urdf"),("Arm14","board_arm.urdf"))}),
            native_binding=dict(original_source=self.original,
                                original_scoring_source_sha256={n:self.hashes[n] for n in audit.SCORE_FILES})))
        self.plan_path = self.root/"comparison.json"; save(self.plan_path,self.plan)
        save(self.run/"experiment-binding.json",dict(comparison_sha256=audit.digest(self.plan_path),
            comparison_fingerprint=self.plan["fingerprint"],setup_controller=audit.setup_protocol(),
            original_source=self.original,source_sha256=self.hashes))
        probes=[]
        for kind,index in (("bucket",1),("rim",1),("table",2),("floor",3),("robot",4)):
            peaks=np.zeros((35,6));peaks[:,index]=1.
            probes.append(dict(kind=kind,all_passed=True,per_cell_passed=[True]*35,peak_N=peaks.tolist()))
        probes.append(dict(kind="foreign_cell_collision_filter",all_passed=True,peak_N=[0.]*6,foreign_ball_final_z_m=-2.))
        save(self.run/"probes.json",probes)
        rim=np.zeros((35,6));rim[:,1]=1.
        save(self.run/"rim-rays.json",dict(rays=[dict(azimuth_deg=0,first_contact_N=rim.tolist())],
            passed=[True]*35,directions=[[1.,0.]]*35,clearance_m=self.scene["ball"]["radius_m"]+.012))
        self.trials=[]
        manifests=[]
        for i,method in enumerate(audit.METHODS):
            t=self.make_trial(method,i);self.trials.append(t)
            name=f"trials/{method}-{i:05d}.json.gz"
            receipt=compressed(self.run/name,t)
            manifests.append(dict(method=method,file=name,sha256=receipt["sha256"],
                metrics=dict(wall_s=1.,cpu_s=.2,peak_rss_mib=20.,torch_peak_allocated_mib=0.)))
        self.result=dict(schema=audit.RESULT_SCHEMA,status="completed",comparison_fingerprint=self.plan["fingerprint"],
            optimizer_updates=0,native_requests=70,checkpoint_sha256_before=self.cp_sha,checkpoint_sha256_after=self.cp_sha,
            source_sha256=self.hashes,trials=manifests,
            control_files={n:audit.digest(self.run/n) for n in ("probes.json","rim-rays.json")})
        save(self.run/"result.json",self.result)

    def make_trial(self,method,index):
        histories=[];rows=[]
        for cell,action in zip(self.cells,self.actions):
            target=endpoint_targets(self.scene,cell,action)
            center=target["board_world"][:3,3].tolist();normal=target["board_normal_world"].tolist()
            drop=target["Arm10"][:3,3].tolist()
            normal_drift=float(np.rad2deg(np.arccos(np.clip(np.dot(normal,normal),-1,1))))
            history=[]
            for j in range(7):
                history.append(dict(iteration=0,measurement_start_step=300+j*60,physics_step=360+j*60,
                    board_center_m=center,board_normal=normal,dropper_flange_m=drop,
                    previous_board_center_m=center,previous_board_normal=normal,previous_dropper_flange_m=drop,
                    position_error_mm=0.,normal_error_deg=normal_drift,dropper_position_error_mm=0.,
                    position_drift_mm=0.,normal_drift_deg=normal_drift,dropper_drift_mm=0.,joint_speed_rad_s=0.,
                    arms={n:dict(position_rad=[0.]*6,velocity_rad_s=[0.]*6,target_rad=[0.]*6) for n in audit.ARMS},
                    articulation_joint_names=dict(Arm10=[f"joint{k}" for k in range(1,7)]+["finger_left_joint","finger_right_joint"],Arm14=[f"joint{k}" for k in range(1,7)]),
                    all_articulation_joint_velocity=dict(Arm10=[0.]*8,Arm14=[0.]*6)))
            histories.append(history)
            prepared=preflight(self.scene,cell,action,self.links,self.starts)
            self.assertEqual(prepared["reason"],"outer_link_length_bound")
            row=dict(action,**expected_empty(self.scene,cell,1/240),cell_id=cell["id"],split="train",
                invalid=prepared["reason"],qualification=prepared["qualification"],path_proxy=prepared["path"],
                path_rejection=None,settled=history[-1],settled_board_center_m=center,
                settled_board_normal=normal,scoring_board_normal=normal,release_ball_center_m=target["release_center_m"].tolist())
            rows.append(row)
        arrays={n:np.zeros((35,6)).tolist() for n in audit.ARMS}
        state={n:{k:copy.deepcopy(arrays[n]) for k in ("position_rad","velocity_rad_s","target_rad")} for n in audit.ARMS}
        gains={n:{k:np.ones((35,6)).tolist() for k in ("effort","stiffness","damping")} for n in audit.ARMS}
        transit=[dict(phase="start",step=0,state=state,original_gains=gains),dict(phase="tail",step=60,state=state),
                 dict(phase="restored_original_hold",step=300,state=state,restored_gains=gains)]
        return dict(schema="sircus-shot-benchmark-native-trial-v1",comparison_fingerprint=self.plan["fingerprint"],
            mode=method,batch=index,optimizer_updates=0,cells=self.cells,actions=copy.deepcopy(self.actions),enabled=[True]*35,
            proposal_ik_seeds=[None]*35,source=self.original,split_manifest_sha256="c"*64,
            learner_packet=None,training_request_order=None,rewards=[-1.]*35,reward_features=[None]*35,
            home=dict(protocol="canonical-episode-home-reset-before-visible-drive-v1",episode_reset=True,
                path_eligible=[True]*35,path_checks=[dict(accepted=True)]*35,physics_step=0,
                target_joint_rad=arrays,measured_joint_rad=arrays,measured_joint_velocity_rad_s=arrays),
            rows=rows,release_step=720,score_end_step=720,physics=None,origins_m=np.zeros((35,3)).tolist(),
            setup_controller=audit.setup_protocol(),settling_history=histories,transit_history=transit,correction_history=[])

    def verify(self):
        return audit.audit_run(self.run,self.plan_path,checkpoint_path=self.checkpoint,assets_path=self.assets)

    def rebind_trial(self,index,mutate):
        trial=copy.deepcopy(self.trials[index]);mutate(trial)
        receipt=self.result["trials"][index]
        saved=compressed(self.run/receipt["file"],trial)
        receipt["sha256"]=saved["sha256"]
        save(self.run/"result.json",self.result)

    def test_complete_synthetic_all_rejected_evidence_keeps_all_70_requests(self):
        result=self.verify()
        self.assertEqual(result["status"],"passed")
        self.assertEqual(result["native_requests"],70)
        self.assertEqual(result["rescored_releases"],0)
        self.assertEqual([result["methods"][m]["counts"]["rejected"] for m in audit.METHODS],[35,35])
        self.assertTrue(result["startup_controls"]["per_ray_force_gates_recomputed"])
        self.assertTrue(all(t["settlement"]["full_articulation_velocity_gate_verified"] for t in result["trials"]))

    def test_policy_updates_changed_checkpoint_and_incomplete_budget_fail(self):
        for key,value in (("optimizer_updates",1),("native_requests",69),("checkpoint_sha256_after","0"*64)):
            before=copy.deepcopy(self.result);self.result[key]=value;save(self.run/"result.json",self.result)
            with self.subTest(key=key),self.assertRaises(ValueError):self.verify()
            self.result=before
        save(self.run/"result.json",self.result)
        self.checkpoint.write_bytes(b"changed weights")
        with self.assertRaisesRegex(ValueError,"checkpoint changed"):self.verify()

    def test_rehashed_action_or_heldout_membership_cannot_replace_frozen_request(self):
        for mutate in (lambda t:t["actions"][0].update(gap_m=.3),lambda t:t["cells"][0].update(split="test"),
                       lambda t:t.update(learner_packet={}),lambda t:t.update(training_request_order=[0])):
            with self.subTest(mutate=mutate):
                self.rebind_trial(0,mutate)
                with self.assertRaises(ValueError):self.verify()

    def test_rehashed_geometry_admission_cannot_hide_rejected_request(self):
        self.rebind_trial(0,lambda t:t["rows"][0]["qualification"].update(accepted=True))
        with self.assertRaisesRegex(ValueError,"geometry qualification"):self.verify()

    def test_original_score_and_full_articulation_velocity_are_recomputed(self):
        self.rebind_trial(0,lambda t:t["rewards"].__setitem__(0,0.))
        with self.assertRaisesRegex(ValueError,"reward"):self.verify()
        self.rebind_trial(0,lambda t:t["settling_history"][0][-1]["all_articulation_joint_velocity"]["Arm10"].__setitem__(6,.02))
        with self.assertRaisesRegex(ValueError,"joint-speed gate"):self.verify()

    def test_resource_and_startup_contact_tampering_fail_closed(self):
        self.result["trials"][0]["metrics"]["wall_s"]=-1.;save(self.run/"result.json",self.result)
        with self.assertRaisesRegex(ValueError,"resource measurement"):self.verify()
        self.result["trials"][0]["metrics"]["wall_s"]=1.
        probes=audit.read(self.run/"probes.json");probes[0]["peak_N"][0][1]=0.
        save(self.run/"probes.json",probes)
        self.result["control_files"]["probes.json"]=audit.digest(self.run/"probes.json")
        save(self.run/"result.json",self.result)
        with self.assertRaisesRegex(ValueError,"force gates"):self.verify()

    def test_unlisted_trial_source_or_camera_changes_fail(self):
        (self.run/"trials/extra.json.gz").write_bytes(b"unlisted")
        with self.assertRaisesRegex(ValueError,"Unlisted"):self.verify()
        (self.run/"trials/extra.json.gz").unlink()
        save(self.run/"selected-camera.json",dict(eye_m=[4,5,6]))
        with self.assertRaisesRegex(ValueError,"camera changed"):self.verify()

    def test_plan_cannot_enable_heldout_or_hide_predecline_denominator(self):
        for mutate in (lambda p:p["cells"][0].update(split="validation"),lambda p:p.update(request_count=69),
                       lambda p:p["cohorts"][0]["ik_seeds"].__setitem__(0,{})):
            plan=copy.deepcopy(self.plan);mutate(plan);plan=signed(plan)
            with self.subTest(mutate=mutate),self.assertRaises(ValueError):audit.validate_plan(plan)

    def test_module_import_does_not_initialize_policy_or_simulator(self):
        code=("import sys;sys.path.insert(0,%r);import candidate_native_audit;"
              "assert 'torch' not in sys.modules;assert not any(n.startswith(('isaacsim','isaaclab')) for n in sys.modules)") % str(SOURCE)
        subprocess.run([sys.executable,"-c",code],check=True)


if __name__ == "__main__":
    unittest.main()
