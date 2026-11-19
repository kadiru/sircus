"""Mock startup instrumentation, with original numeric probe method bodies.

Only extracted function ASTs run on arrays and a fake step counter. No native
classes, simulator, device, subprocess, environment or learned model is created.
"""

import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import candidate_continuation_scene as subject
from candidate_continuation_journal import write_new_json
from candidate_continuation_ledger import canonical_hash

SOURCE=Path(__file__).resolve().parents[1]


def method_body(file,class_name,name):
    """Compile one numeric method; never import/execute native module setup."""
    tree=ast.parse((SOURCE/file).read_text())
    klass=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==class_name)
    function=copy.deepcopy(next(n for n in klass.body if isinstance(n,ast.FunctionDef) and n.name==name))
    namespace=dict(np=np,atomic=write_new_json)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),str(SOURCE/file),"exec"),namespace)
    return namespace[name]


class MockProbeEnv:
    exposed_rim_probes=method_body("film_cylinder_farm.py","FilmShots","exposed_rim_probes")
    probes=method_body("parallel_runtime.py","ParallelShots","probes")

    def __init__(self,directory,scene,cells,*,rims=1):
        self.dest=directory;self.cfg=copy.deepcopy(scene);self.n=35
        self.origins=np.asarray([[i*3.,0.,0.] for i in range(35)])
        self.bucket_xy=np.asarray([c["bucket_xy_m"] for c in cells])
        self.foreign_probe_index=7;self.probe_positions=dict(table=[1.5,-.6,.25],floor=[1.8,-.65,-.5],robot=[0.,0.,.75])
        self.ball=types.SimpleNamespace(data=types.SimpleNamespace(root_pos_w=np.zeros((35,3)),root_lin_vel_w=np.zeros((35,3))))
        self.steps=0;self.placements=0;self.rims=rims;self.time=0.;self.force_override=None

    def set_balls(self,xyz,world=False):
        self.ball.data.root_pos_w=np.asarray(xyz).copy()+(0. if world else self.origins)
        self.ball.data.root_lin_vel_w=np.zeros((35,3));self.placements+=1

    def tick(self):
        self.steps+=1;self.time+=.001
        if self.placements==self.rims+6:self.ball.data.root_pos_w[0,2]=-2.

    def forces(self):
        force=np.zeros((35,6))
        i=self.placements-1
        if i==self.rims-1:force[:,1]=1.
        elif self.rims <= i < self.rims+5:
            target=[1,1,2,3,4][i-self.rims];force[:,target]=1.
        if self.force_override is not None:self.force_override(self,force)
        return force

    def assert_buckets_fixed(self):return 0.


class SceneStartupTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.scene=json.loads((SOURCE/"scene.json").read_text())
        self.cells=[dict(id=f"train-{i:03d}",split="train",bucket_xy_m=[.6+i*.001,-.1]) for i in range(35)]

    def run_startup(self,rims=1):
        env=MockProbeEnv(self.root,self.scene,self.cells,rims=rims)
        published=[]
        recorder=subject.StartupRecorder(env,self.root/"startup",publish=lambda records,parking:published.append((records,parking)),
            clock=lambda:env.time,cpu=lambda:env.time*.5)
        receipt=recorder.run()
        return env,receipt,published

    def audit(self,env):return subject.audit_startup(self.root/"startup",scene=self.scene,cells=self.cells,origins=env.origins)

    def rewrite_receipt(self,receipt):
        value=copy.deepcopy(receipt);value.pop("fingerprint",None);value["fingerprint"]=canonical_hash(value)
        (self.root/"startup/receipt.json").write_text(json.dumps(value,allow_nan=False))
        return value

    def mutate_cohort(self,receipt,index,mutate):
        receipt=copy.deepcopy(receipt);entry=receipt["evidence"][index]
        path=self.root/"startup"/entry["file"]
        raw=json.loads(path.read_text());mutate(raw)
        path.write_text(json.dumps(raw,allow_nan=False));entry["sha256"]=subject.digest(path)
        return self.rewrite_receipt(receipt)

    def test_original_ast_prefix_contains_no_launcher_or_training_dispatch(self):
        tree=subject.scene_program((SOURCE/"film_cylinder_farm.py").read_text())
        compile(tree,"mock-original-scene-prefix","exec")
        function=tree.body[-1]
        self.assertEqual(function.name,"_construct_original_scene")
        text=ast.unparse(function)
        self.assertIn("np.array_equal",text)
        self.assertNotIn("AppLauncher(",text)
        self.assertNotIn("run_benchmark(",text)
        self.assertNotIn("run_study(",text)
        self.assertNotIn("env.exposed_rim_probes(",text)

    def test_missing_origin_boundary_cannot_include_a_training_tail(self):
        raw="import isaaclab.sim as sim_utils\ndef main():\n    env = None\n    if True:\n        train_forever()\n"
        with self.assertRaises(ValueError):subject.scene_program(raw)

    def test_original_numeric_helpers_minimum_schedule_composes_with_actual_auditor(self):
        env,receipt,published=self.run_startup()
        checked=self.audit(env)
        self.assertEqual(checked["status"],"passed")
        self.assertEqual((checked["cohorts"],checked["initialized_trajectories"],checked["physics_steps"]),(7,245,1488))
        self.assertEqual(env.steps,1488);self.assertEqual(env.placements,8)
        self.assertEqual(len(receipt["parking"]),1)
        self.assertEqual(len(published),7)
        self.assertTrue(all(type(row["initialized_states_sha256"]) is str for row in receipt["records"]))
        self.assertNotIn("set_balls",vars(env));self.assertNotIn("tick",vars(env))

    def test_maximum_eight_ray_schedule_retains_original_limits(self):
        env,receipt,_=self.run_startup(8)
        checked=self.audit(env)
        self.assertEqual((checked["cohorts"],checked["initialized_trajectories"],checked["physics_steps"]),(14,490,1824))
        self.assertEqual([r["azimuth_deg"] for r in receipt["records"] if r["kind"]=="rim_ray"],[0,90,180,270,45,135,225,315])
        self.assertFalse(receipt["records"][6]["passed"]);self.assertTrue(receipt["records"][7]["passed"])

    def test_rehashed_clock_gap_does_not_pass(self):
        env,receipt,_=self.run_startup()
        self.mutate_cohort(receipt,1,lambda r:r.update(start_step=r["start_step"]+1))
        with self.assertRaisesRegex(ValueError,"clock/sample"):self.audit(env)

    def test_rehashed_placement_shift_does_not_pass(self):
        env,receipt,_=self.run_startup()
        entry=receipt["evidence"][0];rawpath=self.root/"startup"/entry["file"]
        raw=json.loads(rawpath.read_text());path=self.root/"startup"/raw["initialized_file"]
        initial=json.loads(path.read_text());initial["position_m"][0][0]+=.01
        path.write_text(json.dumps(initial));raw["initialized_sha256"]=subject.digest(path)
        rawpath.write_text(json.dumps(raw));entry["sha256"]=subject.digest(rawpath)
        receipt["records"][0]["initialized_states_sha256"]=raw["initialized_sha256"]
        self.rewrite_receipt(receipt)
        with self.assertRaisesRegex(ValueError,"placement/contact"):self.audit(env)

    def test_rehashed_contact_attribution_cannot_keep_success(self):
        env,receipt,_=self.run_startup()
        self.mutate_cohort(receipt,1,lambda r:r["forces_N"][0][0].__setitem__(1,0.))
        # One removed sample still leaves a contact peak, so remove the entire target channel.
        raw=json.loads((self.root/"startup/cohort-0001.json").read_text())
        for force in raw["forces_N"]:force[0][1]=0.
        path=self.root/"startup/cohort-0001.json";path.write_text(json.dumps(raw))
        receipt["evidence"][1]["sha256"]=subject.digest(path);self.rewrite_receipt(receipt)
        with self.assertRaisesRegex(ValueError,"placement/contact"):self.audit(env)

    def test_no_hidden_parking_probe_or_repeated_cohort(self):
        env,receipt,_=self.run_startup()
        parking=receipt["parking"][0];path=self.root/"startup"/parking["file"]
        raw=json.loads(path.read_text());raw["forces_N"]=[np.zeros((35,6)).tolist()]
        path.write_text(json.dumps(raw));parking["sha256"]=subject.digest(path);self.rewrite_receipt(receipt)
        with self.assertRaisesRegex(ValueError,"hidden probe"):self.audit(env)

    def test_extra_partial_or_unknown_evidence_cannot_be_hidden(self):
        env,_,_=self.run_startup();(self.root/"startup/extra.json").write_text("{}")
        with self.assertRaisesRegex(ValueError,"Unlisted"):self.audit(env)

    def test_initialized_state_count_cannot_omit_one_ball(self):
        env,receipt,_=self.run_startup()
        receipt["records"][0]["placements"]=34
        self.rewrite_receipt(receipt)
        with self.assertRaisesRegex(ValueError,"35 initialized"):self.audit(env)

    def test_initialized_velocity_is_independently_checked_after_rehash(self):
        env,receipt,_=self.run_startup()
        entry=receipt["evidence"][0];rawpath=self.root/"startup"/entry["file"]
        raw=json.loads(rawpath.read_text());path=self.root/"startup"/raw["initialized_file"]
        initial=json.loads(path.read_text());initial["velocity_m_s"][0][2]=.1
        path.write_text(json.dumps(initial));raw["initialized_sha256"]=subject.digest(path)
        rawpath.write_text(json.dumps(raw));entry["sha256"]=subject.digest(rawpath)
        receipt["records"][0]["initialized_states_sha256"]=raw["initialized_sha256"]
        self.rewrite_receipt(receipt)
        with self.assertRaisesRegex(ValueError,"placement/contact"):self.audit(env)

    def test_source_identity_mismatch_cannot_pass(self):
        env,receipt,_=self.run_startup()
        receipt["helper_sha256"]["parallel_runtime.py"]="0"*64;self.rewrite_receipt(receipt)
        with self.assertRaisesRegex(ValueError,"source identity"):self.audit(env)

    def test_costs_must_match_raw_cohort_evidence(self):
        env,receipt,_=self.run_startup()
        self.mutate_cohort(receipt,2,lambda raw:raw.update(wall_s=raw["wall_s"]+1.))
        with self.assertRaisesRegex(ValueError,"cost source"):self.audit(env)

    def test_exception_restores_inherited_methods_and_preserves_partial_evidence(self):
        env=MockProbeEnv(self.root,self.scene,self.cells)
        env.force_override=lambda obj,force:force.__setitem__((0,0),float("nan"))
        recorder=subject.StartupRecorder(env,self.root/"startup",clock=lambda:env.time,cpu=lambda:env.time*.5)
        with self.assertRaisesRegex(ValueError,"Nonfinite"):recorder.run()
        self.assertNotIn("set_balls",vars(env));self.assertNotIn("tick",vars(env))
        self.assertTrue((self.root/"startup/failure.json").exists())
        self.assertTrue((self.root/"startup/incomplete-current.json").exists())
        self.assertFalse((self.root/"startup/receipt.json").exists())

    def test_instrumentation_rejects_preexisting_tick_overrides(self):
        env=MockProbeEnv(self.root,self.scene,self.cells);env.tick=lambda:None
        recorder=subject.StartupRecorder(env,self.root/"startup")
        with self.assertRaisesRegex(ValueError,"inherited methods"):recorder.run()


if __name__=="__main__":unittest.main()
