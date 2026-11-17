"""Invented evidence exercises the independent hookup; no native execution."""
import copy
from contextlib import contextmanager
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import location_comparison_verify as verifier
from location_comparison_native import NativeEngine, PARENT_SHA
from location_comparison_bounds import Bounds
from location_comparison_mock import FakeClock
from location_comparison_runtime import ComparisonRunner, Journal
from test_location_comparison_native import FakeDependencies, FakeEnvironment, sha
from test_location_comparison_runtime import synthetic_plan
from test_location_comparison_audit import reseal, first


class FullVerificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from camera_options import preview_origins
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.plan = synthetic_plan()
        cls.manifest = dict(fingerprint="a"*64, files=dict(parent=dict(path=str(cls.root/"parent.fake"), sha256=PARENT_SHA)))
        cls.scene = dict(simulation=dict(physics_dt_s=1/240))
        cls.experiment = dict(source=dict(geometry_fingerprint="b"*64, synthetic=True))
        env, deps = FakeEnvironment(), FakeDependencies()
        env.origins = preview_origins()
        original_cells = copy.deepcopy(env.spec["cells"])
        clock = FakeClock()
        bounds = Bounds(cls.plan, clock, lambda:dict(free_storage_bytes=10*1024**3))
        engine = NativeEngine(env, cls.manifest, cls.experiment, cls.root, cls.plan,
                              guard=bounds.check, dependencies=deps, clock=clock)
        cls.trace = ComparisonRunner(cls.plan, engine, bounds, Journal(), clock=clock).run()
        engine.close()
        assert cls.trace["events"][-1]["payload"]["status"] == "complete_audit_pending"
        (cls.root/"native/contract.json").write_text(json.dumps(dict(scene=cls.scene,
            fingerprint=cls.experiment["source"]["geometry_fingerprint"], cells=original_cells)))
        cls.manifest["files"]["source_contract"] = dict(path=str(cls.root/"native/contract.json"),
            sha256=verifier.sha(cls.root/"native/contract.json"))
        for name in ("poses", "camera"):
            file = cls.root/f"{name}.json"
            file.write_text("{}")
            cls.manifest["files"][name] = dict(path=str(file), sha256=verifier.sha(file))
        def load_checkpoint(path, receipt, plan, experiment):
            saved = json.loads(path.read_text())
            if saved["metadata"]["state"] != receipt["state"]:
                raise ValueError("Invented checkpoint metadata changed")
            return dict(policy=object(), state=saved["metadata"]["state"],
                        actor_hash=saved["metadata"]["state"]["weight_hash"],optimizer_config={"synthetic":True})
        cls.checks = dict(
            inputs=lambda manifest, plan:dict(experiment=cls.experiment, scene=cls.scene, files=manifest["files"]),
            parent=lambda path, experiment:dict(weight_hash=sha(0), optimizer_hash=sha(292),optimizer_config={"synthetic":True}),
            checkpoint=load_checkpoint,
            policy=lambda policy, packet, cells:dict(maximum_logp_error=0., paid_samples=len(cells)),
            geometry=lambda trial, scene, links:dict(unchanged=sum(trial["enabled"])),
            native=lambda *args:dict(synthetic=True, full_rate_reader="injected"),
            startup=lambda *args, **kwargs:dict(status="passed", synthetic=True))
        (cls.root/"comparison-trace.json").write_text(json.dumps(cls.trace))
        cls.child = dict(status="synthetic_complete_pending_audit",synthetic=True, app_closed=True,error=None,
            runtime_fingerprint=cls.manifest["fingerprint"],comparison_fingerprint=cls.plan["fingerprint"],
            session_id="f"*32,trace_sha256=verifier.sha(cls.root/"comparison-trace.json"))
        (cls.root/"child-exit.json").write_text(json.dumps(cls.child))
        from location_comparison_watchdog import signed, SCHEMA
        cls.exit = signed(dict(schema=SCHEMA,runtime_fingerprint=cls.manifest["fingerprint"],
            comparison_fingerprint=cls.plan["fingerprint"],session_id=cls.child["session_id"],
            status="native_exit_zero_independent_audit_required",exit_code=0,reason=None,
            lock_release_safe=True,native_application_starts=1,
            final_snapshot=cls.trace["events"][-1]["payload"]["bounds"]))
        (cls.root/"comparison-watchdog-exit.json").write_text(json.dumps(cls.exit))
        cls.report = cls.run_audit()

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    @classmethod
    def run_audit(cls, trace=None, checks=None):
        actual = cls.trace if trace is None else trace
        (cls.root/"comparison-trace.json").write_text(json.dumps(actual))
        child = dict(cls.child, trace_sha256=verifier.sha(cls.root/"comparison-trace.json"))
        (cls.root/"child-exit.json").write_text(json.dumps(child))
        try:
            return verifier.audit(cls.root, cls.manifest, cls.plan, actual,
                                  synthetic=True, checks=cls.checks if checks is None else checks)
        finally:
            (cls.root/"comparison-trace.json").write_text(json.dumps(cls.trace))
            (cls.root/"child-exit.json").write_text(json.dumps(cls.child))

    @contextmanager
    def change_file(self, filename, raw):
        path = self.root/filename
        old = path.read_bytes() if path.exists() else None
        path.write_bytes(raw)
        try:
            yield path
        finally:
            if old is None:
                path.unlink()
            else:
                path.write_bytes(old)

    def test_complete_fake_bridge_audits_all_states_requests_and_updates(self):
        self.assertEqual(self.report["status"], "synthetic_verified")
        self.assertEqual(self.report["paid_policy_samples"], 18520)
        self.assertEqual(self.report["terminal_batches"], 539)
        self.assertEqual(self.report["update_artifacts"], 128)
        # 131 sampling states + three initial + eight milestones + two finals.
        self.assertEqual(self.report["checkpoint_count"], 144)
        for key in ("checkpoint_bytes_independently_verified", "model_likelihoods_independently_verified", "native_physics_verified"):
            self.assertIs(self.report[key], False)

    def test_mock_readers_cannot_certify_a_native_run(self):
        changed = copy.deepcopy(self.trace)
        changed["mode"] = "native"
        with self.assertRaisesRegex(ValueError, "Injected checks"):
            verifier.audit(self.root, self.manifest, self.plan, changed, checks=self.checks)
        with self.assertRaisesRegex(ValueError, "evidence mode"):
            verifier.audit(self.root, self.manifest, self.plan, self.trace)
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES":"0"}):
            with self.assertRaisesRegex(ValueError, "CPU-only"):
                verifier.audit(self.root, self.manifest, self.plan, changed)

    def test_checkpoint_byte_change_and_initial_parent_mismatch_fail(self):
        name = first(self.trace,"checkpoint")["receipt"]["file"]
        with self.change_file(name, b"changed"):
            with self.assertRaisesRegex(ValueError, "bytes changed"):
                self.run_audit()
        checks = dict(self.checks, parent=lambda *args:dict(weight_hash="e"*64, optimizer_hash=sha(292),optimizer_config={"synthetic":True}))
        with self.assertRaisesRegex(ValueError, "original seed22"):
            self.run_audit(checks=checks)

    def test_raw_native_byte_change_fails(self):
        name = first(self.trace,"terminal")["evidence"]["file"]
        with self.change_file(name,b"changed"):
            with self.assertRaisesRegex(ValueError, "bytes changed"):
                self.run_audit()

    def test_raw_result_cannot_be_relabelled_after_resealing(self):
        changed = copy.deepcopy(self.trace)
        terminal = first(changed,"terminal")
        path = self.root/terminal["evidence"]["file"]
        packet = verifier.read(path)
        packet["raw_rows"][0]["route_success"] = False
        raw = gzip.compress(json.dumps(packet).encode(), mtime=0)
        terminal["evidence"]["sha256"] = hashlib.sha256(raw).hexdigest()
        reseal(changed)
        with self.change_file(terminal["evidence"]["file"],raw):
            with self.assertRaisesRegex(ValueError, "raw native rows"):
                self.run_audit(changed)

    def test_physics_bytes_and_startup_failure_rejected(self):
        terminal = first(self.trace,"terminal")
        name = "native/"+terminal["evidence"]["physics"]["file"]
        with self.change_file(name,b"changed"):
            with self.assertRaisesRegex(ValueError, "bytes changed"):
                self.run_audit()
        checks = dict(self.checks, startup=lambda *a,**kw:dict(status="failed"))
        with self.assertRaisesRegex(ValueError, "startup physics"):
            self.run_audit(checks=checks)

    def test_update_artifact_counter_change_rejected(self):
        update = first(self.trace,"update")
        name = f"updates/{update['model']}-{update['after']['updates']:05d}.json"
        raw = verifier.read(self.root/name)
        raw["state_after"]["optimizer_steps"] += 1
        with self.change_file(name,json.dumps(raw).encode()):
            with self.assertRaisesRegex(ValueError, "PPO update evidence"):
                self.run_audit()

    def test_update_epoch_budget_and_actor_hash_are_checked(self):
        update = first(self.trace,"update")
        name = f"updates/{update['model']}-{update['after']['updates']:05d}.json"
        raw = verifier.read(self.root/name)
        raw["statistics"]["epochs"].pop()
        with self.change_file(name,json.dumps(raw).encode()):
            with self.assertRaisesRegex(ValueError,"four-pass"):
                self.run_audit()
        raw = verifier.read(self.root/name)
        raw["statistics"]["actor_after"] = "a"*64
        with self.change_file(name,json.dumps(raw).encode()):
            with self.assertRaisesRegex(ValueError,"actor chain"):
                self.run_audit()

    def test_contract_copy_and_watchdog_exit_are_independent_gates(self):
        name = "native/contract.json"
        raw = verifier.read(self.root/name)
        raw["cells"][0]["bucket_xy_m"][0] += .01
        with self.change_file(name,json.dumps(raw).encode()):
            with self.assertRaisesRegex(ValueError,"Constructed native scene"):
                self.run_audit()
        raw = copy.deepcopy(self.exit)
        raw["exit_code"] = 1
        from location_comparison_watchdog import signed
        raw.pop("fingerprint")
        with self.change_file("comparison-watchdog-exit.json",json.dumps(signed(raw)).encode()):
            with self.assertRaisesRegex(ValueError,"one-start clean native exit"):
                self.run_audit()

    def test_extra_partial_artifact_prevents_complete_claim(self):
        with self.change_file("comparison-native/trial-99999-failure.json.gz",b"partial"):
            with self.assertRaisesRegex(ValueError, "Unlisted, partial"):
                self.run_audit()

    def test_incomplete_or_missing_reader_does_not_pass(self):
        trace = copy.deepcopy(self.trace)
        trace["events"] = trace["events"][:2]
        with self.assertRaisesRegex(ValueError, "Incomplete comparison"):
            self.run_audit(trace)
        checks = dict(self.checks)
        del checks["native"]
        with self.assertRaisesRegex(ValueError, "Incomplete verifier"):
            self.run_audit(checks=checks)

    def test_json_duplicate_nonfinite_and_symlink_guards(self):
        for raw in (b'{"a":1,"a":2}', b'{"x":NaN}'):
            with self.change_file("bad.json",raw):
                with self.assertRaises(ValueError): verifier.read(self.root/"bad.json")
        with self.assertRaisesRegex(ValueError,"Escaped"):
            verifier.artifact(self.root,"../escape")
        path = self.root/"linked.json"
        path.symlink_to(self.root/"poses.json")
        try:
            with self.assertRaisesRegex(ValueError,"linked"):
                verifier.artifact(self.root,"linked.json")
        finally:
            path.unlink()


class IndependentLikelihoodTests(unittest.TestCase):
    def test_default_reader_composition_uses_original_auditors_without_starting_them(self):
        from seed22_continuation_audit import verify_native, verify_geometry
        from candidate_continuation_scene import audit_startup
        checks = verifier._default_checks()
        self.assertIs(checks["native"],verify_native)
        self.assertIs(checks["startup"],audit_startup)
        self.assertIs(checks["checkpoint"],verifier.load_model)
        self.assertIs(checks["policy"],verifier.verify_policy)
        from test_location_comparison_native import cells
        trial = dict(cells=cells(count=35),synthetic=True)
        unchanged = copy.deepcopy(trial)
        with mock.patch("seed22_continuation_audit.verify_geometry",return_value={"synthetic":True}) as reader:
            self.assertEqual(checks["geometry"](trial,{},{}),{"synthetic":True})
            self.assertEqual(len(reader.call_args.args[0]["observations"]),35)
        self.assertEqual(trial,unchanged)

    def test_paid_partial_cohort_gaussian_and_deterministic_checks(self):
        import numpy as np
        from continuous_policy import build_observations
        from test_location_comparison_native import cells
        class Tensor(np.ndarray):
            def numpy(self): return np.asarray(self)
        def tensor(value,dtype=None): return np.asarray(value,dtype=dtype).view(Tensor)
        @contextmanager
        def no_grad(): yield
        torch = SimpleNamespace(as_tensor=tensor,tensor=tensor,float32=np.float32,no_grad=no_grad)
        # This invented constant distribution is not seed22 or a trained model.
        class FakePolicy:
            def distribution(self, observations):
                return SimpleNamespace(loc=tensor(np.zeros_like(observations)),scale=tensor(np.ones_like(observations)))
            def log_prob(self, observations, latent):
                return tensor((-.5*latent**2-.5*np.log(2*np.pi)
                    - 2*(np.log(2)-latent-np.logaddexp(0.,-2*latent))).sum(axis=-1))
        context = cells(count=3,split="test")
        policy = FakePolicy()
        obs = torch.tensor(build_observations(context))
        latent = torch.tensor(np.zeros((3,5)))
        logp = policy.log_prob(obs,latent).tolist()
        proposal = dict(deterministic=True, rows=[dict(observation=o,latent=[0.]*5,
            actions_normalized=[0.]*5,logp=p) for o,p in zip(obs.tolist(),logp)])
        with mock.patch.dict(sys.modules,{"torch":torch}):
            report = verifier.verify_policy(policy,proposal,context)
            self.assertEqual(report["paid_samples"],3)
            proposal["rows"][0]["logp"] += .01
            with self.assertRaisesRegex(ValueError,"likelihood"):
                verifier.verify_policy(policy,proposal,context)
            proposal["rows"][0]["logp"] -= .01
            proposal["rows"][0]["observation"][0] += .01
            with self.assertRaisesRegex(ValueError,"inputs"):
                verifier.verify_policy(policy,proposal,context)


class CheckpointReaderTests(unittest.TestCase):
    """Exercise the real validation branches with fake CPU tensor/model readers."""
    def setUp(self):
        # Import before injecting torch: no real checkpoint or policy is loaded.
        import continuous_policy
        import candidate_continuation_checkpoint
        self.plan = dict(fingerprint="a"*64)
        self.experiment = dict(source=dict(geometry_fingerprint="b"*64),
            protocol=dict(fingerprint="c"*64,domain=dict(fingerprint="d"*64)))
        optimizer = dict(state={0:dict(step=548,exp_avg=[0.]),1:dict(step=548,exp_avg=[0.])},
                         param_groups=[dict(lr=.0003)])
        self.state = dict(model_id="broader",weight_hash=sha(64), optimizer_hash=sha(optimizer),
            updates=137,attempts=18960,optimizer_steps=548,parent_checkpoint_sha256=PARENT_SHA,
            action_rng_hash=sha([1,2]),optimizer_rng_hash=sha(dict(cpu=[3,4],cuda=[])))
        self.saved = dict(synthetic=False,seed=20261122,comparison_fingerprint=self.plan["fingerprint"],
            parent_checkpoint_sha256=PARENT_SHA,state=copy.deepcopy(self.state),weight_hash=sha(64),
            actor_weight_hash=sha("actor64"),state_dict=dict(weight=[0.,1.]),optimizer=optimizer,
            action_rng_state=[1,2],optimizer_rng_state=dict(cpu=[3,4],cuda=[]))
        self.model = SimpleNamespace(parameters=lambda:iter([object(),object()]))
        class Finite:
            def all(self): return self
            def item(self): return True
        self.torch = SimpleNamespace(isfinite=lambda value:Finite(),is_tensor=lambda value:False)
        self.receipt = dict(file="checkpoints/broader-final.pt",sha256="e"*64,state=self.state)

    def load(self):
        with mock.patch.dict(sys.modules,{"torch":self.torch}), \
                mock.patch("continuous_policy.load_checkpoint",return_value=(self.model,self.saved)) as loader, \
                mock.patch("candidate_continuation_checkpoint.state_hash",side_effect=sha):
            result = verifier.load_model(Path("invented-never-opened.pt"),self.receipt,self.plan,self.experiment)
            self.assertEqual(loader.call_args.kwargs["device"],"cpu")
            self.assertEqual(loader.call_args.kwargs["expected_updates"],137)
            return result

    def test_checkpoint_reader_verifies_original_shapes_adam_and_rng_without_learning(self):
        result = self.load()
        self.assertEqual(result["state"],self.state)
        self.assertEqual(result["actor_hash"],sha("actor64"))

    def test_synthetic_checkpoint_or_wrong_adam_or_rng_cannot_pass(self):
        original = copy.deepcopy(self.saved)
        for mutation,reason in (
            (lambda s:s.update(synthetic=True),"comparison"),
            (lambda s:s["optimizer"]["param_groups"][0].update(lr=.01),"learning rate"),
            (lambda s:s["optimizer"]["state"][0].update(step=549),"Adam counters"),
            (lambda s:s.update(action_rng_state=[999]),"random stream")):
            self.saved = copy.deepcopy(original)
            mutation(self.saved)
            with self.assertRaisesRegex(ValueError,reason): self.load()


if __name__ == "__main__":
    unittest.main()
