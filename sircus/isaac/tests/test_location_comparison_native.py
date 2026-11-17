"""Invented native/policy bridge tests: no Torch, simulator, shots or learning."""

import copy
from contextlib import contextmanager, ExitStack
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import location_comparison_native as native


def cells(start=0, count=35, split="train"):
    return [dict(id=f"target-{start+i}", target_id=f"target-{start+i}",
                 request_id=f"request-{start+i}", split=split,
                 bucket_xy_m=[.40 + i*.001, -.35], physical_slot=i) for i in range(count)]


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class FakeDependencies:
    synthetic = True

    def __init__(self):
        self.updates = []
        self.installations = 0
        self.cleaned = 0

    def install(self, env):
        self.installations += 1
        def cleanup():
            self.cleaned += 1
        return cleanup

    def prepare_environment(self, env):
        pass

    def restore(self, action_seed, optimizer_seed):
        return dict(version=0, steps=292, action_rng=action_seed, optimizer_rng=optimizer_seed)

    def state(self, handle):
        return dict(weight_hash=sha(handle["version"]), optimizer_hash=sha(handle["steps"]),
                    optimizer_steps=handle["steps"], action_rng_hash=sha(handle["action_rng"]),
                    optimizer_rng_hash=sha(handle["optimizer_rng"]))

    def observations(self, context):
        result = []
        for cell in context:
            x, y = cell["bucket_xy_m"]
            dx, dy = .67-x, -.10-y
            angle = math.atan2(dy, dx)
            result.append([(x-.67)/.9, (y+.25)/.625, math.hypot(dx,dy), math.sin(angle), math.cos(angle)])
        return result

    def sample(self, handle, context, deterministic):
        if not deterministic:
            handle["action_rng"] += len(context)
        return dict(latent=[[.1]*5 for c in context], logp=[-.3 for c in context],
                    actions_normalized=[[.0996679946]*5 for c in context])

    def update(self, handle, examples):
        self.updates.append(copy.deepcopy(examples))
        before = sha(handle["version"])
        handle["version"] += 1
        handle["steps"] += 4
        handle["optimizer_rng"] += 4
        return dict(samples=140, synthetic=True, before=before, after=sha(handle["version"]),
                    actor_before=before, actor_after=sha(handle["version"]),
                    epochs=[dict(samples=140, loss=0.) for _ in range(4)])

    def save_checkpoint(self, handle, path, metadata):
        Path(path).write_text(json.dumps(dict(handle=handle, metadata=metadata)))

    def actions(self, context, normalized, label):
        return [dict(id=f"{label}-{i}", radius_m=p[1]) for i, p in enumerate(normalized)]

    def features(self, env, context, rows):
        return [dict(synthetic=True) for row in rows]

    def reward(self, row, features):
        return -1. if row["invalid"] else 1.6 if row["route_success"] else -.3

    def save_physics(self, path, samples):
        Path(path).write_text(json.dumps(dict(synthetic=True, samples=samples)))

    def setup_protocol(self):
        return dict(synthetic=True, original_guards="delegated")

    def resource_sample(self):
        return dict(lifetime_peak_rss_mib=1., sampled_gpu_used_mib=0., synthetic=True)


class FakeEnvironment:
    def __init__(self):
        self.n = 35
        self.spec = dict(cells=cells())
        self.sim = SimpleNamespace(device="synthetic")
        self.origins = np.zeros((35, 3))
        self.dt = 1/240
        self.steps = 0
        self.actual_bucket_xy = None
        self.fail = False
        self.calls = []
        self.frames = []
        self.physics_samples = []

    def tick(self):
        self.steps += 1

    def home(self):
        self.actual_bucket_xy = self.bucket_xy.copy()
        self.calls.append(("home", self.actual_bucket_xy.copy()))
        self.tick()
        return dict(synthetic=True, episode_reset=True)

    def assert_buckets_fixed(self):
        np.testing.assert_array_equal(self.actual_bucket_xy, self.bucket_xy)
        self.calls.append(("readback", self.bucket_xy.copy()))

    def shot_batch(self, actions, record_video=False, enabled=None):
        self.calls.append(("shot", enabled.tolist()))
        self.proposed_actions = copy.deepcopy(actions)
        # Invent an execution-side repair; original policy proposals must survive.
        actions[0]["radius_m"] += .01
        self.tick()
        self.release_step = self.steps
        self.physics_samples = [dict(position_m=[[0., 0., .1]]*35)]
        if self.fail:
            raise RuntimeError("synthetic dispatch failure")
        for name in ("geometry_records", "geometry_starts", "proposal_ik_seeds", "preflight_records",
                     "qualifications", "path_checks", "transit_history", "correction_history"):
            setattr(self, name, [dict(synthetic=True)]*35)
        self.path_reasons = [None]*35
        self.scoring_normals = np.asarray([[0., 0., 1.]]*35)
        self.geometry_enabled_this_batch = True
        self.geometry_filter_synthetic = True
        self.geometry_wall_s = .02
        self.geometry_process_cpu_s = .01
        rows = [dict(invalid=("inactive" if not enabled[i] else "setup_not_quiet" if i % 3 == 2 else None),
                     route_success=bool(enabled[i] and i % 3 == 0), route_quality=2 if i % 3 == 0 else 0)
                for i in range(35)]
        return rows, dict(settling=[[]]*35, trajectory_m=[[]]*35)


class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name)
        self.env, self.deps = FakeEnvironment(), FakeDependencies()
        self.guards = 0
        def guard():
            self.guards += 1
            return True
        self.engine = native.NativeEngine(self.env,
            dict(files=dict(parent=dict(sha256=native.PARENT_SHA))), dict(source=dict(synthetic=True)),
            self.output, dict(fingerprint="c"*64), guard=guard, dependencies=self.deps)
        self.addCleanup(self.engine.close)
        self.engine.restore("A", 101, 102)

    def dispatch(self, start=0, count=35, split="train", model="A"):
        context = cells(start, count, split)
        packet = self.engine.propose(model, context, deterministic=split != "train")
        result = self.engine.execute(context, packet)
        examples = [dict(proposal=p, outcome=r) for p, r in zip(packet["rows"], result["rows"])]
        return context, packet, result, examples

    def test_restore_independent_identical_parent_and_rng_streams(self):
        a = self.engine.state("A")
        b = self.engine.restore("B", 101, 102)
        self.assertEqual({k:v for k,v in a.items() if k != "model_id"},
                         {k:v for k,v in b.items() if k != "model_id"})
        self.engine.propose("A", cells(), False)
        self.assertNotEqual(self.engine.state("A")["action_rng_hash"], self.engine.state("B")["action_rng_hash"])
        with self.assertRaises(ValueError):
            self.engine.restore("A", 101, 102)

    def test_partial_assessment_resets_actual_targets_and_pads_without_paid_rows(self):
        before = self.engine.state("A")
        _, _, result, _ = self.dispatch(500, 25, "test")
        self.assertEqual(len(result["rows"]), 25)
        self.assertEqual(self.env.active_mask.sum(), 25)
        self.assertEqual(len(self.env.spec["cells"]), 35)
        np.testing.assert_array_equal(self.env.actual_bucket_xy[:25], np.asarray([r["bucket_xy_m"] for r in cells(500,25,"test")]))
        self.assertEqual([x[0] for x in self.env.calls], ["home", "readback", "shot", "readback"])
        self.assertEqual(self.engine.state("A"), before)
        self.assertTrue(result["evidence"]["synthetic"])
        self.assertGreater(self.guards, 4)

    def test_raw_proposals_and_all_rejects_reach_one_140_request_update(self):
        examples = []
        for start in range(0, 140, 35):
            _, _, result, batch = self.dispatch(start)
            self.assertEqual(len(result["rows"]), 35)
            self.assertTrue(any(row["outcome"] == "rejected" for row in result["rows"]))
            examples.extend(batch)
        state = self.engine.learn("A", examples)
        self.assertEqual((state["attempts"], state["updates"], state["optimizer_steps"]), (10140,74,296))
        self.assertEqual(self.deps.updates[0], examples)
        self.assertTrue(all(e["proposal"]["latent"] == [.1]*5 for e in examples))
        self.assertTrue(all(e["proposal"]["actions_normalized"] == [.0996679946]*5 for e in examples))
        with self.assertRaises(ValueError):
            self.engine.learn("A", examples)

    def test_target_mismatch_repeat_and_reward_tampering_fail(self):
        context, packet, result, examples = self.dispatch()
        with self.assertRaises(ValueError):
            self.engine.execute(context, packet)
        new_context = cells(100)
        new_packet = self.engine.propose("A", new_context, False)
        new_context[0]["bucket_xy_m"][0] += .1
        with self.assertRaises(ValueError):
            self.engine.execute(new_context, new_packet)
        for start in range(35,140,35):
            examples.extend(self.dispatch(start)[3])
        examples[0]["outcome"]["reward"] += 1
        with self.assertRaises(ValueError):
            self.engine.learn("A", examples)
        self.assertFalse(self.deps.updates)

    def test_checkpoint_keeps_optimizer_rng_and_refuses_overwrite(self):
        receipt = self.engine.checkpoint("A", "initial")
        self.assertEqual(receipt["state"]["optimizer_steps"], 292)
        payload = json.loads((self.output/receipt["file"]).read_text())
        self.assertIn("optimizer_rng", payload["handle"])
        self.assertEqual(receipt["sha256"], hashlib.sha256((self.output/receipt["file"]).read_bytes()).hexdigest())
        with self.assertRaises(ValueError):
            self.engine.checkpoint("A", "initial")

    def test_failed_native_dispatch_keeps_partial_trace_and_cannot_retry(self):
        self.env.fail = True
        context = cells()
        packet = self.engine.propose("A", context, False)
        tick = self.env.tick
        with self.assertRaises(RuntimeError):
            self.engine.execute(context, packet)
        self.assertEqual(self.env.tick, tick)
        self.assertFalse(self.env.capture_trial)
        folder = self.output/"comparison-native"
        self.assertTrue((self.output/"native/trial-00000-incomplete.npz").exists())
        failure = json.loads(gzip.decompress((folder/"trial-00000-failure.json.gz").read_bytes()))
        self.assertEqual(failure["status"], "incomplete_no_retry")
        with self.assertRaises(ValueError):
            self.engine.execute(context, packet)

    def test_native_requires_authorization_before_importing_dependencies(self):
        with self.assertRaisesRegex(ValueError, "not execution authorization"):
            native.NativeEngine(self.env, {}, {}, self.output, {}, guard=lambda:None)

    def test_false_guard_stops_before_rng_or_native_activity(self):
        before = self.engine.state("A")
        self.engine.guard = lambda: False
        with self.assertRaisesRegex(ValueError, "guard did not pass"):
            self.engine.propose("A", cells(), False)
        self.assertEqual(self.engine.state("A"), before)
        self.assertEqual(self.env.calls, [])

    def test_every_learned_state_is_saved_before_sampling_without_changing_model(self):
        before = self.engine.state("A")
        context, packet, result, examples = self.dispatch()
        receipt = packet["checkpoint"]
        self.assertEqual(receipt["state"], before)
        self.assertEqual(receipt["file"], "checkpoints/A-sampling-00073.pt")
        self.assertTrue((self.output/receipt["file"]).is_file())
        for start in range(35,140,35):
            _, next_packet, _, more = self.dispatch(start)
            self.assertEqual(next_packet["checkpoint"], receipt)
            examples.extend(more)
        self.engine.learn("A", examples)
        before = self.engine.state("A")
        _, next_packet, _, _ = self.dispatch(140)
        self.assertEqual(next_packet["checkpoint"]["state"], before)
        self.assertEqual(next_packet["checkpoint"]["file"], "checkpoints/A-sampling-00074.pt")

    def test_terminal_receipt_requires_recorded_frame_bytes_to_exist_and_match(self):
        original = self.env.shot_batch
        def fake_frames(*args, **kwargs):
            result = original(*args, **kwargs)
            self.env.frames = [dict(frame=0,jpeg_sha256=hashlib.sha256(b"invented image").hexdigest())]
            return result
        self.env.shot_batch = fake_frames
        # Deliberately claim a missing prescheduled frame. It cannot be reported
        # as a completed batch even though fake physical rows were returned.
        with self.assertRaisesRegex(ValueError,"frame bytes"):
            self.dispatch()
        self.assertFalse((self.output/"comparison-native/trial-00000.json.gz").exists())
        self.assertTrue((self.output/"comparison-native/trial-00000-failure.json.gz").exists())

    def test_production_update_wrapper_isolates_entropy_rng_with_fake_torch_only(self):
        class State:
            def __init__(self, value): self.value = value
            def clone(self): return State(self.value)
        class Torch:
            float32 = "fake-float32"
            def __init__(self):
                self.global_state = State(999)
                self.random = SimpleNamespace(fork_rng=self.fork)
            def get_rng_state(self): return self.global_state
            def set_rng_state(self, state): self.global_state = state.clone()
            def tensor(self, data, device, dtype): return copy.deepcopy(data)
            @contextmanager
            def fork(self, devices):
                old = self.global_state.clone()
                try: yield
                finally: self.global_state = old
        torch = Torch()
        inputs = []
        def invented_update(policy, optimizer, observations, latent, logp, returns, spec):
            inputs.append((observations, latent, logp, returns, spec))
            torch.global_state.value += 7
            return dict(synthetic=True)
        dependency = native.ProductionDependencies.__new__(native.ProductionDependencies)
        dependency.torch, dependency.devices, dependency.device = torch, [], "synthetic"
        dependency.policy = SimpleNamespace(update=invented_update)
        examples = [dict(proposal=dict(observation=[0.]*5, latent=[.1]*5, logp=-.3),
                         outcome=dict(reward=-1.)) for _ in range(140)]
        handles = [dict(policy=object(), optimizer=object(), optimizer_rng=dict(cpu=State(10), cuda=[]))
                   for _ in range(2)]
        for handle in handles:
            dependency.update(handle, examples)
            self.assertEqual(handle["optimizer_rng"]["cpu"].value, 17)
            self.assertEqual(torch.global_state.value, 999)
        self.assertEqual(inputs[0], inputs[1])
        self.assertEqual(inputs[0][1], [[.1]*5]*140)

    def test_full_orchestrator_uses_bridge_with_only_invented_native_policy_dependencies(self):
        from test_location_comparison_runtime import synthetic_plan
        from location_comparison_mock import FakeClock
        from location_comparison_bounds import Bounds
        from location_comparison_runtime import ComparisonRunner, Journal
        from location_comparison_audit import audit_trace
        plan = synthetic_plan()
        clock = FakeClock()
        bounds = Bounds(plan, clock, lambda: dict(free_storage_bytes=10*1024**3))
        # A separate engine is required: setUp already restored the test model A.
        with tempfile.TemporaryDirectory() as directory:
            env, dependencies = FakeEnvironment(), FakeDependencies()
            engine = native.NativeEngine(env, dict(files=dict(parent=dict(sha256=native.PARENT_SHA))),
                dict(source=dict(synthetic=True)), directory, plan,
                guard=bounds.check, dependencies=dependencies, clock=clock)
            self.addCleanup(engine.close)
            trace = ComparisonRunner(plan, engine, bounds, Journal(), clock=clock).run()
            final = trace["events"][-1]["payload"]
            self.assertEqual(final["status"], "complete_audit_pending", final["reason"])
            self.assertEqual(final["counts"]["assessment"]["completed"], 600)
            self.assertEqual(len(dependencies.updates), 128)
            for arm in ("original_usable", "broader"):
                self.assertEqual(final["counts"]["train"][arm]["completed"], 8960)
                self.assertEqual(engine.state(arm)["optimizer_steps"], 548)
            report = audit_trace(plan, trace)
            self.assertIn("synthetic", report["status"])
            self.assertTrue(all(event["payload"]["evidence"]["synthetic"]
                for event in trace["events"] if event["kind"] == "terminal"))


class NativeInputTests(unittest.TestCase):
    def test_native_import_binding_rejects_shadowed_or_preloaded_other_source(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory).resolve()
            path = source/"invented_helper.py"
            path.write_text("# Invented helper, never imported\n")
            manifest = dict(source_sha256={path.name: native._file_hash(path)})
            module = SimpleNamespace(__file__=str(path))
            with mock.patch.object(native, "SOURCE", source), \
                    mock.patch.dict(sys.modules, {"invented_helper":module}), \
                    mock.patch.object(native.importlib.util, "find_spec", return_value=SimpleNamespace(origin=str(path))) as find:
                native.validate_source_imports(manifest, ("invented_helper",))
                find.return_value.origin = str(source/"elsewhere.py")
                with self.assertRaisesRegex(ValueError, "resolves outside"):
                    native.validate_source_imports(manifest, ("invented_helper",))
                find.return_value.origin = str(path)
                module.__file__ = str(source/"elsewhere.py")
                with self.assertRaisesRegex(ValueError, "Loaded native helper"):
                    native.validate_source_imports(manifest)

    def test_native_binding_checks_reject_changed_bytes_or_resealed_helper_before_torch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source = root/"source"
            source.mkdir()
            helper = source/"continuous_geometry_layer.py"
            helper.write_text("# Synthetic binding fixture only\n")
            helper_sha = native._file_hash(helper)
            cfg = dict(synthetic=True)
            experiment = dict(source=dict(geometry_fingerprint="e"*64))
            files = {}
            for role, value in (("parent", "synthetic checkpoint"), ("experiment", experiment), ("scene", cfg)):
                path = root/(role+".json")
                path.write_text(json.dumps(value))
                files[role] = dict(path=str(path), sha256=native._file_hash(path))
            manifest = dict(files=files, source_sha256={helper.name:helper_sha})
            plan = dict(bindings=dict(continuous_geometry_layer=helper_sha))
            plan["fingerprint"] = native._hash(plan)
            with ExitStack() as stack:
                for name, value in (("SOURCE",source), ("CRITICAL_SOURCES",{helper.name:helper_sha}),
                    ("PREPARED_FINGERPRINT",plan["fingerprint"]), ("PARENT_SHA",files["parent"]["sha256"]),
                    ("EXPERIMENT_SHA",files["experiment"]["sha256"]), ("SCENE_SHA",files["scene"]["sha256"])):
                    stack.enter_context(mock.patch.object(native,name,value))
                self.assertTrue(native.validate_native_inputs(manifest,experiment,plan,
                    env_cfg=cfg,env_geometry_fingerprint="e"*64))
                changed = copy.deepcopy(manifest)
                Path(files["parent"]["path"]).write_text("changed checkpoint")
                changed["files"]["parent"]["sha256"] = native._file_hash(files["parent"]["path"])
                with self.assertRaisesRegex(ValueError,"immutable native input"):
                    native.validate_native_inputs(changed,experiment,plan,env_cfg=cfg,env_geometry_fingerprint="e"*64)
                Path(files["parent"]["path"]).write_text(json.dumps("synthetic checkpoint"))
                helper.write_text("# Changed synthetic helper\n")
                changed = copy.deepcopy(manifest)
                changed["source_sha256"][helper.name] = native._file_hash(helper)
                with self.assertRaisesRegex(ValueError,"helper changed"):
                    native.validate_native_inputs(changed,experiment,plan,env_cfg=cfg,env_geometry_fingerprint="e"*64)


if __name__ == "__main__":
    unittest.main()
