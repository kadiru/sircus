"""Fabricated full-runtime integration; no Torch model, physics or learning runs."""

from collections import Counter
import copy
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import seed22_continuation_contract as contract
import seed22_continuation_native as native
import seed22_continuation_watchdog as watchdog
import seed22_continuation as entry


class Tensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def __getitem__(self, key): return Tensor(self.value[key])
    def detach(self): return self
    def cpu(self): return self
    def tolist(self): return self.value.tolist()


def plain(value):
    if isinstance(value, Tensor): return value.tolist()
    if isinstance(value, dict): return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)): return [plain(item) for item in value]
    return value


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".synthetic-tmp")
    temporary.write_text(json.dumps(plain(value), allow_nan=False))
    temporary.replace(path)


def summary(rows, returns):
    return dict(attempts=len(rows), captures=sum(row["route_success"] for row in rows),
        clean=sum(row["route_quality"] == 2 for row in rows),
        rejected=sum(bool(row["invalid"]) for row in rows),
        failures=dict(Counter(row["invalid"] or ("captured" if row["route_success"] else
            "external_contact" if row["external_contact_steps"] else "released_miss") for row in rows)),
        mean_reward=float(np.mean(returns)) if returns else None)


def cells(role):
    return [dict(id=f"{role}-{i:03d}", split=role,
        bucket_xy_m=[.10 + (i % 7) * .12, -.60 + (i // 7) * .12])
        for i in range(70 if role == "test" else 35)]


class RuntimeFixture:
    def __init__(self, directory, *, fail_train=False):
        self.output = Path(directory).resolve()
        self.fail_train = fail_train
        self.dispatches, self.updates = [], []
        self.saved = {}
        self.clock = 100.
        self.policy = None
        self.parent = None
        self.env = None
        self.low = np.array([-1., .28, .14, .08, .2])
        self.high = np.array([1., 1.2, .6, .56, .9])
        self.experiment = dict(source=dict(geometry_fingerprint="f" * 64),
                               protocol=dict(domain=dict(fingerprint="d" * 64)))
        self.split = dict(train=cells("train"), validation=cells("validation"),
                          test=cells("test"), fingerprint="c" * 64)
        urdf = self.output / "synthetic.urdf"
        urdf.write_text("<robot>" + "".join(f'<joint name="joint{i}"><limit effort="12"/></joint>'
                                           for i in range(1, 7)) + "</robot>")
        self.urdf = urdf
        scene = self.output / "synthetic-scene.json"
        scene.write_text("{}")
        self.source = self.output / "synthetic-source"
        self.source.mkdir()
        self.assets = self.output / "synthetic-assets"
        self.assets.mkdir()
        self.manifest = dict(protocol=dict(fingerprint="a" * 64),
            files={name: dict(path=str(urdf), sha256=contract.PARENT_CHECKPOINT_SHA256)
                   for name in ("camera", "dropper_urdf", "board_urdf", "parent")},
            assets=str(self.assets), source_run=str(self.source), source_sha256={})
        self.manifest["files"]["scene"] = dict(path=str(scene), sha256=hashlib.sha256(b"{}").hexdigest())

    def now(self):
        self.clock += .01
        return self.clock

    def restore(self, manifest, experiment, device):
        model = SimpleNamespace(version=0)
        optimizer = SimpleNamespace(state_dict=lambda: dict(state="synthetic-restored"))
        if self.policy is None: self.policy = model
        else: self.parent = model
        return model, optimizer, dict(weight_hash="0" * 64)

    @staticmethod
    def weight_hash(model): return f"{model.version:064x}"

    @staticmethod
    def observations(contexts):
        return np.asarray([[*cell["bucket_xy_m"], .7, 0., 1.] for cell in contexts], dtype=np.float32)

    def sample(self, model, obs, *, generator, deterministic):
        count = len(obs.value)
        latent = (np.zeros((count, 5)) if deterministic else
                  np.repeat((.125 + np.arange(count) * .001)[:, None], 5, axis=1))
        return dict(latent=Tensor(latent), logp=Tensor(-2. - np.arange(count) * .01),
                    actions_normalized=Tensor(np.tanh(latent)))

    def update(self, model, optimizer, obs, latent, old_logp, returns, spec):
        # These are fabricated counters, not gradient operations or a real model.
        self.updates.append(dict(obs=obs.value.copy(), latent=latent.value.copy(),
            logp=old_logp.value.copy(), returns=returns.value.copy(), spec=dict(spec)))
        before = self.weight_hash(model)
        model.version += 1
        return dict(before=before, after=self.weight_hash(model), samples=len(returns.value),
                    epochs=[dict(samples=len(returns.value)) for _ in range(4)])

    def save(self, payload, path):
        self.saved[Path(path).name] = plain(payload)
        Path(path).write_text(json.dumps(plain(payload), sort_keys=True))

    def compressed(self, path, packet):
        raw = json.dumps(plain(packet), allow_nan=False).encode()
        Path(path).write_bytes(gzip.compress(raw))
        return dict(file=Path(path).name, sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())

    def construct(self, *, output, assets, source_run, camera, device):
        Path(output).mkdir()
        fixture = self

        class Env:
            def __init__(self):
                self.dest = Path(output)
                self.sim = SimpleNamespace(device="synthetic-cpu")
                self.cfg = {}
                self.steps = 0
                self.dt = .01
                self.origins = np.zeros((35, 3))
                self.spec = dict(cells=cells("train"))
                self.scoring_normals = [np.array([0., 0., 1.])] * 35
                self.qualifications = [dict(accepted=True)] * 35
                self.path_reasons = [None] * 35
                self.path_checks = [dict(accepted=True)] * 35
                self.preflight_records = [dict(synthetic=True)] * 35
                self.transit_history = self.correction_history = []
                self.geometry_filter_synthetic = False
                self.geometry_wall_s = self.geometry_process_cpu_s = self.setup_wall_s = .02

            def tick(self):
                self.steps += 1

            def home(self):
                return dict(physics_step=self.steps, synthetic=True)

            def shot_batch(self, actions, *, record_video, enabled):
                # Exercise durable reservation before every assessment/native call.
                requests = list((fixture.output / "requests").glob("*.json"))
                current = [json.loads(path.read_text()) for path in requests
                           if json.loads(path.read_text())["index"] == self.trial_batch]
                assert len(current) == 1
                assert current[0]["reservation"]["payload"]["enabled_count"] == int(sum(enabled))
                self.proposed_actions = [dict(action) for action in actions]
                self.geometry_records = [dict(synthetic=True)] * 35
                self.geometry_starts = [dict(synthetic=True)] * 35
                self.geometry_enabled_this_batch = self.trial_mode != "raw_parent"
                if self.geometry_enabled_this_batch:
                    for i in np.flatnonzero(enabled): actions[i]["height_m"] += .001
                fixture.dispatches.append(dict(mode=self.trial_mode, enabled=enabled.copy(),
                    proposal=self.proposed_actions, executed=[dict(action) for action in actions]))
                self.release_step = self.steps
                self.tick()
                self.physics_samples = [dict(position_m=np.zeros((35, 3)).tolist(),
                    velocity_m_s=np.zeros((35, 3)).tolist(), forces_N=np.zeros((35, 6)).tolist())]
                if fixture.fail_train and self.trial_mode == "train":
                    raise RuntimeError("synthetic native failure after reservation")
                rows = []
                for i, active in enumerate(enabled):
                    invalid = None if active else "no_selected_setting"
                    caught = bool(active)
                    if self.trial_mode == "train" and i == 9:
                        invalid, caught = "no_geometry_support", False
                    elif self.trial_mode == "train" and i == 17:
                        caught = False
                    rows.append(dict(invalid=invalid, route_success=caught,
                        route_quality=1 if caught else 0, external_contact_steps=0))
                return rows, dict(settling=[], trajectory_m=[])

        self.env = Env()
        return self.env, dict(spec=dict(cells=cells("train"), fingerprint="f" * 64))

    def modules(self):
        torch = SimpleNamespace(tensor=lambda value, **kwargs: Tensor(value),
            as_tensor=lambda value, **kwargs: Tensor(value),
            cat=lambda group: Tensor(np.concatenate([x.value for x in group])),
            Generator=lambda **kwargs: SimpleNamespace(manual_seed=lambda seed:
                SimpleNamespace(get_state=lambda: Tensor([seed]))),
            manual_seed=lambda seed: None, set_float32_matmul_precision=lambda precision: None,
            get_rng_state=lambda: Tensor([1]), cuda=SimpleNamespace(get_rng_state=lambda: Tensor([2]),
                mem_get_info=lambda: (8 * 1024**3, 24 * 1024**3)),
            backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
                                     cudnn=SimpleNamespace(allow_tf32=True)), save=self.save)
        policy = SimpleNamespace(build_observations=self.observations, sample=self.sample,
            physical_actions=lambda values, low, high: low + (np.asarray(values) + 1) * (high - low) / 2,
            checkpoint_payload=lambda model, optimizer, **kwargs:
                dict(kwargs, optimizer_present=optimizer is not None, weight_hash=self.weight_hash(model)),
            weight_hash=self.weight_hash, update=self.update)
        reward = SimpleNamespace(build_features=lambda *args: dict(synthetic=True),
            reward=lambda row, features: -1. if row["invalid"] else 1.5 if row["route_success"] else -.6)
        names = ("bearing_offset_rad", "radius_m", "height_m", "gap_m", "tilt_rad")
        return {"torch": torch, "continuous_policy": policy,
            "candidate_continuation_scene": SimpleNamespace(construct_scene=self.construct,
                StartupRecorder=lambda *args: SimpleNamespace(run=lambda: dict(status="completed"))),
            "continuous_feasibility": SimpleNamespace(LOW=self.low, HIGH=self.high,
                decode_action=lambda cell, action, name: dict(zip(names, map(float, action)), id=name)),
            "prepare_assets": SimpleNamespace(chain=lambda root: "synthetic-chain"),
            "presentation_recording": SimpleNamespace(save_physics=lambda path, samples:
                Path(path).write_bytes(b"synthetic-physics")),
            "seed22_continuation_geometry": SimpleNamespace(install=lambda env: None),
            "shot_benchmark_native": SimpleNamespace(summary=summary, atomic=write_json, compressed_json=self.compressed),
            "shot_benchmark_reward": reward,
            "settled_transit": SimpleNamespace(protocol=lambda: dict(synthetic=True)),
            "candidate_continuation_checkpoint": SimpleNamespace(state_hash=lambda value: "b" * 64)}

    def run(self):
        first = dict(call_index=0, batch_index=0, logical_round=0, attempt_start=0,
            attempt_stop=3, attempts=3, site_indices=[2, 9, 17],
            enabled=[i in (2, 9, 17) for i in range(35)], milestone_after=None)
        original_read_text = Path.read_text
        original_digest = native.digest
        def read_text(path, *args, **kwargs):
            if str(path) == "/proc/meminfo":
                return "MemAvailable: 16777216 kB\n"
            return original_read_text(path, *args, **kwargs)
        def digest(path):
            return contract.PARENT_CHECKPOINT_SHA256 if Path(path) == self.urdf else original_digest(path)
        with patch.dict(sys.modules, self.modules()), \
                patch.object(native, "restore_parent", self.restore), \
                patch.object(entry, "SOURCE", self.source), \
                patch.object(native, "digest", digest), \
                patch.object(Path, "read_text", read_text), \
                patch.object(contract, "batch_schedule", return_value=[first]), \
                patch.object(native.time, "monotonic", self.now), \
                patch.object(native.shutil, "disk_usage", return_value=SimpleNamespace(free=100 * 1024**3)), \
                patch.dict("os.environ", {"SIRCUS_CONTINUATION_DEADLINE_MONOTONIC": "100000",
                    "SIRCUS_CONTINUATION_STARTED_MONOTONIC": "100"}):
            return native.run(self.manifest, self.experiment, self.split, self.output, "a" * 32)


class RuntimeIntegrationTests(unittest.TestCase):
    def test_first_assessments_partial_train_one_update_and_frozen_final(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = RuntimeFixture(directory)
            result = fixture.run()
            self.assertEqual(result["status"], "native_complete_audit_pending")
            self.assertEqual([d["mode"] for d in fixture.dispatches[:3]], ["raw_parent", "monitor", "train"])
            self.assertEqual((result["train_requested"], result["train_completed"]), (3, 3))
            self.assertEqual(result["train"]["captures"], 1)
            self.assertEqual(result["train"]["rejected"], 1)
            self.assertEqual(result["train"]["failures"]["released_miss"], 1)
            self.assertEqual(result["new_updates"], 1)
            self.assertEqual(result["new_optimizer_steps"], 4)
            self.assertEqual(len(fixture.updates), 1)
            update = fixture.updates[0]
            np.testing.assert_array_equal(update["obs"], fixture.observations(fixture.split["train"])[[2, 9, 17]])
            np.testing.assert_array_equal(update["latent"],
                np.repeat((.125 + np.array([2, 9, 17]) * .001)[:, None], 5, axis=1))
            np.testing.assert_array_equal(update["logp"], -2. - np.array([2, 9, 17]) * .01)
            np.testing.assert_array_equal(update["returns"], [1.5, -1., -.6])
            self.assertEqual(fixture.parent.version, 0)
            self.assertEqual(fixture.policy.version, 1)
            train_dispatch = fixture.dispatches[2]
            self.assertAlmostEqual(train_dispatch["executed"][2]["height_m"] -
                                   train_dispatch["proposal"][2]["height_m"], .001)
            self.assertEqual(result["ledger"]["train_reserved"], 3)
            self.assertEqual(result["ledger"]["train_unresolved"], 0)
            self.assertEqual(result["ledger"]["assessment_reserved"], 315)
            progress = json.loads((Path(directory) / "progress.json").read_text())
            self.assertEqual(watchdog.read_progress(Path(directory) / "progress.json",
                session_id="a" * 32, runtime_fingerprint="a" * 64), progress)
            self.assertEqual(fixture.saved["initial.pt"]["update_count"], 73)
            self.assertTrue(fixture.saved["initial.pt"]["optimizer_present"])
            self.assertEqual(fixture.saved["final.pt"]["update_count"], 74)
            self.assertEqual(fixture.saved["final.pt"]["attempted_shots"], 10003)
            self.assertEqual(contract.TRAIN_REQUEST_CAP, 30000)
            self.assertEqual(contract.WALL_SECONDS, 28800)

    def test_native_failure_preserves_reserved_cost_and_partial_physics(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = RuntimeFixture(directory, fail_train=True)
            with self.assertRaisesRegex(RuntimeError, "synthetic native failure"):
                fixture.run()
            failure = json.loads((Path(directory) / "failure.json").read_text())
            self.assertEqual(failure["status"], "incomplete_no_retry")
            self.assertEqual((failure["train_requested"], failure["train_completed"]), (3, 0))
            self.assertEqual(failure["ledger"]["train_unresolved"], 3)
            self.assertTrue(failure["ledger"]["closed"])
            self.assertTrue((Path(directory) / "native" / "incomplete-last-physics.npz").is_file())
            self.assertTrue((Path(directory) / "requests" / "train-00002.json").is_file())
            self.assertFalse((Path(directory) / "result.json").exists())
            self.assertEqual(len(fixture.updates), 0)
            self.assertNotIn("final.pt", fixture.saved)


class ParentRestoreIntegrationTests(unittest.TestCase):
    def restore_fixture(self, step=292):
        # Exercise the real entry adapter with fabricated saved moments. This
        # does not load a checkpoint, create a neural network or run Adam.
        parameter = object()
        policy = SimpleNamespace(parameters=lambda: iter([parameter]))
        saved = dict(seed=20261122, optimizer=dict(state={0: dict(step=step,
            exp_avg="synthetic-first-moment", exp_avg_sq="synthetic-second-moment")},
            param_groups=[dict(lr=.0003, params=[0])]))
        before = copy.deepcopy(saved)
        calls = []

        class Adam:
            def __init__(self, parameters, lr):
                self.parameters = list(parameters)
                self.param_groups = [dict(lr=lr)]
                self.state = {}

            def load_state_dict(self, value):
                calls.append(copy.deepcopy(value))
                self.param_groups = copy.deepcopy(value["param_groups"])
                self.state = {parameter: copy.deepcopy(value["state"][0])}

        torch = SimpleNamespace(optim=SimpleNamespace(Adam=Adam), is_tensor=lambda value: False)
        def load_checkpoint(path, **kwargs):
            self.assertEqual(kwargs["expected_updates"], 73)
            self.assertEqual(kwargs["expected_attempted_shots"], 10000)
            self.assertEqual(kwargs["device"], "cpu")
            return policy, saved
        manifest = dict(files=dict(parent=dict(path="synthetic-parent-not-opened.pt")))
        experiment = dict(protocol=dict(fingerprint="a" * 64, domain=dict(fingerprint="b" * 64)),
                          source=dict(geometry_fingerprint="c" * 64))
        with patch.dict(sys.modules, {"torch": torch,
                "continuous_policy": SimpleNamespace(load_checkpoint=load_checkpoint)}):
            result = entry.restore_parent(manifest, experiment)
        self.assertEqual(saved, before)
        return result, calls, before

    def test_actual_adapter_restores_inherited_moments_before_return(self):
        (policy, optimizer, saved), calls, original = self.restore_fixture()
        self.assertEqual(calls, [original["optimizer"]])
        self.assertEqual(next(iter(optimizer.state.values())), original["optimizer"]["state"][0])
        self.assertEqual(optimizer.param_groups[0]["lr"], .0003)

    def test_actual_adapter_refuses_wrong_inherited_adam_step(self):
        with self.assertRaisesRegex(ValueError, "Parent Adam steps changed"):
            self.restore_fixture(step=293)


if __name__ == "__main__":
    unittest.main()
