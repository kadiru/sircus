"""Synthetic saved-evidence audit tests; no model, optimizer or simulator runs."""

import copy
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import seed22_continuation_audit as subject
from seed22_continuation_contract import (
    Ledger, PARENT_CHECKPOINT_SHA256, ORIGINAL_EXPERIMENT_SHA256,
    ORIGINAL_SPLIT_SHA256, GEOMETRY_LAYER_SHA256, create_fresh_splits,
    make_protocol, signed,
)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))


def packet(path, value):
    raw = json.dumps(value, allow_nan=False).encode()
    with gzip.open(path, "wb") as stream:
        stream.write(raw)
    receipt = dict(file=path.name, sha256=subject.digest(path), bytes=path.stat().st_size,
                   uncompressed_sha256=hashlib.sha256(raw).hexdigest())
    write(Path(str(path) + ".receipt.json"), receipt)
    return receipt


class Fixture:
    """Ten entirely fabricated cohorts plus one fake learning update."""

    def __init__(self, root):
        self.root, self.native = root, root / "native"
        for name in ("requests", "events", "checkpoints", "native/trials", "native/frames", "native/startup"):
            (root / name).mkdir(parents=True, exist_ok=True)
        self.session = "a" * 32
        train = []
        for i in range(35):
            row, column = divmod(i, 7)
            train.append(dict(id=f"train-{i:03d}", split="train", tile_index=i,
                grid_row=row, grid_column=column,
                bucket_xy_m=[-.10 + (column + .5) * 1.50 / 7, -.70 + (row + .5) * .90 / 5]))
        self.splits = create_fresh_splits(train, [])
        self.protocol = make_protocol(PARENT_CHECKPOINT_SHA256, ORIGINAL_EXPERIMENT_SHA256,
            ORIGINAL_SPLIT_SHA256, GEOMETRY_LAYER_SHA256, {"synthetic.py": "0" * 64},
            self.splits["exclusions_fingerprint"])
        poses, camera = root / "poses.json", root / "camera.json"
        write(poses, dict(dropper_joint_rad=[0.] * 6, board_joint_rad=[0.] * 6))
        write(camera, dict(eye_m=[0., 0., 1.], target_m=[0., 0., 0.]))
        self.manifest = dict(schema="sircus-seed22-continuation-inputs-v1", synthetic=True,
            protocol=self.protocol, files={"poses": dict(path=str(poses), sha256=subject.digest(poses)),
                                         "camera": dict(path=str(camera), sha256=subject.digest(camera))})
        self.spec = signed(dict(scene=dict(synthetic=True), cells=train))
        self.experiment = dict(source=dict(geometry_fingerprint=self.spec["fingerprint"], synthetic=True),
                               protocol=dict(domain=dict(fingerprint="d" * 64)))
        write(root / "inputs.json", self.manifest)
        write(root / "splits.json", self.splits)
        write(self.native / "contract.json", self.spec)
        self.cps = {}
        for name, attempts, updates, state, actor, optimizer in (
            ("parent", 0, 0, "0" * 64, "a" * 64, None),
            ("initial", 0, 0, "0" * 64, "a" * 64, "c" * 64),
            ("before-00000", 0, 0, "0" * 64, "a" * 64, "c" * 64),
            ("final", 35, 1, "1" * 64, "b" * 64, "d" * 64),
        ):
            file = "checkpoints/" + name + ".pt"
            model = dict(state_hash=state, actor_hash=actor, optimizer_sha256=optimizer,
                         additional_attempts=attempts, additional_updates=updates,
                         split_fingerprint=self.splits["fingerprint"])
            write(root / file, model)
            self.cps[name] = dict(file=file, sha256=subject.digest(root / file), state_hash=state,
                                  additional_attempts=attempts, additional_updates=updates)
        ledger, receipts, evals = Ledger(self.session), {}, {}
        self.train_file = None
        fake_rows = [dict(invalid=None, route_success=False, route_quality=0, external_contact_steps=0) for _ in range(35)]
        self.summary = dict(attempts=35, captures=0, clean=0, rejected=0,
                            failures=dict(released_miss=35), mean_reward=-.8)
        for i, exp in enumerate(subject.expected_trials(self.splits, 35)):
            method, role = exp["method"], exp["role"]
            cp = (self.cps["parent"] if method in ("raw_parent", "parent_geometry") else
                  self.cps["before-00000"] if role == "train" else
                  self.cps["initial"] if exp["attempts"] == 0 else self.cps["final"])
            plain = dict(actions_normalized=[[0.] * 5 for _ in range(35)],
                         latent=[[0.] * 5 for _ in range(35)], logp=[0.] * 35, deterministic=role != "train")
            stem, name = f"{method}-{i:05d}", f"trials/{method}-{i:05d}.json.gz"
            reservation = ledger.reserve(i, role, 35)
            request = dict(index=i, role=role, method=method, checkpoint=cp,
                cells=exp["cells"], enabled=exp["enabled"], request_order=exp["order"],
                learner_packet=plain, reservation=reservation,
                protocol_fingerprint=self.protocol["fingerprint"],
                additional_requested=exp["attempts"] + (35 if role == "train" else 0), elapsed_s=float(i))
            write(root / "requests" / (stem + ".json"), request)
            trial = dict(schema="sircus-seed22-native-trial-v1", mode=method, batch=i, checkpoint=cp,
                cells=exp["cells"], enabled=exp["enabled"], rows=copy.deepcopy(fake_rows),
                actions_normalized=plain["actions_normalized"],
                learner_packet={k: v for k, v in plain.items() if k != "actions_normalized"},
                rewards=[-.8] * 35, training_request_order=exp["order"],
                source=self.experiment["source"], protocol_fingerprint=self.protocol["fingerprint"],
                reservation_sha256=subject.digest(root / "requests" / (stem + ".json")),
                frames=[], filming=exp["filming"], physics=None)
            receipt = packet(self.native / name, trial)
            receipts[name] = receipt["sha256"]
            completion = ledger.complete(i, role, 35)
            write(root / "events" / (stem + ".json"), dict(index=i, file=name, receipt=receipt,
                ledger_event=completion, ledger=ledger.snapshot(), elapsed_s=float(i) + .1))
            if role == "train":
                self.train_file = name
            else:
                key = (method, role, exp["attempts"])
                group = evals.setdefault(key, dict(method=method, split=role,
                    additional_attempts=exp["attempts"], checkpoint=cp, files=[], counts=copy.deepcopy(self.summary),
                    elapsed_s=float(i) + .1))
                group["files"].append(name)
                if len(group["files"]) == 2:
                    group["counts"].update(attempts=70, failures=dict(released_miss=70))
        self.update = dict(before="0" * 64, before_weight_hash="0" * 64, after="1" * 64,
            actor_before="a" * 64, actor_after="b" * 64, samples=35,
            additional_attempts=35, source_files=[self.train_file], update_index=0,
            epochs=[dict(loss=.1, samples=35) for _ in range(4)], elapsed_s=5.)
        write(root / "events/update-00001.json", self.update)
        write(root / "learning-curve.json", dict(curve=[dict(self.summary, additional_attempts=35,
            new_updates=0, elapsed_s=5.)]))
        self.assessments = list(evals.values())
        write(root / "assessments.json", dict(assessments=self.assessments))
        finish = ledger.finish("time_budget_reserved_final_assessment")
        write(root / "events/ledger-finish.json", finish)
        write(root / "ledger-history.json", dict(events=ledger.history()))
        write(root / "parent-restoration.json", dict(parent_sha256=PARENT_CHECKPOINT_SHA256,
            restored_weight_hash="0" * 64, optimizer_restored=True, parent_steps=292,
            parent_attempts=10000, parent_updates=73, exploration_seed=20261130,
            optimizer_state_sha256="c" * 64))
        self.result = signed(dict(schema="sircus-seed22-continuation-result-v1", status="native_complete_audit_pending",
            session_id=self.session, protocol_fingerprint=self.protocol["fingerprint"],
            stop_reason="time_budget_reserved_final_assessment", train_requested=35, train_completed=35,
            train=self.summary, parent_attempts=10000, parent_updates=73, new_updates=1, new_optimizer_steps=4,
            final_checkpoint=self.cps["final"], assessments=self.assessments, trial_sha256=receipts,
            update_events=1, elapsed_s=28000., ledger=ledger.snapshot(), model_promoted=False))
        self.save_result()
        from candidate_continuation_ledger import canonical_hash as compact_hash
        def wd(value):
            value["fingerprint"] = compact_hash(value)
            return value
        identity = dict(pid=123, start_token="synthetic:1", cookie=self.session)
        owner = wd(dict(schema="sircus-seed22-continuation-watchdog-v1", session_id=self.session,
            runtime_fingerprint=self.protocol["fingerprint"], shared_lock="/workspace/scene.lock",
            gpu_uuid="GPU-ab1950b0-01dd-0fab-407f-2d17936b27f4", max_requests=30000,
            storage_floor_bytes=6 * 1024**3, start_monotonic=10., deadline_monotonic=28810., native_owner=identity))
        exit_record = wd(dict(schema="sircus-seed22-continuation-watchdog-v1", session_id=self.session,
            runtime_fingerprint=self.protocol["fingerprint"], status="native_exit_zero_independent_audit_required",
            native_application_starts=1, exit_code=0, reason=None, lock_release_safe=True, native_owner=identity,
            train_requested=35, train_completed=35, wall_s=28001.))
        write(root / "watchdog-owner.json", owner)
        write(root / "watchdog-exit.json", exit_record)
        def checkpoint(path, cp, *_):
            value = subject.read(path)
            return dict(value, policy={"synthetic": True})
        self.checks = dict(inputs=lambda _: dict(experiment=self.experiment, splits=self.splits,
            files=self.manifest["files"]), checkpoint=checkpoint,
            policy=lambda *args, **kwargs: 0., geometry=lambda *args: dict(synthetic=True),
            native=lambda *args: dict(synthetic=True), startup=lambda *args, **kwargs: dict(status="passed"))

    def save_result(self):
        write(self.root / "result.json", self.result)

    def audit(self):
        return subject.audit(self.root, self.manifest, subject.digest(self.root / "result.json"),
                             synthetic=True, checks=self.checks)

    def replace_packet(self, name, mutate):
        path = self.native / name
        with gzip.open(path, "rb") as stream:
            value = json.loads(stream.read())
        mutate(value)
        receipt = packet(path, value)
        self.result["trial_sha256"][name] = receipt["sha256"]
        body = {k: v for k, v in self.result.items() if k != "fingerprint"}
        self.result = signed(body)
        self.save_result()


class AuditTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.fixture = Fixture(Path(temporary.name).resolve())

    def test_complete_mock_evidence_only_gets_synthetic_pass(self):
        result = self.fixture.audit()
        self.assertEqual(result["status"], "synthetic_passed")
        self.assertTrue(result["synthetic"])
        self.assertEqual(result["train"]["attempts"], 35)
        self.assertEqual(result["learning"]["updates"], 1)
        self.assertEqual(result["learning"]["optimizer_steps"], 4)
        self.assertEqual([x["attempts"] for x in result["final_test"].values()], [70, 70, 70])
        self.assertFalse(result["learning"]["cpu_bitwise_adam_replay"])
        self.assertFalse(result["model_promoted"])

    def test_injected_checks_cannot_certify_production(self):
        f = self.fixture
        manifest = copy.deepcopy(f.manifest)
        manifest["synthetic"] = False
        with self.assertRaisesRegex(ValueError, "Injected checks"):
            subject.audit(f.root, manifest, subject.digest(f.root / "result.json"), checks=f.checks)

    def test_result_needs_external_hash(self):
        f = self.fixture
        with self.assertRaisesRegex(ValueError, "pinned result"):
            subject.audit(f.root, f.manifest, "0" * 64, synthetic=True, checks=f.checks)

    def test_native_missing_or_extra_packet_fails(self):
        (self.fixture.native / "trials/extra-99999.json.gz").write_bytes(b"synthetic")
        with self.assertRaisesRegex(ValueError, "extra native trial"):
            self.fixture.audit()

    def test_raw_likelihood_packet_cannot_be_replaced_with_execution_action(self):
        f = self.fixture
        f.replace_packet(f.train_file, lambda p: p["actions_normalized"][0].__setitem__(0, .1))
        with self.assertRaisesRegex(ValueError, "reserved original policy sample"):
            f.audit()

    def test_paid_mask_cannot_drop_a_failure(self):
        f = self.fixture
        f.replace_packet(f.train_file, lambda p: p["enabled"].__setitem__(0, False))
        with self.assertRaisesRegex(ValueError, "mask"):
            f.audit()

    def test_held_out_data_cannot_enter_update_sources(self):
        f = self.fixture
        update = copy.deepcopy(f.update)
        update["source_files"] = [f.assessments[-1]["files"][0]]
        write(f.root / "events/update-00001.json", update)
        with self.assertRaisesRegex(ValueError, "raw-sample source"):
            f.audit()

    def test_optimizer_epoch_count_cannot_change(self):
        f = self.fixture
        update = copy.deepcopy(f.update)
        update["epochs"] = update["epochs"][:3]
        write(f.root / "events/update-00001.json", update)
        with self.assertRaisesRegex(ValueError, "optimizer epochs"):
            f.audit()

    def test_first_train_cannot_mutate_the_initial_policy_without_update(self):
        f = self.fixture
        check = f.checks["checkpoint"]
        def mutated(path, cp, *args):
            result = check(path, cp, *args)
            if path.name == "before-00000.pt":
                result["state_hash"] = "9" * 64
            return result
        f.checks["checkpoint"] = mutated
        with self.assertRaisesRegex(ValueError, "without a recorded learning update"):
            f.audit()

    def test_first_train_cannot_mutate_restored_adam_moments_without_update(self):
        f = self.fixture
        check = f.checks["checkpoint"]
        def mutated(path, cp, *args):
            result = check(path, cp, *args)
            if path.name == "before-00000.pt":
                result["optimizer_sha256"] = "9" * 64
            return result
        f.checks["checkpoint"] = mutated
        with self.assertRaisesRegex(ValueError, "without a recorded learning update"):
            f.audit()

    def test_reported_actor_after_must_match_the_actual_next_model(self):
        f = self.fixture
        update = copy.deepcopy(f.update)
        update["actor_after"] = "9" * 64
        write(f.root / "events/update-00001.json", update)
        with self.assertRaisesRegex(ValueError, "actor update identity"):
            f.audit()

    def test_exact_ledger_history_is_required(self):
        f = self.fixture
        history = subject.read(f.root / "ledger-history.json")
        history["events"][0]["payload"]["enabled_count"] = 34
        write(f.root / "ledger-history.json", history)
        with self.assertRaisesRegex(ValueError, "ledger history"):
            f.audit()

    def test_learning_curve_recounts_all_requests(self):
        f = self.fixture
        curve = subject.read(f.root / "learning-curve.json")
        curve["curve"][0]["attempts"] = 34
        write(f.root / "learning-curve.json", curve)
        with self.assertRaisesRegex(ValueError, "TRAIN curve"):
            f.audit()

    def test_checkpoint_bytes_cannot_be_replaced(self):
        (self.fixture.root / "checkpoints/final.pt").write_bytes(b"different synthetic weights")
        with self.assertRaisesRegex(ValueError, "Checkpoint bytes"):
            self.fixture.audit()

    def test_trial_filename_cannot_escape(self):
        with self.assertRaisesRegex(ValueError, "Escaped"):
            subject.evidence(self.fixture.root, "../result.json")

    def test_a_failed_native_run_cannot_pass(self):
        write(self.fixture.root / "failure.json", dict(synthetic=True))
        with self.assertRaisesRegex(ValueError, "Complete unpromoted"):
            self.fixture.audit()

    def test_missing_test_cohort_cannot_be_averaged_away(self):
        f = self.fixture
        (f.native / f.assessments[-1]["files"][-1]).unlink()
        with self.assertRaisesRegex(ValueError, "native trial"):
            f.audit()


class PaddingAndScheduleTests(unittest.TestCase):
    def test_reviewed_guard_adapter_hash_cannot_change(self):
        current = dict(subject.PINNED_GUARD_ADAPTERS)
        current["joint_feasibility.py"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "guard adapter bytes changed"):
            subject.verify_guard_adapters({}, current, {})

    def test_controller_ast_allows_only_the_pinned_prepare_body(self):
        original = "class SettledTransitShots(Base):\n    def prepare(self, actions):\n        return old(actions)\n    def release(self):\n        return 2\n"
        adapted = original.replace("old(actions)", "preflight(actions)")
        proof = subject.verify_controller_definitions("settled_transit.py", original, adapted)
        self.assertTrue(proof["all_other_class_function_definitions_unchanged"])
        with self.assertRaisesRegex(ValueError, "controller class/function body"):
            subject.verify_controller_definitions("settled_transit.py", original, adapted.replace("return 2", "return 3"))
        with self.assertRaisesRegex(ValueError, "controller class/function body"):
            subject.verify_controller_definitions("settled_transit.py", original, adapted.replace("self, actions", "self, other"))

    def test_film_ast_retains_every_physical_class_and_helper(self):
        original = "class Native:\n    def tick(self):\n        return 240\ndef helper():\n    return 1\ndef main():\n    return old()\n"
        adapted = original.replace("return old()", "return new_entry()")
        subject.verify_controller_definitions("film_cylinder_farm.py", original, adapted)
        for changed in (adapted.replace("return 240", "return 120"), adapted.replace("return 1", "return 2")):
            with self.assertRaisesRegex(ValueError, "controller class/function body"):
                subject.verify_controller_definitions("film_cylinder_farm.py", original, changed)

    def test_padding_view_changes_only_unpaid_label(self):
        row = dict(invalid="no_selected_setting", qualification=dict(accepted=False, reason="inactive_split"),
            route_success=False, success=False, route_quality=0, path_proxy=None, path_rejection=None)
        packet_value = dict(enabled=[True] + [False] * 34, rows=[dict(row, invalid="paid_decline")] + [copy.deepcopy(row) for _ in range(34)],
            preflight_records=[None] * 35, proposal_ik_seeds=[None] * 35,
            geometry_records=[None] * 35, geometry_starts=[None] * 35,
            reward_features=[None] * 35, rewards=[-1.] * 35, trajectory_m=[[] for _ in range(35)])
        original = copy.deepcopy(packet_value)
        view, slots = subject.padding_guard_view(packet_value)
        self.assertEqual(slots, list(range(1, 35)))
        self.assertEqual(view["rows"][0], original["rows"][0])
        self.assertEqual(packet_value, original)
        self.assertEqual(view["rows"][1]["invalid"], "inactive_split")
        packet_value["rewards"][1] = 1.5
        with self.assertRaisesRegex(ValueError, "Unpaid padding"):
            subject.padding_guard_view(packet_value)

    def test_request_count_must_end_at_a_completed_native_batch(self):
        splits = {"train": [{}] * 35, "validation": [{}] * 35, "test": [{}] * 70}
        with self.assertRaisesRegex(ValueError, "fixed-schedule prefix"):
            subject.expected_trials(splits, 1)
        self.assertEqual(len(subject.expected_trials(splits, 35)), 10)


if __name__ == "__main__":
    unittest.main()
