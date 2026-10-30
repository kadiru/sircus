"""Evidence persistence, cancellation and held-out selection checks."""
import asyncio
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from physics import default_config
from trial_store import TrialStore
from training_loop import train


def feasible(config, action):
    return {"arm_reachable": True, "board_arm_reachable": True}


def fake_sim(config, action, seed=0, frames=False):
    return {"reward": -abs(action["u_mm"] - 65), "retained": action["u_mm"] == 65,
            "outcome": "retained" if action["u_mm"] == 65 else "miss",
            "impact_uv_mm": [action["u_mm"], action["v_mm"]],
            "frames": [{"t": 0}] if frames else [], "seed": seed,
            "max_support_tilt_deg": 1.25, "peak_board_force_n": 12.0}


class TrainingDataTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.config = default_config()
        self.config["board"]["size_mm"] = [200, 300, 5]
        self.config["measurement_provenance"] = {"board_size": "user_measured"}

    def tearDown(self):
        self.temp.cleanup()

    async def test_freezes_policy_before_heldout_and_records_effective_noise(self):
        store = TrialStore(self.config, root=self.root, session_id="freeze-test")
        frozen_bytes = []

        def simulator(config, action, seed=0, frames=False):
            if config["simulation"]["noise_xy_mm"]:
                frozen = (store.archive / "policy.json").read_bytes()
                frozen_bytes.append(frozen)
                result = fake_sim(config, action, seed, frames)
                result["reward"] = 1000  # Held-out rewards may not alter selection.
                return result
            return fake_sim(config, action, seed, frames)

        with patch("training_loop.simulate", simulator):
            output = await train(self.config, seed=8, budget=8, feasibility=feasible, store=store)
        self.assertEqual(output["best_action"]["u_mm"], 65)
        self.assertEqual(len(frozen_bytes), 10)
        self.assertEqual(len({hashlib.sha256(b).hexdigest() for b in frozen_bytes}), 1)
        self.assertEqual(frozen_bytes[-1], (store.archive / "policy.json").read_bytes())
        self.assertEqual(len(output["history"]), 8)
        self.assertEqual(output["test"]["trials"], 10)
        train_row = json.loads((store.archive / "trial-train-0001.json").read_text())
        test_row = json.loads((store.archive / "trial-test-0001.json").read_text())
        self.assertEqual(train_row["scene_config"]["board"]["size_mm"], [200, 300, 5])
        self.assertEqual(test_row["scene_config"]["simulation"]["noise_xy_mm"], 2)
        self.assertNotEqual(train_row["evaluation_config_fingerprint"], test_row["evaluation_config_fingerprint"])
        self.assertEqual(train_row["config_fingerprint"], test_row["config_fingerprint"])
        self.assertEqual(len(set(row["seed"] for row in output["test"]["results"])), 10)
        self.assertEqual(len(list(store.archive.glob("*.json"))), 21)  # 18trials +manifest/policy/summary

    async def test_archive_survives_destructive_sync_and_records_centre_edge(self):
        store = TrialStore(self.config, root=self.root, session_id="durability")
        record = store.record_trial("manual", 1, {"u_mm": 85, "v_mm": 0, "gap_mm": 80},
                                    3, {"reward": .2, "retained": False, "outcome": "miss",
                                        "impact_uv_mm": [85, 0], "peak_board_force_n": 15})
        archive = store.archive / "trial-manual-0001.json"
        copy = store.outbox / (record["record_id"] + ".json")
        self.assertEqual(archive.read_bytes(), copy.read_bytes())
        copy.unlink()  # Viam's documented successful-sync behavior.
        self.assertEqual(json.loads(archive.read_text())["impact_region"], "edge")
        self.assertEqual(json.loads(archive.read_text())["telemetry"]["peak_board_force_n"], 15)
        self.assertFalse(store.status()["cloud_upload_verified"])
        with self.assertRaises(FileExistsError):
            store.record_trial("manual", 1, {}, 3, {})
        self.assertFalse(list(store.staging.iterdir()))

    async def test_stop_preserves_completed_trials_and_no_unfrozen_policy(self):
        store = TrialStore(self.config, root=self.root, session_id="cancel-test")
        stop = asyncio.Event()

        def progress(value):
            if value["phase"] == "training" and value["trial"] == 3:
                stop.set()

        with patch("training_loop.simulate", fake_sim):
            with self.assertRaises(asyncio.CancelledError):
                await train(self.config, budget=8, feasibility=feasible, on_progress=progress,
                            stop_event=stop, store=store)
        summary = json.loads((store.archive / "summary.json").read_text())["summary"]
        self.assertEqual(summary["status"], "cancelled")
        self.assertEqual(summary["completed_train_trials"], 3)
        self.assertFalse((store.archive / "policy.json").exists())
        self.assertEqual(len(list(store.archive.glob("trial-*.json"))), 3)

    async def test_unreachable_board_is_not_selected_and_every_failure_is_saved(self):
        store = TrialStore(self.config, root=self.root, session_id="unreachable")

        async def no_board(config, action):
            return {"arm_reachable": True, "board_arm_reachable": False, "reason": "board pose"}

        with patch("training_loop.simulate", side_effect=AssertionError("must not run physics")):
            with self.assertRaisesRegex(ValueError, "both arms reachable"):
                await train(self.config, budget=8, feasibility=no_board, store=store)
        rows = [json.loads(p.read_text()) for p in store.archive.glob("trial-*.json")]
        self.assertEqual(len(rows), 8)
        self.assertTrue(all(row["outcome"] == "unreachable_arm_target" for row in rows))
        self.assertFalse((store.archive / "policy.json").exists())

    async def test_failed_attempt_records_input_and_is_not_a_training_sample(self):
        store = TrialStore(self.config, root=self.root, session_id="failed-attempt")
        with patch("training_loop.simulate", side_effect=RuntimeError("integration stopped")):
            with self.assertRaisesRegex(RuntimeError, "integration stopped"):
                await train(self.config, budget=8, feasibility=feasible, store=store)
        record = json.loads((store.archive / "trial-train-0001.json").read_text())
        self.assertEqual(record["outcome"], "evaluation_failed")
        self.assertEqual(record["result"]["evaluation_stage"], "physics")
        self.assertFalse(record["result"]["valid_for_training"])
        self.assertIsNone(record["reward"])
        self.assertEqual(record["action"], {"u_mm": 0, "v_mm": 0, "gap_mm": 180})
        summary = json.loads((store.archive / "summary.json").read_text())["summary"]
        self.assertEqual(summary["completed_train_trials"], 0)
        self.assertFalse(summary["policy_frozen"])

    async def test_cancel_during_attempt_records_cancellation(self):
        store = TrialStore(self.config, root=self.root, session_id="cancel-attempt")
        entered = asyncio.Event()

        async def waiting_feasibility(config, action):
            entered.set()
            await asyncio.Event().wait()

        task = asyncio.create_task(train(self.config, budget=8, feasibility=waiting_feasibility, store=store))
        await entered.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        record = json.loads((store.archive / "trial-train-0001.json").read_text())
        self.assertEqual(record["outcome"], "evaluation_cancelled")
        self.assertEqual(record["result"]["evaluation_stage"], "feasibility")
        self.assertFalse(record["result"]["valid_for_training"])
        self.assertIsNone(record["reward"])
        summary = json.loads((store.archive / "summary.json").read_text())["summary"]
        self.assertEqual(summary["status"], "cancelled")

    async def test_model_limit_cannot_win_policy_selection(self):
        store = TrialStore(self.config, root=self.root, session_id="model-limit")

        def bounded_sim(config, action, seed=0, frames=False):
            result = fake_sim(config, action, seed, frames)
            if action["u_mm"] == 65:
                result.update(reward=1000, model_limit="support deflection exceeded", outcome="model_limit")
            return result

        with patch("training_loop.simulate", bounded_sim):
            output = await train(self.config, budget=8, feasibility=feasible, store=store)
        self.assertNotEqual(output["best_action"]["u_mm"], 65)
        invalid = [row for row in output["history"] if row["outcome"] == "model_limit"]
        self.assertEqual(len(invalid), 1)
        self.assertFalse(invalid[0]["valid_for_training"])

    async def test_cancel_heldout_keeps_original_frozen_policy(self):
        store = TrialStore(self.config, root=self.root, session_id="cancel-heldout")
        stop = asyncio.Event()
        policy_before = []

        def progress(value):
            if value["phase"] == "testing" and value["trial"] == 1:
                policy_before.append((store.archive / "policy.json").read_bytes())
                stop.set()

        with patch("training_loop.simulate", fake_sim):
            with self.assertRaises(asyncio.CancelledError):
                await train(self.config, budget=8, feasibility=feasible, on_progress=progress,
                            stop_event=stop, store=store)
        self.assertEqual((store.archive / "policy.json").read_bytes(), policy_before[0])
        summary = json.loads((store.archive / "summary.json").read_text())["summary"]
        self.assertTrue(summary["policy_frozen"])
        self.assertEqual(summary["completed_test_trials"], 1)
        self.assertEqual(summary["completed_train_trials"], 8)

    async def test_incomplete_json_never_published_to_watched_directory(self):
        store = TrialStore(self.config, root=self.root, session_id="json-check")
        before = set(store.outbox.iterdir())
        with self.assertRaises(ValueError):
            store.record_trial("train", 1, {}, 1, {"reward": float("nan")})
        self.assertEqual(set(store.outbox.iterdir()), before)
        with self.assertRaises(ValueError):
            TrialStore(self.config, root=self.root, session_id="../escape")


if __name__ == "__main__":
    unittest.main()
