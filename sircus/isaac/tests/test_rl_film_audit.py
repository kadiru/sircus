"""Synthetic CPU fixtures for the separate native categorical-film audit.

No simulator or hardware is started. The native menu is reused only as geometry;
fixture rewards below are deliberately synthetic and establish no capture rate.
"""

import copy
import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ISAAC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ISAAC))
from action_training import load_expanded_training_menu
from rl_film_audit import (
    FILMED, SCHEMA, SETUP_PROTOCOL, canonical_hash, digest, diversity,
    safe_file, validate_indices, validate_study, verify_checkpoint_chain,
)

BASELINE = ISAAC.parents[1] / "baselines/isaac-expanded-farm-20261102"
HAS_TORCH = importlib.util.find_spec("torch") is not None


def bundle_fixture():
    from rl_film_study import make_study

    spec = json.loads((BASELINE / "contract.json").read_text())
    menus, _ = load_expanded_training_menu(BASELINE / "expanded-menu-qualified.json", spec)
    source = dict(menu_sha256=digest(BASELINE / "expanded-menu-qualified.json"),
                  camera_sha256=digest(BASELINE / "selected-camera.json"))
    return spec, menus, make_study(dict(spec=spec, menus=menus, source=source))


class AuditContractTests(unittest.TestCase):
    def test_signed_budget_keeps_historical_geometry_separate(self):
        spec, _, study = bundle_fixture()
        self.assertEqual(spec["training_batches"], 10)
        self.assertEqual(validate_study(study), study["fingerprint"])
        changed = copy.deepcopy(study)
        changed["training_batches"] = 10
        unsigned = dict(changed); unsigned.pop("fingerprint")
        changed["fingerprint"] = canonical_hash(unsigned)
        with self.assertRaisesRegex(ValueError, "plan changed"):
            validate_study(changed)
        changed = copy.deepcopy(study); changed["seed"] += 1
        with self.assertRaisesRegex(ValueError, "fingerprint changed"):
            validate_study(changed)

    def test_category_validation_rejects_bool_float_and_invalid_index(self):
        self.assertEqual(validate_indices([0] * 35).shape, (35,))
        for value in ([False] * 35, [1.0] * 35, [-1] * 35, [24] * 35, [1] * 34):
            with self.assertRaisesRegex(ValueError, "sampled menu categories"):
                validate_indices(value)

    def test_paths_cannot_escape_or_follow_a_symlink(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root / "ok.pt").write_bytes(b"fixture")
            self.assertEqual(safe_file(root, "ok.pt", ".pt"), root / "ok.pt")
            (root / "link.pt").symlink_to(root / "ok.pt")
            for name in ("../ok.pt", "/tmp/ok.pt", "link.pt", "missing.pt"):
                with self.assertRaises(ValueError):
                    safe_file(root, name, ".pt")

    def test_diversity_uses_selected_geometries_not_global_limits(self):
        spec, menus, _ = bundle_fixture()
        trial = dict(selected_indices=[0] * 35)
        summary = diversity([trial, copy.deepcopy(trial)], menus, spec["cells"])
        self.assertEqual(len(summary), 35)
        self.assertTrue(all(row["unique_selected_options"] == 1 for row in summary))
        self.assertTrue(all(row["executed_category_entropy"] == 0 for row in summary))
        self.assertEqual(summary[0]["executed_ranges"]["height_m"],
                         [menus[0][0]["height_m"]] * 2)


@unittest.skipUnless(HAS_TORCH, "Torch CPU is required for policy likelihood fixtures")
class CheckpointAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from farm_contract import reward, statistics
        from rl_film_policy import (
            MenuPolicy, POLICY_VERSION, actions_from_indices, actor_weight_hash,
            build_inputs, checkpoint_payload, sample, update, weight_hash,
        )

        torch.set_num_threads(2)
        torch.manual_seed(20261110)
        cls.fixture_temp = tempfile.TemporaryDirectory()
        cls.fixture = Path(cls.fixture_temp.name)
        cls.spec, cls.menus, cls.study = bundle_fixture()
        run, spec, menus, study = cls.fixture, cls.spec, cls.menus, cls.study
        for name in ("contract.json", "expanded-menu-qualified.json", "selected-camera.json"):
            shutil.copyfile(BASELINE / name, run / name)
        (run / "study.json").write_text(json.dumps(study))
        policy = MenuPolicy()
        optimizer = torch.optim.Adam(policy.parameters(), lr=study["ppo"]["lr"])
        obs_np, candidate_np = build_inputs(spec["cells"], menus)
        obs, candidate = torch.from_numpy(obs_np), torch.from_numpy(candidate_np)
        initial, initial_actor = weight_hash(policy), actor_weight_hash(policy)
        changes, summaries, assessments, buffer = [], [], [], []

        def checkpoint(filename, batch_count):
            torch.save(checkpoint_payload(
                policy, optimizer, study_fingerprint=study["fingerprint"],
                geometry_fingerprint=spec["fingerprint"], menu_sha256=study["menu_sha256"],
                updates=len(changes), training_batches=batch_count, seed=study["seed"],
            ), run / filename)
            return dict(checkpoint_file=filename, checkpoint_sha256=digest(run / filename),
                        policy_hash=weight_hash(policy), actor_hash=actor_weight_hash(policy))

        def trial(filename, mode, count, cp, repeat=None):
            picked = sample(policy, obs, candidate)
            indices = picked["indices"].tolist()
            actions = actions_from_indices(spec["cells"], menus, indices)
            rows = []
            for i, (cell, action, index) in enumerate(zip(spec["cells"], actions, indices)):
                good = (index + i) % 3 == 0
                # Deliberately synthetic, learnable terminal rewards. This is
                # only a policy/audit unit fixture, never native evidence.
                rows.append(dict(action, cell_id=cell["id"], bucket_xy_m=cell["bucket_xy_m"],
                    invalid=None, external_contact_steps=0, board_bounce=True,
                    closest_opening_distance_m=0.02 if good else 0.4,
                    max_retention_s=1.0 if good else 0.0, route_success=good,
                    route_quality=2 if good else 0, single_bounce_success=good))
            returns = torch.tensor([reward(row) for row in rows], dtype=torch.float32)
            value = dict(schema="sircus-native-rl-film-trial-v1", mode=mode, batch=count,
                source_fingerprint=spec["fingerprint"], study_fingerprint=study["fingerprint"],
                menu_sha256=study["menu_sha256"], policy_version=POLICY_VERSION,
                update_count_before=len(changes), setup_protocol=SETUP_PROTOCOL,
                release_enabled=[True] * 35, **cp, observations=obs_np.tolist(),
                candidate_features=candidate_np.tolist(), selected_indices=indices,
                old_logp=picked["logp"].tolist(), entropy=picked["entropy"].tolist(),
                actions=actions, rollout=rows, rewards=returns.tolist(),
                optimizer_samples=35 if mode == "train" else 0,
                setup_visualization="driven" if mode == "train" and count in FILMED else "not-recorded")
            if repeat is not None:
                value.update(after_train_batches=count, repeat=repeat,
                             assessment_seed=study["assessment_seed_base"] + repeat)
            (run / filename).write_text(json.dumps(value))
            summary = dict(batch=count, file=filename, file_sha256=digest(run / filename),
                optimizer_samples=value["optimizer_samples"], selected_indices=indices,
                **cp, **statistics(rows))
            return summary, (obs, candidate, picked["indices"], picked["logp"], returns), rows

        def assess(stage):
            cp = checkpoint(f"assessment-policy-{stage:03d}.pt", stage)
            files, all_rows = [], []
            for repeat in range(8):
                filename = f"assessment-{stage:03d}-{repeat:02d}.json"
                _, _, rows = trial(filename, "assessment", stage, cp, repeat)
                files.append(filename); all_rows.extend(rows)
            assessments.append(dict(after_train_batches=stage, files=files,
                update_count_before=len(changes), **cp, **statistics(all_rows)))

        assess(0)
        for index in range(128):
            cp = checkpoint(f"policy-{index:03d}.pt", index)
            summary, tensors, _ = trial(f"train-batch-{index:03d}.json", "train", index, cp)
            summaries.append(summary); buffer.append(tensors)
            if len(buffer) == 2:
                change = update(policy, optimizer, *[torch.cat(v) for v in zip(*buffer)], study["ppo"])
                change.update(optimizer_batches=[index - 1, index], optimizer_samples=70)
                changes.append(change); buffer.clear()
            if index + 1 in study["assessment_stages"]:
                assess(index + 1)
        final_cp = checkpoint("policy-final.pt", 128)
        report = dict(schema="sircus-native-rl-film-result-v1", status="completed",
            source_fingerprint=spec["fingerprint"], study_fingerprint=study["fingerprint"],
            study=study, source=study["source"], setup_protocol=SETUP_PROTOCOL,
            policy_version=POLICY_VERSION, workcells=35, articulations=70,
            menu_sha256=study["menu_sha256"], training_batches=summaries,
            filmed_batches=FILMED, assessments=assessments, updates=changes,
            training_attempts=4480, optimizer_samples=4480,
            initial_hash=initial, initial_actor_hash=initial_actor,
            final_hash=weight_hash(policy), final_actor_hash=actor_weight_hash(policy),
            final_checkpoint_file=final_cp["checkpoint_file"],
            final_checkpoint_sha256=final_cp["checkpoint_sha256"])
        (run / "result.json").write_text(json.dumps(report))

    @classmethod
    def tearDownClass(cls):
        cls.fixture_temp.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name) / "run"
        shutil.copytree(self.fixture, self.run)
        self.report = json.loads((self.run / "result.json").read_text())

    def write_report(self):
        (self.run / "result.json").write_text(json.dumps(self.report))

    def mutate_train(self, index, change):
        path = self.run / f"train-batch-{index:03d}.json"
        trial = json.loads(path.read_text()); change(trial)
        path.write_text(json.dumps(trial))
        self.report["training_batches"][index]["file_sha256"] = digest(path)
        self.write_report()

    def test_all_training_and_frozen_assessment_likelihoods_are_bound(self):
        result = verify_checkpoint_chain(self.run)
        self.assertEqual(result["schema"], SCHEMA)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["training_actions_checked"], 4480)
        self.assertEqual(result["assessment_actions_checked"], 1120)
        self.assertEqual(result["updates_checked"], 64)
        self.assertEqual(len(result["source_trial_sha256"]), 160)
        self.assertLess(result["category_logprob_max_abs_error"], 1e-6)
        self.assertNotEqual(result["initial_actor_hash"], result["final_actor_hash"])
        self.assertFalse((self.run / "checkpoint-audit.json").exists())

    def test_tampered_log_probability_and_candidate_features_are_rejected(self):
        self.mutate_train(2, lambda t: t["old_logp"].__setitem__(0, t["old_logp"][0] + 0.1))
        with self.assertRaisesRegex(ValueError, "log probabilities"):
            verify_checkpoint_chain(self.run)

    def test_native_reward_cannot_be_replaced_by_an_audio_or_capture_label(self):
        self.mutate_train(2, lambda t: t["rewards"].__setitem__(0, 99.0))
        with self.assertRaisesRegex(ValueError, "rewards differ"):
            verify_checkpoint_chain(self.run)

    def test_optimizers_cannot_consume_assessment_batches(self):
        self.report["updates"][0]["optimizer_batches"] = ["assessment-000-00", 1]
        self.write_report()
        with self.assertRaisesRegex(ValueError, "optimizer membership"):
            verify_checkpoint_chain(self.run)

    def test_category_action_and_scene_inputs_must_match(self):
        self.mutate_train(2, lambda t: t["candidate_features"][0][0].__setitem__(0, 0.123))
        with self.assertRaisesRegex(ValueError, "inputs do not match"):
            verify_checkpoint_chain(self.run)

    def test_assessment_policy_version_and_final_actor_cannot_change(self):
        self.report["assessments"][1]["policy_hash"] = "0" * 64
        self.write_report()
        with self.assertRaisesRegex(ValueError, "assessment checkpoint"):
            verify_checkpoint_chain(self.run)

    def test_checkpoint_bytes_are_verified_before_loading(self):
        path = self.run / "policy-002.pt"
        path.write_bytes(path.read_bytes() + b"unexpected")
        with self.assertRaisesRegex(ValueError, "checksum"):
            verify_checkpoint_chain(self.run)

    def test_assessment_common_seed_and_scripted_setup_cannot_drift(self):
        path = self.run / "assessment-032-03.json"
        trial = json.loads(path.read_text())
        trial["assessment_seed"] += 1
        path.write_text(json.dumps(trial))
        with self.assertRaisesRegex(ValueError, "common seed"):
            verify_checkpoint_chain(self.run)


if __name__ == "__main__":
    unittest.main()
