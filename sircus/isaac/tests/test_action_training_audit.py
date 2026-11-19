"""CPU fixtures prove recorded Gaussian likelihood and checkpoint-chain checks."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
from action_training_audit import SCHEMA, digest, verify_checkpoint_chain

HAS_TORCH = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(HAS_TORCH, "Torch CPU is required for checkpoint likelihood tests")
class ActionTrainingAuditTests(unittest.TestCase):
    def setUp(self):
        import torch
        from cylinder_farm import HIGH, LOGSTD, LOW, WARM, observation, specification
        from farm_policy import Policy, update, weight_hash

        torch.set_num_threads(2)
        torch.manual_seed(147)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name)
        scene = json.loads((SOURCE / "scene.json").read_text())
        self.contract = specification(scene, 10)
        (self.run / "contract.json").write_text(json.dumps(self.contract))
        policy = Policy(LOW, HIGH, WARM, LOGSTD)
        optimizer = torch.optim.Adam(policy.parameters(), lr=self.contract["lr"])
        obs = torch.tensor(
            [observation(c) for c in self.contract["cells"]], dtype=torch.float32
        )
        initial = weight_hash(policy)
        summaries, changes, buffer = [], [], []
        for index in range(10):
            before = weight_hash(policy)
            name = f"policy-{index:02}.pt"
            torch.save(
                dict(
                    state_dict=policy.state_dict(),
                    optimizer=optimizer.state_dict(),
                    source_fingerprint=self.contract["fingerprint"],
                    updates=index // 2,
                    menu_sha256="geometry-menu",
                    weight_hash=before,
                ),
                self.run / name,
            )
            with torch.no_grad():
                distribution = policy.distribution(obs)
                latent = distribution.sample()
                logp = distribution.log_prob(latent).sum(-1)
            returns = torch.sin(latent[:, 0])
            buffer.append((obs, latent, logp, returns))
            clip = dict(
                batch=index,
                checkpoint_file=name,
                checkpoint_sha256=digest(self.run / name),
                source_fingerprint=self.contract["fingerprint"],
                menu_sha256="geometry-menu",
                policy_hash=before,
                observation=obs.tolist(),
                latent=latent.tolist(),
                old_logp=logp.tolist(),
            )
            clip_name = f"clip-{index:02}.json"
            (self.run / clip_name).write_text(json.dumps(clip))
            summaries.append(dict(batch=index, file=clip_name, policy_hash=before))
            if index % 2:
                changes.append(
                    update(
                        policy,
                        optimizer,
                        *[torch.cat(v) for v in zip(*buffer)],
                        self.contract,
                    )
                )
                buffer.clear()
        final = weight_hash(policy)
        torch.save(
            dict(
                state_dict=policy.state_dict(),
                optimizer=optimizer.state_dict(),
                contract=self.contract,
                menu_sha256="geometry-menu",
                weight_hash=final,
            ),
            self.run / "policy-final.pt",
        )
        self.report = dict(
            schema="sircus-action-training-film-v1",
            source_fingerprint=self.contract["fingerprint"],
            fingerprint=self.contract["fingerprint"],
            workcells=35,
            articulations=70,
            initial_hash=initial,
            final_hash=final,
            menu_sha256="geometry-menu",
            batches=summaries,
            updates=changes,
            final_checkpoint_sha256=digest(self.run / "policy-final.pt"),
        )
        self.write_report()

    def write_report(self):
        (self.run / "result.json").write_text(json.dumps(self.report))

    def mutate_clip(self, change):
        path = self.run / "clip-02.json"
        value = json.loads(path.read_text())
        change(value)
        path.write_text(json.dumps(value))

    def test_verifies_real_cpu_checkpoints_and_binds_all_source_files(self):
        report = verify_checkpoint_chain(self.run)
        self.assertEqual(report["schema"], SCHEMA)
        self.assertEqual(report["latent_actions_checked"], 350)
        self.assertEqual(report["updates_checked"], 5)
        self.assertLess(report["latent_logprob_max_abs_error"], 1e-6)
        self.assertEqual(report["final_policy_hash"], self.report["final_hash"])
        self.assertEqual(len(report["source_clip_sha256"]), 10)
        self.assertEqual(len(report["checkpoint_sha256"]), 11)
        self.assertEqual(
            report["source_result_sha256"], digest(self.run / "result.json")
        )
        self.assertFalse((self.run / "checkpoint-audit.json").exists())

    def test_changed_likelihood_is_rejected(self):
        self.mutate_clip(lambda c: c["old_logp"].__setitem__(0, c["old_logp"][0] + 0.1))
        with self.assertRaisesRegex(ValueError, "log probabilities"):
            verify_checkpoint_chain(self.run)

    def test_changed_scene_observation_is_rejected_even_before_likelihood(self):
        self.mutate_clip(lambda c: c["observation"][0].__setitem__(0, 0.123))
        with self.assertRaisesRegex(ValueError, "observations do not match"):
            verify_checkpoint_chain(self.run)

    def test_checkpoint_checksum_and_update_chain_corruption_are_rejected(self):
        path = self.run / "policy-03.pt"
        raw = path.read_bytes()
        path.write_bytes(raw + b"changed")
        with self.assertRaisesRegex(ValueError, "checksum"):
            verify_checkpoint_chain(self.run)
        path.write_bytes(raw)
        self.report["updates"][1]["before"] = "different-policy"
        self.write_report()
        with self.assertRaisesRegex(ValueError, "update chain"):
            verify_checkpoint_chain(self.run)

    def test_final_weights_must_match_metadata_even_when_file_digest_is_updated(self):
        import torch

        path = self.run / "policy-final.pt"
        saved = torch.load(path, map_location="cpu", weights_only=True)
        saved["state_dict"]["logstd"][0] += 0.01
        torch.save(saved, path)
        self.report["final_checkpoint_sha256"] = digest(path)
        self.write_report()
        with self.assertRaisesRegex(ValueError, "Final policy weights"):
            verify_checkpoint_chain(self.run)

    def test_rejects_checkpoint_path_substitution(self):
        self.mutate_clip(
            lambda c: c.__setitem__("checkpoint_file", "../another-policy.pt")
        )
        with self.assertRaisesRegex(ValueError, "identity"):
            verify_checkpoint_chain(self.run)


if __name__ == "__main__":
    unittest.main()
