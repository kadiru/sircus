import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from farm_transfer_probe import VERSION, audit_transfer_probe, validate_checkpoint
from farm_learning import FEATURE_NAMES, VERSION as LEARNING_VERSION

try:
    import torch
except ImportError:
    torch = None


def metadata():
    result = dict(
        policy_kind="candidate-conditioned-categorical",
        source_fingerprint="a" * 64,
        native_bank_sha256="b" * 64,
        updates=4,
        study_contract=dict(
            source_fingerprint="a" * 64,
            version=LEARNING_VERSION,
            candidate_feature_names=list(FEATURE_NAMES),
            actor_input_dim=11,
            observation_dim=5,
            num_envs=35,
            training_batches=8,
            fingerprint="c" * 64,
        ),
    )
    from farm_learning import digest

    unsigned = copy.deepcopy(result["study_contract"])
    unsigned.pop("fingerprint")
    result["study_contract"]["fingerprint"] = digest(unsigned)
    return result


class TestTransferCompatibility(unittest.TestCase):
    def test_completed_matching_candidate_policy_is_required(self):
        saved = metadata()
        self.assertEqual(
            validate_checkpoint(saved, {"fingerprint": "a" * 64}),
            saved["study_contract"],
        )
        for key, value in (
            ("policy_kind", "continuous-gaussian"),
            ("source_fingerprint", "other-scene"),
            ("updates", 0),
            ("native_bank_sha256", "missing"),
        ):
            changed = copy.deepcopy(saved)
            changed[key] = value
            with self.assertRaises(ValueError):
                validate_checkpoint(changed, {"fingerprint": "a" * 64})

    def test_feature_contract_cannot_be_guessed_from_weight_shape(self):
        for key, value in (
            ("candidate_feature_names", ["unknown"] * 6),
            ("actor_input_dim", 10),
            ("observation_dim", 6),
            ("num_envs", 34),
            ("version", "other-version"),
        ):
            saved = metadata()
            saved["study_contract"][key] = value
            with self.assertRaises(ValueError):
                validate_checkpoint(saved, {"fingerprint": "a" * 64})


@unittest.skipIf(
    torch is None, "Torch unavailable on Mac; run in Ubuntu CPU environment"
)
class TestTransferAudit(unittest.TestCase):
    def test_real_frozen_likelihood_audit_and_tamper_rejection(self):
        from expanded_farm import HIGH, LOW, shot
        from farm_learning import (
            candidate_tensor_data,
            choose_actions,
            digest,
            make_policy,
            per_cell,
            summary,
        )
        from farm_learning_runtime import file_hash
        from farm_policy import weight_hash

        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            cells = [
                dict(id=f"cell-{i}", bucket_xy_m=[0.4 + i * 0.01, -0.2])
                for i in range(35)
            ]
            menus = [
                [shot(c, [0, 0.4, 0.3, 0.2, 25]), shot(c, [5, 0.4, 0.32, 0.22, 28])]
                for c in cells
            ]
            spec = dict(
                fingerprint="a" * 64,
                cells=cells,
                action_low=LOW.tolist(),
                action_high=HIGH.tolist(),
            )
            (run / "contract.json").write_text(json.dumps(spec))
            bank = dict(menus=menus, source_fingerprint=spec["fingerprint"])
            bank["fingerprint"] = digest(bank)
            (run / "native-qualified-transfer-bank.json").write_text(json.dumps(bank))
            model = make_policy().cpu()
            saved = metadata()
            saved.update(state_dict=model.state_dict(), weight_hash=weight_hash(model))
            torch.save(saved, run / "source-policy.pt")
            contract = dict(
                version=VERSION,
                source_fingerprint=spec["fingerprint"],
                source_learning_fingerprint=saved["study_contract"]["fingerprint"],
                source_policy_weight_hash=saved["weight_hash"],
                source_checkpoint_sha256=file_hash(run / "source-policy.pt"),
                source_native_bank_sha256=saved["native_bank_sha256"],
                evaluation_repeats=2,
            )
            contract["fingerprint"] = digest(contract)
            (run / "transfer-probe-contract.json").write_text(json.dumps(contract))
            _, features, mask = candidate_tensor_data(cells, menus, LOW, HIGH)
            features, mask = torch.tensor(features), torch.tensor(mask)
            result = dict(
                version=VERSION,
                fingerprint=contract["fingerprint"],
                parent_study_fingerprint=saved["study_contract"]["fingerprint"],
                source_fingerprint=spec["fingerprint"],
                source_learning_fingerprint=saved["study_contract"]["fingerprint"],
                source_checkpoint_sha256=contract["source_checkpoint_sha256"],
                source_native_bank_sha256=saved["native_bank_sha256"],
                policy_updates=0,
                transfer_native_bank_sha256=file_hash(
                    run / "native-qualified-transfer-bank.json"
                ),
                source_policy_weight_hash=saved["weight_hash"],
                stages={},
                summary={},
                per_cell={},
            )
            for deterministic, count in ((True, 1), (False, 2)):
                label = (
                    "transfer_deterministic" if deterministic else "transfer_stochastic"
                )
                stage_rows = []
                result["stages"][label] = []
                for number in range(count):
                    with torch.no_grad():
                        distribution = model.distribution(features, mask)
                        indices = (
                            distribution.probs.argmax(-1)
                            if deterministic
                            else distribution.sample()
                        )
                        logp = distribution.log_prob(indices)
                    actions = choose_actions(menus, indices.numpy())
                    rows = [
                        dict(
                            cell_id=c["id"],
                            invalid=None,
                            board_bounce=True,
                            route_success=False,
                            route_quality=0,
                            single_bounce_success=False,
                            external_contact_steps=0,
                            closest_opening_distance_m=0.2,
                            max_retention_s=0,
                        )
                        for c in cells
                    ]
                    batch = dict(
                        transfer_fingerprint=contract["fingerprint"],
                        source_checkpoint_sha256=contract["source_checkpoint_sha256"],
                        source_native_bank_sha256=contract["source_native_bank_sha256"],
                        transfer_native_bank_sha256=result[
                            "transfer_native_bank_sha256"
                        ],
                        policy_hash=saved["weight_hash"],
                        actions=actions,
                        rows=rows,
                        enabled=[True] * 35,
                        sampled_indices=indices.tolist(),
                        log_prob=logp.tolist(),
                    )
                    filename = f"{label}-{number}.json"
                    (run / filename).write_text(json.dumps(batch))
                    result["stages"][label].append(
                        dict(
                            file=filename,
                            sha256=file_hash(run / filename),
                            **summary(rows),
                        )
                    )
                    stage_rows.extend(rows)
                result["summary"][label] = summary(stage_rows)
                result["per_cell"][label] = per_cell(cells, stage_rows)
            result["stages"]["probe_sweep"] = []
            result["summary"]["probe_sweep"] = summary([])
            result["per_cell"]["probe_sweep"] = per_cell(cells, [])
            (run / "transfer-probe-result.json").write_text(json.dumps(result))
            audit = audit_transfer_probe(run)
            self.assertEqual(audit["frozen_evaluation_samples_checked"], 105)
            self.assertEqual(audit["policy_updates"], 0)
            self.assertLess(audit["log_probability_max_abs_error"], 1e-6)
            batch_path = run / result["stages"]["transfer_stochastic"][0]["file"]
            batch_path.write_text(batch_path.read_text() + " ")
            with self.assertRaisesRegex(ValueError, "batch checksum"):
                audit_transfer_probe(run)


if __name__ == "__main__":
    unittest.main()
