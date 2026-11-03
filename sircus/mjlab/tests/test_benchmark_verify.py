"""Evidence identity and failure-denominator tests; no native engine or GPU."""

from collections import Counter
from contextlib import redirect_stdout
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from sircus.mjlab.benchmark_verify import (
    Checks,
    EvidenceError,
    PRIOR_PATH,
    behavior_log_prob,
    build_run_summary,
    check_behavior_likelihood,
    check_contract_protocol,
    check_row_invariants,
    contained,
    counts,
    digest,
    expected_observation,
    expected_perturbation,
    expected_schedule,
    check_resources,
    independent_sweep_prediction,
    reconstruct_sweep,
    network_forward,
    replay_ppo_update,
    row_reward,
    source_anchor,
    verify_prior_bundle,
    weights_sha256,
)


ROOT = Path(__file__).resolve().parents[3]


def row(identifier="trial-0", **overrides):
    value = dict(
        id=identifier,
        values=[0.6, -0.2, 0.3, 30.0, 0.2],
        bucket_xy_m=[0.6, -0.5],
        native_release=True,
        invalid=False,
        route_success=True,
        route_quality=2,
        board_bounce=True,
        whole_ball_entry=True,
        max_retention_s=1.0,
        closest_opening_distance_m=0.0,
        external_contact_steps=0,
        single_bounce_success=True,
    )
    value.update(overrides)
    return value


def refresh_manifests(bundle):
    """A forged manifest must not bypass semantic source bindings."""
    for kind in ("parent", "followup"):
        directory = bundle / kind
        p = directory / "manifest.json"
        manifest = json.loads(p.read_text())
        for name in manifest:
            manifest[name] = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        p.write_text(json.dumps(manifest, indent=2) + "\n")
    paths = sorted(
        p for p in bundle.rglob("*") if p.is_file() and p.name != "manifest.sha256"
    )
    (bundle / "manifest.sha256").write_text(
        "\n".join(
            hashlib.sha256(p.read_bytes()).hexdigest()
            + "  "
            + str(p.relative_to(bundle))
            for p in paths
        )
        + "\n"
    )



class RunSummaryTests(unittest.TestCase):
    def test_native_summary_and_reject_denominator_compose_without_duplicate_keys(self):
        rows = [
            row("native-1"),
            row("native-2"),
            row("rejected", native_release=False, invalid=True, route_success=False),
        ]
        native = dict(
            native_releases=2,
            provenance={"checked": True},
            resources={"wall_s": 3.0},
        )
        before = copy.deepcopy(native)
        result = build_run_summary("sweep", 20261311, 1024, rows, native)
        encoded = json.dumps(result)
        self.assertEqual(encoded.count('"native_releases"'), 1)
        self.assertEqual(result["requested"], 3)
        self.assertEqual(result["native_releases"], 2)
        self.assertEqual(result["method"], "sweep")
        self.assertEqual(result["seed"], 20261311)
        self.assertEqual(result["training_requests"], 1024)
        self.assertEqual(result["provenance"], native["provenance"])
        self.assertEqual(result["resources"], native["resources"])
        self.assertEqual(native, before)

    def test_mismatched_native_summary_count_is_rejected(self):
        with self.assertRaisesRegex(EvidenceError, "native count"):
            build_run_summary("ppo", 20261311, 1024, [row()], {"native_releases": 0})


class PriorTests(unittest.TestCase):
    def test_actual_published_prior_independently_matches_runner(self):
        from sircus.mjlab.benchmark_contract import load_prior

        prior = load_prior(ROOT / PRIOR_PATH)
        checks = Checks()
        checked = verify_prior_bundle(ROOT, prior, checks=checks)
        self.assertEqual(checked["records"], prior["records"])
        self.assertEqual(checked["records_sha256"], digest(checked["records"]))
        self.assertGreater(checks.count, 350)
        self.assertFalse(checked["full_historical_cost_known"])

    def test_modified_published_bytes_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary) / "bundle"
            shutil.copytree(ROOT / PRIOR_PATH, bundle)
            p = bundle / "followup/selected-settings.json"
            p.write_text(p.read_text() + " ")
            with self.assertRaisesRegex(EvidenceError, "published byte hash"):
                verify_prior_bundle(bundle)

    def test_forged_manifests_do_not_relabel_teacher_joint_configuration(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary) / "bundle"
            shutil.copytree(ROOT / PRIOR_PATH, bundle)
            p = bundle / "followup/selected-settings.json"
            values = json.loads(p.read_text())
            values[0]["configuration"]["q10"][0] += 0.01
            p.write_text(json.dumps(values, indent=2) + "\n")
            refresh_manifests(bundle)
            with self.assertRaisesRegex(EvidenceError, "exact configuration hash"):
                verify_prior_bundle(bundle)

    def test_prior_must_match_runner_records_and_known_cost_limits(self):
        from sircus.mjlab.benchmark_contract import load_prior

        prior = load_prior(ROOT / PRIOR_PATH)
        changed = copy.deepcopy(prior)
        changed["full_historical_cost_known"] = True
        with self.assertRaisesRegex(EvidenceError, "full_historical_cost_known"):
            verify_prior_bundle(ROOT, changed)
        changed = copy.deepcopy(prior)
        changed["records"][0]["q10"][0] += 0.01
        with self.assertRaisesRegex(EvidenceError, "/records"):
            verify_prior_bundle(ROOT, changed)

    def test_manifest_paths_cannot_escape_declared_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(EvidenceError):
                contained(temporary, "../secret")
            with self.assertRaises(EvidenceError):
                contained(temporary, "/absolute")


class TrialTests(unittest.TestCase):
    def test_rejected_requests_stay_in_attempts_and_cannot_capture(self):
        rejected = row(
            "rejected",
            native_release=False,
            invalid="geometry",
            route_success=False,
            route_quality=0,
        )
        values = [row(), rejected]
        check_row_invariants(values, Checks())
        self.assertEqual(counts(values)["attempts"], 2)
        self.assertEqual(counts(values)["native_releases"], 1)
        self.assertEqual(counts(values)["captures"], 1)
        self.assertEqual(counts(values)["invalid"], 1)
        rejected["route_success"] = True
        with self.assertRaisesRegex(EvidenceError, "rejection cannot capture"):
            check_row_invariants(values, Checks())

    def test_duplicate_trial_ids_and_nonfinite_actions_fail(self):
        with self.assertRaisesRegex(EvidenceError, "Unique requested"):
            check_row_invariants([row(), row()], Checks())
        invalid = row()
        invalid["values"][0] = float("nan")
        with self.assertRaisesRegex(EvidenceError, "finite five-action"):
            check_row_invariants([invalid], Checks())

    def test_capture_needs_complete_board_route_and_full_retention(self):
        for override in (
            {"board_bounce": False},
            {"whole_ball_entry": False},
            {"max_retention_s": 0.999},
            {"external_contact_steps": 1},
        ):
            with self.subTest(override=override), self.assertRaisesRegex(
                EvidenceError, "capture invariants"
            ):
                check_row_invariants([row(**override)], Checks())

    def test_external_contact_keeps_capture_bonus_out_of_reward(self):
        external = row(route_success=False, route_quality=0, external_contact_steps=1)
        self.assertAlmostEqual(row_reward(external), 0.15)
        self.assertAlmostEqual(row_reward(row()), 4.55)
        self.assertEqual(row_reward(row(native_release=False, invalid="bounds")), -1)

    def test_recorded_reward_is_independently_checked(self):
        value = row()
        value["learning_reward"] = row_reward(value)
        check_row_invariants([value], Checks())
        value["learning_reward"] += 0.01
        with self.assertRaisesRegex(EvidenceError, "terminal reward"):
            check_row_invariants([value], Checks())


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from sircus.mjlab.benchmark_contract import load_prior, make_splits, protocol

        cls.prior = load_prior(ROOT / PRIOR_PATH)
        cls.contract = dict(
            version="sircus-sweep-rl-benchmark-v1",
            prior_sha256=digest(cls.prior),
            splits=make_splits(),
            protocol=protocol(),
        )
        cls.contract["protocol"]["prior"] = {
            k: copy.deepcopy(v) for k, v in cls.prior.items() if k != "records"
        }
        cls.contract["fingerprint"] = digest(cls.contract)

    def test_fixed_pools_and_equal_budget_contract_pass_independent_reconstruction(
        self,
    ):
        checked = check_contract_protocol(self.contract, self.prior, Checks())
        self.assertEqual(len(checked), 71)
        self.assertEqual(
            Counter(role for role, _ in checked.values()),
            {"train": 35, "validation": 12, "test": 24},
        )

    def test_relabelled_heldout_coordinate_fails_after_refingerprinting(self):
        contract = copy.deepcopy(self.contract)
        contract["splits"]["test"][0]["bucket_xy_m"] = contract["splits"]["train"][0][
            "bucket_xy_m"
        ]
        contract.pop("fingerprint")
        contract["fingerprint"] = digest(contract)
        with self.assertRaisesRegex(EvidenceError, "Prespecified layout coordinate"):
            check_contract_protocol(contract, self.prior, Checks())

    def test_schedule_and_observation_have_no_method_dependent_inputs(self):
        from sircus.mjlab import benchmark_contract as runner
        import numpy as np

        for seed in self.contract["protocol"]["seeds"]:
            self.assertEqual(expected_schedule(seed, 1024), runner.schedule(seed, 1024))
        bucket = [0.435, -0.317]
        anchor = source_anchor(bucket, self.prior["records"])
        self.assertEqual(anchor, runner.anchor_for(bucket, self.prior["records"]))
        np.testing.assert_array_equal(
            expected_observation(bucket, anchor), runner.observation(bucket, anchor)
        )
        for role in ("validation", "test"):
            for context in self.contract["splits"][role]:
                for repeat in range(3):
                    np.testing.assert_array_equal(
                        expected_perturbation(context["id"], repeat),
                        runner.perturbation(context["id"], repeat),
                    )


class LikelihoodTests(unittest.TestCase):
    def test_independent_gaussian_likelihood_and_tampering_detection(self):
        import numpy as np

        state = {"logstd": np.full(5, -1.5, dtype=np.float32)}
        for prefix, width in (("actor", 5), ("critic", 1)):
            for layer, inputs, outputs in ((0, 9, 64), (2, 64, 64), (4, 64, width)):
                state[f"{prefix}.{layer}.weight"] = np.zeros(
                    (outputs, inputs), dtype=np.float32
                )
                state[f"{prefix}.{layer}.bias"] = np.zeros(outputs, dtype=np.float32)
        obs = np.zeros((4, 9), dtype=np.float32)
        latent = np.zeros((4, 5), dtype=np.float32)
        np.testing.assert_array_equal(network_forward(state, obs), latent)
        expected = np.full(4, 5 * (1.5 - 0.5 * np.log(2 * np.pi)))
        np.testing.assert_allclose(
            behavior_log_prob(state, obs, latent), expected, atol=1e-6
        )
        check_behavior_likelihood(state, obs, latent, expected, Checks(), "batch")
        with self.assertRaisesRegex(EvidenceError, "behavior log probabilities"):
            check_behavior_likelihood(
                state, obs, latent, expected + 0.01, Checks(), "batch"
            )
        changed = copy.deepcopy(state)
        changed["actor.4.bias"][0] += 0.001
        self.assertNotEqual(weights_sha256(state), weights_sha256(changed))

    @unittest.skipUnless(
        importlib.util.find_spec("torch"),
        "CPU PyTorch audit runs in the pinned Ubuntu environment",
    )
    def test_cpu_optimizer_update_is_reproduced_and_parameter_tampering_fails(self):
        import numpy as np
        import torch
        from sircus.mjlab.benchmark_policy import ResidualPolicy
        from sircus.mjlab.learning_policy import update

        torch.set_num_threads(1)
        torch.manual_seed(123)
        policy = ResidualPolicy().cpu()
        optimizer = torch.optim.Adam(policy.parameters(), lr=0.0003)
        generator = np.random.default_rng(7)
        for _ in range(2):
            obs = torch.tensor(generator.normal(size=(32, 9)), dtype=torch.float32)
            with torch.no_grad():
                distribution = policy.distribution(obs)
                latent = distribution.sample()
                old = distribution.log_prob(latent).sum(-1)
            reward = torch.tensor(generator.normal(size=32), dtype=torch.float32)
            before = copy.deepcopy(policy.state_dict())
            state = copy.deepcopy(optimizer.state_dict())
            buffer = dict(
                obs=obs.numpy(),
                latent=latent.numpy(),
                old_logp=old.numpy(),
                reward=reward.numpy(),
            )
            update(policy, optimizer, obs, latent, old, reward)
            after = copy.deepcopy(policy.state_dict())
            replay_ppo_update(before, state, after, buffer, Checks(), "batch")
        after["actor.4.bias"][0] += 0.001
        with self.assertRaisesRegex(EvidenceError, "replayed PPO parameter"):
            replay_ppo_update(before, state, after, buffer, Checks(), "tampered")


@unittest.skipUnless(importlib.util.find_spec("torch"), "Execution sweep module is tested in the pinned Ubuntu CPU environment")
class SweepReplayTests(unittest.TestCase):
    def test_full_sweep_schedule_archive_and_training_only_dependencies_replay(self):
        import numpy as np
        from sircus.mjlab.benchmark_contract import action_from_residual, load_prior, make_splits
        from sircus.mjlab.benchmark_policy import GuidedSweep
        # This test is only imported on environments with PyTorch because the
        # execution policy module includes its CPU actor implementation.
        prior = load_prior(ROOT / PRIOR_PATH)
        train = make_splits()["train"]
        seed = 20261311
        learner = GuidedSweep(seed, train)
        snapshots, rows = {0: learner.state()}, []
        # state() is intentionally a shallow view during execution; persisted
        # snapshots are exact immutable JSON values.
        snapshots[0] = copy.deepcopy(snapshots[0])
        schedule = expected_schedule(seed, 1024)
        for offset in range(0, 1024, 32):
            batch = []
            for index in range(offset, offset + 32):
                context = train[schedule[index]]
                anchor = source_anchor(context["bucket_xy_m"], prior["records"])
                residual, metadata = learner.propose(context, anchor)
                failed = index % 11 == 0
                captured = index % 3 == 0 and not failed
                item = row(
                    f"sweep-{index}", role="train", context_id=context["id"],
                    values=action_from_residual(anchor["values"], residual).tolist(),
                    normalized_residual=residual.tolist(), invalid=failed,
                    native_release=not failed, route_success=captured,
                    route_quality=2 if captured else 0,
                    closest_opening_distance_m=(index % 7) * .01,
                    **metadata,
                )
                batch.append(item)
            for item in batch:
                learner.observe(item)
            rows.extend(batch)
            if len(rows) in (256, 512, 1024):
                snapshots[len(rows)] = copy.deepcopy(learner.state())
        result = reconstruct_sweep(seed, train, rows, Checks(), snapshots=snapshots)
        self.assertEqual(result, learner.state())
        context = make_splits()["test"][0]
        anchor = source_anchor(context["bucket_xy_m"], prior["records"])
        predicted, meta = independent_sweep_prediction(result, context, anchor)
        values, original_meta = learner.predict(context, anchor)
        np.testing.assert_array_equal(predicted, values)
        self.assertEqual(meta, original_meta)
        changed = copy.deepcopy(rows)
        dependent = next(r for r in changed if r["source_training_ids"])
        dependent["source_training_ids"] = ["leaked-test-label"]
        with self.assertRaisesRegex(EvidenceError, "TRAIN-only dependencies"):
            reconstruct_sweep(seed, train, changed, Checks(), snapshots=snapshots)


class ResourceTests(unittest.TestCase):
    def test_observed_sampled_peak_and_separate_training_timing(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            resource = dict(wall_s=2., cpu_s=1., geometry_s=0., gpu_s=0.,
                            setup_s=2., evaluation_wall_s=1., gpu_reservation_s=3.1,
                            peak_rss_mb=50., peak_vram_mb=120., torch_allocated_mb=10.,
                            torch_reserved_mb=20., vram_samples=2, vram_sampler_errors=0)
            events = [
                dict(event="nvidia_process_sample", values=dict(used_vram_mib=100.)),
                dict(event="snapshot", values=resource),
                # Sampler event can be written just after a snapshot observes
                # the updated counter; order by sample count, not timestamps.
                dict(event="nvidia_process_sample", values=dict(used_vram_mib=120.)),
            ]
            (directory / "resource-samples.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n")
            (directory / "batches.jsonl").write_text("")
            report = dict(curves=[dict(samples=0, resources=resource)], resources=resource)
            check_resources(directory, report, Checks())
            bad = copy.deepcopy(resource)
            bad["peak_vram_mb"] = 0.
            with (directory / "resource-samples.jsonl").open("a") as stream:
                stream.write(json.dumps(dict(event="snapshot", values=bad)) + "\n")
            report["resources"] = bad
            with self.assertRaisesRegex(EvidenceError, "sampled process peak"):
                check_resources(directory, report, Checks())

    def test_missing_gpu_sample_stays_unknown(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            resource = dict(wall_s=0., evaluation_wall_s=0., gpu_reservation_s=.1,
                            geometry_s=0., gpu_s=0., peak_vram_mb=None,
                            torch_allocated_mb=None, torch_reserved_mb=None, vram_samples=0)
            (directory / "resource-samples.jsonl").write_text(json.dumps(dict(event="snapshot", values=resource)) + "\n")
            (directory / "batches.jsonl").write_text("")
            check_resources(directory, dict(curves=[], resources=resource), Checks())


class CliTests(unittest.TestCase):
    def test_failed_audit_writes_failure_and_never_returns_success(self):
        from sircus.mjlab.benchmark_verify import main
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            study = root / "study"
            study.mkdir()
            output = root / "failed-audit.json"
            argv = ["benchmark_verify", "--study", str(study), "--reference", str(root / "reference"),
                    "--assets", str(root / "assets"), "--repository", str(ROOT), "--output", str(output)]
            with patch("sys.argv", argv), patch("sircus.mjlab.benchmark_verify.audit", side_effect=EvidenceError("bad evidence")), redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    main()
            self.assertEqual(caught.exception.code, 1)
            result = json.loads(output.read_text())
            self.assertEqual(result["status"], "failed")
            self.assertTrue(result["failures"])
            self.assertEqual(list(study.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
