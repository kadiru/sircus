"""Synthetic CPU-shadow evidence and accounting checks; no native shots."""

import copy
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
import joint_feasibility_analysis as analysis


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, allow_nan=False).encode()
    path.write_bytes(gzip.compress(raw) if path.suffix == ".gz" else raw)
    return analysis.digest(path)


def accounting_record(index):
    admitted = index % 4 != 0
    released = index % 4 in (2, 3)
    captured = index % 4 == 3
    return dict(preflight_accepted=admitted, released=released, captured=captured,
                clean=index % 8 == 3,
                preflight_reason=None if admitted else "outer_link_length_bound",
                native_invalid=None if released else (
                    "endpoints_not_settled" if admitted else "outer_link_length_bound"),
                parity=True)


class ShadowTests(unittest.TestCase):
    def setUp(self):
        self.cell = dict(id="synthetic-cell", bucket_xy_m=[0.5, -0.4])
        self.action = dict(id="synthetic-action", marker="exact physical action")
        self.home = {n: [0.0] * 6 for n in ("Arm10", "Arm14")}
        self.links = {n: object() for n in self.home}
        self.targets = {n: np.eye(4) for n in self.home}
        self.targets["Arm10"][0, 3] = 0.3
        self.targets["Arm14"][0, 3] = 0.7
        self.row = dict(qualification=dict(accepted=True, reason=None,
            ik={n: dict(solved=True, residual=0.0, q=[0.0] * 6) for n in self.home}),
            path_proxy=dict(accepted=True, reasons=[]), invalid=None,
            route_success=True, route_quality=2)

    def decision(self, row=None):
        row = self.row if row is None else row
        accepted = bool(row["qualification"]["accepted"] and row.get("path_proxy")
                        and row["path_proxy"]["accepted"])
        reason = row["qualification"]["reason"]
        if reason is None and not accepted:
            reason = row["path_proxy"]["reasons"][0]
        return dict(accepted=accepted, qualification=copy.deepcopy(row["qualification"]),
                    path=copy.deepcopy(row.get("path_proxy")), reason=reason,
                    status="admitted" if accepted else "unresolved",
                    binding=dict(synthetic=True), fingerprint="f" * 64)

    def replay(self, row=None, side_effect=None):
        row = self.row if row is None else row

        def exact_preflight(scene, cell, action, links, starts, **kwargs):
            self.assertEqual(scene, {})
            self.assertEqual(cell, self.cell)
            self.assertEqual(action, self.action)
            self.assertEqual(set(kwargs), {"solver", "solver_protocol"})
            self.assertIn("no-new-search", kwargs["solver_protocol"])
            for name, saved in row["qualification"]["ik"].items():
                fit = kwargs["solver"](self.targets[name], links[name], starts[name], name == "Arm14")
                self.assertEqual(fit, saved)
                # Returning a defensive copy prevents the preflight consumer
                # from rewriting the historical answer.
                fit["q"][0] = 100.0
                self.assertEqual(saved["q"][0], 0.0)
            return self.decision(row)

        with patch.object(analysis, "endpoint_targets", return_value=self.targets), \
             patch.object(analysis, "fit_details", return_value={}) as verify, \
             patch.object(analysis, "preflight", side_effect=side_effect or exact_preflight) as preflight, \
             patch("robust_ik.solve", side_effect=AssertionError("Must not optimize IK in shadow analysis")):
            result = analysis.replay_request({}, self.cell, self.action, row, self.home, self.links)
        return result, verify, preflight

    def test_replay_consumes_only_saved_fits_and_preserves_original_action(self):
        before = copy.deepcopy((self.cell, self.action, self.row, self.home))
        result, verify, preflight = self.replay()
        self.assertTrue(result["exact_action_unchanged"])
        self.assertTrue(result["parity"])
        self.assertTrue(result["captured"])
        self.assertTrue(result["clean"])
        self.assertEqual(verify.call_count, 2)
        self.assertEqual(preflight.call_count, 1)
        self.assertEqual((self.cell, self.action, self.row, self.home), before)

    def test_original_outcomes_do_not_enter_preflight(self):
        captured, _, first = self.replay()
        row = copy.deepcopy(self.row)
        row.update(route_success=False, route_quality=0)
        missed, _, second = self.replay(row)
        self.assertNotEqual(captured["captured"], missed["captured"])
        first_args, second_args = first.call_args, second.call_args
        self.assertEqual(first_args.args[:4], second_args.args[:4])
        for name in self.home:
            np.testing.assert_array_equal(first_args.args[4][name], second_args.args[4][name])
        self.assertEqual(first_args.kwargs["solver_protocol"], second_args.kwargs["solver_protocol"])

    def test_preflight_admission_does_not_count_settling_failure_as_release(self):
        row = copy.deepcopy(self.row)
        row.update(invalid="endpoints_not_settled", route_success=False, route_quality=0)
        result, _, _ = self.replay(row)
        self.assertTrue(result["preflight_accepted"])
        self.assertFalse(result["released"])
        summary = analysis.summarize([result])
        self.assertEqual(summary["execution_declined_after_preflight"], 1)
        self.assertEqual(summary["capture_rate"], 0)
        self.assertIsNone(summary["capture_given_release"])

    def test_unresolved_ik_remains_declined_with_original_request_denominator(self):
        row = copy.deepcopy(self.row)
        row["qualification"] = dict(accepted=False, reason="endpoint_ik_unresolved_Arm10",
            ik={"Arm10": dict(solved=False, residual=0.1, q=[0.0] * 6)})
        row.update(path_proxy=None, invalid="endpoint_ik_unresolved_Arm10",
                   route_success=False, route_quality=0)
        result, _, _ = self.replay(row)
        self.assertEqual(result["preflight_status"], "unresolved")
        summary = analysis.summarize([result])
        self.assertEqual(summary["requests"], 1)
        self.assertEqual(summary["preflight_declined"], 1)
        self.assertEqual(summary["capture_rate"], 0)

    def test_changed_admission_endpoint_or_path_reason_fails_closed(self):
        for field in ("admission", "endpoint", "path"):
            def changed(*args, **kwargs):
                for name in self.home:
                    kwargs["solver"](self.targets[name], self.links[name], self.home[name], name == "Arm14")
                decision = self.decision()
                if field == "admission":
                    decision["accepted"] = False
                elif field == "endpoint":
                    decision["qualification"]["reason"] = "changed"
                else:
                    decision["path"]["reasons"] = ["changed"]
                return decision
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "differs|changed"):
                self.replay(side_effect=changed)

    def test_replay_refuses_missing_fit_or_changed_endpoint_objective(self):
        for change in ("missing", "target"):
            row = copy.deepcopy(self.row)
            if change == "missing":
                del row["qualification"]["ik"]["Arm14"]

            def changed(*args, **kwargs):
                target = self.targets["Arm14"].copy()
                if change == "target":
                    target[0, 3] += 0.001
                kwargs["solver"](target, self.links["Arm14"], self.home["Arm14"], True)
                return self.decision(row)

            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "absent|objective"):
                self.replay(row, changed)

    def test_replay_rejects_duplicate_or_reordered_solver_consumption(self):
        for order in (("Arm10", "Arm10", "Arm14"), ("Arm14", "Arm10")):
            def changed(*args, **kwargs):
                for name in order:
                    kwargs["solver"](self.targets[name], self.links[name], self.home[name], name == "Arm14")
                return self.decision()
            with self.subTest(order=order), self.assertRaisesRegex(ValueError, "sequence"):
                self.replay(side_effect=changed)

    def test_capture_and_clean_labels_cannot_bypass_original_release(self):
        for changes in (dict(invalid="endpoints_not_settled"),
                        dict(route_success=False, route_quality=2)):
            row = copy.deepcopy(self.row)
            row.update(changes)
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "capture"):
                self.replay(row)

    def test_accounting_keeps_requested_positions_and_conditional_rates_distinct(self):
        summary = analysis.summarize([accounting_record(i) for i in range(8)])
        self.assertEqual(summary["requests"], 8)
        self.assertEqual(summary["preflight_admitted"], 6)
        self.assertEqual(summary["preflight_declined"], 2)
        self.assertEqual(summary["execution_declined_after_preflight"], 2)
        self.assertEqual(summary["released"], 4)
        self.assertEqual(summary["released_failures"], 2)
        self.assertEqual(summary["captures"], 2)
        self.assertEqual(summary["clean"], 1)
        self.assertEqual(summary["capture_rate"], 0.25)
        self.assertEqual(summary["capture_given_release"], 0.5)
        self.assertEqual(summary["release_given_preflight"], 4 / 6)
        self.assertEqual(summary["parity_requests"], 8)
        with self.assertRaisesRegex(ValueError, "Empty"):
            analysis.summarize([])
        impossible = accounting_record(0)
        impossible["captured"] = True
        with self.assertRaisesRegex(ValueError, "accounting"):
            analysis.summarize([impossible])

    def test_evidence_hash_compression_and_nonfinite_json_guards(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "packet.json.gz"
            expected = write(path, dict(synthetic=True))
            self.assertEqual(analysis.read(path, expected), dict(synthetic=True))
            path.write_bytes(gzip.compress(b'{"synthetic": false}'))
            with self.assertRaisesRegex(ValueError, "SHA mismatch"):
                analysis.read(path, expected)
            bad = Path(temporary) / "bad.json"
            bad.write_text('{"value": NaN}')
            with self.assertRaisesRegex(ValueError, "Nonfinite"):
                analysis.read(bad)

    def test_evidence_paths_cannot_escape_by_parent_absolute_or_symlink(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            root = parent / "run"
            root.mkdir()
            outside = parent / "outside.json"
            outside.write_text("{}")
            (root / "escape").symlink_to(outside)
            self.assertEqual(analysis.contained(root, "trials/a.json"), (root / "trials/a.json").resolve())
            for relative in ("../outside.json", str(outside), "escape"):
                with self.subTest(relative=relative), self.assertRaisesRegex(ValueError, "relative|escaped"):
                    analysis.contained(root, relative)


class ProductionEvidenceTests(unittest.TestCase):
    """Generate bound mock production evidence rather than committing raw trials."""
    def fixture(self, root):
        seed = 20261121
        run = root / "runs" / f"ppo-seed{seed}"
        run.mkdir(parents=True)
        source = dict(geometry_fingerprint="a" * 64)
        experiment = dict(source=source, split_manifest_sha256="b" * 64)
        experiment_hash = write(root / "experiment.json", experiment)
        checkpoint_path = run / "checkpoints/final.pt"
        checkpoint_path.parent.mkdir()
        checkpoint_path.write_bytes(b"synthetic checkpoint bytes; not a Torch model")
        checkpoint = dict(file="checkpoints/final.pt", sha256=analysis.digest(checkpoint_path),
                          state_hash="c" * 64)
        splits, receipts, totals = {}, {}, {}
        for split, count in (("train", 35), ("validation", 35), ("test", 70)):
            splits[split] = [dict(id=f"{split}-{i}", bucket_xy_m=[i / 1000, -0.4], split=split)
                             for i in range(count)]
            records = [accounting_record(i) for i in range(count)]
            summary = analysis.summarize(records)
            totals[split] = dict(attempts=count, captures=summary["captures"], clean=summary["clean"],
                rejected=count-summary["released"])
            for cohort in range(count // 35):
                cells = splits[split][35 * cohort:35 * (cohort + 1)]
                rows = [dict(cell_id=c["id"], synthetic_record=records[35 * cohort + i])
                        for i, c in enumerate(cells)]
                packet = dict(source=source, split_manifest_sha256=experiment["split_manifest_sha256"],
                    checkpoint=checkpoint, mode="eval-"+split, cells=cells,
                    actions=[dict(id=c["id"]+"-action") for c in cells], rows=rows, enabled=[True]*35,
                    home=dict(measured_joint_rad={n: [[0.0]*6 for _ in range(35)] for n in ("Arm10", "Arm14")},
                              path_eligible=[True]*35))
                relative = f"trials/eval-{split}-{cohort}.json.gz"
                receipts[relative] = write(run / relative, packet)
        proof = dict(no_optimizer_or_flight_search=True, training_attempts=10000,
            checkpoint=checkpoint, state_hash_before=checkpoint["state_hash"],
            state_hash_after=checkpoint["state_hash"], source=source, source_trial_sha256=receipts)
        proof_hash = write(run / "proof-final.json", proof)
        final = dict(training_attempts=10000, proof_file="proof-final.json", **totals)
        progress = dict(status="completed", train=dict(attempts=10000, rejected=6344),
                        checkpoints=[final], source_trial_sha256=receipts)
        audit = dict(status="passed", smoke_only=False, method="ppo", seed=seed,
            paid_train_attempts=10000, source=source,
            split_manifest_sha256=experiment["split_manifest_sha256"],
            source_progress_sha256=write(run / "progress.json", progress),
            source_binding_sha256=write(run / "experiment-binding.json", dict(experiment_sha256=experiment_hash)),
            source_contract_sha256=write(run / "contract.json", dict(scene={})), source_trial_sha256=receipts)
        audit_hash = write(Path(str(run)+".audit.json"), audit)
        published = dict(seed=seed, independent_audit=dict(status="passed", sha256=audit_hash),
            checkpoint=dict(native_file=checkpoint["file"], sha256=checkpoint["sha256"], proof_sha256=proof_hash))
        return run, published, experiment, splits

    def test_complete_bound_mock_evidence_covers_all_original_final_requests(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, published, experiment, splits = self.fixture(root)

            def replay(scene, cell, action, row, home, links):
                return dict(cell_id=cell["id"], bucket_xy_m=cell["bucket_xy_m"],
                            **copy.deepcopy(row["synthetic_record"]))

            with patch.object(analysis, "replay_request", side_effect=replay) as replay_mock:
                result = analysis.analyze_run(root, published, experiment, splits, {})
            self.assertEqual(replay_mock.call_count, 140)
            self.assertEqual([result["splits"][s]["requests"] for s in ("train", "validation", "test")], [35, 35, 70])
            self.assertEqual(result["splits"]["test"]["captures"], 17)
            self.assertEqual(result["splits"]["test"]["preflight_declined"], 18)
            self.assertEqual(result["splits"]["test"]["execution_declined_after_preflight"], 18)
            self.assertEqual(result["original_training_requests"], 10000)
            self.assertEqual(result["original_training_rejected"], 6344)
            self.assertEqual(len(result["trial_sha256"]), 4)

    def test_audit_pending_or_mutated_checkpoint_never_yields_analysis(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, published, experiment, splits = self.fixture(root)
            pending = copy.deepcopy(published)
            pending["independent_audit"]["status"] = "pending"
            with self.assertRaisesRegex(ValueError, "has not passed"):
                analysis.analyze_run(root, pending, experiment, splits, {})
            (run / published["checkpoint"]["native_file"]).write_bytes(b"changed model")
            with self.assertRaisesRegex(ValueError, "model bytes changed"):
                analysis.analyze_run(root, published, experiment, splits, {})

    def test_unbound_trial_rewrite_fails_before_replaying_any_request(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run, published, experiment, splits = self.fixture(root)
            path = run / "trials/eval-train-0.json.gz"
            packet = analysis.read(path)
            packet["rows"][0]["synthetic_record"]["captured"] = True
            write(path, packet)
            with patch.object(analysis, "replay_request") as replay_mock, \
                 self.assertRaisesRegex(ValueError, "SHA mismatch"):
                analysis.analyze_run(root, published, experiment, splits, {})
            replay_mock.assert_not_called()

    def aggregate_fixture(self, root):
        source = dict(geometry_fingerprint="a"*64)
        experiment = dict(source=source, split_manifest_sha256="b"*64)
        experiment_hash = write(root / "experiment.json", experiment)
        splits_hash = write(root / "split-manifest.json", dict(synthetic=True))
        assets = root / "assets"
        assets.mkdir()
        asset_hashes = {}
        for name in ("dropper.urdf", "board_arm.urdf"):
            path = assets / name
            path.write_text("<robot name='synthetic' />")
            asset_hashes["assets/"+name] = analysis.digest(path)
        contract_hash = "c"*64
        receipt = dict(snapshot=dict(source=source),
                       dependency_sha256=dict(asset_hashes, **{"contract.json": contract_hash}))
        receipt_path = root / "geometry-receipt.json"
        write(receipt_path, receipt)
        published = dict(schema="sircus-shot-benchmark-final-comparison-v1", completed_audited_runs=6,
            experiment_sha256=experiment_hash, split_manifest_sha256=splits_hash, source=source,
            runs=[dict(method="ppo", seed=seed, independent_audit=dict(source_contract_sha256=contract_hash))
                  for seed in analysis.SEEDS])
        published_path = root / "published.json"
        write(published_path, published)
        return assets, published_path, receipt_path

    def test_aggregate_binds_original_geometry_and_reports_zero_new_work(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            assets, published, receipt = self.aggregate_fixture(root)
            one_run = dict(splits={s: dict(requests=n) for s, n in (("train", 35), ("validation", 35), ("test", 70))})
            with patch.object(analysis, "chain", return_value=[]), \
                 patch.object(analysis, "analyze_run", return_value=one_run) as analyze_run:
                result = analysis.analyze(root, assets, published, receipt)
            self.assertEqual(analyze_run.call_count, 3)
            self.assertEqual(result["requests"], 420)
            self.assertEqual(result["new_native_shots"], 0)
            self.assertEqual(result["optimizer_updates"], 0)
            self.assertEqual(result["changed_policy_actions"], 0)
            self.assertFalse(result["simulator_started"])
            self.assertFalse(result["release_authorized"])
            self.assertEqual(result["geometry_receipt_sha256"], analysis.digest(receipt))
            unsigned = dict(result)
            fingerprint = unsigned.pop("fingerprint")
            self.assertEqual(fingerprint, analysis.canonical_hash(unsigned))

    def test_changed_urdf_source_or_scene_contract_fails_before_shadow_replay(self):
        for change in ("urdf", "source", "contract"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                assets, published_path, receipt_path = self.aggregate_fixture(root)
                receipt = analysis.read(receipt_path)
                if change == "urdf":
                    (assets / "dropper.urdf").write_text("<robot name='changed' />")
                elif change == "source":
                    receipt["snapshot"]["source"]["geometry_fingerprint"] = "d"*64
                    write(receipt_path, receipt)
                else:
                    receipt["dependency_sha256"]["contract.json"] = "d"*64
                    write(receipt_path, receipt)
                with patch.object(analysis, "chain", return_value=[]), \
                     patch.object(analysis, "analyze_run") as analyze_run, \
                     self.assertRaisesRegex(ValueError, "receipt|contract"):
                    analysis.analyze(root, assets, published_path, receipt_path)
                analyze_run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
