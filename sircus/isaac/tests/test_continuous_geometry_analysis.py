"""Synthetic provenance/accounting tests; no model, simulator or original data."""

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
import continuous_geometry_analysis as analysis
import candidate_train_evidence as collector
from continuous_feasibility import decode_action


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, separators=(",", ":"), allow_nan=False)+"\n")
    return analysis.digest(path)


def fixture(root):
    """Small fake lineage, with deliberately fake arm files never solved."""
    benchmark, assets = root/"benchmark", root/"assets"
    run = benchmark/"runs/ppo-seed20261122"
    run.mkdir(parents=True)
    assets.mkdir()
    source = dict(commit="synthetic", archive_sha256="a"*64)
    scene = json.loads((SOURCE/"scene.json").read_text())
    cells = [dict(id=f"train-{i:03d}", split="train", bucket_xy_m=[.2+i*.01, -.1]) for i in range(35)]
    source_binding_sha = save(run/"experiment-binding.json", dict(source=source))
    contract_sha = save(run/"contract.json", dict(scene=scene))
    split_sha = save(benchmark/"splits.json", dict(train=[dict(c, tile_index=i) for i,c in enumerate(cells)]))
    experiment_sha = save(benchmark/"experiment.json", dict(source=source, split_manifest_file="splits.json"))
    model = run/"checkpoints/final.pt"
    model.parent.mkdir()
    model.write_bytes(b"synthetic bytes: must never be loaded")
    model_sha = analysis.digest(model)
    progress_sha = save(run/"progress.json", dict(source=source, checkpoints=[dict(training_attempts=10000,
        checkpoint_file="checkpoints/final.pt", checkpoint_sha256=model_sha)]))
    audit_sha = save(Path(str(run)+".audit.json"), dict(status="passed", smoke_only=False,
        paid_train_attempts=10000, seed=20261122, source_progress_sha256=progress_sha,
        source_sha256={name:analysis.digest(SOURCE/name) for name in analysis.CORE_FILES}))
    binding = dict(audit_sha256=audit_sha, progress_sha256=progress_sha,
        experiment_sha256=experiment_sha, split_manifest_sha256=split_sha,
        checkpoint_sha256=model_sha, binding_sha256=source_binding_sha, contract_sha256=contract_sha)
    published = root/"published.json"
    save(published, dict(source=source, experiment_sha256=experiment_sha, split_manifest_sha256=split_sha,
        runs=[dict(method="ppo", seed=20261122, independent_audit=dict(sha256=audit_sha,
                   source_progress_sha256=progress_sha), checkpoint=dict(sha256=model_sha))]))
    asset_sha = {}
    for name in ("dropper.urdf", "board_arm.urdf"):
        path = assets/name
        path.write_text("<robot name='synthetic'/>")
        asset_sha["assets/"+name] = analysis.digest(path)
    receipt = root/"baselines/isaac-reach-ik-audit-20261106/summary.json"
    save(receipt, dict(snapshot=dict(source=source), dependency_sha256=dict(asset_sha, **{"contract.json":contract_sha})))
    successes = []
    for i in range(4):
        action = decode_action(cells[i], [0., .4, .35, .24, .5], "synthetic-"+str(i))
        relative = f"trials/train-{i:05d}.json.gz"
        successes.append(dict(success_key=relative+"#0", trial_file=relative, trial_sha256=str(i)*64,
            slot=0, cell=copy.deepcopy(cells[i]), action=action, recorded_starts_rad={n:[0.]*6 for n in collector.ARMS},
            qualification=dict(accepted=True, action=copy.deepcopy(action),
                ik={n:dict(q=[0.]*6, solved=True, residual=0.) for n in collector.ARMS}),
            path_proxy=dict(accepted=True), clean=i<2))
    manifest = {f"trials/train-{i:05d}.json.gz":str(i%10)*64 for i in range(289)}
    evidence = dict(schema=collector.VERSION, seed=20261122, source=source, source_bindings=binding,
        scene=scene, cells=cells, successful_shots=4, clean_successes=2,
        training_totals=dict(attempts=10000, captures=4, clean=2), train_trial_sha256=manifest,
        successes=successes, evaluated_trial_files_read=0, new_native_shots=0, optimizer_updates=0)
    evidence["fingerprint"] = collector.canonical_hash(evidence)
    export = root/"train.json"
    save(export, evidence)
    return dict(export=export, benchmark=benchmark, assets=assets, published=published,
                model=model, fingerprint=evidence["fingerprint"], evidence=evidence)


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.f = fixture(self.root)

    def load(self):
        sha = analysis.digest(self.f["export"])
        with patch.object(analysis, "TRAIN_EXPORT_SHA256", sha), \
             patch.object(analysis, "TRAIN_EXPORT_FINGERPRINT", self.f["fingerprint"]), \
             patch.object(analysis, "SUCCESS_COUNT", 4), patch.object(analysis, "CLEAN_COUNT", 2), \
             patch.object(analysis, "ROOT", self.root), patch.object(analysis, "chain", return_value=[]):
            return analysis.load_inputs(self.f["export"], self.f["benchmark"], self.f["assets"],
                                        self.f["published"], expected_export_sha256=sha)

    def test_loader_uses_measured_starts_and_never_success_witnesses(self):
        inputs = self.load()
        self.assertEqual(len(inputs["cases"]), 4)
        self.assertEqual(sum(c["clean"] for c in inputs["cases"]), 2)
        self.assertTrue(all(c["starts"] == c["seeds"] for c in inputs["cases"]))
        self.assertTrue(all("qualification" not in c and "ik" not in c for c in inputs["cases"]))
        self.assertFalse(inputs["bindings"]["historical_endpoint_witnesses_used"])
        self.assertFalse(inputs["bindings"]["model_loaded"])
        self.assertFalse(inputs["bindings"]["candidate_bank_read"])
        self.assertIn(str(self.f["model"].resolve()), inputs["protected_before"])
        self.assertIn("numpy", inputs["bindings"]["runtime"])

    def test_wrong_raw_export_hash_or_fingerprint_fails(self):
        with self.assertRaisesRegex(ValueError, "export SHA"):
            analysis.load_inputs(self.f["export"], self.f["benchmark"], self.f["assets"], self.f["published"])
        raw = self.f["evidence"]
        raw["successful_shots"] = 3
        save(self.f["export"], raw)
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            self.load()

    def test_changed_model_or_asset_is_rejected_without_loading(self):
        self.f["model"].write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "model bytes changed"):
            self.load()
        self.f["model"].write_bytes(b"synthetic bytes: must never be loaded")
        (self.f["assets"]/"dropper.urdf").write_text("changed")
        with self.assertRaisesRegex(ValueError, "arm asset changed"):
            self.load()

    def test_case_identity_and_derived_action_are_checked(self):
        for change, expected in ((lambda e:e["successes"][0]["cell"].update(split="test"), "non-TRAIN"),
                (lambda e:e["successes"][0]["action"].update(board_x_m=99), "shot geometry")):
            with self.subTest(expected=expected):
                self.f = fixture(self.root/expected.replace(" ", "-"))
                value = self.f["evidence"]
                change(value)
                value.pop("fingerprint")
                value["fingerprint"] = collector.canonical_hash(value)
                self.f["fingerprint"] = value["fingerprint"]
                save(self.f["export"], value)
                with self.assertRaisesRegex(ValueError, expected):
                    self.load()

    def inputs(self):
        value = self.load()
        value["synthetic"] = True
        return value

    @staticmethod
    def fake(job):
        case = job[2]
        index = int(case["cell"]["id"].split("-")[-1])
        classification = ("unchanged", "adjusted", "rejected", "unchanged")[index]
        reason = None if classification == "unchanged" else "endpoint_ik_unresolved_Arm14"
        return dict(case_id=case["case_id"], cell_id=case["cell"]["id"], clean=case["clean"],
            result=dict(classification=classification, original_preflight=dict(reason=reason,
                accepted=classification=="unchanged"), reason=None if classification!="rejected" else "no_verified_repair_within_bounded_work"),
            wall_s=.1, cpu_s=.05)

    def test_analysis_keeps_every_case_and_clean_denominator(self):
        inputs = self.inputs()
        updates = []
        report = analysis.analyze_cases(inputs, evaluate=self.fake, progress=updates.append)
        self.assertEqual(report["counts"], dict(requested=4, unchanged=2, adjusted=1, rejected=1))
        self.assertEqual(report["clean_counts"], dict(requested=2, unchanged=1, adjusted=1, rejected=0))
        self.assertEqual(report["reasons"], {"endpoint_ik_unresolved_Arm14":2})
        self.assertEqual(report["protected_after"], inputs["protected_before"])
        self.assertEqual(report["optimizer_updates"], report["new_native_shots"])
        self.assertEqual(report["worker_cpu_s"], .2)
        self.assertEqual(updates[-1]["completed"], 4)
        self.assertEqual(analysis.unsigned(report)["counts"], report["counts"])

    def test_real_evidence_cannot_replace_geometry_or_use_witness_seed(self):
        inputs = self.load()
        with self.assertRaisesRegex(ValueError, "cannot replace"):
            analysis.analyze_cases(inputs, evaluate=self.fake)
        inputs["synthetic"] = True
        inputs["cases"][0]["seeds"]["Arm10"][0] = .1
        with self.assertRaisesRegex(ValueError, "forbids historical"):
            analysis.analyze_cases(inputs, evaluate=self.fake)

    def test_duplicate_case_missing_clean_or_unbounded_workers_fails(self):
        for change in (lambda v:v["cases"].__setitem__(1, v["cases"][0]),
                       lambda v:v["cases"][0].update(clean=False)):
            inputs = self.inputs(); change(inputs)
            with self.assertRaisesRegex(ValueError, "coverage"):
                analysis.analyze_cases(inputs, evaluate=self.fake)
        with self.assertRaisesRegex(ValueError, "CPU workers"):
            analysis.analyze_cases(self.inputs(), workers=5)

    def test_changed_input_during_evaluation_fails_closed(self):
        inputs = self.inputs()
        def mutate(job):
            self.f["model"].write_bytes(b"changed during check")
            return self.fake(job)
        with self.assertRaisesRegex(ValueError, "during analysis"):
            analysis.analyze_cases(inputs, evaluate=mutate)

    def test_new_only_results_retain_all_rows_and_do_not_overwrite(self):
        report = analysis.analyze_cases(self.inputs(), evaluate=self.fake)
        target = self.root/"result"
        analysis.write_results(target, report, dict(status="passed_synthetic"))
        self.assertEqual(len((target/"retention.csv").read_text().splitlines()), 5)
        self.assertEqual(json.loads((target/"analysis.json").read_bytes())["counts"], report["counts"])
        with self.assertRaisesRegex(ValueError, "replace"):
            analysis.write_results(target, report, dict(status="passed_synthetic"))

    def test_output_cannot_be_inside_original_benchmark_or_assets(self):
        inputs = self.load()
        for directory in (self.f["benchmark"], self.f["assets"]):
            with self.subTest(directory=directory), self.assertRaisesRegex(ValueError, "overlaps"):
                analysis.validate_output(directory/"new-output", inputs)
            self.assertFalse((directory/"new-output").exists())

    def test_real_geometry_and_independent_audit_connect_end_to_end(self):
        # Only this test's artificial exactly-solvable chains are used.
        from test_continuous_geometry_audit import GeometryAuditTests
        from continuous_geometry_layer import filter_shot
        from continuous_geometry_audit import audit_analysis
        toy = GeometryAuditTests()
        toy.setUp()
        cases = [dict(case_id="synthetic-"+str(i), cell=dict(toy.cell, id="synthetic-"+str(i)),
            action=dict(toy.action, id="synthetic-"+str(i)), starts=copy.deepcopy(toy.starts),
            seeds=copy.deepcopy(toy.starts), clean=i==0) for i in range(3)]
        inputs = dict(synthetic=True, scene=toy.scene, links=toy.links, cases=cases,
            expected_counts=dict(requested=3, clean=1),
            protected_before=analysis.snapshot([self.f["model"]]), bindings=dict(synthetic=True))
        def evaluate(job):
            scene, links, case = job
            result = filter_shot(scene, case["cell"], case["action"], links, case["starts"],
                                solver=toy.solve, solver_protocol="synthetic-exact-FK")
            return dict(case_id=case["case_id"], cell_id=case["cell"]["id"], clean=case["clean"],
                        result=result, wall_s=0., cpu_s=0.)
        report = analysis.analyze_cases(inputs, evaluate=evaluate)
        independent = audit_analysis(report, inputs)
        self.assertEqual(independent["status"], "passed_synthetic")
        self.assertEqual(independent["counts"], dict(requested=3, unchanged=3, adjusted=0, rejected=0))


if __name__ == "__main__":
    unittest.main()
