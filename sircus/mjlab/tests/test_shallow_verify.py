"""CPU audit/mocked-orchestration tests; fabricated outcomes are not physics."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np

from sircus.mjlab import lowboard_inputs, lowboard_study, shallow_contract as protocol, shallow_verify as module
from sircus.mjlab.tests import test_lowboard_study as runner_tests
from sircus.mjlab.tests import test_lowboard_verify as prior_tests
from sircus.mjlab.tests.test_distance_contract import outcome
from sircus.mjlab.verify_expansion import Audit


def read(path): return json.loads(Path(path).read_text())
def write(path, value): Path(path).write_text(json.dumps(value))
def checks(a): return {r["name"]: r["passed"] for r in a.checks}


def fixture(root):
    spec, parent = prior_tests.fixture(root)
    spec["settings"] = protocol.settings(spec["scene"], spec["seeds"])
    return spec, parent


class LiteralCandidateTests(unittest.TestCase):
    def test_independently_rebuilt_literal_contexts_match_requested_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            spec, parent = fixture(Path(directory)); a = Audit()
            declared = module.audit_candidate_and_contexts(a, spec, parent)
            self.assertTrue(all(checks(a).values()), checks(a))
            self.assertEqual(declared, protocol.contexts(spec["seeds"], spec["settings"]["targets"]))
            self.assertEqual(len(declared), 12)
            self.assertEqual(spec["settings"]["candidate"]["gap_m"], .3868)
            self.assertNotEqual(spec["settings"]["candidate"]["gap_m"], .3868243380486807)

    def test_earlier_cpu_gap_or_height_target_and_budget_substitution_fail(self):
        for kind in ("gap", "height", "target", "seed", "budget"):
            with tempfile.TemporaryDirectory() as directory:
                spec, parent = fixture(Path(directory))
                if kind == "gap": spec["settings"]["candidate"]["gap_m"] = .3868243380486807
                elif kind == "height": spec["settings"]["candidate"]["board_height_m"] = .36
                elif kind == "target": spec["settings"]["targets"][0]["bucket_xy_m"][0] += .01
                elif kind == "seed": spec["settings"]["seed"] = 20261207
                else: spec["settings"]["budgets"]["maximum_requests"] = 500
                a = Audit(); module.audit_candidate_and_contexts(a, spec, parent)
                with self.subTest(kind=kind): self.assertFalse(all(checks(a).values()), checks(a))


class SelectionAndRowTests(unittest.TestCase):
    def test_actual_capture_only_selection_and_independent_seeded_noise(self):
        with tempfile.TemporaryDirectory() as directory:
            spec, parent = fixture(Path(directory)); declared = module.audit_candidate_and_contexts(Audit(), spec, parent)
            discovery = [c for c in declared if c["stage"] == "shallow"]
            rows = [outcome(c, .3868, hit=i % 2 == 0) for i, c in enumerate(discovery)]
            expected = module.select_observed_candidates(rows, spec["settings"]["targets"])
            self.assertEqual(expected, protocol.select_recipes(rows, spec["settings"]["targets"]))
            self.assertEqual(len(expected), 2)
            repeat = module.expected_repeats(expected)
            self.assertEqual(repeat, protocol.repeat_plan(expected)); self.assertEqual(len(repeat), 66)
            self.assertTrue(all(r["values"][2:] == [.380, 28., .3868]
                for r in repeat if r["phase"] == "nominal"))
            with self.assertRaises(ValueError): module.select_observed_candidates(rows[:-1], spec["settings"]["targets"])

    def test_new_row_namespace_executes_eligible_and_rejected_footprint_paths(self):
        spec, prepared, rows = prior_tests.ActualRowBranchTests().fixture()
        for i, row in enumerate(rows): row["id"] = f"shallow-shot-{i:06d}"
        a = Audit(); module.audit_rows(a, spec, prepared, rows, [], [], dict(requests=[], rows=[]))
        self.assertTrue(all(checks(a).values()), checks(a))
        rows[0]["id"] = "lowboard-shot-000000"
        a = Audit(); module.audit_rows(a, spec, prepared, rows, [], [], dict(requests=[], rows=[]))
        self.assertFalse(checks(a)["unique_sequential_request_ids"])


def fake_flight(hits):
    def run(env, bucket, values, q10, q14, eligible, record):
        n = int(sum(eligible))
        env.last_arrays = {key: np.zeros((960, n, width), np.float32)
            for key, width in (("position_m", 3), ("velocity_m_s", 3), ("forces_N", 6))}
        rows = []; reference = np.asarray([.16275255472120964, -.6365356466995716])
        for ok, xy in zip(eligible, bucket):
            if not ok: rows.append(runner_tests.unresolved(xy)); continue
            hit = bool(np.linalg.norm(xy - reference) < 1e-10 or hits)
            rows.append(dict(invalid=None, native_release=True, route_success=hit, route_quality=int(hit),
                board_bounce=True, whole_ball_entry=hit, closest_opening_distance_m=.01 if hit else .3,
                max_retention_s=float(hit), external_contact_steps=0, single_bounce_success=hit))
        return rows
    return run


class MockCompletionTests(unittest.TestCase):
    def run_mock(self, root, hits):
        parent, parent_spec = runner_tests.fixture(root)
        args = NS(parent_study=parent, assets=root / "assets", output=root / "out", batch=64, wall_time_limit=120)
        def variations(scene, assets, nominal, requests, bounds):
            source = {k: nominal[k] for k in ("values", "bucket_xy_m", "q10", "q14")}
            return [runner_tests.gate(scene, source, dict(id=r["context_id"], bucket_xy_m=source["bucket_xy_m"],
                board_height_m=r["values"][2], tilt_deg=r["values"][3]), r["gap_m"],
                eligible=r["repeat_index"] != 3) for r in requests]
        modules = {
            "sircus.mjlab.bank_study": NS(run_bank_rows=fake_flight(hits), unresolved_row=runner_tests.unresolved),
            "sircus.mjlab.learning_solve": NS(required_asset_paths=lambda _: []),
            "sircus.mjlab.distance_geometry": NS(DistanceGeometry=runner_tests.MockGeometry, audit_distance_variations=variations),
            "sircus.mjlab.timestep_env": NS(TimestepEnvironment=runner_tests.MockEnvironment, physics_settings=lambda s: dict(substeps=s))}
        runner_tests.MockEnvironment.allocated = []
        with patch.dict("sys.modules", modules), patch.object(lowboard_inputs, "PARENT_FINGERPRINT", parent_spec["fingerprint"]), \
                patch.object(lowboard_study, "validate_packages", lambda p: p), contextlib.redirect_stdout(io.StringIO()):
            report = lowboard_study._run(args, protocol=protocol)
        rows = [json.loads(line) for line in (args.output / "records.jsonl").read_text().splitlines()]
        return args, report, rows, runner_tests.MockEnvironment.allocated

    def evaluate(self, root, hits):
        args, report, rows, allocated = self.run_mock(root, hits)
        spec, parent = read(args.output / "contract.json"), read(args.parent_study / "contract.json")
        recipes, prepared = read(args.output / "recipes.json"), read(args.output / "prepared-contexts.json")
        a = Audit(); declared = module.audit_candidate_and_contexts(a, spec, parent)
        module.audit_plans(a, args.output, spec, declared, rows, recipes)
        module.shared.audit_controls(a, args.output, spec, rows)
        positions = {}
        for batch in read(args.output / "batches.json"):
            if batch["cache_file"] is None: continue
            with np.load(args.output / batch["cache_file"], allow_pickle=False) as data:
                for i, name in enumerate(data["record_ids"]): positions[str(name)] = data["position_m"][:, i].copy()
        module.shared.audit_report(a, spec, report, declared, prepared, rows, recipes, positions)
        module.shared.audit_display_selection(a, spec, report, rows, recipes)
        self.assertTrue(all(checks(a).values()), checks(a))
        self.assertEqual([r["id"] for r in rows], [f"shallow-shot-{i:06d}" for i in range(len(rows))])
        return args, report, rows, allocated, spec, declared, recipes

    def test_no_capture_completes28_requests_and_only64_native_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            args, report, rows, allocated, spec, declared, recipes = self.evaluate(Path(directory), False)
            self.assertEqual(len(rows), 28); self.assertEqual(recipes, []); self.assertEqual(allocated, [64])
            self.assertEqual(len(report["trace_examples"]), 4)
            wrong = read(args.output / "coarse-plan.json"); wrong[0]["gap_m"] = .3868243380486807
            write(args.output / "coarse-plan.json", wrong)
            a = Audit(); module.audit_plans(a, args.output, spec, declared, rows, recipes)
            self.assertFalse(checks(a)["exact_four_discovery_requests_with_literal_gap"])

    def test_four_captures_run_exactly396_paired_repeat_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            args, report, rows, allocated, spec, declared, recipes = self.evaluate(Path(directory), True)
            self.assertEqual(len(rows), 424); self.assertEqual(len(recipes), 4)
            self.assertEqual(sorted(allocated), [32, 64, 128])
            self.assertEqual(sum(r["role"] == "engineering_repeat" for r in rows), 396)
            self.assertEqual(len(report["trace_examples"]), 12)
            for repeat_id in {r["repeat_id"] for r in rows if r["role"] == "engineering_repeat"}:
                group = [r for r in rows if r.get("repeat_id") == repeat_id]
                self.assertEqual(sorted(r["substeps"] for r in group), [32, 64, 128])
                self.assertTrue(all(all(r[k] == group[0][k] for k in ("values", "bucket_xy_m", "q10", "q14")) for r in group))
            a = Audit(); module.audit_plans(a, args.output, spec, declared, rows[:-1], recipes)
            self.assertFalse(checks(a)["complete_bounded_requests_no_retries_or_hidden_shots"])
            write(args.output / "refinement-plan.json", [dict(gap_m=.39)])
            a = Audit(); module.audit_plans(a, args.output, spec, declared, rows, recipes)
            self.assertFalse(checks(a)["no_refinement_or_other_development_requests"])


if __name__ == "__main__": unittest.main()
