"""Corruption checks for independent corrected-contact evidence accounting."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from sircus.mjlab.contract import ScoreStream, FORCE_NAMES
from sircus.mjlab.corrected_verify import (Audit, digest, sha256, audit_native_batch,
    audit_grid_denominators, audit_split_identity, audit_geometry_caches, independently_select, audit_confirmations)
from sircus.mjlab.corrected_verify import audit_home_layouts


class CorrectedVerificationTests(unittest.TestCase):
    def layouts(self):
        return [dict(id=f"grid-{r}-{c}", bucket_xy_m=[.1 + .16 * c, -.65 + .2 * r]) for r in range(5) for c in range(7)]

    def results(self, audit):
        return {c["name"]: c["passed"] for c in audit.checks}

    def native_fixture(self):
        scene = json.loads((Path(__file__).resolve().parents[2] / "isaac/scene.json").read_text())
        p = np.tile([.5, -.2, -.3], (960, 1, 1)); v = np.zeros_like(p); f = np.zeros((960, 1, 6))
        s = ScoreStream(scene, [.16, -.64], [0., 0., 1.]); s.step(p[0, 0], v[0, 0], dict(zip(FORCE_NAMES, f[0, 0])))
        row = dict(s.result(), id="new-native", bucket_xy_m=[.16, -.64], board_normal=[0., 0., 1.],
            contact_protocol=dict(substeps=64, board_mode="articulated", gravity_mode="passive_only"), native_release=True, invalid=None)
        return scene, [row], dict(position_m=p, velocity_m_s=v, forces_N=f), ["new-native"]

    def test_all_35_including_unsupported_are_required(self):
        cells = self.layouts(); cells[0]["endpoint_eligible"] = 0
        a = Audit(); audit_grid_denominators(a, self.layouts(), cells)
        self.assertTrue(all(self.results(a).values()))
        cells.pop(0); a = Audit(); audit_grid_denominators(a, self.layouts(), cells)
        self.assertFalse(self.results(a)["all_development_cells_in_report"])

    def test_duplicate_cell_cannot_hide_missing_layout(self):
        cells = self.layouts(); cells[-1] = copy.deepcopy(cells[0])
        a = Audit(); audit_grid_denominators(a, self.layouts(), cells)
        self.assertFalse(self.results(a)["all_development_cells_in_report"])

    def test_splits_are_new_disjoint_and_bound_to_grid(self):
        layouts = self.layouts(); splits = dict(train=[r["bucket_xy_m"] for r in layouts], validation=[[.234, -.534]], test=[[.342, -.432]])
        a = Audit(); audit_split_identity(a, splits, layouts, [[.1, .2]])
        self.assertTrue(all(self.results(a).values()))
        a = Audit(); audit_split_identity(a, splits, layouts, [[.342, -.432]])
        self.assertFalse(self.results(a)["new_validation_test_exclude_historical_coordinates"])
        splits["test"] = splits["validation"]
        a = Audit(); audit_split_identity(a, splits, layouts, [])
        self.assertFalse(self.results(a)["future_splits_unique_and_disjoint"])

    def test_native_saved_arrays_rescore(self):
        scene, rows, arrays, ids = self.native_fixture()
        a = Audit(); audit_native_batch(a, scene, rows, arrays, ids, label="batch")
        self.assertTrue(all(self.results(a).values()), self.results(a))

    def test_legacy_physics_cannot_be_relabelled_corrected(self):
        scene, rows, arrays, ids = self.native_fixture(); rows[0]["contact_protocol"]["substeps"] = 8
        a = Audit(); audit_native_batch(a, scene, rows, arrays, ids, label="batch")
        self.assertFalse(self.results(a)["batch_corrected_articulated_protocol"])

    def test_mismatched_record_id_prevents_rescore_claim(self):
        scene, rows, arrays, ids = self.native_fixture()
        a = Audit(); audit_native_batch(a, scene, rows, arrays, ["old-native"], label="batch")
        self.assertFalse(self.results(a)["batch_record_id_identity"])
        self.assertNotIn("batch_independent_scalar_rescore", self.results(a))

    def test_false_success_and_altered_trace_are_rejected(self):
        scene, rows, arrays, ids = self.native_fixture(); rows[0]["route_success"] = True
        a = Audit(); audit_native_batch(a, scene, rows, arrays, ids, label="batch")
        self.assertFalse(self.results(a)["batch_independent_scalar_rescore"])
        scene, rows, arrays, ids = self.native_fixture(); arrays["position_m"][:, :, 2] = .5
        a = Audit(); audit_native_batch(a, scene, rows, arrays, ids, label="batch")
        self.assertFalse(self.results(a)["batch_independent_scalar_rescore"])

    def test_cached_joint_substitution_and_omitted_request_rejected(self):
        cache = dict(record_ids=np.asarray(["request"]), values=np.asarray([[.5, -.2, .4, 30., .3]]),
            bucket=np.asarray([[.2, -.6]]), q10=np.zeros((1, 1, 6)), q14=np.ones((1, 1, 6)),
            mask=np.ones((1, 1), bool), reasons=np.zeros((1, 1), int), minimum_clearance_lower_bound_m=np.full((1, 1), .006))
        row = dict(id="request", geometry_cache_file="endpoints.npz", geometry_cache_row=0,
            values=cache["values"][0].tolist(), bucket_xy_m=cache["bucket"][0].tolist(),
            q10=[0.] * 6, q14=[1.] * 6, geometry_eligible=True, geometry_reason=0)
        a = Audit(); audit_geometry_caches(a, [row], {"endpoints.npz": cache})
        self.assertTrue(all(self.results(a).values()))
        row["q14"][0] += .01
        a = Audit(); audit_geometry_caches(a, [row], {"endpoints.npz": cache})
        self.assertFalse(self.results(a)["all_record_request_joint_cache_associations"])
        a = Audit(); audit_geometry_caches(a, [], {"endpoints.npz": cache})
        self.assertFalse(self.results(a)["no_cached_request_or_outcome_omitted"])

    def test_outcome_free_selection_respects_masks_and_seed(self):
        values = np.array([[.5 + i * .01, -.2, .4, 30., .3] for i in range(8)])
        mask = np.array([True, False, True, True, False, True, True, True])
        chosen = independently_select(values, mask, 4, 20261127)
        self.assertEqual(chosen, independently_select(values, mask, 4, 20261127))
        self.assertEqual(len(chosen), 4)
        self.assertTrue(all(mask[i] for i in chosen))

    def test_zero_separation_bound_cannot_be_relabelled_fixed_penetration(self):
        layouts = self.layouts(); records = []
        for i, cell in enumerate(layouts):
            passed = i > 0
            raw = dict(passed=passed, reason=None if passed else "clearance", minimum_clearance_m=.006 if passed else 0.,
                closest_pair=["Arm10/link_base/collision/0", "bucket/wall00"], common_home_sha256="home",
                margin_contacts=[])
            summary = {k: raw[k] for k in ("passed", "reason", "minimum_clearance_m", "closest_pair", "common_home_sha256")}
            summary.update(category="home_clear" if passed else "staging_clearance_blocked", fixed_base_penetrations=[])
            records.append(dict(cell, evidence=raw, home_geometry=summary))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); path = root / "common-home-layouts.json"; path.write_text(json.dumps(records))
            spec = dict(grid_layouts=layouts, common_home_sha256="home", common_home_layouts_sha256=sha256(path))
            report = dict(cells=[dict(r, home_geometry=p["home_geometry"]) for r, p in zip(layouts, records)],
                counts=dict(home_clear_positions=34, grid_positions=35), common_home_layouts_sha256=sha256(path))
            a = Audit(); audit_home_layouts(a, root, spec, report)
            self.assertTrue(all(self.results(a).values()))
            records[0]["home_geometry"]["category"] = "fixed_base_bucket_penetration"
            path.write_text(json.dumps(records)); spec["common_home_layouts_sha256"] = report["common_home_layouts_sha256"] = sha256(path)
            a = Audit(); audit_home_layouts(a, root, spec, report)
            self.assertFalse(self.results(a)["initial_home_categories_bound_to_actual_evidence"])


class QualificationAuditTests(unittest.TestCase):
    def fixture(self, root):
        scene = dict(scope="mock geometry, no dynamics")
        cell = dict(id="dev-grid-00-00", row=0, column=0, bucket_xy_m=[.2, -.6])
        source = dict(id="new-source", cell_id=cell["id"], candidate_id="candidate", phase="search",
            values=[.5, -.2, .4, 30., .3], q10=[0.] * 6, q14=[1.] * 6,
            bucket_xy_m=cell["bucket_xy_m"], native_release=True, route_success=True)
        home = dict(version="sircus-fixed-common-home-v1", passed=True, scene_sha256=digest(scene), q10=[0.] * 6, q14=[0.] * 6)
        home["sha256"] = digest(home)
        identity = dict(version="sircus-fixed-common-home-transit-admission-v1", scene_sha256=digest(scene),
            common_home_sha256=home["sha256"], **{k: source[k] for k in ("bucket_xy_m", "values", "q10", "q14")})
        scope = "fixed_common_home_to_exact_selected_shot_and_reverse_sampled_geometry"
        body = dict(passed=True, request=identity, proof_scope=scope, returned_home_joint_max_error_rad=0.,
            exact_selected_joint_max_error_rad=0., minimum_clearance_lower_bound_m=.006)
        proof = dict(passed=True, proof_scope=scope, common_home_sha256=home["sha256"], request_sha256=digest(identity), report=body)
        rng = np.random.default_rng(20261127 + 1000)
        nominal = [dict(source, id=f"new-source-nominal-{i:02d}", phase="nominal") for i in range(3)]
        fresh = [dict(source, id=f"new-source-confirmation-{i:02d}", phase="confirmation",
            values=(np.array(source["values"]) + rng.normal(0, [.001, .001, .002, .35, .002])).tolist()) for i in range(20)]
        for row in [*nominal, *fresh]:
            request = {k: row[k] for k in ("bucket_xy_m", "values", "q10", "q14")}
            row.update(confirmation_source_id=source["id"], geometry_eligible=True, motion_admitted=True,
                motion_admission_file=f'variation-{row["id"]}.json', invalid=None)
            variation = dict(passed=True, common_home_sha256=home["sha256"], nominal_request_sha256=proof["request_sha256"],
                request=request, request_sha256=digest(request), local_edges=[dict(passed=True, minimum_clearance_m=.006)] * 4)
            (root / row["motion_admission_file"]).write_text(json.dumps(variation))
        entry = dict(source=source, transit_file="transit.json", nominal_ids=[r["id"] for r in nominal],
            fresh_ids=[r["id"] for r in fresh], noise_seed=20261127 + 1000, nominal_attempts=3, nominal_captures=3,
            fresh_attempts=20, fresh_captures=20, qualified=True)
        spec = dict(scene=scene, grid_layouts=[cell], common_home=home, common_home_sha256=home["sha256"], fingerprint="mock", physics={}, budgets=dict(top=1))
        report = dict(cells=[dict(cell, confirmed_sources=[entry], search_captures=1)], qualified_sources=[entry],
            cells_with_native_capture=1, cells_with_qualified_source=1,
            transit_attempts=[dict(source_id=source["id"], cell_id=cell["id"], passed=True, file="transit.json")])
        catalogue = dict(contract_fingerprint="mock", physics={}, sources=[entry]); catalogue["fingerprint"] = digest(catalogue)
        for name, value in (("common-home.json", home), ("transit.json", proof), ("confirmed-catalogue.json", catalogue),
            ("confirmation-plan-dev-grid-00-00-source-00.json", dict(source_id=source["id"], noise_seed=entry["noise_seed"], nominal=nominal, fresh=fresh))):
            (root / name).write_text(json.dumps(value))
        report.update(catalogue_fingerprint=catalogue["fingerprint"], catalogue_sha256=sha256(root / "confirmed-catalogue.json"))
        return spec, report, [source, *nominal, *fresh]

    def check(self, root, fixture):
        a = Audit(); audit_confirmations(a, root, *fixture)
        return {c["name"]: c["passed"] for c in a.checks}

    def test_valid_fresh_qualification_and_common_home_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); f = self.fixture(root); result = self.check(root, f)
            self.assertTrue(all(result.values()), result)

    def test_missing_failed_confirmation_cannot_shrink_denominator(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); f = self.fixture(root); f[1]["qualified_sources"][0]["fresh_ids"].pop()
            result = self.check(root, f)
            self.assertFalse(result["new-source_all_three_and_twenty_attempts"])
            self.assertFalse(result["no_confirmation_failure_or_source_group_omitted"])

    def test_own_endpoint_cycle_cannot_claim_common_home(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); f = self.fixture(root); proof = json.loads((root / "transit.json").read_text())
            proof["proof_scope"] = "own_endpoint_round_trip"
            (root / "transit.json").write_text(json.dumps(proof))
            self.assertFalse(self.check(root, f)["new-source_exact_source_fixed_home_proof"])

    def test_failed_local_motion_cannot_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); f = self.fixture(root); row = f[2][-1]
            path = root / row["motion_admission_file"]; v = json.loads(path.read_text()); v["passed"] = False
            path.write_text(json.dumps(v)); row["motion_admitted"] = False
            self.assertFalse(self.check(root, f)[row["id"] + "_failed_motion_never_releases_or_scores"])

    def test_omitted_failed_transit_candidate_cannot_disappear(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); f = self.fixture(root)
            extra = copy.deepcopy(f[2][0]); extra["id"] = "second-captured-source"
            f[2].append(extra); f[0]["budgets"]["top"] = 2
            self.assertFalse(self.check(root, f)["every_declared_top_capture_gets_transit_attempt_including_failures"])

    def test_passed_transit_must_receive_full_confirmation_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); f = self.fixture(root); f[1]["cells"][0]["confirmed_sources"] = []
            self.assertFalse(self.check(root, f)["every_passed_transit_source_has_full_nominal_and_confirmation_group"])


if __name__ == "__main__":
    unittest.main()
