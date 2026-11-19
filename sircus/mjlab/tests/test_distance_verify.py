import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from sircus.mjlab import distance_verify as module
from sircus.mjlab.verify_expansion import Audit, digest, sha256


def write(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def checks(a):
    return {x["name"]: x["passed"] for x in a.checks}


def parent_fixture(root):
    geometry = dict(values=[.47, -.28, .4, 32., .3], bucket_xy_m=[.16, -.64],
        q10=[0.] * 6, q14=[1.] * 6)
    recipes = [dict(id=name, source=dict(copy.deepcopy(geometry), id="source-" + name,
        context_id="context-" + name, native_release=True, route_success=True, invalid=None))
        for name in module.PERFECT_PARENT_RECIPES]
    parent = dict(scene={"scene": "fixture"}, packages={}, asset_sha256={},
        source_sha256={name: "unchanged" for name in module.CORE_SOURCE})
    parent["fingerprint"] = digest(parent)
    write(root, "contract.json", parent)
    write(root, "recipes.json", recipes)
    (root / "records.jsonl").write_text("".join(json.dumps(r["source"]) + "\n" for r in recipes))
    manifest = {name: sha256(root / name) for name in ("contract.json", "recipes.json", "records.jsonl")}
    write(root, "evidence-manifest.json", manifest)
    report = dict(status="completed", phase="completed", contract_fingerprint=parent["fingerprint"],
        evidence_sha256=manifest, selected_recipes=[dict(id=r["id"], source_record_id=r["source"]["id"], by_substeps={
            str(s): dict(attempts=23, captures=23) for s in (32, 64, 128)}) for r in recipes])
    write(root, "report.json", report)
    write(root, "audit-neighborhood.json", dict(status="passed", failures=[], checks_count=1,
        contract_fingerprint=parent["fingerprint"], checks=[dict(name="fixture", passed=True)]))
    spec = {k: copy.deepcopy(parent[k]) for k in ("scene", "packages", "asset_sha256", "source_sha256")}
    spec["seeds"] = [dict(id=r["id"], **{k:r["source"][k] for k in module.GEOMETRY_KEYS}) for r in recipes]
    spec["seed_provenance"] = [dict(id=r["id"], parent_source_record_id=r["source"]["id"],
        parent_context_id=r["source"]["context_id"], geometry_sha256=digest(seed))
        for r,seed in zip(recipes,spec["seeds"])]
    spec["parent_study"] = dict(contract_fingerprint=parent["fingerprint"], files_sha256={
        name: sha256(root / name) for name in ("contract.json", "report.json", "audit-neighborhood.json",
            "evidence-manifest.json", "recipes.json")})
    return spec, parent["fingerprint"]


class ParentEvidenceTests(unittest.TestCase):
    def test_exact_parent_raw_inputs_and_geometry_seeds_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); spec, fingerprint = parent_fixture(root); a = Audit()
            with patch.object(module, "PARENT_FINGERPRINT", fingerprint):
                module.audit_parent(a, spec, root)
            self.assertTrue(all(checks(a).values()), checks(a))

    def test_wrong_parent_fingerprint_or_rewritten_raw_evidence_is_rejected(self):
        for kind in ("fingerprint", "raw", "seed", "source", "omitted_input"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory); spec, fingerprint = parent_fixture(root)
                if kind == "fingerprint": fingerprint = "not-the-parent"
                elif kind == "raw": (root / "records.jsonl").write_text('{"id":"changed"}\n')
                elif kind == "seed": spec["seeds"][0]["q14"][0] += .01
                elif kind == "source": spec["source_sha256"][module.CORE_SOURCE[0]] = "other"
                else: spec["parent_study"]["files_sha256"].pop("recipes.json")
                a = Audit()
                with patch.object(module, "PARENT_FINGERPRINT", fingerprint):
                    module.audit_parent(a, spec, root)
                with self.subTest(kind=kind): self.assertFalse(all(checks(a).values()), checks(a))


def paired_fixture():
    rows = []; positions = {}
    for i in range(33):
        for step in (32, 64, 128):
            row = dict(id=f"r{i:02d}-{step}", repeat_id=f"r{i:02d}", substeps=step,
                phase="nominal" if i < 3 else "fresh", native_release=True, route_success=True)
            rows.append(row); positions[row["id"]] = np.zeros((960, 3))
    return rows, positions


class PairedSensitivityTests(unittest.TestCase):
    def test_all_33_requests_and_nominal_positions_are_used(self):
        rows, positions = paired_fixture()
        result = module.paired_sensitivity(rows, positions)
        self.assertEqual(result, dict(paired_attempts=33, capture_agreement_64_128=33,
            nominal_position_delta_350ms_max_m=0., paired_engineering_gate=True,
            all_resolution_repeat_gate=True))

    def test_coarse_step_threshold_is_separate_from_fine_pair_gate(self):
        rows, positions = paired_fixture()
        for row in rows:
            if row["substeps"] == 32 and row["repeat_id"] in ("r03", "r04", "r05", "r06"):
                row["route_success"] = False
        result = module.paired_sensitivity(rows, positions)
        self.assertTrue(result["paired_engineering_gate"])
        self.assertFalse(result["all_resolution_repeat_gate"])

    def test_nominal_failure_rejected_request_missing_step_and_drift_fail(self):
        for kind in ("nominal", "fresh", "agreement", "missing", "drift"):
            rows, positions = paired_fixture()
            if kind == "nominal": rows[1]["route_success"] = False
            elif kind == "fresh":
                for row in rows:
                    if row["substeps"] in (64, 128) and row["repeat_id"] in ("r03", "r04", "r05", "r06"):
                        row["route_success"] = False; row["native_release"] = False
            elif kind == "agreement":
                for row in rows:
                    if row["substeps"] == 64 and row["repeat_id"] in ("r03", "r04", "r05", "r06"):
                        row["route_success"] = False
            elif kind == "missing": rows.pop()
            else: positions["r00-128"][83, 0] = .00501
            result = module.paired_sensitivity(rows, positions)
            with self.subTest(kind=kind): self.assertFalse(result["paired_engineering_gate"])


class TableFootprintTests(unittest.TestCase):
    def test_outer_radius_not_inner_opening_controls_full_footprint(self):
        scene = dict(table=dict(center_m=[0., 0., 0.], size_m=[2., 1., .05]),
            bucket=dict(outer_top_diameter_m=.2, outer_bottom_diameter_m=.158))
        self.assertAlmostEqual(module.table_footprint_clearance(scene, [.9, 0.]), 0.)
        self.assertAlmostEqual(module.table_footprint_clearance(scene, [.901, 0.]), -.001)
        self.assertAlmostEqual(module.table_footprint_clearance(scene, [0., .4]), 0.)
        self.assertAlmostEqual(module.table_footprint_clearance(scene, [0., .41]), -.01)

    def test_invalid_coordinates_are_not_repositioned(self):
        scene = dict(table=dict(center_m=[0., 0., 0.], size_m=[2., 1., .05]),
            bucket=dict(outer_top_diameter_m=.2, outer_bottom_diameter_m=.158))
        for point in ([0.], [0., float("nan")]):
            with self.subTest(point=point), self.assertRaises(ValueError):
                module.table_footprint_clearance(scene, point)


class ContextBranchTests(unittest.TestCase):
    def fixture(self, rotated=False):
        from sircus.mjlab.admission import VERIFIED_PARK_Q10
        from sircus.mjlab.distance_geometry import table_footprint
        scene = dict(table=dict(center_m=[0., 0., 0.], size_m=[2., 2., .05]),
            bucket=dict(outer_top_diameter_m=.2, outer_bottom_diameter_m=.158))
        seed = dict(id="recipe-00", values=[0., 0., .4, 30., .3],
            bucket_xy_m=[.4, 0.], q10=[0.] * 6, q14=[1.] * 6)
        context = dict(id="context", source_anchor_id=seed["id"], bucket_xy_m=[0., .5] if rotated else [.5, 0.],
            board_height_m=.4, tilt_deg=30., distance_id="d", ray_id="r", distance_m=.5)
        proof = dict(version="sircus-fixed-board-gap-window-geometry-v1",
            scene_sha256=digest(scene), bucket_xy_m=context["bucket_xy_m"], board_world=np.eye(4).tolist(),
            q14=seed["q14"], park_q10=list(VERIFIED_PARK_Q10))
        row = dict(context, q14=seed["q14"], board_admitted=True, board_world=proof["board_world"],
            board_branch_sha256=digest(proof), source_anchor_geometry_sha256=digest(seed),
            wrapper_version="sircus-bucket-distance-frozen-branch-geometry-v1",
            bucket_footprint=table_footprint(scene, context["bucket_xy_m"]),
            same_source_bearing=not rotated, bearing_comparison_tolerance=1e-12)
        if rotated:
            row["rotated_board_derivation"] = dict(admitted=True, preserve_known=False,
                old_joints_role="numerical_continuation_only", source_anchor_geometry_sha256=digest(seed),
                bucket_xy_m=context["bucket_xy_m"], branch_checks=[dict(passed=True)])
        return dict(scene=scene, seeds=[seed]), [context], [row]

    def test_same_ray_source_and_rotated_proof_are_both_supported(self):
        for rotated in (False, True):
            spec, declared, prepared = self.fixture(rotated); a = Audit()
            module.audit_contexts(a, spec, declared, prepared)
            with self.subTest(rotated=rotated): self.assertTrue(all(checks(a).values()), checks(a))

    def test_branch_substitution_provenance_footprint_and_rotation_tampering_fail(self):
        for kind in ("hash", "source", "footprint", "rotation", "same_ray"):
            spec, declared, prepared = self.fixture(kind == "rotation")
            if kind == "hash": prepared[0]["board_branch_sha256"] = "other"
            elif kind == "source": prepared[0]["source_anchor_geometry_sha256"] = "other"
            elif kind == "footprint": prepared[0]["bucket_footprint"]["required_edge_margin_m"] = 0.
            elif kind == "rotation": prepared[0]["rotated_board_derivation"]["admitted"] = False
            else:
                from sircus.mjlab.admission import VERIFIED_PARK_Q10
                row = prepared[0]; row["q14"] = [2.] * 6
                row["board_branch_sha256"] = digest(dict(version="sircus-fixed-board-gap-window-geometry-v1",
                    scene_sha256=digest(spec["scene"]), bucket_xy_m=row["bucket_xy_m"], board_world=row["board_world"],
                    q14=row["q14"], park_q10=list(VERIFIED_PARK_Q10)))
            a = Audit(); module.audit_contexts(a, spec, declared, prepared)
            with self.subTest(kind=kind): self.assertFalse(all(checks(a).values()), checks(a))


def distance_fixture():
    from sircus.mjlab import distance_contract as protocol
    root = Path(__file__).resolve().parents[3]
    scene = json.loads((root / "sircus/isaac/scene.json").read_text())
    seeds = [dict(id=name, values=[.4708757793012148, -.2778934014509746, .4, 30., .3],
        bucket_xy_m=[.16275255472120964, -.6365356466995716], q10=[0.] * 6, q14=[1.] * 6)
        for name in protocol.SEED_IDS]
    return dict(scene=scene, seeds=seeds, settings=protocol.settings(scene, seeds))


class TargetAndPlanTests(unittest.TestCase):
    def test_exact_rays_and_unsupported_far_targets_are_preserved(self):
        spec = distance_fixture(); a = Audit(); targets = module.audit_targets(a, spec)
        self.assertTrue(all(checks(a).values()), checks(a))
        self.assertEqual([r["id"] for r in targets if not r["bucket_on_table"]],
            ["original-d06", "corner-d05", "corner-d06"])
        for field in ("bucket_xy_m", "bucket_on_table", "table_margin_m"):
            bad = copy.deepcopy(spec)
            bad["settings"]["targets"][2][field] = [0., 0.] if field == "bucket_xy_m" else "changed"
            a = Audit(); module.audit_targets(a, bad)
            with self.subTest(field=field): self.assertFalse(all(checks(a).values()))

    def plan_fixture(self, root):
        from sircus.mjlab import distance_contract as protocol
        spec = distance_fixture(); targets = spec["settings"]["targets"]
        contexts = protocol.contexts(spec["seeds"], targets)
        coarse = [dict(p, id=f"coarse-{i}", substeps=64, route_success=False, native_release=False,
            invalid="mock_rejection") for i, p in enumerate(protocol.coarse_plan(contexts))]
        extension = protocol.extension_contexts(spec["seeds"], targets, coarse)
        extra = [dict(p, id=f"extra-{i}", substeps=64, route_success=False, native_release=False,
            invalid="mock_rejection") for i, p in enumerate(protocol.extension_plan(extension))]
        controls = [dict(role="control", phase="controls", substeps=64) for _ in range(24)]
        for name, value in {"declared-contexts.json": contexts + extension,
                "extension-contexts.json": extension, "coarse-plan.json": protocol.coarse_plan(contexts),
                "extension-plan.json": protocol.extension_plan(extension), "refinement-plan.json": [],
                "frozen-repeat-plan.json": []}.items(): write(root, name, value)
        return spec, controls + coarse + extra

    def test_full_miss_plans_account_for_all_unsupported_and_extension_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); spec, rows = self.plan_fixture(root); a = Audit()
            contexts = module.audit_development_plans(a, root, spec, rows, [])
            self.assertEqual(len(contexts), 208); self.assertEqual(len(rows), 3272)
            self.assertTrue(all(checks(a).values()), checks(a))

    def test_omitted_requests_and_changed_extension_or_refinement_plans_fail(self):
        for kind in ("request", "extension", "refinement", "future_repeat"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory); spec, rows = self.plan_fixture(root)
                if kind == "request": rows.pop(24)
                elif kind == "extension": write(root, "extension-contexts.json", [])
                elif kind == "refinement": write(root, "refinement-plan.json", [dict(gap_m=.33)])
                else: rows.append(dict(role="engineering_repeat", phase="fresh", substeps=64, repeat_id="not-frozen"))
                a = Audit(); module.audit_development_plans(a, root, spec, rows, [])
                with self.subTest(kind=kind): self.assertFalse(all(checks(a).values()), checks(a))


if __name__ == "__main__": unittest.main()
