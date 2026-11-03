"""CPU-only audit/adversarial fixtures; mocked flights are not physics evidence."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from sircus.mjlab import lowboard_contract as protocol, lowboard_verify as module
from sircus.mjlab import distance_contract
from sircus.mjlab.tests.test_distance_contract import parent_inputs
from sircus.mjlab.tests import test_lowboard_study as runner_tests
from sircus.mjlab.tests.test_timestep_env import options
from sircus.mjlab.timestep_env import validate_options
from sircus.mjlab.verify_expansion import Audit, digest, sha256


def write(root, name, value):
    (root / name).write_text(json.dumps(value))


def checks(a):
    return {r["name"]: r["passed"] for r in a.checks}


def fixture(root):
    scene, seeds = parent_inputs()
    for seed in seeds: seed.update(q10=[0.] * 6, q14=[1.] * 6)
    source = {name: "unchanged:" + name for name in module.CORE_SOURCE}
    publisher, (old, new) = next(iter(module.PUBLICATION_OVERRIDE.items()))
    source[publisher] = old
    parent = dict(version="sircus-bucket-distance-development-v1", scene=scene, seeds=seeds,
        settings=distance_contract.settings(scene, seeds), source_sha256=source, asset_sha256={}, packages={})
    parent["fingerprint"] = digest(parent)
    write(root, "contract.json", parent)
    rows = [dict(copy.deepcopy(seed), id=f"parent-{seed['id']}-{i}", role="control", substeps=64,
        source_anchor_id=seed["id"], native_release=True, route_success=True, invalid=None)
        for seed in seeds for i in range(3)]
    (root / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    manifest = {name: sha256(root / name) for name in ("contract.json", "records.jsonl")}
    write(root, "evidence-manifest.json", manifest)
    write(root, "report.json", dict(status="completed", phase="completed", contract_fingerprint=parent["fingerprint"], evidence_sha256=manifest))
    write(root, "audit-distance.json", dict(status="passed", contract_fingerprint=parent["fingerprint"],
        checks_count=1, checks=[dict(passed=True)], failures=[]))
    spec = dict(scene=scene, seeds=copy.deepcopy(seeds), settings=protocol.settings(scene, seeds),
        source_sha256=dict(source, **{publisher: new}), asset_sha256={}, packages={},
        source_compatibility={publisher: dict(parent_sha256=old, current_sha256=new,
            purpose="Read-only display adapter published after parent execution; not imported by native physics")},
        seed_provenance=[dict(id=seed["id"], parent_control_record_ids=[r["id"] for r in rows if r["source_anchor_id"] == seed["id"]],
            geometry_sha256=digest(seed)) for seed in seeds])
    spec["parent_study"] = dict(contract_fingerprint=parent["fingerprint"], files_sha256={n: sha256(root / n)
        for n in ("contract.json", "report.json", "audit-distance.json", "evidence-manifest.json", "records.jsonl")})
    return spec, parent


class ParentAndSourceTests(unittest.TestCase):
    def test_exact_parent_control_ancestry_and_single_publication_override(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); spec, parent = fixture(root); a = Audit()
            with patch.object(module, "PARENT_FINGERPRINT", parent["fingerprint"]): module.audit_parent(a, spec, root)
            self.assertTrue(all(checks(a).values()), checks(a))

    def test_raw_seed_provenance_and_native_source_tampering_fail(self):
        for kind in ("raw", "seed", "lineage", "native", "publisher", "override"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory); spec, parent = fixture(root)
                publisher = next(iter(module.PUBLICATION_OVERRIDE))
                if kind == "raw": (root / "records.jsonl").write_text((root / "records.jsonl").read_text() + "\n")
                elif kind == "seed": spec["seeds"][0]["q14"][0] += .001
                elif kind == "lineage": spec["seed_provenance"][0]["parent_control_record_ids"][0] = "other"
                elif kind == "native": spec["source_sha256"][module.CORE_SOURCE[0]] = "changed"
                elif kind == "publisher": spec["source_sha256"][publisher] = "unapproved-new-publisher"
                else: spec["source_compatibility"] = {}
                a = Audit()
                with patch.object(module, "PARENT_FINGERPRINT", parent["fingerprint"]): module.audit_parent(a, spec, root)
                with self.subTest(kind=kind): self.assertFalse(all(checks(a).values()), checks(a))


class GridTests(unittest.TestCase):
    def test_independent_grid_assignment_and_ceiling_match_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            spec, parent = fixture(Path(directory)); a = Audit()
            expected = module.audit_targets_and_grid(a, spec, parent)
            self.assertTrue(all(checks(a).values()), checks(a))
            self.assertEqual(expected, protocol.contexts(spec["seeds"], spec["settings"]["targets"]))
            self.assertEqual(len(expected), 136)
            coarse = protocol.coarse_plan(expected, spec["scene"], spec["seeds"])
            self.assertEqual(len(coarse), 640)
            self.assertEqual(set(r["release_headroom_m"] for r in coarse), {.005, .015, .03, .05, .08})

    def test_target_movement_control_relabeling_or_cap_change_is_not_hidden(self):
        for kind in ("target", "control", "cap"):
            with tempfile.TemporaryDirectory() as directory:
                spec, parent = fixture(Path(directory))
                if kind == "target": spec["settings"]["targets"][0]["bucket_xy_m"][0] += .001
                elif kind == "control": spec["settings"]["control_target"]["distance_id"] = "corner-d01"
                else: spec["settings"]["reach_limit"]["ball_center_height_m"] += .001
                a = Audit(); module.audit_targets_and_grid(a, spec, parent)
                with self.subTest(kind=kind): self.assertFalse(all(checks(a).values()), checks(a))


class UsedOptionsTests(unittest.TestCase):
    def fixture(self, root, steps):
        effective = {str(s): validate_options(*options(s), s) for s in steps}
        write(root, "effective-options.json", effective)
        return dict(effective_native_options=copy.deepcopy(effective)), [dict(native_release=True, substeps=s) for s in steps]

    def test_zero_recipe_completed_case_requires_only_actual64_allocation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); report, rows = self.fixture(root, [64]); a = Audit()
            module.audit_options(a, root, report, rows)
            self.assertTrue(all(checks(a).values()), checks(a))
            report, _ = self.fixture(root, [32, 64]); a = Audit(); module.audit_options(a, root, report, rows)
            self.assertFalse(checks(a)["only_actually_used_native_engines_claimed"])

    def test_all_three_repeat_allocations_and_strict_vendor_conversion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); report, rows = self.fixture(root, [32, 64, 128]); a = Audit()
            module.audit_options(a, root, report, rows); self.assertTrue(all(checks(a).values()))
            value = report["effective_native_options"]; value["64"]["warp"]["tolerance"][0] = float(np.float32(1e-8))
            write(root, "effective-options.json", value); a = Audit(); module.audit_options(a, root, report, rows)
            self.assertFalse(checks(a)["actual_allocated_native_warp_options_64"])


class ActualRowBranchTests(unittest.TestCase):
    def fixture(self):
        from sircus.mjlab.distance_geometry import table_footprint
        scene, seeds = parent_inputs(); seed = dict(seeds[0], q10=[0.] * 6, q14=[1.] * 6)
        spec = dict(scene=scene, settings=protocol.settings(scene, [dict(s, q10=[0.] * 6, q14=[1.] * 6) for s in seeds]))
        prepared, rows = [], []
        for index, eligible in enumerate((True, False)):
            target = spec["settings"]["control_target"] if eligible else spec["settings"]["targets"][0]
            context = dict(id="control:fixture" if eligible else "far:fixture", source_anchor_id=seed["id"],
                source_anchor_geometry_sha256=digest(seed), bucket_xy_m=target["bucket_xy_m"],
                board_height_m=seed["values"][2], tilt_deg=seed["values"][3], q14=seed["q14"],
                board_branch_sha256="fixed-board-branch", **{k: target[k] for k in ("distance_id", "ray_id", "distance_m")})
            geometry = dict(context_id=context["id"], values=seed["values"], bucket_xy_m=context["bucket_xy_m"],
                q10=seed["q10"], q14=seed["q14"], eligible=eligible, reason=None if eligible else "workspace_prior_reject",
                board_branch_sha256=context["board_branch_sha256"], minimum_clearance_lower_bound_m=.01 if eligible else None,
                source_anchor_id=seed["id"], source_anchor_geometry_sha256=context["source_anchor_geometry_sha256"],
                wrapper_version="sircus-bucket-distance-frozen-branch-geometry-v1",
                bucket_footprint=table_footprint(scene, context["bucket_xy_m"]),
                **{k: context[k] for k in ("distance_id", "ray_id", "distance_m")})
            row = dict(copy.deepcopy(geometry), id=f"lowboard-shot-{index:06d}", context_id=context["id"],
                gap_m=seed["values"][4], role="control" if eligible else "development", phase="controls" if eligible else "coarse",
                substeps=64, native_release=eligible, invalid=None if eligible else "workspace_prior_reject",
                route_success=eligible, route_quality=1 if eligible else 0, board_bounce=eligible, whole_ball_entry=eligible,
                max_retention_s=1. if eligible else 0., external_contact_steps=0, closest_opening_distance_m=.01,
                single_bounce_success=eligible, geometry=copy.deepcopy(geometry))
            row["learning_reward"] = module.row_reward(row)
            prepared.append(copy.deepcopy(context)); rows.append(row)
        return spec, prepared, rows

    def test_eligible_and_rejected_actual_row_paths_execute(self):
        spec, prepared, rows = self.fixture(); a = Audit()
        module.audit_rows(a, spec, prepared, rows, [], [], dict(requests=[], rows=[]))
        self.assertTrue(all(checks(a).values()), checks(a))

    def test_rejection_clearance_frozen_joint_and_namespace_tampering_fail(self):
        for kind in ("rejected_capture", "rejected_native", "clearance", "q14", "namespace"):
            spec, prepared, rows = self.fixture()
            if kind == "rejected_capture": rows[1]["route_success"] = True
            elif kind == "rejected_native": rows[1]["native_release"] = True
            elif kind == "clearance":
                rows[0]["minimum_clearance_lower_bound_m"] = .004
                rows[0]["geometry"]["minimum_clearance_lower_bound_m"] = .004
            elif kind == "q14":
                rows[0]["q14"][0] += .01; rows[0]["geometry"]["q14"][0] += .01
            else: rows[0]["id"] = "distance-shot-000000"
            a = Audit(); module.audit_rows(a, spec, prepared, rows, [], [], dict(requests=[], rows=[]))
            with self.subTest(kind=kind): self.assertFalse(all(checks(a).values()), checks(a))

    def test_local_repeat_geometry_and_joint_limit_branch_execute(self):
        spec, _, base = self.fixture(); source = base[0]
        nominal = {k: copy.deepcopy(source[k]) for k in module.GEOMETRY_KEYS}
        request = dict(context_id="recipe-00-local-03", recipe_id="recipe-00", repeat_id="recipe-00-repeat-03",
            role="engineering_repeat", phase="fresh", gap_m=source["gap_m"], values=copy.deepcopy(source["values"]),
            **{k: source[k] for k in ("distance_id", "ray_id", "distance_m")})
        request["values"][2] += .001
        geometry = copy.deepcopy(source["geometry"])
        geometry.update(context_id=request["context_id"], values=request["values"], wrapper_version="sircus-nominal-branch-local-variation-v1",
            nominal_geometry_sha256=digest(nominal), distance_wrapper_version="sircus-bucket-distance-local-variation-v1")
        geometry["q10"][0] += .02
        rows = []
        for i, step in enumerate((32, 64, 128)):
            row = copy.deepcopy(source); row.update(copy.deepcopy(request)); row.update(copy.deepcopy(geometry))
            row.update(id=f"lowboard-shot-{i:06d}", context_id=request["context_id"], substeps=step,
                geometry=copy.deepcopy(geometry))
            rows.append(row)
        recipes = [dict(id="recipe-00", source=source, bucket_xy_m=source["bucket_xy_m"])]
        gates = dict(requests=[request], rows=[geometry])
        a = Audit(); module.audit_rows(a, spec, [], rows, recipes, [request], gates)
        self.assertTrue(all(checks(a).values()), checks(a))
        for row in rows:
            row["q10"][0] = .101; row["geometry"]["q10"][0] = .101
        gates["rows"][0]["q10"][0] = .101
        a = Audit(); module.audit_rows(a, spec, [], rows, recipes, [request], gates)
        self.assertFalse(checks(a)["every_requested_target_and_frozen_source_branch_preserved"])


class MockCompletionTests(unittest.TestCase):
    def evaluate(self, directory, hits):
        args, report, rows = runner_tests.LowboardStudyTests().run_mock(Path(directory), hits)
        spec = json.loads((args.output / "contract.json").read_text())
        parent = json.loads((args.parent_study / "contract.json").read_text())
        declared = module.audit_targets_and_grid(Audit(), spec, parent)
        recipes = json.loads((args.output / "recipes.json").read_text())
        prepared = json.loads((args.output / "prepared-contexts.json").read_text())
        positions = {}
        for batch in json.loads((args.output / "batches.json").read_text()):
            if batch["cache_file"] is None: continue
            with np.load(args.output / batch["cache_file"], allow_pickle=False) as arrays:
                for i, name in enumerate(arrays["record_ids"]): positions[str(name)] = arrays["position_m"][:, i].copy()
        a = Audit(); module.audit_controls(a, args.output, spec, rows)
        module.audit_plans(a, args.output, spec, declared, rows, recipes)
        module.audit_report(a, spec, report, declared, prepared, rows, recipes, positions)
        module.audit_display_selection(a, spec, report, rows, recipes)
        self.assertTrue(all(checks(a).values()), checks(a))
        return args, report, spec, declared, rows, recipes

    def test_zero_recipe_completed_plan_report_and_posthoc_misses(self):
        with tempfile.TemporaryDirectory() as directory:
            args, report, spec, declared, rows, recipes = self.evaluate(directory, False)
            self.assertEqual(recipes, []); self.assertEqual(len(rows), 664)
            # A global plan edit is detected without native execution.
            coarse = json.loads((args.output / "coarse-plan.json").read_text()); coarse[0]["gap_m"] += .001
            write(args.output, "coarse-plan.json", coarse)
            a = Audit(); module.audit_plans(a, args.output, spec, declared, rows, recipes)
            self.assertFalse(checks(a)["all_640_exact_unclipped_near_ceiling_requests"])

    def test_max_eight_recipes_and_frozen_three_resolution_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            args, report, spec, declared, rows, recipes = self.evaluate(directory, True)
            self.assertEqual(len(recipes), 8)
            self.assertEqual(sum(r["role"] == "engineering_repeat" for r in rows), 792)
            self.assertLessEqual(len(rows), 1968)
            # Neither a dropped repeat nor a forged post-hoc display choice can pass.
            removed = rows[:-1]; a = Audit(); module.audit_plans(a, args.output, spec, declared, removed, recipes)
            self.assertFalse(checks(a)["complete_budget_and_no_hidden_retry_or_resampling"])
            report["trace_examples"][-1]["record_id"] = "not-a-linked-record"
            a = Audit(); module.audit_display_selection(a, spec, report, rows, recipes)
            self.assertFalse(checks(a)["native_nominal_and_explicit_posthoc_development_examples"])


if __name__ == "__main__": unittest.main()
