"""Mock orchestration tests; fabricated flights are never physics evidence."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np

from sircus.mjlab import distance_contract as protocol
from sircus.mjlab import distance_study as study
from sircus.mjlab.distance_geometry import table_footprint
from sircus.mjlab.learning_contract import digest


ROOT = Path(__file__).resolve().parents[3]


def write(path, value):
    path.write_text(json.dumps(value))


def parent_fixture(root):
    directory = root / "parent"; directory.mkdir()
    scene = json.loads((ROOT / "sircus/isaac/scene.json").read_text())
    curated = json.loads((ROOT / "baselines/mjlab-height-tilt-20261102/completed/recipe-comparison.json").read_text())
    sources = {r["recipe_id"]: r for r in curated["recipes"]}
    recipes = []
    for name in protocol.SEED_IDS:
        row = sources[name]
        geometry = dict(values=[.4708757793012148, -.2778934014509746,
            row["board_height_m"], row["tilt_deg"], row["drop_gap_m"]],
            bucket_xy_m=[.16275255472120964, -.6365356466995716], q10=[0.] * 6, q14=[1.] * 6)
        recipes.append(dict(id=name, source=dict(**geometry, id="source-" + name,
            context_id="context-" + name, native_release=True, route_success=True, invalid=None)))
    spec = dict(scene=scene, packages={}, asset_sha256={}, source_sha256={})
    spec["fingerprint"] = digest(spec)
    write(directory / "contract.json", spec); write(directory / "recipes.json", recipes)
    manifest = {name: study.sha(directory / name) for name in ("contract.json", "recipes.json")}
    write(directory / "evidence-manifest.json", manifest)
    write(directory / "report.json", dict(status="completed", contract_fingerprint=spec["fingerprint"],
        evidence_sha256=manifest, selected_recipes=[dict(id=r["id"], source_record_id=r["source"]["id"],
        by_substeps={str(s): dict(attempts=23, captures=23) for s in protocol.SUBSTEPS}) for r in recipes]))
    write(directory / "audit-neighborhood.json", dict(status="passed", failures=[],
        contract_fingerprint=spec["fingerprint"], checks=[dict(passed=True)]))
    return directory


def gate(scene, source, context, gap, eligible=True):
    values = source["values"][:2] + [context["board_height_m"], context["tilt_deg"], gap]
    footprint = table_footprint(scene, context["bucket_xy_m"])
    admitted = eligible and footprint["passed"]
    return dict(context_id=context["id"], eligible=admitted,
        reason=None if admitted else "mock_rejection", values=values,
        bucket_xy_m=context["bucket_xy_m"], q10=source["q10"], q14=source["q14"],
        board_branch_sha256="mock:" + context["id"],
        minimum_clearance_lower_bound_m=.01 if admitted else None)


class FakeGeometry:
    def __init__(self, scene, assets, seeds, bounds):
        self.scene = scene; self.seeds = {s["id"]: s for s in seeds}; self.contexts = {}

    def prepare_contexts(self, contexts):
        self.contexts.update({c["id"]: c for c in contexts})
        return [dict(**c, board_admitted=table_footprint(self.scene, c["bucket_xy_m"])["passed"],
            reason=None, board_branch_sha256="mock:" + c["id"]) for c in contexts]

    def audit(self, ids, gaps):
        return [gate(self.scene, self.seeds[self.contexts[name]["source_anchor_id"]],
            self.contexts[name], gap) for name, gap in zip(ids, gaps)]


class FakeEnvironment:
    def __init__(self, scene, assets, n, bounds, substeps):
        self.substeps = substeps; self.effective_options = dict(substeps=substeps)


def unresolved(bucket):
    return dict(invalid="mock", native_release=False, route_success=False, route_quality=0,
        board_bounce=False, whole_ball_entry=False, closest_opening_distance_m=999.,
        max_retention_s=0., external_contact_steps=0, single_bounce_success=False)


def fake_flight(env, bucket, values, q10, q14, eligible, record):
    n = int(sum(eligible))
    env.last_arrays = dict(position_m=np.zeros((960, n, 3), np.float32),
        velocity_m_s=np.zeros((960, n, 3), np.float32), forces_N=np.zeros((960, n, 6), np.float32))
    result = []
    board = np.asarray([.4708757793012148, -.2778934014509746])
    reference = np.asarray([.16275255472120964, -.6365356466995716]) - board
    for ok, xy, value in zip(eligible, bucket, values):
        if not ok:
            result.append(unresolved(xy)); continue
        ray = xy - board
        original = abs(ray[0] * reference[1] - ray[1] * reference[0]) < 1e-12
        # One coarse corner context scores, triggering the prespecified far
        # extension; lower-tilt extension contexts score in this mock.
        success = bool(original or value[3] < 30.)
        result.append(dict(invalid=None, native_release=True, route_success=success,
            route_quality=1 if success else 0, board_bounce=True, whole_ball_entry=success,
            closest_opening_distance_m=.01 if success else .3, max_retention_s=1. if success else 0.,
            external_contact_steps=0, single_bounce_success=success,
            contact_protocol=dict(substeps=env.substeps, board_mode="articulated", gravity_mode="passive_only")))
    return result


class DistanceStudyTests(unittest.TestCase):
    def test_parent_tampering_and_failed_audit_block_preparation(self):
        for kind in ("raw", "audit", "seed"):
            with tempfile.TemporaryDirectory() as directory:
                parent = parent_fixture(Path(directory))
                if kind == "raw": (parent / "recipes.json").write_text("[]")
                elif kind == "audit":
                    audit = study.read(parent / "audit-neighborhood.json"); audit["checks"][0]["passed"] = False
                    write(parent / "audit-neighborhood.json", audit)
                else:
                    report = study.read(parent / "report.json")
                    report["selected_recipes"][0]["by_substeps"]["32"]["captures"] = 22
                    write(parent / "report.json", report)
                with self.subTest(kind=kind), self.assertRaises(ValueError): study.prepare_inputs(parent)

    def test_full_mock_preserves_targets_conditional_extension_and_paired_rejections(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); parent = parent_fixture(root)
            args = NS(parent_study=parent, assets=root / "assets", output=root / "out",
                batch=64, wall_time_limit=120)
            def variations(scene, assets, nominal, requests, bounds):
                source = {k: nominal[k] for k in ("values", "bucket_xy_m", "q10", "q14")}
                return [gate(scene, source, dict(id=r["context_id"], bucket_xy_m=source["bucket_xy_m"],
                    board_height_m=r["values"][2], tilt_deg=r["values"][3]),
                    r["values"][4], eligible=r["repeat_index"] != 3) for r in requests]
            modules = {
                "sircus.mjlab.bank_study": NS(run_bank_rows=fake_flight, unresolved_row=unresolved),
                "sircus.mjlab.learning_solve": NS(required_asset_paths=lambda _: []),
                "sircus.mjlab.distance_geometry": NS(DistanceGeometry=FakeGeometry, audit_distance_variations=variations),
                "sircus.mjlab.timestep_env": NS(TimestepEnvironment=FakeEnvironment, physics_settings=lambda s: dict(substeps=s))}
            with patch.dict("sys.modules", modules), patch.object(study, "validate_packages", lambda p: p), \
                    contextlib.redirect_stdout(io.StringIO()):
                report = study._run(args)
            rows = [json.loads(x) for x in (args.output / "records.jsonl").read_text().splitlines()]
            self.assertEqual(report["status"], "completed"); self.assertEqual(len(report["targets"]), 14)
            self.assertEqual(report["counts"]["requests"], len(rows)); self.assertLessEqual(len(rows), 5589)
            self.assertEqual(sum(r["role"] == "control" for r in rows), 24)
            self.assertEqual(sum(r["phase"] == "coarse" for r in rows), 1904)
            self.assertEqual(len(study.read(args.output / "extension-contexts.json")), 96)
            self.assertEqual(sum(r["phase"] == "far_extension" for r in rows), 1344)
            for target in report["targets"]:
                group = [r for r in rows if r["distance_id"] == target["id"]]
                self.assertTrue(all(r["bucket_xy_m"] == target["bucket_xy_m"] for r in group))
                if not target["bucket_on_table"]:
                    self.assertTrue(group); self.assertTrue(all(not r["native_release"] and r["invalid"] for r in group))
            for repeat in {r["repeat_id"] for r in rows if r["role"] == "engineering_repeat"}:
                group = [r for r in rows if r.get("repeat_id") == repeat]
                self.assertEqual(sorted(r["substeps"] for r in group), [32, 64, 128])
                self.assertTrue(all(all(r[k] == group[0][k] for k in ("values", "bucket_xy_m", "q10", "q14")) for r in group))
                if group[0]["repeat_index"] == 3:
                    self.assertTrue(all(r["invalid"] and not r["native_release"] for r in group))
            for name, expected in report["evidence_sha256"].items():
                self.assertEqual(study.sha(args.output / name), expected)


if __name__ == "__main__": unittest.main()
