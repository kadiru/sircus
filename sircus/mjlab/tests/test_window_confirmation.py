"""Bounded confirmation selection/provenance checks; no GPU/native jobs."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from sircus.mjlab import window_contract as protocol
from sircus.mjlab.bank_study import unresolved_row
from sircus.mjlab.contact_study import atomic
from sircus.mjlab.corrected_contract import sha
from sircus.mjlab.contract import FORCE_NAMES, ScoreStream
from sircus.mjlab.learning_contract import digest, reward
from sircus.mjlab.window_confirmation import (PARENT_FILES, SCOPE, SEED,
    audit_confirmation, load_parent, plan_confirmation)
from sircus.mjlab.window_verify import stats


def original_rows():
    settings = protocol.settings(); contexts = settings["contexts"]["train"]
    chosen = [next(c for c in contexts if c["board_height_m"] == h and abs(c["tilt_deg"] - 36.154791124080255) < 1e-9)
              for h in (.3973878458850598, .5)]
    points = [[(.20, False), (.22475, True), (.25, False), (.272, True), (.32, True), (.3704375, True), (.40, False)],
              [(.15, False), (.1775, True), (.1814375, True), (.19, False), (.209, True), (.24, True), (.2680625, True), (.29, False)]]
    rows = []
    for c, values in zip(chosen, points):
        for gap, hit in values:
            rows.append(dict(id=f"train-{len(rows)}", context_id=c["id"], role="train", method="shared",
                gap_m=gap, route_success=hit, native_release=True))
    return settings, chosen, rows


def write_parent(path):
    path.mkdir(); settings, chosen, rows = original_rows()
    scene = json.loads((Path(__file__).resolve().parents[2] / "isaac/scene.json").read_text())
    contexts = [dict(c, q14=[1.] * 6, board_branch_sha256="fixed-" + c["id"]) for c in chosen]
    spec = dict(settings=settings, scene=scene, physics=dict(test="mock"), anchor_geometry=dict(test="geometry"))
    spec["fingerprint"] = digest(spec); atomic(path / "contract.json", spec)
    (path / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows)); atomic(path / "prepared-contexts.json", contexts)
    atomic(path / "audit-window.json", dict(status="passed", failures=[], contract_fingerprint=spec["fingerprint"]))
    manifest = {n: sha(path / n) for n in ("contract.json", "records.jsonl", "prepared-contexts.json")}
    atomic(path / "evidence-manifest.json", manifest)
    atomic(path / "report.json", dict(status="completed", contract_fingerprint=spec["fingerprint"], observed_windows=protocol.windows(rows), evidence_sha256=manifest))
    return spec, contexts


def write_confirmation(root):
    parent = root / "parent"; original, contexts = write_parent(parent)
    _, recipes, plan, _ = load_parent(parent)
    output = root / "confirmation"; output.mkdir(); source = root / "source"; source.mkdir(); assets = root / "assets"; assets.mkdir()
    (source / "source.py").write_text("# immutable mock source\n"); (assets / "arm.urdf").write_text("<robot/>\n")
    spec = dict(parent_contract_fingerprint=original["fingerprint"], parent_file_sha256={n: sha(parent / n) for n in PARENT_FILES},
        source_sha256={"source.py": sha(source / "source.py")}, asset_sha256={"assets/arm.urdf": sha(assets / "arm.urdf")},
        scene=original["scene"], physics=original["physics"], anchor_geometry=original["anchor_geometry"], frozen_contexts=contexts,
        recipes=recipes, plan_sha256=digest(plan), scope=SCOPE)
    spec["fingerprint"] = digest(spec); atomic(output / "contract.json", spec); atomic(output / "plan.json", plan)
    selected = {c["id"]: c for c in contexts}; rows = []
    for request in plan:
        c = selected[request["context_id"]]
        g = dict(q10=[0.] * 6, q14=c["q14"], board_branch_sha256=c["board_branch_sha256"], eligible=False,
            minimum_clearance_lower_bound_m=None, bucket_xy_m=[.16, -.64], values=[.47, -.28, c["board_height_m"], c["tilt_deg"], request["gap_m"]])
        row = dict(unresolved_row(g["bucket_xy_m"]), **request); row.update(g)
        row.update(geometry=copy.deepcopy(g), native_cache_file=None, native_cache_row=None); row["learning_reward"] = reward(row); rows.append(row)
    (output / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    atomic(output / "batches.json", [dict(packet=i, record_ids=[r["id"] for r in rows[n:n + 64]], native_record_ids=[], native_cache_file=None) for i, n in enumerate((0, 64))])
    effective = dict(timestep_s=1 / 15360, solver=2, integrator=3, cone=0, iterations=100, tolerance=1e-8,
        gravity=[0., 0., -9.81], substeps=64, board_mode="articulated", gravity_mode="passive_only")
    atomic(output / "effective-options.json", effective)
    report = dict(status="completed", contract_fingerprint=spec["fingerprint"], parent_contract_fingerprint=original["fingerprint"], recipes=[], effective_native_options=effective)
    for recipe in recipes:
        rr = [r for r in rows if r["recipe_id"] == recipe["id"]]
        report["recipes"].append(dict(recipe, **{p: stats(rr if p == "total" else [r for r in rr if r["phase"] == p]) for p in ("nominal", "fresh", "total")}))
    refresh_manifest(output, report)
    return output, parent, source, assets, rows


def refresh_manifest(output, report=None):
    report = report or json.loads((output / "report.json").read_text())
    manifest = {str(p.relative_to(output)): sha(p) for p in output.rglob("*") if p.is_file() and p.name not in ("report.json", "evidence-manifest.json") and not p.name.startswith("audit")}
    report["evidence_sha256"] = manifest; atomic(output / "evidence-manifest.json", manifest); atomic(output / "report.json", report)


class WindowConfirmationTests(unittest.TestCase):
    def test_all_four_train_runs_have_midpoints_and_92_seeded_attempts(self):
        settings, _, rows = original_rows(); observed = protocol.windows(rows)
        recipes, plan = plan_confirmation(rows, settings, observed)
        self.assertEqual(len(plan), 92); self.assertEqual(len(recipes), 4)
        np.testing.assert_allclose([r["gap_m"] for r in recipes], [.22475, .32121875, .17946875, .23853125])
        self.assertEqual(sum(r["phase"] == "nominal" for r in plan), 12)
        rng = np.random.default_rng(SEED)
        for recipe in recipes:
            selected = [p for p in plan if p["recipe_id"] == recipe["id"]]
            self.assertEqual([r["repeat_index"] for r in selected], list(range(23)))
            np.testing.assert_array_equal([r["gap_noise_m"] for r in selected[3:]], rng.normal(0, .002, 20))
            self.assertTrue(all(r["gap_m"] == r["nominal_gap_m"] + r["gap_noise_m"] for r in selected))

    def test_sac_and_heldout_hits_cannot_select_confirmation_recipes(self):
        settings, _, rows = original_rows(); observed = protocol.windows(rows)
        expected = plan_confirmation(rows, settings, observed)
        for role, method in (("train", "sac"), ("validation", "shared"), ("test", "adaptive_surrogate")):
            bad = dict(rows[0], role=role, method=method, route_success=True, gap_m=.21)
            self.assertEqual(plan_confirmation(rows + [bad], settings, observed), expected)
        with self.assertRaises(ValueError): plan_confirmation(rows, settings, [])
        rows[1]["route_success"] = False
        with self.assertRaises(ValueError): plan_confirmation(rows, settings, observed)

    def test_parent_must_be_completed_audited_and_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory) / "parent"; write_parent(parent)
            self.assertEqual(len(load_parent(parent)[2]), 92)
            report = json.loads((parent / "report.json").read_text()); report["status"] = "running"; atomic(parent / "report.json", report)
            with self.assertRaises(ValueError): load_parent(parent)
            report["status"] = "completed"; atomic(parent / "report.json", report)
            with (parent / "records.jsonl").open("a") as stream: stream.write("{}\n")
            with self.assertRaises(ValueError): load_parent(parent)

    def test_all_invalid_attempts_remain_audited_not_silently_dropped(self):
        with tempfile.TemporaryDirectory() as directory:
            output, parent, source, assets, _ = write_confirmation(Path(directory))
            result = audit_confirmation(output, parent, source, assets)
            self.assertEqual(result["status"], "passed", result["failures"])
            self.assertEqual(result["independently_rescored_throws"], 0)
            rows = [json.loads(line) for line in (output / "records.jsonl").read_text().splitlines()][:-1]
            (output / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows)); refresh_manifest(output)
            result = audit_confirmation(output, parent, source, assets)
            self.assertEqual(result["status"], "failed")

    def test_source_and_asset_changes_fail_despite_self_consistent_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            output, parent, source, assets, _ = write_confirmation(Path(directory))
            (source / "source.py").write_text("# changed\n"); (assets / "arm.urdf").write_text("<robot changed='yes'/>\n")
            result = audit_confirmation(output, parent, source, assets)
            failed = {c["name"] for c in result["failures"]}
            self.assertTrue({"executed_source", "native_assets"}.issubset(failed))

    def test_loose_solver_tolerance_cannot_pass_effective_options_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            output, parent, source, assets, _ = write_confirmation(Path(directory))
            report = json.loads((output / "report.json").read_text()); report["effective_native_options"]["tolerance"] = 1.5e-8
            atomic(output / "effective-options.json", report["effective_native_options"]); refresh_manifest(output, report)
            failed = {c["name"] for c in audit_confirmation(output, parent, source, assets)["failures"]}
            self.assertIn("effective_corrected64_options_and_report", failed)

    def test_changed_frozen_branch_and_fake_release_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            output, parent, source, assets, rows = write_confirmation(Path(directory))
            rows[0]["q14"] = [999.] * 6; rows[1]["native_release"] = True
            (output / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows)); refresh_manifest(output)
            failed = {c["name"] for c in audit_confirmation(output, parent, source, assets)["failures"]}
            self.assertTrue({"frozen_geometry_rejections_and_reward", "batch_native_membership_0", "every_attempt_and_release_once"}.issubset(failed))

    def test_each_raw_trace_is_independently_rescored_and_bound_to_record_id(self):
        with tempfile.TemporaryDirectory() as directory:
            output, parent, source, assets, rows = write_confirmation(Path(directory))
            scene = json.loads((output / "contract.json").read_text())["scene"]
            row = rows[0]; position = np.tile([.5, -.2, -.3], (960, 1, 1)); velocity = np.zeros_like(position); force = np.zeros((960, 1, 6))
            scorer = ScoreStream(scene, row["bucket_xy_m"], [0., 0., 1.])
            scorer.step(position[0, 0], velocity[0, 0], dict(zip(FORCE_NAMES, force[0, 0])))
            row.update(scorer.result()); row.update(eligible=True, invalid=None, native_release=True, board_normal=[0., 0., 1.],
                contact_protocol=dict(substeps=64, board_mode="articulated", gravity_mode="passive_only"),
                minimum_clearance_lower_bound_m=.006, native_cache_file="native-00.npz", native_cache_row=0)
            row["geometry"].update(eligible=True, minimum_clearance_lower_bound_m=.006); row["learning_reward"] = reward(row)
            batches = json.loads((output / "batches.json").read_text()); batches[0].update(native_record_ids=[row["id"]], native_cache_file="native-00.npz")
            atomic(output / "batches.json", batches)
            np.savez_compressed(output / "native-00.npz", position_m=position, velocity_m_s=velocity, forces_N=force, record_ids=np.asarray([row["id"]]))
            (output / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            report = json.loads((output / "report.json").read_text())
            for recipe in report["recipes"]:
                rr = [r for r in rows if r["recipe_id"] == recipe["id"]]
                recipe.update({p: stats(rr if p == "total" else [r for r in rr if r["phase"] == p]) for p in ("nominal", "fresh", "total")})
            refresh_manifest(output, report)
            result = audit_confirmation(output, parent, source, assets)
            self.assertEqual(result["status"], "passed", result["failures"])
            self.assertEqual(result["independently_rescored_throws"], 1)
            np.savez_compressed(output / "native-00.npz", position_m=position, velocity_m_s=velocity, forces_N=force, record_ids=np.asarray(["wrong-record"]))
            refresh_manifest(output)
            failed = {c["name"] for c in audit_confirmation(output, parent, source, assets)["failures"]}
            self.assertIn("batch-0:record_ids", failed)


if __name__ == "__main__": unittest.main()
