"""Quarantined 92-shot engineering repeats of TRAIN-observed release-gap runs.

No policy update, inference search, retry, clipping or promotion is performed.
Only the completed original study's shared/adaptive TRAIN records select recipes.
"""
import argparse
from collections import Counter
import copy
import importlib.metadata
import json
from pathlib import Path

import numpy as np

from . import window_contract as protocol
from .contact_study import atomic
from .corrected_contract import sha, validate_packages
from .corrected_grid import source_manifest, storage_check
from .learning_contract import digest, reward
from .verify_expansion import Audit
from .window_verify import _inside, audit_native, lines, read, row_reward, stats

VERSION = "sircus-train-window-engineering-confirmation-v1"
SEED = 20261204
SCOPE = "Post-study TRAIN-derived engineering repeats; excluded from original learner comparison and all policy selection; held endpoints only, no path/actuator validation"
PARENT_FILES = ("contract.json", "report.json", "records.jsonl", "prepared-contexts.json", "audit-window.json", "evidence-manifest.json")
DISPLAY_ONLY_SOURCE_EXCEPTION = "sircus/mjlab/prepare_window_report.py"


def source_changes(original, current):
    changes = {n: dict(parent=h, confirmation=current.get(n)) for n, h in original.items() if current.get(n) != h}
    if set(changes) - {DISPLAY_ONLY_SOURCE_EXCEPTION}:
        raise ValueError("Original source changed outside the documented display-only exception: " + str(sorted(changes)))
    return changes


def plan_confirmation(rows, settings, observed):
    train = [r for r in rows if r.get("role") == "train" and r.get("method") in ("shared", "adaptive_surrogate")]
    rebuilt = protocol.windows(train)
    if rebuilt != observed:
        raise ValueError("Original observed windows do not reconstruct from shared/adaptive TRAIN")
    contexts = {c["id"]: c for c in settings["contexts"]["train"]}
    recipes, plans, rng = [], [], np.random.default_rng(SEED)
    for window in rebuilt:
        for run in window["observed_hit_runs"]:
            c = contexts[window["context_id"]]; lo, hi = run["low_observed_gap_m"], run["high_observed_gap_m"]
            recipe = dict(id=f"recipe-{len(recipes):02d}", context_id=c["id"], board_height_m=c["board_height_m"],
                tilt_deg=c["tilt_deg"], observed_low_gap_m=lo, observed_high_gap_m=hi, gap_m=(lo + hi) / 2)
            recipes.append(recipe)
            for i, noise in enumerate([0.] * 3 + rng.normal(0, .002, 20).tolist()):
                plans.append(dict(id=f"confirmation-{len(plans):03d}", recipe_id=recipe["id"], context_id=c["id"],
                    role="engineering_confirmation", method="train_window_repeat", phase="nominal" if i < 3 else "fresh",
                    repeat_index=i, nominal_gap_m=recipe["gap_m"], gap_noise_m=float(noise), gap_m=float(recipe["gap_m"] + noise)))
    expected = sorted([(.22475, .22475), (.272, .3704375), (.1775, .1814375), (.209, .2680625)])
    actual = sorted((r["observed_low_gap_m"], r["observed_high_gap_m"]) for r in recipes)
    if len(recipes) != 4 or not np.allclose(actual, expected, atol=1e-10, rtol=0):
        raise ValueError("This bounded confirmation requires the four declared original TRAIN runs")
    return recipes, plans


def load_parent(study):
    study = Path(study); spec, report, audit = [read(study / name) for name in ("contract.json", "report.json", "audit-window.json")]
    payload = dict(spec); fp = payload.pop("fingerprint", None)
    if digest(payload) != fp or report.get("contract_fingerprint") != fp or audit.get("contract_fingerprint") != fp:
        raise ValueError("Original contract/report/audit fingerprint mismatch")
    if report.get("status") != "completed" or audit.get("status") != "passed" or audit.get("failures"):
        raise ValueError("Completed independently audited original study required")
    manifest = read(study / "evidence-manifest.json")
    if manifest != report.get("evidence_sha256") or any(sha(_inside(study, name)) != value for name, value in manifest.items()):
        raise ValueError("Original evidence changed after its audit")
    recipes, plans = plan_confirmation(lines(study / "records.jsonl"), spec["settings"], report["observed_windows"])
    if spec["settings"]["action_bounds"] != protocol.BOUNDS:
        raise ValueError("Action bounds differ from the original study")
    return spec, recipes, plans, read(study / "prepared-contexts.json")


def audit_confirmation(output, parent_study, source, assets):
    """CPU audit of exact plans, frozen branches, rejected slots and every trace."""
    output = Path(output); spec, report = read(output / "contract.json"), read(output / "report.json")
    rows, plan, batches = lines(output / "records.jsonl"), read(output / "plan.json"), read(output / "batches.json")
    a = Audit(); payload = dict(spec); fp = payload.pop("fingerprint", None)
    a.check("contract_fingerprint", digest(payload) == fp == report.get("contract_fingerprint"))
    a.check("completed_quarantined_confirmation", report.get("status") == "completed" and report.get("parent_contract_fingerprint") == spec["parent_contract_fingerprint"] and spec.get("scope") == SCOPE)
    original, recipes, regenerated, parent_contexts = load_parent(parent_study)
    a.check("immutable_audited_parent_and_train_only_plan", original["fingerprint"] == spec["parent_contract_fingerprint"]
        and spec["parent_file_sha256"] == {n: sha(Path(parent_study) / n) for n in PARENT_FILES}
        and recipes == spec["recipes"] and regenerated == plan)
    for key, root, identities in (("executed_source", source, spec["source_sha256"]), ("native_assets", Path(assets).parent, spec["asset_sha256"])):
        a.check(key, bool(identities) and all(sha(_inside(root, n)) == h for n, h in identities.items()))
    a.check("original_core_source_unchanged", source_changes(original.get("source_sha256", {}), spec["source_sha256"]) == spec.get("display_only_source_changes", {}))
    parent_contexts = {c["id"]: c for c in parent_contexts}
    a.check("original_scene_physics_anchor_and_frozen_branches", all(spec[k] == original[k] for k in ("scene", "physics", "anchor_geometry"))
        and all(c["q14"] == parent_contexts[c["id"]]["q14"] and c["board_branch_sha256"] == parent_contexts[c["id"]]["board_branch_sha256"] for c in spec["frozen_contexts"]))
    if spec.get("packages"):
        a.check("matching_auditor_packages", all(importlib.metadata.version(n) == v for n, v in spec["packages"].items()))
    effective = read(output / "effective-options.json")
    fixed = dict(timestep_s=1 / 15360, solver=2, integrator=3, cone=0, iterations=100,
        gravity=[0., 0., -9.81], substeps=64, board_mode="articulated", gravity_mode="passive_only")
    a.check("effective_corrected64_options_and_report", effective == report.get("effective_native_options")
        and all(effective.get(k) == v for k, v in fixed.items()) and np.isclose(effective.get("tolerance", 0), 1e-8, rtol=0, atol=1e-12))
    a.check("complete_92_frozen_requests", len(rows) == len(plan) == 92 and [r["id"] for r in rows] == [r["id"] for r in plan]
        and len({r["id"] for r in rows}) == 92 and spec["plan_sha256"] == digest(plan)
        and all(all(r.get(k) == v for k, v in p.items()) for r, p in zip(rows, plan)))
    prepared = {c["id"]: c for c in spec["frozen_contexts"]}; bad = []
    for r in rows:
        g = r["geometry"]; c = prepared[r["context_id"]]
        if (r["q14"] != c["q14"] or r["board_branch_sha256"] != c["board_branch_sha256"]
                or any(r[k] != g[k] for k in ("q10", "q14", "values", "bucket_xy_m", "eligible"))
                or r["values"][4] != r["gap_m"] or r["native_release"] != r["eligible"]
                or (not r["eligible"] and (not r.get("invalid") or r["route_success"]))
                or (r["eligible"] and r["minimum_clearance_lower_bound_m"] < .005)
                or not np.isclose(r["learning_reward"], row_reward(r), rtol=0, atol=1e-9)):
            bad.append(r["id"])
    a.check("frozen_geometry_rejections_and_reward", not bad, bad)
    lookup = {r["id"]: r for r in rows}; batch_ids, rescored = [], []
    for batch in batches:
        batch_ids.extend(batch["record_ids"])
        if any(i not in lookup for i in batch["record_ids"] + batch["native_record_ids"]):
            a.check("batch_records_present_" + str(batch["packet"]), False); continue
        native = [lookup[i] for i in batch["native_record_ids"]]
        a.check("batch_native_membership_" + str(batch["packet"]), batch["native_record_ids"] == [i for i in batch["record_ids"] if lookup[i]["native_release"]])
        if native:
            with np.load(_inside(output, batch["native_cache_file"]), allow_pickle=False) as arrays:
                audit_native(a, spec["scene"], native, arrays, "batch-" + str(batch["packet"]))
            rescored.extend(r["id"] for r in native)
        else:
            a.check("empty_native_batch_" + str(batch["packet"]), batch["native_cache_file"] is None)
        a.check("batch_cache_pointers_" + str(batch["packet"]), all(r.get("native_cache_file") == batch["native_cache_file"]
            and r.get("native_cache_row") == i for i, r in enumerate(native)) and all(lookup[i].get("native_cache_file") is None
            and lookup[i].get("native_cache_row") is None for i in batch["record_ids"] if not lookup[i]["native_release"]))
    a.check("every_attempt_and_release_once", batch_ids == [r["id"] for r in rows]
        and Counter(rescored) == Counter(r["id"] for r in rows if r["native_release"]))
    manifest = read(output / "evidence-manifest.json")
    files = {str(p.relative_to(output)) for p in output.rglob("*") if p.is_file() and p.name not in ("report.json", "evidence-manifest.json") and not p.name.startswith("audit")}
    a.check("complete_evidence_manifest", set(manifest) == files and manifest == report.get("evidence_sha256") and all(sha(_inside(output, n)) == h for n, h in manifest.items()))
    for recipe in report["recipes"]:
        rr = [r for r in rows if r["recipe_id"] == recipe["id"]]
        for phase in ("nominal", "fresh", "total"):
            a.stats(recipe["id"] + ":" + phase, recipe[phase], stats(rr if phase == "total" else [r for r in rr if r["phase"] == phase]))
    failures = [c for c in a.checks if not c["passed"]]
    return dict(version=VERSION, status="failed" if failures else "passed", contract_fingerprint=fp,
        checks=a.checks, checks_count=len(a.checks), failures=failures, independently_rescored_throws=len(rescored), scope=SCOPE)


def run(args, parent):
    from .bank_study import run_bank_rows
    from .contact_env import ContactEnvironment
    from .learning_solve import required_asset_paths
    from .window_geometry import WindowGeometry
    original, recipes, plan, original_contexts = parent; scene = copy.deepcopy(original["scene"])
    geometry = WindowGeometry(scene, args.assets, original["anchor_geometry"], bounds=protocol.BOUNDS)
    wanted = list(dict.fromkeys(r["context_id"] for r in recipes)); prior = {r["id"]: r for r in original_contexts}
    contexts = geometry.prepare_contexts([dict(id=i, board_height_m=prior[i]["board_height_m"], tilt_deg=prior[i]["tilt_deg"]) for i in wanted])
    if any(not c["board_admitted"] or any(c[k] != prior[c["id"]][k] for k in ("q14", "board_branch_sha256")) for c in contexts):
        raise RuntimeError("Frozen board context differs from original; no branch replacement allowed")
    source = source_manifest(Path(__file__).resolve().parents[2])
    changes = source_changes(original["source_sha256"], source)
    packages = validate_packages({n: importlib.metadata.version(n) for n in original["packages"]})
    assets = {n: sha(args.assets.parent / n) for n in required_asset_paths(args.assets)}
    if packages != original["packages"] or assets != original["asset_sha256"]:
        raise RuntimeError("Original package or native asset identities changed")
    args.output.mkdir(parents=True, exist_ok=False)
    storage_check(args.output)
    spec = dict(version=VERSION, parent_contract_fingerprint=original["fingerprint"], parent_file_sha256={n: sha(args.study / n) for n in PARENT_FILES},
        source_sha256=source, display_only_source_changes=changes, packages=packages, asset_sha256=assets,
        scene=scene, physics=original["physics"], anchor_geometry=original["anchor_geometry"], frozen_contexts=contexts,
        settings=dict(seed=SEED, nominal=3, fresh=20, gap_sigma_m=.002, bounds=protocol.BOUNDS, clipping=False, retries=0),
        recipes=recipes, plan_sha256=digest(plan), scope=SCOPE)
    spec["fingerprint"] = digest(spec); atomic(args.output / "contract.json", spec); atomic(args.output / "plan.json", plan)
    report = dict(version=VERSION, status="running", contract_fingerprint=spec["fingerprint"], parent_contract_fingerprint=original["fingerprint"], settings=spec["settings"], physics=spec["physics"], scope=SCOPE)
    atomic(args.output / "report.json", report)
    env = ContactEnvironment(scene, args.assets, n=64, bounds=protocol.BOUNDS, substeps=64, board_mode="articulated", gravity_mode="passive_only")
    opt = env.sim.mj_model.opt
    if (not np.isclose(opt.timestep, 1 / 15360) or int(opt.solver) != 2 or int(opt.integrator) != 3 or int(opt.cone) != 0
            or opt.iterations != 100 or not np.isclose(opt.tolerance, 1e-8, rtol=0, atol=1e-12) or not np.array_equal(opt.gravity, [0, 0, -9.81])):
        raise RuntimeError("Effective corrected-contact native options differ")
    effective = dict(timestep_s=float(opt.timestep), solver=int(opt.solver), integrator=int(opt.integrator), cone=int(opt.cone),
        iterations=int(opt.iterations), tolerance=float(opt.tolerance), gravity=opt.gravity.tolist(), substeps=64, board_mode="articulated", gravity_mode="passive_only")
    atomic(args.output / "effective-options.json", effective); report["effective_native_options"] = effective
    rows, batches = [], []
    for offset in range(0, 92, 64):
        chunk = plan[offset:offset + 64]; gates = geometry.audit([r["context_id"] for r in chunk], [r["gap_m"] for r in chunk])
        eligible = np.asarray([g["eligible"] for g in gates]); indices = np.flatnonzero(eligible); packet = len(batches)
        atomic(args.output / f"geometry-{packet:02d}.json", gates)
        measured = run_bank_rows(env, np.asarray([g["bucket_xy_m"] for g in gates]), np.asarray([g["values"] for g in gates]),
            np.asarray([g["q10"] for g in gates]), np.asarray([g["q14"] for g in gates]), eligible, record=True)
        if any(bool(r["native_release"]) != bool(ok) for r, ok in zip(measured, eligible)): raise RuntimeError("Native endpoint admission changed")
        cache = f"native-{packet:02d}.npz" if len(indices) else None
        if cache: np.savez_compressed(args.output / cache, **env.last_arrays, record_ids=np.asarray([chunk[i]["id"] for i in indices]))
        mapping = {int(i): j for j, i in enumerate(indices)}
        for i, (request, g, result) in enumerate(zip(chunk, gates, measured)):
            row = dict(result, **request); row.update({k: g[k] for k in ("values", "bucket_xy_m", "q10", "q14", "eligible", "board_branch_sha256", "minimum_clearance_lower_bound_m")})
            row.update(geometry=g, native_cache_file=cache if eligible[i] else None, native_cache_row=mapping.get(i))
            if not eligible[i]: row["invalid"] = g["reason"]
            row["learning_reward"] = reward(row); rows.append(row)
        batches.append(dict(packet=packet, record_ids=[r["id"] for r in chunk], native_record_ids=[chunk[i]["id"] for i in indices], native_cache_file=cache))
        (args.output / "records.jsonl").write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in rows)); atomic(args.output / "batches.json", batches)
        report["storage"] = storage_check(args.output)
    report["recipes"] = []
    for recipe in recipes:
        rr = [r for r in rows if r["recipe_id"] == recipe["id"]]; out = dict(recipe, release_center_height_m=recipe["board_height_m"] + scene["ball"]["radius_m"] + recipe["gap_m"],
            board_to_bucket_distance_m=float(np.linalg.norm(np.asarray(rr[0]["values"][:2]) - rr[0]["bucket_xy_m"])), scope=SCOPE)
        out["distance_m"] = out["board_to_bucket_distance_m"]
        out.update({phase: stats(rr if phase == "total" else [r for r in rr if r["phase"] == phase]) for phase in ("nominal", "fresh", "total")}); report["recipes"].append(out)
    report["evidence_sha256"] = {str(p.relative_to(args.output)): sha(p) for p in sorted(args.output.rglob("*")) if p.is_file() and p.name not in ("report.json", "evidence-manifest.json") and not p.name.startswith("audit")}
    atomic(args.output / "evidence-manifest.json", report["evidence_sha256"]); report["status"] = "completed"; atomic(args.output / "report.json", report)
    audit = audit_confirmation(args.output, args.study, Path(__file__).resolve().parents[2], args.assets); atomic(args.output / "audit-window-confirmation.json", audit)
    if audit["status"] != "passed": raise RuntimeError("Independent confirmation evidence audit failed")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "assets", "output", "gpu-lock"): parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(); parent = load_parent(args.study)
    from .gpu_slot import gpu_slot
    with gpu_slot(args.gpu_lock, timeout_s=7200): run(args, parent)


if __name__ == "__main__": main()
