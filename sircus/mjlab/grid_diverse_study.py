"""Additive, diversity-controlled follow-up to an audited 35-cell study.

Every proposed endpoint is checked before the finite native budget is selected.
Old trials remain immutable saved evidence. Only newly selected recipes receive
this follow-up's independently frozen 3+30 repeats. No policy is trained here.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np

from .corrected_contract import PHYSICS, NOISE, geometry_hint, grid_layouts, sha
from .expanded_actions import WIDE_BOUNDS
from .grid_study import (
    StudyPlanner,
    candidate_pool,
    historical_hints,
    jsonable,
    native_capture,
    study_identity as core_identity,
    utc,
)
from .grid_study_contract import make_report, validate_report
from .learning_contract import digest
from .sweep_contract import proposals, rank_row
from .sweep_worker import atomic, contained, load_reference

VERSION = "sircus-grid35-diverse-followup-v2"
PARENT_VERSION = "sircus-grid35-parent-evidence-v1"
SEED = 20261201
BROAD_METHODS = {"broad_space_filling", "inverse_ballistic_ring"}
CPU_CANDIDATES, NATIVE_BUDGET, BROAD_QUOTA = 1024, 128, 64
TARGET_SELECTION = dict(
    nominal_required=3,
    nominal_attempts=3,
    fresh_required=27,
    fresh_attempts=30,
    missing_or_blocked=True,
)
SELECTED_SHOT_RULE = dict(
    candidate_limit=6,
    ranking=[
        "qualification_fresh_captures",
        "qualification_fresh_clean_captures",
        "native_search_rank",
    ],
    native_search_rank="sweep_contract.rank_row(search_source)",
    early_stop=dict(
        nominal_captures=3,
        nominal_attempts=3,
        fresh_captures=20,
        fresh_attempts=20,
        fresh_clean_captures=20,
    ),
    frozen_validation="Never used for recipe selection or fallback",
)
SNAPSHOT_FILES = (
    "contract.json",
    "report.json",
    "frozen-selection.json",
    "manifest.json",
    "audit.json",
)


def read(path):
    return json.loads(Path(path).read_text())


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _fingerprinted(value, label):
    raw = copy.deepcopy(value)
    expected = raw.pop("fingerprint", None)
    _require(expected == digest(raw), "Changed " + label + " fingerprint")
    return expected


def study_identity(ref):
    """The native/core sources are unchanged; the additive reader is explicit."""
    identity = core_identity(ref)
    name = "sircus/mjlab/grid_diverse_study.py"
    identity["source_sha256"][name] = sha(Path(__file__).resolve())
    return identity


def load_parent_snapshot(parent_dir, expected_metadata=None, *, audit_path=None):
    """Validate compact copied parent bindings, without loading native arrays.

    The launcher separately verifies every full parent manifest file and frozen
    source before using parent outcomes. A reader may use the compact copies
    only after matching this metadata to the current follow-up contract.
    """
    parent = Path(parent_dir).resolve()
    audit_file = Path(audit_path).resolve() if audit_path else parent / "audit.json"
    contract, report, selected, manifest, audit = (
        read(parent / "contract.json"),
        read(parent / "report.json"),
        read(parent / "frozen-selection.json"),
        read(parent / "manifest.json"),
        read(audit_file),
    )
    fingerprint = _fingerprinted(contract, "parent contract")
    validate_report(report)
    _require(report["status"] == "completed", "Parent study is not completed")
    _require(
        report["identity"] == contract["identity"], "Parent report identity differs"
    )
    _require(report["protocol"] == contract["protocol"], "Parent protocol differs")
    _require(contract["identity"]["physics"] == PHYSICS, "Parent physics differs")
    _require(isinstance(manifest, dict) and manifest, "Parent manifest is empty")
    for name in ("contract.json", "report.json", "frozen-selection.json"):
        _require(
            manifest.get(name) == sha(parent / name), "Parent manifest differs: " + name
        )
    _require(
        audit.get("status") == "passed" and audit.get("failures") == [],
        "Parent audit did not pass",
    )
    _require(audit.get("checks", 0) > 0, "Parent audit has no checks")
    _require(
        audit.get("contract_fingerprint") == fingerprint,
        "Parent audit contract differs",
    )
    for name, field in (
        ("manifest.json", "study_manifest_sha256"),
        ("report.json", "report_sha256"),
        ("frozen-selection.json", "frozen_selection_sha256"),
    ):
        _require(
            audit.get(field) == sha(parent / name),
            "Parent audit evidence differs: " + name,
        )
    for field in ("source_sha256", "asset_sha256", "physics"):
        _require(
            audit.get(field) == contract["identity"][field],
            "Parent audit identity differs: " + field,
        )
    _require(
        audit.get("packages") == contract["packages"],
        "Parent audited dependencies differ",
    )
    actual_selected = {
        row["id"]: row["selected"] for row in report["cells"] if row.get("selected")
    }
    _require(selected == actual_selected, "Parent frozen selection changed")
    metadata = dict(
        version=PARENT_VERSION,
        contract_fingerprint=fingerprint,
        report_fingerprint=report["fingerprint"],
        study_manifest_sha256=sha(parent / "manifest.json"),
        report_sha256=sha(parent / "report.json"),
        frozen_selection_sha256=sha(parent / "frozen-selection.json"),
        audit_sha256=sha(audit_file),
        source_sha256=contract["identity"]["source_sha256"],
        asset_sha256=contract["identity"]["asset_sha256"],
        physics=PHYSICS,
    )
    if expected_metadata is not None:
        _require(
            metadata == expected_metadata,
            "Copied parent snapshot does not match this follow-up",
        )
    return dict(
        root=parent,
        contract=contract,
        report=report,
        frozen_selection=selected,
        manifest=manifest,
        audit=audit,
        audit_path=audit_file,
        metadata=metadata,
    )


def load_parent(study, audit_path, parent_source, ref, *, repository=None):
    """Before new work, verify all immutable parent files, sources and packages."""
    snapshot = load_parent_snapshot(study, audit_path=audit_path)
    parent, manifest = snapshot["root"], snapshot["manifest"]
    for name, expected in manifest.items():
        _require(
            sha(contained(parent, name)) == expected, "Parent evidence changed: " + name
        )
    actual_files = {
        str(p.relative_to(parent))
        for p in parent.rglob("*")
        if p.is_file() and p.name != "manifest.json"
    }
    _require(set(manifest) == actual_files, "Parent manifest does not cover all files")
    identity = snapshot["contract"]["identity"]
    root = Path(repository or Path(__file__).resolve().parents[2])
    for name, expected in identity["source_sha256"].items():
        _require(
            sha(contained(parent_source, name)) == expected,
            "Frozen parent source changed: " + name,
        )
        _require(
            sha(contained(root, name)) == expected,
            "Current core source differs from parent: " + name,
        )
    _require(
        identity["contract_fingerprint"] == ref["contract"]["fingerprint"],
        "Parent native scene contract differs",
    )
    _require(identity["scene_sha256"] == digest(ref["scene"]), "Parent scene differs")
    _require(
        identity["asset_sha256"] == ref["asset_hashes"], "Parent actual assets differ"
    )
    _require(
        snapshot["contract"]["packages"] == ref["packages"],
        "Parent actual packages differ",
    )
    for name, expected in ref["source_hashes"].items():
        _require(
            identity["source_sha256"].get(name) == expected,
            "Parent native dependency differs: " + name,
        )
    ledger = read(parent / "ledger.json")
    expected_report = make_report(
        ledger["attempts"],
        identity=identity,
        protocol=snapshot["contract"]["protocol"],
        status=snapshot["report"]["status"],
        states=ledger["states"],
        started_at=ledger["started_at"],
        updated_at=snapshot["report"]["updated_at"],
        validation_by_cell=ledger["validation"],
    )
    _require(
        expected_report == snapshot["report"],
        "Parent ledger does not reconstruct the audited report",
    )
    snapshot["ledger"] = ledger
    return snapshot


def parent_attempt(snapshot, cell):
    """Choose the parent's frozen winner, or its last unresolved attempt."""
    attempts = snapshot["ledger"]["attempts"].get(cell["id"], [])
    _require(bool(attempts), "Parent cell has no audited attempt")
    selected = cell.get("selected")
    if selected:
        chosen = [
            a for a in attempts if a["request"]["request_id"] == selected["request_id"]
        ]
        _require(len(chosen) == 1, "Parent selected attempt is not unique")
        attempt = chosen[0]
        _require(
            digest(attempt["result"]["configuration"])
            == selected["configuration_sha256"],
            "Parent selected configuration changed",
        )
    else:
        attempt = attempts[-1]
    stage = "saved" if attempt["stage"] == "saved_reference" else attempt["stage"]
    prefix = f"cells/{cell['id']}/{stage}/"
    files = {key: prefix + key + ".json" for key in ("request", "result")}
    for key, name in files.items():
        path = contained(snapshot["root"], name)
        _require(
            snapshot["manifest"].get(name) == sha(path),
            "Parent original file changed: " + name,
        )
        _require(
            read(path) == attempt[key], "Parent original payload differs from ledger"
        )
    return attempt, files


def targeted_cell_ids(report):
    """Parent final findings select new development work, not new test results."""
    targets = []
    for cell in report["cells"]:
        validation = cell.get("frozen_validation") or {}
        nominal, fresh = validation.get("nominal", {}), validation.get("fresh", {})
        if (
            not cell.get("selected")
            or validation.get("status") != "completed"
            or nominal.get("attempts") != 3
            or nominal.get("captures") != 3
            or fresh.get("attempts") != 30
            or fresh.get("captures") is None
            or fresh["captures"] < 27
        ):
            targets.append(cell["id"])
    return targets


def archived_wrapper(snapshot, cell, request, identity):
    """Keep original IDs/provenance nested; the new association is only a reader."""
    original, files = parent_attempt(snapshot, cell)
    result = copy.deepcopy(original["result"])
    old_provenance, old_metrics = copy.deepcopy(result["provenance"]), copy.deepcopy(
        result.get("metrics", {})
    )
    parent = snapshot["metadata"]
    result.update(request_id=request["request_id"], method="catalog_lookup")
    result["metrics"].update(
        cpu_candidates=0,
        geometry_rejected=0,
        search_cpu_candidates=0,
        search_geometry_rejected=0,
        search_requests=0,
        confirmation_requests=0,
        search_captures=0,
        elapsed_s=0.0,
    )
    result["message"] = (
        "Archived parent evidence; this follow-up has not rerun these qualification or frozen-validation trials"
    )
    result["provenance"].update(
        identity,
        saved_evidence=True,
        search_at_inference=False,
        policy_used=False,
        device="saved native RTX4090 parent evidence",
        parent_evidence=dict(
            parent_contract_fingerprint=parent["contract_fingerprint"],
            parent_report_fingerprint=parent["report_fingerprint"],
            parent_manifest_sha256=parent["study_manifest_sha256"],
            parent_audit_sha256=parent["audit_sha256"],
            original_request_id=original["request"]["request_id"],
            original_request_payload_sha256=digest(original["request"]),
            original_result_payload_sha256=digest(original["result"]),
            original_result_sha256=sha(contained(snapshot["root"], files["result"])),
            original_stage=original["stage"],
            original_request_file=files["request"],
            original_result_file=files["result"],
            original_provenance=old_provenance,
            original_metrics=old_metrics,
            scope="Archived parent record; current source hashes identify the reader, not a new native rollout",
        ),
    )
    return result, original, files


def parameter_summary(rows, bucket):
    """Describe actually proposed/admitted/selected requested parameters."""
    if not rows:
        return dict(
            count=0,
            family_counts={},
            ranges={},
            distance_m=None,
            release_ball_center_z_m=None,
        )
    values = np.asarray([r["values"] for r in rows], float)
    _require(
        values.shape == (len(rows), 5) and np.isfinite(values).all(),
        "Invalid diversity parameter rows",
    )
    names = ("board_x_m", "board_y_m", "board_height_m", "tilt_deg", "gap_m")
    distance = np.linalg.norm(values[:, :2] - np.asarray(bucket), axis=1)
    release = values[:, 2] + values[:, 4] + 0.0375
    return dict(
        count=len(rows),
        distinct_actions=len({tuple(v) for v in values}),
        family_counts=dict(
            Counter(r.get("proposal", {}).get("method", "unspecified") for r in rows)
        ),
        ranges={
            name: [float(values[:, i].min()), float(values[:, i].max())]
            for i, name in enumerate(names)
        },
        distance_m=[float(distance.min()), float(distance.max())],
        release_ball_center_z_m=[float(release.min()), float(release.max())],
        scope="Requested action ranges; measured native positions remain in shot records",
    )


def _farthest(values, candidates, count, *, selected=(), seed=SEED):
    """Seeded ties, normalized geometry, no flight labels or acquisition score."""
    ids = np.asarray(list(candidates), dtype=int)
    if count <= 0 or not len(ids):
        return []
    rng = np.random.default_rng(seed)
    ids = ids[rng.permutation(len(ids))]
    low, high = np.asarray(WIDE_BOUNDS["low"]), np.asarray(WIDE_BOUNDS["high"])
    normalized = (np.asarray(values) - low) / (high - low)
    picked = []
    if selected:
        nearest = np.linalg.norm(
            normalized[ids, None] - normalized[np.asarray(selected)][None], axis=2
        ).min(axis=1)
    else:
        picked.append(int(ids[0]))
        nearest = np.linalg.norm(normalized[ids] - normalized[picked[0]], axis=1)
    while len(picked) < min(count, len(ids)):
        nearest[np.isin(ids, picked)] = -1
        chosen = int(ids[int(np.argmax(nearest))])
        picked.append(chosen)
        nearest = np.minimum(
            nearest, np.linalg.norm(normalized[ids] - normalized[chosen], axis=1)
        )
    return picked


def select_diverse(rows, *, budget=NATIVE_BUDGET, broad_quota=BROAD_QUOTA, seed=SEED):
    """Reserve broad/inverse native slots after every endpoint is audited."""
    _require(
        isinstance(budget, int) and not isinstance(budget, bool) and budget > 0,
        "Positive native budget required",
    )
    _require(
        isinstance(broad_quota, int)
        and not isinstance(broad_quota, bool)
        and 0 <= broad_quota <= budget,
        "Invalid broad quota",
    )
    values = np.asarray([r["values"] for r in rows], float).reshape(-1, 5)
    eligible = [i for i, r in enumerate(rows) if r.get("geometry_eligible") is True]
    broad = [
        i
        for i in eligible
        if rows[i].get("proposal", {}).get("method") in BROAD_METHODS
    ]
    local = [i for i in eligible if i not in set(broad)]
    chosen = _farthest(values, broad, min(broad_quota, len(broad)), seed=seed)
    chosen += _farthest(
        values,
        local,
        min(budget - len(chosen), len(local)),
        selected=chosen,
        seed=seed + 1,
    )
    if len(chosen) < min(budget, len(eligible)):
        rest = [i for i in eligible if i not in set(chosen)]
        chosen += _farthest(
            values, rest, budget - len(chosen), selected=chosen, seed=seed + 2
        )
    _require(
        len(chosen) == len(set(chosen)) and all(i in eligible for i in chosen),
        "Native diversity selection is invalid",
    )
    return chosen


def qualification_rank(source, record):
    """Rank only development qualification evidence, before frozen validation."""
    return (
        record.get("fresh_counts", {}).get("captures", -1),
        record.get("quality", {}).get("clean", -1),
        rank_row(source),
    )


def perfect_qualification(record):
    return bool(
        record.get("qualified")
        and record.get("nominal_counts") == dict(captures=3, attempts=3)
        and record.get("fresh_counts") == dict(captures=20, attempts=20)
        and record.get("quality", {}).get("clean") == 20
    )


def diverse_pool(
    scene, bucket, hints, *, seed, extra=(), exclude=(), limit=CPU_CANDIDATES
):
    """Allocate half the proposed endpoints to broad/inverse full-range samples."""
    _require(2 <= limit <= CPU_CANDIDATES, "Diversity pool limit must be2..1024")
    low, high = np.asarray(WIDE_BOUNDS["low"]), np.asarray(WIDE_BOUNDS["high"])
    seen = {tuple(float(v) for v in row) for row in exclude}
    local, broad = [], []

    def add(target, value, meta):
        a = np.asarray(value, float)
        if (
            a.shape != (5,)
            or not np.isfinite(a).all()
            or not ((a >= low) & (a <= high)).all()
        ):
            return
        if np.linalg.norm(a[:2] - bucket) <= 0.15 or tuple(a) in seen:
            return
        seen.add(tuple(a))
        target.append(
            (a, dict(meta, scope="Proposal only; no inherited capture or admission"))
        )

    anchors = [
        h for h in list(extra) + list(hints) if np.array_equal(h["bucket_xy_m"], bucket)
    ]
    diverse_anchors = []
    for anchor in anchors:
        a = np.asarray(anchor["values"])
        if not any(
            np.linalg.norm((a - np.asarray(b["values"])) / (high - low)) < 0.05
            for b in diverse_anchors
        ):
            diverse_anchors.append(anchor)
        if len(diverse_anchors) == 4:
            break
    # These are deliberately joint energy/angle changes, rather than tiny gap
    # offsets around a table-assisted shot. Native physics supplies all labels.
    for tilt in (12.0, 20.0, 28.0, 36.0, 44.0, 52.0, 60.0):
        for gap in (0.06, 0.12, 0.20, 0.30, 0.42, 0.54, 0.64):
            for anchor in diverse_anchors:
                a = np.asarray(anchor["values"], float).copy()
                a[3:] = tilt, gap
                add(
                    local,
                    a,
                    dict(
                        method="joint_gap_tilt_sweep",
                        source_hint=anchor,
                        source_record_sha256=anchor.get("source_record_sha256"),
                    ),
                )
    local_values, local_meta = candidate_pool(
        scene,
        bucket,
        hints,
        limit=1024,
        seed=seed,
        expanded=True,
        extra=extra,
        exclude=exclude,
    )
    for value, meta in zip(local_values, local_meta):
        add(local, value, meta)
    broad_values, broad_meta = proposals(
        scene, bucket, limit=1024, seed=seed + 7919, hints=()
    )
    for value, meta in zip(broad_values, broad_meta):
        if meta.get("method") in BROAD_METHODS:
            add(broad, value, meta)
    broad_count = min(limit // 2, len(broad))
    local_count = min(limit - broad_count, len(local))
    chosen = local[:local_count] + broad[:broad_count]
    if len(chosen) < limit:
        chosen += (broad[broad_count:] + local[local_count:])[: limit - len(chosen)]
    _require(
        len(chosen) == limit, "Could not form the full distinct bounded proposal pool"
    )
    # Order cannot consume the native budget: every endpoint is audited first.
    return np.asarray([v for v, _ in chosen]), [m for _, m in chosen]


class DiversePlanner(StudyPlanner):
    def search(
        self,
        env,
        request,
        values,
        metadata,
        *,
        confirmation_candidates=6,
        seed=SEED,
        progress=None,
    ):
        home = self.admission.check_home(request["bucket_xy"])
        atomic(self.directory / "home.json", home)
        if not home.get("passed"):
            return self.outcome(request, None, status="common_home_unresolved")
        checked = []
        for offset in range(0, len(values), 32):
            self.storage_check()
            admitted = self.audit(
                request["bucket_xy"],
                values[offset : offset + 32],
                metadata[offset : offset + 32],
            )
            checked.extend(admitted)
            self.counts["search_cpu_candidates"] += len(admitted)
            self.counts["search_geometry_rejected"] += sum(
                not r["geometry_eligible"] for r in admitted
            )
            if progress:
                progress(dict(self.counts))
        indices = select_diverse(
            checked, budget=request["search_budget"], broad_quota=BROAD_QUOTA, seed=seed
        )
        chosen = [checked[i] for i in indices]
        broad_available = sum(
            r["geometry_eligible"]
            and r.get("proposal", {}).get("method") in BROAD_METHODS
            for r in checked
        )
        broad_selected = sum(
            r.get("proposal", {}).get("method") in BROAD_METHODS for r in chosen
        )
        selection = dict(
            method="family_quota_and_normalized_farthest_point",
            seed=seed,
            planned_cpu_candidates=len(values),
            checked_cpu_candidates=len(checked),
            all_proposals_audited=len(values) == len(checked),
            native_budget=request["search_budget"],
            requested_broad_quota=BROAD_QUOTA,
            available_broad=broad_available,
            selected_broad=broad_selected,
            quota_satisfied=broad_selected >= min(BROAD_QUOTA, broad_available),
            selected_indices=indices,
            selected_ids=[r["id"] for r in chosen],
            selection_sha256=digest(chosen),
            planned=parameter_summary(
                [dict(values=v.tolist(), proposal=m) for v, m in zip(values, metadata)],
                request["bucket_xy"],
            ),
            checked=parameter_summary(checked, request["bucket_xy"]),
            eligible=parameter_summary(
                [r for r in checked if r["geometry_eligible"]], request["bucket_xy"]
            ),
            selected=parameter_summary(chosen, request["bucket_xy"]),
        )
        atomic(self.directory / "diversity.json", jsonable(selection))
        if chosen:
            self.execute(env, chosen, "search")
        successful = sorted(
            [r for r in self.rows if r["phase"] == "search" and native_capture(r)],
            key=rank_row,
            reverse=True,
        )
        selected, cohort, qualified = None, None, []
        for i, source in enumerate(successful[:confirmation_candidates]):
            current, nominal, fresh = self.confirm(env, source, seed=seed + 10000 + i)
            if selected is None or current.get("fresh_counts", {}).get(
                "captures", -1
            ) > cohort.get("fresh_counts", {}).get("captures", -1):
                selected, cohort = nominal[0] if nominal else source, current
            if current["qualified"]:
                qualified.append(
                    (qualification_rank(source, current), nominal[0], current)
                )
                # The declared objectives cannot improve beyond all20clean
                # captures. Frozen validation remains entirely untouched.
                if perfect_qualification(current):
                    break
        if qualified:
            _, selected, cohort = max(qualified, key=lambda item: item[0])
            return self.outcome(request, selected, cohort)
        if selected is not None:
            return self.outcome(request, selected, cohort)
        best = max(self.rows, key=rank_row) if self.rows else None
        return self.outcome(request, best)


def parent_geometry(snapshot, cell_id):
    """Parent rows supply geometry seeds; outcomes are not copied as labels."""
    hints, excluded, identities = [], [], {}
    for name, expected in snapshot["manifest"].items():
        if (
            name.startswith("cells/" + cell_id + "/")
            and name.endswith("/shots.jsonl")
            and "/frozen-repeat/" not in name
        ):
            path = contained(snapshot["root"], name)
            _require(sha(path) == expected, "Parent warm-start rows changed")
            identities[name] = expected
            rows = [
                json.loads(line)
                for line in path.read_text().splitlines()
                if line.strip()
            ]
            native = [
                r
                for r in rows
                if r.get("phase") == "search" and r.get("native_release")
            ]
            excluded.extend(r["values"] for r in native)
            for row in sorted(native, key=rank_row, reverse=True):
                hint = geometry_hint(row, "audited_parent_geometry_only")
                if hint is not None:
                    hints.append(hint)
    return hints, excluded, identities


def protocol(parent, targets):
    return dict(
        version=VERSION,
        inherited_baseline=True,
        frozen_scope="new_selected_only",
        parent_contract_fingerprint=parent["contract_fingerprint"],
        parent_manifest_sha256=parent["study_manifest_sha256"],
        parent_audit_sha256=parent["audit_sha256"],
        targeted_cell_ids=targets,
        grid=grid_layouts(),
        target_selection=TARGET_SELECTION,
        stages=[
            dict(
                name="baseline", scope="Archived audited parent evidence; no new work"
            ),
            dict(
                name="expanded",
                cpu_candidates=CPU_CANDIDATES,
                native_budget=NATIVE_BUDGET,
                broad_native_quota=BROAD_QUOTA,
                confirmation_candidates=6,
            ),
        ],
        selection_method="family_quota_and_normalized_farthest_point",
        selected_shot_rule=SELECTED_SHOT_RULE,
        action_order=["board_x_m", "board_y_m", "board_height_m", "tilt_deg", "gap_m"],
        bounds=WIDE_BOUNDS,
        qualification=dict(nominal=3, fresh=20, required_fresh=18),
        frozen_validation=dict(
            nominal=3,
            fresh=30,
            seed=SEED + 100000,
            selection="Freeze all choices before these trials; no fallback or reselection from frozen outcomes",
        ),
        seed=SEED,
        perturbation_std=NOISE.tolist(),
        physics=PHYSICS,
        staging="Unchanged fixed common home; failed home/placement geometry is explicit",
        scope="Additive searched development settings, not RL or unseen spatial generalization",
    )


def run(args):
    from .gpu_slot import gpu_slot

    ref = load_reference(args.reference, args.assets)
    parent = load_parent(args.parent_study, args.parent_audit, args.parent_source, ref)
    identity = study_identity(ref)
    ref["source_hashes"] = identity["source_sha256"]
    parent_cells = {c["id"]: c for c in parent["report"]["cells"]}
    targets = targeted_cell_ids(parent["report"])
    hints = historical_hints(ref)
    parent_rows = {cell_id: parent_geometry(parent, cell_id) for cell_id in targets}
    parent_hint_manifest = {
        name: expected
        for _, _, files in parent_rows.values()
        for name, expected in files.items()
    }
    contract = dict(
        identity=identity,
        protocol=protocol(parent["metadata"], targets),
        hints_sha256=digest(hints),
        parent_hints_sha256=digest({k: v[0] for k, v in parent_rows.items()}),
        parent_row_manifest=parent_hint_manifest,
        packages=ref["packages"],
        parent=parent["metadata"],
    )
    contract["fingerprint"] = digest(contract)
    directory = args.output
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "contract.json").exists():
        _require(
            read(directory / "contract.json") == contract,
            "Follow-up output identity changed; use a new directory",
        )
    if (directory / "manifest.json").exists():
        _require(
            read(directory / "report.json")["status"] == "completed",
            "Unexpected immutable output manifest",
        )
        print(
            json.dumps(dict(status="already_completed", output=str(directory))),
            flush=True,
        )
        return
    atomic(directory / "contract.json", contract)
    atomic(directory / "parent-hints.json", {k: v[0] for k, v in parent_rows.items()})
    atomic(directory / "parent-row-manifest.json", parent_hint_manifest)
    copies = directory / "parent"
    copies.mkdir(exist_ok=True)
    for name in SNAPSHOT_FILES:
        original = (
            parent["audit_path"] if name == "audit.json" else parent["root"] / name
        )
        target = copies / name
        if target.exists():
            _require(sha(target) == sha(original), "Archived parent bytes changed")
        else:
            shutil.copyfile(original, target)
    load_parent_snapshot(copies, contract["parent"])
    ledger_path = directory / "ledger.json"
    ledger = (
        read(ledger_path)
        if ledger_path.exists()
        else dict(started_at=utc(), attempts={}, states={}, validation={})
    )

    def publish(status="running"):
        atomic(ledger_path, ledger)
        report = make_report(
            ledger["attempts"],
            identity=identity,
            protocol=contract["protocol"],
            status=status,
            states=ledger["states"],
            started_at=ledger["started_at"],
            updated_at=utc(),
            validation_by_cell=ledger["validation"],
        )
        validate_report(report)
        atomic(directory / "report.json", report)
        print(
            json.dumps(
                dict(
                    status=status,
                    summary=report["summary"],
                    updated_at=report["updated_at"],
                )
            ),
            flush=True,
        )
        return report

    def store(layout, stage, request, result, path):
        result["provenance"].update(
            source_sha256=identity["source_sha256"],
            study_fingerprint=contract["fingerprint"],
            request_sha256=digest(
                dict(
                    request=request,
                    study_fingerprint=contract["fingerprint"],
                    stage=stage,
                )
            ),
        )
        atomic(path / "request.json", request)
        atomic(path / "result.json", result)
        manifest = {
            str(p.relative_to(path)): sha(p)
            for p in path.rglob("*")
            if p.is_file() and p.name != "manifest.json"
        }
        atomic(path / "manifest.json", manifest)
        attempt = dict(
            stage=stage,
            request=request,
            result=result,
            result_sha256=sha(path / "result.json"),
            evidence_sha256=manifest,
            result_url="/api/planner-grid/result/" + layout["id"],
        )
        metrics = result.get("metrics", {})
        if (
            not result["provenance"].get("saved_evidence")
            and "fresh_clean_captures" in metrics
        ):
            attempt["quality"] = dict(
                clean=metrics["fresh_clean_captures"],
                assisted=metrics["fresh_assisted_captures"],
                cohort="fresh",
            )
        ledger["attempts"].setdefault(layout["id"], []).append(attempt)
        report = publish()
        cell = next(c for c in report["cells"] if c["id"] == layout["id"])
        if cell.get("selected"):
            chosen = next(
                a["result"]
                for a in ledger["attempts"][layout["id"]]
                if a["request"]["request_id"] == cell["selected"]["request_id"]
            )
            atomic(directory / "results" / (layout["id"] + ".json"), chosen)

    layouts = grid_layouts()
    for layout in layouts:
        if ledger["attempts"].get(layout["id"]):
            continue
        path = directory / "cells" / layout["id"] / "baseline"
        path.mkdir(parents=True, exist_ok=True)
        request = dict(
            request_id=layout["id"] + "-archived-parent",
            mode="search",
            bucket_xy=layout["bucket_xy_m"],
            constraints=dict(board_position=None, dropper_position=None),
            search_budget=16,
            cpu_candidates=64,
        )
        result, original, files = archived_wrapper(
            parent, parent_cells[layout["id"]], request, identity
        )
        for key, name in files.items():
            shutil.copyfile(
                contained(parent["root"], name), path / ("original-" + key + ".json")
            )
        ledger["states"][layout["id"]] = (
            "saved_verified" if result.get("success") else "queued"
        )
        store(layout, "baseline", request, result, path)
    publish()
    if args.prepare_only:
        return
    from .contact_env import ContactEnvironment

    with gpu_slot(args.gpu_lock, timeout_s=1800, poll_s=5):
        planner = DiversePlanner(ref, args.assets, directory / "initialization")
        env = None

        def get_env():
            nonlocal env
            if env is None:
                env = ContactEnvironment(
                    ref["scene"],
                    args.assets,
                    n=32,
                    bounds=WIDE_BOUNDS,
                    substeps=64,
                    board_mode="articulated",
                    gravity_mode="passive_only",
                )
                planner.check_options(env)
            return env

        for index, layout in enumerate(layouts):
            if layout["id"] not in targets or any(
                a["stage"] == "expanded" for a in ledger["attempts"][layout["id"]]
            ):
                continue
            path = directory / "cells" / layout["id"] / "expanded"
            _require(
                not path.exists(),
                "Incomplete stage evidence was preserved; use a new output directory to retry",
            )
            planner.start_stage(path, layout["id"], "expanded")
            ledger["states"][layout["id"]] = "searching"
            publish()
            request = dict(
                request_id=layout["id"] + "-diverse-expanded",
                mode="search",
                bucket_xy=layout["bucket_xy_m"],
                constraints=dict(board_position=None, dropper_position=None),
                search_budget=NATIVE_BUDGET,
                cpu_candidates=CPU_CANDIDATES,
            )
            begin = time.monotonic()
            home = planner.admission.check_home(layout["bucket_xy_m"])
            if not home.get("passed"):
                atomic(path / "home.json", home)
                result = planner.outcome(request, None, status="common_home_unresolved")
            else:
                extra, exclude, source_files = parent_rows[layout["id"]]
                values, metadata = diverse_pool(
                    ref["scene"],
                    layout["bucket_xy_m"],
                    hints,
                    extra=extra,
                    exclude=exclude,
                    seed=SEED + index,
                )
                atomic(
                    path / "proposals.json",
                    dict(
                        values=values.tolist(),
                        metadata=metadata,
                        seed=SEED + index,
                        parent_row_manifest=source_files,
                        planned=parameter_summary(
                            [
                                dict(values=v.tolist(), proposal=m)
                                for v, m in zip(values, metadata)
                            ],
                            layout["bucket_xy_m"],
                        ),
                    ),
                )
                result = planner.search(
                    get_env(),
                    request,
                    values,
                    metadata,
                    confirmation_candidates=6,
                    seed=SEED + index * 100,
                )
            result["metrics"]["elapsed_s"] = time.monotonic() - begin
            ledger["states"][layout["id"]] = (
                "verified_fresh"
                if result.get("success")
                else "blocked" if not home.get("passed") else "unresolved"
            )
            store(layout, "expanded", request, result, path)
        report = publish("validating")
        selected = {
            c["id"]: c["selected"] for c in report["cells"] if c.get("selected")
        }
        frozen = directory / "frozen-selection.json"
        if frozen.exists():
            _require(
                read(frozen) == selected,
                "Frozen choices changed after validation started",
            )
        else:
            atomic(frozen, selected)
        for index, layout in enumerate(layouts):
            choice = selected.get(layout["id"])
            if (
                not choice
                or choice["source"] != "fresh_study"
                or layout["id"] in ledger["validation"]
            ):
                continue
            attempt = next(
                a
                for a in ledger["attempts"][layout["id"]]
                if a["request"]["request_id"] == choice["request_id"]
            )
            config = attempt["result"]["configuration"]
            _require(
                digest(config) == choice["configuration_sha256"],
                "Frozen recipe changed",
            )
            native_rows = [
                json.loads(line)
                for line in (
                    directory / "cells" / layout["id"] / "expanded" / "shots.jsonl"
                )
                .read_text()
                .splitlines()
            ]
            original = next(
                r
                for r in native_rows
                if r.get("phase") == "nominal"
                and r.get("q10") == config["q10"]
                and r.get("q14") == config["q14"]
            )
            source = dict(
                id=layout["id"] + "-frozen",
                bucket_xy_m=layout["bucket_xy_m"],
                values=original["values"],
                q10=config["q10"],
                q14=config["q14"],
            )
            path = directory / "cells" / layout["id"] / "frozen-repeat"
            _require(
                not path.exists(),
                "Incomplete frozen cohort preserved; use a new output directory",
            )
            planner.start_stage(path, layout["id"], "frozen-repeat")
            cohort, nominal, fresh = planner.confirm(
                get_env(),
                source,
                seed=SEED + 100000 + index,
                fresh_n=30,
                prefix="frozen",
            )
            validation = dict(
                status=(
                    "completed" if len(nominal) == 3 and len(fresh) == 30 else "blocked"
                ),
                configuration_sha256=digest(config),
                seed=SEED + 100000 + index,
                nominal=cohort.get(
                    "nominal_counts", dict(captures=None, attempts=None)
                ),
                fresh=cohort.get("fresh_counts", dict(captures=None, attempts=None)),
            )
            if "quality" in cohort:
                validation["quality"] = cohort["quality"]
            atomic(path / "validation.json", validation)
            atomic(
                path / "manifest.json",
                {
                    str(p.relative_to(path)): sha(p)
                    for p in path.rglob("*")
                    if p.is_file() and p.name != "manifest.json"
                },
            )
            ledger["validation"][layout["id"]] = validation
            publish("validating")
        publish("completed")
        atomic(
            directory / "manifest.json",
            {
                str(p.relative_to(directory)): sha(p)
                for p in directory.rglob("*")
                if p.is_file() and p.name != "manifest.json"
            },
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "reference",
        "assets",
        "gpu-lock",
        "output",
        "parent-study",
        "parent-audit",
        "parent-source",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
