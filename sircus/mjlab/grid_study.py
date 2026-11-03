"""Finite 35-target development search and independently frozen repeats.

The existing corrected native physics, scoring, geometry and staging contracts
are unchanged. Historical captured poses are warm starts, never new labels.
This runner owns only its output directory and the shared 4090 slot.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import numpy as np

from .corrected_contract import PHYSICS, NOISE, geometry_hint, grid_layouts, sha
from .expanded_actions import WIDE_BOUNDS
from .learning_contract import digest
from .sweep_contract import proposals, rank_row
from .sweep_worker import (
    NativePlanner,
    atomic,
    cached_result,
    exact_source,
    load_reference,
    result_for,
)

VERSION = "sircus-grid35-expanded-sweep-v1"
SEED = 20261130
PROTOCOL = dict(
    version=VERSION,
    grid=grid_layouts(),
    stages=[
        dict(
            name="baseline",
            cpu_candidates=64,
            native_budget=16,
            confirmation_candidates=3,
        ),
        dict(
            name="expanded",
            cpu_candidates=1024,
            native_budget=128,
            confirmation_candidates=6,
        ),
    ],
    action_order=["board_x_m", "board_y_m", "board_height_m", "tilt_deg", "gap_m"],
    bounds=WIDE_BOUNDS,
    qualification=dict(nominal=3, fresh=20, required_fresh=18),
    frozen_validation=dict(
        nominal=3,
        fresh=30,
        seed=SEED + 100000,
        selection="Freeze all selected settings before these trials; never select a fallback on their outcomes",
    ),
    perturbation_std=NOISE.tolist(),
    seed=SEED,
    physics=PHYSICS,
    staging="Unchanged fixed common home; failed home/placement geometry is explicit",
    scope="Development search and repeatability, not RL inference or held-out spatial generalization",
)


def utc():
    return datetime.now(timezone.utc).isoformat()


def jsonable(value):
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def native_capture(row):
    return bool(
        row.get("native_release")
        and row.get("route_success")
        and not row.get("invalid")
        and row.get("scalar_score_audit_passed")
    )


def study_identity(ref):
    source = dict(ref["source_hashes"])
    root = Path(__file__).resolve().parents[2]
    for name in ("sircus/mjlab/grid_study.py", "sircus/mjlab/grid_study_contract.py"):
        source[name] = sha(root / name)
    return dict(
        contract_fingerprint=ref["contract"]["fingerprint"],
        scene_sha256=digest(ref["scene"]),
        asset_sha256=ref["asset_hashes"],
        source_sha256=source,
        physics=PHYSICS,
    )


def historical_hints(ref):
    path = ref["root"] / "shots.jsonl"
    if sha(path) != ref["public"]["artifacts_sha256"]["shots.jsonl"]:
        raise ValueError("Historical native shot evidence changed")
    hints = []
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if (
            row.get("phase") == "search"
            and row.get("native_release")
            and row.get("route_success")
            and not row.get("invalid")
        ):
            hint = geometry_hint(row, "corrected_grid_search")
            if hint is not None:
                hints.append(hint)
    # Qualified references are first in an exact-target query, but their old
    # outcomes remain solely in the saved-reference display.
    preferred = [
        geometry_hint(item["source"], "qualified_reference_geometry")
        for item in ref["catalogue"]["sources"]
        if item.get("qualified")
    ]
    return [row for row in preferred + hints if row is not None]


def candidate_pool(
    scene, bucket, hints, *, limit, seed, expanded=False, extra=(), exclude=()
):
    """Interleave gap/H/tilt/radius/bearing refinements with broad proposals."""
    low, high = np.asarray(WIDE_BOUNDS["low"]), np.asarray(WIDE_BOUNDS["high"])
    bucket = np.asarray(bucket, float)
    rows, metadata = [], []
    seen = {tuple(float(x) for x in value) for value in exclude}
    local = [
        row
        for row in list(extra) + list(hints)
        if np.array_equal(row["bucket_xy_m"], bucket)
    ]

    def add(value, meta):
        a = np.asarray(value, float)
        if (
            len(rows) >= limit
            or a.shape != (5,)
            or not np.isfinite(a).all()
            or not ((a >= low) & (a <= high)).all()
        ):
            return
        if np.linalg.norm(a[:2] - bucket) <= 0.15 or tuple(a) in seen:
            return
        seen.add(tuple(a))
        rows.append(a)
        metadata.append(
            dict(meta, scope="Geometry proposal only; no inherited flight label")
        )

    for hint in local:
        add(
            hint["values"],
            dict(
                method="exact_historical_geometry",
                source_hint=hint,
                source_record_sha256=hint.get("source_record_sha256"),
            ),
        )
    if expanded and local:
        # Normalize geometry to choose diverse warm starts without outcome
        # labels. Previously measured near misses may also become local seeds.
        distinct = []
        for hint in local:
            a = np.asarray(hint["values"])
            if not any(
                np.linalg.norm((a - np.asarray(other["values"])) / (high - low)) < 0.025
                for other in distinct
            ):
                distinct.append(hint)
            if len(distinct) >= 8:
                break
        offsets = []
        for axis, magnitudes in (
            (4, (0.001, 0.002, 0.004, 0.008, 0.016, 0.032)),
            (2, (0.005, 0.015, 0.030)),
            (3, (0.5, 1.0, 2.0, 4.0)),
            ("radius", (0.010, 0.025, 0.050)),
            ("bearing", (3.0, 7.5, 15.0)),
        ):
            for magnitude in magnitudes:
                for sign in (-1, 1):
                    offsets.append((axis, sign * magnitude))
        for axis, delta in offsets:
            for hint in distinct:
                a = np.asarray(hint["values"], float).copy()
                if axis == "radius":
                    v = a[:2] - bucket
                    a[:2] += delta * v / np.linalg.norm(v)
                elif axis == "bearing":
                    t = math.radians(delta)
                    a[:2] = bucket + np.asarray(
                        [[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]]
                    ) @ (a[:2] - bucket)
                else:
                    a[axis] += delta
                add(
                    a,
                    dict(
                        method="local_interleaved_sweep",
                        axis=str(axis),
                        delta=delta,
                        source_hint=hint,
                        source_record_sha256=hint.get("source_record_sha256"),
                    ),
                )
        rng = np.random.default_rng(seed)
        for i in range(min(256, limit // 3)):
            hint = distinct[i % len(distinct)]
            a = np.asarray(hint["values"]) + rng.normal(
                0, [0.008, 0.008, 0.015, 1.5, 0.015]
            )
            add(
                a,
                dict(
                    method="local_joint_parameter_variation",
                    source_hint=hint,
                    source_record_sha256=hint.get("source_record_sha256"),
                ),
            )
    broad, meta = proposals(
        scene,
        bucket,
        limit=1024 if expanded else max(256, limit),
        seed=seed,
        hints=hints,
    )
    for a, m in zip(broad, meta):
        add(a, m)
    return np.asarray(rows).reshape(-1, 5), metadata


class StudyPlanner(NativePlanner):
    def start_stage(self, directory, cell_id, stage):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.packet, self.rows, self.audit_records = 0, [], []
        self.cell_id, self.stage = cell_id, stage
        self.counts = dict(
            cpu_candidates=0,
            geometry_rejected=0,
            search_requests=0,
            confirmation_requests=0,
            search_cpu_candidates=0,
            search_geometry_rejected=0,
            search_captures=0,
        )
        self.confirmations = []

    def audit(self, bucket, values, metadata=None):
        rows = super().audit(bucket, values, metadata)
        for row in rows:
            row["id"] = f"{self.cell_id}-{self.stage}-{row['id']}"
        self.audit_records.extend(copy.deepcopy(rows))
        atomic(self.directory / "geometry.json", jsonable(self.audit_records))
        return rows

    def execute(self, env, candidates, phase):
        rows = super().execute(env, candidates, phase)
        if phase == "search":
            self.counts["search_captures"] += sum(native_capture(r) for r in rows)
        return rows

    def check_options(self, env):
        o = env.sim.mj_model.opt
        actual = dict(
            timestep=float(o.timestep),
            solver=int(o.solver),
            cone=int(o.cone),
            integrator=int(o.integrator),
            iterations=int(o.iterations),
            tolerance=float(o.tolerance),
            gravity=np.asarray(o.gravity).tolist(),
        )
        if actual != self.ref["public"]["effective_native_options"]:
            raise ValueError("Effective native options changed")

    def confirm(self, env, source, *, seed, fresh_n=20, prefix="qualification"):
        source = dict(source)
        proof = self.admission.admit(source)
        record = dict(
            source_id=source["id"],
            source_values=source["values"],
            source_q10=source["q10"],
            source_q14=source["q14"],
            seed=seed,
            perturbation_std=NOISE.tolist(),
            purpose=prefix,
            motion=proof,
            nominal=[],
            fresh=[],
            qualified=False,
        )
        path = self.directory / f"{prefix}-{len(self.confirmations):03d}.json"
        self.confirmations.append(record)
        if not proof.get("passed"):
            atomic(path, jsonable(record))
            return record, [], []
        exact = dict(
            source, geometry_eligible=True, geometry_reason="endpoint_admitted"
        )
        nominal = self.execute(
            env,
            [dict(exact, id=source["id"] + f"-{prefix}-nominal-{i}") for i in range(3)],
            "nominal",
        )
        record["nominal"] = [r["id"] for r in nominal]
        record["nominal_counts"] = dict(
            captures=sum(map(native_capture, nominal)), attempts=len(nominal)
        )
        if not all(native_capture(r) for r in nominal) and prefix == "qualification":
            atomic(path, jsonable(record))
            return record, nominal, []
        jitter = np.asarray(source["values"]) + np.random.default_rng(seed).normal(
            0, NOISE, (fresh_n, 5)
        )
        hint = {key: source[key] for key in ("values", "bucket_xy_m", "q10", "q14")}
        fresh = self.audit(
            source["bucket_xy_m"], jitter, [dict(source_hint=hint) for _ in jitter]
        )
        record["variations"] = []
        for i, row in enumerate(fresh):
            row["id"] = source["id"] + f"-{prefix}-fresh-{i}"
            proof2 = (
                self.admission.connect_variation(source, row)
                if row["geometry_eligible"]
                else dict(passed=False, reason=row["geometry_reason"])
            )
            row["geometry_eligible"] &= bool(proof2.get("passed"))
            if not row["geometry_eligible"] and proof2.get("reason"):
                row["geometry_reason"] = proof2["reason"]
            record["variations"].append(
                dict(
                    id=row["id"],
                    values=row["values"],
                    q10=row["q10"],
                    q14=row["q14"],
                    geometry_reason=row["geometry_reason"],
                    admitted=row["geometry_eligible"],
                    local_path=proof2,
                )
            )
        repeated = self.execute(env, fresh, "fresh")
        record["fresh"] = [r["id"] for r in repeated]
        record["nominal_counts"] = dict(
            captures=sum(map(native_capture, nominal)), attempts=len(nominal)
        )
        record["fresh_counts"] = dict(
            captures=sum(map(native_capture, repeated)), attempts=len(repeated)
        )
        record["quality"] = dict(
            clean=sum(
                native_capture(r) and r.get("route_quality") == 2 for r in repeated
            ),
            assisted=sum(
                native_capture(r) and r.get("route_quality") == 1 for r in repeated
            ),
            cohort="fresh",
        )
        record["qualified"] = bool(
            all(map(native_capture, nominal))
            and len(repeated) == 20
            and sum(map(native_capture, repeated)) >= 18
        )
        atomic(path, jsonable(record))
        return record, nominal, repeated

    def outcome(self, request, source, confirmation=None, *, status=None):
        counts = dict(self.counts)
        if confirmation:
            if source is None or any(
                source.get(name) != confirmation.get(record_name)
                for name, record_name in (
                    ("values", "source_values"),
                    ("q10", "source_q10"),
                    ("q14", "source_q14"),
                )
            ):
                raise ValueError(
                    "Confirmation cohort belongs to another exact shot configuration"
                )
            counts.update(
                nominal_attempts=len(confirmation["nominal"]),
                nominal_captures=confirmation.get("nominal_counts", {}).get(
                    "captures", 0
                ),
                fresh_attempts=len(confirmation["fresh"]),
                fresh_captures=confirmation.get("fresh_counts", {}).get("captures", 0),
                fresh_clean_captures=confirmation.get("quality", {}).get("clean", 0),
                fresh_assisted_captures=confirmation.get("quality", {}).get(
                    "assisted", 0
                ),
            )
        qualified = bool(confirmation and confirmation["qualified"])
        status = status or (
            "verified_shot"
            if qualified
            else (
                "capture_found_confirmation_unresolved"
                if source and source.get("route_success")
                else "no_verified_solution_within_budget"
            )
        )
        result = result_for(
            self.ref,
            request,
            source,
            status=status,
            counts=counts,
            motion=confirmation["motion"] if confirmation else None,
            qualified=qualified,
            message=(
                "Fresh study qualification passed"
                if qualified
                else "No qualified setting in this finite stage; inspect geometry, native outcomes and candidate-specific repeat records"
            ),
        )
        if result.get("configuration") and confirmation:
            result["metrics"]["confirmation_configuration_sha256"] = digest(
                result["configuration"]
            )
        return result

    def search(
        self,
        env,
        request,
        values,
        metadata,
        *,
        confirmation_candidates,
        seed,
        progress=None,
    ):
        home = self.admission.check_home(request["bucket_xy"])
        atomic(self.directory / "home.json", home)
        if not home.get("passed"):
            return self.outcome(request, None, status="common_home_unresolved")
        for offset in range(0, len(values), 32):
            admitted = self.audit(
                request["bucket_xy"],
                values[offset : offset + 32],
                metadata[offset : offset + 32],
            )
            self.counts["search_cpu_candidates"] += len(admitted)
            self.counts["search_geometry_rejected"] += sum(
                not r["geometry_eligible"] for r in admitted
            )
            remaining = request["search_budget"] - self.counts["search_requests"]
            group = [r for r in admitted if r["geometry_eligible"]][:remaining]
            if group:
                self.execute(env, group, "search")
            if progress:
                progress(dict(self.counts))
            if self.counts["search_requests"] >= request["search_budget"]:
                break
        successful = sorted(
            [r for r in self.rows if r["phase"] == "search" and native_capture(r)],
            key=rank_row,
            reverse=True,
        )
        selected, cohort = None, None
        for i, source in enumerate(successful[:confirmation_candidates]):
            current, nominal, fresh = self.confirm(env, source, seed=seed + i)
            if selected is None or current.get("fresh_counts", {}).get(
                "captures", -1
            ) > cohort.get("fresh_counts", {}).get("captures", -1):
                selected, cohort = (nominal[0] if nominal else source), current
            if current["qualified"]:
                return self.outcome(request, nominal[0], current)
        if selected is not None:
            return self.outcome(request, selected, cohort)
        best = max(self.rows, key=rank_row) if self.rows else None
        return self.outcome(request, best)


def run(args):
    from .grid_study_contract import make_report, validate_report
    from .gpu_slot import gpu_slot
    from .contact_env import ContactEnvironment

    ref = load_reference(args.reference, args.assets)
    identity = study_identity(ref)
    # Results emitted by this runner include its additional frozen source code.
    ref["source_hashes"] = identity["source_sha256"]
    hints = historical_hints(ref)
    directory = args.output
    directory.mkdir(parents=True, exist_ok=True)
    contract = dict(
        identity=identity,
        protocol=PROTOCOL,
        hints_sha256=digest(hints),
        packages=ref["packages"],
    )
    contract["fingerprint"] = digest(contract)
    if (directory / "contract.json").exists() and json.loads(
        (directory / "contract.json").read_text()
    ) != contract:
        raise ValueError("Study output identity changed; use a new directory")
    atomic(directory / "contract.json", contract)
    ledger_path = directory / "ledger.json"
    ledger = (
        json.loads(ledger_path.read_text())
        if ledger_path.exists()
        else dict(started_at=utc(), attempts={}, states={}, validation={})
    )
    layouts = grid_layouts()

    def publish(status="running"):
        atomic(ledger_path, ledger)
        report = make_report(
            ledger["attempts"],
            identity=identity,
            protocol=PROTOCOL,
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

    def store(layout, stage, request, result, stage_dir):
        result["provenance"]["source_sha256"] = identity["source_sha256"]
        result["provenance"]["study_fingerprint"] = contract["fingerprint"]
        result["provenance"]["request_sha256"] = digest(
            dict(
                request=request, study_fingerprint=contract["fingerprint"], stage=stage
            )
        )
        atomic(stage_dir / "result.json", result)
        atomic(stage_dir / "request.json", request)
        manifest = {
            str(p.relative_to(stage_dir)): sha(p)
            for p in stage_dir.rglob("*")
            if p.is_file() and p.name != "manifest.json"
        }
        atomic(stage_dir / "manifest.json", manifest)
        attempt = dict(
            stage=stage,
            request=request,
            result=result,
            result_sha256=sha(stage_dir / "result.json"),
            evidence_sha256=manifest,
            result_url="/api/planner-grid/result/" + layout["id"],
        )
        metrics = result.get("metrics", {})
        if (
            not result["provenance"].get("saved_evidence")
            and "fresh_clean_captures" in metrics
        ):
            attempt["quality"] = dict(
                clean=metrics.get("fresh_clean_captures", 0),
                assisted=metrics.get("fresh_assisted_captures", 0),
                cohort="fresh",
            )
        ledger["attempts"].setdefault(layout["id"], []).append(attempt)
        selected_report = publish()
        cell = next(c for c in selected_report["cells"] if c["id"] == layout["id"])
        if cell.get("selected"):
            selected_id = cell["selected"]["request_id"]
            chosen = next(
                a["result"]
                for a in ledger["attempts"][layout["id"]]
                if a["request"]["request_id"] == selected_id
            )
            atomic(directory / "results" / (layout["id"] + ".json"), chosen)

    # Saved catalogue numbers remain historical. They are visible immediately,
    # then receive new trials rather than being counted as this study's work.
    for layout in layouts:
        if layout["id"] in ledger["attempts"]:
            continue
        request = dict(
            request_id=layout["id"] + "-saved",
            mode="search",
            bucket_xy=layout["bucket_xy_m"],
            constraints=dict(board_position=None, dropper_position=None),
            search_budget=16,
            cpu_candidates=64,
        )
        source = exact_source(ref, request)
        ledger["states"][layout["id"]] = "queued"
        if source:
            stage_dir = directory / "cells" / layout["id"] / "saved"
            stage_dir.mkdir(parents=True, exist_ok=True)
            store(
                layout,
                "saved_reference",
                request,
                cached_result(ref, request, source),
                stage_dir,
            )
    publish()
    if args.prepare_only:
        return
    with gpu_slot(args.gpu_lock, timeout_s=1800, poll_s=5):
        planner = StudyPlanner(ref, args.assets, directory / "initialization")
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

        for stage_index, stage in enumerate(PROTOCOL["stages"]):
            for index, layout in enumerate(layouts):
                old = ledger["attempts"].get(layout["id"], [])
                if any(a["stage"] == stage["name"] for a in old):
                    continue
                if stage_index and any(
                    a["result"].get("success")
                    and not a["result"]["provenance"].get("saved_evidence")
                    for a in old
                ):
                    continue
                home = planner.admission.check_home(layout["bucket_xy_m"])
                ledger["states"][layout["id"]] = "searching"
                publish()
                path = directory / "cells" / layout["id"] / stage["name"]
                planner.start_stage(path, layout["id"], stage["name"])
                request = dict(
                    request_id=layout["id"] + "-" + stage["name"],
                    mode="search",
                    bucket_xy=layout["bucket_xy_m"],
                    constraints=dict(board_position=None, dropper_position=None),
                    search_budget=stage["native_budget"],
                    cpu_candidates=stage["cpu_candidates"],
                )
                begin = time.monotonic()
                if not home.get("passed"):
                    atomic(path / "home.json", home)
                    result = planner.outcome(
                        request, None, status="common_home_unresolved"
                    )
                elif stage_index == 0 and (source := exact_source(ref, request)):
                    row = dict(
                        source["source"],
                        geometry_eligible=True,
                        geometry_reason="endpoint_admitted",
                    )
                    admitted = planner.audit(
                        layout["bucket_xy_m"], [row["values"]], [dict(source_hint=row)]
                    )[0]
                    if admitted["geometry_eligible"]:
                        cohort, nominal, fresh = planner.confirm(
                            get_env(), admitted, seed=SEED + index * 100
                        )
                        result = planner.outcome(
                            request, nominal[0] if nominal else row, cohort
                        )
                    else:
                        result = planner.outcome(request, None)
                else:
                    extra, exclude = [], []
                    if stage_index:
                        old_path = directory / "cells" / layout["id"] / "baseline"
                        if (old_path / "geometry.json").exists():
                            exclude = [
                                r["values"]
                                for r in json.loads(
                                    (old_path / "geometry.json").read_text()
                                )
                            ]
                        if (old_path / "shots.jsonl").exists():
                            previous = [
                                json.loads(line)
                                for line in (old_path / "shots.jsonl")
                                .read_text()
                                .splitlines()
                            ]
                            extra = [
                                geometry_hint(r, "new_baseline_geometry")
                                for r in sorted(previous, key=rank_row, reverse=True)[
                                    :8
                                ]
                                if r.get("native_release")
                            ]
                            extra = [r for r in extra if r is not None]
                    values, metadata = candidate_pool(
                        ref["scene"],
                        layout["bucket_xy_m"],
                        hints,
                        limit=stage["cpu_candidates"],
                        seed=SEED + index + stage_index * 1000,
                        expanded=bool(stage_index),
                        extra=extra,
                        exclude=exclude,
                    )
                    atomic(
                        path / "proposals.json",
                        dict(
                            values=values.tolist(),
                            metadata=metadata,
                            seed=SEED + index + stage_index * 1000,
                        ),
                    )
                    result = planner.search(
                        get_env(),
                        request,
                        values,
                        metadata,
                        confirmation_candidates=stage["confirmation_candidates"],
                        seed=SEED + index * 100 + stage_index * 10000,
                    )
                result["metrics"]["elapsed_s"] = time.monotonic() - begin
                ledger["states"][layout["id"]] = (
                    "verified_fresh"
                    if result.get("success")
                    else "blocked" if not home.get("passed") else "unresolved"
                )
                store(layout, stage["name"], request, result, path)

        # Freeze all choices once, then assess repeatability without reselection.
        report = publish("validating")
        selected = {
            c["id"]: c["selected"] for c in report["cells"] if c.get("selected")
        }
        atomic(directory / "frozen-selection.json", selected)
        for index, layout in enumerate(layouts):
            if layout["id"] not in selected or layout["id"] in ledger["validation"]:
                continue
            choice = selected[layout["id"]]
            attempt = next(
                a
                for a in ledger["attempts"][layout["id"]]
                if a["request"]["request_id"] == choice["request_id"]
            )
            config = attempt["result"]["configuration"]
            if digest(config) != choice["configuration_sha256"]:
                raise ValueError("Frozen selection identity changed")
            path = directory / "cells" / layout["id"] / "frozen-repeat"
            planner.start_stage(path, layout["id"], "frozen-repeat")
            source = dict(
                id=layout["id"] + "-frozen",
                bucket_xy_m=layout["bucket_xy_m"],
                values=[
                    *config["board_center"][:2],
                    config["board_center"][2],
                    config["tilt_deg"],
                    config["gap_m"],
                ],
                q10=config["q10"],
                q14=config["q14"],
            )
            # Native FK board_center can differ slightly from requested values;
            # exact original action parameters are saved by the request stage.
            original_stage = attempt["stage"]
            if original_stage == "saved_reference":
                source["values"] = exact_source(ref, attempt["request"])["source"][
                    "values"
                ]
            else:
                shots = (
                    directory / "cells" / layout["id"] / original_stage / "shots.jsonl"
                )
                first = next(
                    json.loads(line)
                    for line in shots.read_text().splitlines()
                    if json.loads(line).get("native_release")
                    and json.loads(line).get("phase") == "nominal"
                    and json.loads(line).get("q10") == config["q10"]
                    and json.loads(line).get("q14") == config["q14"]
                )
                source["values"] = first["values"]
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
    for name in ("reference", "assets", "gpu-lock", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
