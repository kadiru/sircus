"""Bounded, corrected-contact shot search; no policy/checkpoint dependency.

Exact published recipes can replay hash-bound saved evidence without a GPU.
Every new bucket position needs fresh strict geometry and native confirmation.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import importlib.metadata
import json
from pathlib import Path
import shutil
import time
import uuid
import signal

import numpy as np

from .corrected_contract import NOISE, PHYSICS, sha, validate_packages
from .expanded_actions import WIDE_BOUNDS, project
from .learning_contract import constraints, digest

VERSION = "sircus-corrected-sweep-planner-v1"
CRITICAL = (
    "sircus/mjlab/contact_env.py",
    "sircus/mjlab/contact_model.py",
    "sircus/mjlab/model.py",
    "sircus/mjlab/contract.py",
    "sircus/mjlab/learning_score.py",
    "sircus/mjlab/corrected_geometry.py",
    "sircus/mjlab/stable_geometry.py",
    "sircus/mjlab/stable_ik.py",
    "sircus/mjlab/expanded_actions.py",
    "sircus/mjlab/bank_geometry.py",
    "sircus/mjlab/bank_study.py",
    "sircus/mjlab/transit_audit.py",
    "sircus/mjlab/admission.py",
    "sircus/mjlab/learning_env.py",
    "sircus/mjlab/learning_ik.py",
    "sircus/mjlab/learning_reach.py",
    "sircus/mjlab/learning_contract.py",
    "sircus/mjlab/run.py",
    "sircus/mjlab/corrected_contract.py",
    "sircus/mjlab/corrected_admission.py",
    "sircus/mjlab/gpu_slot.py",
    "sircus/mjlab/expand_geometry.py",
    "sircus/isaac/geometry.py",
    "sircus/isaac/rl_contract.py",
    "sircus/isaac/entry_quality.py",
    "sircus/isaac/board_route.py",
)
MIN_FREE = 512 * 1024**2
MAX_CACHE = 256 * 1024**2


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, allow_nan=False, indent=2) + "\n")
    temp.replace(path)


def contained(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root):
        raise ValueError("Evidence path must remain inside its declared directory")
    return path


def load_reference(reference, assets, *, repository=None):
    """Bind the native reference to committed evidence, actual assets and code."""
    repository = Path(repository or Path(__file__).resolve().parents[2])
    reference, assets = Path(reference), Path(assets)
    public_path = (
        repository / "baselines/mjlab-corrected-grid-20261102/native-results.json"
    )
    public = json.loads(public_path.read_text())
    if public.get("status") != "completed" or public.get("physics") != PHYSICS:
        raise ValueError(
            "Published corrected native reference is incomplete or incompatible"
        )
    for name in ("report.json", "contract.json", "confirmed-catalogue.json"):
        if sha(reference / name) != public["artifacts_sha256"][name]:
            raise ValueError(
                "Native reference differs from published evidence: " + name
            )
    report = json.loads((reference / "report.json").read_text())
    spec = json.loads((reference / "contract.json").read_text())
    unsigned = dict(spec)
    fingerprint = unsigned.pop("fingerprint")
    if digest(unsigned) != fingerprint or fingerprint != public["contract_fingerprint"]:
        raise ValueError("Reference scene/contract identity changed")
    if report.get("status") != "completed" or spec["physics"] != PHYSICS:
        raise ValueError("Corrected native study must be completed")
    source = {}
    for name in CRITICAL:
        actual = sha(repository / name)
        if spec["source_sha256"].get(name) != actual:
            raise ValueError("Corrected native source differs: " + name)
        source[name] = actual
    for name in ("sircus/mjlab/sweep_worker.py", "sircus/mjlab/sweep_contract.py"):
        source[name] = sha(repository / name)
    for name, expected in spec["asset_sha256"].items():
        if sha(contained(assets.parent, name)) != expected:
            raise ValueError("Native asset changed: " + name)
    packages = validate_packages(
        {
            name: importlib.metadata.version(name)
            for name in (
                "mjlab",
                "mujoco",
                "mujoco-warp",
                "torch",
                "warp-lang",
                "numpy",
                "ml-dtypes",
            )
        }
    )
    catalogue = json.loads((reference / "confirmed-catalogue.json").read_text())
    cat = dict(catalogue)
    cat_hash = cat.pop("fingerprint")
    if (
        digest(cat) != cat_hash
        or catalogue["physics"] != PHYSICS
        or catalogue["contract_fingerprint"] != fingerprint
    ):
        raise ValueError(
            "Confirmed catalogue belongs to a different scene/contact model"
        )
    return dict(
        root=reference,
        scene=spec["scene"],
        contract=spec,
        report=report,
        public=public,
        catalogue=catalogue,
        source_hashes=source,
        asset_hashes=spec["asset_sha256"],
        packages=packages,
    )


def evidence_file(ref, name):
    path = contained(ref["root"], name)
    expected = ref["report"]["evidence_sha256"].get(name)
    if not expected or sha(path) != expected:
        raise ValueError("Cached trajectory/path evidence changed: " + name)
    return path


def read_query_cache(directory, key, source_hashes):
    """Never reuse an unbound or accidentally modified result/trace bundle."""
    directory = Path(directory)
    if not (directory / "result.json").is_file():
        return None
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, expected in manifest.items():
        if sha(contained(directory, name)) != expected:
            raise ValueError("Planner cached evidence changed: " + name)
    if "result.json" not in manifest:
        raise ValueError("Cache manifest omits its result")
    result = json.loads((directory / "result.json").read_text())
    identity = result.get("provenance", {})
    if (
        identity.get("request_sha256") != key
        or identity.get("source_sha256") != source_hashes
    ):
        raise ValueError("Planner cache identity mismatch")
    return result if result.get("success") else None


def exact_source(ref, request):
    if any(value is not None for value in request["constraints"].values()):
        return None
    # Exact numerical identity, never rounded-distance or nearby-position lookup.
    rows = [
        item
        for item in ref["catalogue"]["sources"]
        if item.get("qualified") is True
        and item["source"]["bucket_xy_m"] == request["bucket_xy"]
    ]
    if not rows:
        return None
    return max(
        rows,
        key=lambda item: (
            item["fresh_captures"],
            item["source"].get("route_quality", 0),
            item["source"]["id"],
        ),
    )


def cached_result(ref, request, item):
    source = copy.deepcopy(item["source"])
    cache = evidence_file(ref, source["native_cache_file"])
    with np.load(cache, allow_pickle=False) as arrays:
        index = source["native_cache_row"]
        if str(arrays["record_ids"][index]) != source["id"]:
            raise ValueError("Cached trajectory row belongs to another shot")
        positions = arrays["position_m"][:, index].copy()
    motion = json.loads(evidence_file(ref, item["transit_file"]).read_text())
    if not motion.get("passed"):
        raise ValueError("Qualified catalogue path was not admitted")
    source["trajectory_m"] = positions[3::4].tolist()
    source["trajectory_dt_s"] = 1 / 60
    return result_for(
        ref,
        request,
        source,
        "catalog_lookup",
        "verified_shot",
        "Exact saved simulated shot: confirmed catalogue evidence; no new trial was run",
        counts=dict(
            cpu_candidates=0,
            geometry_rejected=0,
            search_requests=0,
            confirmation_requests=0,
            nominal_captures=item["nominal_captures"],
            nominal_attempts=item["nominal_attempts"],
            fresh_captures=item["fresh_captures"],
            fresh_attempts=item["fresh_attempts"],
        ),
        motion=motion,
        qualified=True,
        saved_evidence=True,
    )


def result_for(
    ref,
    request,
    row=None,
    method="sweep_planner",
    status="no_verified_solution_within_budget",
    message="No verified shot found within this finite search budget",
    *,
    counts=None,
    motion=None,
    qualified=False,
    saved_evidence=False,
):
    row = row or {}
    success = bool(
        status == "verified_shot"
        and qualified
        and row.get("route_success")
        and not row.get("invalid")
        and row.get("scalar_score_audit_passed")
        and (row.get("native_release") or saved_evidence)
        and motion
        and motion.get("passed")
    )
    config = None
    if row.get("native_release") or row.get("scalar_score_audit_passed"):
        config = dict(
            board_center=row["board_center_m"],
            board_normal=row["board_normal"],
            board_quaternion_wxyz=row["board_quaternion_wxyz"],
            release_center=row["release_ball_center_m"],
            dropper_center=row["release_ball_center_m"],
            q10=row["q10"],
            q14=row["q14"],
            link_positions10=row["link_positions10"],
            link_positions14=row["link_positions14"],
            tilt_deg=row["values"][3],
            gap_m=row["values"][4],
        )
    trajectory = [
        dict(time=(i + 1) * row.get("trajectory_dt_s", 1 / 60), ball=point)
        for i, point in enumerate(row.get("trajectory_m", []))
    ]
    return dict(
        request_id=request["request_id"],
        bucket_xy_m=request["bucket_xy"],
        method=method,
        status_native=status,
        success=success,
        message=message,
        configuration=config,
        trajectory=trajectory,
        constraints=request["constraints"],
        motion=motion,
        qualified=qualified,
        metrics=dict(
            **(counts or {}),
            board_bounces=row.get("board_contact_episodes", 0),
            retention_s=row.get("max_retention_s", 0),
            quality=row.get("route_quality", 0),
            invalid=row.get("invalid"),
            external_contacts=row.get("external_contact_steps", 0),
            scalar_score_audit_passed=bool(row.get("scalar_score_audit_passed")),
        ),
        provenance=dict(
            version=VERSION,
            contract_fingerprint=ref["contract"]["fingerprint"],
            scene_sha256=digest(ref["scene"]),
            asset_sha256=ref["asset_hashes"],
            source_sha256=ref["source_hashes"],
            physics=PHYSICS,
            engine="mjlab / native MuJoCo Warp",
            device="RTX4090" if not saved_evidence else "saved native RTX4090 evidence",
            search_at_inference=method == "sweep_planner",
            saved_evidence=saved_evidence,
            endpoint_configuration_only=True,
            policy_used=False,
            scope="Prescribed held endpoints; no learned grasp/transit or sustained impact-compliance dynamics. Motion proof is sampled geometry only.",
        ),
    )


class NativePlanner:
    """A worker owns its CPU models and acquires the shared GPU for new shots."""

    def __init__(self, ref, assets, directory, native_batch=32):
        from .corrected_geometry import CorrectedGeometry
        from .corrected_admission import TransitAdmission

        self.ref, self.assets, self.directory = ref, Path(assets), Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.geometry = CorrectedGeometry(ref["scene"], assets)
        self.admission = TransitAdmission(
            ref["scene"], assets, home=ref["contract"]["common_home"]
        )
        self.batch = native_batch
        self.packet = 0
        self.rows = []
        self.counts = dict(
            cpu_candidates=0,
            geometry_rejected=0,
            search_requests=0,
            confirmation_requests=0,
        )

    def storage_check(self):
        free = shutil.disk_usage(self.directory).free
        used = sum(p.stat().st_size for p in self.directory.rglob("*") if p.is_file())
        if free < MIN_FREE or used > MAX_CACHE:
            raise RuntimeError(
                "Planner storage budget reached; no evidence or other files were deleted"
            )

    def audit(self, bucket, values, metadata=None):
        from .corrected_geometry import REASONS

        metadata = metadata or [{} for _ in values]
        n = len(values)
        q10, q14, known_values, known_bucket = (
            np.full((n, size), np.nan) for size in (6, 6, 5, 2)
        )
        for i, meta in enumerate(metadata):
            hint = meta.get("source_hint")
            if hint and all(
                key in hint for key in ("q10", "q14", "values", "bucket_xy_m")
            ):
                for target, key in (
                    (q10, "q10"),
                    (q14, "q14"),
                    (known_values, "values"),
                    (known_bucket, "bucket_xy_m"),
                ):
                    target[i] = hint[key]
        xy = np.repeat(np.asarray([bucket], float), n, axis=0)
        mask, detail = self.geometry.audit_values(
            xy,
            np.asarray(values, float)[:, None],
            bounds=WIDE_BOUNDS,
            known_q10=q10,
            known_q14=q14,
            known_values=known_values,
            known_bucket_xy=known_bucket,
            chunk_size=32,
        )
        self.counts["cpu_candidates"] += n
        self.counts["geometry_rejected"] += n - int(mask.sum())
        rows = []
        for i, value in enumerate(values):
            rows.append(
                dict(
                    id=f"candidate-{self.counts['cpu_candidates']-n+i:05d}",
                    bucket_xy_m=list(bucket),
                    values=np.asarray(value).tolist(),
                    q10=detail["q10"][i, 0].tolist(),
                    q14=detail["q14"][i, 0].tolist(),
                    geometry_eligible=bool(mask[i, 0]),
                    geometry_reason=REASONS[int(detail["reasons"][i, 0])],
                    proposal=metadata[i],
                )
            )
        return rows

    def execute(self, env, candidates, phase):
        from .bank_study import run_bank_rows
        from .contract import ScoreStream, FORCE_NAMES

        results = []
        for start in range(0, len(candidates), self.batch):
            self.storage_check()
            group = candidates[start : start + self.batch]
            eligible = np.asarray([r["geometry_eligible"] for r in group])
            rows = run_bank_rows(
                env,
                np.asarray([r["bucket_xy_m"] for r in group]),
                np.asarray([r["values"] for r in group]),
                np.asarray([r["q10"] for r in group]),
                np.asarray([r["q14"] for r in group]),
                eligible,
                record=True,
            )
            indices = np.flatnonzero(eligible)
            filename = f"native-{self.packet:04d}.npz"
            self.packet += 1
            if len(indices):
                arrays = env.last_arrays
                np.savez_compressed(
                    self.directory / filename,
                    **arrays,
                    record_ids=np.asarray([group[i]["id"] for i in indices]),
                )
                for column, index in enumerate(indices):
                    row = rows[index]
                    if not row.get("native_release"):
                        raise RuntimeError(
                            "Strict geometry/native endpoint admission disagreement"
                        )
                    scalar = ScoreStream(
                        self.ref["scene"],
                        group[index]["bucket_xy_m"],
                        row["board_normal"],
                    )
                    for p, v, f in zip(
                        arrays["position_m"][:, column],
                        arrays["velocity_m_s"][:, column],
                        arrays["forces_N"][:, column],
                    ):
                        scalar.step(p, v, dict(zip(FORCE_NAMES, f)))
                    measured = scalar.result()
                    for key in (
                        "route_success",
                        "route_quality",
                        "complete_success",
                        "board_bounce",
                        "whole_ball_entry",
                        "external_contact_steps",
                        "board_contact_episodes",
                        "physics_steps",
                    ):
                        if measured[key] != row[key]:
                            raise RuntimeError(
                                "Independent scalar scorer mismatch: " + key
                            )
                    row.update(
                        scalar_score_audit_passed=True,
                        native_cache_file=filename,
                        native_cache_row=column,
                    )
            for row, candidate in zip(rows, group):
                row.update(
                    id=candidate["id"],
                    values=candidate["values"],
                    phase=phase,
                    geometry_reason=candidate["geometry_reason"],
                    proposal=candidate.get("proposal"),
                    q10=candidate["q10"],
                    q14=candidate["q14"],
                )
                results.append(row)
                self.rows.append(row)
                compact = {k: v for k, v in row.items() if k != "trajectory_m"}
                with (self.directory / "shots.jsonl").open("a") as stream:
                    stream.write(json.dumps(compact, allow_nan=False) + "\n")
            self.counts[
                "search_requests" if phase == "search" else "confirmation_requests"
            ] += len(group)
        return results

    def run(self, request):
        from .sweep_contract import proposals, rank_row
        from .contact_env import ContactEnvironment

        bucket = request["bucket_xy"]
        home = self.admission.check_home(bucket)
        if not home.get("passed"):
            return result_for(
                self.ref,
                request,
                status="common_home_unresolved",
                message="The prescribed common home is obstructed in this layout; another staging pose would need separate planning",
                counts=self.counts,
            )
        hints = [entry["source"] for entry in self.ref["catalogue"]["sources"]]
        values, metadata = proposals(
            self.ref["scene"],
            bucket,
            limit=request["cpu_candidates"],
            seed=20261128,
            hints=hints,
            constraints=request["constraints"],
        )
        env = None
        successful = []
        search_rows = []
        checked = []
        for offset in range(0, len(values), 32):
            admitted = self.audit(
                bucket, values[offset : offset + 32], metadata[offset : offset + 32]
            )
            checked.extend(admitted)
            remaining = request["search_budget"] - self.counts["search_requests"]
            eligible = [row for row in admitted if row["geometry_eligible"]][:remaining]
            if not eligible:
                if remaining <= 0:
                    break
                continue
            if env is None:
                env = ContactEnvironment(
                    self.ref["scene"],
                    self.assets,
                    n=self.batch,
                    bounds=WIDE_BOUNDS,
                    substeps=64,
                    board_mode="articulated",
                    gravity_mode="passive_only",
                )
                options = env.sim.mj_model.opt
                expected = self.ref["public"]["effective_native_options"]
                actual = dict(
                    timestep=float(options.timestep),
                    solver=int(options.solver),
                    cone=int(options.cone),
                    integrator=int(options.integrator),
                    iterations=int(options.iterations),
                    tolerance=float(options.tolerance),
                    gravity=np.asarray(options.gravity).tolist(),
                )
                if actual != expected:
                    raise RuntimeError(
                        "Effective native options differ from confirmed corrected physics"
                    )
            rows = self.execute(env, eligible, "search")
            search_rows.extend(rows)
            successful.extend(
                row
                for row in rows
                if row.get("native_release") and row.get("route_success")
            )
            if self.counts["search_requests"] >= request["search_budget"]:
                break
        atomic(self.directory / "geometry.json", checked)
        if not successful:
            best = max(search_rows, key=rank_row) if search_rows else None
            return result_for(self.ref, request, best, counts=self.counts)
        rng = np.random.default_rng(20261129)
        for source in sorted(successful, key=rank_row, reverse=True)[:3]:
            motion = self.admission.admit(source)
            atomic(self.directory / (source["id"] + "-motion.json"), motion)
            if not motion.get("passed"):
                continue
            exact = dict(
                source, geometry_eligible=True, geometry_reason="endpoint_admitted"
            )
            nominal = self.execute(
                env,
                [dict(exact, id=source["id"] + f"-nominal-{i}") for i in range(3)],
                "nominal",
            )
            if not all(
                row.get("native_release") and row.get("route_success")
                for row in nominal
            ):
                continue
            fresh_values = np.asarray(source["values"]) + rng.normal(0, NOISE, (20, 5))
            locks = constraints(20)
            for api, key in (
                ("board_position", "board"),
                ("dropper_position", "dropper"),
            ):
                if request["constraints"][api] is not None:
                    locks[key][:] = request["constraints"][api]
            fresh_values, allowed = project(
                fresh_values, locks, self.ref["scene"]["ball"]["radius_m"], WIDE_BOUNDS
            )
            hint = {key: source[key] for key in ("values", "bucket_xy_m", "q10", "q14")}
            fresh = self.audit(
                bucket, fresh_values, [dict(source_hint=hint) for _ in range(20)]
            )
            for i, candidate in enumerate(fresh):
                candidate["id"] = source["id"] + f"-fresh-{i}"
                candidate["geometry_eligible"] &= bool(allowed[i])
                if candidate["geometry_eligible"]:
                    proof = self.admission.connect_variation(source, candidate)
                    candidate["geometry_eligible"] &= bool(proof.get("passed"))
            repeated = self.execute(env, fresh, "fresh")
            captured = sum(
                bool(row.get("native_release") and row.get("route_success"))
                for row in repeated
            )
            self.counts.update(
                nominal_attempts=3,
                nominal_captures=3,
                fresh_attempts=20,
                fresh_captures=captured,
            )
            if captured < 18:
                continue
            best = nominal[0]
            return result_for(
                self.ref,
                request,
                best,
                status="verified_shot",
                message=f"Confirmed simulated shot: 3/3 nominal and {captured}/20 fresh placement variations; sampled common-home path admitted",
                counts=self.counts,
                motion=motion,
                qualified=True,
            )
        best = max(search_rows, key=rank_row)
        return result_for(
            self.ref,
            request,
            best,
            status="capture_found_confirmation_unresolved",
            message="A search shot captured, but repeatability or common-home path qualification did not pass; not ready to run",
            counts=self.counts,
        )


def run(args):
    from .sweep_contract import validate_request, request_key

    ref = load_reference(args.reference, args.assets)
    request = validate_request(json.loads(args.request.read_text()), ref["scene"])
    key = request_key(
        ref["scene"], ref["asset_hashes"], request, source_hashes=ref["source_hashes"]
    )
    cache = args.cache / key
    result_path = cache / "result.json"
    saved = read_query_cache(cache, key, ref["source_hashes"])
    if saved is not None:
        saved.update(request_id=request["request_id"], method="catalog_lookup")
        saved["provenance"].update(saved_evidence=True, search_at_inference=False)
        return atomic(args.output, saved)
    source = exact_source(ref, request)
    if source:
        result = cached_result(ref, request, source)
        result["provenance"]["request_sha256"] = key
        return atomic(args.output, result)
    from .gpu_slot import gpu_slot

    try:
        with gpu_slot(args.gpu_lock, timeout_s=args.queue_timeout, poll_s=1):
            cache.mkdir(parents=True, exist_ok=True)
            attempt = cache / ("attempt-" + uuid.uuid4().hex)
            planner = NativePlanner(ref, args.assets, attempt)
            begin = time.monotonic()
            result = planner.run(request)
            result["metrics"]["elapsed_s"] = time.monotonic() - begin
            result["provenance"]["request_sha256"] = key
            result["provenance"]["evidence_directory"] = attempt.name
            atomic(result_path, result)
            manifest = {
                str(p.relative_to(cache)): sha(p)
                for p in cache.rglob("*")
                if p.is_file() and p.name != "manifest.json"
            }
            atomic(cache / "manifest.json", manifest)
    except TimeoutError:
        result = result_for(
            ref,
            request,
            status="busy",
            message="The 4090 is running another experiment. Existing work was left untouched; retry after it finishes.",
        )
    atomic(args.output, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("reference", "assets", "gpu-lock", "cache", "request", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--queue-timeout", type=float, default=0)
    from .learning_solve import _bound_child_lifetime

    _bound_child_lifetime()
    signal.alarm(600)
    run(parser.parse_args())
