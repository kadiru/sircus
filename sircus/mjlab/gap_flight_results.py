"""Read-only, CPU-derived contact results for the audited30-shot correction.

Original native files remain immutable. This module verifies their full proof,
rescans bound dense traces and writes a compact derivative outside that study.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path

import numpy as np

from .prepare_gap_flight_report import correction, sha, read
from .benchmark_verify import contained
from .calibrated_score import FORCE_THRESHOLD_N
from .calibrated_score_oracle import score_trace
from .xy650_verify import first_flight


VERSION = "sircus-gap-flight-contact-results-v1"
CONTEXTS = tuple("gap-coverage-test-"+code for code in ("12", "24", "48"))
NAMES = ("board", "bucket", "table", "floor", "robot", "unattributed")
SCOPE = ("Audited fixed64 development repeats on three selected previous failures. "
    "Robot force is an aggregate channel, not a measured link identity. Contact is force>0.01N. "
    "Before-terminal means after the first board rebound and before the unchanged scorer terminates; "
    "the tail is separately inspected through the full4s native recording. Gap changes clearance and impact energy. "
    "No fresh holdout, noise,128-step, driven-arm-motion or physical validation.")


class Checks:
    def __init__(self): self.count = 0
    def require(self, condition, label):
        self.count += 1
        if not bool(condition): raise ValueError(label)


def validate_packet(arrays, cohort, trace_spec, checks):
    """Authenticate dimensions, identity and all recorded numeric channels."""
    required = {"record_ids", "position_m", "velocity_m_s", "forces_N", "time_s", "dt_s", "native_time_s",
        "contact_distance_m", "contact_geom_ids", "contact_position_m", "contact_normal"}
    checks.require(required <= set(arrays), "Complete native arrays required")
    ids = arrays["record_ids"].tolist(); n = len(cohort)
    checks.require(isinstance(ids, list) and len(ids) == n and len(set(ids)) == n, "Unique complete cache IDs")
    columns = [row["native_cache_row"] for row in cohort]
    checks.require(all(type(c) is int for c in columns) and sorted(columns) == list(range(n)), "Every native cache column assigned exactly once")
    for row in cohort:
        checks.require(ids[row["native_cache_row"]] == row["id"], "Native cache column belongs to its exact requested record")
    dt = float(trace_spec["native_dt_s"]); ticks = int(round(float(trace_spec["horizon_s"])/dt))
    checks.require(np.asarray(arrays["dt_s"]).shape == () and float(arrays["dt_s"]) == dt and dt == 1/15360,
                   "Exact fixed64 timestep required")
    shapes = dict(position_m=(ticks+1, n, 3), velocity_m_s=(ticks+1, n, 3), forces_N=(ticks, n, 6),
        time_s=(ticks+1,), native_time_s=(ticks+1, n), contact_distance_m=(ticks, n, 2),
        contact_geom_ids=(ticks, n, 2, 2), contact_position_m=(ticks, n, 2, 3), contact_normal=(ticks, n, 2, 3))
    for key, shape in shapes.items(): checks.require(arrays[key].shape == shape, "Native array dimensions/"+key)
    for key in ("position_m", "velocity_m_s", "forces_N", "time_s", "native_time_s", "contact_position_m", "contact_normal"):
        checks.require(np.isfinite(arrays[key]).all(), "Finite native numeric channel/"+key)
    checks.require(np.array_equal(arrays["time_s"], np.arange(ticks+1)*dt), "Canonical interval times unchanged")
    checks.require((np.diff(arrays["native_time_s"], axis=0) > 0).all(), "Native clocks advance")
    checks.require((arrays["forces_N"] >= 0).all(), "Force magnitudes nonnegative")
    distances = arrays["contact_distance_m"]
    checks.require((np.isfinite(distances) | np.isposinf(distances)).all(), "Contact distances finite or explicitly absent")
    present = np.isfinite(distances); geoms = arrays["contact_geom_ids"]
    checks.require(np.issubdtype(geoms.dtype, np.integer), "Integral contact geometry IDs")
    checks.require((geoms[present] >= 0).all() and (geoms[~present] == -1).all(), "Absent and present contact identities distinct")
    lengths = np.linalg.norm(arrays["contact_normal"], axis=-1)
    checks.require((~present | (np.abs(lengths-1) <= 1e-4) | ((lengths == 0) & (np.abs(distances) <= 2e-6))).all(),
                   "Native contact normals unit or audited precision tangencies")


def scan_contact(row, arrays, metric, checks=None):
    """Use the same0.01N threshold; absent rebound evidence stays unknown."""
    checks = checks or Checks()
    result = dict(id=row["id"], repeat=row["repeat"], native_release=row.get("native_release") is True,
        capture=bool(row.get("native_release") and row.get("route_success") and not row.get("invalid") and row.get("invalid_physics") is False),
        clean=bool(row.get("route_success") and row.get("route_quality") == 2), route_quality=row.get("route_quality"),
        whole_ball_entry=row.get("whole_ball_entry"), retention_s=row.get("max_retention_s"),
        entry_class=row.get("entry_class"), rim_band_contact=row.get("rim_band_contact"),
        bucket_contact_before_entry=row.get("bucket_contact_before_entry"), exited_after_entry=row.get("exited_after_entry"),
        minimum_mouth_clearance_mm=row.get("minimum_mouth_clearance_mm"),
        observation_status="unknown", first_post_bounce_contact_channels=None, first_contact_delay_ms=None,
        robot_contact_intervals_before_terminal=None, robot_contact_intervals_after_terminal=None,
        robot_contact_intervals_post_rebound=None, peak_robot_force_N_before_terminal=None,
        peak_robot_force_N_after_terminal=None, first_robot_contact_time_s=None, last_robot_contact_time_s=None,
        rebound_time_s=None, terminal_time_s=None, trace_end_time_s=None)
    if row.get("native_release") is not True:
        result["observation_status"] = "missing_no_native_release"; return result
    if not metric.get("available"):
        result["observation_status"] = "missing_no_board_rebound"; return result
    forces, times = np.asarray(arrays["forces_N"]), np.asarray(arrays["time_s"])
    start, terminal = metric["start_index"], row["physics_steps"]
    checks.require(type(start) is int and type(terminal) is int and 0 <= start <= terminal <= len(forces), "Observed rebound precedes scorer terminal")
    checks.require(forces.ndim == 2 and forces.shape[1] == 6 and times.shape == (len(forces)+1,), "Aligned scalar interval traces")
    checks.require(FORCE_THRESHOLD_N == .01, "Existing scoring contact threshold unchanged")
    active = np.flatnonzero(forces[:, 4] > FORCE_THRESHOLD_N)
    before = active[(active >= start) & (active < terminal)]
    after = active[active >= terminal]
    post = active[active >= start]
    peak = lambda left, right: float(forces[left:right, 4].max()) if right > left else 0.
    result.update(observation_status="observed", first_post_bounce_contact_channels=list(metric["cutoff_contact_channels"]),
        first_contact_delay_ms=(metric["end_time_s"]-metric["start_time_s"])*1000 if metric["cutoff_reason"] == "contact" else None,
        robot_contact_intervals_before_terminal=len(before), robot_contact_intervals_after_terminal=len(after),
        robot_contact_intervals_post_rebound=len(post), peak_robot_force_N_before_terminal=peak(start, terminal),
        peak_robot_force_N_after_terminal=peak(terminal, len(forces)),
        first_robot_contact_time_s=float(times[post[0]]) if len(post) else None,
        last_robot_contact_time_s=float(times[post[-1]+1]) if len(post) else None,
        rebound_time_s=float(times[start]), terminal_time_s=float(times[terminal]), trace_end_time_s=float(times[-1]))
    return result


def summarize(items):
    observed = [r for r in items if r["observation_status"] == "observed"]
    known = len(observed) == len(items)
    result = dict(requested=len(items), native_releases=sum(r["native_release"] for r in items),
        captures=sum(r["capture"] for r in items), clean=sum(r["clean"] for r in items),
        contact_observations=len(observed), contact_unknown=len(items)-len(observed),
        first_post_bounce_contact_counts=dict(Counter(channel for r in observed for channel in r["first_post_bounce_contact_channels"])),
        first_contact_delay_ms=[r["first_contact_delay_ms"] for r in items], retention_s=[r["retention_s"] for r in items])
    for segment in ("before_terminal", "after_terminal", "post_rebound"):
        values = [r["robot_contact_intervals_"+segment] for r in observed]
        result["robot_contact_trials_"+segment] = sum(v > 0 for v in values) if known else None
        result["robot_contact_intervals_"+segment] = sum(values) if known else None
    for segment in ("before_terminal", "after_terminal"):
        values = [r["peak_robot_force_N_"+segment] for r in observed]
        result["peak_robot_force_N_"+segment] = max(values) if known and values else None
    for key in ("rim_band_contact", "bucket_contact_before_entry", "exited_after_entry"):
        values = [r[key] for r in items]
        result[key+"_trials"] = sum(values) if all(type(v) is bool for v in values) else None
    result["route_quality_counts"] = dict(Counter(str(r["route_quality"]) for r in items if r["route_quality"] is not None))
    return result


def build(study, audit):
    study, audit = Path(study).resolve(), Path(audit).resolve()
    result = correction(study, audit)
    if result.get("status") != "passed" or result.get("audit", {}).get("full_checks_verified") is not True:
        raise ValueError("Completed correction with fully verified native proof required")
    spec, proof, manifest = read(study/"contract.json"), read(audit), read(study/"evidence-manifest.json")
    rows = [json.loads(line) for line in (study/"records.jsonl").read_text().splitlines()]
    checks = Checks(); checks.require(len(rows) == len({r["id"] for r in rows}) == 30, "All30 unique requested slots")
    helpers = ("calibrated_score.py", "calibrated_score_oracle.py", "xy650_verify.py", "prepare_gap_flight_report.py")
    source_hashes = {"sircus/mjlab/"+name: sha(Path(__file__).with_name(name)) for name in helpers}
    for name in helpers[:3]:
        relative = "sircus/mjlab/"+name
        checks.require(source_hashes[relative] == spec["native_source_sha256"][relative], "Exact original contact/scorer helper/"+name)
    checks.require([r["id"] for r in rows] == [r["id"] for r in spec["plan"]], "Literal frozen request order")
    checks.require(Counter(r["role"] for r in rows) == Counter(diagnostic=18, positive_control=6, negative_control=6), "Controls distinct from18 development requests")
    grouped = {}
    for row in rows:
        checks.require(type(row.get("native_release")) is bool, "Explicit native outcome")
        if row["native_release"]: grouped.setdefault(row["native_cache_file"], []).append(row)
    observed = {}; caches = {}
    calibration = spec["identity"]["calibration"]["score_config"]
    for filename, cohort in grouped.items():
        path = contained(study/"native", filename); fingerprint = sha(path)
        checks.require(manifest["native/"+filename] == fingerprint, "Manifest-bound raw native cache")
        for row in cohort: checks.require(row["native_cache_sha256"] == fingerprint, "Record-bound raw native cache")
        with np.load(path, allow_pickle=False) as archive: arrays = {key: archive[key] for key in archive.files}
        validate_packet(arrays, cohort, spec["identity"]["trace_schema"], checks)
        caches["native/"+filename] = fingerprint
        for row in cohort:
            column = row["native_cache_row"]
            trace = {k: v[:, column] if v.ndim >= 2 and k != "time_s" else v for k, v in arrays.items() if k != "record_ids"}
            scored = score_trace(spec["scene"], row["bucket_xy_m"], row["board_normal"], calibration, trace)
            for key in ("complete_success", "route_success", "route_quality", "whole_ball_entry", "external_contact_steps", "physics_steps", "invalid_physics",
                        "entry_class", "rim_band_contact", "bucket_contact_before_entry", "exited_after_entry"):
                checks.require(scored[key] == row[key], "Independent scalar score/"+key)
            checks.require(abs(scored["max_retention_s"]-row["max_retention_s"]) <= 1e-9, "Independent retention clock")
            expected_margin, saved_margin = scored["minimum_mouth_clearance_mm"], row["minimum_mouth_clearance_mm"]
            checks.require(expected_margin == saved_margin if expected_margin is None or saved_margin is None
                           else abs(expected_margin-saved_margin) <= 1e-7, "Independent mouth clearance diagnostic")
            metric = first_flight(spec["scene"], row["bucket_xy_m"], row["board_normal"], calibration, trace, valid=row.get("invalid_physics") is False)
            checks.require(metric == row["first_flight"], "Independent first-contact timing and channels")
            observed[row["id"]] = scan_contact(row, trace, metric, checks)
    for row in rows:
        if not row["native_release"]: observed[row["id"]] = scan_contact(row, {}, {}, checks)
    contacts = []; cases = []
    for method, variant in (("original", "original"), ("corrected", "gap_plus60")):
        group = [r for r in rows if r["role"] == "diagnostic" and r["variant"] == variant]
        checks.require(len(group) == 9, "Exactly9 requested attempts per variant")
        contacts.append(dict(method=method, **summarize([observed[r["id"]] for r in group])))
    for context in CONTEXTS:
        entry = dict(context_id=context)
        original_values = None
        for method, variant in (("original", "original"), ("corrected", "gap_plus60")):
            group = [r for r in rows if r["context_id"] == context and r["variant"] == variant]
            checks.require(len(group) == 3 and [r["repeat"] for r in group] == [0, 1, 2], "Three ordered repeats per case and variant")
            values, bucket = group[0]["values"], group[0]["bucket_xy_m"]
            checks.require(all(r["values"] == values and r["bucket_xy_m"] == bucket for r in group), "All repeats preserve exact requested geometry")
            if method == "original":
                original_values = values
                entry.update(bucket_xy_m=bucket, board_xyz_m=values[:3], board_tilt_deg=values[3],
                             board_bucket_distance_mm=float(np.linalg.norm(np.asarray(values[:2])-bucket)*1000))
            else:
                checks.require(values[:4] == original_values[:4] and values[4] == original_values[4]+.06
                               and bucket == entry["bucket_xy_m"], "Only requested release gap increases60mm")
            entry[method] = dict(gap_mm=values[4]*1000,
                release_ball_center_z_mm=(values[2]+values[4]+spec["scene"]["ball"]["radius_m"])*1000,
                **summarize([observed[r["id"]] for r in group]))
            entry[method]["records"] = [observed[r["id"]] for r in group]
        cases.append(entry)
    for comparison, contact in zip(result["comparisons"], contacts):
        for key in ("requested", "native_releases", "captures", "clean"):
            checks.require(comparison["counts"][key] == contact[key], "Derived cohort retains audited count/"+key)
    result.update(contact_comparison=contacts, case_comparison=cases,
        control_summary=dict(requested=12, source="unchanged matched controls", gates=copy.deepcopy(proof["control_gates"])),
        contact_provenance=dict(version=VERSION, verified=True, full_checks_verified=True, checks_count=proof["checks_count"],
            derived_checks_count=checks.count, plan_sha256=result["plan_sha256"], report_sha256=result["report_sha256"],
            manifest_sha256=result["manifest_sha256"], audit_sha256=sha(audit), source_sha256=sha(__file__),
            helper_sources_sha256=source_hashes, native_cache_sha256=caches, records_sha256=sha(study/"records.jsonl"),
            force_threshold_N=FORCE_THRESHOLD_N, force_channel="robot", force_channel_index=4,
            trace_horizon_s=spec["identity"]["trace_schema"]["horizon_s"], substeps=64,
            calibration_fingerprint=calibration["fingerprint"], scope=SCOPE))
    result["audit"]["source_sha256"] = sha(audit)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("study", "audit", "output"): parser.add_argument("--"+key, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.resolve().is_relative_to(args.study.resolve()):
        raise ValueError("New derivative output directory outside immutable native study required")
    result = build(args.study, args.audit)
    args.output.mkdir(parents=True)
    path = args.output/"correction-results.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    (args.output/"checksums.json").write_text(json.dumps({path.name: sha(path)}, indent=2)+"\n")
    print(json.dumps(dict(status=result["status"], contacts=result["contact_comparison"], output=str(path))))


if __name__ == "__main__": main()
