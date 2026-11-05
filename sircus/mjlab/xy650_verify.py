"""Independent CPU audit of frozen translated-plane 650 mm development shots.

Capture labels retain the contact-calibrated oracle. The new first-flight
diagnostic is recomputed independently from saved dense observations; it does
not import the producer diagnostic, ranking, plan or fingerprint helpers.
"""
from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
import copy
import itertools
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

from .benchmark_verify import EvidenceError, contained, digest, read, sha
from . import clean650_verify as clean_audit
from . import replay650_verify as replay

VERSION = "sircus-xy650-independent-audit-v1"
STUDY_VERSION = "sircus-xy650-shift-study-v1"
METRIC_VERSION = "sircus-xy650-first-flight-aperture-v1"
DISTANCE_RULE = "translated-board-and-bucket-fixed650-bearing-v1"
DX = (-.01, 0., .01)
DY = (.04, .06, .08)
HEIGHTS = (.36, .38, .40)
TILTS = (26., 28., 30.)
GAPS = (.39, .41, .43)
OFFSETS = ((-.005, 0., .005), (-.5, 0., .5), (-.005, 0., .005))
STEPS = (64, 128)
MAX_REQUESTS = 388
FORCE_NAMES = ("board", "bucket", "table", "floor", "robot", "unattributed")
AuditChecks = replay.AuditChecks
statistics = clean_audit.statistics
native_valid = clean_audit.native_valid
clean = clean_audit.clean
control_gate = clean_audit.control_gate
qualify = clean_audit.qualify


def target_key(row):
    """Collapse representation-only duplicates, without changing any action."""
    return tuple(round(float(v), 9) for v in [*row["values"], *row["bucket_xy_m"]])


def order_key(row):
    first = row.get("first_flight") or {}
    crossing = bool(first.get("available") and first.get("first_crossing") is not None and first.get("mouth_clearance_m") is not None)
    margin = first.get("mouth_clearance_m")
    distance = first.get("closest_opening_distance_m")
    margin = float(margin) if isinstance(margin, (int, float)) and math.isfinite(margin) else -1e9
    distance = float(distance) if isinstance(distance, (int, float)) and math.isfinite(distance) else 1e6
    return (-int(clean(row)), -int(replay.captured(row)), -int(crossing), -margin, distance, row["id"])


def select(rows, count, *, unique=False):
    selected, seen = [], set()
    for row in sorted((r for r in rows if native_valid(r)), key=order_key):
        key = target_key(row)
        if unique and key in seen:
            continue
        seen.add(key);selected.append(row)
        if len(selected) == count:
            break
    return selected


def initial_inputs(seed):
    """Literal253 queries: nine27-shot grids, nine translated references, original."""
    result = []
    for dx, dy in itertools.product(DX, DY):
        for height, tilt, gap in itertools.product(HEIGHTS, TILTS, GAPS):
            result.append(dict(values=[seed["values"][0]+dx, seed["values"][1]+dy, height, tilt, gap],
                               bucket_xy_m=[seed["bucket_xy_m"][0]+dx, seed["bucket_xy_m"][1]+dy],
                               translation_xy_m=[dx, dy]))
    for dx, dy in itertools.product(DX, DY):
        result.append(dict(values=[seed["values"][0]+dx, seed["values"][1]+dy, .38, 28., .3868],
                           bucket_xy_m=[seed["bucket_xy_m"][0]+dx, seed["bucket_xy_m"][1]+dy],
                           translation_xy_m=[dx, dy]))
    result.append(dict(values=copy.deepcopy(seed["values"]), bucket_xy_m=copy.deepcopy(seed["bucket_xy_m"]),
                       translation_xy_m=[0., 0.]))
    return result


def local_inputs(source):
    values = source["values"]
    return [[values[0], values[1], values[2]+h, values[3]+t, values[4]+g]
            for h, t, g in itertools.product(*OFFSETS)]


def first_flight(scene, bucket_xy, normal, calibration, arrays, *, valid=True):
    """Independent scalar state scan; force-only contact cutoff is diagnostic.

    A contact interval k joins states k and k+1. A flight ending at k includes
    its start state k, never the potentially already-deflected end state k+1.
    Neither this metric nor its signed aperture margin changes capture labels.
    """
    empty = dict(version=METRIC_VERSION, crossing_plane="ball_center_at_rim", available=False,
                 reason=None, start_index=None, end_index=None, start_time_s=None, end_time_s=None,
                 cutoff_reason=None, cutoff_contact_channels=[], first_crossing=None,
                 mouth_clearance_m=None, closest_opening_distance_m=None, sample_count=0)
    if type(valid) is not bool:
        raise ValueError("Explicit first-flight validity boolean required")
    if not valid:
        return dict(empty, reason="invalid_physics")
    if not isinstance(arrays, Mapping) or any(k not in arrays for k in ("position_m", "velocity_m_s", "forces_N")) \
            or not any(k in arrays for k in ("time_s", "dt_s")):
        return dict(empty, reason="missing_trace")
    from .calibrated_score_oracle import validate_calibration
    validate_calibration(calibration)
    position = np.asarray(arrays["position_m"], dtype=float)
    velocity = np.asarray(arrays["velocity_m_s"], dtype=float)
    force = np.asarray(arrays["forces_N"], dtype=float)
    size = len(position) if position.ndim else 0
    if position.shape != (size, 3) or size < 2 or velocity.shape != position.shape or force.shape != (size-1, 6):
        raise ValueError("Wrong first-flight dense trace shape")
    if not all(np.isfinite(v).all() for v in (position, velocity, force)) or np.any(force < 0):
        raise ValueError("Nonfinite or negative first-flight trace")
    dt = float(np.asarray(arrays["dt_s"]).item()) if "dt_s" in arrays else None
    if dt is not None and (not math.isfinite(dt) or dt <= 0):
        raise ValueError("Invalid first-flight timestep")
    times = np.asarray(arrays["time_s"], dtype=float) if "time_s" in arrays else np.arange(size)*dt
    if times.shape != (size,) or not np.isfinite(times).all() or abs(times[0]) > 1e-12 or np.min(np.diff(times)) <= 0:
        raise ValueError("Invalid first-flight timestamps")
    if dt is not None and not np.allclose(np.diff(times), dt, rtol=1e-9, atol=1e-12):
        raise ValueError("Contradictory first-flight cadence")
    if "contact_distance_m" in arrays:
        contact = np.asarray(arrays["contact_distance_m"], dtype=float)
        if contact.shape != (size-1, 2) or np.isnan(contact).any() or np.isneginf(contact).any():
            raise ValueError("Invalid first-flight contact distances")
    axis, bucket = np.asarray(normal, dtype=float), np.asarray(bucket_xy, dtype=float)
    if axis.shape != (3,) or bucket.shape != (2,) or not np.isfinite(axis).all() or not np.isfinite(bucket).all() \
            or abs(float(np.linalg.norm(axis))-1.) > 1e-6:
        raise ValueError("Invalid first-flight references")
    planes = np.asarray(calibration["compiled_planes"], dtype=float)
    radius = float(scene["ball"]["radius_m"])
    rim = float(calibration.get("compiled_rim_m", scene["bucket"]["assumed_bottom_thickness_m"]+scene["bucket"]["assumed_interior_depth_m"]))
    if planes.ndim != 2 or planes.shape[1:] != (4,) or len(planes) < 3 or not np.isfinite(planes).all() \
            or not math.isfinite(radius) or radius <= 0 or not math.isfinite(rim) \
            or np.any(np.hypot(planes[:, 0], planes[:, 1]) <= 0):
        raise ValueError("Invalid first-flight aperture geometry")
    bucket_geometry = scene["bucket"]
    floor = float(bucket_geometry["assumed_bottom_thickness_m"])
    top = floor+float(bucket_geometry["assumed_interior_depth_m"])
    epsilon = float(calibration["geometry_epsilon_m"])
    if not 0 <= floor < top or len(planes) != int(bucket_geometry["segments"]) \
            or max(calibration["wall_tolerance_m"], calibration["floor_tolerance_m"], epsilon) >= radius:
        raise ValueError("Inconsistent first-flight bucket geometry")
    angles = sorted(math.atan2(-row[1], -row[0]) % (2*math.pi) for row in planes)
    for index, angle in enumerate(angles):
        if abs(angle-(index+.5)*2*math.pi/len(planes))*bucket_geometry["outer_top_diameter_m"]/2 > epsilon:
            raise ValueError("Wrong first-flight compiled wall directions")
    for height, diameter in ((floor, bucket_geometry["inner_bottom_diameter_m"]), (top, bucket_geometry["inner_top_diameter_m"])):
        expected = diameter*math.cos(math.pi/len(planes))/2
        for nx, ny, nz, b in planes:
            if abs((nz*height+b)/math.hypot(nx, ny)-expected) > epsilon:
                raise ValueError("Wrong first-flight compiled wall geometry")
    for name, expected in (("compiled_floor_m", floor), ("compiled_rim_m", top)):
        if name in calibration and abs(float(calibration[name])-expected) > epsilon:
            raise ValueError("Wrong first-flight compiled vertical geometry")
    latest_contact, start = None, None
    for state in range(1, size):
        if force[state-1, 0] > .01:
            latest_contact = float(times[state])
        elif latest_contact is not None and times[state]-latest_contact <= .05+1e-12 \
                and float(np.dot(velocity[state], axis)) > .05:
            start = state
            break
    if start is None:
        return dict(empty, reason="no_board_rebound")
    end, channels, cutoff = size-1, [], "horizon"
    for interval in range(start, size-1):
        active = [name for column, name in enumerate(FORCE_NAMES) if force[interval, column] > .01]
        if active:
            end, channels, cutoff = interval, active, "contact"
            break
    differences = position[start:end+1]-np.asarray([bucket[0], bucket[1], rim])
    closest = float(np.sqrt(np.sum(differences*differences, axis=1)).min())
    crossing, margin = None, None
    for right in range(start+1, end+1):
        left = right-1
        upper, lower = float(position[left, 2]), float(position[right, 2])
        if upper > rim and lower <= rim and upper > lower:
            fraction = (upper-rim)/(upper-lower)
            xy = position[left, :2]+fraction*(position[right, :2]-position[left, :2])
            relative = xy-bucket
            distances = [(nx*relative[0]+ny*relative[1]+nz*rim+b)/math.hypot(nx, ny)
                         for nx, ny, nz, b in planes]
            margin = float(min(distances)-radius)
            crossing = dict(time_s=float(times[left]+fraction*(times[right]-times[left])),
                            interval_index=left, fraction=float(fraction), xy_m=xy.tolist())
            break
    return dict(empty, available=True, reason=None, start_index=start, end_index=end,
                start_time_s=float(times[start]), end_time_s=float(times[end]), cutoff_reason=cutoff,
                cutoff_contact_channels=channels, first_crossing=crossing, mouth_clearance_m=margin,
                closest_opening_distance_m=closest, sample_count=end-start+1)


def same(actual, expected, checks, label):
    """Compare complete nested numerical diagnostics without hiding missing keys."""
    if isinstance(expected, dict):
        checks.require(isinstance(actual, dict), label+"/object required")
        checks.equal(set(actual), set(expected), label+"/exact fields")
        for key, value in expected.items():
            same(actual[key], value, checks, label+"/"+key)
    elif isinstance(expected, (list, tuple)):
        checks.require(isinstance(actual, (list, tuple)), label+"/list required")
        checks.equal(len(actual), len(expected), label+"/length")
        for index, value in enumerate(expected):
            same(actual[index], value, checks, label+"/"+str(index))
    elif isinstance(expected, float):
        checks.numeric(actual, expected, label, 1e-10)
    else:
        checks.equal(actual, expected, label)


def auditor_identity(producer_source, producer_hashes, checks):
    root = Path(__file__).resolve().parents[2]
    own = "sircus/mjlab/xy650_verify.py"
    dependencies = {}
    for module in tuple(sys.modules.values()):
        name = getattr(module, "__name__", "")
        filename = getattr(module, "__file__", None)
        if not filename or not (name.startswith("sircus.mjlab.") or name.startswith("sircus.isaac.")):
            continue
        path = Path(filename).resolve()
        checks.require(path.is_relative_to(root), "One explicit XY auditor source snapshot/"+name)
        relative = str(path.relative_to(root))
        if len(Path(relative).parts) != 3 or path.suffix != ".py" or relative == own:
            continue
        if relative.startswith("sircus/isaac/") and relative not in producer_hashes:
            continue
        checks.require(relative in producer_hashes, "Actual imported dependency declared/"+relative)
        actual = sha(path)
        checks.equal(actual, producer_hashes[relative], "Unchanged imported XY audit dependency/"+relative)
        dependencies[relative] = actual
    checks.require("sircus/mjlab/calibrated_score_oracle.py" in dependencies, "Independent calibrated scorer retained")
    checks.equal(sha(contained(producer_source, own)), producer_hashes[own], "Original XY producer checker bytes retained")
    return dict(version=VERSION, auditor_sha256=sha(Path(__file__)),
                original_producer_auditor_sha256=producer_hashes[own], dependencies_sha256=dependencies)


def audit_raised_parent(study, bindings, checks):
    """Require the exact completed parent, not merely a plausible prior score."""
    names = {"contract.json", "report.json", "audit-clean650.json", "records.jsonl", "evidence-manifest.json"}
    checks.equal(set(bindings), names, "XY study binds complete raised parent inputs")
    for name, expected in bindings.items():
        checks.equal(sha(contained(study, name)), expected, "Exact frozen raised parent/"+name)
    spec, report, proof = (read(study/name) for name in ("contract.json", "report.json", "audit-clean650.json"))
    unsigned = dict(spec);fingerprint = unsigned.pop("fingerprint", None)
    checks.equal(digest(unsigned), fingerprint, "Raised parent complete identity")
    checks.equal(fingerprint, "fe28a1277b7d7e32428dd736346d9336ca274d7a0fe9c0dc11ac61bab6970036", "Exact prespecified raised parent")
    checks.equal(spec["version"], "sircus-clean650-height-tilt-gap-raised-v3", "Raised parent protocol preserved")
    checks.equal((report["version"], report["status"], report["phase"]),
                 (spec["version"], "completed", "completed"), "Raised parent completed")
    checks.equal(report["contract_fingerprint"], fingerprint, "Raised report exact identity")
    checks.equal(proof["version"], "sircus-clean650-independent-audit-v3", "Original raised independent audit")
    checks.equal(proof["status"], "passed", "Raised parent independent audit passed")
    checks.require(not proof.get("failures") and proof["checks_count"] == len(proof["checks"]) > 0
                   and all(c.get("passed") is True for c in proof["checks"]), "Complete raised audit strictly passed")
    checks.equal(proof["contract_fingerprint"], fingerprint, "Raised audit exact identity")
    checks.equal(proof["report_sha256"], sha(study/"report.json"), "Raised audit binds report bytes")
    checks.equal(proof["evidence_manifest_sha256"], sha(study/"evidence-manifest.json"), "Raised audit binds all evidence bytes")
    manifest = replay.check_manifest(study, checks)
    checks.equal(report["evidence_sha256"], manifest, "Raised report binds raw manifest")
    checks.equal(proof["requested_trials"], 163, "Original raised requested denominator")
    checks.equal(proof["native_releases"], 120, "Original raised native denominator")
    checks.equal(proof["gate"], report["gate"], "Raised outcome gate unchanged")
    return spec


def audit_first_flight(study, rows, scene, calibration, checks):
    """Recompute new diagnostics only from the exact saved native columns."""
    by_cache = {}
    for row in rows:
        if row["native_release"]:
            by_cache.setdefault((row["native_directory"], row["native_cache_file"]), []).append(row)
        else:
            checks.equal(row["first_flight"], None, row["id"]+"/reject has no observed first flight")
    for (directory, name), members in by_cache.items():
        with np.load(contained(study, directory)/name, allow_pickle=False) as archive:
            for row in members:
                column = row["native_cache_row"]
                arrays = {key: (archive[key][:, column] if archive[key].ndim >= 2 else archive[key])
                          for key in archive.files if key != "record_ids"}
                expected = first_flight(scene, row["bucket_xy_m"], row["board_normal"], calibration, arrays,
                                        valid=native_valid(row))
                same(row["first_flight"], expected, checks, row["id"]+"/independent first flight")


def scene_fingerprint(scene, calibration, assets):
    return digest(dict(scene=scene, calibration=calibration, asset_sha256=assets,
                       capture_version="sircus-calibrated-native-capture-v2"))


def query_fingerprint(group, scene_fp):
    return digest(dict(scene_fingerprint=scene_fp, values=group["values"], bucket_xy_m=group["bucket_xy_m"],
                       distance_rule="historical-control-unchanged" if group["role"].endswith("_control") else DISTANCE_RULE,
                       source_record_id=group["source_record_id"], source_record_sha256=group["source_record_sha256"]))


def execution_fingerprint(row):
    return digest({key: row[key] for key in ("id", "query_fingerprint", "substeps", "repeat_index", "q10", "q14",
                                            "native_initial_q10", "native_initial_q14")})


def make_plan(groups, phase, *, steps, repeats):
    rows = clean_audit.make_plan(groups, phase, steps=steps, repeats=repeats)
    lookup = {group["id"]: group for group in groups}
    for row in rows:
        group = lookup[row["group_id"]]
        for key in ("query_fingerprint", "query_kind", "translation_m", "source_record_id", "source_record_sha256",
                    "source_query_fingerprint"):
            if key in group:
                row[key] = copy.deepcopy(group[key])
    return rows


def audit_initial(groups, seed, controls, checks):
    checks.equal(len(groups), 255, "Two unchanged controls and253 prespecified development queries")
    for actual, source in zip(groups[:2], controls):
        checks.equal(actual["id"], source["id"], "Historical control group identity")
        checks.equal(actual["role"], source["role"], "Historical control role")
        clean_audit.source_fields(actual, source, checks, actual["id"])
        for key in ("q10", "q14"):
            checks.equal(actual[key], source[key], "Original control joint branch/"+key)
        checks.equal(actual["phase"], "controls", "Historical control phase")
        checks.equal(actual["query_kind"], "historical_control", "Historical controls distinguished from650 search")
        checks.equal(actual["translation_m"], [0., 0.], "Controls are never translated")
    expected = initial_inputs(seed)
    for index, (actual, target) in enumerate(zip(groups[2:], expected)):
        ident = f"xy-grid-{index:04d}" if index < 243 else f"xy-reference-{index-243:02d}" if index < 252 else "reference-650"
        kind = "translated_grid" if index < 243 else "translated_reference" if index < 252 else "original_reference"
        checks.equal(actual["id"], ident, "Literal ordered translated query ID")
        checks.equal(actual["query_kind"], kind, ident+"/literal query type")
        checks.equal(actual["role"], "discovery", ident+"/development role")
        checks.equal(actual["phase"], "coarse", ident+"/development phase")
        checks.numeric(actual["values"], target["values"], ident+"/exact translated H/T/G", 1e-12)
        checks.numeric(actual["bucket_xy_m"], target["bucket_xy_m"], ident+"/bucket translated with board", 1e-12)
        checks.equal(actual["translation_m"], target["translation_xy_m"], ident+"/fixed XY offset")
        for key in ("source_record_id", "source_record_sha256"):
            checks.equal(actual[key], seed[key], ident+"/exact original source/"+key)
        if index == 252 and actual["eligible"]:
            for key in ("q10", "q14"):
                checks.equal(actual[key], seed[key], "Unshifted reference preserves historical branch/"+key)


def audit_refinement(groups, selected, checks):
    checks.equal(len({target_key(row) for row in selected}), len(selected), "All refinement seeds are distinct geometries")
    checks.equal(len(groups), 27*len(selected), "All27 local proposals per unique seed retained")
    for seed_index, source in enumerate(selected):
        for index, (actual, values) in enumerate(zip(groups[27*seed_index:27*(seed_index+1)], local_inputs(source))):
            i, j, k = np.unravel_index(index, (3, 3, 3))
            ident = f"refine-s{seed_index:02d}-h{i}t{j}g{k}"
            checks.equal(actual["id"], ident, "Literal refinement offset identity")
            checks.equal(actual["role"], "discovery", ident+"/development role")
            checks.equal(actual["phase"], "refinement", ident+"/refinement phase")
            checks.equal(actual["query_kind"], "local_refinement", ident+"/refinement type")
            checks.numeric(actual["values"], values, ident+"/literal narrow offsets", 1e-12)
            checks.equal(actual["bucket_xy_m"], source["bucket_xy_m"], ident+"/fixed local bucket")
            checks.equal(actual["translation_m"], source["translation_m"], ident+"/fixed translated plane")
            checks.equal(actual["source_record_id"], source["id"], ident+"/preceding observed source")
            checks.equal(actual["source_record_sha256"], digest(source), ident+"/exact preceding record")
            checks.equal(actual["source_query_fingerprint"], source["query_fingerprint"], ident+"/explicit query ancestry")


def audit_verification(groups, selected, checks):
    checks.equal(len({target_key(row) for row in selected}), len(selected), "Verification targets are distinct geometries")
    clean_audit.audit_verification(groups, selected, checks)
    for group, source in zip(groups, selected):
        checks.equal(group["phase"], "verification", group["id"]+"/verification phase")
        checks.equal(group["query_kind"], "nominal_verification", group["id"]+"/nominal exact query")
        checks.equal(group["translation_m"], source["translation_m"], group["id"]+"/source plane offset")
        checks.equal(group["source_query_fingerprint"], source["query_fingerprint"], group["id"]+"/source query identity")


def audit_fingerprints(spec, report, groups, rows, seed, checks):
    scene_fp = scene_fingerprint(spec["scene"], spec["calibration"], spec["asset_sha256"])
    checks.equal(spec["scene_fingerprint"], scene_fp, "Static scene includes actual geometry/material/calibration identity")
    checks.equal(report["scene_fingerprint"], scene_fp, "Report static scene identity")
    original_delta = np.asarray(seed["bucket_xy_m"])-seed["values"][:2]
    for group in groups:
        label = group["id"]
        checks.equal(group["query_fingerprint"], query_fingerprint(group, scene_fp), label+"/complete geometric and provenance query")
        if not group["role"].endswith("_control"):
            actual_delta = np.asarray(group["bucket_xy_m"])-group["values"][:2]
            checks.numeric(actual_delta, original_delta, label+"/translated fixed bearing", 1e-12)
            checks.numeric(float(np.linalg.norm(actual_delta)), .65, label+"/650mm distance", 1e-12)
    execution_ids = []
    for row in rows:
        if row["native_release"]:
            expected = execution_fingerprint(row)
            checks.equal(row["execution_fingerprint"], expected, row["id"]+"/exact requested and measured execution identity")
            execution_ids.append(expected)
        else:
            checks.equal(row["execution_fingerprint"], None, row["id"]+"/unreleased query has no execution identity")
    checks.equal(len(set(execution_ids)), len(execution_ids), "Every actual execution identity is unique")


def audit_settings(settings, checks):
    expected = dict(translations=dict(dx_m=list(DX), dy_m=list(DY), rule=DISTANCE_RULE),
                    grid=dict(heights_m=list(HEIGHTS), tilts_deg=list(TILTS), gaps_m=list(GAPS), reference=[.38, 28., .3868]),
                    coarse_requests=253, translated_grid_requests=243, translated_reference_requests=9, unshifted_reference_requests=1,
                    refinement_seeds=3, refinement_max_requests=81,
                    refinement_offsets=dict(zip(("height_m", "tilt_deg", "gap_m"), map(list, OFFSETS))),
                    verification_sources=5, verification_dedup_decimals=9, refinement_dedup_decimals=9,
                    repeats_per_resolution=3, substeps=list(STEPS), max_requests=MAX_REQUESTS,
                    minimum_clearance_m=.005, max_drift_350ms_m=.005, first_flight_version=METRIC_VERSION)
    for key, value in expected.items():
        checks.equal(settings[key], value, "Literal frozen XY protocol/"+key)
    checks.require("noise" in settings["verification_scope"].lower() and "no" in settings["verification_scope"].lower(),
                   "Noise robustness explicitly excluded")


def audit_group_metadata(actual, group, scene, checks, label):
    clean_audit.audit_group_metadata(actual, group, scene, checks, label)
    for key in ("phase", "translation_m", "query_kind", "query_fingerprint", "source_query_fingerprint"):
        if key in group:
            checks.equal(actual[key], group[key], label+"/explicit query metadata/"+key)


def audit_report(report, rows, groups, verification_groups, sources, positions, pre, post, scene, checks):
    stat = clean_audit.audit_statistics
    stat(report["counts"], rows, checks, "All requested XY shots")
    for field, key in (("phase_counts", "phase"), ("substep_counts", "substeps")):
        checks.equal(report["counts"][field], dict(Counter(str(row[key]) for row in rows)), "Complete XY accounting/"+field)
    checks.equal(report["control_gates"], dict(controls_pre=pre, controls_post=post), "Both independent control gates")
    checks.equal(len(report["groups"]), len(groups), "Every requested geometry group displayed")
    for actual, group in zip(report["groups"], groups):
        audit_group_metadata(actual, group, scene, checks, group["id"])
        for step in STEPS:
            stat(actual["by_substeps"][str(step)], [row for row in rows if row["group_id"] == group["id"] and row["substeps"] == step],
                 checks, group["id"]+"/actual outcomes/"+str(step))
    keys = [(phase, group["id"], group["role"], step)
            for phase in ("controls_pre", "controls_post") for group in groups[:2] for step in STEPS]
    checks.equal([(row["phase"], row["group_id"], row["role"], row["substeps"]) for row in report["controls"]],
                 keys, "All control phases, groups and resolutions visible")
    for actual in report["controls"]:
        cohort = [row for row in rows if all(row[key] == actual[key] for key in ("phase", "group_id", "substeps"))]
        stat(actual["stats"], cohort, checks, "Exact reported control cohort")
    development = [row for row in rows if row["phase"] in ("coarse", "refinement")]
    top = select(development, 10)
    checks.equal([row["id"] for row in report["top_settings"]], [row["id"] for row in top], "Top10 independently ranked first-flight development records")
    lookup = {group["id"]: group for group in groups}
    for actual, source in zip(report["top_settings"], top):
        group = lookup[source["group_id"]]
        audit_group_metadata(dict(actual, id=group["id"]), group, scene, checks, source["id"])
        checks.equal(actual["group_id"], group["id"], source["id"]+"/actual source group")
        checks.equal(actual["substeps"], source["substeps"], source["id"]+"/development resolution")
        stat(actual["stats"], [source], checks, source["id"]+"/single recorded observation")
        for key in ("minimum_mouth_clearance_mm", "closest_opening_distance_m", "execution_fingerprint"):
            checks.equal(actual[key], source.get(key), source["id"]+"/actual metric/"+key)
        same(actual["first_flight"], source["first_flight"], checks, source["id"]+"/actual diagnostic")
    checks.equal(len(report["verification"]), len(verification_groups), "Every selected verification target reported")
    qualified = 0
    for actual, group, source in zip(report["verification"], verification_groups, sources):
        audit_group_metadata(actual, group, scene, checks, group["id"])
        cohort = [row for row in rows if row["group_id"] == group["id"]]
        for step in STEPS:
            stat(actual["by_substeps"][str(step)], [row for row in cohort if row["substeps"] == step], checks,
                 group["id"]+"/matched repeat outcomes/"+str(step))
        replay.check_pairing(cohort, checks)
        checks.equal(actual["sensitivity"], replay.paired_sensitivity(cohort, positions), group["id"]+"/matched initialized controls and drift")
        result = qualify(cohort, positions, controls_passed=pre["status"] == post["status"] == "passed", source_clean=clean(source))
        checks.equal(actual["qualification"], result, group["id"]+"/independent nominal clean qualification")
        qualified += int(result["qualified"])
    status = "blocked_controls" if pre["status"] != "passed" or post["status"] != "passed" else "passed" if qualified else "failed"
    gate = dict(status=status, controls_pre_passed=pre["status"] == "passed", controls_post_passed=post["status"] == "passed",
                qualified_settings=qualified, more_ppo_started=False)
    checks.equal(report["gate"], gate, "Final independently reconstructed XY qualification")
    return gate


def audit_traces(study, report, rows, groups, scene, checks):
    lookup = {group["id"]: group for group in groups}
    expected = [row for row in rows if row["phase"] == "controls_pre" and row["repeat_index"] == 0 and row["native_release"]]
    expected += select([row for row in rows if row["phase"] in ("coarse", "refinement")], 3)
    expected += [row for row in rows if row["phase"] == "verification" and row["repeat_index"] == 0 and row["native_release"]]
    checks.equal([trace["record_id"] for trace in report["trace_examples"]], [row["id"] for row in expected], "Only prespecified illustrative native records")
    checks.require(len(expected) <= 17, "At most17 illustrative actual native trajectories")
    for trace, row in zip(report["trace_examples"], expected):
        group = lookup[row["group_id"]]
        for key in ("group_id", "role", "substeps", "whole_ball_entry", "max_retention_s", "external_contact_steps", "route_success",
                    "route_quality", "rim_band_contact", "invalid", "invalid_physics", "scalar_score_audit_passed", "execution_fingerprint"):
            checks.equal(trace[key], row[key], row["id"]+"/actual illustrated native field/"+key)
        for key in ("translation_m", "query_kind", "query_fingerprint", "source_query_fingerprint"):
            if key in group:
                checks.equal(trace[key], group[key], row["id"]+"/illustrated query provenance/"+key)
        values = group["values"]
        for key, value in dict(board_xy_m=values[:2], bucket_xy_m=group["bucket_xy_m"], board_height_m=values[2], tilt_deg=values[3], gap_m=values[4],
                               release_center_height_m=values[2]+values[4]+scene["ball"]["radius_m"],
                               distance_m=float(np.linalg.norm(np.asarray(group["bucket_xy_m"])-values[:2]))).items():
            checks.equal(trace[key], value, row["id"]+"/actual illustrated geometry/"+key)
        same(trace["first_flight"], row["first_flight"], checks, row["id"]+"/illustrated measured first flight")
        with np.load(contained(study, row["native_directory"])/row["native_cache_file"], allow_pickle=False) as archive:
            points = archive["position_m"][::row["substeps"]*4, row["native_cache_row"]]
        checks.numeric(trace["position_m"], points, row["id"]+"/exact saved native illustrated positions", 0)
        checks.numeric(trace["sample_dt_s"], 1/60, row["id"]+"/actual illustrated cadence", 0)
        checks.require(bool(trace.get("selection_scope")), "Illustration selection and scope explicit")


def audit(study, source, assets, *, shallow_study, positive_study, positive_audit, replay_study, coarse_study, raised_study):
    started, checks = time.monotonic(), AuditChecks()
    study, source, assets = map(Path, (study, source, assets))
    manifest = replay.check_manifest(study, checks)
    spec, report = read(study/"contract.json"), read(study/"report.json")
    unsigned = dict(spec);fingerprint = unsigned.pop("fingerprint", None)
    checks.equal(digest(unsigned), fingerprint, "Complete frozen XY contract identity")
    checks.equal(spec["version"], STUDY_VERSION, "Explicit bounded translated-plane protocol")
    checks.equal(report["version"], STUDY_VERSION, "New result version remains separate from historical studies")
    checks.equal(report["contract_fingerprint"], fingerprint, "Report exact complete contract")
    checks.equal((report["status"], report["phase"]), ("completed", "completed"), "Finite development search completed")
    checks.equal(spec["evidence_kind"], "native_calibrated_xy650_sweep", "Actual calibrated native shot evidence")
    checks.equal(report["evidence_sha256"], manifest, "Report binds every native artifact")
    replay.check_source_files(source, spec["source_sha256"], checks, "Immutable actual native source")
    paths = list((source/"sircus/mjlab").glob("*.py")) + [source/"sircus/mjlab/requirements.txt"]
    paths += [source/"sircus/isaac"/name for name in ("geometry.py", "rl_contract.py", "entry_quality.py", "board_route.py")]
    checks.equal(set(spec["source_sha256"]), {str(path.relative_to(source)) for path in paths}, "All actual execution and audit modules declared")
    from .learning_solve import required_asset_paths
    actual_assets = {name: sha(contained(assets.parent, name)) for name in required_asset_paths(assets)}
    checks.equal(spec["asset_sha256"], actual_assets, "Exact actual collision assets")
    replay.audit_packages(spec["packages"], checks)
    calibration = replay.check_calibration(study, spec["calibration"], checks)
    parent, originals = replay.audit_input_sources(shallow_study, positive_study, positive_audit,
                                                 spec["input_bindings"], checks, spec["calibration"])
    prior, _ = clean_audit.audit_prior_replay(Path(replay_study), spec["input_bindings"]["replay"], checks)
    coarse_parent = clean_audit.audit_coarse_parent(Path(coarse_study), spec["input_bindings"]["coarse"], checks)
    raised_parent = audit_raised_parent(Path(raised_study), spec["input_bindings"]["raised"], checks)
    for name, ancestor in (("original", parent), ("matched replay", prior), ("coarse", coarse_parent), ("raised", raised_parent)):
        for key in ("scene", "calibration", "packages", "asset_sha256"):
            checks.equal(spec[key], ancestor[key], "Unchanged static physical model and calibration/"+name+"/"+key)
    frozen = {"initial-groups.json", "initial-geometry.json", "initial-plan.json", "calibration.json", "verification.json"}
    checks.equal(set(spec["frozen_files_sha256"]), frozen, "Every pre-outcome input declared")
    for name, expected in spec["frozen_files_sha256"].items():
        checks.equal(sha(contained(study, name)), expected, "Frozen original pre-outcome bytes/"+name)
    audit_settings(spec["settings"], checks)
    checks.equal(report["settings"], spec["settings"], "Reported exact prespecified protocol")
    seed = next(group for group in originals if group["role"] == "candidate")
    controls = [group for group in originals if group["role"].endswith("_control")]
    initial, initial_proofs = read(study/"initial-groups.json"), read(study/"initial-geometry.json")
    audit_initial(initial, seed, controls, checks)
    before = make_plan(initial[:2], "controls_pre", steps=STEPS, repeats=3)
    coarse_plan = make_plan(initial[2:], "coarse", steps=(64,), repeats=1)
    after = make_plan(initial[:2], "controls_post", steps=STEPS, repeats=3)
    checks.equal(read(study/"initial-plan.json"), dict(controls_pre=before, coarse=coarse_plan, controls_post=after), "Literal complete initial and final control/coarse plans")
    groups, proofs = read(study/"groups.json"), read(study/"geometry.json")
    checks.equal(groups[:255], initial, "Initial geometry queries never changed after results")
    checks.equal(proofs[:255], initial_proofs, "Initial endpoint proofs never changed after results")
    clean_audit.audit_geometry(groups, proofs, spec["scene"], assets, checks)
    rows = replay.read_records(study/"records.jsonl")
    checks.require(12 <= len(rows) <= MAX_REQUESTS, "Finite388-request maximum including every rejection")
    checks.equal([row["id"] for row in rows[:12]], [row["id"] for row in before], "Controls precede all translated development shots")
    pre = control_gate(rows[:12])
    checks.equal(read(study/"initial-control-gate.json"), pre, "Independently reconstructed initial control gate")
    selected = select([row for row in rows if row["phase"] == "coarse"], 3, unique=True) if pre["status"] == "passed" else []
    refined = read(study/"refinement-groups.json")
    audit_refinement(refined, selected, checks)
    refinement_plan = make_plan(refined, "refinement", steps=(64,), repeats=1)
    verification_sources = select([row for row in rows if row["phase"] in ("coarse", "refinement")], 5, unique=True) if pre["status"] == "passed" else []
    verified = read(study/"verification-groups.json")
    audit_verification(verified, verification_sources, checks)
    verification_plan = make_plan(verified, "verification", steps=STEPS, repeats=3)
    checks.equal(groups, initial+refined+verified, "Final groups exactly concatenate the opened phases")
    stages = read(study/"adaptive-stages.json")
    checks.equal(len(stages), 2, "Exactly two frozen adaptive stages; no rescue iterations")
    prefix = rows[:12+len(coarse_plan)] if pre["status"] == "passed" else rows[:12]
    clean_audit.audit_adaptive_stage(study, stages, 0, "refinement", refined, selected, refinement_plan, prefix, proofs, checks, initial_count=255)
    prefix = rows[:12+len(coarse_plan)+len(refinement_plan)] if pre["status"] == "passed" else rows[:12]
    clean_audit.audit_adaptive_stage(study, stages, 1, "verification", verified, verification_sources, verification_plan, prefix, proofs, checks, initial_count=255)
    expected = before+coarse_plan+refinement_plan+verification_plan+after if pre["status"] == "passed" else before
    clean_audit.audit_native_rows(rows, expected, groups, checks)
    audit_fingerprints(spec, report, groups, rows, seed, checks)
    replay.audit_batches(study, rows, checks)
    post = control_gate([row for row in rows if row["phase"] == "controls_post"])
    if pre["status"] != "passed":
        post["status"] = "not_run"
    checks.equal(read(study/"final-control-gate.json"), post, "Final independent controls retained")
    actual_steps = {row["substeps"] for row in rows if row["native_release"]}
    checks.require(actual_steps in (set(), set(STEPS)), "Both native resolutions present, or wholly rejected controls")
    if actual_steps:
        options = read(study/"effective-options.json")
        replay.audit_options(options, checks)
        replay.audit_compiled(study, spec["scene"], assets, calibration, options, checks)
        positions, traces = replay.audit_native(study, rows, spec["scene"], calibration, assets, checks)
    else:
        checks.require(not (study/"effective-options.json").exists(), "No invented native options")
        checks.require(not list((study/"native").rglob("*.npz")), "No invented native caches")
        positions, traces = {}, {}
    audit_first_flight(study, rows, spec["scene"], calibration, checks)
    gate = audit_report(report, rows, groups, verified, verification_sources, positions, pre, post, spec["scene"], checks)
    audit_traces(study, report, rows, groups, spec["scene"], checks)
    identity = auditor_identity(source, spec["source_sha256"], checks)
    return dict(version=VERSION, status="passed", contract_fingerprint=fingerprint, scene_fingerprint=spec["scene_fingerprint"],
                checks_count=checks.count, checks=checks.entries, failures=[], auditor_sha256=sha(Path(__file__)),
                report_sha256=sha(study/"report.json"), evidence_manifest_sha256=sha(study/"evidence-manifest.json"),
                auditor_identity=identity, producer_source_sha256=spec["source_sha256"], requested_trials=len(rows),
                native_releases=len(traces), gate=gate, elapsed_s=time.monotonic()-started,
                limitations=["Finite translated-plane development queries, not a learned policy or held-out workspace coverage.",
                             "First-flight aperture margin is a force-cutoff diagnostic, not a capture/clean-entry label.",
                             "Nominal held-endpoint repeats only; no noise, grasp/transit, actuator, perception or physical validation."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "source", "assets", "shallow-study", "positive-study", "positive-audit", "replay-study", "coarse-study", "raised-study", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.study.resolve()) and args.output.name != "audit-xy650.json":
        raise ValueError("Only designated new audit may be written inside frozen evidence")
    try:
        result = audit(args.study, args.source, args.assets, shallow_study=args.shallow_study, positive_study=args.positive_study,
                       positive_audit=args.positive_audit, replay_study=args.replay_study, coarse_study=args.coarse_study, raised_study=args.raised_study)
    except (EvidenceError, ValueError, KeyError, OSError) as error:
        result = dict(version=VERSION, status="failed", failures=[str(error)])
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
        raise
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({key: result[key] for key in ("status", "checks_count", "requested_trials", "native_releases", "gate")}))


if __name__ == "__main__":
    main()
