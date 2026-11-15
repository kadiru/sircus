"""Estimate simple rebound coefficients from audited, disjoint native traces."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from .benchmark_verify import contained, digest, read, sha
from .calibrated_score import FORCE_NAMES, FORCE_THRESHOLD_N, REBOUND_NORMAL_SPEED_M_S, _trace
from .intergrid_candidate_replay_plan import FLIGHT_MISSES, VERSION as REPLAY_VERSION, command_key
from .intergrid_route_fallback_audit import VERSION as FALLBACK_AUDIT_VERSION
from .intergrid_route_fallback_study import VERSION as FALLBACK_VERSION
from .intergrid_route_refit_inputs import _audited_source


VERSION = "sircus-intergrid-rebound-calibration-v1"
FREE_WINDOW_S = 0.002
GRAVITY = np.array([0.0, 0.0, -9.81])
ARCHIVE_ARRAYS = ("record_ids", "position_m", "velocity_m_s", "forces_N", "time_s", "dt_s")
SURFACES = ("board", "table")
COEFFICIENTS = ("normal_restitution", "tangent_retention")


def _impact(times, velocity, contacts, surface, start, end, normal):
    """Measure velocity on both sides of a single-surface force interval."""
    if contacts[start:end].sum(axis=1).max() != 1 or not contacts[start:end, surface].all():
        return None
    before = int(np.searchsorted(times, times[start] - FREE_WINDOW_S, side="right") - 1)
    after = int(np.searchsorted(times, times[end] + FREE_WINDOW_S, side="left"))
    if before < 0 or after >= len(times) or contacts[before:start].any() or contacts[end:after].any():
        return None
    impact_time = (times[start] + times[end]) / 2
    incoming = np.median(
        velocity[before:start + 1] + (impact_time - times[before:start + 1, None]) * GRAVITY,
        axis=0,
    )
    outgoing = np.median(
        velocity[end:after + 1] + (impact_time - times[end:after + 1, None]) * GRAVITY,
        axis=0,
    )
    normal_in, normal_out = float(incoming @ normal), float(outgoing @ normal)
    if normal_in >= -REBOUND_NORMAL_SPEED_M_S or normal_out <= REBOUND_NORMAL_SPEED_M_S:
        return None
    tangent_in, tangent_out = incoming - normal_in * normal, outgoing - normal_out * normal
    tangent_sq = float(tangent_in @ tangent_in)
    return dict(start=start, end=end, normal_restitution=-normal_out / normal_in,
                tangent_retention=(float(tangent_out @ tangent_in) / tangent_sq
                                   if tangent_sq > REBOUND_NORMAL_SPEED_M_S**2 else None))


def _trace_impacts(row, arrays):
    index, steps = row["native_cache_row"], row["physics_steps"]
    if (type(index) is not int or not 0 <= index < len(arrays["record_ids"])
            or arrays["record_ids"][index] != row["id"] or type(steps) is not int or steps < 1):
        raise ValueError("Native rebound trace differs from its row identity")
    _, velocity, force, times, _ = _trace(dict(
        position_m=arrays["position_m"][:steps + 1, index],
        velocity_m_s=arrays["velocity_m_s"][:steps + 1, index],
        forces_N=arrays["forces_N"][:steps, index],
        time_s=arrays["time_s"][:steps + 1], dt_s=arrays["dt_s"],
    ))
    if len(force) != steps:
        raise ValueError("Native rebound trace does not cover its scoring interval")
    contacts = force > FORCE_THRESHOLD_N
    edges = np.flatnonzero(np.diff(np.r_[False, contacts.any(axis=1), False]))
    normal = np.asarray(row["board_normal"], dtype=float)
    if normal.shape != (3,) or not np.isclose(np.linalg.norm(normal), 1.0, atol=1e-6, rtol=0):
        raise ValueError("Audited trace requires a unit board normal")
    normal /= np.linalg.norm(normal)
    result = {}
    for name, axis, pair in zip(SURFACES, (normal, np.array([0., 0., 1.])),
                                zip(edges[::2], edges[1::2], strict=True)):
        start, end = map(int, pair)
        impact = _impact(times, velocity, contacts, FORCE_NAMES.index(name), start, end, axis)
        if impact is None:
            break
        result[name] = impact
    return result


def _summarize(samples, surface, coefficient):
    by_position = defaultdict(list)
    for sample in samples:
        value = sample["impacts"].get(surface, {}).get(coefficient)
        if value is not None:
            by_position[sample["position"]].append(value)
    medians = [float(np.median(values)) for values in by_position.values()]
    if not medians:
        raise ValueError(f"No independent native samples for {surface} {coefficient}")
    value = float(np.median(medians))
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"Invalid fitted coefficient for {surface} {coefficient}")
    return value, dict(traces=sum(map(len, by_position.values())),
                       positions=len(by_position),
                       position_median_quartiles=np.quantile(medians, [0.25, 0.5, 0.75]).tolist())


def fit_rebound_calibration(fallback_study, fallback_audit, replay_plan):
    """Fit median effective rebound coefficients without using any replay shot."""
    study = Path(fallback_study)
    report, manifest, audit, binding, _ = _audited_source(
        study, Path(fallback_audit), FALLBACK_VERSION, ("rows.jsonl", "layouts.json"),
    )
    if audit["version"] != FALLBACK_AUDIT_VERSION or audit["test_releases"] != report["test_releases"]:
        raise ValueError("Independent route fallback audit required for rebound calibration")
    layouts = {row["id"]: row["bucket_xy_m"] for row in read(study / "layouts.json")["layouts"]}
    excluded = {case["layout"]["id"]: case["layout"]["bucket_xy_m"] for case in replay_plan["cases"]}
    if (replay_plan["version"] != REPLAY_VERSION or len(excluded) != FLIGHT_MISSES
            or any(layouts.get(name) != xy for name, xy in excluded.items())):
        raise ValueError("Every measured replay position must be excluded from calibration")
    excluded_xy = {tuple(xy) for xy in excluded.values()}
    packets, commands = defaultdict(list), set()
    released = 0
    with (study / "rows.jsonl").open() as stream:
        for row in map(json.loads, stream):
            if row["phase"] != "test" or row["native_release"] is not True:
                continue
            released += 1
            if (row["bucket_xy_m"] != layouts[row["context_id"]] or row["role"] != "test"
                    or row["adapter_directory"] != "native-heading"):
                raise ValueError("Native rebound row does not bind an audited layout and engine")
            if (row["context_id"] in excluded or tuple(row["bucket_xy_m"]) in excluded_xy
                    or row["invalid_physics"] is not False or row["scalar_score_audit_passed"] is not True):
                continue
            key = command_key(row)
            if key in commands:
                continue
            commands.add(key)
            packets[f"native-heading/{row['native_cache_file']}"].append(row)
    if released != audit["test_releases"]:
        raise ValueError("Native rebound calibration ledger release count differs")

    samples, trace_hashes = [], {}
    for name, rows in sorted(packets.items()):
        path = contained(study, name)
        trace_hashes[name] = sha(path)
        if {row["native_cache_sha256"] for row in rows} != {trace_hashes[name]} or manifest[name] != trace_hashes[name]:
            raise ValueError("Native rebound trace differs from audited bytes")
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in ARCHIVE_ARRAYS}
        for row in rows:
            samples.append(dict(id=row["id"], position=row["context_id"], impacts=_trace_impacts(row, arrays)))

    coefficients, counts = {}, {}
    for surface in SURFACES:
        for coefficient in COEFFICIENTS:
            key = f"{surface}_{coefficient}"
            coefficients[key], counts[key] = _summarize(samples, surface, coefficient)
    result = dict(version=VERSION, **coefficients, counts=counts,
                  source=binding, excluded_positions=sorted(excluded),
                  unique_source_traces=len(samples), unique_source_positions=len({row["position"] for row in samples}),
                  trace_manifest_sha256=digest(trace_hashes), samples_sha256=digest(samples),
                  free_window_s=FREE_WINDOW_S, gravity_m_s2=GRAVITY.tolist(),
                  method="Median within position, then median across positions. The first contact must be board "
                         "and the next contact must be table to contribute a table rebound. Two milliseconds "
                         "of free flight on either side are projected to the midpoint of the isolated contact.",
                  measured_geometry_model=True)
    result["fingerprint"] = digest(result)
    return result
