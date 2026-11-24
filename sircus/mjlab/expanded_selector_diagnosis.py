"""Diagnose eight audited saved flights; no simulation steps or policy changes.

Force intervals establish measured contact categories. Optional CPU collision
reconstruction names possible robot links, separately from those measurements.
Interpolated rim crossings describe the observed flight, not a counterfactual
corrected shot or proof that one changed parameter caused the paired outcome.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

from .benchmark_verify import Checks, contained, digest, fingerprint, read, sha
from .calibrated_score import FORCE_NAMES, FORCE_THRESHOLD_N
from .hybrid_miss_diagnosis import episodes

VERSION = "sircus-expanded-selector-saved-diagnosis-v1"
CASE_NUMBERS = (4, 10, 19, 47)
METHODS = ("current", "expanded")
SCORE_FIELDS = (
    "route_success", "route_class", "board_first_confirmed", "board_first_status",
    "first_board_contact_start_s", "first_board_rebound_s", "whole_ball_entry",
    "first_mouth_passage_time_s", "whole_ball_entry_time_s", "capture_time_s",
    "max_retention_s", "retention_resets", "exited_after_entry",
    "closest_opening_distance_m", "minimum_mouth_clearance_mm", "rim_band_contact",
    "bucket_contact_before_entry", "invalid_physics", "route_failure_reason",
    "board_contact_episodes", "table_contact_steps", "robot_contact_steps", "floor_contact_steps",
    "unattributed_contact_steps", "physics_steps", "scoring_version",
)


def rim_crossings(times, positions, bucket_xy, board_center, geometry, *, after_s):
    """Interpolate descending center-at-rim crossings from adjacent saved states.

    Clearance uses the full ball radius at the center/rim plane. A negative
    value is overlap with the aperture boundary at this crossing, not a failure
    label: rim-assisted captures and later entries remain possible.
    """
    t, p, bucket, board = map(np.asarray, (times, positions, bucket_xy, board_center))
    if (t.ndim != 1 or len(t) < 2 or p.shape != (len(t), 3) or bucket.shape != (2,)
            or board.shape != (3,) or not all(np.isfinite(x).all() for x in (t, p, bucket, board))
            or np.any(np.diff(t) <= 0) or not math.isfinite(after_s)):
        raise ValueError("Finite, aligned observed states required")
    rim, radius = geometry["rim_m"], geometry["ball_radius_m"]
    planes = np.asarray(geometry["inward_rim_halfplanes"], float)
    direction = bucket - board[:2]
    if np.linalg.norm(direction) <= 1e-10 or planes.ndim != 2 or planes.shape[1] != 3:
        raise ValueError("Defined board-to-bucket axis and rim planes required")
    direction /= np.linalg.norm(direction)
    lateral = np.array([-direction[1], direction[0]])
    found = []
    for i in np.flatnonzero((p[:-1, 2] >= rim) & (p[1:, 2] < rim)):
        alpha = float((p[i, 2] - rim) / (p[i, 2] - p[i+1, 2]))
        when = float(t[i] + alpha * (t[i+1] - t[i]))
        if when < after_s:
            continue
        point = p[i] + alpha * (p[i+1] - p[i])
        error = point[:2] - bucket
        found.append(dict(time_s=when, interval_start_tick=int(i), alpha=alpha,
            position_m=point.tolist(), along_error_m=float(error @ direction),
            lateral_error_m=float(error @ lateral), radial_error_m=float(np.linalg.norm(error)),
            aperture_clearance_m=float(np.min(planes[:, :2] @ error + planes[:, 2]) - radius)))
    return found


def setting_delta(current, expanded):
    """Expanded minus current; raw joint differences preserve branch changes."""
    vector_keys = ("command_board_center_m", "board_center_m", "release_ball_center_m", "q10_rad", "q14_rad")
    result = {key: (np.asarray(expanded[key]) - np.asarray(current[key])).tolist() for key in vector_keys}
    for key in ("board_height_m", "tilt_deg", "drop_gap_m", "heading_offset_deg", "twist_deg", "board_bucket_distance_m"):
        result[key] = float(expanded[key] - current[key])
    result["joint_delta_scope"] = "Literal held-joint subtraction in radians, without wrapping revolute joints"
    return result


def verify_parent(study, audit_path, checks):
    """Bind every artifact consumed below to the unchanged independently audited parent."""
    report, audit = read(study / "report.json"), read(audit_path)
    checks.equal((report["status"], audit["status"]), ("completed", "passed"), "Completed audited source")
    checks.equal(audit["version"], "sircus-expanded-selector-independent-audit-v1", "Expected source audit")
    checks.equal(audit["report_sha256"], sha(study / "report.json"), "Audited report binding")
    manifest_hash = sha(study / "evidence-manifest.json")
    checks.equal(audit["evidence_manifest_sha256"], manifest_hash, "Audited manifest binding")
    checks.equal(report["evidence_manifest_sha256"], manifest_hash, "Report manifest binding")
    manifest, verified = read(study / "evidence-manifest.json"), {}

    def bound(name):
        path = contained(study, name)
        if name not in verified:
            checks.equal(sha(path), manifest[name], "Bound artifact " + name)
            verified[name] = manifest[name]
        return path

    for name, key in (("contract.json", "contract_sha256"), ("choices.json", "choices_sha256")):
        checks.equal(sha(bound(name)), audit[key], "Audited " + name)
    return report, audit, bound, verified


def reconstructed_contacts(gate, row, t, p, f, events, checks):
    """CPU geometry only; the force channel does not contain measured link IDs."""
    from .link_clearance_diagnosis import attribution
    gate.colliders(row["native_initial_q10"], row["native_initial_q14"], row["bucket_xy_m"])
    body = gate.m.body("Arm14/impact_board").id
    checks.numeric(gate.d.xpos[body], row["board_center_m"], "Reconstructed board center", 2e-6)
    checks.numeric(gate.d.xmat[body].reshape(3, 3)[:, 2], row["board_normal"], "Reconstructed board normal", 2e-6)
    checks.require(row["max_integrated_joint_excursion_rad"] < 1e-4, "Held endpoint reconstruction applicable")
    ball_id = gate.m.geom("ball/geom").id
    robot_events = [event for event in events if event["channel"] == "robot"]
    results = []
    for event in robot_events:
        channel = FORCE_NAMES.index("robot")
        peak = event["start_tick"] + int(np.argmax(f[event["start_tick"]:event["end_tick"], channel]))
        ticks = sorted({event["start_tick"], min(event["start_tick"]+1, len(t)-1), peak})
        samples = []
        for tick in ticks:
            gate.d.qpos[gate.ball:gate.ball+3] = p[tick]
            gate.mj.mj_kinematics(gate.m, gate.d)
            distances = [(float(gate.mj.mj_geomDistance(gate.m, gate.d, ball_id, item["index"], 10., None)),
                          item["name"]) for item in gate.local]
            samples.append(dict(tick=tick, time_s=float(t[tick]), **attribution(distances)))
        results.append(dict(measured_interval_start_s=event["start_s"], samples=samples))
    return results


def measured_flight(t, p, v, f, row, score, geom, events):
    crossings = rim_crossings(t, p, row["bucket_xy_m"], row["board_center_m"], geom,
                             after_s=score["first_board_rebound_s"] or 0.)
    for crossing in crossings:
        previous = [e for e in events if e["channel"] != "board" and e["end_s"] <= crossing["time_s"]]
        crossing["previous_nonboard_contact"] = max(previous, key=lambda e: e["end_s"])["channel"] if previous else None
    rebound = score["first_board_rebound_s"]
    tick = int(np.searchsorted(t, rebound)) if rebound is not None else None
    bucket_contact = next((e for e in events if e["channel"] == "bucket"), None)
    first_table = next((e for e in events if e["channel"] == "table"), None)
    bucket_geometry = None
    if bucket_contact:
        point = np.asarray(bucket_contact["position_m"])
        direction = np.asarray(row["bucket_xy_m"]) - np.asarray(row["board_center_m"][:2])
        direction /= np.linalg.norm(direction)
        offset = point[:2] - row["bucket_xy_m"]
        bucket_geometry = dict(radial_center_distance_m=float(np.linalg.norm(offset)),
            along_center_error_m=float(offset @ direction),
            ball_center_above_rim_m=float(point[2] - geom["rim_m"]),
            ball_top_above_rim_m=float(point[2] + geom["ball_radius_m"] - geom["rim_m"]),
            vertical_velocity_before_m_s=bucket_contact["velocity_before_m_s"][2],
            ascending=bucket_contact["velocity_before_m_s"][2] > 0.,
            near_side=float(offset @ direction) < 0.,
            scope="Observed ball center and velocity at first measured bucket-force interval; does not identify a specific bucket wall/rim collider")
    return dict(descending_center_rim_crossings=crossings,
        first_descending_center_rim=crossings[0] if crossings else None,
        first_descending_center_rim_after_table=next((x for x in crossings
            if first_table and x["time_s"] >= first_table["end_s"]), None),
        measured_rebound_state=None if tick is None else dict(time_s=float(t[tick]),
            position_m=p[tick].tolist(), velocity_m_s=v[tick].tolist()),
        first_bucket_contact=bucket_contact, first_bucket_contact_geometry=bucket_geometry,
        first_table_contact=first_table,
        first_robot_contact=next((e for e in events if e["channel"] == "robot"), None),
        terminal=dict(time_s=float(t[-1]), position_m=p[-1].tolist()),
        crossing_scope="Linear interpolation between native states; positive along error is beyond bucket, negative is short. The first crossing can intentionally be short before a table rebound; it is not a failure criterion. Aperture clearance is descriptive geometry, not the capture scorer.")


def selection_details(site, slot, request, checkpoint, checks):
    from .neural_shot_model import predict
    matches = [p for stage in site["plan"].values() for p in stage["proposals"]
               if p["id"] == request["proposal_id"]]
    checks.equal(len(matches), 1, "One authenticated selected pre-IK proposal")
    proposal = matches[0]
    ranked = slot["ranked_proposal_ids"]
    source = request.get("source_hint", {})
    return dict(proposal_id=proposal["id"], proposal_rank=ranked.index(proposal["id"])+1 if proposal["id"] in ranked else None,
        checked_proposals=slot["checked_proposals"], candidate_pool=slot["candidate_pool"],
        source_id=request.get("training_recipe_id", request.get("source_teacher_id")),
        source_teacher_id=request.get("source_teacher_id"), source_setting_key=request.get("source_setting_key"),
        source_bucket_xy_m=request.get("training_bucket_xy_m", source.get("bucket_xy_m")),
        source_distance_m=request.get("training_distance_m"), source_observed_route=request.get("source_observed_route"),
        intended_route=request["intended_route"], probabilities=predict(checkpoint, proposal),
        probability_use="used_for_neural_proposal_order" if proposal["id"] in ranked else "posthoc_score_of_unranked_joint_alternative",
        probability_scope="Recomputed from frozen checkpoint and authenticated original pre-IK proposal; not calibrated guarantees",
        prediction=slot["prediction"], predicted_flight_risk=slot["predicted_flight_risk"],
        decision_s=site["decision_latency_s"], request_s=site["request_wall_s"],
        return_reason=site["return_reason"], adapter=site["stalled_reference"]["selected_adapter"])


def findings(methods):
    facts, hypotheses = [], []
    for method, shot in methods.items():
        outcome, flight = shot["outcome"], shot["flight_measurements"]
        start = f"{method.capitalize()}: "
        if outcome["capture"]:
            facts.append(start + f"captured by {outcome['route_class']} route at {outcome['capture_time_s']:.3f} s; "
                         f"{outcome['retention_resets']} retention reset(s).")
        else:
            facts.append(start + ("no whole-ball entry" if not outcome["whole_ball_entry"] else
                                  f"entered but longest retention was {outcome['max_retention_s']:.3f} s") + ".")
        if outcome["board_contact_episodes"] > 1:
            facts.append(start + f"recorded {outcome['board_contact_episodes']} distinct board-contact intervals.")
        first_robot = flight["first_robot_contact"]
        if first_robot:
            facts.append(start + f"measured robot contact at {first_robot['start_s']:.3f} s; link identity is reconstructed separately.")
        bucket_geometry = flight["first_bucket_contact_geometry"]
        if bucket_geometry and bucket_geometry["ascending"]:
            facts.append(start + f"first bucket contact occurred while rising at {bucket_geometry['vertical_velocity_before_m_s']:.2f} m/s; "
                f"ball center was {1000*bucket_geometry['radial_center_distance_m']:.1f} mm from bucket center.")
        crossing = flight["first_descending_center_rim_after_table"] or flight["first_descending_center_rim"]
        if crossing:
            stage = "first descending rim-height passage after table contact" if flight["first_descending_center_rim_after_table"] else "first descending rim-height passage"
            facts.append(start + stage + f" was {1000*abs(crossing['along_error_m']):.1f} mm "
                f"{'beyond' if crossing['along_error_m']>=0 else 'short of'} bucket along the board-to-bucket axis "
                f"(signed lateral error {1000*crossing['lateral_error_m']:.1f} mm).")
            if crossing["previous_nonboard_contact"] == "bucket":
                facts.append(start + "that rim-height passage occurred after a bucket collision, so it does not describe the unobstructed approach.")
        if not outcome["capture"]:
            if first_robot:
                hypotheses.append(start + "test a different arm configuration or flight path; contact occurrence alone does not prove it caused the miss.")
            else:
                hypotheses.append(start + "compare the saved predicted rebound and rim crossing with measured flight before tuning height, tilt or gap; no robot contact was recorded.")
    hypotheses.append("Multiple shot parameters and source candidates changed together; these paired observations do not isolate a causal parameter or prove a correction.")
    return facts, hypotheses


def diagnose(study, audit_path, assets):
    from . import board_first_score_oracle as oracle
    from .arm_clearance_selector import ArmClearanceGate
    from .consistent_flight_selector import geometry
    from .expanded_selector_audit import verify_initial_trace
    from .saved_shot_video import verify_assets
    from .table_assisted_env import _same

    study, audit_path, assets = map(Path, (study, audit_path, assets))
    checks = Checks()
    report, audit, bound, verified = verify_parent(study, audit_path, checks)
    contract = read(bound("contract.json"))
    fingerprint(contract, checks, "Parent contract")
    checks.equal(contract["physics_substeps"], 64, "Native64 source")
    checks.equal(digest(contract["scene"]), contract["identity"]["scene_sha256"], "Frozen scene")
    verify_assets(assets, contract["identity"]["asset_sha256"])
    root = Path(__file__).resolve().parents[2]
    for name in ("model.py", "calibrated_score.py", "calibrated_score_oracle.py", "board_first_score_oracle.py",
                 "neural_shot_model.py", "arm_clearance_selector.py", "consistent_flight_selector.py",
                 "hybrid_miss_diagnosis.py", "link_clearance_diagnosis.py", "benchmark_verify.py",
                 "expanded_selector_audit.py", "table_assisted_env.py", "saved_shot_video.py"):
        name = "sircus/mjlab/" + name
        checks.equal(sha(root/name), contract["source_sha256"][name], "Frozen source " + name)
    checks.equal(sha(root/"sircus/isaac/geometry.py"), contract["source_sha256"]["sircus/isaac/geometry.py"], "Frozen arm geometry")
    calibration = read(bound("calibration.json"))["score_config"]
    geom = geometry(contract["scene"], calibration)
    rows_list = [json.loads(line) for line in bound("rows.jsonl").read_text().splitlines()]
    rows = {row["id"]: row for row in rows_list}
    checks.equal(len(rows), len(rows_list), "Unique source rows")
    slots = {slot["id"]: slot for slot in read(bound("choices.json"))["slots"]}
    plans = {(w["site"]["layout"]["id"], w["variant"]): w["site"] for w in read(bound("plans.json"))}
    checkpoints = {method: read(bound(method + "-checkpoint.json")) for method in METHODS}
    gate = ArmClearanceGate(contract["scene"], assets)
    cases = []
    for number in CASE_NUMBERS:
        case_id, methods = f"expanded-selector-test-{number:02d}", {}
        for method in METHODS:
            identifier = case_id + "-" + method
            row, slot, site = rows[identifier], slots[identifier], plans[(case_id, method)]
            request = slot["request"]
            checks.equal(request["phase"], "test", "Frozen test request")
            expected = dict(request, phase="test_" + method)
            checks.equal({key: row[key] for key in expected}, expected, "Executed frozen command " + identifier)
            checks.equal((row["native_release"], row["invalid_physics"], row["contact_protocol"]["substeps"]),
                         (True, False, 64), "Valid native64 " + identifier)
            packet = bound(row["request_packet"])
            checks.equal(sha(packet), row["request_packet_sha256"], "Bound request packet")
            packet_data = read(packet)
            checks.equal([r for r in packet_data["rows"] if r["id"] == identifier], [request], "Packet contains exact frozen request")
            checks.equal((packet_data["phase"], packet_data["adapter_directory"], packet_data["control"]),
                         (row["phase"], row["adapter_directory"], False), "Packet metadata matches native row")
            relative = row["adapter_directory"] + "/" + row["native_cache_file"]
            path = bound(relative)
            checks.equal(sha(path), row["native_cache_sha256"], "Bound native arrays")
            with np.load(path, allow_pickle=False) as archive:
                column = row["native_cache_row"]
                checks.equal(str(archive["record_ids"][column]), identifier, "Exact native column")
                verify_initial_trace(row, archive, column, checks)
                arrays = {name: archive[name][:, column] for name in ("position_m", "velocity_m_s", "forces_N", "contact_distance_m")}
                arrays.update({name: archive[name] for name in ("time_s", "dt_s") if name in archive})
            score = oracle.score_trace(contract["scene"], row["bucket_xy_m"], row["board_normal"], calibration, arrays)
            _same(row["board_first_score"], score, "Independent v4 rescoring " + identifier)
            checks.count += len(score)
            n = score["physics_steps"]
            checks.equal(n, row["physics_steps"], "Same scoring terminal")
            t = arrays["time_s"][:n+1]
            p, v, f = arrays["position_m"][:n+1], arrays["velocity_m_s"][:n+1], arrays["forces_N"][:n]
            events = episodes(t, p, v, f)
            checks.equal(sum(e["end_tick"]-e["start_tick"] for e in events if e["channel"] == "robot"),
                         score["robot_contact_steps"], "Measured robot interval count")
            values = row["values"]
            settings = dict(command_board_center_m=values[:3], board_center_m=row["board_center_m"],
                board_height_m=values[2], tilt_deg=values[3], drop_gap_m=values[4],
                heading_offset_deg=row["board_heading_offset_deg"], twist_deg=row["board_twist_deg"],
                release_ball_center_m=row["release_ball_center_m"],
                q10_rad=row["native_initial_q10"], q14_rad=row["native_initial_q14"],
                board_normal=row["board_normal"], board_quaternion_wxyz=row["board_quaternion_wxyz"],
                board_bucket_distance_m=math.dist(values[:2], row["bucket_xy_m"]))
            methods[method] = dict(id=identifier, settings=settings,
                selection=selection_details(site, slot, request, checkpoints[method], checks),
                outcome=dict(capture=score["route_success"], **{key: score[key] for key in SCORE_FIELDS}),
                measured_events=events, flight_measurements=measured_flight(t, p, v, f, row, score, geom, events),
                reconstructed_robot_contacts=reconstructed_contacts(gate, row, t, p, f, events, checks),
                trace=dict(relative_file=relative, sha256=row["native_cache_sha256"], column=column,
                           terminal_tick=n, duration_s=float(t[-1]), time_step_s=float(t[1]-t[0])))
        checks.require(methods["current"]["outcome"]["capture"] != methods["expanded"]["outcome"]["capture"], "Discordant pair")
        facts, hypotheses = findings(methods)
        cases.append(dict(id=case_id, bucket_xy_m=rows[case_id+"-current"]["bucket_xy_m"],
            change="expanded_gained" if methods["expanded"]["outcome"]["capture"] else "expanded_lost",
            methods=methods, expanded_minus_current=setting_delta(methods["current"]["settings"], methods["expanded"]["settings"]),
            findings=facts, hypotheses=hypotheses))
    return dict(version=VERSION, status="completed", cpu_only=True, physics_substeps=64,
        checks_count=checks.count, cases=cases, geometry=geom,
        parent_comparison=report["comparison"]["paired"],
        source=dict(study=str(study), report_sha256=sha(study/"report.json"), audit_sha256=sha(audit_path),
            manifest_sha256=sha(study/"evidence-manifest.json"), choices_sha256=sha(study/"choices.json"),
            contract_sha256=sha(study/"contract.json")), verified_artifacts=verified,
        model_sha256=gate.model_sha256, diagnostic_source_sha256=sha(__file__), mujoco_version=gate.mj.__version__,
        scope="Eight selected historical native64 flights, not new trials or a generalization score. Force channels measure contact categories; link identity is inferred at held endpoints. Later robot/table contacts are permitted by v4 scoring. No stepping, GPU allocation, training, source overwrite or automatic planner change.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "audit", "assets", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        parser.error("Set CUDA_VISIBLE_DEVICES='' for this CPU-only saved-evidence diagnosis")
    if args.output.exists() or args.output.resolve().is_relative_to(args.study.resolve()):
        parser.error("Use a new output file outside historical evidence")
    result = diagnose(args.study, args.audit, args.assets)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(status=result["status"], cases=len(result["cases"]), checks_count=result["checks_count"])))


if __name__ == "__main__":
    main()
