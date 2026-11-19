"""Verify fresh PPO records and cut only quiet native-camera outcome tails.

Every batch is retained, including rejected endpoints and misses. Video cuts do
not change the full-rate one-second capture score or become training labels.
"""

import argparse
import hashlib
import json
import math
import shutil
import tempfile
from pathlib import Path

import numpy as np
from board_route import BoardRoute, FORCE_THRESHOLD_N
from action_training import (
    task_dispatch,
    training_formats,
    load_expanded_training_menu,
    verify_expanded_binding,
)
from camera_options import preview_origins
from cylinder_farm import COLS, RADIUS, ROWS, observation, project_actions
from dark_export import inside_frame
from entry_quality import EntryQuality
from export_cylinder_film import (
    encode_movie,
    matches_action,
    read,
    require,
    verified_menus,
)
from farm_contract import reward, statistics
from rl_contract import ShotScore
from saved_camera import load_camera
from settling_contract import accepted
from silver_style import stage_settings

FPS = 30
BATCHES = 10
FOCUS = 1
MOVE_FRAMES = 36
RELEASE_STRIDE = 8
DT = 1 / 240
FORCES = ("board", "bucket", "table", "floor", "robot", "unattributed")
BACKDROP = {
    "renderer_settings": {
        "/rtx/background/source/type": 2,
        "/rtx/background/source/color": [0.22, 0.23, 0.24],
    },
    "documentation": "https://docs.omniverse.nvidia.com/materials-and-rendering/latest/rtx-renderer_common.html",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(value, shape, message):
    result = np.asarray(value, dtype=float)
    require(result.shape == shape and np.isfinite(result).all(), message)
    return result


def close_record(actual, expected, message):
    for name, value in expected.items():
        if isinstance(value, (bool, str)) or value is None:
            equal = actual.get(name) == value
        elif isinstance(value, dict):
            close_record(actual.get(name, {}), value, message)
            continue
        else:
            equal = name in actual and np.allclose(
                actual[name], value, atol=2e-6, rtol=1e-5
            )
        require(equal, f"{message}: {name}")


def score_focus(clip, row, scene, origin):
    """Replay recorded 240 Hz forces and positions using the original scorers."""
    telemetry = clip["focus_telemetry"]
    require(telemetry, "Missing foreground telemetry")
    start = clip["release_step"]
    require(
        [t["physics_step"] for t in telemetry]
        == list(range(start + 1, start + 1 + len(telemetry))),
        "Foreground telemetry is not continuous at 240 Hz",
    )
    bucket = dict(scene["bucket"], center_xy_m=row["bucket_xy_m"])
    scorer = ShotScore(bucket, scene["ball"]["radius_m"])
    quality = EntryQuality(bucket, scene["ball"]["radius_m"])
    route = BoardRoute(DT)
    normal = finite(row["settled_board_normal"], (3,), "Invalid settled board normal")
    require(
        abs(np.linalg.norm(normal) - 1) < 1e-3,
        "Settled board normal is not unit length",
    )
    active = not bool(row["invalid"])
    entry_start = None
    success_entry = None
    scoring_stop = None
    first_board = None
    for item in telemetry:
        pos = finite(item["ball_world_m"], (3,), "Invalid foreground position") - origin
        vel = finite(item["ball_velocity_m_s"], (3,), "Invalid foreground velocity")
        forces = item["forces_N"]
        require(set(forces) == set(FORCES), "Missing contact channels")
        require(
            all(math.isfinite(f) and f >= 0 for f in forces.values()),
            "Invalid contact force",
        )
        if forces["board"] > FORCE_THRESHOLD_N and first_board is None:
            first_board = item["physics_step"]
        if active and item["physics_step"] - start <= 960:
            scorer.step(pos, forces["board"], DT, float(vel @ normal))
            quality.step(pos, forces["bucket"], scorer.entered)
            route.step(forces)
            if scorer.retention > 0:
                if entry_start is None:
                    entry_start = item["physics_step"]
            else:
                entry_start = None
            if scorer.success:
                success_entry = entry_start
            if scorer.success or pos[2] < -0.2 or item["physics_step"] - start == 960:
                scoring_stop = item["physics_step"]
                active = False
    require(not active, "Foreground telemetry ends before the scoring horizon")
    if not row["invalid"]:
        scored = scorer.result()
        measured_quality = quality.result(scored["success"])
        measured_route = route.result(
            scored["success"], measured_quality["quality_score"]
        )
        close_record(row, scored, "Foreground scoring replay disagrees")
        close_record(row, measured_quality, "Foreground entry quality disagrees")
        close_record(row, measured_route, "Foreground route scoring disagrees")
    return dict(
        scoring_stop_step=scoring_stop,
        physical_retained_capture=bool(scorer.success and not row["invalid"]),
        successful_retention_start_step=success_entry,
        first_board_contact_step=first_board,
    )


def terminal_cut(clip, row, audit, origin, padding_s=0.2):
    """Select an inclusive source-frame span without discarding any batch.

    A rim touch is never terminal. Successful retention uses its final qualifying
    entry interval. Otherwise retain all bucket interactions before a measured
    table/floor outcome; unresolved trajectories run through the full horizon.
    """
    frames, telemetry = clip["frames"], clip["focus_telemetry"]
    release = [f for f in frames if f["phase"] == "release"]
    require(release, "Missing release frames")
    if row["invalid"]:
        event, reason = clip["release_step"], "endpoint_rejected"
    elif audit["physical_retained_capture"]:
        event, reason = (
            audit["successful_retention_start_step"],
            "retained_bucket_entry",
        )
        require(event is not None, "Capture has no qualifying entry interval")
    else:
        bucket_steps = [
            t["physics_step"]
            for t in telemetry
            if t["forces_N"]["bucket"] > FORCE_THRESHOLD_N
        ]
        last_bucket = max(bucket_steps, default=clip["release_step"])
        landed = next(
            (
                t
                for t in telemetry
                if t["physics_step"] > last_bucket
                and max(t["forces_N"]["table"], t["forces_N"]["floor"])
                > FORCE_THRESHOLD_N
            ),
            None,
        )
        if landed is not None:
            event = landed["physics_step"]
            reason = (
                "floor_contact"
                if landed["forces_N"]["floor"] > FORCE_THRESHOLD_N
                else "table_contact"
            )
        else:
            exited = next(
                (
                    t
                    for t in telemetry
                    if t["physics_step"] > last_bucket
                    and np.asarray(t["ball_world_m"])[2] - origin[2] < -0.2
                ),
                None,
            )
            event = (
                exited["physics_step"]
                if exited is not None
                else telemetry[-1]["physics_step"]
            )
            reason = (
                "workspace_exit_below_table"
                if exited is not None
                else "timeout_unresolved"
            )
    through = min(telemetry[-1]["physics_step"], event + math.ceil(padding_s / DT))
    end = next((f for f in release if f["physics_step"] >= through), release[-1])
    start_frame, end_frame = frames[0]["frame"], end["frame"]
    require(end_frame >= frames[MOVE_FRAMES - 1]["frame"], "Cut lost arm setup frames")
    nearest = min(release, key=lambda f: abs(f["physics_step"] - event))
    item = min(telemetry, key=lambda t: abs(t["physics_step"] - event))
    return dict(
        source_start_frame=start_frame,
        source_end_frame_inclusive=end_frame,
        source_frames=end_frame - start_frame + 1,
        terminal_reason=reason,
        terminal_physics_step=event,
        terminal_event_in_view=inside_frame(nearest, item["ball_world_m"]),
        physical_retained_capture=audit["physical_retained_capture"],
        route_qualified_capture=bool(row["route_success"]),
        invalid=row["invalid"],
        scoring_stop_step=audit["scoring_stop_step"],
        scoring_retention_s=row["max_retention_s"],
        padding_s=padding_s,
    )


def verify_action_geometry(row, action, cell, expanded=False):
    """The fifth action is measured radial distance, never an implicit menu tie."""
    expected_radius = action["radius_m"] if expanded else RADIUS
    require(
        np.allclose(row["bucket_xy_m"], cell["bucket_xy_m"], atol=1e-7, rtol=0)
        and abs(row["distance_m"] - expected_radius) < 1e-7,
        "Bucket or radius changed",
    )
    if expanded:
        measured = np.linalg.norm(
            np.asarray(row["bucket_xy_m"]) - [row["board_x_m"], row["board_y_m"]]
        )
        require(
            abs(measured - row["radius_m"]) < 1e-7
            and abs(row["radius_m"] - action["radius_m"]) < 1e-7,
            "Expanded board XY and radius disagree",
        )


def verify_probes(run, count):
    probes = read(run / "probes.json")
    require(
        [p["kind"] for p in probes]
        == ["bucket", "rim", "table", "floor", "robot", "foreign_cell_collision_filter"]
        and all(p["all_passed"] for p in probes)
        and all(p["per_cell_passed"] == [True] * count for p in probes[:-1])
        and read(run / "rim-rays.json")["passed"] == [True] * count,
        "Contact, rim or cell-isolation probe failed",
    )


def verify(run):
    run = Path(run)
    report, spec = read(run / "result.json"), read(run / "contract.json")
    task, expanded = task_dispatch(spec)
    formats = training_formats(spec)
    fingerprint = spec["fingerprint"]
    require(
        hashlib.sha256(
            json.dumps(
                {k: v for k, v in spec.items() if k != "fingerprint"}, sort_keys=True
            ).encode()
        ).hexdigest()
        == fingerprint
        and report["fingerprint"] == report["source_fingerprint"] == fingerprint,
        "Training contract fingerprint mismatch",
    )
    require(
        report["schema"] == formats["report_schema"]
        and report["filming_protocol"] == formats["protocol"],
        "Not a fresh training film",
    )
    require(
        spec["training_batches"] == BATCHES and spec["batches_per_update"] == 2,
        "Wrong training budget",
    )
    cells = spec["cells"]
    require(
        len(cells) == ROWS * COLS == report["workcells"] == 35
        and report["articulations"] == 70,
        "Wrong workcell count",
    )
    require(
        len({tuple(c["bucket_xy_m"]) for c in cells}) == 35,
        "Bucket sites are not distinct",
    )
    require(
        np.array_equal(preview_origins(), report["origins_m"]),
        "Display origins changed",
    )
    require(
        report["fps"] == FPS
        and report["resolution"] == [1920, 1080]
        and report["physics_dt_s"] == DT,
        "Cadence or resolution mismatch",
    )
    binding = verify_expanded_binding(run, spec, report) if expanded else None
    camera = (
        binding["camera"]
        if expanded
        else load_camera(run / "selected-camera.json", fingerprint)
    )
    camera_hash = digest(run / "selected-camera.json")
    require(
        camera == report["camera"] and camera_hash == report["selected_camera_sha256"],
        "Selected camera changed",
    )
    expected_stage = json.loads(json.dumps(stage_settings()))
    expected_stage["camera_backdrop"] = BACKDROP
    require(
        report["stage"] == expected_stage, "Selected silver lighting/backdrop changed"
    )
    verify_probes(run, 35)
    menus, menu_audit = (
        load_expanded_training_menu(run / "expanded-menu-qualified.json", spec)
        if expanded
        else verified_menus(run, cells)
    )
    provenance = read(run / "menu-provenance.json")
    require(
        menu_audit["file_sha256"] == report["menu_sha256"] == provenance["file_sha256"]
        and provenance == report["menu_provenance"]
        and provenance["source_fingerprint"] == fingerprint,
        "Geometry menu lineage mismatch",
    )
    require(
        len(report["batches"]) == BATCHES and len(report["updates"]) == 5,
        "Incomplete batches or PPO updates",
    )
    current, total = report["initial_hash"], 0
    rows, chapters, checkpoints = [], [], []
    observations = [task.observation(c) for c in cells]
    for batch in range(BATCHES):
        clip, summary = read(run / f"clip-{batch:02}.json"), report["batches"][batch]
        require(
            clip["schema"] == formats["clip_schema"]
            and clip["filming_protocol"] == report["filming_protocol"]
            and clip["batch"] == batch
            and clip["focus"] == FOCUS
            and summary["batch"] == batch,
            "Missing or reordered batch",
        )
        require(
            clip["source_fingerprint"] == fingerprint
            and clip["menu_sha256"] == report["menu_sha256"]
            and clip["selected_camera_sha256"] == camera_hash,
            "Clip lineage changed",
        )
        require(
            clip["policy_hash"] == summary["policy_hash"] == current,
            "Clip is not from the current on-policy weights",
        )
        filename = f"policy-{batch:02}.pt"
        require(
            clip["checkpoint_file"] == filename
            and digest(run / filename) == clip["checkpoint_sha256"],
            "Sampling checkpoint digest mismatch",
        )
        checkpoints.append(
            dict(file=filename, sha256=clip["checkpoint_sha256"], weight_hash=current)
        )
        if expanded:
            require(
                clip["task_version"] == spec["version"]
                and clip["action_dim"] == formats["action_dim"]
                and clip["camera_binding_sha256"] == report["camera_binding_sha256"]
                and clip["camera_source_fingerprint"]
                == report["camera_source_fingerprint"],
                "Expanded clip task/camera binding changed",
            )
        latent = finite(
            clip["latent"], (35, formats["action_dim"]), "Incomplete latent samples"
        )
        finite(clip["old_logp"], (35,), "Incomplete log probabilities")
        finite(clip["rewards"], (35,), "Incomplete reward samples")
        require(
            np.allclose(clip["observation"], observations, atol=1e-7, rtol=0),
            "Policy observations changed",
        )
        actions = task.project_actions(cells, latent.astype(np.float32), menus)
        records = clip["rollout"]
        require(
            clip["release_enabled"] == [bool(m) for m in menus]
            and summary["optimizer_samples"] == 35
            and summary["supported_cells"] == sum(bool(m) for m in menus),
            "Release mask or original optimizer sampling changed",
        )
        require(len(clip["trajectory_m"]) == 35, "Missing per-cell trajectories")
        require(
            len(records) == len(clip["actions"]) == 35, "Incomplete sampled attempts"
        )
        for index, (cell, row, action, logged, ret) in enumerate(
            zip(cells, records, actions, clip["actions"], clip["rewards"])
        ):
            require(
                row["cell_id"] == cell["id"]
                and matches_action(logged, action)
                and matches_action(row, action),
                "Sampled/projected action mismatch",
            )
            verify_action_geometry(row, action, cell, expanded)
            require(
                np.isclose(ret, reward(row), atol=1e-6)
                and np.isclose(ret, row["ppo_reward"], atol=1e-6)
                and row["geometry_menu_empty"] == (not bool(menus[index])),
                "PPO reward or menu diagnostic disagrees with scored row",
            )
            require(
                bool(menus[index]) or bool(row["invalid"]), "Menu-empty cell released"
            )
            if not row["invalid"]:
                release = [
                    row["board_x_m"],
                    row["board_y_m"],
                    row["height_m"] + row["gap_m"] + spec["scene"]["ball"]["radius_m"],
                ]
                require(
                    np.allclose(
                        row["release_ball_center_m"], release, atol=1e-7, rtol=0
                    )
                    and accepted(row["settled"])
                    and row["settled"]["dropper_position_error_mm"] <= 2,
                    "Release target or settling failed",
                )
            if row["route_success"]:
                require(
                    not row["invalid"]
                    and row["board_bounce"]
                    and row["whole_ball_entry"]
                    and row["max_retention_s"] >= 1 - 1e-8
                    and not row["external_contact_steps"],
                    "Unqualified capture",
                )
        close_record(summary, statistics(records), "Batch totals disagree")
        frames = clip["frames"]
        move = [f for f in frames if f["phase"] == "move"]
        release = [f for f in frames if f["phase"] == "release"]
        require(
            len(move) == MOVE_FRAMES
            and frames[:MOVE_FRAMES] == move
            and release
            and frames[MOVE_FRAMES:] == release,
            "Missing complete arm setup or release sequence",
        )
        require(
            clip["physics_dt_s"] == DT and clip["release_stride"] == RELEASE_STRIDE,
            "Clip physics cadence changed",
        )
        require(
            summary["start_frame"] == total and summary["frames"] == len(frames),
            "Batch frame totals disagree",
        )
        for i, frame in enumerate(frames):
            require(
                frame["frame"] == total + i
                and np.allclose(frame["eye_m"], camera["eye_m"], atol=2e-4, rtol=0)
                and np.allclose(
                    frame["target_m"], camera["target_m"], atol=2e-5, rtol=0
                ),
                "Native camera is not fixed or frames are missing",
            )
            positions = finite(
                frame["all_ball_world_m"], (35, 3), "Missing background ball tracks"
            )
            require(
                np.allclose(
                    frame["focus_ball_world_m"], positions[FOCUS], atol=1e-7, rtol=0
                ),
                "Foreground and full-farm track disagree",
            )
        require(
            all(b["physics_step"] > a["physics_step"] for a, b in zip(move, move[1:]))
            and release[0]["physics_step"] == clip["release_step"]
            and all(
                b["physics_step"] - a["physics_step"] == RELEASE_STRIDE
                for a, b in zip(release, release[1:])
            ),
            "Release frame cadence changed",
        )
        motion = {}
        for name in ("Arm10", "Arm14"):
            joints = finite(
                [f["focus_joints_rad"][name] for f in move],
                (MOVE_FRAMES, 6),
                "Missing measured setup joints",
            )
            motion[name] = float(np.max(np.ptp(joints, axis=0)))
        require(
            clip["telemetry_end_step"]
            == clip["capture_end_step"]
            == clip["focus_telemetry"][-1]["physics_step"]
            and clip["score_end_step"] <= clip["capture_end_step"]
            and clip["score_end_step"] == summary["score_end_step"]
            and clip["release_step"] == summary["release_step"],
            "Score/capture horizon mismatch",
        )
        for index, row in enumerate(records):
            if not row["invalid"]:
                require(
                    np.allclose(
                        np.asarray(release[0]["all_ball_world_m"])[index]
                        - report["origins_m"][index],
                        row["release_ball_center_m"],
                        atol=0.003,
                        rtol=0,
                    ),
                    "Recorded release disagrees with requested pose",
                )
        audit = score_focus(
            clip, records[FOCUS], spec["scene"], np.asarray(report["origins_m"][FOCUS])
        )
        cut = terminal_cut(
            clip, records[FOCUS], audit, np.asarray(report["origins_m"][FOCUS])
        )
        activity = set()
        for item in clip["focus_telemetry"]:
            require(
                set(item["activity"]) == set(FORCES)
                and all(
                    isinstance(i, int) and 0 <= i < 35
                    for group in item["activity"].values()
                    for i in group
                ),
                "Invalid background contact indices",
            )
            require(
                all(
                    (FOCUS in item["activity"][name])
                    == (item["forces_N"][name] > FORCE_THRESHOLD_N)
                    for name in FORCES
                ),
                "Foreground contact and activity disagree",
            )
            for index in item["activity"]["board"]:
                if index != FOCUS and not records[index]["invalid"]:
                    nearest = min(
                        release,
                        key=lambda f: abs(f["physics_step"] - item["physics_step"]),
                    )
                    if inside_frame(nearest, nearest["all_ball_world_m"][index]):
                        activity.add(index)
        chapters.append(
            dict(
                batch=batch,
                title=f"Training batch {batch+1} — {cut['terminal_reason']}",
                start_frame=total,
                frames=len(frames),
                focus=cells[FOCUS]["id"],
                motion_excursion_rad=motion,
                background_visible_board_contacts=len(activity),
                **cut,
            )
        )
        rows.extend(records)
        total += len(frames)
        if batch % 2:
            change = report["updates"][batch // 2]
            require(
                change["before"] == current
                and change["after"] != current
                and len(change["epochs"]) == spec["epochs"],
                "PPO update chain broke",
            )
            current = change["after"]
    require(
        current == report["final_hash"]
        and digest(run / "policy-final.pt") == report["final_checkpoint_sha256"],
        "Final checkpoint mismatch",
    )
    require(
        report["frames"] == total
        and [p.name for p in sorted((run / "frames").glob("*.jpg"))]
        == [f"{i:05d}.jpg" for i in range(total)],
        "Incomplete native RGB sequence",
    )
    require(
        report["attempts"] == report["optimizer_samples"] == len(rows) == 350
        and report["captures"] == statistics(rows)["captures"],
        "Missing training attempts or outcome totals",
    )
    require(
        all(
            max(c["motion_excursion_rad"][name] for c in chapters) > 0.01
            for name in ("Arm10", "Arm14")
        ),
        "No measured setup motion for both arms",
    )
    checkpoint_audit = verify_checkpoint_audit(
        run, report, checkpoints, formats["audit_schema"]
    )
    return dict(
        schema=formats["export_schema"],
        **(
            dict(
                task_version=spec["version"],
                action_dim=formats["action_dim"],
                camera_source_fingerprint=report["camera_source_fingerprint"],
                camera_binding=binding,
            )
            if expanded
            else {}
        ),
        checkpoint_audit=checkpoint_audit,
        source_fingerprint=fingerprint,
        selected_camera_sha256=camera_hash,
        camera=camera,
        resolution=report["resolution"],
        fps=FPS,
        training=statistics(rows),
        releases=sum(not r["invalid"] for r in rows),
        ppo_updates=5,
        initial_weight_hash=report["initial_hash"],
        final_weight_hash=current,
        final_checkpoint_sha256=report["final_checkpoint_sha256"],
        sampling_checkpoints=checkpoints,
        raw_frames=total,
        raw_duration_s=total / FPS,
        geometry_menu=menu_audit,
        chapters=chapters,
        scope="Actual on-policy endpoint-shot PPO training; scripted setup and initialized releases. All batches retained. Action-only visual cuts do not alter full-rate one-second scoring; no held-out evaluation or learned grasp/path claim.",
    )


def verify_checkpoint_audit(
    run,
    report,
    checkpoints,
    expected_schema="sircus-action-training-checkpoint-audit-v1",
):
    """Bind an optional separately executed, CPU-only checkpoint/logp audit."""
    path = run / "checkpoint-audit.json"
    if not path.exists():
        return dict(
            verified=False,
            scope="Checkpoint bytes and recorded update chain verified; Torch weight loading and latent log-probability replay await the separate CPU audit.",
        )
    audit = read(path)
    expected = {
        f"clip-{i:02}.json": digest(run / f"clip-{i:02}.json") for i in range(BATCHES)
    }
    checkpoints = {c["file"]: c["sha256"] for c in checkpoints}
    checkpoints["policy-final.pt"] = report["final_checkpoint_sha256"]
    require(
        audit["schema"] == expected_schema
        and audit["source_result_sha256"] == digest(run / "result.json")
        and audit["source_contract_sha256"] == digest(run / "contract.json")
        and audit["source_clip_sha256"] == expected
        and audit["checkpoint_sha256"] == checkpoints
        and audit["final_policy_hash"] == report["final_hash"],
        "CPU checkpoint audit is stale or belongs to another run",
    )
    require(
        audit["sampled_policy_hashes"] == [b["policy_hash"] for b in report["batches"]],
        "CPU checkpoint audit sampling chain disagrees",
    )
    if expected_schema == "sircus-action-training-checkpoint-audit-v2":
        require(
            audit["task_version"] == report["task_version"]
            and audit["action_dim"] == report["action_dim"] == 5
            and audit["camera_source_fingerprint"]
            == report["camera_source_fingerprint"]
            and audit["camera_binding_sha256"] == report["camera_binding_sha256"]
            and audit["native_menu_sha256"] == report["menu_sha256"],
            "CPU expanded task/camera/menu audit mismatch",
        )
    return dict(verified=True, file=path.name, sha256=digest(path), audit=audit)


def render_action_cut(run, output, chapters, resolution):
    """Reindex original native JPEGs only; no compositing, interpolation or overlays."""
    stitched = []
    with tempfile.TemporaryDirectory(prefix="sircus-training-cut-") as tmp:
        numbered = Path(tmp)
        count = 0
        for chapter in chapters:
            start = count
            for source in range(
                chapter["source_start_frame"], chapter["source_end_frame_inclusive"] + 1
            ):
                (numbered / f"{count:05d}.jpg").symlink_to(
                    (run / "frames" / f"{source:05d}.jpg").resolve()
                )
                count += 1
            stitched.append(dict(chapter, start_frame=start, frames=count - start))
        movie = encode_movie(numbered, output, stitched, resolution)
    return dict(movie, chapters=stitched, native_pixels_only=True, playback_multiple=1)


def export(run, output, raw_movie=False):
    summary = verify(run)
    output.mkdir(parents=True, exist_ok=False)
    summary["action_movie"] = render_action_cut(
        run, output / "action-training.mp4", summary["chapters"], summary["resolution"]
    )
    if raw_movie:
        summary["untrimmed_movie"] = encode_movie(
            run / "frames",
            output / "action-training-untrimmed.mp4",
            summary["chapters"],
            summary["resolution"],
        )
    poster = summary["chapters"][0]["start_frame"] + MOVE_FRAMES
    shutil.copyfile(run / "frames" / f"{poster:05d}.jpg", output / "training.jpg")
    (output / "film.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "chapters"}))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--untrimmed", action="store_true")
    args = parser.parse_args()
    export(args.run, args.output, args.untrimmed)
