"""Audit and edit the native RTX frames from the compact-farm replay."""

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
from board_route import FORCE_THRESHOLD_N
from recording_plans import recording_plan
from dark_stage import FLOOR_Z_M, origins, stage_settings
from export_cylinder_film import encode_movie, read, require
from feasibility_contract import dynamic_cell
from rl_contract import ShotScore

FPS = 30
ACTION_FIELDS = (
    "id",
    "azimuth_deg",
    "azimuth_offset_deg",
    "board_x_m",
    "board_y_m",
    "height_m",
    "gap_m",
    "tilt_deg",
    "cell_id",
    "bucket_xy_m",
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inside_frame(frame, point):
    """Check the logged contact/release projects inside the 23 mm perspective shot."""
    eye, aim = np.asarray(frame["eye_m"]), np.asarray(frame["target_m"])
    direction = aim - eye
    forward = direction / np.linalg.norm(direction)
    right = np.cross(forward, [0, 0, 1])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    ray = np.asarray(point) - eye
    depth = float(ray @ forward)
    return (
        depth > 0.04
        and abs(ray @ right) * 23 < depth * 18
        and abs(ray @ up) * 23 < depth * 10.125
    )


def event_frame(contact, release, label):
    if contact is None:
        return None
    step = contact["physics_step"]
    require(
        np.isfinite(contact["force_N"]) and contact["force_N"] > FORCE_THRESHOLD_N,
        f"{label} has no measured contact",
    )
    point = contact["ball_world_m"]
    require(
        np.shape(point) == (3,) and np.isfinite(point).all(),
        f"{label} has an invalid point",
    )
    if not release[0]["physics_step"] <= step <= release[-1]["physics_step"]:
        return None
    best = min(release, key=lambda f: abs(f["physics_step"] - step))
    return best["frame"] if inside_frame(best, point) else None


def assess_focus(chapter, clip, row, bucket, radius, origin):
    frames = clip["frames"]
    release = frames[chapter["move_frames"] :]
    contacts = clip["focus_contacts"]
    require(not row["invalid"], "Focused arm never released")
    ball = np.asarray([f["focus_ball_world_m"] for f in release])
    require(
        ball.shape == (len(release), 3) and np.isfinite(ball).all(),
        "Missing focused ball track",
    )
    require(
        np.allclose(ball[0] - origin, row["release_ball_center_m"], atol=0.003, rtol=0),
        "Ball did not start at its recorded release target",
    )
    board = event_frame(contacts["board"], release, "Board")
    floor = event_frame(contacts["floor"], release, "Floor")
    require(chapter["board_bounce"] == bool(row["board_bounce"]), "Bounce log mismatch")
    captured = bool(row["route_success"])
    require(chapter["captured"] == captured, "Capture log mismatch")
    literal_floor = contacts["floor"] is not None
    require(chapter["literal_floor"] == literal_floor, "Floor log mismatch")
    if literal_floor:
        floor_z = contacts["floor"]["ball_world_m"][2] - origin[2]
        require(
            abs(floor_z - (FLOOR_Z_M + radius)) < 0.025,
            "Floor force was not at floor height",
        )
    # A release-frame sample at the bucket must also agree with its whole-ball geometry.
    # The physics scorer's full-rate retention is checked independently below.
    landing = None
    if captured:
        require(
            row["board_bounce"]
            and row["whole_ball_entry"]
            and row["max_retention_s"] >= 1 - 1e-8
            and row["external_contact_steps"] == 0,
            "Unqualified claimed capture",
        )
        scorer = ShotScore(dict(bucket, center_xy_m=row["bucket_xy_m"]), radius)
        scorer.bounced = True
        scorer.was_above = True
        for f, pos in zip(release, ball):
            scorer.step(pos - origin, 0.0, 1 / FPS)
            if scorer.retention > 0 and inside_frame(f, pos):
                landing = f["frame"]
                break
    if chapter["action_only"]:
        require(
            board is not None and row["board_bounce"],
            f"Missing visible board rebound at {chapter['shot']}",
        )
        if captured:
            require(
                landing is not None and landing > board,
                f"Bucket entry outside shot {chapter['shot']}",
            )
        if literal_floor:
            require(
                floor is not None and floor > board,
                f"Floor arrival outside shot {chapter['shot']}",
            )
    return dict(
        visible_board_frame=board,
        visible_bucket_entry_frame=landing,
        visible_floor_frame=floor,
        visible_bucket_contact_frame=event_frame(contacts["bucket"], release, "Bucket"),
        visible_table_contact_frame=event_frame(contacts["table"], release, "Table"),
    )


def verify_resume(run, result):
    info = result.get("resumed")
    if info is None:
        return
    name = info["previous_run"]
    require(Path(name).name == name, "Invalid resume origin")
    previous = run.parent / name
    require(
        all(p["all_passed"] for p in read(previous / "probes.json")),
        "The original capture prefix failed contact checks",
    )
    sha = hashlib.sha256()
    for i in range(info["completed_chapters"]):
        filename = f"clip-{i:02}.json"
        require(
            digest(run / filename)
            == digest(previous / filename)
            == info["clip_sha256"][filename],
            "The resumed prefix changed its original shot records",
        )
        for f in read(run / filename)["frames"]:
            filename = f"{f['frame']:05d}.jpg"
            raw = (run / "frames" / filename).read_bytes()
            require(
                raw == (previous / "frames" / filename).read_bytes(),
                "Resumed RGB changed",
            )
            sha.update(filename.encode() + raw)
    require(sha.hexdigest() == info["frames_sha256"], "Resumed RGB digest disagrees")


def validate_station_series(chapters):
    """Count distinct recorded releases; planned slots and re-used frames do not count."""
    from station_flight import MIN_FOREGROUND_DROPS

    require(
        len(chapters) >= MIN_FOREGROUND_DROPS,
        "Need at least 10 recorded foreground drops",
    )
    require(
        len({c["focus"] for c in chapters}) == 1,
        "Station series changed its foreground table",
    )
    require(
        len({c["start_frame"] for c in chapters}) == len(chapters),
        "Duplicate recorded drop",
    )
    require(
        all(
            c["station"]
            and c["visible_board_frame"] is not None
            and c["background_drops_in_view"] >= 2
            for c in chapters
        ),
        "Need measured foreground drops and background activity in every shot",
    )


def verify(run, source):
    result = read(run / "result.json")
    plan_name = result.get("recording_plan", "dark-tour-v1")
    shots, camera_pose = recording_plan(plan_name)
    verify_resume(run, result)
    parent, spec = read(source / "result.json"), read(source / "contract.json")
    require(result["version"] == "sircus-dark-farm-replay-v1", "Unknown replay")
    require(
        result["source_fingerprint"] == parent["fingerprint"] == spec["fingerprint"]
        and result["source_weights_sha256"] == parent["final_hash"],
        "Wrong source run",
    )
    require(
        result["stage"] == json.loads(json.dumps(stage_settings()))
        and np.array_equal(result["origins_m"], origins())
        and (result["workcells"], result["articulations"]) == (35, 70),
        "Wrong farm",
    )
    require(
        result["fps"] == FPS
        and result["resolution"] == [1920, 1080]
        and np.isclose(result["physics_dt_s"], 1 / 240, atol=1e-10, rtol=0),
        "Unexpected timing or resolution",
    )
    checks = read(run / "probes.json")
    require(
        [p["kind"] for p in checks]
        == ["bucket", "rim", "table", "floor", "robot", "foreign_cell_collision_filter"]
        and all(p["all_passed"] for p in checks)
        and all(p["per_cell_passed"] == [True] * 35 for p in checks[:-1])
        and read(run / "rim-rays.json")["passed"] == [True] * 35,
        "Probe failure",
    )
    require(len(result["chapters"]) == len(shots), "Missing directed shots")
    source_rows, source_hash = {}, {}
    chapters, total, attempts, valid, captures = [], 0, 0, 0, 0
    focus_moves = []
    for i, ch in enumerate(result["chapters"]):
        plan = shots[i]
        clip = read(run / f"clip-{i:02}.json")
        require(
            (
                ch["shot"],
                ch["source_batch"],
                ch["frames"],
                ch["move_frames"],
                ch["stride"],
            )
            == (
                plan.name,
                plan.source_batch,
                plan.frames,
                plan.move_frames,
                plan.stride,
            ),
            "Chapter disagrees with film plan",
        )
        frames, rows = clip["frames"], clip["rollout"]
        require(
            clip["chapter"] == ch
            and len(rows) == 35
            and len(clip["action_source_batches"]) == 35
            and ch["start_frame"] == total
            and ch["frames"] == len(frames)
            and [f["frame"] for f in frames] == list(range(total, total + len(frames))),
            "Shot or frame mismatch",
        )
        move = ch["move_frames"]
        stride = ch["stride"]
        release = frames[move:]
        require(
            stride > 0
            and 0 <= move < len(frames)
            and all(f["phase"] == "move" for f in frames[:move])
            and all(f["phase"] == "release" for f in release)
            and all(
                b["physics_step"] - a["physics_step"] == stride
                for a, b in zip(release, release[1:])
            ),
            f"Invalid phase/physics cadence in {ch['shot']}",
        )
        for f in frames:
            require(
                np.shape(f["eye_m"]) == np.shape(f["target_m"]) == (3,)
                and np.isfinite([f["eye_m"], f["target_m"]]).all()
                and np.linalg.norm(np.subtract(f["target_m"], f["eye_m"])) > 0.01,
                "Missing camera pose",
            )
        for j, (row, src) in enumerate(zip(rows, clip["action_source_batches"])):
            if src not in source_rows:
                p = source / f"clip-{src:02}.json"
                source_hash[src], source_rows[src] = digest(p), read(p)["rollout"]
            original = source_rows[src][j]
            require(
                all(row[k] == original[k] for k in ACTION_FIELDS),
                f"Replay action changed in {ch['shot']}, cell {j}",
            )
        require(
            clip["source_clip_sha256"] == source_hash[ch["source_batch"]],
            "Source clip digest mismatch",
        )
        require(
            ch["focus"]
            == (rows[plan.focus]["cell_id"] if plan.focus is not None else None),
            "Focused cell changed",
        )
        geometry = [dynamic_cell(cell, row) for cell, row in zip(spec["cells"], rows)]
        for k, frame in enumerate(frames):
            ball = (
                frame.get("focus_ball_world_m") if frame["phase"] == "release" else None
            )
            eye, aim = camera_pose(plan, k, geometry, origins(), ball)
            require(
                np.allclose(eye, frame["eye_m"], atol=0.00021, rtol=0)
                and np.allclose(aim, frame["target_m"], atol=0.00003, rtol=0),
                f"Recorded camera path disagrees in {ch['shot']} at frame {k}",
            )
        focused = {}
        if ch["focus"] is not None:
            j = [r["cell_id"] for r in rows].index(ch["focus"])
            focused = assess_focus(
                ch,
                clip,
                rows[j],
                spec["scene"]["bucket"],
                spec["scene"]["ball"]["radius_m"],
                origins()[j],
            )
            if plan_name == "fixed-station-v1":
                # The same recorded batch must also show active neighboring tables.
                focused["background_drops_in_view"] = sum(
                    not r["invalid"]
                    and r["board_bounce"]
                    and inside_frame(
                        release[0], origins()[k] + r["release_ball_center_m"]
                    )
                    for k, r in enumerate(rows)
                    if k != j
                )
            if ch.get("station"):
                excursions = clip["motion"]["max_joint_excursion_deg"]
                for arm in ("Arm10", "Arm14"):
                    joints = np.array(
                        [f["focus_joints_rad"][arm] for f in frames[:move]]
                    )
                    require(
                        joints.shape == (move, 6)
                        and np.isfinite(joints).all()
                        and np.isclose(
                            excursions[arm],
                            np.rad2deg(np.ptp(joints, axis=0)).max(),
                            atol=1e-5,
                        ),
                        f"Missing or inconsistent station joint measurements in {ch['shot']}",
                    )
                focus_moves.append((ch["focus"], excursions))
                focused["joint_excursion_deg"] = excursions
        chapters.append(dict(ch, **focused))
        total += len(frames)
        attempts += len(rows)
        valid += sum(not r["invalid"] for r in rows)
        captures += sum(bool(r["route_success"]) for r in rows)
    require(
        total == result["frames"]
        and [p.name for p in sorted((run / "frames").glob("*.jpg"))]
        == [f"{k:05d}.jpg" for k in range(total)],
        "Frame sequence incomplete",
    )
    actions = [c for c in chapters if c["action_only"]]
    if plan_name == "fixed-station-v1":
        validate_station_series(actions)
    require(
        any(
            c["captured"] and c["visible_bucket_entry_frame"] is not None
            for c in actions
        ),
        "No filmed close-up capture",
    )
    require(
        any(
            not c["captured"] and c["visible_floor_frame"] is not None for c in actions
        ),
        "No filmed close-up miss reaching the physical floor",
    )
    require(
        len(focus_moves) >= 2
        and len({name for name, _ in focus_moves}) == 1
        and all(max(m[k] for _, m in focus_moves) >= 1 for k in ("Arm10", "Arm14")),
        "Repeated station or measured joint motion missing",
    )
    return dict(
        version=result["version"],
        recording_plan=plan_name,
        scope="Native PhysX replay of previously sampled PPO actions",
        source_fingerprint=parent["fingerprint"],
        source_weights_sha256=parent["final_hash"],
        stage=result["stage"],
        workcells=35,
        arms=70,
        fps=FPS,
        resolution=result["resolution"],
        frames=total,
        resumed=result.get("resumed"),
        full_duration_s=total / FPS,
        attempts=attempts,
        releases=valid,
        captures=captures,
        focused_bounces=sum(c["visible_board_frame"] is not None for c in actions),
        focused_captures=sum(c["captured"] for c in actions),
        focused_floor_arrivals=sum(
            c["visible_floor_frame"] is not None for c in actions
        ),
        chapters=chapters,
    )


def highlight_span(chapter):
    """Original frame IDs. Keep 0.6 seconds of travel and 0.5 seconds beyond landing."""
    first = chapter["start_frame"]
    last = first + chapter["frames"]
    begin = first + max(0, chapter["move_frames"] - 18)
    impact = chapter.get("visible_board_frame")
    landing = chapter.get("visible_floor_frame")
    if chapter.get("captured"):
        landing = chapter["visible_bucket_entry_frame"]
    if landing is None and impact is not None:
        other_contacts = [
            chapter.get(k)
            for k in ("visible_bucket_contact_frame", "visible_table_contact_frame")
        ]
        landing = max(
            (t for t in other_contacts if t is not None and t > impact), default=None
        )
    end = min(last, landing + 16) if landing is not None else last
    return begin, end


def render_cut(frames_dir, output, chapters, resolution, step=1):
    """Encode linked native frames; fast cuts keep every tenth frame across edits."""
    relative, spans = [], []
    count = source_count = 0
    with tempfile.TemporaryDirectory(dir=output.parent) as temp:
        target = Path(temp)
        for c in chapters:
            begin, end = highlight_span(c)
            first_kept = begin + (-source_count % step)
            indices = range(first_kept, end, step)
            require(
                len(indices) > 0,
                "A trimmed shot contains no frames at its playback rate",
            )
            relative.append(dict(c, start_frame=count, frames=len(indices)))
            spans.append(
                dict(
                    shot=c["shot"],
                    first_source_frame=begin,
                    end_source_frame_exclusive=end,
                    first_sampled_frame=first_kept,
                    encoded_frames=len(indices),
                )
            )
            for src_index in indices:
                (target / f"{count:05d}.jpg").symlink_to(
                    (frames_dir / f"{src_index:05d}.jpg").resolve()
                )
                count += 1
            source_count += end - begin
        movie = encode_movie(target, output, relative, resolution)
    del movie[
        "source_start_frame"
    ]  # An edit contains multiple noncontiguous source ranges.
    return dict(
        movie, source_spans=spans, source_frames=source_count, playback_multiple=step
    )


def contact_sheet(frames_dir, dest, chapters):
    from PIL import Image, ImageDraw

    captions = ["Release", "Board", "Landing", "After"]
    canvas = Image.new("RGB", (4 * 336, len(chapters) * 215), "#121416")
    draw = ImageDraw.Draw(canvas)
    for j, c in enumerate(chapters):
        start = c["start_frame"] + c["move_frames"]
        _, end = highlight_span(c)
        last = end - 1
        landing = (
            c["visible_bucket_entry_frame"]
            if c["captured"]
            else c["visible_floor_frame"]
        )
        frames = [start, c["visible_board_frame"], landing, last]
        for k, (index, title) in enumerate(zip(frames, captions)):
            x, y = k * 336, j * 215
            if index is not None:
                with Image.open(frames_dir / f"{index:05d}.jpg") as original:
                    original.thumbnail((332, 187))
                    canvas.paste(original, (x, y))
            outcome = (
                "bucket"
                if c["captured"]
                else "floor" if c["visible_floor_frame"] is not None else "miss"
            )
            label = (
                f"{c['focus']}  {title}  {outcome}  |  f{index}"
                if index is not None
                else f"{c['focus']}  {title}  | no measured landing"
            )
            draw.text((x + 3, y + 191), label, fill="#dddddd")
    canvas.save(dest, quality=91)


def encode_all(run, summary, dest):
    chapters, frames = summary["chapters"], run / "frames"
    actions = [c for c in chapters if c["action_only"]]
    station = [c for c in chapters if c.get("station")]
    summary["full_movie"] = encode_movie(
        frames, dest / "dark-farm.mp4", chapters, summary["resolution"]
    )
    summary["action_movie"] = render_cut(
        frames, dest / "dark-drops.mp4", actions, summary["resolution"]
    )
    summary["station_movie"] = render_cut(
        frames, dest / "station.mp4", station, summary["resolution"]
    )
    summary["station_10x_movie"] = render_cut(
        frames, dest / "station-10x.mp4", station, summary["resolution"], step=10
    )
    contact_sheet(frames, dest / "contact-sheet.jpg", actions)
    (dest / "film.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def export(run, source, dest):
    summary = verify(run, source)
    if dest.exists():
        raise FileExistsError(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        build = Path(tmp) / "export"
        build.mkdir()
        encode_all(run, summary, build)
        build.rename(dest)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    report = export(args.run, args.source, args.output)
    print(json.dumps({k: v for k, v in report.items() if k != "chapters"}))
