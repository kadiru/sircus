"""Verify the actual training rollouts and assemble their consecutive Isaac RGB frames."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
from cylinder_farm import COLS, RADIUS, ROWS, menu_batches, origins, project_actions
from farm_contract import reward, statistics
from farm_flight import SHOTS, camera_pose, choose_front
from feasibility_contract import dynamic_cell


def read(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def matches_action(actual, expected):
    for key, value in expected.items():
        if key not in actual:
            return False
        if isinstance(value, str):
            if actual[key] != value:
                return False
        else:
            # Across NumPy builds a float32 tanh ULP can span 0.000022°
            # when mapped onto the 360° ring. Compare in the field's units.
            atol = {"azimuth_deg": 3e-5, "azimuth_offset_deg": 3e-5, "tilt_deg": 2e-5}.get(
                key, 5e-7
            )
            if not np.isclose(actual[key], value, atol=atol, rtol=0):
                return False
    return True


def verified_menus(run, cells):
    path = run / "menu.json"
    manifest = read(path)
    menus, checks = manifest["menus"], manifest["checks"]
    batches = list(menu_batches(cells))
    require(
        len(menus) == len(cells) and len(checks) == len(batches) and manifest["rule"],
        "Incomplete geometry menu",
    )
    expected = [[] for _ in cells]
    for proposals, reasons in zip(batches, checks):
        require(
            len(proposals) == len(cells) == len(reasons), "Incomplete geometry check"
        )
        for cell, (proposal, reason) in enumerate(zip(proposals, reasons)):
            if reason is None:
                expected[cell].append(proposal)
            else:
                require(
                    isinstance(reason, str) and bool(reason),
                    "Missing pose rejection reason",
                )
    for menu, accepted in zip(menus, expected):
        require(
            len(menu) == len(accepted)
            and all(
                set(a) == set(b) and matches_action(a, b)
                for a, b in zip(menu, accepted)
            ),
            "Geometry menu disagrees with its accepted endpoint proposals",
        )
    return menus, dict(
        rule=manifest["rule"],
        file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        probes=len(batches) * len(cells),
        accepted_per_cell=[len(m) for m in menus],
    )


def verify(run):
    spec, report = read(run / "contract.json"), read(run / "result.json")
    fp = spec.pop("fingerprint")
    require(
        hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest() == fp
        and fp == report["fingerprint"],
        "Scene or training contract changed",
    )
    cells = spec["cells"]
    menus, menu_audit = verified_menus(run, cells)
    require(
        menu_audit["file_sha256"] == report["menu_sha256"],
        "Menu checksum disagrees with the completed training run",
    )
    require(
        len(cells) == ROWS * COLS == report["workcells"]
        and report["articulations"] == 70
        and len({tuple(c["bucket_xy_m"]) for c in cells}) == 35
        and np.array_equal(origins(), spec["origins_m"]),
        "Wrong workspace layout",
    )
    probes = read(run / "probes.json")
    expected_kinds = [
        "bucket",
        "rim",
        "table",
        "floor",
        "robot",
        "foreign_cell_collision_filter",
    ]
    require(
        [p["kind"] for p in probes] == expected_kinds
        and all(p["all_passed"] for p in probes)
        and all(p["per_cell_passed"] == [True] * len(cells) for p in probes[:-1])
        and read(run / "rim-rays.json")["passed"] == [True] * len(cells),
        "Contact or isolation probe failed",
    )
    require(
        len(report["updates"]) == len(SHOTS) // 2
        and len(report["batches"]) == len(SHOTS),
        "Incomplete policy update or film",
    )
    current = report["initial_hash"]
    total = 0
    rows, chapters = [], []
    first_front = None
    for i, (name, frame_count, stride) in enumerate(SHOTS):
        clip = read(run / f"clip-{i:02}.json")
        batch = report["batches"][i]
        records, frames = clip["rollout"], clip["frames"]
        require(
            clip["shot"] == name
            and len(records) == 35
            and len(frames) == frame_count
            and clip["policy_hash"] == batch["policy_hash"] == current,
            "Wrong training batch",
        )
        requested = project_actions(
            cells, np.asarray(clip["latent"], dtype=np.float32), menus
        )
        require(len(clip["rewards"]) == len(records), "Missing rewards")
        geometry = []
        for cell, row, action, ret in zip(cells, records, requested, clip["rewards"]):
            require(
                row["cell_id"] == cell["id"]
                and np.allclose(row["bucket_xy_m"], cell["bucket_xy_m"])
                and matches_action(row, action)
                and abs(row["distance_m"] - RADIUS) < 1e-8,
                f"Action or bucket mismatch in {name}, cell {cell['id']}",
            )
            expected = [
                row["board_x_m"],
                row["board_y_m"],
                row["height_m"] + row["gap_m"] + spec["scene"]["ball"]["radius_m"],
            ]
            require(
                np.allclose(expected, row["release_ball_center_m"], atol=1e-7)
                and np.isclose(ret, reward(row), atol=1e-6),
                "Release or reward mismatch",
            )
            if row["route_success"]:
                require(
                    not row["invalid"]
                    and row["board_bounce"]
                    and row["whole_ball_entry"]
                    and row["max_retention_s"] >= 1 - 1e-8
                    and row["external_contact_steps"] == 0,
                    "Unqualified capture",
                )
            geometry.append(dynamic_cell(cell, action))
        require(
            all(np.isclose(batch[k], v) for k, v in statistics(records).items()),
            "Batch outcome totals disagree",
        )
        focus = choose_front(
            [not r["invalid"] for r in records],
            name,
            first_front if name == "front-right" else None,
        )
        require(focus == clip["focus"], "Front table selected inconsistently")
        focus_row = records[focus] if focus is not None else {}
        require(
            batch["shot"] == name
            and batch["start_frame"] == total
            and batch["frames"] == frame_count
            and batch["physics_stride"] == stride
            and batch["focus"] == focus_row.get("cell_id")
            and batch["focus_bounce"] == focus_row.get("board_bounce")
            and batch["focus_capture"] == focus_row.get("route_success"),
            "Shot log disagrees with rollout",
        )
        if name == "front-left":
            first_front = focus
        for k, frame in enumerate(frames):
            eye, target = camera_pose(i, k, geometry, focus)
            require(
                frame["frame"] == total + k
                and frame["physics_step"] == frames[0]["physics_step"] + k * stride
                and np.allclose(eye, frame["eye_m"], atol=2e-4)
                and np.allclose(target, frame["target_m"], atol=2e-5),
                "Camera or physics frame mismatch",
            )
        chapters.append(
            dict(
                shot=name,
                start_frame=total,
                frames=frame_count,
                start_s=total / 30,
                focus=batch["focus"],
                board_bounce=batch["focus_bounce"],
                captured=batch["focus_capture"],
            )
        )
        rows.extend(records)
        total += frame_count
        if i % 2:
            change = report["updates"][i // 2]
            require(
                current == change["before"] and current != change["after"],
                "PPO weights failed to update from filmed batches",
            )
            current = change["after"]
    require(
        current == report["final_hash"]
        and report["frames"] == total
        and [p.name for p in sorted((run / "frames").glob("*.jpg"))]
        == [f"{i:05d}.jpg" for i in range(total)],
        "Incomplete frames or checkpoint chain",
    )
    bucket = spec["scene"]["bucket"]
    rim = bucket["assumed_interior_depth_m"] + bucket["assumed_bottom_thickness_m"]
    low = [r for r in rows if not r["invalid"] and r["settled_board_center_m"][2] < rim]
    return dict(
        version=report["version"],
        scene_fingerprint=fp,
        gpu=report["gpu"],
        rows=ROWS,
        columns=COLS,
        arms=report["articulations"],
        radius_mm=1000 * RADIUS,
        training=statistics(rows),
        releases=sum(not r["invalid"] for r in rows),
        geometry_menu=menu_audit,
        board_bounces=sum(r["board_bounce"] for r in rows),
        low_board_releases=len(low),
        low_board_captures=sum(r["route_success"] for r in low),
        ppo_updates=len(report["updates"]),
        final_weights_sha256=current,
        checkpoint_file_sha256=hashlib.sha256(
            (run / "cylinder-policy.pt").read_bytes()
        ).hexdigest(),
        fps=30,
        frames=total,
        duration_s=total / 30,
        resolution=report["resolution"],
        chapters=chapters,
    )


def encode_movie(frames_dir, movie, chapters, resolution):
    """Encode a contiguous interval of raw frames with chapters relative to the cut."""
    start = chapters[0]["start_frame"]
    count = 0
    meta = [";FFMETADATA1"]
    titles = {
        "orbit-start": "Orbit around the farm",
        "orbit-back": "Far side of the farm",
        "orbit-return": "Return to the front row",
        "descend": "Descent toward the tables",
        "through-aisle": "Flight between workcells",
        "front-left": "First front-row drop",
        "front-right": "Second front-row drop",
        "overhead-field": "The 35 tables from above",
        "overhead-bounce": "Single table from above",
        "aerial-exit": "Departure over the farm",
    }
    for c in chapters:
        require(c["start_frame"] == start + count, "Chapter sequence is not contiguous")
        title = c["title"] if "title" in c else titles[c["shot"]]
        if c.get("focus"):
            title += f" ({c['focus']})"
        meta.extend(
            [
                "[CHAPTER]",
                "TIMEBASE=1/30",
                f"START={count}",
                f"END={count+c['frames']}",
                f"title={title}",
            ]
        )
        count += c["frames"]
    metadata = movie.with_suffix(".ffmeta")
    metadata.write_text("\n".join(meta) + "\n")
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-n",
            "-framerate",
            "30",
            "-start_number",
            str(start),
            "-i",
            str(frames_dir / "%05d.jpg"),
            "-i",
            str(metadata),
            "-map",
            "0:v:0",
            "-map_metadata",
            "1",
            "-map_chapters",
            "1",
            "-an",
            "-c:v",
            "libx264",
            "-frames:v",
            str(count),
            "-pix_fmt",
            "yuv420p",
            "-threads",
            "4",
            "-preset",
            "medium",
            "-crf",
            "17",
            "-movflags",
            "+faststart",
            str(movie),
        ],
        check=True,
    )
    data = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-count_frames",
                "-show_streams",
                "-show_chapters",
                "-of",
                "json",
                str(movie),
            ]
        )
    )
    stream = next(s for s in data["streams"] if s["codec_type"] == "video")
    require(
        int(stream["nb_read_frames"]) == count
        and stream["avg_frame_rate"] == "30/1"
        and stream["codec_name"] == "h264"
        and [stream["width"], stream["height"]] == resolution
        and len(data["chapters"]) == len(chapters),
        "Encoded movie did not match the camera log",
    )
    for c, encoded in zip(chapters, data["chapters"]):
        expected_start = (c["start_frame"] - start) / 30
        require(
            abs(float(encoded["start_time"]) - expected_start) < 0.002
            and abs(float(encoded["end_time"]) - expected_start - c["frames"] / 30)
            < 0.002,
            "Encoded chapter timestamps disagree with raw frames",
        )
        require(
            encoded.get("tags", {}).get("title", "").startswith(
                c["title"] if "title" in c else titles[c["shot"]]
            ),
            "Encoded chapter title is missing or incorrect",
        )
    return dict(
        file=movie.name,
        sha256=hashlib.sha256(movie.read_bytes()).hexdigest(),
        source_start_frame=start,
        frames=count,
        duration_s=count / 30,
    )


def export(run, out):
    summary = verify(run)
    out.mkdir(parents=True, exist_ok=False)
    full = encode_movie(
        run / "frames",
        out / "sircus-drone-farm.mp4",
        summary["chapters"],
        summary["resolution"],
    )
    front = [
        c for c in summary["chapters"] if c["shot"] in ("front-left", "front-right")
    ]
    require(
        len(front) == 2 and front[0]["focus"] != front[1]["focus"],
        "Missing distinct front-row shots",
    )
    summary["movie_sha256"] = full["sha256"]
    summary["front_row_movie"] = encode_movie(
        run / "frames", out / "sircus-front-row.mp4", front, summary["resolution"]
    )
    # Preserve the original pixels of the release thumbnail for inspection.
    poster = front[0]["start_frame"] + 18
    shutil.copyfile(run / "frames" / f"{poster:05d}.jpg", out / "front-row.jpg")
    (out / "film.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "chapters"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    a = parser.parse_args()
    export(a.run, a.output)
