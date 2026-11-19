"""CPU-only, native-geometry movie of four audited height/tilt recordings.

This replays saved ball centers and held arm joints. It never integrates physics,
interpolates trajectories, invents arm motion, or changes the historical scores.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path

import numpy as np

from .saved_shot_video import (SOURCE_MODULES, build_held_scene, contained, digest,
                              nearest_tick, read, require, sha, software_backend,
                              vector, verify_assets)

VERSION = "sircus-height-tilt-recorded-video-v1"
FPS = 30
CONTACTS = ("board", "bucket", "table", "floor", "robot", "unattributed")
RECORDS = (
    ("height-tilt-table-training-test-00-g23", "Case 00 / board, robot, bucket / no entry"),
    ("height-tilt-table-training-test-18-g23", "Case 18 / board, robot, table / no entry"),
    ("height-tilt-table-training-test-01-g17", "Case 01 / no robot contact / no entry"),
    ("height-tilt-controls_pre-control-table_control-r0", "Positive control / table-assisted capture"),
)


def first_events(times, forces, row):
    """Forces are logged at the pre-integration state at the same native tick."""
    stop = int(row["physics_steps"])
    require(forces.shape == (len(times) - 1, 6) or forces.shape == (len(times), 6),
            "Native force categories or shape changed")
    require(0 < stop < len(times) and stop <= len(forces), "Scored native terminal is outside recording")
    events = {}
    for index, category in enumerate(CONTACTS):
        hits = np.flatnonzero(forces[:stop, index] > .01)
        if len(hits):
            events[category] = float(times[int(hits[0])])
    for key, field in (("whole_ball_entry", "whole_ball_entry_time_s"), ("capture", "capture_time_s")):
        value = row.get(field)
        if value is not None:
            require(math.isfinite(value) and 0 <= value <= times[stop], "Scored event outside scored interval")
            events[key] = float(value)
    return events


def load_recordings(a):
    study = Path(a.study).resolve()
    summary = read(a.summary)
    provenance = summary["provenance"]
    paths = {"report": study / "report.json", "audit": Path(a.audit),
             "rows": study / "rows.jsonl", "contract": study / "contract.json",
             "admission_matrix": study / "admission-matrix.json",
             "evidence_manifest": study / "evidence-manifest.json"}
    for name, path in paths.items():
        require(sha(path) == provenance[name + "_sha256"], "Audited bytes changed: " + name)
    report, audit, contract, manifest = (read(paths[name]) for name in ("report", "audit", "contract", "evidence_manifest"))
    require(report["status"] == "completed" and report["phase"] == "completed", "Completed native study required")
    require(audit["status"] == "passed" and audit["full_checks_verified"] is True
            and audit["native_checks_verified"] is True and audit["failures"] == [], "Passed full native audit required")
    require(audit["checks_count"] == len(audit["checks"]) > 0
            and all(c["passed"] is True for c in audit["checks"]), "All independent checks must pass")
    unsigned = dict(contract)
    fingerprint = unsigned.pop("fingerprint")
    require(fingerprint == digest(unsigned) == provenance["contract_fingerprint"]
            == audit["contract_fingerprint"] == report["contract_fingerprint"], "Contract fingerprint mismatch")
    require(audit["report_sha256"] == provenance["report_sha256"]
            and audit["evidence_manifest_sha256"] == provenance["evidence_manifest_sha256"]
            and audit["admission_matrix_sha256"] == provenance["admission_matrix_sha256"], "Audit does not bind study")
    require(report["evidence_sha256"] == manifest, "Report and manifest disagree")
    for name in ("rows", "contract", "admission_matrix"):
        require(manifest[paths[name].name] == provenance[name + "_sha256"], "Manifest changed: " + name)
    scene = contract["scene"]
    require(digest(scene) == contract["identity"]["scene_sha256"], "Original scene changed")
    assets = verify_assets(a.assets, contract["identity"]["asset_sha256"])
    repository = Path(__file__).resolve().parents[2]
    sources = {name: sha(repository / name) for name in SOURCE_MODULES}
    for name, value in sources.items():
        require(contract["identity"]["source_sha256"].get(name) == value, "Original model source changed: " + name)
    rows = [json.loads(line) for line in paths["rows"].read_text().splitlines() if line.strip()]
    require(len({r["id"] for r in rows}) == len(rows), "Duplicate source record IDs")
    lookup = {row["id"]: row for row in rows}
    bundles = []
    dt = contract["identity"]["trace_schema"]["native_dt_s"]
    require(dt == 1 / 15360, "Canonical native64 timestep required")
    for identifier, label in RECORDS:
        require(identifier in lookup, "Selected record missing: " + identifier)
        row = lookup[identifier]
        require(row.get("native_release") is True and row.get("invalid_physics") is False
                and row.get("contact_protocol", {}).get("substeps") == 64, "Valid native64 release required")
        cache = contained(study / "native", row["native_cache_file"])
        cache_hash = sha(cache)
        require(cache_hash == row["native_cache_sha256"] == manifest["native/" + row["native_cache_file"]],
                "Exact selected archive changed")
        column = row["native_cache_row"]
        require(type(column) is int and column >= 0, "Exact archive column required")
        with np.load(cache, allow_pickle=False) as archive:
            require(column < len(archive["record_ids"]) and str(archive["record_ids"][column]) == identifier,
                    "Archive record identity mismatch")
            times = np.asarray(archive["time_s"], dtype=float)
            positions = np.asarray(archive["position_m"][:, column], dtype=float)
            forces = np.asarray(archive["forces_N"][:, column], dtype=float)
        require(times.ndim == 1 and len(times) > 1 and np.isfinite(times).all()
                and np.allclose(times, np.arange(len(times)) * dt, rtol=0, atol=1e-12), "Recorded native clock changed")
        require(positions.shape == (len(times), 3) and np.isfinite(positions).all() and np.isfinite(forces).all(),
                "Finite recorded trajectory and contacts required")
        require(np.array_equal(positions[0], np.asarray(row["native_initial_ball_center_m"])), "Release center differs")
        for key in ("q10", "q14"):
            vector(row["native_initial_" + key], 6, key)
        events = first_events(times, forces, row)
        if identifier == RECORDS[-1][0]:
            require(row.get("route_success") is True and row.get("table_assisted_capture") is True
                    and "capture" in events, "Expected positive table-assisted control required")
        else:
            require(row.get("route_success") is False and row.get("whole_ball_entry") is False,
                    "Expected no-entry diagnostic miss required")
        if identifier in (RECORDS[0][0], RECORDS[1][0]):
            require("robot" in events and events["robot"] > events["board"], "Expected post-board robot contact required")
        if identifier == RECORDS[2][0]:
            require("robot" not in events, "Expected no-robot comparison required")
        bundles.append(dict(contract=contract, scene=scene, row=row, label=label, time_s=times,
                            positions=positions, events=events, record_sha256=digest(row), cache_sha256=cache_hash,
                            native_dt_s=dt, scored_until_s=float(times[int(row["physics_steps"])])))
    shared = dict(source={k: provenance[k] for k in ("report_sha256", "audit_sha256", "contract_fingerprint")},
                  input_sha256={**{key: sha(path) for key, path in paths.items()}, "summary": sha(a.summary)},
                  asset_sha256=assets, model_source_sha256=sources, scene_sha256=digest(scene))
    return bundles, shared


def frame_plan(bundles, *, fps=FPS):
    """Actual native samples only, including exact contact freezes between samples."""
    require(type(fps) is int and 1 <= fps <= 60, "Bounded positive integer FPS required")
    frames, clips = [], []
    for clip_index, bundle in enumerate(bundles):
        row, times, events = bundle["row"], bundle["time_s"], bundle["events"]
        captured = row["route_success"]
        end = bundle["scored_until_s"]
        require(0 < end <= times[-1], "Scored native terminal is outside recording")
        start = len(frames) / fps

        def add(segment, view, speed, requested, count=1):
            tick = nearest_tick(times, requested)
            require(float(times[tick]) <= end, "Frame exceeds scored native interval")
            for _ in range(count):
                frames.append(dict(frame=len(frames), video_time_s=len(frames) / fps, clip_index=clip_index,
                                   record_id=row["id"], segment=segment, view=view, playback_speed=speed,
                                   source_tick=tick, source_time_s=float(times[tick])))

        add("title_hold", "overview", 0., 0., round(.6 * fps))
        for i in range(math.ceil(end * fps)):
            add("normal", "overview", 1., min(end, i / fps))
        freeze = events.get("robot")
        frozen = False
        for i in range(math.ceil(end / .25 * fps)):
            wanted = min(end, i / fps * .25)
            if freeze is not None and not frozen and wanted >= freeze:
                add("robot_contact_hold", "close", 0., freeze, round(.7 * fps))
                frozen = True
            add("slow", "close", .25, wanted)
        add("outcome_hold", "close", 0., end, round(.7 * fps))
        clips.append(dict(record_id=row["id"], context_id=row["context_id"], label=bundle["label"],
                          start_s=start, end_s=len(frames) / fps, values=row["values"],
                          bucket_xy_m=row["bucket_xy_m"], capture=captured, whole_ball_entry=row["whole_ball_entry"],
                          events=events, source_interval_s=[0., end], scored_until_s=bundle["scored_until_s"],
                          record_sha256=bundle["record_sha256"], native_cache_sha256=bundle["cache_sha256"],
                          native_cache_row=row["native_cache_row"],
                          held_joints_rad={k: row["native_initial_" + k] for k in ("q10", "q14")}))
    require(len(frames) <= 60 * fps, "Movie exceeds bounded one-minute duration")
    return frames, clips


def cameras(bundle):
    board = np.asarray(bundle["row"]["board_center_m"])
    bucket = np.asarray([*bundle["row"]["bucket_xy_m"], 0.])
    direction = bucket[:2] - board[:2]
    side = np.asarray([direction[1], -direction[0]])
    azimuth = math.degrees(math.atan2(side[1], side[0]))
    target = (.5 * (board + bucket)).tolist()
    target[2] = max(.30, float(board[2]) * .75)
    return dict(overview=dict(lookat=[.65, -.30, .40], distance=2.6, azimuth=azimuth + 20., elevation=-28.),
                close=dict(lookat=target, distance=1.65, azimuth=azimuth, elevation=-18.),
                reverse=dict(lookat=target, distance=1.65, azimuth=azimuth + 180., elevation=-18.))


def stage(item, bundle):
    if item["segment"] == "robot_contact_hold":
        return "Recorded robot contact (>0.01 N)"
    if item["segment"] == "outcome_hold":
        return "Audited outcome: one-second capture" if bundle["row"]["route_success"] else "Audited outcome: no whole-ball entry"
    events, time = bundle["events"], item["source_time_s"]
    if "capture" in events and time >= events["capture"]:
        return "CAPTURE / retained inside for one continuous second"
    if "whole_ball_entry" in events and time >= events["whole_ball_entry"]:
        return "Whole ball inside / retention clock running"
    occurred = [(value, name) for name, value in events.items() if name in CONTACTS and value <= time]
    if occurred:
        return "Recorded contact: " + max(occurred)[1]
    return "Release / falling toward board"


def render(a):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Hide both GPUs before importing the renderer")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "LP_NUM_THREADS"):
        os.environ[name] = "2"
    require(a.width >= 640 and a.height >= 360 and a.width <= 1920 and a.height <= 1080
            and a.width % 2 == a.height % 2 == 0, "Even dimensions between640x360 and1920x1080 required")
    output = Path(a.output).resolve()
    require(output.suffix == ".mp4", "MP4 output required")
    for path in (a.study, a.assets):
        require(not output.is_relative_to(Path(path).resolve()), "Keep rendered outputs outside original evidence/assets")
    manifest_path = output.with_suffix(".preview.json" if a.preview_only else ".json")
    preview_dir = output.with_name(output.stem + "-previews")
    require(not manifest_path.exists() and (a.preview_only or not output.exists()), "Refusing to overwrite rendered evidence")
    bundles, shared = load_recordings(a)
    plan, clips = frame_plan(bundles)
    overrides = {} if a.cameras is None else read(a.cameras)
    output.parent.mkdir(parents=True, exist_ok=True)
    if a.preview_only:
        preview_dir.mkdir(exist_ok=False)
    import mujoco
    from OpenGL import GL
    from PIL import Image, ImageDraw, ImageFont
    import imageio.v2 as imageio
    fonts = []
    for size in (27, 21, 18):
        paths = ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf")
        fonts.append(next((ImageFont.truetype(p, size) for p in paths if Path(p).is_file()), ImageFont.load_default()))
    previews, model_hashes, view_settings = [], {}, {}
    writer, renderer, backend = None, None, None
    try:
        if not a.preview_only:
            writer = imageio.get_writer(output, fps=FPS, codec="libx264", quality=8, macro_block_size=1,
                                       ffmpeg_params=["-threads", "2", "-movflags", "+faststart"])
        for index, bundle in enumerate(bundles):
            model, data, ball_q, model_hash = build_held_scene(bundle, a.assets)
            model_hashes[bundle["row"]["id"]] = model_hash
            model.vis.global_.offwidth = max(a.width, model.vis.global_.offwidth)
            model.vis.global_.offheight = max(a.height, model.vis.global_.offheight)
            renderer = mujoco.Renderer(model, height=a.height, width=a.width)
            backend = software_backend((GL.glGetString(GL.GL_RENDERER) or b"").decode(),
                                       (GL.glGetString(GL.GL_VENDOR) or b"").decode(), os.environ.get("CUDA_VISIBLE_DEVICES"))
            views = cameras(bundle)
            views.update(overrides.get(bundle["row"]["id"], {}))
            for view in views.values():
                vector(view["lookat"], 3, "camera target")
                require(all(math.isfinite(view[k]) for k in ("distance", "azimuth", "elevation"))
                        and view["distance"] > 0, "Finite camera settings required")
            view_settings[bundle["row"]["id"]] = views
            camera = mujoco.MjvCamera()
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE

            def frame(item):
                data.qpos[ball_q:ball_q + 3] = bundle["positions"][item["source_tick"]]
                data.time = item["source_time_s"]
                mujoco.mj_forward(model, data)
                settings = views[item["view"]]
                camera.lookat[:] = settings["lookat"]
                for key in ("distance", "azimuth", "elevation"):
                    setattr(camera, key, settings[key])
                renderer.update_scene(data, camera=camera)
                image = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(image)
                draw.rectangle((0, 0, a.width, 106), fill="#101722")
                draw.text((20, 9), bundle["label"], font=fonts[0], fill="white")
                speed = "HOLD" if item["playback_speed"] == 0 else f'{item["playback_speed"]:g}x'
                draw.text((20, 45), f'{speed} | actual time {item["source_time_s"]:.4f}s | {stage(item, bundle)}',
                          font=fonts[1], fill="#85dec2")
                values = bundle["row"]["values"]
                draw.text((20, 76), f'Board center {values[2]*1000:.1f} mm | tilt {values[3]:.2f} deg | drop gap {values[4]*1000:.1f} mm',
                          font=fonts[2], fill="#c7d2df")
                draw.rectangle((0, a.height - 54, a.width, a.height), fill="#101722")
                draw.text((20, a.height - 46), "Saved MuJoCo/mjlab simulation / held arm poses / no new physics or synthesized motion", font=fonts[2], fill="#c7d2df")
                draw.text((20, a.height - 24), f'Native64 | robot contact is an aggregate category | {bundle["row"]["id"]}', font=fonts[2], fill="#c7d2df")
                return np.asarray(image)

            if a.preview_only:
                events = {"release": 0., **{key: value for key, value in bundle["events"].items()
                                          if key in ("board", "robot", "table", "bucket", "capture")}}
                for view in views:
                    for name, when in events.items():
                        tick = nearest_tick(bundle["time_s"], when)
                        item = dict(view=view, source_tick=tick, source_time_s=float(bundle["time_s"][tick]),
                                    playback_speed=0., segment="robot_contact_hold" if name == "robot" else "preview")
                        path = preview_dir / f"{index}-{view}-{name}.jpg"
                        Image.fromarray(frame(item)).save(path, quality=94)
                        previews.append(dict(name=path.name, sha256=sha(path), record_id=bundle["row"]["id"], **item))
            else:
                for item in plan:
                    if item["clip_index"] == index:
                        writer.append_data(frame(item))
            renderer.close()
            renderer = None
        if writer is not None:
            writer.close()
            writer = None
    finally:
        if renderer is not None:
            renderer.close()
        if writer is not None:
            writer.close()
    result = dict(version=VERSION, status="preview_complete" if a.preview_only else "completed", **shared,
                  renderer_source_sha256=sha(__file__),
                  helper_source_sha256={"sircus/mjlab/saved_shot_video.py": sha(Path(__file__).with_name("saved_shot_video.py"))},
                  full_model_xml_sha256=model_hashes, backend=backend,
                  packages={name: importlib.metadata.version(name) for name in ("mujoco", "numpy", "Pillow", "imageio")},
                  width=a.width, height=a.height, fps=FPS, duration_s=len(plan) / FPS, clips=clips,
                  cameras=view_settings, camera_overrides_sha256=None if a.cameras is None else sha(a.cameras),
                  native_dt_s=1 / 15360, native_steps_added=0, held_arm_poses=True,
                  sampled_ball_centers_only=True, ball_spin_recorded=False, interpolation=False,
                  historical_labels_unchanged=True, contact_threshold_N=.01, contact_force_order=list(CONTACTS),
                  frames=[] if a.preview_only else plan, frames_sha256=None if a.preview_only else digest(plan),
                  output_sha256=None if a.preview_only else sha(output), video_bytes=None if a.preview_only else output.stat().st_size,
                  previews=previews, scope="Three inspected development misses and one positive control; recorded native64 replay, not policy validation.")
    manifest_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(status=result["status"], manifest=str(manifest_path), duration_s=result["duration_s"], previews=len(previews))))
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "audit", "summary", "assets", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--cameras", type=Path, help="Optional record-ID to camera-name settings JSON; camera changes only")
    p.add_argument("--preview-only", action="store_true")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    return p


if __name__ == "__main__":
    render(parser().parse_args())
