"""Render a hash-verified historical ball trace without running new physics.

Only recorded ball-center samples are displayed. The native experiment held
both arms at saved endpoint joints; no unrecorded arm motion is invented here.
GL backend selection is external and must report a software renderer.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

VERSION = "sircus-recorded-table-assisted-video-v1"
FPS = 30
END_S = 2.5
SOURCE_MODULES = ("sircus/mjlab/model.py", "sircus/isaac/geometry.py")


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(value, label):
    if not value:
        raise ValueError(label)


def vector(value, size, label):
    array = np.asarray(value, dtype=float)
    require(array.shape == (size,) and np.isfinite(array).all(), "Finite " + label + " required")
    return array


def contained(root, name):
    root = Path(root).resolve()
    require(isinstance(name, str) and name and not Path(name).is_absolute() and ".." not in Path(name).parts,
            "Relative evidence or asset path required")
    path = (root / name).resolve()
    require(path.is_relative_to(root) and path.is_file(), "Missing or escaped input: " + name)
    return path


def verify_assets(assets, expected):
    """Bind every URDF, seed file and collision mesh used by the full model."""
    assets = Path(assets).resolve()
    root = assets.parent
    paths = {assets / name for name in ("dropper.urdf", "board_arm.urdf", "poses.json")}
    for name in ("dropper.urdf", "board_arm.urdf"):
        for mesh in ET.parse(assets / name).findall(".//collision/geometry/mesh"):
            filename = mesh.get("filename", "")
            if filename.startswith("/workspace/"):
                path = root / filename.removeprefix("/workspace/")
            else:
                path = Path(filename)
                if not path.is_absolute():
                    path = assets / path
            paths.add(path.resolve())
    require(all(p.is_relative_to(root) for p in paths), "Collision assets escaped their declared tree")
    actual = {str(p.relative_to(root)): sha(p) for p in sorted(paths)}
    require(actual == expected, "Original complete collision assets changed")
    return actual


def nearest_tick(times, wanted):
    """Select an actual recorded sample, choosing the earlier sample on a tie."""
    require(math.isfinite(wanted) and times[0] <= wanted <= times[-1], "Requested frame is outside recorded time")
    right = int(np.searchsorted(times, wanted, side="left"))
    if right == 0:
        return 0
    if right == len(times):
        return len(times) - 1
    return right - 1 if wanted - times[right - 1] <= times[right] - wanted else right


def frame_plan(times, *, fps=FPS, end_s=END_S):
    """15 seconds: overview1x,0.5s hold,close-up0.25x,2s final hold."""
    require(type(fps) is int and fps > 0 and 0 < end_s <= times[-1], "Valid recorded timeline required")
    result = []
    segments = (("normal", "overview", 1., end_s, False),
                ("normal_hold", "overview", 0., .5, True),
                ("slow", "close", .25, end_s / .25, False),
                ("capture_hold", "close", 0., 2., True))
    for name, view, speed, duration, hold in segments:
        for i in range(round(duration * fps)):
            requested = end_s if hold else min(end_s, i / fps * speed)
            tick = nearest_tick(times, requested)
            result.append(dict(frame=len(result), video_time_s=len(result) / fps, segment=name,
                               view=view, playback_speed=speed, requested_source_time_s=requested,
                               source_tick=tick, source_time_s=float(times[tick])))
    return result


def software_backend(renderer_string, vendor_string, cuda_visible):
    text = renderer_string.lower()
    require(any(name in text for name in ("llvmpipe", "softpipe", "swrast")),
            "Software OpenGL required; refusing renderer " + renderer_string)
    require(cuda_visible == "", "Replay requires CUDA_VISIBLE_DEVICES='' so neither GPU is used")
    return dict(renderer=renderer_string, vendor=vendor_string, software_only=True,
                cuda_visible_devices=cuda_visible, backend=os.environ.get("MUJOCO_GL", ""))


def load_recording(a):
    evidence = read(a.evidence)
    require(evidence.get("version") == "sircus-table-assisted-history-example-v1"
            and evidence.get("status") == "verified_historical_example", "Verified historical example required")
    provenance = evidence["provenance"]
    contract_path = Path(a.study) / "contract.json"
    require(sha(contract_path) == provenance["original_contract_sha256"], "Original contract bytes changed")
    contract = read(contract_path)
    unsigned = dict(contract)
    require(unsigned.pop("fingerprint") == digest(unsigned), "Original contract fingerprint changed")
    calibration_path = Path(a.study) / "calibration.json"
    require(sha(calibration_path) == provenance["original_calibration_file_sha256"], "Original calibration changed")
    source = read(a.scene)
    scene = source.get("scene", source)
    require(digest(scene) == contract["identity"]["scene_sha256"], "Scene differs from historical simulation")
    asset_hashes = verify_assets(a.assets, contract["identity"]["asset_sha256"])
    repository = Path(__file__).resolve().parents[2]
    source_hashes = {name: sha(repository / name) for name in SOURCE_MODULES}
    for name, value in source_hashes.items():
        require(contract["identity"]["source_sha256"].get(name) == value, "Historical model source differs: " + name)
    require(sha(a.metadata) == provenance["metadata"]["sha256"], "Historical metadata bytes changed")
    lines = Path(a.metadata).read_text().splitlines()
    line = provenance["metadata"]["line"]
    require(type(line) is int and 1 <= line <= len(lines), "Original metadata line missing")
    row = json.loads(lines[line - 1])
    require(row["id"] == evidence["shot"]["id"], "Historical shot ID mismatch")
    expected_hash = provenance["native_archive"]["sha256"]
    require(sha(a.cache) == expected_hash == row["native_cache_sha256"], "Historical native cache bytes changed")
    require(Path(a.cache).name == row["native_cache_file"], "Historical native cache filename differs")
    column = provenance["native_archive"]["row"]
    require(type(column) is int and column >= 0 and column == row["native_cache_row"], "Historical native row mismatch")
    with np.load(a.cache, allow_pickle=False) as archive:
        ids = archive["record_ids"]
        require(column < len(ids) and str(ids[column]) == row["id"], "Archive record identity mismatch")
        time_s = np.asarray(archive["time_s"], dtype=float)
        positions = np.asarray(archive["position_m"][:, column], dtype=float)
    require(time_s.ndim == 1 and len(time_s) > 1 and np.isfinite(time_s).all()
            and time_s[0] == 0 and np.all(np.diff(time_s) > 0), "Monotonic recorded clock required")
    require(positions.shape == (len(time_s), 3) and np.isfinite(positions).all(), "Finite recorded ball centers required")
    dt = contract["identity"]["trace_schema"]["native_dt_s"]
    require(dt == 1 / 15360 and np.allclose(time_s, np.arange(len(time_s)) * dt, atol=1e-12, rtol=0),
            "Canonical fixed64 native ticks changed")
    require(time_s[-1] >= END_S and evidence["events"]["capture_s"] <= END_S,
            "Complete capture and requested replay interval required")
    require(row.get("native_release") is True and row.get("invalid_physics") is False,
            "Valid executed historical shot required")
    require(row.get("contact_protocol", {}).get("substeps") == 64, "Historical fixed64 protocol required")
    for key in ("success", "complete_success", "route_success", "board_bounce", "whole_ball_entry", "max_retention_s", "invalid_physics"):
        require(row[key] == evidence["original_score"][key], "Original score changed: " + key)
    require(np.allclose(positions[0], row["native_initial_ball_center_m"], atol=0, rtol=0), "Recorded release center differs")
    require(np.allclose(positions[0], evidence["shot"]["release_ball_center_m"], atol=1e-6, rtol=0), "Evidence release center differs")
    for key in ("q10", "q14"):
        vector(row["native_initial_" + key], 6, key)
    for key in ("bucket_xy_m", "board_center_m"):
        wanted = evidence["shot"]["settled_board_center_m" if key == "board_center_m" else key]
        require(np.allclose(row[key], wanted, atol=1e-9, rtol=0), "Evidence geometry differs: " + key)
    return dict(evidence=evidence, contract=contract, scene=scene, row=row, time_s=time_s, positions=positions,
                input_sha256=dict(contract=sha(contract_path), calibration=sha(calibration_path),
                                  metadata=sha(a.metadata), cache=expected_hash, evidence=sha(a.evidence), scene_file=sha(a.scene)),
                asset_sha256=asset_hashes, model_source_sha256=source_hashes,
                scene_sha256=digest(scene), native_dt_s=dt)


def build_held_scene(bundle, assets):
    import mujoco
    from .model import build_scene
    require(importlib.metadata.version("mujoco") == bundle["contract"]["packages"]["mujoco"],
            "Use historical pinned MuJoCo version for compiled geometry")
    xml = build_scene(bundle["scene"], assets, solref=(.004, .15), contact_friction=.35)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    row = bundle["row"]
    for role, key in (("Arm10", "q10"), ("Arm14", "q14")):
        for j, value in enumerate(row["native_initial_" + key], 1):
            data.qpos[model.jnt_qposadr[model.joint(f"{role}/joint{j}").id]] = value
    for side in ("left", "right"):
        data.qpos[model.jnt_qposadr[model.joint(f"Arm10/finger_{side}_joint").id]] = float(np.float32(.025))
    ball_q = int(model.jnt_qposadr[model.joint("ball/free").id])
    data.qpos[ball_q:ball_q + 3] = bundle["positions"][0]
    data.qpos[ball_q + 3:ball_q + 7] = [1., 0., 0., 0.]
    bucket_id = model.body("bucket").id
    data.mocap_pos[model.body_mocapid[bucket_id]] = row["native_initial_bucket_center_m"]
    data.mocap_quat[model.body_mocapid[bucket_id]] = [1., 0., 0., 0.]
    mujoco.mj_forward(model, data)
    board_id = model.body("Arm14/impact_board").id
    require(np.allclose(data.xpos[board_id], row["board_center_m"], atol=2e-6, rtol=0), "Rendered board center differs from native pose")
    require(np.allclose(data.xmat[board_id].reshape(3, 3)[:, 2], row["board_normal"], atol=2e-6, rtol=0), "Rendered board normal differs")
    require(np.allclose(data.xpos[bucket_id], [*row["bucket_xy_m"], 0], atol=1e-6, rtol=0), "Rendered bucket differs")
    return model, data, ball_q, hashlib.sha256(xml.encode()).hexdigest()


def cameras(bundle):
    board = np.asarray(bundle["row"]["board_center_m"])
    bucket = np.asarray([*bundle["row"]["bucket_xy_m"], 0.])
    direction = bucket[:2] - board[:2]
    side = np.asarray([direction[1], -direction[0]])
    azimuth = math.degrees(math.atan2(side[1], side[0]))
    return dict(overview=dict(lookat=[.65, -.30, .31], distance=2.7, azimuth=azimuth + 20., elevation=-32.),
                close=dict(lookat=[*(.5 * (board[:2] + bucket[:2])), .25], distance=1.55, azimuth=azimuth, elevation=-30.))


def stage(time_s, events):
    if time_s >= events["capture_s"]:
        return "CAPTURED  /  one second inside"
    if time_s >= events["whole_ball_entry_s"]:
        return "Whole ball inside  /  retention clock running"
    if time_s >= events["first_bucket_contact_s"]:
        return "Bucket and rim contacts"
    if time_s >= events["table_bounce"]["start_s"]:
        return "Table bounce  /  ball rises toward bucket"
    if time_s >= events["board_impact"]["start_s"]:
        return "Board bounce  /  ball heads toward table"
    return "Ball released  /  falling toward board"


def render(a):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Hide both GPUs before importing the renderer")
    require(a.width >= 640 and a.height >= 360 and a.width % 2 == a.height % 2 == 0, "Even video dimensions of at least640x360 required")
    output = Path(a.output).resolve()
    for path in (a.study, a.assets):
        require(not output.is_relative_to(Path(path).resolve()), "Video must stay outside original evidence/assets")
    require(output.suffix.lower() == ".mp4", "MP4 output required")
    manifest_path = output.with_suffix(".preview.json" if a.preview_only else ".json")
    preview_dir = output.with_name(output.stem + "-previews")
    require(not manifest_path.exists() and (a.preview_only or not output.exists()), "Refusing to overwrite rendered evidence")
    bundle = load_recording(a)
    model, data, ball_q, model_sha = build_held_scene(bundle, a.assets)
    import mujoco
    from OpenGL import GL
    from PIL import Image, ImageDraw, ImageFont
    import imageio.v2 as imageio
    output.parent.mkdir(parents=True, exist_ok=True)
    model.vis.global_.offwidth = max(a.width, model.vis.global_.offwidth)
    model.vis.global_.offheight = max(a.height, model.vis.global_.offheight)
    views = cameras(bundle)
    renderer = mujoco.Renderer(model, height=a.height, width=a.width)
    backend = software_backend((GL.glGetString(GL.GL_RENDERER) or b"").decode(),
                               (GL.glGetString(GL.GL_VENDOR) or b"").decode(), os.environ.get("CUDA_VISIBLE_DEVICES"))
    fonts = []
    for size in (28, 23, 19):
        font = None
        for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf"):
            if Path(path).is_file():
                font = ImageFont.truetype(path, size); break
        fonts.append(font or ImageFont.load_default())
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
        draw.rectangle((0, 0, a.width, 104), fill="#101722")
        draw.text((22, 12), "Table-assisted capture / recorded simulation replay", font=fonts[0], fill="white")
        speed = "HOLD" if item["playback_speed"] == 0 else f'{item["playback_speed"]:g}x playback'
        draw.text((22, 51), f'{speed}   |   native time {item["source_time_s"]:.3f}s   |   '+stage(item["source_time_s"], bundle["evidence"]["events"]), font=fonts[1], fill="#7ce9b1")
        draw.rectangle((0, a.height - 65, a.width, a.height), fill="#101722")
        draw.text((22, a.height - 57), "Recorded ball samples; held arm poses. No new physics or invented arm motion.", font=fonts[2], fill="#c7d2df")
        draw.text((22, a.height - 30), "Table-assisted task capture accepted. Historical strict scorer excluded table contact.", font=fonts[2], fill="#c7d2df")
        return np.asarray(image)

    plan = frame_plan(bundle["time_s"])
    previews = []
    writer = None
    try:
        if a.preview_only:
            preview_dir.mkdir(exist_ok=False)
            events = bundle["evidence"]["events"]
            for view in views:
                for name, when in (("release", 0.), ("board", events["board_impact"]["start_s"]),
                                   ("table", events["table_bounce"]["start_s"]),
                                   ("bucket", events["first_bucket_contact_s"]), ("capture", events["capture_s"])):
                    tick = nearest_tick(bundle["time_s"], when)
                    item = dict(view=view, playback_speed=1. if view == "overview" else .25,
                                source_tick=tick, source_time_s=float(bundle["time_s"][tick]))
                    path = preview_dir / f"{view}-{name}.jpg"
                    Image.fromarray(frame(item)).save(path, quality=94)
                    previews.append(dict(name=path.name, sha256=sha(path), **item))
        else:
            writer = imageio.get_writer(output, fps=FPS, codec="libx264", quality=8, macro_block_size=1)
            for item in plan:
                writer.append_data(frame(item))
            writer.close(); writer = None
    finally:
        if writer is not None:
            writer.close()
        renderer.close()
    report = dict(version=VERSION, status="preview_complete" if a.preview_only else "completed",
                  shot_id=bundle["row"]["id"], input_sha256=bundle["input_sha256"], asset_sha256=bundle["asset_sha256"],
                  scene_sha256=bundle["scene_sha256"], model_source_sha256=bundle["model_source_sha256"],
                  renderer_source_sha256=sha(__file__), full_model_xml_sha256=model_sha, backend=backend,
                  packages={name: importlib.metadata.version(name) for name in ("mujoco", "numpy", "Pillow", "imageio")},
                  width=a.width, height=a.height, fps=FPS, cameras=views, native_dt_s=bundle["native_dt_s"],
                  native_steps_added=0, held_arm_poses=True, sampled_ball_centers_only=True, ball_spin_recorded=False,
                  source_interval_s=[0., END_S], original_score=bundle["evidence"]["original_score"],
                  intended_task_success=True, intended_task_success_basis="User accepts table-assisted whole-ball bucket capture retained for one continuous second",
                  historical_labels_unchanged=True, policy_reliability_established=False,
                  frames=[] if a.preview_only else plan, frames_sha256=None if a.preview_only else digest(plan),
                  output_sha256=None if a.preview_only else sha(output), previews=previews,
                  scope="Visual replay of one recorded native training attempt, with static saved joint poses; no new rollout, training, scoring relabel or synthesized trajectory.")
    manifest_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(status=report["status"], manifest=str(manifest_path), frames=len(report["frames"]), previews=len(previews))))
    return report


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for key in ("study", "metadata", "cache", "evidence", "scene", "assets", "output"):
        result.add_argument("--" + key, type=Path, required=True)
    result.add_argument("--width", type=int, default=1280)
    result.add_argument("--height", type=int, default=720)
    result.add_argument("--preview-only", action="store_true")
    return result


if __name__ == "__main__":
    render(parser().parse_args())
