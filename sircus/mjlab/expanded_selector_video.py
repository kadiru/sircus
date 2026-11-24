"""Paired videos of frozen expanded-selector native64 recordings, without stepping.

Only recorded ball centers are played back; arms remain at the saved prescribed
endpoints. The renderer proves the EGL device's PCI identity before creating a
context and leases only the authorized 4090. No simulation outcomes are changed.
"""
from __future__ import annotations

import argparse
import ctypes
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np

from . import hybrid_miss_video as recorded
from .gpu_slot import GPU_UUID
from .height_tilt_video import CONTACTS, first_events
from .saved_shot_video import contained, digest, nearest_tick, read, require, sha, vector

VERSION = "sircus-expanded-selector-recorded-pairs-v1"
METHODS = ("current", "expanded")
CASES = ("04", "10", "19", "47")
SCORING_VERSION = "sircus-board-first-native-capture-v4"
FPS = 30
SPEED = .25


def normalize_pci(value):
    parts = value.strip().lower().split(":")
    require(len(parts) == 3, "PCI address required")
    return f"{int(parts[0], 16):04x}:{int(parts[1], 16):02x}:{parts[2]}"


def select_device(devices, inventory, expected=GPU_UUID):
    """Pure selection: CUDA visibility alone does not select an EGL device."""
    matches = [r for r in inventory if r["uuid"] == expected]
    require(len(matches) == 1 and "RTX 4090" in matches[0]["name"], "Authorized 4090 identity required")
    pci = normalize_pci(matches[0]["pci"])
    choices = [r for r in devices if "EGL_NV_device_cuda" in r["extensions"].split()
               and r.get("pci") and normalize_pci(r["pci"]) == pci]
    require(len(choices) == 1, "Exactly one NVIDIA EGL device must map to the authorized PCI address")
    return dict(choices[0], uuid=expected, name=matches[0]["name"], pci=pci)


def egl_metadata():
    """Enumerate device metadata only: no display, context, or GPU work."""
    lib = ctypes.CDLL("libEGL.so.1")
    lib.eglGetProcAddress.restype = ctypes.c_void_p
    lib.eglGetProcAddress.argtypes = [ctypes.c_char_p]

    def extension(name, result, *arguments):
        address = lib.eglGetProcAddress(name.encode())
        require(bool(address), "Required EGL metadata extension missing: " + name)
        return ctypes.CFUNCTYPE(result, *arguments)(address)

    query = extension("eglQueryDevicesEXT", ctypes.c_uint, ctypes.c_int,
                      ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int))
    string = extension("eglQueryDeviceStringEXT", ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int)
    devices = (ctypes.c_void_p * 64)()
    count = ctypes.c_int()
    require(query(64, devices, ctypes.byref(count)) and 0 < count.value < 64,
            "Bounded EGL device enumeration required")
    result = []
    for i in range(count.value):
        extensions = (string(devices[i], 0x3055) or b"").decode()
        item = dict(index=i, extensions=extensions, pci=None)
        for name, token, required in (("render_node", 0x3377, "EGL_EXT_device_drm_render_node"),
                                      ("card_node", 0x3233, "EGL_EXT_device_drm")):
            if required not in extensions.split():
                continue
            value = string(devices[i], token)
            if not value:
                continue
            node = Path(value.decode())
            sys_device = Path("/sys/class/drm") / node.name / "device"
            require(sys_device.exists(), "EGL DRM device must have a sysfs PCI mapping")
            mapped = sys_device.resolve().name
            if item["pci"]:
                require(normalize_pci(item["pci"]) == normalize_pci(mapped), "EGL DRM mappings disagree")
            item.update({name: str(node), "pci": mapped})
        result.append(item)
    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,name,pci.bus_id",
                                   "--format=csv,noheader"], text=True)
    inventory = []
    for line in raw.splitlines():
        uuid, name, pci = [x.strip() for x in line.split(",")]
        inventory.append(dict(uuid=uuid, name=name, pci=pci))
    return select_device(result, inventory)


def configure_egl():
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == GPU_UUID, "Select only the authorized 4090 UUID")
    require(os.environ.get("MUJOCO_GL") == "egl" and os.environ.get("PYOPENGL_PLATFORM") == "egl",
            "Explicit EGL rendering environment required")
    import sys
    require("mujoco" not in sys.modules, "Map EGL before importing MuJoCo")
    device = egl_metadata()
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(device["index"])
    return device


def validate_backend(renderer, vendor, device):
    require("RTX 4090" in renderer and "NVIDIA" in vendor and device["uuid"] == GPU_UUID,
            "The rendering context must be the independently mapped NVIDIA 4090")
    return dict(renderer=renderer, vendor=vendor, device=device)


def load_pair(a):
    require(a.case in CASES, "One of the four audited discordant cases required")
    study = Path(a.study).resolve()
    report, contract, manifest, audit = recorded.verify_envelope(study, a.audit)
    require(contract["methods"] == list(METHODS), "Unexpected comparison methods")
    require(contract["scoring_version"] == SCORING_VERSION, "Board-first v4 scoring required")
    choices_file = contained(study, "choices.json")
    require(sha(choices_file) == audit["choices_sha256"] == manifest["choices.json"], "Frozen choices differ")
    choices = read(choices_file)["slots"]
    bundles = {}
    for method in METHODS:
        identifier = f"expanded-selector-test-{a.case}-{method}"
        bundle = recorded.load_recording(SimpleNamespace(study=study, audit=a.audit, assets=a.assets, record=identifier))
        row = bundle["row"]
        matching = [s for s in choices if s["id"] == identifier]
        require(len(matching) == 1 and matching[0]["method"] == method, "Unique frozen selection required")
        require(matching[0]["bucket_xy_m"] == row["bucket_xy_m"], "Selected and recorded bucket differ")
        score = row.get("board_first_score", {})
        require(score.get("scoring_version") == SCORING_VERSION and score.get("invalid_physics") is False,
                "Unchanged valid v4 score required")
        require(type(score.get("route_success")) is bool and score["physics_steps"] == row["physics_steps"],
                "V4 terminal and capture identity required")
        bundle["score"] = score
        # Replace only the event labels in this replay bundle, never the saved row.
        # The original row hash remains bound to the unmodified historical record.
        bundle["events"] = {k: v for k, v in bundle["events"].items() if k in CONTACTS}
        for name in ("whole_ball_entry", "capture"):
            t = score.get(name + "_time_s")
            if t is not None:
                require(math.isfinite(t) and 0 <= t <= bundle["scored_until_s"], "V4 event outside recorded terminal")
                bundle["events"][name] = t
        relative = row["adapter_directory"] + "/" + row["effective_options_file"]
        options_file = contained(study, relative)
        require(sha(options_file) == row["effective_options_sha256"] == manifest[relative], "Native options differ")
        bundle["options"] = read(options_file)
        bundle["input_sha256"].update({"choices.json": sha(choices_file), "native_options": sha(options_file)})
        bundles[method] = bundle
    require(bundles["current"]["row"]["bucket_xy_m"] == bundles["expanded"]["row"]["bucket_xy_m"],
            "Both methods must use exactly the same target")
    require(bundles["current"]["score"]["route_success"] != bundles["expanded"]["score"]["route_success"],
            "Only discordant outcomes belong in these paired replays")
    return bundles


def verified_scene(bundle, assets):
    from .capture_calibration import compiled_geometry, option_record
    model, data, ball_q, xml_hash = recorded.build_scene(bundle, assets)
    for name, value in bundle["options"]["native"].items():
        if name == "gravity":
            model.opt.gravity[:] = value
        else:
            setattr(model.opt, name, value)
    require(option_record(model) == bundle["options"]["native"], "Replayed model options differ")
    geometry = compiled_geometry(model, bundle["scene"])
    require(geometry["geometry_sha256"] == bundle["row"]["calibrated_geometry_sha256"],
            "Replayed collision geometry differs from the measured native trace")
    return model, data, ball_q, xml_hash


def frame_plan(bundles, fps=FPS):
    require(type(fps) is int and 1 <= fps <= 60, "Bounded integer FPS required")
    end = max(b["scored_until_s"] for b in bundles.values())
    frames = []

    def add(wanted, segment, count=1):
        samples = {}
        for method in METHODS:
            bundle = bundles[method]
            stop = bundle["scored_until_s"]
            require(0 < stop <= bundle["time_s"][-1], "Scored interval outside recording")
            tick = nearest_tick(bundle["time_s"], min(wanted, stop))
            samples[method] = dict(source_tick=tick, source_time_s=float(bundle["time_s"][tick]),
                                   terminal_hold=bool(wanted >= stop))
        for _ in range(count):
            frames.append(dict(frame=len(frames), video_time_s=len(frames) / fps,
                               segment=segment, playback_speed=SPEED if segment == "slow" else 0,
                               common_time_s=wanted, methods=samples))
    add(0, "title", round(.5 * fps))
    for i in range(math.ceil(end / SPEED * fps)):
        add(min(end, i / fps * SPEED), "slow")
    add(end, "outcome", fps)
    return frames


def common_camera(bundles):
    rows = [bundles[m]["row"] for m in METHODS]
    points = np.asarray([p for row in rows for p in
                         (row["board_center_m"], row["native_initial_ball_center_m"], [*row["bucket_xy_m"], .12])])
    target = (points.min(axis=0) + points.max(axis=0)) / 2
    direction = np.asarray(rows[0]["bucket_xy_m"]) - np.mean([r["board_center_m"][:2] for r in rows], axis=0)
    side = np.asarray([direction[1], -direction[0]])
    return dict(lookat=target.tolist(), distance=max(1.5, float(np.linalg.norm(np.ptp(points, axis=0))) * 1.8),
                azimuth=math.degrees(math.atan2(side[1], side[0])) + 15, elevation=-25)


def status_line(sample, bundle):
    if sample["terminal_hold"]:
        return "Recording ended — final state held"
    t = sample["source_time_s"]
    if "capture" in bundle["events"] and t >= bundle["events"]["capture"]:
        return "Capture recorded: retained one second"
    occurred = [(v, k) for k, v in bundle["events"].items() if v <= t]
    if not occurred:
        return "Released: following saved ball samples"
    _, name = max(occurred)
    return "Whole-ball entry recorded" if name == "whole_ball_entry" else "Recorded contact: " + name


def method_receipt(bundle):
    row = bundle["row"]
    return dict(record_id=row["id"], record_sha256=bundle["record_sha256"],
                native_archive_sha256=row["native_cache_sha256"], native_column=row["native_cache_row"],
                capture=bundle["score"]["route_success"], scoring_version=SCORING_VERSION,
                source_interval_s=[0, bundle["scored_until_s"]], events=bundle["events"],
                q10=row["native_initial_q10"], q14=row["native_initial_q14"],
                bucket_xy_m=row["bucket_xy_m"], values=row["values"],
                calibrated_geometry_sha256=row["calibrated_geometry_sha256"],
                input_sha256=bundle["input_sha256"])


def render(a):
    output = Path(a.output).resolve()
    require(output.suffix == ".mp4", "MP4 output required")
    receipt_path = output.with_suffix(".preview.json" if a.preview_only else ".json")
    for root in (a.study, a.assets):
        require(not output.is_relative_to(Path(root).resolve()), "Keep media outside immutable evidence/assets")
    require(not output.exists() and not receipt_path.exists(), "Refusing to overwrite replay evidence")
    bundles = load_pair(a)
    if a.validate_only:
        return dict(status="validated", methods={m: method_receipt(b) for m, b in bundles.items()})
    device = configure_egl()
    from .gpu_slot import gpu_slot, foreign_processes
    with gpu_slot(a.gpu_lock, timeout_s=a.gpu_wait_s, poll_s=10):
        require(not foreign_processes(), "4090 became occupied before rendering")
        return render_leased(a, output, receipt_path, bundles, device)


def render_leased(a, output, receipt_path, bundles, device):
    import mujoco
    from OpenGL import GL
    from PIL import Image, ImageDraw, ImageFont
    import imageio.v2 as imageio
    from contextlib import ExitStack
    frames = frame_plan(bundles)
    camera_settings = read(a.camera) if a.camera else common_camera(bundles)
    vector(camera_settings["lookat"], 3, "camera target")
    require(all(math.isfinite(camera_settings[k]) for k in ("distance", "azimuth", "elevation")) and
            camera_settings["distance"] > 0, "Finite positive camera required")
    width, height, panel, header, footer = 1280, 720, 640, 170, 62
    viewport = height - header - footer
    fonts = [ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", n) for n in (26, 22, 18, 16)]
    output.parent.mkdir(parents=True, exist_ok=True)
    models, renderers, backend, model_hashes = {}, {}, {}, {}
    previews = []
    with ExitStack() as stack:
        for method in METHODS:
            model, data, ball_q, model_hash = verified_scene(bundles[method], a.assets)
            models[method] = model, data, ball_q
            model_hashes[method] = model_hash
            model.vis.global_.offwidth = max(panel, model.vis.global_.offwidth)
            model.vis.global_.offheight = max(viewport, model.vis.global_.offheight)
            renderer = stack.enter_context(mujoco.Renderer(model, height=viewport, width=panel))
            backend[method] = validate_backend((GL.glGetString(GL.GL_RENDERER) or b"").decode(),
                                               (GL.glGetString(GL.GL_VENDOR) or b"").decode(), device)
            renderers[method] = renderer
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = camera_settings["lookat"]
        for key in ("distance", "azimuth", "elevation"):
            setattr(camera, key, camera_settings[key])

        def draw_frame(item):
            picture = Image.new("RGB", (width, height), "#101722")
            draw = ImageDraw.Draw(picture)
            xy = bundles["current"]["row"]["bucket_xy_m"]
            draw.text((18, 8), f"Position {a.case} | bucket ({xy[0]*1000:.1f}, {xy[1]*1000:.1f}) mm", font=fonts[0], fill="white")
            draw.text((18, 43), "Saved native64 simulation | quarter speed (0.25×) | identical camera", font=fonts[2], fill="#c6d3df")
            for index, method in enumerate(METHODS):
                bundle = bundles[method]; sample = item["methods"][method]
                model, data, ball_q = models[method]
                data.qpos[ball_q:ball_q+3] = bundle["positions"][sample["source_tick"]]
                data.time = sample["source_time_s"]
                mujoco.mj_forward(model, data)
                renderers[method].update_scene(data, camera=camera)
                picture.paste(Image.fromarray(renderers[method].render()), (index * panel, header))
                x = index * panel + 16
                captured = bundle["score"]["route_success"]
                color = "#85dec2" if captured else "#ffbb89"
                name = "Current planner" if method == "current" else "Expanded selector"
                draw.text((x, 77), name + (" — CAPTURE" if captured else " — NO CAPTURE"), font=fonts[1], fill=color)
                clock = f"Simulation t = {sample['source_time_s']:.3f} s"
                draw.text((x, 111), clock, font=fonts[2], fill="white")
                draw.text((x, 139), status_line(sample, bundle), font=fonts[3], fill="#c6d3df")
                values = bundle["row"]["values"]
                draw.text((x, height-54), f"Board {values[2]*1000:.0f} mm · tilt {values[3]:.1f}° · gap {values[4]*1000:.0f} mm", font=fonts[3], fill="#c6d3df")
            draw.line((panel, 72, panel, height-30), fill="#536274", width=2)
            draw.text((18, height-27), "Arms held at saved endpoints. Recorded ball centers; no new physics, invented arm motion or ball spin.", font=fonts[3], fill="#91a1b6")
            return picture

        if a.preview_only:
            directory = output.with_name(output.stem + "-previews")
            directory.mkdir(exist_ok=False)
            moments = sorted({0., *[t for b in bundles.values() for t in b["events"].values()],
                              max(b["scored_until_s"] for b in bundles.values())})
            for i, t in enumerate(moments):
                item = min(frames, key=lambda f: abs(f["common_time_s"] - t))
                path = directory / f"frame-{i:02d}.jpg"
                draw_frame(item).save(path, quality=92)
                previews.append(dict(file=path.name, sha256=sha(path), frame=item))
        else:
            with imageio.get_writer(output, fps=FPS, codec="libx264", quality=7, macro_block_size=1,
                                    ffmpeg_params=["-threads", "2", "-movflags", "+faststart"]) as writer:
                for item in frames:
                    writer.append_data(np.asarray(draw_frame(item)))
    receipt = dict(version=VERSION, status="preview_complete" if a.preview_only else "completed",
                   case_id=f"expanded-selector-test-{a.case}", methods={m: method_receipt(b) for m,b in bundles.items()},
                   renderer_source_sha256=sha(__file__), model_xml_sha256=model_hashes,
                   model_source_sha256=bundles["current"]["model_source_sha256"],
                   asset_sha256=bundles["current"]["assets_sha256"], backend=backend,
                   packages={n:importlib.metadata.version(n) for n in ("mujoco", "numpy", "Pillow", "imageio")},
                   width=width, height=height, fps=FPS, duration_s=len(frames)/FPS, playback_speed=SPEED,
                   camera=camera_settings, frames=[] if a.preview_only else frames, previews=previews,
                   native_steps_added=0, held_arm_poses=True, interpolation=False,
                   sampled_ball_centers_only=True, ball_spin_recorded=False, historical_labels_unchanged=True,
                   output_sha256=None if a.preview_only else sha(output), video_bytes=None if a.preview_only else output.stat().st_size)
    receipt_path.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return dict(status=receipt["status"], output=str(output), manifest=str(receipt_path), duration_s=receipt["duration_s"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "audit", "assets", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--case", choices=CASES, required=True)
    p.add_argument("--camera", type=Path)
    p.add_argument("--preview-only", action="store_true")
    p.add_argument("--validate-only", action="store_true")
    p.add_argument("--gpu-lock", type=Path, default=Path("/home/kadir/sircus-isaac/scene.lock"))
    p.add_argument("--gpu-wait-s", type=float, default=30)
    print(json.dumps(render(p.parse_args()), allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
