"""Capture native camera choices while every simulated workcell stays frozen."""

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from camera_options import DISPLAY_SLOT, SOURCE_BATCH, SOURCE_FOCUS, camera_options
from parallel_runtime import atomic


def configure_backdrop():
    """Override the renderer's sky color while retaining the selected lighting.

    NVIDIA RTX Common documents these background source settings separately
    from scene lighting. No additional dome replaces the existing studio fill.
    """
    import carb

    values = {
        "/rtx/background/source/type": 2,
        "/rtx/background/source/color": [0.22, 0.23, 0.24],
    }
    settings = carb.settings.get_settings()
    for name, value in values.items():
        settings.set(name, value)
    return dict(
        renderer_settings=values,
        documentation="https://docs.omniverse.nvidia.com/materials-and-rendering/latest/rtx-renderer_common.html",
    )


def state_digest(env):
    arrays = [a.data.joint_pos.cpu().numpy() for a in env.arms.values()]
    arrays += [
        env.ball.data.root_pos_w.cpu().numpy(),
        env.bucket.data.root_pos_w.cpu().numpy(),
    ]
    return hashlib.sha256(
        np.concatenate([a.flatten() for a in arrays]).tobytes()
    ).hexdigest()


def render_options(env, source):
    """No stepping, dropping, optimization or object-dependent camera tracking."""
    env.focus = SOURCE_FOCUS
    initial, physics_step = state_digest(env), env.steps
    variants = []
    for option in camera_options():
        eye, aim = np.array(option["eye_m"]), np.array(option["target_m"])
        # Flush the temporal renderer after a camera cut without advancing PhysX.
        import torch

        env.camera.set_world_poses_from_view(
            torch.as_tensor(eye[None], dtype=torch.float32, device=env.sim.device),
            torch.as_tensor(aim[None], dtype=torch.float32, device=env.sim.device),
        )
        for _ in range(24):
            env.sim.render()
        env.write_rgb(eye, aim, 1)
        frame = env.frames[-1]
        filename = f"camera-{option['id']:02}.jpg"
        destination = env.dest / filename
        shutil.copyfile(env.frames_path / f"{frame['frame']:05d}.jpg", destination)
        if env.steps != physics_step or state_digest(env) != initial:
            raise RuntimeError("The comparison scene advanced between cameras")
        variants.append(
            dict(
                option,
                file=filename,
                sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                physics_step=physics_step,
                camera_readback=dict(eye_m=frame["eye_m"], target_m=frame["target_m"]),
            )
        )
    manifest = dict(
        version="sircus-silver-camera-options-v1",
        renderer="Isaac Sim 5.1.0 native RTX camera",
        resolution=[1920, 1080],
        focal_length_mm=23,
        horizontal_aperture_mm=36,
        vertical_aperture_mm=20.25,
        appearance=env.args.appearance,
        stage=env.stage_metadata,
        focus_source=f"r{SOURCE_FOCUS // 7 + 1}c{SOURCE_FOCUS % 7 + 1}",
        focus_display=f"r{DISPLAY_SLOT // 7 + 1}c{DISPLAY_SLOT % 7 + 1}",
        workcell_origin_permutation_note="Whole source workcells 1 and 6 exchange display positions; their arms, bucket, board and ball remain together. All 35 unique target assignments are preserved.",
        origins_m=env.origins.tolist(),
        source_batch=SOURCE_BATCH,
        source_clip_sha256=hashlib.sha256(
            (source / f"clip-{SOURCE_BATCH:02}.json").read_bytes()
        ).hexdigest(),
        source_scene_fingerprint=json.loads((source / "contract.json").read_text())[
            "fingerprint"
        ],
        frozen_state_sha256=initial,
        same_state_all_cameras=True,
        capture_source_sha256={
            name: hashlib.sha256(
                Path(__file__).with_name(name).read_bytes()
            ).hexdigest()
            for name in (
                "camera_options.py",
                "camera_preview.py",
                "film_cylinder_farm.py",
                "silver_style.py",
            )
        },
        scope="Static composition previews using saved articulated poses; no new rollout, training result or physical validation.",
        variants=variants,
    )
    atomic(env.dest / "camera-options.json", manifest)
