"""Isaac-only randomized capture; frozen sensor-only worker; separate truth scoring."""

import copy
import datetime
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from fixed_rl import FixedShot
from perception_evaluate import compare
from perception_stream import publish_sensor, publish_truth
from perception_worker import load_sensor
from PIL import Image
from prepare_assets import fk, matrix
from pxr import Gf
from validation_contract import (
    CONDITIONS,
    VERSION,
    layouts,
    noisy_depth,
    paired_source,
    summarize,
    visible_sphere_pixels,
)


def run(sim, arms, ball, camera, cfg, poses, args, atomic):
    out = Path(args.output)
    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = out / f"perception-study-{stamp}"
    for name in ["sensors", "evaluation_truth", "estimates"]:
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    frozen = hashlib.sha256(
        (Path(__file__).parent / "perception_estimator.py").read_bytes()
    ).hexdigest()
    plan = list(layouts(args.validation_seed, args.validation_layouts))
    base = matrix([cfg["arms"]["base_spacing_m"], 0, 0], [0, 0, np.pi])
    mount = matrix(
        [0, 0, cfg["board"]["tool_offset_m"]], [0, 0, cfg["board"]["mount_yaw_rad"]]
    )
    env = FixedShot(sim, arms, ball, camera, None, cfg, poses, args, atomic)
    for layout in plan:
        q = np.array(poses["board_joint_rad"]) + layout["board_joint_delta_rad"]
        target = base @ fk(q, env.links) @ mount
        layout["board_joint_rad"] = q.tolist()
        layout["board_center_m"] = target[:3, 3].tolist()
        layout["board_normal"] = target[:3, 2].tolist()
        if layout["ball_mode"] == "above_board":
            layout["ball_position_m"] = (target[:3, 3] + [0, 0, 0.16]).tolist()
        elif layout["ball_mode"] == "between_board_bucket":
            layout["ball_position_m"] = [
                *(0.5 * (target[:2, 3] + layout["bucket_xy_m"])),
                0.27,
            ]
    atomic(
        run_dir / "plan.json",
        {
            "contract": VERSION,
            "seed": args.validation_seed,
            "layouts": plan,
            "conditions": CONDITIONS,
            "estimator_sha256": frozen,
            "note": "Evaluation-only plan; never supplied to estimator",
        },
    )
    bucket = sim.stage.GetPrimAtPath("/World/Bucket").GetAttribute("xformOp:translate")
    light = sim.stage.GetPrimAtPath("/World/Light")
    intensity = light.GetAttribute("inputs:intensity")
    color = light.GetAttribute("inputs:color")
    assert bucket and intensity and color
    records = []
    excluded = []
    frame = 0
    started = time.monotonic()
    planned = len(plan) * len(CONDITIONS)

    def progress(state, current=None):
        summary = summarize(records, planned, excluded)
        summary.update(
            state=state,
            run=run_dir.name,
            seed=args.validation_seed,
            estimator_sha256=frozen,
            elapsed_s=time.monotonic() - started,
            timestamp_utc=datetime.datetime.now(datetime.UTC).isoformat(),
            current=current,
            condition_parameters=CONDITIONS,
            board_center_range_m={
                "min": np.min([p["board_center_m"] for p in plan], axis=0).tolist(),
                "max": np.max([p["board_center_m"] for p in plan], axis=0).tolist(),
            },
            bucket_xy_range_m={
                "min": np.min([p["bucket_xy_m"] for p in plan], axis=0).tolist(),
                "max": np.max([p["bucket_xy_m"] for p in plan], axis=0).tolist(),
            },
        )
        atomic(run_dir / "summary.json", summary)
        atomic(out / "validation.json", summary)

    progress("running")
    for layout in plan:
        if shutil.disk_usage(out).free < 3 * 1024**3:
            raise RuntimeError(
                "Less than3GiB free; validation stopped without deleting data"
            )
        center = np.array(layout["board_center_m"])
        bxy = layout["bucket_xy_m"]
        q14 = np.array(layout["board_joint_rad"])
        if np.any(q14 < env.lo) or np.any(q14 > env.hi) or not (0.2 < center[2] < 0.65):
            excluded.append(
                {
                    "layout": layout["layout"],
                    "reason": "joint or height bound; no resampling",
                }
            )
            progress("running")
            continue
        arm = arms["Arm14"]
        q = arm.data.default_joint_pos.clone()
        for j in range(6):
            q[0, arm.joint_names.index(f"joint{j + 1}")] = float(q14[j])
        arm.write_joint_state_to_sim(q, torch.zeros_like(q))
        arm.set_joint_position_target(q)
        arm.reset()
        bucket.Set(Gf.Vec3d(*bxy, 0.0))
        actual_cfg = copy.deepcopy(cfg)
        actual_cfg["bucket"]["center_xy_m"] = bxy
        ball_state = ball.data.default_root_state.clone()
        ball_state[0, :3] = torch.tensor(layout["ball_position_m"], device=sim.device)
        ball_state[0, 7:] = 0

        def render(ball_state=ball_state):
            nonlocal frame
            for _ in range(8):
                ball.write_root_state_to_sim(ball_state)
                for a in arms.values():
                    a.write_data_to_sim()
                sim.step(render=False)
                for a in arms.values():
                    a.update(env.dt)
                ball.update(env.dt)
            sim.render()
            camera.update(env.dt * 8, force_recompute=True)
            frame += 1

        captures = {}
        for condition in CONDITIONS:
            source = paired_source(condition["name"])
            source_truth = None
            if source:
                rgb, clean, source_truth = captures[source]
                frame += 1
            else:
                intensity.Set(condition["intensity"])
                color.Set(Gf.Vec3f(*condition["color"]))
                for _ in range(24):
                    render()
                rgb = (
                    camera.data.output["rgb"][0, :, :, :3]
                    .cpu()
                    .numpy()
                    .astype(np.uint8)
                )
                clean = (
                    camera.data.output["distance_to_image_plane"][0, :, :, 0]
                    .cpu()
                    .numpy()
                    .copy()
                )
            seed = (
                args.validation_seed
                + layout["layout"] * 100
                + CONDITIONS.index(condition)
            )
            depth = noisy_depth(clean, condition["sigma_m"], condition["dropout"], seed)
            ident = f"{layout['layout']:03d}-{condition['name']}"
            frame_id = publish_sensor(out, frame, rgb, depth, camera)
            if source_truth is None:
                publish_truth(out, frame_id, actual_cfg, arms, ball)
                truth = json.loads((out / "evaluation_truth/latest.json").read_text())
                truth["lighting_readback"] = {
                    "intensity": float(intensity.Get()),
                    "color": list(color.Get()),
                }
                truth["source_render_frame_id"] = frame_id
            else:
                truth = copy.deepcopy(source_truth)
                truth["frame_id"] = frame_id
            sensor = load_sensor(out / "sensors/latest.npz")
            truth["ball_visible_pixels"] = visible_sphere_pixels(
                clean,
                sensor["intrinsics"],
                sensor["camera_to_world"],
                truth["objects"]["ball"]["center_m"],
                cfg["ball"]["radius_m"],
            )
            truth["rendered_rgb_mean"] = rgb.reshape(-1, 3).mean(0).tolist()
            truth["depth_valid_before"] = int(
                (np.isfinite(clean) & (clean > 0.05) & (clean < 5)).sum()
            )
            truth["depth_valid_after"] = int(np.isfinite(depth).sum())
            truth["layout"] = layout
            truth["condition"] = condition
            truth["paired_source"] = source
            if source is None:
                captures[condition["name"]] = (
                    rgb.copy(),
                    clean.copy(),
                    copy.deepcopy(truth),
                )
            atomic(out / "evaluation_truth/latest.json", truth)
            shutil.copyfile(
                out / "sensors/latest.npz", run_dir / "sensors" / f"{ident}.npz"
            )
            atomic(run_dir / "evaluation_truth" / f"{ident}.json", truth)
            Image.fromarray(rgb).save(out / "latest.tmp.jpg")
            (out / "latest.tmp.jpg").replace(out / "latest.jpg")
            Image.fromarray(
                np.uint8(
                    np.clip(1 - np.nan_to_num(depth, nan=5, posinf=5) / 5, 0, 1) * 255
                )
            ).save(out / "depth-preview.png")
            atomic(
                out / "status.json",
                {
                    "timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(),
                    "frame": frame,
                    "state": "randomized_perception_validation",
                    "gpu": torch.cuda.get_device_name(0),
                    "camera": {"valid_depth_pixels": int(np.isfinite(depth).sum())},
                    "arms": {
                        n: {"joint_names": a.joint_names} for n, a in arms.items()
                    },
                    "max_joint_target_error_rad": float(
                        max(
                            (a.data.joint_pos - a.data.joint_pos_target).abs().max()
                            for a in arms.values()
                        )
                    ),
                    "training": {
                        "phase": "static perception validation; no policy updates",
                        "layout": layout["layout"],
                        "condition": condition["name"],
                    },
                },
            )
            # Existing independent worker reads ONLY the sensor bundle; wait for the
            # exact ID. Timeout is recorded as missing, never replaced by truth.
            estimate = None
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                try:
                    candidate = json.loads((out / "perception.json").read_text())
                    if candidate["frame_id"] == frame_id:
                        estimate = candidate
                        break
                except (OSError, ValueError):
                    pass
                time.sleep(0.03)
            if estimate is None:
                estimate = {
                    "frame_id": frame_id,
                    "timestamp_utc": str(sensor["timestamp_utc"]),
                    "objects": {
                        n: {"status": "missing", "reason": "worker timeout"}
                        for n in truth["objects"]
                    },
                }
            atomic(run_dir / "estimates" / f"{ident}.json", estimate)
            report = compare(estimate, truth)
            report.update(
                layout=layout["layout"],
                condition=condition["name"],
                ball_mode=layout["ball_mode"],
                ball_visible_pixels=truth["ball_visible_pixels"],
                worker_timeout=estimate["objects"]["ball"].get("reason")
                == "worker timeout",
                sample=ident,
            )
            records.append(report)
            atomic(run_dir / "records.json", records)
            if (out / "perception.jpg").exists() and not report["worker_timeout"]:
                shutil.copyfile(
                    out / "perception.jpg", run_dir / "estimates" / f"{ident}.jpg"
                )
                shutil.copyfile(
                    out / "perception.jpg", out / "validation-example.tmp.jpg"
                )
                (out / "validation-example.tmp.jpg").replace(
                    out / "validation-example.jpg"
                )
            progress("running", ident)
    assert (
        frozen
        == hashlib.sha256(
            (Path(__file__).parent / "perception_estimator.py").read_bytes()
        ).hexdigest()
    ), "Detector changed during study"
    progress("completed")
    atomic(
        out / "result.json",
        {
            "passed": bool(records),
            "state": "perception_validation_completed",
            "run": run_dir.name,
            "observations": len(records),
            "scope": "Infrastructure completion only; detector accuracy is in validation.json",
        },
    )
    print("PERCEPTION_VALIDATION_COMPLETE", run_dir, flush=True)
