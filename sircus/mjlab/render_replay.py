"""Render recorded GPU ball positions in MuJoCo's native renderer, with fixed arm poses."""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import imageio.v2 as imageio

from .model import build_scene


def render(a):
    manifest = json.loads(a.manifest.read_text())
    first = json.loads(a.first.read_text())
    second = json.loads(a.second.read_text())
    lookup = {s["shot_id"]: s for s in manifest["shots"]}
    aa = {r["shot_id"]: r for r in first["shots"] if r["repeat"] == 0}
    bb = {r["shot_id"]: r for r in second["shots"] if r["repeat"] == 0}
    wanted = []
    for flags in ((True, True), (False, True), (True, False), (False, False)):
        found = [
            k
            for k in lookup
            if (aa[k]["route_success"], bb[k]["route_success"]) == flags
        ]
        if found:
            wanted.append(found[0])
    a.output.parent.mkdir(parents=True, exist_ok=True)
    frames = 0
    shots = []
    writer = imageio.get_writer(a.output, fps=30, codec="libx264", quality=8)
    from PIL import Image, ImageDraw

    for key in wanted:
        s = lookup[key]
        model = mujoco.MjModel.from_xml_string(
            build_scene(manifest["scene"], a.assets, bucket_xy=s["bucket_xy_m"])
        )
        data = mujoco.MjData(model)
        for role, values in s["final_joint_positions_rad"].items():
            names = [f"{role}/joint{i+1}" for i in range(6)]
            if role == "Arm10":
                names += [f"{role}/finger_left_joint", f"{role}/finger_right_joint"]
            for name, v in zip(names, values):
                j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                data.qpos[model.jnt_qposadr[j]] = v
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball/free")
        q = int(model.jnt_qposadr[j])
        mocap = int(
            model.body_mocapid[
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bucket")
            ]
        )
        data.mocap_pos[mocap] = [*s["bucket_xy_m"], 0]
        renderer = mujoco.Renderer(model, height=480, width=640)
        for i in range(
            0, min(len(aa[key]["trajectory_m"]), len(bb[key]["trajectory_m"])), 2
        ):
            images = []
            for label, r in (
                ("Isaac PhysX trajectory", aa[key]),
                ("mjlab GPU trajectory", bb[key]),
            ):
                data.qpos[q : q + 3] = r["trajectory_m"][i]
                data.qpos[q + 3 : q + 7] = [1, 0, 0, 0]
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera="front")
                picture = Image.fromarray(renderer.render())
                d = ImageDraw.Draw(picture)
                d.rectangle((0, 0, 640, 40), fill="#101319")
                status = "CAPTURE" if r["route_success"] else "MISS"
                d.text((12, 6), f"{label}  |  {key}  |  {status}", fill="white")
                d.text(
                    (12, 22),
                    "Common native MuJoCo renderer; fixed arm poses",
                    fill="#aabbcc",
                )
                images.append(np.asarray(picture))
            writer.append_data(np.concatenate(images, axis=1))
            frames += 1
        renderer.close()
        shots.append(key)
    writer.close()
    a.output.with_suffix(".json").write_text(
        json.dumps(
            dict(
                scope="Native MuJoCo rendering of recorded GPU trajectories from both engines; fixed joint poses; no image synthesis",
                input_sha256={
                    "first": hashlib.sha256(a.first.read_bytes()).hexdigest(),
                    "second": hashlib.sha256(a.second.read_bytes()).hexdigest(),
                },
                fps=30,
                frames=frames,
                shots=shots,
            ),
            indent=2,
        )
    )
    print(a.output, frames, "frames")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for field in ("manifest", "first", "second", "assets", "output"):
        p.add_argument("--" + field, type=Path, required=True)
    render(p.parse_args())
