"""Render appearance choices in the native Isaac camera at one fixed scene state."""

import argparse
import copy
import hashlib
import json
import os
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--assets", type=Path, default=Path("/workspace/assets"))
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--palette", choices=("original", "gray"), default="original")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not args.enable_cameras:
    parser.error("--enable_cameras is required")
args.output.mkdir(exist_ok=False)
app = AppLauncher(args).app

import isaaclab.sim as sim_utils
import numpy as np
import torch
from dark_stage import FLOOR_Z_M, origins
from feasibility_runtime import FeasibilityShots
from isaaclab.scene import InteractiveScene
from isaaclab.sensors import Camera, CameraCfg
from isaacsim.core.cloner import Cloner
from parallel_assets import bucket_asset, robot_body_filters, scene_config
from parallel_runtime import atomic
from PIL import Image
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade
from scipy.spatial.transform import Rotation

TABLES = [
    ("Cool gray", (0.23, 0.31, 0.34), False),
    ("Pale ash", (0.62, 0.48, 0.28), True),
    ("Honey oak", (0.38, 0.20, 0.07), True),
    ("Walnut", (0.16, 0.07, 0.025), True),
    ("Charcoal", (0.055, 0.070, 0.085), False),
]
FLOORS = [
    ("Mid-gray", (0.20, 0.20, 0.20), (0.20, 0.20, 0.20), "Neutral gray", False),
    ("Warm concrete", (0.46, 0.43, 0.38), (0.46, 0.43, 0.38), "Warm concrete", False),
    (
        "White square",
        (0.82, 0.82, 0.80),
        (0.045, 0.05, 0.06),
        "Charcoal surround",
        True,
    ),
    (
        "Cool gray square",
        (0.40, 0.46, 0.50),
        (0.78, 0.78, 0.76),
        "Off-white surround",
        True,
    ),
]


if args.palette == "gray":
    TABLES = [
        ("Cool gray", (0.23, 0.31, 0.34), False),
        ("Silver gray", (0.40, 0.44, 0.47), False),
        ("Cloud blue", (0.33, 0.45, 0.52), False),
        ("Slate blue", (0.16, 0.25, 0.31), False),
        ("Satin blue", (0.28, 0.42, 0.47), False),
    ]
    FLOORS = [
        ("Medium gray", (0.22, 0.23, 0.24), (0.22, 0.23, 0.24), "Medium gray", False),
        (
            "Light gray square",
            (0.38, 0.40, 0.42),
            (0.21, 0.23, 0.25),
            "Slate-gray surround",
            True,
        ),
    ]


class PreviewScene(InteractiveScene):
    def clone_environments(self, copy_from_source=False):
        positions = origins()
        Cloner.clone(
            self.cloner,
            source_prim_path=self.env_prim_paths[0],
            prim_paths=self.env_prim_paths,
            positions=positions,
            replicate_physics=True,
            copy_from_source=copy_from_source,
            enable_env_ids=True,
            clone_in_fabric=self.cfg.clone_in_fabric,
        )
        self._default_env_origins = torch.as_tensor(positions, device=self.device)


def material(stage, name, color, texture=None, roughness=0.88, specular=0.04):
    root = f"/World/LookMaterials/{name}"
    mat = UsdShade.Material.Define(stage, root)
    shader = UsdShade.Shader.Define(stage, root + "/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(1)
    shader.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set(
        (specular, specular, specular)
    )
    if texture is not None:
        uv = UsdShade.Shader.Define(stage, root + "/UV")
        uv.CreateIdAttr("UsdPrimvarReader_float2")
        uv.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        tex = UsdShade.Shader.Define(stage, root + "/Grain")
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(texture))
        tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            uv.CreateOutput("result", Sdf.ValueTypeNames.Float2)
        )
        shader.GetInput("diffuseColor").ConnectToSource(
            tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        )
    mat.CreateSurfaceOutput().ConnectToSource(
        shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    )
    return mat


def plane(stage, name, center, size, z):
    mesh = UsdGeom.Mesh.Define(stage, "/World/LookSurfaces/" + name)
    cx, cy = center
    x, y = np.array(size) / 2
    mesh.CreatePointsAttr(
        [
            (cx - x, cy - y, z),
            (cx + x, cy - y, z),
            (cx + x, cy + y, z),
            (cx - x, cy + y, z),
        ]
    )
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateSubdivisionSchemeAttr("none")
    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, "vertex"
    ).Set([(0, 0), (1, 0), (1, 1), (0, 1)])
    return mesh


def grain_png(color, path):
    # Seeded texture for the actual USD tabletop material. Grain runs along X.
    rng = np.random.default_rng(35)
    u, v = np.meshgrid(np.linspace(0, 1, 1024), np.linspace(0, 1, 512))
    warp = v * 35 + 0.35 * np.sin(u * 12 + v * 7) + 0.11 * np.sin(u * 29)
    tone = 0.91 + 0.055 * np.sin(warp * 2 * np.pi) + 0.035 * np.sin(warp * 19)
    tone += rng.normal(0, 0.013, u.shape)
    linear = np.clip(np.array(color)[None, None, :] * tone[:, :, None], 0, 1)
    srgb = np.where(
        linear <= 0.0031308, linear * 12.92, 1.055 * linear ** (1 / 2.4) - 0.055
    )
    Image.fromarray(np.uint8(np.clip(srgb * 255, 0, 255))).save(path)


def main():
    torch.set_num_threads(2)
    cfg = json.loads(Path(__file__).with_name("scene.json").read_text())
    poses = json.loads((args.assets / "poses.json").read_text())
    source_spec = json.loads((args.source / "source-contract.json").read_text())
    spec = dict(num_envs=35, cells=source_spec["cells"])
    lighting = dict(
        dome_fill=250,
        distant_intensity=1500,
        angular_size_deg=2.0,
        rotation_xyz_deg=[-25, -35, 0],
    )
    if args.palette == "gray":
        lighting = dict(
            dome_fill=900,
            distant_intensity=360,
            angular_size_deg=18.0,
            rotation_xyz_deg=[-25, -35, 0],
            fills=[
                dict(
                    name="LeftFill",
                    intensity=180,
                    angle=35.0,
                    rotation_xyz_deg=[25, 30, 150],
                ),
                dict(
                    name="RightFill",
                    intensity=140,
                    angle=40.0,
                    rotation_xyz_deg=[-20, 25, -70],
                ),
            ],
        )
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(
            dt=cfg["simulation"]["physics_dt_s"],
            device=args.device,
            render=sim_utils.RenderCfg(
                enable_shadows=True,
                enable_ambient_occlusion=args.palette != "gray",
                enable_direct_lighting=True,
            ),
        )
    )
    bucket = args.output / "bucket.usda"
    bucket_asset(bucket, cfg["bucket"])
    scfg, filters = scene_config(
        cfg, poses, args.assets, bucket, 35, robot_body_filters(args.assets)
    )
    scfg.Floor.spawn.visible = False
    scfg.light.spawn.intensity = lighting["dome_fill"]
    scfg.light.spawn.color = (1.0, 1.0, 1.0)
    scene = PreviewScene(scfg)
    light = sim_utils.DistantLightCfg(
        intensity=lighting["distant_intensity"], angle=lighting["angular_size_deg"]
    )
    q = Rotation.from_euler("xyz", lighting["rotation_xyz_deg"], degrees=True).as_quat()
    light.func("/World/LookKey", light, orientation=tuple(map(float, q[[3, 0, 1, 2]])))
    for fill in lighting.get("fills", []):
        light_cfg = sim_utils.DistantLightCfg(
            intensity=fill["intensity"], angle=fill["angle"]
        )
        rot = Rotation.from_euler(
            "xyz", fill["rotation_xyz_deg"], degrees=True
        ).as_quat()
        light_cfg.func(
            "/World/" + fill["name"],
            light_cfg,
            orientation=tuple(map(float, rot[[3, 0, 1, 2]])),
        )
    camera = Camera(
        CameraCfg(
            prim_path="/World/LookCamera",
            width=1920,
            height=1080,
            update_period=0,
            update_latest_camera_pose=True,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=23,
                horizontal_aperture=36,
                vertical_aperture=20.25,
                clipping_range=(0.04, 300),
            ),
        )
    )
    sim.reset()
    env = FeasibilityShots(
        sim,
        scene,
        None,
        None,
        cfg,
        poses,
        args,
        copy.deepcopy(spec),
        filters,
        args.output,
    )
    clip = json.loads((args.source / "clip-04.json").read_text())
    for name, arm in env.arms.items():
        q = torch.tensor(
            [r["final_joint_positions_rad"][name] for r in clip["rollout"]],
            device=sim.device,
        )
        arm.write_joint_state_to_sim(q, torch.zeros_like(q))
        arm.set_joint_position_target(q)
        arm.reset()
    for _ in range(120):
        env.tick()
    releases = np.array(
        [
            r["release_ball_center_m"] if not r["invalid"] else [0.1, -0.8, 0.04]
            for r in clip["rollout"]
        ]
    )
    env.set_balls(releases)
    env.tick()
    # Every appearance receives the same frozen articulation and ball state.
    state = (
        torch.cat(
            [a.data.joint_pos.flatten() for a in env.arms.values()]
            + [env.ball.data.root_pos_w.flatten()]
        )
        .cpu()
        .numpy()
    )
    stage = sim_utils.get_current_stage()
    center = np.array(cfg["table"]["center_m"][:2])
    size = np.array(cfg["table"]["size_m"][:2])
    world = origins()
    low = world[:, :2].min(axis=0) + center - size / 2
    high = world[:, :2].max(axis=0) + center + size / 2
    side = float(max(high - low) + 6)
    surround = plane(stage, "Surround", center, [240, 240], FLOOR_Z_M)
    pad = plane(stage, "Pad", center, [side, side], FLOOR_Z_M + 0.0003)
    tops = [
        plane(stage, f"TableTop{i:02}", xy[:2] + center, size, 0.0002)
        for i, xy in enumerate(world)
    ]
    tables = []
    for name, color, wood in TABLES:
        texture = (
            args.output / (name.lower().replace(" ", "-") + ".png") if wood else None
        )
        if wood:
            grain_png(color, texture)
        satin = args.palette == "gray" and name == "Satin blue"
        tables.append(
            material(
                stage,
                name.replace(" ", ""),
                color,
                texture,
                roughness=0.28 if satin else 0.72 if args.palette == "gray" else 0.88,
                specular=0.10 if satin else 0.04,
            )
        )
    near = world[2]
    views = {
        "close": (near + [1.95, -3.7, 1.85], near + [0.67, -0.2, -0.13]),
        "overview": (np.array([18.0, -24.0, 17.5]), np.array([0.67, -0.25, -0.2])),
    }
    manifest = dict(
        renderer="Isaac Sim 5.1.0 native RTX camera",
        palette=args.palette,
        resolution=[1920, 1080],
        lighting=lighting,
        source_scene_fingerprint=source_spec["fingerprint"],
        camera_close=[x.tolist() for x in views["close"]],
        camera_overview=[x.tolist() for x in views["overview"]],
        floor_square_size_m=[side, side],
        floor_square_bounds_xy_m=[
            (center - side / 2).tolist(),
            (center + side / 2).tolist(),
        ],
        table_footprint_bounds_xy_m=[low.tolist(), high.tolist()],
        minimum_floor_padding_m=float(
            np.min(
                np.concatenate([low - (center - side / 2), (center + side / 2) - high])
            )
        ),
        workcells=35,
        arms=70,
        frozen_state_sha256=hashlib.sha256(state.tobytes()).hexdigest(),
        variants=[],
    )
    for floor_index, (
        floor_name,
        pad_color,
        background,
        background_name,
        square,
    ) in enumerate(FLOORS):
        floor_mat = material(stage, f"Floor{floor_index}", pad_color)
        backdrop_mat = material(stage, f"Backdrop{floor_index}", background)
        UsdShade.MaterialBindingAPI.Apply(surround.GetPrim()).Bind(backdrop_mat)
        UsdShade.MaterialBindingAPI.Apply(pad.GetPrim()).Bind(floor_mat)
        (pad.MakeVisible if square else pad.MakeInvisible)()
        for table_index, (table_name, table_color, wood) in enumerate(TABLES):
            n = floor_index * 5 + table_index + 1
            for i in range(35):
                root = stage.GetPrimAtPath(f"/World/envs/env_{i}/Table")
                for prim in Usd.PrimRange(root):
                    if prim.IsA(UsdGeom.Gprim):
                        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                            tables[table_index]
                        )
                UsdShade.MaterialBindingAPI.Apply(tops[i].GetPrim()).Bind(
                    tables[table_index]
                )
                (tops[i].MakeVisible if wood else tops[i].MakeInvisible)()
            files = []
            for view, (eye, aim) in views.items():
                camera.set_world_poses_from_view(
                    torch.tensor(eye[None], dtype=torch.float32, device=sim.device),
                    torch.tensor(aim[None], dtype=torch.float32, device=sim.device),
                )
                for _ in range(24):
                    sim.render()
                camera.update(1 / 60, force_recompute=True)
                rgb = (
                    camera.data.output["rgb"][0, :, :, :3]
                    .cpu()
                    .numpy()
                    .astype(np.uint8)
                )
                filename = (
                    f"look-{n:02}.jpg" if view == "close" else f"overview-{n:02}.jpg"
                )
                Image.fromarray(rgb).save(args.output / filename, quality=96)
                files.append(
                    dict(
                        file=filename,
                        sha256=hashlib.sha256(
                            (args.output / filename).read_bytes()
                        ).hexdigest(),
                    )
                )
            manifest["variants"].append(
                dict(
                    id=n,
                    name=f"{table_name} / {floor_name}",
                    table=table_name,
                    floor=floor_name,
                    background=background_name,
                    table_linear_rgb=table_color,
                    floor_linear_rgb=pad_color,
                    background_linear_rgb=background,
                    wood_grain=wood,
                    files=files,
                )
            )
            atomic(args.output / "manifest.json", manifest)
            print(f"Rendered look {n:02}: {table_name}, {floor_name}", flush=True)
    after = (
        torch.cat(
            [a.data.joint_pos.flatten() for a in env.arms.values()]
            + [env.ball.data.root_pos_w.flatten()]
        )
        .cpu()
        .numpy()
    )
    if not np.array_equal(state, after):
        raise RuntimeError("Physics state changed between appearance previews")
    manifest["frozen_state_verified"] = True
    atomic(args.output / "manifest.json", manifest)


try:
    main()
except BaseException:
    atomic(args.output / "failure.json", dict(error=traceback.format_exc()))
    traceback.print_exc()
    os._exit(1)
app.close(skip_cleanup=True)
