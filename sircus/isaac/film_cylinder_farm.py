"""Film the live RTX camera while its 35 articulated PhysX shots train a shared PPO policy."""

import argparse
import copy
import hashlib
import json
import os
import shutil
import time
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--assets", default="/workspace/assets")
p.add_argument("--output", type=Path, required=True)
p.add_argument(
    "--dark-replay",
    type=Path,
    help="Replay recorded training actions on the compact film stage",
)
p.add_argument(
    "--resume-filming", type=Path, help="Resume a completed prefix at a camera cut"
)
p.add_argument(
    "--recording-plan",
    choices=("fixed-station-v1", "dark-tour-v1", "chosen-farm-v1"),
    default="fixed-station-v1",
    help="Camera/release plan for --dark-replay",
)
p.add_argument("--appearance", choices=("dark", "silver"), default="dark")
p.add_argument("--preview-only", action="store_true")
p.add_argument(
    "--camera-previews",
    action="store_true",
    help="Render ten fixed silver camera options",
)
p.add_argument("--camera-option", type=int, choices=range(1, 11))
p.add_argument("--camera-right-m", type=float, default=0.0)
p.add_argument("--camera-z-m", type=float)
p.add_argument("--scout-snapshot", action="store_true")
p.add_argument("--camera-json", type=Path)
p.add_argument(
    "--training-camera-json",
    type=Path,
    help="Record all new PPO batches at this fixed camera, with driven setup",
)
p.add_argument(
    "--geometry-menu-source",
    type=Path,
    help="Completed matching run supplying geometry-only preflight menus",
)
p.add_argument(
    "--expanded-pose-menu",
    type=Path,
    help="Expanded offline endpoint candidates for native qualification and 5-D PPO",
)
p.add_argument("--expanded-qualify-only", action="store_true")
p.add_argument("--expanded-options-per-cell", type=int, default=4)
p.add_argument(
    "--presentation-source", type=Path,
    help="Completed native expanded run supplying geometry-only filming menus",
)
p.add_argument("--presentation-camera-json", type=Path)
p.add_argument("--presentation-batches", type=int, default=30)
p.add_argument("--rl-study-source", type=Path,
               help="Completed native expanded run supplying geometry-only PPO options")
p.add_argument("--rl-study-camera-json", type=Path)
p.add_argument("--continuous-source", type=Path)
p.add_argument("--continuous-camera-json", type=Path)
p.add_argument("--continuous-batches", type=int, default=64)
p.add_argument("--continuous-feasibility", type=Path)
p.add_argument("--feasibility-guided-support", type=Path, help="Opt-in geometry-only local residual support; requires continuous mode")
p.add_argument("--diverse-settled-support", type=Path, help="Opt-in diverse geometry regions and measured driven setup; requires continuous mode")
p.add_argument("--settling-check-source", type=Path, help="Replay four fixed prior training requests for setup-only qualification, without releases or PPO")
p.add_argument("--shot-benchmark-plan", type=Path, help="Separate versioned sweep/PPO experiment; never changes historical pilot contracts")
p.add_argument("--shot-benchmark-method", choices=("sweep", "ppo"))
p.add_argument("--shot-benchmark-seed", type=int)
p.add_argument("--shot-benchmark-smoke", action="store_true")
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
if args.shot_benchmark_plan is not None:
    if (args.continuous_source is None or args.shot_benchmark_method is None
            or args.shot_benchmark_seed is None or args.diverse_settled_support is not None
            or args.feasibility_guided_support is not None or args.settling_check_source is not None
            or args.continuous_feasibility is not None or args.continuous_batches != 64):
        p.error("Benchmark requires its plan, method, seed and continuous source, without historical support/check modes")
elif args.shot_benchmark_method is not None or args.shot_benchmark_seed is not None or args.shot_benchmark_smoke:
    p.error("Benchmark options require --shot-benchmark-plan")
from presentation_recording import validate_options

try:
    presentation = validate_options(
        args.presentation_source, args.presentation_camera_json,
        args.presentation_batches, appearance=args.appearance,
        incompatible=bool(
            args.training_camera_json or args.geometry_menu_source
            or args.expanded_pose_menu or args.expanded_qualify_only
            or args.expanded_options_per_cell != 4 or args.dark_replay
            or args.resume_filming or args.camera_json or args.preview_only
            or args.camera_previews or args.camera_option is not None
            or args.scout_snapshot or args.camera_right_m != 0
            or args.camera_z_m is not None
            or args.recording_plan != "fixed-station-v1"
        ),
    )
except ValueError as error:
    p.error(str(error))
from rl_film_study import validate_options as validate_rl_options
try:
    rl_study = validate_rl_options(
        args.rl_study_source, args.rl_study_camera_json,
        appearance=args.appearance,
        incompatible=bool(
            presentation or args.training_camera_json or args.geometry_menu_source
            or args.expanded_pose_menu or args.expanded_qualify_only
            or args.expanded_options_per_cell != 4 or args.dark_replay
            or args.resume_filming or args.camera_json or args.preview_only
            or args.camera_previews or args.camera_option is not None
            or args.scout_snapshot or args.camera_right_m != 0
            or args.camera_z_m is not None
            or args.recording_plan != "fixed-station-v1"
            or args.presentation_batches != 30
        ),
    )
except ValueError as error:
    p.error(str(error))
from continuous_pilot import validate_options as validate_continuous_options
try:
    continuous = validate_continuous_options(
        args.continuous_source, args.continuous_camera_json, args.continuous_batches,
        appearance=args.appearance,
        incompatible=bool(
            presentation or rl_study or args.training_camera_json or args.geometry_menu_source
            or args.expanded_pose_menu or args.expanded_qualify_only
            or args.expanded_options_per_cell != 4 or args.dark_replay
            or args.resume_filming or args.camera_json or args.preview_only
            or args.camera_previews or args.camera_option is not None
            or args.scout_snapshot or args.camera_right_m != 0
            or args.camera_z_m is not None
            or args.recording_plan != "fixed-station-v1"
            or args.presentation_batches != 30
        ),
    )
except ValueError as error:
    p.error(str(error))
if args.diverse_settled_support is not None and (not continuous or args.feasibility_guided_support is not None):
    p.error("Diverse settled support requires continuous mode and excludes guided mode")
if args.settling_check_source is not None and args.diverse_settled_support is None:
    p.error("Setup-only check requires diverse-settled mode")
if args.feasibility_guided_support is not None and not continuous:
    p.error("--feasibility-guided-support requires continuous mode")
training_camera = args.training_camera_json is not None
menu_inputs = int(args.geometry_menu_source is not None) + int(
    args.expanded_pose_menu is not None
)
if (training_camera and menu_inputs != 1) or (not training_camera and menu_inputs):
    p.error(
        "Training-camera recording requires exactly one legacy or expanded menu source"
    )
if args.expanded_qualify_only and args.expanded_pose_menu is None:
    p.error("--expanded-qualify-only requires --expanded-pose-menu")
if not 2 <= args.expanded_options_per_cell <= 32:
    p.error("Expanded native menus require 2..32 options per cell")
if args.expanded_options_per_cell != 4 and args.expanded_pose_menu is None:
    p.error("--expanded-options-per-cell requires --expanded-pose-menu")
if training_camera and (
    args.dark_replay
    or args.resume_filming
    or args.camera_json
    or args.camera_previews
    or args.preview_only
    or args.appearance != "silver"
    or args.recording_plan != "fixed-station-v1"
):
    p.error("Training-camera recording requires silver and no replay/preview options")
if not args.enable_cameras:
    p.error("--enable_cameras is required")
if args.camera_previews and (not args.dark_replay or args.appearance != "silver"):
    p.error("--camera-previews requires --dark-replay and --appearance silver")
if args.camera_option is not None and not args.camera_previews:
    p.error("--camera-option requires --camera-previews")
if args.scout_snapshot and not args.camera_previews:
    p.error("--scout-snapshot requires --camera-previews")
if args.recording_plan == "chosen-farm-v1" and args.camera_json is None:
    p.error("chosen-farm-v1 requires --camera-json")
if args.camera_json is not None and (
    not args.dark_replay or args.appearance != "silver"
):
    p.error("--camera-json requires --dark-replay and --appearance silver")
if (
    args.camera_right_m != 0 or args.camera_z_m is not None
) and args.camera_option is None:
    p.error("Camera adjustments require --camera-option")
if shutil.disk_usage(args.output.parent).free < 3 * 1024**3:
    p.error("Less than 3 GiB free for filming")
args.output.mkdir(exist_ok=False)
app = AppLauncher(args).app

import isaaclab.sim as sim_utils
import numpy as np
import torch
from cylinder_farm import (
    HIGH,
    LOGSTD,
    LOW,
    SEED,
    WARM,
    menu_batches,
    observation,
    origins,
    project_actions,
    specification,
)
from farm_contract import reward, statistics
from farm_flight import SHOTS, camera_pose, choose_front
from farm_policy import Policy, update, weight_hash
from feasibility_runtime import FeasibilityShots
from isaaclab.scene import InteractiveScene
from isaaclab.sensors import Camera, CameraCfg
from isaacsim.core.cloner import Cloner
from parallel_assets import bucket_asset, robot_body_filters, scene_config
from parallel_runtime import atomic
from PIL import Image
from scipy.spatial.transform import Rotation


class RectangularScene(InteractiveScene):
    grid_origins = staticmethod(origins)

    def clone_environments(self, copy_from_source=False):
        # Isaac Lab ordinarily uses a square GridCloner. Supply the actual ordered
        # transforms to Isaac's underlying cloner before Physics replicates the assets.
        positions = self.grid_origins()
        if len(positions) != self.num_envs:
            raise ValueError("Each of the 35 grid sites needs its own clone")
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


class FilmShots(FeasibilityShots):
    first_step = None
    index = None
    first_front = None
    next_frame = 0

    def exposed_rim_probes(self):
        """Find a clear diagnostic ray when a parked link overhangs a bucket."""
        bucket = self.cfg["bucket"]
        top = bucket["assumed_interior_depth_m"] + bucket["assumed_bottom_thickness_m"]
        self.rim_probe_height_m = self.cfg["ball"]["radius_m"] + 0.012
        self.rim_probe_directions = np.zeros((self.n, 2))
        found = np.zeros(self.n, dtype=bool)
        evidence = []
        for angle in [0, 90, 180, 270, 45, 135, 225, 315]:
            direction = np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))])
            xy = self.bucket_xy + direction * bucket["inner_top_diameter_m"] / 2
            self.set_balls(
                np.column_stack([xy, np.full(self.n, top + self.rim_probe_height_m)])
            )
            first = np.zeros((self.n, 6))
            for _ in range(48):
                self.tick()
                f = self.forces()
                touched = (first.max(axis=1) <= 0.01) & (f.max(axis=1) > 0.01)
                first[touched] = f[touched]
            clear = (first[:, 1] > 0.01) & (
                first[:, [0, 2, 3, 4, 5]].max(axis=1) <= 0.01
            )
            self.rim_probe_directions[clear & ~found] = direction
            found |= clear
            evidence.append(dict(azimuth_deg=angle, first_contact_N=first.tolist()))
            if found.all():
                break
        atomic(
            self.dest / "rim-rays.json",
            dict(
                rays=evidence,
                passed=found.tolist(),
                directions=self.rim_probe_directions.tolist(),
                clearance_m=self.rim_probe_height_m,
            ),
        )
        if not found.all():
            raise RuntimeError(f"No unobstructed rim ray for {np.flatnonzero(~found)}")

    def prepare(self, actions):
        result = super().prepare(actions)
        name = SHOTS[self.index][0]
        valid = [reason is None for reason in result[3]]
        self.focus = choose_front(
            valid, name, self.first_front if name == "front-right" else None
        )
        if name == "front-left":
            self.first_front = self.focus
        return result

    def qualify_menu(self, cells):
        """Measure commandable actions before the first training throw."""
        menus = [[] for _ in cells]
        checks = []
        for proposals in menu_batches(cells):
            _, _, _, reasons = super().prepare(proposals)
            for options, action, reason in zip(menus, proposals, reasons):
                if reason is None:
                    options.append(action)
            checks.append(reasons)
            atomic(
                self.dest / "menu-progress.json",
                dict(batches=len(checks), options_per_cell=[len(m) for m in menus]),
            )
        atomic(
            self.dest / "menu.json",
            dict(
                menus=menus,
                checks=checks,
                rule="Both endpoints pass IK and measured settling before a ball is released.",
            ),
        )
        if sum(bool(m) for m in menus[:7]) < 2:
            raise RuntimeError(
                "Fewer than two front-row sites have a qualified release"
            )
        return menus

    def tick(self):
        super().tick()
        if self.first_step is not None:
            self.capture_frame()

    def render(self):
        # ParallelShots calls this immediately after releasing each batch.
        if self.record_video and self.first_step is None:
            self.first_step = self.steps
            self.capture_frame()

    def capture_frame(self):
        _, length, stride = SHOTS[self.index]
        frame = len(self.frames)
        if frame >= length or self.steps - self.first_step != frame * stride:
            return
        eye, target = camera_pose(self.index, frame, self.spec["cells"], self.focus)
        self.write_rgb(eye, target, stride)

    def write_rgb(self, eye, target, elapsed_steps):
        device = self.sim.device
        self.camera.set_world_poses_from_view(
            torch.as_tensor(eye[None], dtype=torch.float32, device=device),
            torch.as_tensor(target[None], dtype=torch.float32, device=device),
        )
        # Render at a fixed physics step. Flush RTX's previous camera/physics frame.
        for _ in range(3):
            self.sim.render()
        self.camera.update(elapsed_steps * self.dt, force_recompute=True)
        data = self.camera.data
        pos = data.pos_w[0].cpu().numpy()
        q = data.quat_w_ros[0].cpu().numpy()
        optical = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()[:, 2]
        expected = (target - eye) / np.linalg.norm(target - eye)
        if np.linalg.norm(pos - eye) > 0.0002 or optical @ expected < 0.99999:
            raise RuntimeError("RTX camera readback missed its shot")
        rgb = data.output["rgb"][0, :, :, :3].cpu().numpy().astype(np.uint8)
        Image.fromarray(rgb).save(
            self.frames_path / f"{self.next_frame:05d}.jpg", quality=93
        )
        record = dict(
            frame=self.next_frame,
            physics_step=self.steps,
            eye_m=pos.tolist(),
            target_m=target.tolist(),
        )
        if self.focus is not None:
            record["focus_ball_world_m"] = (
                self.ball.data.root_pos_w[self.focus].cpu().numpy().tolist()
            )
        self.frames.append(record)
        self.next_frame += 1

    def film(self, index, shots):
        self.index, self.frames, self.first_step = index, [], None
        rows, raw = self.shot_batch(shots, record_video=True)
        # If all episodes terminate early, keep the live physics clock and camera
        # rolling for the end of the planned shot; terminal scores stay frozen.
        while len(self.frames) < SHOTS[index][1]:
            self.tick()
        self.first_step = None
        return rows, dict(
            shot=SHOTS[index][0],
            focus=self.focus,
            frames=self.frames,
            rollout=rows,
            trajectory_m=raw["trajectory_m"],
        )


def main():
    started = time.monotonic()
    dest = args.output
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    cfg = json.loads(Path(__file__).with_name("scene.json").read_text())
    poses = json.loads((Path(args.assets) / "poses.json").read_text())
    presentation_bundle = None
    if presentation or rl_study or continuous:
        from presentation_recording import load_source

        presentation_bundle = load_source(
            args.continuous_source if continuous else (args.rl_study_source if rl_study else args.presentation_source),
            args.continuous_camera_json if continuous else (args.rl_study_camera_json if rl_study else args.presentation_camera_json),
        )
        spec = copy.deepcopy(presentation_bundle["spec"])
        if spec["scene"] != cfg:
            raise ValueError("Presentation requires the unchanged source scene")
    elif args.expanded_pose_menu is not None:
        from expanded_farm import make_spec

        spec = make_spec(cfg, len(SHOTS))
    else:
        spec = specification(cfg, len(SHOTS))
    saved_camera = None
    camera_binding = None
    camera_path = (
        args.continuous_camera_json or args.rl_study_camera_json or args.presentation_camera_json
        or args.training_camera_json or args.camera_json
    )
    if camera_path is not None:
        if args.expanded_pose_menu is not None or presentation or rl_study or continuous:
            from expanded_farm import bind_camera
            from camera_options import preview_origins

            saved_camera, camera_binding = bind_camera(
                camera_path, spec, preview_origins(), dest / "selected-camera.json"
            )
            atomic(dest / "camera-binding.json", camera_binding)
        else:
            from saved_camera import load_camera

            saved_camera = load_camera(camera_path, spec["fingerprint"])
            shutil.copyfile(camera_path, dest / "selected-camera.json")
    if continuous:
        shutil.copyfile(args.continuous_source / "contract.json", dest / "contract.json")
    elif presentation:
        shutil.copyfile(args.presentation_source / "contract.json", dest / "contract.json")
    elif rl_study:
        shutil.copyfile(args.rl_study_source / "contract.json", dest / "contract.json")
    else:
        atomic(
            dest / ("source-contract.json" if args.dark_replay else "contract.json"), spec
        )
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(
            dt=cfg["simulation"]["physics_dt_s"],
            render_interval=8,
            device=args.device,
            physx=sim_utils.PhysxCfg(enable_ccd=False),
            render=(
                sim_utils.RenderCfg(enable_ambient_occlusion=False)
                if args.appearance == "silver"
                else sim_utils.RenderCfg()
            ),
        )
    )
    bucket = dest / "bucket.usda"
    bucket_asset(bucket, cfg["bucket"])
    scfg, filters = scene_config(
        cfg,
        poses,
        args.assets,
        bucket,
        spec["num_envs"],
        robot_body_filters(args.assets),
    )
    layout = origins
    scene_type = RectangularScene
    shots_type = FilmShots
    if args.dark_replay or training_camera or presentation or rl_study or continuous:
        import dark_stage
        from dark_replay import ReplayShots, run_replay

        if args.appearance == "silver":
            from silver_style import configure_stage, stage_settings
        else:
            from dark_stage import configure_stage, stage_settings
        configure_stage(scfg, cfg)
        layout = dark_stage.origins
        if (
            args.camera_previews
            or args.recording_plan == "chosen-farm-v1"
            or training_camera
            or presentation
            or rl_study
            or continuous
        ):
            from camera_options import preview_origins
            from camera_preview import configure_backdrop

            layout = preview_origins
            preview_backdrop = configure_backdrop()

        class CompactScene(RectangularScene):
            grid_origins = staticmethod(layout)

        if continuous:
            from continuous_pilot import ContinuousShots
            if args.shot_benchmark_plan is not None:
                from shot_benchmark_native import BenchmarkShots
                ContinuousShots = BenchmarkShots
            elif args.diverse_settled_support is not None:
                from settled_transit import SettledTransitShots
                ContinuousShots = SettledTransitShots

            class CompactShots(ContinuousShots, FilmShots):
                pass

        elif rl_study:
            from rl_film_study import RLStudyShots

            class CompactShots(RLStudyShots, FilmShots):
                pass

        elif presentation:
            from presentation_recording import PresentationShots

            class CompactShots(PresentationShots, FilmShots):
                pass

        elif training_camera:
            from action_training import ActionTrainingShots

            class CompactShots(ActionTrainingShots, FilmShots):
                pass

        else:

            class CompactShots(ReplayShots, FilmShots):
                pass

        scene_type, shots_type = CompactScene, CompactShots
    scene = scene_type(scfg)
    camera = Camera(
        CameraCfg(
            prim_path="/World/CinemaCamera",
            width=1920,
            height=1080,
            update_period=0,
            update_latest_camera_pose=True,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=23,
                horizontal_aperture=36,
                vertical_aperture=20.25,
                clipping_range=(0.04, 150),
            ),
        )
    )
    sim.reset()
    env = shots_type(
        sim, scene, camera, None, cfg, poses, args, copy.deepcopy(spec), filters, dest
    )
    if not np.array_equal(env.origins, layout()):
        raise ValueError("Isaac workcell origins disagree with the 5 by 7 layout")
    if args.dark_replay or training_camera or presentation or rl_study or continuous:
        env.stage_metadata = stage_settings()
        env.saved_camera = saved_camera
        if camera_binding is not None:
            env.camera_binding = camera_binding
        if (
            args.camera_previews
            or args.recording_plan == "chosen-farm-v1"
            or training_camera
            or presentation
            or rl_study
            or continuous
        ):
            env.stage_metadata["camera_backdrop"] = preview_backdrop
        if args.preview_only or args.camera_previews:
            from recording_plans import recording_plan

            shots, view_pose = (
                recording_plan(args.recording_plan, saved_camera)
                if args.recording_plan == "chosen-farm-v1"
                else recording_plan(args.recording_plan)
            )
            source_clip = json.loads((args.dark_replay / "clip-06.json").read_text())[
                "rollout"
            ]
            env.focus, env.frames = 1, []
            for name, arm in env.arms.items():
                q = torch.tensor(
                    [r["final_joint_positions_rad"][name] for r in source_clip],
                    device=sim.device,
                )
                arm.write_joint_state_to_sim(q, torch.zeros_like(q))
                arm.set_joint_position_target(q)
                arm.reset()
            for _ in range(120):
                env.tick()
            env.set_balls(
                np.array(
                    [
                        (
                            r["release_ball_center_m"]
                            if not r["invalid"]
                            else [0.1, -0.8, 0.04]
                        )
                        for r in source_clip
                    ]
                )
            )
            env.tick()
            if args.camera_previews:
                from camera_preview import render_options

                render_options(env, args.dark_replay)
                return
            eye, aim = view_pose(shots[0], 0, env.spec["cells"], env.origins)
            camera.set_world_poses_from_view(
                torch.tensor(eye[None], dtype=torch.float32, device=sim.device),
                torch.tensor(aim[None], dtype=torch.float32, device=sim.device),
            )
            for _ in range(24):
                sim.render()
            env.write_rgb(eye, aim, 1)
            shutil.copyfile(env.frames_path / "00000.jpg", dest / "preview.jpg")
            atomic(
                dest / "preview.json",
                dict(
                    appearance=args.appearance,
                    stage=env.stage_metadata,
                    eye_m=eye.tolist(),
                    target_m=aim.tolist(),
                ),
            )
            return
        env.foreign_probe_index = 7
    env.exposed_rim_probes()
    # Place the tabletop probe in the gap between four neighboring grid sites.
    env.probe_positions["table"] = [1.313, -0.625, 0.25]
    probes = env.probes()
    atomic(dest / "probes.json", probes)
    if continuous:
        from continuous_pilot import run_study

        if args.shot_benchmark_plan is not None:
            from shot_benchmark_native import run_benchmark
            run_benchmark(env, presentation_bundle, args.shot_benchmark_plan,
                          args.shot_benchmark_method, args.shot_benchmark_seed,
                          smoke=args.shot_benchmark_smoke)
            return
        if args.continuous_feasibility is None:
            raise ValueError("Continuous pilot requires a completed geometry-only feasibility map")

        if args.diverse_settled_support is not None:
            from diverse_pilot import run_study as run_diverse, run_setup_check
            if args.settling_check_source is not None:
                run_setup_check(env, presentation_bundle, args.continuous_feasibility, args.settling_check_source)
            else:
                run_diverse(env, presentation_bundle, args.continuous_batches, args.continuous_feasibility, args.diverse_settled_support)
        elif args.feasibility_guided_support is not None:
            from feasibility_guided_pilot import run_study as run_guided
            run_guided(env, presentation_bundle, args.continuous_batches, args.continuous_feasibility, args.feasibility_guided_support)
        else:
            run_study(env, presentation_bundle, args.continuous_batches, args.continuous_feasibility)
        return
    if rl_study:
        from rl_film_study import run_study

        run_study(env, presentation_bundle)
        return
    if presentation:
        from presentation_recording import run_presentation

        run_presentation(env, presentation_bundle, args.presentation_batches)
        return
    if training_camera:
        from action_training import run_training

        if args.expanded_pose_menu is not None:
            from expanded_pose_native import qualify_native_menu

            env.expanded_menu_path = qualify_native_menu(
                env,
                spec,
                args.expanded_pose_menu,
                desired_per_cell=args.expanded_options_per_cell,
            )
            if args.expanded_qualify_only:
                return
        run_training(env, spec, args.geometry_menu_source)
        return
    if args.dark_replay:
        run_replay(env, args.dark_replay, spec)
        return
    menus = env.qualify_menu(spec["cells"])
    menu_hash = hashlib.sha256((dest / "menu.json").read_bytes()).hexdigest()
    policy = Policy(LOW, HIGH, WARM, LOGSTD).to(sim.device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=spec["lr"])
    initial = weight_hash(policy)
    obs = torch.tensor(
        [observation(c) for c in spec["cells"]], dtype=torch.float32, device=sim.device
    )
    batches, changes, buffer = [], [], []
    for i, (name, _, stride) in enumerate(SHOTS):
        before = weight_hash(policy)
        with torch.no_grad():
            dist = policy.distribution(obs)
            latent = dist.sample()
            logp = dist.log_prob(latent).sum(-1)
        rows, clip = env.film(
            i, project_actions(spec["cells"], latent.cpu().numpy(), menus)
        )
        returns = torch.tensor([reward(r) for r in rows], device=sim.device)
        buffer.append((obs, latent, logp, returns))
        clip.update(
            policy_hash=before,
            latent=latent.cpu().tolist(),
            rewards=returns.cpu().tolist(),
        )
        atomic(dest / f"clip-{i:02}.json", clip)
        focus = clip["focus"]
        sample = dict(
            shot=name,
            start_frame=clip["frames"][0]["frame"],
            frames=len(clip["frames"]),
            physics_stride=stride,
            focus=rows[focus]["cell_id"] if focus is not None else None,
            focus_bounce=rows[focus]["board_bounce"] if focus is not None else None,
            focus_capture=rows[focus]["route_success"] if focus is not None else None,
            policy_hash=before,
            **statistics(rows),
        )
        batches.append(sample)
        if len(buffer) == spec["batches_per_update"]:
            changes.append(
                update(policy, optimizer, *[torch.cat(a) for a in zip(*buffer)], spec)
            )
            buffer.clear()
        atomic(dest / "progress.json", dict(batches=batches, updates=len(changes)))
        print(json.dumps(sample), flush=True)
    final = weight_hash(policy)
    if final == initial or len(changes) != len(SHOTS) // 2:
        raise RuntimeError("Missing measured PPO updates")
    torch.save(
        dict(state_dict=policy.state_dict(), contract=spec, menu_sha256=menu_hash),
        dest / "cylinder-policy.pt",
    )
    atomic(
        dest / "result.json",
        dict(
            version=spec["version"],
            fingerprint=spec["fingerprint"],
            menu_sha256=menu_hash,
            gpu=torch.cuda.get_device_name(),
            workcells=env.n,
            articulations=2 * env.n,
            initial_hash=initial,
            final_hash=final,
            updates=changes,
            batches=batches,
            frames=env.next_frame,
            resolution=[1920, 1080],
            fps=30,
            physics_dt_s=env.dt,
            max_bucket_drift_m=env.peak_bucket_drift,
            contact_probes=[p["kind"] for p in probes],
            seconds=time.monotonic() - started,
            source_sha256={
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(Path(__file__).parent.glob("*.py"))
            },
        ),
    )


try:
    main()
except BaseException:
    atomic(args.output / "failure.json", dict(error=traceback.format_exc()))
    traceback.print_exc()
    os._exit(1)
app.close(skip_cleanup=True)
