"""Opt-in fresh-allocation environment for paired32/64/128 timestep studies.

``TimestepEnvironment`` is a lazy factory returning a ContactEnvironment subclass.
The inherited flight/scoring loop is unchanged. Native dependencies are imported
only on construction; inspecting options and contracts is CPU-only. No existing
environment or allocated Warp model has its timestep changed by this adapter.
"""
import json
import numbers
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .gpu_slot import GPU_UUID

VERSION = "sircus-articulated-passive-timestep-sensitivity-v1"
SUBSTEPS = (32, 64, 128)


def checked_substeps(substeps):
    if isinstance(substeps, bool) or not isinstance(substeps, numbers.Integral) or substeps not in SUBSTEPS:
        raise ValueError("This opt-in experiment supports exactly32/64/128 substeps")
    return int(substeps)


def physics_settings(substeps):
    substeps = checked_substeps(substeps)
    return dict(version=f"{VERSION}-{substeps}", parent_contact_model="sircus-held-board-contact-v2",
        substeps=substeps, native_dt_s=1 / (240 * substeps), scoring_dt_s=1 / 240,
        horizon_s=4., board_mode="articulated", gravity_mode="passive_only",
        solref=[.004, .15], contact_friction=.35, solver="newton",
        solver_iterations=100, solver_tolerance=1e-8, cone="pyramidal", integrator="implicitfast",
        scope="Paired prescribed endpoint ball flights; no timestep-convergence or actuator-tracking claim")


def _option_array(value):
    # Warp arrays expose numpy(); a tensor/fake array may expose detach/cpu too.
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    result = np.asarray(value)
    if result.size == 0 or not np.isfinite(result).all():
        raise RuntimeError("Finite nonempty effective Warp options required")
    return result


def validate_options(native, warp, substeps):
    """Read/check every effective option used by the controlled intervention.

    Native options retain double precision. Warp timestep/tolerance/gravity are
    float32 device arrays in pinned MuJoCo Warp3.11; compare their stored values
    exactly with the float32 representation, including every expanded world.
    """
    substeps = checked_substeps(substeps)
    expected_dt = 1 / (240 * substeps)
    if (not np.isfinite(native.timestep) or not np.isclose(native.timestep, expected_dt, rtol=0, atol=1e-15)
            or int(native.solver) != 2 or int(native.integrator) != 3 or int(native.cone) != 0
            or int(native.iterations) != 100 or not np.isfinite(native.tolerance)
            or not np.isclose(native.tolerance, 1e-8, rtol=0, atol=1e-12)
            or not np.array_equal(np.asarray(native.gravity), [0., 0., -9.81])):
        raise RuntimeError("Effective native model differs from the declared timestep experiment")
    dt, tolerance, gravity = [_option_array(getattr(warp, name)) for name in ("timestep", "tolerance", "gravity")]
    if (any(value.dtype != np.dtype('float32') for value in (dt, tolerance, gravity))
            or not np.all(dt == np.float32(expected_dt)) or not np.all(tolerance == np.float32(1e-8))
            or gravity.shape[-1:] != (3,) or not np.all(gravity == np.asarray([0., 0., -9.81], np.float32))
            or int(warp.solver) != 2 or int(warp.integrator) != 3 or int(warp.cone) != 0
            or int(warp.iterations) != 100):
        raise RuntimeError("Allocated Warp model differs from its native timestep/options")
    return dict(version=f"{VERSION}-{substeps}", substeps=substeps, timestep_s=float(native.timestep),
        solver=int(native.solver), integrator=int(native.integrator), cone=int(native.cone),
        iterations=int(native.iterations), tolerance=float(native.tolerance), gravity=np.asarray(native.gravity).tolist(),
        board_mode="articulated", gravity_mode="passive_only", scoring_dt_s=1 / 240,
        warp=dict(timestep_s=dt.tolist(), tolerance=tolerance.tolist(), gravity=gravity.tolist(),
            solver=int(warp.solver), integrator=int(warp.integrator), cone=int(warp.cone), iterations=int(warp.iterations),
            dtypes=dict(timestep=str(dt.dtype), tolerance=str(tolerance.dtype), gravity=str(gravity.dtype)),
            precision="Stored float32 values; checked in every expanded world"))


def _dependencies():
    import mujoco
    import torch
    from mjlab.sim import Simulation, SimulationCfg, MujocoCfg
    from .contact_env import ContactEnvironment
    from .contact_model import make_model
    from .expanded_actions import checked_bounds, OLD_BOUNDS
    from .learning_ik import ShotIK
    from .run import sensor_config
    return SimpleNamespace(**locals())


def _initialize(self, scene, assets, n, bounds, substeps, deps):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != GPU_UUID:
        raise RuntimeError("Select ONLY the authorized RTX4090 UUID")
    torch, mujoco = deps.torch, deps.mujoco
    if torch.cuda.device_count() != 1 or "4090" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Select ONLY the authorized RTX4090")
    self.substeps, self.board_mode, self.gravity_mode = substeps, "articulated", "passive_only"
    self.scene, self.n, self.device = scene, n, "cuda:0"
    self.action_bounds = deps.OLD_BOUNDS if bounds is None else dict(bounds)
    deps.checked_bounds(self.action_bounds)
    spec, self.attached_model = deps.make_model(scene, assets, self.board_mode)
    self.sensors = deps.sensor_config(spec, spec.compile())
    self.m = spec.compile()
    # These values are applied BEFORE put_model/CUDA graph allocation by mjlab1.6.
    # The old ContactEnvironment and any already allocated engine are untouched.
    options = deps.MujocoCfg(timestep=1 / (240 * substeps), integrator="implicitfast", cone="pyramidal",
        solver="newton", iterations=100, tolerance=1e-8, gravity=(0., 0., -9.81))
    self.sim = deps.Simulation(num_envs=n, cfg=deps.SimulationCfg(mujoco=options, nconmax=512, njmax=2048),
        model=self.m, device=self.device)
    self.effective_options = validate_options(self.sim.mj_model.opt, self.sim.model.opt, substeps)
    self.physics_version = self.effective_options["version"]
    for sensor in self.sensors:
        sensor.initialize(self.sim.mj_model, self.sim.model, self.sim.data, self.device)
    self.ik = deps.ShotIK(scene, assets, device=self.device)
    self.seeds = json.loads((Path(assets) / "poses.json").read_text())
    m = self.m
    joint = lambda name: m.joint(name).id
    self.ball_q = int(m.jnt_qposadr[joint("ball/free")]); self.ball_v = int(m.jnt_dofadr[joint("ball/free")])
    self.board = m.body("Arm14/impact_board").id; self.flange = m.body("Arm10/link6").id
    self.bucket = m.body("bucket").id; self.mocap = int(m.body_mocapid[self.bucket])
    self.board_mocap = int(m.body_mocapid[self.board])
    self.q10 = [int(m.jnt_qposadr[joint(f"Arm10/joint{k}")]) for k in range(1, 7)]
    self.q14 = [int(m.jnt_qposadr[joint(f"Arm14/joint{k}")]) for k in range(1, 7)]
    self.arm_q = self.q10 + self.q14
    self.arm_v = [int(m.jnt_dofadr[joint(f"Arm{role}/joint{k}")]) for role in (10, 14) for k in range(1, 7)]
    self.actuator_q = [int(m.jnt_qposadr[j]) for j in m.actuator_trnid[:, 0]]
    self.fingers = [int(m.jnt_qposadr[joint(f"Arm10/finger_{side}_joint")]) for side in ("left", "right")]
    self.cpu, self.attached_data = mujoco.MjData(m), mujoco.MjData(self.attached_model)
    self.links = {role: [m.body(f"Arm{role}/link{k}").id for k in range(1, 7)] for role in (10, 14)}
    self.bucket_geoms = [i for i in range(m.ngeom) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith("bucket/")]
    self.last_arrays = None


def TimestepEnvironment(scene, assets, n=64, bounds=None, *, substeps=64,
                        board_mode="articulated", gravity_mode="passive_only"):
    """Create a fresh native engine with unchanged parent run/scoring methods."""
    substeps = checked_substeps(substeps)
    if isinstance(n, bool) or not isinstance(n, numbers.Integral) or not 1 <= n <= 128:
        raise ValueError("Finite experiment world count must be1..128")
    if board_mode != "articulated" or gravity_mode != "passive_only":
        raise ValueError("This experiment fixes articulated board and passive-only gravity")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != GPU_UUID:
        raise RuntimeError("Select ONLY the authorized RTX4090 UUID")
    deps = _dependencies()
    class TimestepContactEnvironment(deps.ContactEnvironment):
        def __init__(self):
            _initialize(self, scene, assets, int(n), bounds, substeps, deps)
    return TimestepContactEnvironment()
