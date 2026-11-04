"""Additive, read-only pose instrumentation for prescribed-endpoint throws.

The calibrated environment, stepping, contacts and scorers are unchanged. A
scoped logger records post-integration joint states on the existing native
stream. After the complete rollout, CPU MuJoCo kinematics reconstructs transforms
from those states. No forward/kinematics call touches the running GPU model.

Every interval resets both arms to their commanded endpoint before stepping.
Consequently contact interval[k,k+1] uses the held pre-step board pose; a tiny
integrated excursion at state[k+1] is separately observable and is reset before
the next interval. These data do not establish compliant or controlled-arm
settling, a physical gripper release, or the contact point on the board.
"""

from contextlib import contextmanager
import threading
import time

import numpy as np

from .calibrated_env import CalibratedEnvironment
from .calibrated_score import FORCE_THRESHOLD_N


VERSION = "sircus-native-impact-pose-log-v1"
FK_UNIFORM_HZ = 240
IMPACT_WINDOW_S = .010
_LOGGER_LOCK = threading.RLock()


def model_ids(model):
    """Resolve named native joints/bodies instead of assuming model ordering."""
    return dict(
        q=[int(model.jnt_qposadr[model.joint(f"Arm{r}/joint{k}").id])
           for r in (10, 14) for k in range(1, 7)],
        v=[int(model.jnt_dofadr[model.joint(f"Arm{r}/joint{k}").id])
           for r in (10, 14) for k in range(1, 7)],
        board=model.body("Arm14/impact_board").id,
        flange=model.body("Arm10/link6").id,
        links10=[model.body(f"Arm10/link{k}").id for k in range(1, 7)],
        links14=[model.body(f"Arm14/link{k}").id for k in range(1, 7)],
    )


def sample_indices(times, forces, *, hz=FK_UNIFORM_HZ, window_s=IMPACT_WINDOW_S):
    """Uniform full-horizon FK plus every native state around first impact.

    Force[k] describes [time[k],time[k+1]]. Both endpoints of that interval and
    the native samples within the declared padding are always included.
    """
    times, forces = np.asarray(times, float), np.asarray(forces, float)
    if (times.ndim != 1 or len(times) < 2 or forces.shape != (len(times)-1,)
            or not np.isfinite(times).all() or not np.isfinite(forces).all()
            or np.any(np.diff(times) <= 0) or hz <= 0 or window_s < 0):
        raise ValueError("Finite ordered native times and one force per interval required")
    dt = float(times[1]-times[0])
    if not np.allclose(np.diff(times), dt, atol=1e-12, rtol=0):
        raise ValueError("Pose instrumentation requires the constant native clock")
    stride = round(1/(hz*dt))
    if stride < 1 or abs(stride*dt-1/hz) > 1e-12:
        raise ValueError("Uniform FK rate must divide the native step rate")
    selected = set(range(0, len(times), stride)) | {len(times)-1}
    hit = np.flatnonzero(forces > FORCE_THRESHOLD_N)
    first = int(hit[0]) if len(hit) else -1
    if first >= 0:
        near = np.flatnonzero((times >= times[first]-window_s-1e-12)
                             & (times <= times[first+1]+window_s+1e-12))
        selected.update(int(i) for i in near)
    return np.asarray(sorted(selected), dtype=np.int64), first


def _fk(model, data, ids, joints):
    """CPU FK of measured integrated joints, without integration or dynamics."""
    import mujoco

    data.qpos[:] = model.qpos0
    data.qpos[ids["q"]] = joints
    data.qvel[:] = 0
    mujoco.mj_kinematics(model, data)
    return dict(
        board_center_m=data.xpos[ids["board"]].copy(),
        board_quaternion_wxyz=data.xquat[ids["board"]].copy(),
        board_normal=data.xmat[ids["board"]].reshape(3, 3)[:, 2].copy(),
        flange10_center_m=data.xpos[ids["flange"]].copy(),
        flange10_quaternion_wxyz=data.xquat[ids["flange"]].copy(),
        link_positions10_m=data.xpos[ids["links10"]].copy(),
        link_positions14_m=data.xpos[ids["links14"]].copy(),
    )


def reconstruct_pose_arrays(model, joints, velocity, times, forces, ball, ball_velocity):
    """Build ragged-per-world FK history, padded with an explicit validity mask."""
    import mujoco

    q, v, times = np.asarray(joints), np.asarray(velocity), np.asarray(times, float)
    ball, ball_velocity, forces = map(np.asarray, (ball, ball_velocity, forces))
    if (q.ndim != 3 or q.shape[-1] != 12 or v.shape != q.shape
            or q.shape[0] != len(times) or forces.shape != (len(times)-1, q.shape[1], 6)
            or ball.shape != (*q.shape[:2], 3) or ball_velocity.shape != ball.shape
            or not all(np.isfinite(a).all() for a in (q, v, times, forces, ball, ball_velocity))):
        raise ValueError("Complete finite joint, ball and force state histories required")
    ids, data = model_ids(model), mujoco.MjData(model)
    selections = [sample_indices(times, forces[:, world, 0]) for world in range(q.shape[1])]
    size = max(len(item[0]) for item in selections)
    shape = (size, q.shape[1])
    result = dict(
        pose_time_s=times.copy(), pose_joint_qpos_rad=q.copy(), pose_joint_qvel_rad_s=v.copy(),
        # The frozen environment slices every multidimensional observation as
        # value[:, world], so even constant metadata needs a sample axis.
        pose_held_joint_qpos_rad=q[:1].copy(),
        pose_sample_state_index=np.full(shape, -1, dtype=np.int64),
        pose_sample_valid=np.zeros(shape, dtype=bool), pose_sample_time_s=np.zeros(shape),
        pose_first_impact_interval=np.asarray([item[1] for item in selections], dtype=np.int64),
    )
    suffixes = dict(board_center_m=(3,), board_normal=(3,), board_quaternion_wxyz=(4,),
                    flange10_center_m=(3,), flange10_quaternion_wxyz=(4,),
                    link_positions10_m=(6, 3), link_positions14_m=(6, 3),
                    ball_center_m=(3,), ball_velocity_m_s=(3,))
    for name, tail in suffixes.items():
        result["pose_"+name] = np.zeros((*shape, *tail), dtype=np.float64)
    for world, (indices, _) in enumerate(selections):
        count = len(indices)
        result["pose_sample_state_index"][:count, world] = indices
        result["pose_sample_valid"][:count, world] = True
        result["pose_sample_time_s"][:count, world] = times[indices]
        result["pose_ball_center_m"][:count, world] = ball[indices, world]
        result["pose_ball_velocity_m_s"][:count, world] = ball_velocity[indices, world]
        for slot, index in enumerate(indices):
            for key, value in _fk(model, data, ids, q[index, world]).items():
                result["pose_"+key][slot, world] = value
    return result


def _angle_degrees(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.rad2deg(np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1), np.sum(a*b, axis=-1)))


def pose_summary(arrays, world, initial_row=None):
    """Derive compact per-shot observations from the stored pose histories."""
    valid = arrays["pose_sample_valid"][:, world]
    states = arrays["pose_sample_state_index"][valid, world]
    q = arrays["pose_joint_qpos_rad"][:, world]
    v = arrays["pose_joint_qvel_rad_s"][:, world]
    centres = arrays["pose_board_center_m"][valid, world]
    normals = arrays["pose_board_normal"][valid, world]
    first = int(arrays["pose_first_impact_interval"][world])
    times = arrays["pose_time_s"]

    def observation(index):
        slot = int(np.flatnonzero(states == index)[0])
        return dict(native_state_index=int(index), time_s=float(times[index]),
                    board_center_m=centres[slot].tolist(), board_normal=normals[slot].tolist(),
                    board_quaternion_wxyz=arrays["pose_board_quaternion_wxyz"][valid, world][slot].tolist(),
                    flange10_center_m=arrays["pose_flange10_center_m"][valid, world][slot].tolist(),
                    q10_rad=q[index, :6].tolist(), q14_rad=q[index, 6:].tolist(),
                    qvel10_rad_s=v[index, :6].tolist(), qvel14_rad_s=v[index, 6:].tolist(),
                    ball_center_m=arrays["pose_ball_center_m"][valid, world][slot].tolist(),
                    ball_velocity_m_s=arrays["pose_ball_velocity_m_s"][valid, world][slot].tolist())

    dt = float(times[1]-times[0])
    initial = observation(0)
    result = dict(
        version=VERSION, status="verified", native_joint_hz=1/dt, fk_uniform_hz=FK_UNIFORM_HZ,
        dense_impact_window_s=IMPACT_WINDOW_S, native_states=len(times), fk_samples=int(valid.sum()),
        joint_order=[f"Arm{r}/joint{k}" for r in (10, 14) for k in range(1, 7)],
        initial=initial, held_pre_step_board_pose=dict(
            board_center_m=initial["board_center_m"], board_normal=initial["board_normal"],
            board_quaternion_wxyz=initial["board_quaternion_wxyz"]),
        first_impact_interval=None, impact_before=None, impact_after=None,
        maximum_postintegration_board_drift_mm=float(np.linalg.norm(centres-centres[0], axis=1).max()*1000),
        maximum_postintegration_board_normal_error_deg=float(_angle_degrees(normals, normals[0]).max()),
        board_extrema_scope="Sampled FK: uniform240Hz over full recorded horizon plus every native state within10ms of first measured impact interval",
        maximum_integrated_joint_excursion_rad=float(np.abs(q-q[0]).max()),
        maximum_integrated_joint_velocity_rad_s=float(np.abs(v).max()),
        arm10_maximum_integrated_joint_excursion_rad=float(np.abs(q[:, :6]-q[0, :6]).max()),
        arm14_maximum_integrated_joint_excursion_rad=float(np.abs(q[:, 6:]-q[0, 6:]).max()),
        arm10_peak_joint_velocity_rad_s=float(np.abs(v[:, :6]).max()),
        arm14_peak_joint_velocity_rad_s=float(np.abs(v[:, 6:]).max()),
        pose_provenance="CPU MuJoCo FK reconstructed from recorded post-integration joint states; no live GPU forward call",
        contact_time_convention="force[k] describes interval[t[k],t[k+1]] using held pre-step endpoints; state[k+1] is integrated before next endpoint reset",
        settling_status="not_applicable_prescribed_endpoints_reset_each_native_interval",
        physical_release_status="not_tested_ball_spawned_free_flight",
        compliance_status="not_validated_endpoint_hold_discards_accumulated_arm_response",
    )
    if first >= 0:
        result["first_impact_interval"] = dict(index=first, start_s=float(times[first]), end_s=float(times[first+1]))
        # state[first] is the previous integration endpoint, before the current
        # interval's arm reset; the actual collision board is held_pre_step.
        result["impact_before"] = observation(first)
        result["impact_after"] = observation(first+1)
    if initial_row is not None:
        result["initial_fk_vs_native_board_error_mm"] = float(np.linalg.norm(
            centres[0]-np.asarray(initial_row["board_center_m"]))*1000)
        result["initial_fk_vs_native_normal_error_deg"] = float(_angle_degrees(
            normals[0], np.asarray(initial_row["board_normal"])))
        result["initial_ball_release_error_mm"] = float(np.linalg.norm(
            np.asarray(initial["ball_center_m"])-initial_row["release_ball_center_m"])*1000)
        if (result["initial_fk_vs_native_board_error_mm"] > .002
                or result["initial_fk_vs_native_normal_error_deg"] > .002):
            raise RuntimeError("Independent initial CPU FK disagrees with native board observation")
    return result


def verify_pose_arrays(model, arrays, rows=None):
    """Independent CPU replay of every exported FK sample from dense raw joints.

    Uses fresh MjData and mj_fwdPosition rather than the producer's mj_kinematics
    helper. This validates transform reconstruction and state/time association;
    it cannot independently validate the GPU integrator or physical compliance.
    """
    import mujoco

    ids, data = model_ids(model), mujoco.MjData(model)
    q, v, times = (np.asarray(arrays[k]) for k in
                   ("pose_joint_qpos_rad", "pose_joint_qvel_rad_s", "pose_time_s"))
    if q.shape != v.shape or q.ndim != 3 or q.shape[-1] != 12 or len(times) != q.shape[0]:
        raise ValueError("Invalid dense pose state shapes")
    if not all(np.isfinite(a).all() for a in (q, v, times)):
        raise ValueError("Non-finite dense pose state")
    if not np.array_equal(times, arrays["time_s"]):
        raise ValueError("Pose and ball clocks differ")
    if not np.array_equal(arrays["pose_held_joint_qpos_rad"], q[:1]):
        raise ValueError("Held endpoint differs from the recorded initial joints")
    checked, maximum = 0, 0.
    for world in range(q.shape[1]):
        expected, impact = sample_indices(times, arrays["forces_N"][:, world, 0])
        valid = arrays["pose_sample_valid"][:, world]
        indices = arrays["pose_sample_state_index"][valid, world]
        if (not np.array_equal(indices, expected) or np.any(arrays["pose_sample_state_index"][~valid, world] != -1)
                or int(arrays["pose_first_impact_interval"][world]) != impact
                or not np.array_equal(arrays["pose_sample_time_s"][valid, world], times[indices])):
            raise ValueError("FK selection, impact window or native index/time identity changed")
        for slot, index in zip(np.flatnonzero(valid), indices):
            data.qpos[:] = model.qpos0
            data.qpos[ids["q"]] = q[index, world]
            data.qvel[:] = 0
            data.qvel[ids["v"]] = v[index, world]
            mujoco.mj_fwdPosition(model, data)
            tests = dict(board_center_m=data.xpos[ids["board"]],
                         board_quaternion_wxyz=data.xquat[ids["board"]],
                         board_normal=data.xmat[ids["board"]].reshape(3, 3)[:, 2],
                         flange10_center_m=data.xpos[ids["flange"]],
                         flange10_quaternion_wxyz=data.xquat[ids["flange"]],
                         link_positions10_m=data.xpos[ids["links10"]],
                         link_positions14_m=data.xpos[ids["links14"]],
                         ball_center_m=arrays["position_m"][index, world],
                         ball_velocity_m_s=arrays["velocity_m_s"][index, world])
            for key, expected_value in tests.items():
                actual = arrays["pose_"+key][slot, world]
                if not np.isfinite(actual).all() or not np.allclose(actual, expected_value, atol=1e-10, rtol=0):
                    raise ValueError("CPU FK evidence mismatch: "+key)
                maximum = max(maximum, float(np.abs(actual-expected_value).max()))
            checked += 1
    summaries = [pose_summary(arrays, i, rows[i] if rows else None) for i in range(q.shape[1])]
    return dict(status="passed", checked_fk_samples=checked,
                maximum_numeric_difference=maximum, summaries=summaries,
                scope="Independent CPU FK of saved integrated joints; not independent GPU/physical dynamics")


class _PoseLog:
    def __init__(self, original, simulation, count, ticks, *args):
        import torch

        self.base = original(simulation, count, ticks, *args)
        self.sim, self.count = simulation, count
        self.ids = model_ids(simulation.mj_model)
        device = simulation.device if hasattr(simulation, "device") else "cuda:0"
        self.q = torch.empty((ticks+1, count, 12), device=device)
        self.v = torch.empty_like(self.q)
        self._state(0)

    def _state(self, index):
        self.q[index].copy_(self.sim.data.qpos[:self.count, self.ids["q"]])
        self.v[index].copy_(self.sim.data.qvel[:self.count, self.ids["v"]])

    def step(self, index, force):
        self.base.step(index, force)
        self._state(index+1)

    def arrays(self, count=None):
        values = self.base.arrays(count)
        count = self.count if count is None else count
        start = time.monotonic()
        values.update(reconstruct_pose_arrays(self.sim.mj_model,
            self.q[:, :count].cpu().numpy(), self.v[:, :count].cpu().numpy(),
            values["time_s"], values["forces_N"], values["position_m"], values["velocity_m_s"]))
        self.postprocess_s = time.monotonic()-start
        return values


@contextmanager
def pose_logger_scope(simulation):
    """Restore the frozen logger even on failure; isolated runner only.

    CalibratedEnvironment imports its logger inside run(), so substitution can
    instrument that call without copying or editing its physics loop. A process
    lock serializes instrumentation scopes. Any unexpected second simulation
    entering the scope is rejected, not silently instrumented.
    """
    from . import calibrated_native_log

    with _LOGGER_LOCK:
        original, captured = calibrated_native_log.DenseNativeLog, []

        def factory(current, *args, **kwargs):
            if current is not simulation or kwargs:
                raise RuntimeError("Pose logger used by an unexpected simulation/call")
            log = _PoseLog(original, current, *args)
            captured.append(log)
            return log

        calibrated_native_log.DenseNativeLog = factory
        try:
            yield captured
        finally:
            calibrated_native_log.DenseNativeLog = original


class ImpactPoseEnvironment(CalibratedEnvironment):
    """Same prescribed-endpoint simulation, with additive actual-joint logging."""
    def run(self, bucket, values, locks, *, record=False):
        with pose_logger_scope(self.sim) as captured:
            rows, metadata = super().run(bucket, values, locks, record=record)
        if len(captured) != 1 or self.last_arrays is None:
            raise RuntimeError("Native pose logger did not record exactly one complete batch")
        for world, row in enumerate(rows):
            row["pose_logging"] = pose_summary(self.last_arrays, world, row)
        metadata.update(pose_logging_version=VERSION,
                        pose_fk_postprocess_s=captured[0].postprocess_s,
                        pose_scope="New instrumentation; frozen calibrated physics/scorer unchanged")
        return rows, metadata
