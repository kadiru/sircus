"""V2 prescribed-endpoint physics with dense native observations.

The original environment/scorer remains untouched. Each requested endpoint must
already have a strict geometric setup proof; no actions are clipped or replaced.
Labels are computed after the native rollout and checked by a separate oracle.
"""

from contextlib import contextmanager
import time

import numpy as np

from .contact_env import ContactEnvironment
from .corrected_contract import PHYSICS


VERSION = "sircus-calibrated-held-endpoint-native-v2"


@contextmanager
def native_reset_scope(sim):
    """Order the Torch reset mask and all Warp work on one native stream."""
    import torch
    import warp as wp

    stream = torch.cuda.ExternalStream(wp.get_stream(sim.wp_device).cuda_stream)
    with torch.cuda.stream(stream):
        sim.reset()
        yield


class CalibratedEnvironment(ContactEnvironment):
    def __init__(self, scene, assets, calibration, n=32, bounds=None):
        from .calibrated_score import validate_calibration
        from .capture_calibration import compiled_geometry

        self.calibration = validate_calibration(calibration)
        super().__init__(scene, assets, n=n, bounds=bounds, substeps=64,
                         board_mode=PHYSICS["board_mode"], gravity_mode=PHYSICS["gravity_mode"])
        self.compiled_geometry = compiled_geometry(self.sim.mj_model, scene)
        if not np.allclose(self.compiled_geometry["compiled_planes"], self.calibration["compiled_planes"],
                           rtol=0, atol=self.calibration["geometry_epsilon_m"]):
            raise ValueError("Runtime bucket planes differ from calibrated collision geometry")
        self.native_geom_ids = self.compiled_geometry["contact_geom_ids"]

    def run(self, bucket, values, locks, *, record=False):
        import torch
        from .calibrated_native_log import DenseNativeLog
        from .calibrated_score import score_trace
        from .calibrated_score_oracle import score_trace as oracle_score_trace

        if not record:
            raise ValueError("Calibrated native releases require dense audit observations")
        started = time.monotonic()
        bucket, values = np.asarray(bucket, dtype=float), np.asarray(values, dtype=float)
        count = len(bucket)
        if bucket.shape != (count, 2) or values.shape != (count, 5) or not 0 < count <= self.n:
            raise ValueError("Invalid calibrated native batch")
        if not np.isfinite(bucket).all() or not np.isfinite(values).all():
            raise ValueError("Non-finite native action")
        projected, allowed = self.project(values, locks, self.scene["ball"]["radius_m"])
        if not np.asarray(allowed).all() or not np.allclose(projected, values, rtol=0, atol=1e-12):
            raise ValueError("Native actions must pass their original bounds without projection")
        if np.any(np.linalg.norm(bucket-values[:, :2], axis=1) <= .15):
            raise ValueError("Native board/bucket minimum-distance bound")
        pad = lambda a: np.concatenate([a, np.repeat(a[-1:], self.n-count, axis=0)])
        bucket_p, values_p = pad(bucket), pad(values)
        ik = self.ik.solve(bucket_p, values_p[:, :3], values_p[:, 3], values_p[:, 4],
                           initial10=self.seeds["dropper_joint_rad"], initial14=self.seeds["board_joint_rad"], starts=4, iterations=90)
        q = np.tile(self.m.qpos0, (self.n, 1))
        q[:, self.q10] = ik["q10"].cpu().numpy()
        q[:, self.q14] = ik["q14"].cpu().numpy()
        q[:, self.fingers] = .025
        q[:, self.ball_q:self.ball_q+3] = ik["targets"]["release_center_m"]
        q[:, self.ball_q+3:self.ball_q+7] = [1, 0, 0, 0]
        collision = self._collision_gate(q[:count], bucket_p[:count])
        valid = ik["valid"][:count].cpu().numpy() & np.asarray([not pairs for pairs in collision])
        if not valid.all():
            raise RuntimeError("Strict admission disagrees with actual native endpoint geometry")
        sim = self.sim
        with native_reset_scope(sim):
            targets = torch.tensor(np.column_stack([bucket_p, np.zeros(self.n)]), device=self.device, dtype=torch.float32)
            sim.data.mocap_pos[:, self.mocap, :] = targets
            sim.data.mocap_quat[:, self.mocap, :] = torch.tensor([1, 0, 0, 0], device=self.device)
            qt = torch.tensor(q, device=self.device, dtype=torch.float32)
            sim.data.qpos[:] = qt
            sim.data.qvel[:] = 0
            sim.data.ctrl[:] = qt[:, self.actuator_q]
            if self.board_mode == "kinematic":
                bp, bq = self._board_poses(q)
                sim.data.mocap_pos[:, self.board_mocap, :] = torch.tensor(bp, device=self.device, dtype=torch.float32)
                sim.data.mocap_quat[:, self.board_mocap, :] = torch.tensor(bq, device=self.device, dtype=torch.float32)
            sim.forward()
            native_initial_q = sim.data.qpos[:, :self.ball_q].cpu().numpy().copy()
            expected = targets[:, None, :] + torch.tensor(self.m.geom_pos[self.bucket_geoms], device=self.device, dtype=torch.float32)
            collider_error = float(torch.linalg.vector_norm(sim.data.geom_xpos[:, self.bucket_geoms]-expected, dim=-1).max())
            if collider_error > 1e-5:
                raise RuntimeError("Native bucket collision transforms disagree with query")
            centers = sim.data.xpos[:, self.board, :].clone()
            bucket_centers = sim.data.xpos[:, self.bucket, :].clone()
            normals = sim.data.xmat[:, self.board, :].reshape(self.n, 3, 3)[:, :, 2].clone()
            quaternions = sim.data.xquat[:, self.board, :].cpu().tolist()
            links = {role: sim.data.xpos[:, ids, :].cpu().tolist() for role, ids in self.links.items()}
            poserr = torch.linalg.vector_norm(centers-torch.tensor(values_p[:, :3], device=self.device), dim=-1).cpu().numpy()
            normerr = np.rad2deg(np.arccos(np.clip((normals.cpu().numpy()*ik["targets"]["board_normal_world"]).sum(1), -1, 1)))
            droperr = torch.linalg.vector_norm(sim.data.xpos[:, self.flange]-torch.tensor(ik["targets"]["release_center_m"]+[0, 0, .095], device=self.device), dim=-1).cpu().numpy()
            if not ((poserr[:count] < .002) & (normerr[:count] < .5) & (droperr[:count] < .002)).all():
                raise RuntimeError("Native held poses disagree with the admitted shot")
            for sensor in self.sensors:
                sensor.reset()
            tags = np.zeros(self.m.ngeom, dtype=np.int32)
            tags[self.native_geom_ids["walls"]] = 1
            tags[self.native_geom_ids["base"]] = 2
            ticks = 960*self.substeps
            log = DenseNativeLog(sim, count, ticks, self.ball_q, self.ball_v, self.native_geom_ids["ball"], tags)
            drift = torch.zeros(self.n, device=self.device)
            excursion = torch.zeros(self.n, device=self.device)
            peak_joint_velocity = torch.zeros(self.n, device=self.device)
            native_start = time.monotonic()
            for tick in range(ticks):
                sim.data.qpos[:, :self.ball_q] = qt[:, :self.ball_q]
                sim.data.qvel[:, :self.ball_v] = 0
                sim.data.qfrc_applied[:, :self.ball_v] = 0
                sim.step()
                excursion = torch.maximum(excursion, (sim.data.qpos[:, self.arm_q]-qt[:, self.arm_q]).abs().amax(1))
                peak_joint_velocity = torch.maximum(peak_joint_velocity, sim.data.qvel[:, self.arm_v].abs().amax(1))
                for sensor in self.sensors:
                    sensor.update(1/(240*self.substeps))
                vectors = [sensor.data.force for sensor in self.sensors]
                norms = [torch.linalg.vector_norm(v, dim=-1).sum(1) for v in vectors[:5]]
                residual = torch.linalg.vector_norm(vectors[-1].sum(1)+sum(v.sum(1) for v in vectors[:5]), dim=-1)
                force = torch.stack([*norms, residual], -1)
                log.step(tick, force)
                if (tick+1) % self.substeps == 0:
                    drift = torch.maximum(drift, torch.linalg.vector_norm(sim.data.xpos[:, self.board]-centers, dim=-1))
            torch.cuda.synchronize()
            native_s = time.monotonic()-native_start
            transfer_start = time.monotonic()
            arrays = log.arrays()
            transfer_s = time.monotonic()-transfer_start
        self.last_arrays = arrays
        score_start = time.monotonic()
        center_rows, normal_rows = centers[:count].cpu().numpy(), normals[:count].cpu().numpy()
        rows = []
        keys = ("success", "route_success", "complete_success", "route_quality", "board_bounce", "whole_ball_entry",
                "physics_steps", "external_contact_steps", "board_contact_episodes", "invalid_physics")
        for i in range(count):
            trace = {key: (value[:, i] if value.ndim >= 2 and key not in ("time_s",) else value)
                     for key, value in arrays.items()}
            primary = score_trace(self.scene, bucket[i], normal_rows[i], self.calibration, trace)
            independent = oracle_score_trace(self.scene, bucket[i], normal_rows[i], self.calibration, trace)
            for key in keys:
                if primary[key] != independent[key]:
                    raise RuntimeError("Independent calibrated native scorer disagreement: "+key)
            if abs(primary["max_retention_s"]-independent["max_retention_s"]) > 1e-9:
                raise RuntimeError("Independent retention-clock disagreement")
            row = dict(primary)
            row.update(
                native_release=True, scalar_score_audit_passed=True,
                invalid="native_physics_outside_calibrated_domain" if primary["invalid_physics"] else None,
                bucket_xy_m=bucket[i].tolist(), values=values[i].tolist(),
                board_center_m=center_rows[i].tolist(), board_normal=normal_rows[i].tolist(),
                release_ball_center_m=ik["targets"]["release_center_m"][i].tolist(),
                native_initial_ball_center_m=arrays["position_m"][0, i].tolist(),
                native_initial_bucket_center_m=bucket_centers[i].cpu().tolist(),
                bucket_command_center_m=targets[i].cpu().tolist(),
                q10=q[i, self.q10].tolist(), q14=q[i, self.q14].tolist(),
                native_initial_q10=native_initial_q[i, self.q10].tolist(),
                native_initial_q14=native_initial_q[i, self.q14].tolist(),
                pose_gate=dict(board_position_error_mm=float(poserr[i]*1000), board_normal_error_deg=float(normerr[i]), dropper_error_mm=float(droperr[i]*1000)),
                endpoint_collision_pairs=collision[i], max_board_drift_mm=float(drift[i]*1000), bucket_collider_error_m=collider_error,
                max_integrated_joint_excursion_rad=float(excursion[i]), peak_integrated_joint_velocity_rad_s=float(peak_joint_velocity[i]),
                contact_protocol=dict(substeps=self.substeps, board_mode=self.board_mode, gravity_mode=self.gravity_mode),
                board_quaternion_wxyz=quaternions[i],
                link_positions10=[[0, 0, 0], *links[10][i]],
                link_positions14=[[self.scene["arms"]["base_spacing_m"], 0, 0], *links[14][i]],
                trajectory_m=arrays["position_m"][self.substeps*4::self.substeps*4, i].tolist(), trajectory_dt_s=1/60,
                calibrated_geometry_sha256=self.compiled_geometry["geometry_sha256"],
                native_clock_drift_s=float(np.max(np.abs(arrays["native_time_s"][:, i]-arrays["time_s"]))),
                time_provenance="canonical tick*verified native dt; native accumulated clock separately recorded; contact distance/position/normal use pre-integration p[k] for interval[k,k+1]; position[k+1] is post-integration",
            )
            rows.append(row)
        return rows, dict(seconds=time.monotonic()-started, native_rollout_s=native_s, raw_transfer_s=transfer_s,
                          score_and_oracle_s=time.monotonic()-score_start, batch_worlds=self.n, actual_shots=count)
