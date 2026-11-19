"""Simulation-only endpoint controller with measured, gated board settling."""

import numpy as np
from generalized_rl import SceneShot
from prepare_assets import matrix
from robust_ik import solve
from scipy.spatial.transform import Rotation
from settling_contract import MAX_CORRECTIONS, accepted, correction, pose_errors


class SettledShot(SceneShot):
    def tick(self, render=False):
        # Supported PhysX model-based feed-forward; gravity stays enabled.
        # Only this versioned simulated controller uses these torques.
        for arm in self.arms.values():
            arm.set_joint_effort_target(
                arm.root_physx_view.get_gravity_compensation_forces()
            )
        super().tick(render=render)

    def ik(self, target, name):
        key = "board_joint_rad" if name == "Arm14" else "dropper_joint_rad"
        r = solve(target, self.links, self.poses[key], name == "Arm14")
        if not r["solved"]:
            self.ik_failure_component = name
            raise ValueError("Endpoint IK unresolved at original acceptance tolerance")
        return np.array(r["q"])

    def before_release(self, target):
        arm = self.arms["Arm14"]
        idx = arm.body_names.index("impact_board")
        command = target.copy()
        base = matrix([self.cfg["arms"]["base_spacing_m"], 0, 0], [0, 0, np.pi])
        mount = matrix(
            [0, 0, self.cfg["board"]["tool_offset_m"]],
            [0, 0, self.cfg["board"]["mount_yaw_rad"]],
        )
        records = []
        for iteration in range(MAX_CORRECTIONS + 1):
            previous = arm.data.body_pos_w[0, idx].cpu().numpy().copy()
            previous_q = arm.data.body_quat_w[0, idx].cpu().numpy()
            previous_normal = Rotation.from_quat(previous_q[[1, 2, 3, 0]]).as_matrix()[
                :, 2
            ]
            for _ in range(60):
                self.tick()
            center = arm.data.body_pos_w[0, idx].cpu().numpy().copy()
            q = arm.data.body_quat_w[0, idx].cpu().numpy()
            normal = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()[:, 2]
            speed = max(
                float(a.data.joint_vel[0].abs().max()) for a in self.arms.values()
            )
            measured = dict(
                pose_errors(target, center, normal),
                joint_speed_rad_s=speed,
                center_m=center.tolist(),
                normal_world=normal.tolist(),
                iteration=iteration,
                position_drift_mm=float(np.linalg.norm(center - previous) * 1000),
                normal_drift_deg=float(
                    np.rad2deg(np.arccos(np.clip(normal @ previous_normal, -1, 1)))
                ),
                command_center_m=command[:3, 3].tolist(),
                target_joints=self.q["Arm14"].tolist(),
                measured_joints=arm.data.joint_pos[0, :6].cpu().tolist(),
            )
            records.append(measured)
            if accepted(measured):
                return {
                    "settled_board": records,
                    "release_normal_world": normal.tolist(),
                }
            if iteration == MAX_CORRECTIONS:
                break
            # Feedback changes position targets only; no teleport/reset after the first endpoint initialization.
            command = correction(command, target, center, normal)
            endpoint = np.linalg.inv(base) @ command @ np.linalg.inv(mount)
            r = solve(endpoint, self.links, self.q["Arm14"], True)
            if not r["solved"]:
                break
            self.q["Arm14"] = np.array(r["q"])
            joints = arm.data.joint_pos_target.clone()
            for i in range(6):
                joints[0, arm.joint_names.index(f"joint{i + 1}")] = r["q"][i]
            arm.set_joint_position_target(joints)
            for _ in range(120):
                self.tick()
        return {"invalid": "board_not_settled", "settled_board": records}
