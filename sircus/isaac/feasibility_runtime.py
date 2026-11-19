"""Opt-in variable board/release XY; historical parallel and PPO tasks unchanged."""

import copy

import numpy as np
from feasibility_contract import dynamic_cell
from parallel_runtime import ParallelShots
from prepare_assets import matrix


class FeasibilityShots(ParallelShots):
    def __init__(self, *args, **kwargs):
        self.probe_positions = {
            "table": [1.50, -0.60, 0.25],
            "floor": [1.8, -0.65, -0.5],
            "robot": [0, 0, 0.75],
        }
        super().__init__(*args, **kwargs)
        self.base_cells = copy.deepcopy(self.spec["cells"])

    def prepare(self, actions):
        self.spec["cells"] = [
            dynamic_cell(c, a) for c, a in zip(self.base_cells, actions)
        ]
        releases, normals, records, invalid = super().prepare(actions)
        self.release_centers, self.release_normals = self.board_poses()
        for i, c in enumerate(self.spec["cells"]):
            if c["distance_m"] < 0.28:
                invalid[i] = "board_bucket_separation"
        return releases, normals, records, invalid

    def shot_batch(self, actions, record_video=False, enabled=None):
        rows, raw = super().shot_batch(actions, record_video, enabled)
        centers, normals = self.board_poses()
        for i, row in enumerate(rows):
            c = self.spec["cells"][i]
            row.update(
                bucket_xy_m=list(c["bucket_xy_m"]),
                distance_m=c["distance_m"],
                heading_deg=c["heading_deg"],
                settled_board_center_m=self.release_centers[i].tolist(),
                settled_board_normal=self.release_normals[i].tolist(),
                final_board_center_m=centers[i].tolist(),
                final_board_normal=normals[i].tolist(),
                final_joint_positions_rad={
                    k: v.data.joint_pos[i].cpu().numpy().tolist()
                    for k, v in self.arms.items()
                },
            )
        return rows, raw

    def render(self):
        if self.camera is not None:
            super().render()

    def direct_drop_audit(self, heights):
        pivot = self.links[0][0][:3, 3]
        reach_bound = sum(np.linalg.norm(link[0][:3, 3]) for link in self.links[1:])
        result = []
        for cell in self.base_cells:
            xy = cell["bucket_xy_m"]
            horizontal = float(np.linalg.norm(np.array(xy) - pivot[:2]))
            # This is an outer bound from link lengths, not an IK-search failure.
            beyond = horizontal > reach_bound + 0.095
            trials = []
            for height in heights:
                target = matrix([*xy, height + 0.095], [np.pi, 0, 0])
                if np.linalg.norm(target[:3, 3] - pivot) > reach_bound:
                    trials.append(
                        {
                            "ball_center_z_m": height,
                            "solved": False,
                            "reason": "outside_link_length_bound",
                        }
                    )
                else:
                    fit = self._ik(target, "Arm10")
                    trials.append(
                        {
                            "ball_center_z_m": height,
                            **{
                                k: fit[k]
                                for k in (
                                    "solved",
                                    "position_error_mm",
                                    "orientation_error_deg",
                                    "stage",
                                )
                            },
                        }
                    )
            result.append(
                {
                    "cell_id": cell["id"],
                    "horizontal_distance_m": horizontal,
                    "flange_outer_bound_m": float(reach_bound),
                    "ball_outer_bound_m": float(reach_bound + 0.095),
                    "beyond_geometric_outer_bound": bool(beyond),
                    "tested_heights": trials,
                    "scope": "Endpoint IK for downward tool at three ball heights; no path certification. Only the link-length outer bound establishes unreachable XY in this model.",
                }
            )
        return result
