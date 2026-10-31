"""Fixed-layout, single-decision cooperative RL contract; no simulator imports."""

import hashlib
import json
import math

import numpy as np

VERSION = "sircus-isaac-fixed-shot-v4"
# Tilt from horizontal, heading offset from bucket-facing, vertical ball clearance.
LOW = np.array([12.0, -15.0, 0.06])
HIGH = np.array([45.0, 15.0, 0.25])
INITIAL_MEAN = np.array([-0.6, 0.5, -0.7])


def action_values(latent):
    return LOW + (np.tanh(latent) + 1) * 0.5 * (HIGH - LOW)


def fingerprint(cfg):
    return hashlib.sha256(
        json.dumps(
            {
                "scene": cfg,
                "task": VERSION,
                "low": LOW.tolist(),
                "high": HIGH.tolist(),
                "ball_restitution_combine": "max",
                "retention_s": 1.0,
                "floor_contact_tolerance_m": 0.001,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()


class ShotScore:
    def __init__(self, bucket, radius):
        self.b = bucket
        self.r = radius
        self.bounced = False
        self.retention = 0.0
        self.max_retention = 0.0
        self.success = False
        self.closest = 2.0
        self.entered = False
        self.was_above = False
        self.contact_seen = False
        self.since_board_contact = float("inf")
        angles = (
            (np.arange(bucket["segments"]) + 0.5) * 2 * math.pi / bucket["segments"]
        )
        self.wall_normals = np.column_stack([np.cos(angles), np.sin(angles)])

    def step(self, pos, board_force, dt, rebound_normal_speed=0.0):
        x, y, z = pos
        b, r = self.b, self.r
        floor = b["assumed_bottom_thickness_m"]
        top = floor + b["assumed_interior_depth_m"]
        rho = math.hypot(x - b["center_xy_m"][0], y - b["center_xy_m"][1])
        slope = (
            (b["inner_top_diameter_m"] - b["inner_bottom_diameter_m"])
            / 2
            / b["assumed_interior_depth_m"]
        )
        inner = b["inner_bottom_diameter_m"] / 2 + slope * (z - floor)
        # Match all 64 actual wall planes, rather than an inscribed circular proxy.
        inner *= math.cos(math.pi / b["segments"])
        slope *= math.cos(math.pi / b["segments"])
        projection = float(
            np.max(self.wall_normals @ (np.array([x, y]) - b["center_xy_m"]))
        )
        whole = (
            z + r <= top
            and z - r >= floor - 0.001
            and (inner - projection) / math.sqrt(1 + slope * slope) >= r
        )
        self.contact_seen |= board_force > 0.01
        self.since_board_contact = (
            0.0 if board_force > 0.01 else self.since_board_contact + dt
        )
        self.bounced |= (
            self.contact_seen
            and self.since_board_contact <= 0.05
            and board_force <= 0.01
            and rebound_normal_speed > 0.05
        )
        if self.bounced:
            self.closest = min(self.closest, math.hypot(rho, z - (top - r)))
            self.was_above |= z + r > top
        if self.bounced and self.was_above and whole:
            self.entered = True
            self.retention += dt
        else:
            self.retention = 0.0
        self.max_retention = max(self.max_retention, self.retention)
        self.success |= self.retention >= 1.0 - 1e-9

    def result(self):
        reward = (
            float(self.bounced)
            + 2 * math.exp(-self.closest / 0.12)
            + 2 * min(self.max_retention, 1.0)
            + 10 * float(self.success)
        )
        return {
            "reward": reward,
            "success": bool(self.success),
            "board_bounce": bool(self.bounced),
            "whole_ball_entry": bool(self.entered),
            "max_retention_s": self.max_retention,
            "closest_opening_distance_m": self.closest,
        }
