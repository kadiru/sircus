"""Additional entry-quality label; never changes the historical capture scorer."""

import math

import numpy as np


class EntryQuality:
    def __init__(self, bucket, radius):
        self.b, self.r = bucket, radius
        a = (np.arange(bucket["segments"]) + 0.5) * 2 * math.pi / bucket["segments"]
        self.normals = np.column_stack([np.cos(a), np.sin(a)])
        self.contact = False
        self.peak = 0.0
        self.margin = float("inf")
        self.entry = False
        self.saw_mouth = False
        self.rim_band_contact = False
        self.exited_after_entry = False

    def step(self, pos, bucket_force, entered):
        if not np.isfinite(bucket_force):
            raise ValueError("Invalid bucket contact observation")
        top = self.b["assumed_interior_depth_m"] + self.b["assumed_bottom_thickness_m"]
        self.rim_band_contact |= bucket_force > 0.01 and pos[2] + self.r >= top - 0.002
        if self.entry:
            self.exited_after_entry |= pos[2] + self.r > top + 1e-6
            return
        self.peak = max(self.peak, float(bucket_force))
        self.contact |= bucket_force > 0.01
        dz = float(pos[2] - top)
        if abs(dz) <= self.r:
            self.saw_mouth = True
            slice_radius = math.sqrt(max(0, self.r**2 - dz**2))
            projection = float(
                np.max(self.normals @ (np.array(pos[:2]) - self.b["center_xy_m"]))
            )
            clearance = (
                self.b["inner_top_diameter_m"]
                / 2
                * math.cos(math.pi / self.b["segments"])
                - projection
                - slice_radius
            )
            self.margin = min(self.margin, clearance)
        self.entry = bool(entered)

    def result(self, captured):
        clean = bool(
            captured
            and self.entry
            and self.saw_mouth
            and not self.contact
            and not self.rim_band_contact
            and not self.exited_after_entry
            and self.margin >= 0.001
        )
        return {
            "quality_score": 2 if clean else 1 if captured else 0,
            "entry_class": "clean_capture"
            if clean
            else "assisted_or_marginal_capture"
            if captured
            else "miss",
            "bucket_contact_before_entry": bool(self.contact),
            "rim_band_contact": bool(self.rim_band_contact),
            "exited_after_entry": bool(self.exited_after_entry),
            "peak_bucket_force_before_entry_N": self.peak,
            "minimum_mouth_clearance_mm": self.margin * 1000
            if self.saw_mouth
            else None,
            "quality_scope": "240 Hz bucket contact and polygon-mouth clearance; clean requires >=1mm entry margin, no pre-entry/rim-band contact, no subsequent partial exit; historical capture unchanged",
        }
