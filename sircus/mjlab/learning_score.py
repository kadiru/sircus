"""Batched Torch transcription of the unchanged 240 Hz shot scoring contract.

This is a throughput adapter, not a new success definition.  ``rows`` is checked
against ``ScoreStream`` on complete streams in test_learning_score.py.  Double
precision counters deliberately preserve the original one-second boundary.
"""

import math

import torch

from .contract import FORCE_NAMES, ROUTE_VERSION, SCORER_VERSION


class BatchScore:
    """Score independent worlds without per-tick device-to-host readback.

    ``forces`` columns follow ``contract.FORCE_NAMES``.  Finished worlds freeze
    at exactly the same capture/fall boundary as ScoreStream.  The simulation
    may continue stepping these worlds; none of those later samples is scored.
    """

    def __init__(self, scene, bucket_xy_m, normals, device=None):
        if scene["simulation"]["physics_dt_s"] != 1 / 240:
            raise ValueError("Scoring requires the original 240 Hz cadence")
        self.xy = torch.as_tensor(bucket_xy_m, dtype=torch.float64, device=device)
        self.normal = torch.as_tensor(normals, dtype=torch.float64, device=device)
        if (
            self.xy.ndim != 2
            or self.xy.shape[1] != 2
            or self.normal.shape != (len(self.xy), 3)
            or not bool(torch.isfinite(self.xy).all())
            or not bool(torch.isfinite(self.normal).all())
            or not bool(
                torch.allclose(
                    torch.linalg.vector_norm(self.normal, dim=1),
                    torch.ones(len(self.xy), dtype=torch.float64, device=device),
                    atol=1e-5,
                    rtol=0,
                )
            )
        ):
            raise ValueError("Expected finite bucket XYs and unit board normals")
        self.device = self.xy.device
        self.n = len(self.xy)
        self.dt = 1 / 240
        self.bucket = scene["bucket"]
        self.r = scene["ball"]["radius_m"]
        self.floor = self.bucket["assumed_bottom_thickness_m"]
        self.top = self.floor + self.bucket["assumed_interior_depth_m"]
        segments = self.bucket["segments"]
        self.cosine = math.cos(math.pi / segments)
        a = torch.arange(segments, dtype=torch.float64, device=self.device) + 0.5
        a *= 2 * math.pi / segments
        self.planes = torch.stack((a.cos(), a.sin()), dim=1)
        self.slope = (
            (
                self.bucket["inner_top_diameter_m"]
                - self.bucket["inner_bottom_diameter_m"]
            )
            / 2
            / self.bucket["assumed_interior_depth_m"]
        )
        self.retention = self._float()
        self.max_retention = self._float()
        self.closest = self._float(2)
        self.since_contact = self._float(float("inf"))
        self.quiet = self._float(0.05)
        self.margin = self._float(float("inf"))
        self.bucket_peak = self._float()
        self.peaks = torch.zeros((self.n, 6), dtype=torch.float64, device=self.device)
        for name in (
            "bounced",
            "success",
            "entered",
            "was_above",
            "contact_seen",
            "entry",
            "saw_mouth",
            "rim",
            "exited",
            "bucket_contact",
            "finished",
        ):
            setattr(self, name, self._bool())
        self.episodes = torch.zeros(self.n, dtype=torch.int64, device=self.device)
        self.external = torch.zeros_like(self.episodes)
        self.steps = torch.zeros_like(self.episodes)

    def _float(self, fill=0):
        return torch.full((self.n,), fill, dtype=torch.float64, device=self.device)

    def _bool(self):
        return torch.zeros(self.n, dtype=torch.bool, device=self.device)

    def step(self, position, velocity, forces):
        """Consume one outer 240 Hz sample; forces are max norms over substeps."""
        p = torch.as_tensor(position, dtype=torch.float64, device=self.device)
        v = torch.as_tensor(velocity, dtype=torch.float64, device=self.device)
        if isinstance(forces, dict):
            if set(forces) != set(FORCE_NAMES):
                raise ValueError("Contact attribution requires every force category")
            forces = torch.stack([forces[k] for k in FORCE_NAMES], dim=1)
        f = torch.as_tensor(forces, dtype=torch.float64, device=self.device)
        if p.shape != (self.n, 3) or v.shape != p.shape or f.shape != (self.n, 6):
            raise ValueError("Wrong batched scoring dimensions")
        # Avoid synchronizing GPU every tick.  The CUDA async assertion raises
        # on corrupt simulator data; CPU calls retain immediate ValueErrors.
        active = ~self.finished
        valid_rows = torch.isfinite(p).all(dim=1) & torch.isfinite(v).all(dim=1)
        valid_rows &= torch.isfinite(f).all(dim=1) & (f >= 0).all(dim=1)
        valid = (valid_rows | ~active).all()
        if self.device.type == "cuda":
            torch._assert_async(valid, "Invalid batched score observation")
        elif not bool(valid):
            raise ValueError("Invalid batched score observation")
        z = p[:, 2]
        delta = p[:, :2] - self.xy
        projection = (delta @ self.planes.T).amax(dim=1)
        inner = (
            self.bucket["inner_bottom_diameter_m"] / 2 + self.slope * (z - self.floor)
        ) * self.cosine
        whole = (
            (z + self.r <= self.top + 1e-6)
            & (z - self.r >= self.floor - 0.001)
            & (
                (inner - projection) / math.sqrt(1 + (self.slope * self.cosine) ** 2)
                >= self.r - 1e-6
            )
        )
        board_contact = f[:, 0] > 0.01
        self.contact_seen |= active & board_contact
        self.since_contact = torch.where(
            active,
            torch.where(board_contact, 0, self.since_contact + self.dt),
            self.since_contact,
        )
        self.bounced |= (
            active
            & self.contact_seen
            & (self.since_contact <= 0.05)
            & ~board_contact
            & ((v * self.normal).sum(dim=1) > 0.05)
        )
        distance = torch.sqrt((delta**2).sum(dim=1) + (z - (self.top - self.r)) ** 2)
        self.closest = torch.where(
            active & self.bounced, torch.minimum(self.closest, distance), self.closest
        )
        self.was_above |= active & self.bounced & (z + self.r > self.top)
        inside = self.bounced & self.was_above & whole
        self.entered |= active & inside
        self.retention = torch.where(
            active, torch.where(inside, self.retention + self.dt, 0), self.retention
        )
        self.max_retention = torch.maximum(self.max_retention, self.retention)
        self.success |= active & (self.retention >= 1 - 1e-9)

        bucket_contact = f[:, 1] > 0.01
        self.rim |= active & bucket_contact & (z + self.r >= self.top - 0.002)
        self.exited |= active & self.entry & (z + self.r > self.top + 1e-6)
        before = active & ~self.entry
        self.bucket_peak = torch.where(
            before, torch.maximum(self.bucket_peak, f[:, 1]), self.bucket_peak
        )
        self.bucket_contact |= before & bucket_contact
        mouth = before & ((z - self.top).abs() <= self.r)
        self.saw_mouth |= mouth
        slice_radius = (self.r**2 - (z - self.top) ** 2).clamp_min(0).sqrt()
        clearance = (
            self.bucket["inner_top_diameter_m"] / 2 * self.cosine
            - projection
            - slice_radius
        )
        self.margin = torch.where(
            mouth, torch.minimum(self.margin, clearance), self.margin
        )
        self.entry |= before & self.entered

        self.episodes += (active & board_contact & (self.quiet >= 0.05 - 1e-9)).long()
        self.quiet = torch.where(
            active, torch.where(board_contact, 0, self.quiet + self.dt), self.quiet
        )
        self.external += (active & (f[:, 2:] > 0.01).any(dim=1)).long()
        self.peaks = torch.where(
            active[:, None], torch.maximum(self.peaks, f), self.peaks
        )
        self.steps += active.long()
        self.finished |= active & (self.success | (z < -0.2))

    def tensors(self):
        """Device tensors, including historical reward and additive route labels."""
        clean = (
            self.success
            & self.entry
            & self.saw_mouth
            & ~self.bucket_contact
            & ~self.rim
            & ~self.exited
            & (self.margin >= 0.001)
        )
        quality = torch.where(clean, 2, self.success.long())
        route = self.success & (self.episodes >= 1) & (self.external == 0)
        return dict(
            reward=(
                self.bounced.double()
                + 2 * torch.exp(-self.closest / 0.12)
                + 2 * self.max_retention.clamp_max(1)
                + 10 * self.success.double()
            ),
            success=self.success,
            board_bounce=self.bounced,
            whole_ball_entry=self.entered,
            max_retention_s=self.max_retention,
            closest_opening_distance_m=self.closest,
            quality_score=quality,
            bucket_contact_before_entry=self.bucket_contact,
            rim_band_contact=self.rim,
            exited_after_entry=self.exited,
            peak_bucket_force_before_entry_N=self.bucket_peak,
            minimum_mouth_clearance_mm=self.margin * 1000,
            route_success=route,
            route_quality=torch.where(route, quality, 0),
            single_bounce_success=route & (self.episodes == 1),
            board_contact_episodes=self.episodes,
            external_contact_steps=self.external,
            complete_success=route,
            physics_steps=self.steps,
        )

    def rows(self):
        """Export ScoreStream-compatible JSON records, synchronizing only here."""
        values = {k: v.detach().cpu().tolist() for k, v in self.tensors().items()}
        peaks = self.peaks.detach().cpu().tolist()
        saw_mouth = self.saw_mouth.detach().cpu().tolist()
        rows = []
        for i in range(self.n):
            row = {k: v[i] for k, v in values.items()}
            row["entry_class"] = (
                "clean_capture"
                if row["quality_score"] == 2
                else (
                    "assisted_or_marginal_capture"
                    if row["quality_score"] == 1
                    else "miss"
                )
            )
            if not saw_mouth[i]:
                row["minimum_mouth_clearance_mm"] = None
            row.update(
                quality_scope="240 Hz bucket contact and polygon-mouth clearance; clean requires >=1mm entry margin, no pre-entry/rim-band contact, no subsequent partial exit; historical capture unchanged",
                scoring_version=SCORER_VERSION,
                route_version=ROUTE_VERSION,
                contact_peaks_N=dict(zip(FORCE_NAMES, peaks[i])),
            )
            rows.append(row)
        return rows
