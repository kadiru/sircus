"""Additive contact-based board-only route definition. Historical capture is unchanged."""

import math

VERSION = "sircus-board-route-v1"
FORCE_THRESHOLD_N = 0.01
CONTACT_FREE_S = 0.05


class BoardRoute:
    def __init__(self, dt):
        self.dt = dt
        self.quiet_s = CONTACT_FREE_S
        self.board_episodes = 0
        self.external_steps = 0
        self.peaks = {}

    def step(self, forces):
        if not forces or any(not math.isfinite(v) or v < 0 for v in forces.values()):
            raise ValueError("Invalid contact observation")
        if forces["board"] > FORCE_THRESHOLD_N:
            if self.quiet_s >= CONTACT_FREE_S - 1e-9:
                self.board_episodes += 1
            self.quiet_s = 0
        else:
            self.quiet_s += self.dt
        external = False
        for name, value in forces.items():
            self.peaks[name] = max(self.peaks.get(name, 0), value)
            if name not in ("board", "bucket"):
                external |= value > FORCE_THRESHOLD_N
        self.external_steps += int(external)

    def result(self, success, quality):
        route = bool(success and self.board_episodes >= 1 and not self.external_steps)
        return {
            "route_version": VERSION,
            "route_success": route,
            "route_quality": quality if route else 0,
            "single_bounce_success": bool(route and self.board_episodes == 1),
            "board_contact_episodes": self.board_episodes,
            "external_contact_steps": self.external_steps,
            "contact_peaks_N": self.peaks,
        }
