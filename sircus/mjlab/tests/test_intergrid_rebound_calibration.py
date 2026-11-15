"""Exercise boundary measurements and contact isolation on known traces."""

import unittest

import numpy as np

from sircus.mjlab.calibrated_score import FORCE_NAMES
from sircus.mjlab.intergrid_rebound_calibration import GRAVITY, _trace_impacts


class ReboundCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.times = np.arange(31) / 1000.
        self.normal = np.array([0.6, 0., 0.8])
        self.velocity = np.array([0., 0., -2.]) + self.times[:, None] * GRAVITY
        self.forces = np.zeros((30, len(FORCE_NAMES)))
        self.add_impact("board", 6, 8, self.normal, .65, .8)
        self.add_impact("table", 16, 18, np.array([0., 0., 1.]), .6, .9,
                        incoming_z=-1.5)

    def add_impact(self, surface, start, end, normal, restitution, retention, *, incoming_z=None):
        midpoint = (self.times[start] + self.times[end]) / 2
        if incoming_z is not None:
            self.velocity[start - 4:, 2] = incoming_z + (self.times[start - 4:] - midpoint) * GRAVITY[2]
        incoming = self.velocity[start] + (midpoint - self.times[start]) * GRAVITY
        normal_in = float(incoming @ normal)
        outgoing = retention * (incoming - normal_in * normal) - restitution * normal_in * normal
        self.velocity[end:] = outgoing + (self.times[end:, None] - midpoint) * GRAVITY
        self.forces[start:end, FORCE_NAMES.index(surface)] = 20.

    def summarize(self):
        arrays = dict(record_ids=["shot"], position_m=np.zeros((31, 1, 3)),
                      velocity_m_s=self.velocity[:, None, :], forces_N=self.forces[:, None, :],
                      time_s=self.times, dt_s=.001)
        return _trace_impacts(dict(id="shot", native_cache_row=0, physics_steps=30,
                                   board_normal=self.normal.tolist()), arrays)

    def test_isolated_rebounds_recover_known_velocity_ratios(self):
        measured = self.summarize()
        self.assertEqual(set(measured), {"board", "table"})
        for surface, normal, tangent in (("board", .65, .8), ("table", .6, .9)):
            self.assertAlmostEqual(measured[surface]["normal_restitution"], normal)
            self.assertAlmostEqual(measured[surface]["tangent_retention"], tangent)

    def test_intervening_bucket_contact_excludes_table_estimate(self):
        self.forces[12, FORCE_NAMES.index("bucket")] = 20.
        self.assertEqual(set(self.summarize()), {"board"})

    def test_overlap_with_board_contact_excludes_both_estimates(self):
        self.forces[7, FORCE_NAMES.index("robot")] = 20.
        self.assertEqual(self.summarize(), {})


if __name__ == "__main__":
    unittest.main()
