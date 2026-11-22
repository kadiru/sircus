"""Checks of contact timing boundaries, separate channels, and trace validity."""
import unittest

import numpy as np

from sircus.mjlab.hybrid_miss_diagnosis import episodes


class EpisodeTests(unittest.TestCase):
    def test_pre_contact_and_post_contact_endpoints_are_not_shifted(self):
        time = np.arange(6) * .1
        pos = np.arange(18).reshape(6, 3)
        vel = -pos
        forces = np.zeros((5, 6))
        forces[1:3, 0] = [2., 3.]
        forces[4, 4] = 4.
        out = episodes(time, pos, vel, forces)
        self.assertEqual([x["channel"] for x in out], ["board", "robot"])
        self.assertEqual(out[0]["start_tick"], 1)
        self.assertEqual(out[0]["end_tick"], 3)
        self.assertEqual(out[0]["position_m"], pos[1].tolist())
        self.assertEqual(out[0]["velocity_after_m_s"], vel[3].tolist())
        self.assertEqual(out[1]["end_tick"], 5)

    def test_contacts_on_two_surfaces_stay_separate(self):
        forces = np.zeros((3, 6))
        forces[1, [1, 4]] = 5.
        out = episodes(np.arange(4), np.zeros((4, 3)), np.zeros((4, 3)), forces)
        self.assertEqual({x["channel"] for x in out}, {"bucket", "robot"})
        self.assertTrue(all(x["start_tick"] == 1 for x in out))

    def test_threshold_is_strict(self):
        forces = np.zeros((2, 6))
        forces[0, 0] = .01
        self.assertEqual(episodes(np.arange(3), np.zeros((3, 3)), np.zeros((3, 3)), forces), [])

    def test_missing_endpoint_or_nonfinite_force_is_rejected(self):
        time, pos, vel, forces = np.arange(3), np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((2, 6))
        with self.assertRaises(ValueError):
            episodes(time, pos[:-1], vel, forces)
        forces[0, 0] = np.nan
        with self.assertRaises(ValueError):
            episodes(time, pos, vel, forces)


if __name__ == "__main__":
    unittest.main()
