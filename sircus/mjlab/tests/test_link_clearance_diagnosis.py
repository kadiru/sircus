"""Contact evidence alignment and honest link-attribution uncertainty."""
import unittest

import numpy as np

from sircus.mjlab.link_clearance_diagnosis import attribution, compact, force_summary


class LinkDiagnosisTests(unittest.TestCase):
    def test_unique_overlap_has_named_link_but_no_measured_identity_claim(self):
        result = attribution([(.01, "dropper/finger"), (-.0002, "bouncer/link3"), (.005, "bouncer/link4")])
        self.assertEqual(result["status"], "unique_overlap")
        self.assertEqual(result["inferred_geometry"], "bouncer/link3")
        self.assertEqual(result["second_nearest_geometry"], "bouncer/link4")
        self.assertAlmostEqual(result["nearest_signed_distance_m"], -.0002)

    def test_multiple_overlaps_do_not_guess_identity(self):
        result = attribution([(-.002, "bouncer/link3"), (-.001, "bouncer/link4")])
        self.assertEqual(result["status"], "multiple_overlaps")
        self.assertIsNone(result["inferred_geometry"])
        self.assertEqual(result["overlapping_geometries"], ["bouncer/link3", "bouncer/link4"])

    def test_no_overlap_does_not_turn_nearest_into_contact(self):
        result = attribution([(0., "bouncer/link3"), (.1, "dropper/finger")])
        self.assertEqual(result["status"], "no_overlap_at_saved_sample")
        self.assertIsNone(result["inferred_geometry"])

    def test_rejects_incomplete_distances_and_duplicate_identity(self):
        for rows in ([], [(np.nan, "x")], [(np.inf, "x")], [(0., "x"), (.1, "x")]):
            with self.assertRaises(ValueError):
                attribution(rows)

    def test_force_first_event_is_not_global_force_peak(self):
        t = np.arange(6) * .1
        p, v, f = np.zeros((6, 3)), np.zeros((6, 3)), np.zeros((5, 6))
        f[1, 4], f[3:5, 4] = 3., [8., 20.]
        result = force_summary(t, p, v, f)
        self.assertEqual(result["first_robot"]["start_tick"], 1)
        self.assertEqual(result["peak_tick"], 4)
        self.assertEqual(result["robot_ticks"], 3)

    def test_no_robot_force_and_malformed_trace(self):
        t, p, v, f = np.arange(3), np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((2, 6))
        f[0, 0] = 10.
        result = force_summary(t, p, v, f)
        self.assertIsNone(result["first_robot"])
        self.assertIsNone(result["peak_tick"])
        self.assertEqual(result["robot_ticks"], 0)
        with self.assertRaises(ValueError):
            force_summary(t, p[:-1], v, f)

    def test_compact_preserves_all_attempts_and_uncertainty(self):
        cases = [dict(id="target", bucket_xy_m=[1., 0.], shots=3, robot_contact_shots=2,
            reconstructed_first_link_counts={"link3": 1, "multiple_overlaps": 1, "no_robot_force": 1},
            results=[dict(measured_first_robot=dict(start_s=.4)), dict(measured_first_robot=dict(start_s=.3)),
                     dict(measured_first_robot=None)])]
        result = compact(cases)[0]
        self.assertEqual(result["trials"], 3)
        self.assertEqual(result["robot_contacts"], 2)
        self.assertEqual(result["first_contact_time_s"], dict(min=.3, max=.4))
        self.assertEqual(result["first_contact_link_counts"]["multiple_overlaps"], 1)


if __name__ == "__main__":
    unittest.main()
