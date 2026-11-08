import unittest

import numpy as np

from sircus.mjlab import arm_clearance_distance as precise
from sircus.mjlab import arm_clearance_selector as old


class SupportClearanceTests(unittest.TestCase):
    def box(self):
        return dict(name="box", planes=old.box_planes([0, 0, 0], [1, 1, 1]),
                    vertices=precise.box_vertices([0, 0, 0], [1, 1, 1]))

    def test_corner_distance_refinement_retains_sphere_radius_and_margin(self):
        collider = self.box()
        point = np.array([1.06, 1.06, 0.])
        directions = lambda c, p: np.tile([2**-.5, 2**-.5, 0.], (len(p), 1))
        original = old.path_bound(point, [0, 0, 0], .01, [0, 0, 0], [collider], .07)
        refined = precise.certify_path(point, [0, 0, 0], .01, [0, 0, 0], [collider], .07, directions)
        self.assertFalse(original['passed'])
        self.assertTrue(refined['passed'])
        self.assertAlmostEqual(refined['minimum_clearance_lower_bound_m'], np.sqrt(2)*.06-.07-1e-10)

    def test_bad_direction_cannot_certify_an_intersection(self):
        c = self.box()
        directions = lambda c, p: np.tile([-1., 0., 0.], (len(p), 1))
        result = precise.certify_path([1.01, 0, 0], [0, 0, 0], .01, [0, 0, 0], [c], .04, directions)
        self.assertFalse(result['passed'])

    def test_all_vertices_matter_and_unnormalized_normal_is_rejected(self):
        vertices = self.box()['vertices']
        result = precise.support_bound([[2, 0, 0]], vertices, [[1, 0, 0]])
        self.assertEqual(result.tolist(), [1.])
        with self.assertRaises(ValueError):
            precise.support_bound([[2, 0, 0]], vertices, [[2, 0, 0]])

    def test_contact_between_samples_remains_rejected(self):
        c = dict(name='thin', planes=old.box_planes([0, 0, 0], [.0001, 1, 1]),
                 vertices=precise.box_vertices([0, 0, 0], [.0001, 1, 1]))
        def normals(c, p):
            return np.column_stack([np.where(p[:, 0] < 0, -1., 1.), np.zeros((len(p), 2))])
        result = precise.certify_path([-.002, 0, 0], [2, 0, 0], .002, [0, 0, 0], [c], .0001, normals)
        self.assertFalse(result['passed'])
        self.assertEqual(result['between_sample_allowance_m'], .002)

    def test_accepted_near_unit_direction_cannot_inflate_distance(self):
        vertices = self.box()['vertices']
        plain = precise.support_bound([[2, 0, 0]], vertices, [[1, 0, 0]])
        scaled = precise.support_bound([[2, 0, 0]], vertices, [[1+5e-10, 0, 0]])
        np.testing.assert_array_equal(plain, scaled)

    def test_horizon_and_safety_constants_are_unchanged(self):
        changed = {k for k in old.RULE if precise.RULE[k] != old.RULE[k]}
        self.assertEqual(changed, {'version'})
        self.assertEqual(precise.RULE['end'], 'first_descending_rim_band_start')


if __name__ == '__main__':
    unittest.main()
