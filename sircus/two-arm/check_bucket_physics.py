"""Analytic bucket geometry checks plus reduced-order contact regressions."""
import copy
import math
import unittest

from physics import (bucket_contact, bucket_contains_sphere, bucket_profile,
                     default_config, simulate, validate_config)


def config():
    c = default_config()
    c['board'].update(center_mm=[600, 0, 350], size_mm=[200, 300, 5], rpy_deg=[0, 30, 90])
    c['table'].update(center_mm=[550, 0, -5], size_mm=[850, 800, 40])
    c['bin'] = {'shape': 'tapered_bucket', 'center_xy_mm': [680, 245],
                'inner_top_diameter_mm': 177, 'outer_top_diameter_mm': 200,
                'inner_bottom_diameter_mm': 150, 'outer_bottom_diameter_mm': 158,
                'reported_height_mm': 175, 'height_interpretation': 'unresolved',
                'interior_depth_mm': 175, 'bottom_thickness_mm': 8,
                'floor_top_mm': 27, 'contact_damping_s_m': .6}
    return c


def point(c, rho, z, angle=0):
    x, y = c['bin']['center_xy_mm']
    return [x + rho * math.cos(angle), y + rho * math.sin(angle), z]


class BucketGeometryChecks(unittest.TestCase):
    def setUp(self):
        self.c = config()
        self.p = bucket_profile(self.c)

    def test_reported_height_remains_distinct_from_assumptions(self):
        self.assertEqual(self.c['bin']['reported_height_mm'], 175)
        self.assertEqual(self.c['bin']['height_interpretation'], 'unresolved')
        self.assertEqual(self.p['interior_depth_mm'], 175)
        self.assertEqual(self.p['outer_height_mm'], 183)
        self.assertEqual(self.p['inner_top_radius_mm'], 88.5)
        self.assertEqual(self.p['inner_bottom_radius_mm'], 75)
        self.assertEqual(self.p['outer_top_radius_mm'], 100)
        self.assertEqual(self.p['outer_bottom_radius_mm'], 79)

    def test_open_top_has_no_phantom_lid(self):
        for z in (202, 217, 240):
            self.assertIsNone(bucket_contact(point(self.c, 0, z), 37.5, self.c))

    def test_inner_wall_contact_uses_taper_normal(self):
        slope = 13.5 / 175
        z = 107
        inner_radius = 75 + slope * (z - 27)
        name, hit = bucket_contact(point(self.c, inner_radius - 10, z), 37.5, self.c)
        self.assertEqual(name, 'bucket-inner-wall')
        self.assertAlmostEqual(hit[0], 37.5 - 10 / math.sqrt(1 + slope * slope), places=10)
        self.assertAlmostEqual(hit[1][0], -1 / math.sqrt(1 + slope * slope), places=10)
        self.assertAlmostEqual(hit[1][2], slope / math.sqrt(1 + slope * slope), places=10)

    def test_rim_is_annular_and_blocks_edge_strikes(self):
        name, hit = bucket_contact(point(self.c, 94, 212), 20, self.c)
        self.assertEqual(name, 'bucket-rim')
        self.assertAlmostEqual(hit[0], 10)
        self.assertEqual(hit[1], (0, 0, 1))
        # Grazing the opening edge sees the finite corner, not an infinite wall.
        name, hit = bucket_contact(point(self.c, 85, 212), 20, self.c)
        self.assertEqual(name, 'bucket-rim')
        self.assertLess(hit[1][0], 0)
        self.assertGreater(hit[1][2], 0)

    def test_outside_wall_and_underside_are_solid(self):
        z = 100
        outer_radius = 79 + 21 * (z - 19) / 183
        name, hit = bucket_contact(point(self.c, outer_radius + 5, z), 20, self.c)
        self.assertEqual(name, 'bucket-outer-wall')
        self.assertGreater(hit[1][0], 0)
        self.assertLess(hit[1][2], 0)
        name, hit = bucket_contact(point(self.c, 0, 14), 10, self.c)
        self.assertEqual(name, 'bucket-underside')
        self.assertEqual(hit[1], (0, 0, -1))

    def test_floor_is_finite_and_normal_points_up(self):
        name, hit = bucket_contact(point(self.c, 0, 60), 37.5, self.c)
        self.assertEqual(name, 'bucket-floor')
        self.assertEqual(hit[1], (0, 0, 1))
        self.assertAlmostEqual(hit[0], 4.5)
        self.assertIsNone(bucket_contact(point(self.c, 250, 60), 37.5, self.c))

    def test_axisymmetric_contacts_rotate_with_ball(self):
        base = bucket_contact(point(self.c, 70, 107), 37.5, self.c)
        rotated = bucket_contact(point(self.c, 70, 107, math.pi / 3), 37.5, self.c)
        self.assertEqual(base[0], rotated[0])
        self.assertAlmostEqual(base[1][0], rotated[1][0], places=10)
        self.assertAlmostEqual(rotated[1][1][0], base[1][1][0] * .5, places=10)
        self.assertAlmostEqual(rotated[1][1][1], base[1][1][0] * math.sqrt(3) / 2, places=10)

    def test_capture_checks_inner_frustum_not_opening_box(self):
        z = 27 + 37.5
        self.assertTrue(bucket_contains_sphere(self.c, point(self.c, 40, z), 37.5))
        self.assertFalse(bucket_contains_sphere(self.c, point(self.c, 50, z), 37.5))
        x, y = self.c['bin']['center_xy_mm']
        self.assertFalse(bucket_contains_sphere(self.c, [x + 40, y + 40, z], 37.5))
        self.assertFalse(bucket_contains_sphere(self.c, point(self.c, 0, 200), 37.5))
        self.assertFalse(bucket_contains_sphere(self.c, point(self.c, 0, z - .1), 37.5))
        self.assertTrue(bucket_contains_sphere(self.c, point(self.c, 0, z - .1), 37.5, tolerance_mm=.5))

    def test_capture_is_independent_of_outer_diameters(self):
        other = copy.deepcopy(self.c)
        other['bin'].update(outer_bottom_diameter_mm=250, outer_top_diameter_mm=300)
        for rho in (0, 40, 45, 55, 80):
            pos = point(self.c, rho, 64.5)
            self.assertEqual(bucket_contains_sphere(self.c, pos, 37.5), bucket_contains_sphere(other, pos, 37.5))

    def test_dimensions_reject_inverted_or_impossible_shell(self):
        for key, value in (('inner_top_diameter_mm', 140), ('outer_top_diameter_mm', 170),
                           ('bottom_thickness_mm', 0), ('interior_depth_mm', 50),
                           ('height_interpretation', 'assumed_measured_depth')):
            c = copy.deepcopy(self.c)
            c['bin'][key] = value
            with self.assertRaises(ValueError):
                validate_config(c)

    def test_board_bucket_intersection_is_rejected(self):
        c = copy.deepcopy(self.c)
        c['board'].update(center_mm=[680 + 90, 245, 150], rpy_deg=[0, 0, 0])
        with self.assertRaisesRegex(ValueError, 'intersects bucket'):
            validate_config(c)

    def test_rim_bounce_capture_survives_timestep_refinement(self):
        c = copy.deepcopy(self.c)
        action = {'u_mm': 0, 'v_mm': 0, 'gap_mm': 140}
        coarse = simulate(c, action, seed=42, frames=False)
        c['simulation']['dt_s'] /= 2
        fine = simulate(c, action, seed=42, frames=False)
        for result in (coarse, fine):
            self.assertEqual(result['outcome'], 'retained')
            self.assertTrue(result['first_floor_contact'])
            self.assertIn('bucket-rim', [x['object'] for x in result['contacts']])
            self.assertIn('bucket-floor', [x['object'] for x in result['contacts']])
            self.assertLess(result['max_energy_j'], result['initial_energy_j'] * 1.02)
            self.assertTrue(bucket_contains_sphere(c, result['final_ball_xyz_mm'], 37.5, tolerance_mm=.5))
        # Several board/rim bounces make the final settled point sensitive to dt;
        # both trajectories must remain inside the measured interior.

    def test_bucket_simulation_is_seeded_and_keeps_scene_immutable(self):
        c = copy.deepcopy(self.c)
        before = copy.deepcopy(c)
        action = {'u_mm': 0, 'v_mm': 0, 'gap_mm': 140}
        first = simulate(c, action, seed=42, frames=False)
        second = simulate(c, action, seed=42, frames=True)
        self.assertEqual(first['final_ball_xyz_mm'], second['final_ball_xyz_mm'])
        self.assertEqual(first['outcome'], second['outcome'])
        self.assertEqual(first['container_shape'], 'tapered_bucket')
        self.assertFalse(first['physically_verified'])
        self.assertEqual(c, before)


if __name__ == '__main__':
    unittest.main()
