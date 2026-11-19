"""Synthetic geometry controls; no robot command or native rollout."""
import copy
import math
import unittest

import numpy as np

from sircus.mjlab import arm_clearance_selector as gate


def box(name, center, half_size):
    return dict(name=name, planes=gate.box_planes(center, half_size))


class BoundTests(unittest.TestCase):
    def test_frozen_rule_has_exact_budget_and_scope(self):
        self.assertEqual(gate.MARGIN_M, .002)
        self.assertEqual(gate.MAX_SPACING_S, .002)
        self.assertEqual(gate.MAX_HORIZON_S, 1.5)
        self.assertEqual(gate.RULE['members_required'], 5)
        self.assertEqual(gate.RULE['maximum_candidate_geometry_visits'], 4)
        self.assertFalse(gate.RULE['incoming_drop_checked'])

    def test_box_planes_inside_outside_and_rotated(self):
        p = gate.box_planes([1, 0, 0], [.1, .2, .3])
        self.assertAlmostEqual(float((p[:, :3]@[1, 0, 0]+p[:, 3]).max()), -.1)
        self.assertAlmostEqual(float((p[:, :3]@[1.4, 0, 0]+p[:, 3]).max()), .3)
        rotation = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
        moved = gate.world_planes(p, rotation, [0, 0, 2])
        self.assertAlmostEqual(float((moved[:, :3]@[0, 1, 2]+moved[:, 3]).max()), -.1)

    def test_mesh_hull_contains_known_cube_and_roundoff(self):
        vertices = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
        p = gate.mesh_planes(vertices)
        self.assertEqual(p.shape, (6, 4))
        self.assertTrue(np.allclose(np.linalg.norm(p[:, :3], axis=1), 1.))
        self.assertAlmostEqual(float((p[:, :3]@[0, 0, 0]+p[:, 3]).max()), -1.)

    def test_clear_path_passes_with_reported_continuous_bound(self):
        result = gate.path_bound([0, 0, 1], [1, 0, 0], .3, [0, 0, -9.81],
            [box('Arm10/obstacle', [0, 0, 0], [.05, .05, .05])], .04)
        self.assertTrue(result['passed'])
        self.assertGreater(result['minimum_clearance_lower_bound_m'], .4)
        self.assertLessEqual(result['actual_spacing_s'], .002)
        self.assertAlmostEqual(result['between_sample_allowance_m'],
            result['maximum_speed_m_s']*result['actual_spacing_s']/2)

    def test_radius_detects_collision_when_center_does_not_touch(self):
        result = gate.path_bound([0, 0, 1], [0, 0, 0], .01, [0, 0, -9.81],
            [box('Arm14/hand', [.05, 0, 1], [.02, .02, .02])], .04)
        self.assertFalse(result['passed'])
        self.assertLess(result['minimum_clearance_lower_bound_m'], 0)
        self.assertEqual(result['closest']['geom'], 'Arm14/hand')

    def test_between_sample_hit_cannot_pass_fast_thin_obstacle(self):
        # 1mm obstacle entirely between endpoints; naive sample testing passes.
        result = gate.path_bound([-.1, 0, 0], [100, 0, 0], .002, [0, 0, 0],
            [box('Arm10/thin', [0, 0, 0], [.0005, .01, .01])], .005)
        self.assertEqual(result['samples'], 2)
        self.assertFalse(result['passed'])
        self.assertAlmostEqual(result['between_sample_allowance_m'], .1)

    def test_missing_or_overlong_geometry_never_truncates_to_clear(self):
        for horizon in (0, -1, 1.500001, math.inf, math.nan):
            with self.assertRaises(ValueError): gate.path_bound([0, 0, 0], [1, 0, 0], horizon, [0, 0, -9.81],
                [box('Arm10/part', [3, 0, 0], [.1]*3)], .04)
        with self.assertRaises(ValueError): gate.path_bound([0]*3, [1, 0, 0], .5, [0, 0, -9.81], [], .04)
        with self.assertRaises(ValueError): gate.world_planes(gate.box_planes([0]*3, [1]*3), np.eye(3)*2, [0]*3)

    def test_five_members_all_required_and_endpoint_bound(self):
        # Inject a synthetic fixed collision geometry, without importing MuJoCo.
        instance = object.__new__(gate.ArmClearanceGate)
        instance.model_sha256 = 'a'*64; instance.geometry_sha256 = 'b'*64
        instance.radius = .04; instance.excluded = [gate.RULE['excluded_collision_geometry']]
        instance.nominal_geometry = dict(ball_radius_m=.04, board_size_m=[.2, .3, .005], mount_yaw_rad=math.pi/2)
        instance.colliders = lambda *args: [box('Arm10/part', [0, 0, 0], [.05]*3)]
        launch = gate.nominal_origin([0, 0, .9, 25, .3], [.5, 0], instance.nominal_geometry)
        members = [dict(member=i, full_downward_rim_band_available=True,
            rim_band_start_s=.1, origin_m=launch['origin_m'], velocity_world_m_s=[1, 0, 0]) for i in range(5)]
        flight = dict(launch=launch, members=members)
        a = instance.check([0]*6, [0]*6, [.5, 0], [0, 0, .9, 25, .3], flight)
        self.assertTrue(a['passed']); self.assertEqual(len(a['members']), 5)
        changed = instance.check([.1]*6, [0]*6, [.5, 0], [0, 0, .9, 25, .3], flight)
        self.assertNotEqual(a['endpoint_sha256'], changed['endpoint_sha256'])
        for mutation in ('missing', 'overlong', 'collision'):
            with self.subTest(mutation=mutation):
                f = copy.deepcopy(flight)
                if mutation == 'missing': f['members'].pop()
                elif mutation == 'overlong': f['members'][-1]['rim_band_start_s'] = 1.6
                else: f['members'][-1]['velocity_world_m_s'] = [0, 0, -10]
                self.assertFalse(instance.check([0]*6, [0]*6, [.5, 0], [0, 0, .9, 25, .3], f)['passed'])
        with self.assertRaisesRegex(ValueError, 'geometry differs'):
            instance.check([0]*6, [0]*6, [.5, 0], [0, 0, .8, 25, .3], flight)


if __name__ == '__main__': unittest.main()
