"""Run: python3 -m unittest discover -s sircus/simulation -p check_physics.py -v"""
import copy
import math
import unittest

from physics import default_config, matvec, release_point, rotation, simulate, validate_config


class CompliantBoardChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = default_config()
        cls.center_action = {"u_mm":0,"v_mm":0,"gap_mm":200}
        cls.center = simulate(cls.config,cls.center_action)
        cls.edge_action = {"u_mm":60,"v_mm":-40,"gap_mm":200}
        cls.edge = simulate(cls.config,cls.edge_action,frames=False)

    def test_default_capture_requires_settling(self):
        self.assertEqual(self.center["outcome"],"retained")
        self.assertTrue(self.center["first_floor_contact"])
        self.assertLess(self.center["final_speed_mm_s"],60)
        self.assertFalse(self.center["calibrated"])
        self.assertFalse(self.center["physically_verified"])
        self.assertGreater(self.center["duration_s"],.2)

    def test_board_moves_during_collision(self):
        self.assertGreater(self.center["max_support_displacement_mm"],1)
        transforms = self.center["frames"]
        self.assertTrue(any(math.dist(f["board_center_mm"],self.config["board"]["center_mm"]) > 1 for f in transforms))
        self.assertTrue(any(math.dist(f["board_rpy_deg"],self.config["board"]["rpy_deg"]) > .1 for f in transforms))

    def test_edge_changes_dynamics(self):
        self.assertGreater(math.dist(self.center["outgoing_mm_s"],self.edge["outgoing_mm_s"]),100)
        self.assertGreater(self.edge["max_support_tilt_deg"],self.center["max_support_tilt_deg"])

    def test_time_step_convergence(self):
        c = copy.deepcopy(self.config)
        c["simulation"]["dt_s"] /= 2
        fine = simulate(c,self.edge_action,frames=False)
        self.assertEqual(fine["outcome"],self.edge["outcome"])
        self.assertLess(math.dist(fine["landing_xy_mm"],self.edge["landing_xy_mm"]),5)
        self.assertLess(math.dist(fine["outgoing_mm_s"],self.edge["outgoing_mm_s"]),30)

    def test_energy_does_not_materially_increase(self):
        # Explicit integration introduces bounded numerical error; this is a
        # regression tolerance, not a guarantee for every possible configuration.
        for result in (self.center,self.edge):
            self.assertLess(result["max_energy_j"],result["initial_energy_j"]*1.02)

    def test_first_floor_contact_is_not_retention(self):
        c = copy.deepcopy(self.config)
        c["bin"]["contact_damping_s_m"] = .12
        result = simulate(c,self.center_action,frames=False)
        self.assertTrue(result["first_floor_contact"])
        self.assertFalse(result["retained"])
        self.assertEqual(result["outcome"],"floor_contact_not_retained")

    def test_genuine_miss_is_not_corrected_towards_bin(self):
        c = copy.deepcopy(self.config)
        c["bin"]["center_xy_mm"] = [800,220]
        result = simulate(c,self.center_action,frames=False)
        self.assertFalse(result["retained"])
        self.assertFalse(result["first_floor_contact"])
        self.assertEqual(result["outgoing_mm_s"],self.center["outgoing_mm_s"])

    def test_yaw_changes_launch_direction(self):
        c = copy.deepcopy(self.config)
        c["board"]["rpy_deg"][2] = 90
        c["bin"]["center_xy_mm"] = [800,220]
        result = simulate(c,self.center_action,frames=False)
        original = self.center["outgoing_mm_s"]
        rotated = result["outgoing_mm_s"]
        self.assertAlmostEqual(rotated[0],-original[1],delta=1)
        self.assertAlmostEqual(rotated[1],original[0],delta=1)
        self.assertAlmostEqual(rotated[2],original[2],delta=1)

    def test_full_orientation_matrix(self):
        r = rotation([-23,17,61])
        axes = [matvec(r,v) for v in ((1,0,0),(0,1,0),(0,0,1))]
        for i in range(3):
            self.assertAlmostEqual(sum(v*v for v in axes[i]),1,places=12)
            for j in range(i): self.assertAlmostEqual(sum(x*y for x,y in zip(axes[i],axes[j])),0,places=12)
        self.assertAlmostEqual(axes[0][0]*(axes[1][1]*axes[2][2]-axes[1][2]*axes[2][1])-axes[0][1]*(axes[1][0]*axes[2][2]-axes[1][2]*axes[2][0])+axes[0][2]*(axes[1][0]*axes[2][1]-axes[1][1]*axes[2][0]),1)

    def test_no_silent_clamping(self):
        for field,value in (("u_mm",1000),("v_mm",float("nan")),("gap_mm",10),("gap_mm",True)):
            a = dict(self.center_action)
            a[field] = value
            with self.assertRaises(ValueError): release_point(self.config,a)
        c = copy.deepcopy(self.config)
        c["ball"]["mass_kg"] = float("inf")
        with self.assertRaises(ValueError): validate_config(c)
        c = copy.deepcopy(self.config)
        c["board"]["rpy_deg"][0] = 100
        with self.assertRaises(ValueError): validate_config(c)
        c = copy.deepcopy(self.config)
        c["simulation"]["dt_s"] = .001
        with self.assertRaises(ValueError): validate_config(c)

    def test_intersecting_board_layout_is_rejected(self):
        c = copy.deepcopy(self.config)
        c["board"]["center_mm"][2] = 15
        with self.assertRaisesRegex(ValueError,"intersects table"):
            validate_config(c)

    def test_yawed_board_in_clear_space_is_accepted(self):
        c = copy.deepcopy(self.config)
        c["board"]["rpy_deg"] = [-23,17,61]
        c["board"]["center_mm"][2] = 260
        self.assertEqual(validate_config(c),c)

    def test_offset_mount_converges_with_smaller_step(self):
        c = copy.deepcopy(self.config)
        c["board"]["support_offset_mm"] = [30,-20,-6]
        coarse = simulate(c,self.edge_action,frames=False)
        c["simulation"]["dt_s"] /= 2
        fine = simulate(c,self.edge_action,frames=False)
        self.assertEqual(coarse["outcome"],fine["outcome"])
        self.assertLess(math.dist(coarse["landing_xy_mm"],fine["landing_xy_mm"]),5)
        self.assertLess(math.dist(coarse["outgoing_mm_s"],fine["outgoing_mm_s"]),30)
        self.assertLess(coarse["max_energy_j"],coarse["initial_energy_j"]*1.02)

    def test_seeded_noise_and_frames_do_not_change_outcomes(self):
        c = copy.deepcopy(self.config)
        c["simulation"].update(noise_xy_mm=2,noise_gap_mm=3)
        a = simulate(c,self.center_action,seed=172,frames=True)
        b = simulate(c,self.center_action,seed=172,frames=False)
        self.assertEqual(a["final_ball_xyz_mm"],b["final_ball_xyz_mm"])
        self.assertEqual(a["release_xyz_mm"],b["release_xyz_mm"])
        self.assertNotEqual(a["release_xyz_mm"],a["nominal_release_xyz_mm"])
        self.assertEqual(b["frames"],[])

    def test_configuration_is_not_mutated(self):
        c = default_config()
        before = copy.deepcopy(c)
        simulate(c,self.center_action,frames=False)
        self.assertEqual(c,before)


if __name__ == "__main__":
    unittest.main()
