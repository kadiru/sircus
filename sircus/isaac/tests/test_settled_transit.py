"""Independent pure/mock checks for opt-in driven settlement, without Isaac."""
import ast
import copy
import math
import sys
import types
import unittest
from pathlib import Path
from unittest import mock
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import settled_transit as controller
from prepare_assets import fk
from continuous_policy import torch

class Array:
    def __init__(self, value): self.value = np.asarray(value, dtype=float)
    def clone(self): return Array(self.value.copy())
    def cpu(self): return self
    def numpy(self): return self.value
    def tolist(self): return self.value.tolist()
    def __getitem__(self, key): return Array(self.value[key])
    def __setitem__(self, key, value): self.value[key] = value.value if isinstance(value, Array) else value

class Arm:
    joint_names = [f"joint{i+1}" for i in range(6)]
    def __init__(self):
        self.data = types.SimpleNamespace(joint_pos=Array(np.zeros((1, 6))))
        self.commands = []
    def set_joint_position_target(self, value): self.commands.append(value.value.copy())

def metric(**changes):
    out = dict(position_error_mm=0., normal_error_deg=0., joint_speed_rad_s=0.,
               position_drift_mm=0., normal_drift_deg=0.,
               dropper_position_error_mm=0., dropper_drift_mm=0.,
               dropper_flange_m=[0., 0., 0.],
               arms={n:dict(position_rad=[0.]*6, target_rad=[.01]*6)
                     for n in ("Arm10", "Arm14")})
    out.update(changes)
    return out

class FitRepresentationTests(unittest.TestCase):
    def links(self, last_axis=(0., 0., 1.)):
        return [(np.eye(4), np.array(last_axis if i == 5 else (0., 0., 1.)),
                 -2*math.pi, 2*math.pi) for i in range(6)]
    def test_legal_turn_removal_preserves_full_fk_and_does_not_mutate_input(self):
        links = self.links(); q = [5.9, 0., 0., 0., 0., -5.8]
        fit = dict(solved=True, q=q.copy(), residual=0.); old = copy.deepcopy(fit)
        target = fk(q, links)
        result = controller.equivalent_fit(target, fit, links, np.zeros(6), False)
        self.assertEqual(fit, old)
        np.testing.assert_allclose(fk(result["q"], links), target, atol=1e-12)
        self.assertLess(abs(result["q"][0]), 1.)
        self.assertLess(abs(result["q"][-1]), 1.)
        self.assertTrue(result["equivalent_representation"]["max_change_rad"] > 6.)
    def test_normal_only_null_spin_returns_to_reference_but_full_pose_does_not(self):
        links = self.links(); q = [0., 0., 0., 0., 0., 2.]; reference = np.zeros(6)
        target = fk(q, links)
        a = controller.equivalent_fit(target, dict(solved=True, q=q), links, reference, True)
        b = controller.equivalent_fit(target, dict(solved=True, q=q), links, reference, False)
        self.assertEqual(a["q"][-1], 0.); self.assertEqual(b["q"][-1], 2.)
        np.testing.assert_allclose(fk(a["q"], links)[:3, 2], target[:3, 2], atol=1e-12)
    def test_non_null_wrist_is_not_replaced_and_bad_fit_is_rejected(self):
        links = self.links(last_axis=(1., 0., 0.)); q = [0., 0., 0., 0., 0., 1.]
        result = controller.equivalent_fit(fk(q, links), dict(solved=True, q=q), links, np.zeros(6), True)
        self.assertEqual(result["q"][-1], 1.)
        with self.assertRaises(ValueError):
            controller.equivalent_fit(np.eye(4), dict(solved=True, q=q), links, np.zeros(6), False)
    def test_unsolved_fit_stays_unsolved_and_joint_limits_are_respected(self):
        fit = dict(solved=False, q=[0.]*6)
        self.assertIs(controller.equivalent_fit(np.eye(4), fit, self.links(), np.zeros(6), False), fit)
        links = [(o, a, -1., 6.) for o, a, lo, hi in self.links()]
        result = controller.equivalent_fit(fk([5.9]*6, links), dict(solved=True, q=[5.9]*6), links, [-.8]*6, False)
        self.assertTrue(all(-1. <= q <= 6. for q in result["q"]))
        np.testing.assert_allclose(result["q"], [5.9-2*math.pi]*6)

class GateTests(unittest.TestCase):
    def test_release_tolerances_are_not_loosened(self):
        self.assertTrue(controller.release_ready(metric()))
        for name, value in [("position_error_mm", 2.001), ("normal_error_deg", .5001),
                            ("joint_speed_rad_s", .005001), ("position_drift_mm", .10001),
                            ("normal_drift_deg", .05001), ("dropper_position_error_mm", 2.001)]:
            with self.subTest(gate=name): self.assertFalse(controller.release_ready(metric(**{name:value})))
        self.assertFalse(controller.quiet(metric(dropper_drift_mm=.101)))
    def test_wait_extends_drift_without_invoking_cartesian_feedback(self):
        env = controller.SettledTransitShots(); records = [[]]; calls = []
        def measure(*args):
            calls.append(None)
            m = metric(position_drift_mm=.2 if len(calls) < 10 else 0.)
            return [m], np.zeros((1, 3)), np.array([[0., 0., 1.]])
        env._measure = measure
        env._wait_stable(None, None, 0, records, np.array([True]))
        self.assertEqual(len(calls), 10)
        calls.clear()
        env._measure = lambda *a: (calls.append(None) or ([metric(position_drift_mm=.2)], None, None))
        env._wait_stable(None, None, 0, records, np.array([True]))
        self.assertEqual(len(calls), 32)
    def test_setup_does_not_write_endpoint_state_or_reset_articulations(self):
        tree = ast.parse(Path(controller.__file__).read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SettledTransitShots")
        attrs = [n.func.attr for n in ast.walk(cls) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
        for forbidden in ["write_joint_state_to_sim", "write_root_state_to_sim", "reset"]: self.assertNotIn(forbidden, attrs)

class FeedbackSelectionTests(unittest.TestCase):
    def env(self, metrics):
        env = controller.SettledTransitShots(); env.n = 1
        env.active_mask = np.array([True]); env.trial_mode = "train"
        env.arms = {n:Arm() for n in ("Arm10", "Arm14")}
        env.base_cells = [dict(id="cell")]; env.spec = {"cells":[]}
        env.cfg = dict(arms=dict(base_spacing_m=1.), board=dict(tool_offset_m=.1, mount_yaw_rad=0.))
        env.continuous_links = {n:[] for n in env.arms}; env.ik_cache = {}
        env.approach_endpoints = lambda q: None; env.assert_buckets_fixed = lambda: None
        env._wait_stable = mock.Mock(side_effect=[([m], np.array([[1.,0.,.1]]), np.array([[0.,0.,1.]])) for m in metrics])
        env._ik = mock.Mock(return_value=dict(solved=True, q=[.02]*6))
        return env
    def run_prepare(self, env):
        fit = dict(accepted=True, ik={n:dict(q=[.01]*6) for n in env.arms})
        endpoints = {n:np.eye(4) for n in env.arms}
        with mock.patch("feasibility_contract.dynamic_cell", side_effect=lambda c,a:c.copy()), mock.patch("expanded_pose_menu.endpoint_targets", return_value=endpoints), mock.patch("continuous_feasibility.qualify_action", return_value=fit), mock.patch("continuous_feasibility.approximate_path_check", return_value=dict(accepted=True,reasons=[])) as proxy:
            result = env.prepare([{}])
        return result, proxy
    def test_moving_or_grossly_lagging_arm_does_not_receive_bias_correction(self):
        for m, reason in [(metric(position_error_mm=3.,position_drift_mm=.2), "setup_motion_not_quiet"), (metric(position_error_mm=3., arms={n:dict(position_rad=[0.]*6,target_rad=[.03]*6) for n in ("Arm10","Arm14")}), "setup_transit_incomplete")]:
            with self.subTest(reason=reason):
                env = self.env([m, metric()]); result, _ = self.run_prepare(env)
                self.assertEqual(result[-1], [reason]); env._ik.assert_not_called()
    def test_only_offset_dropper_is_corrected_and_proxy_checks_other_actual_goal(self):
        env = self.env([metric(dropper_position_error_mm=3.), metric()])
        result, proxy = self.run_prepare(env)
        self.assertEqual(result[-1], [None]); self.assertEqual(env._ik.call_count, 1)
        self.assertEqual(env._ik.call_args.args[1], "Arm10")
        self.assertEqual(env.correction_history[0]["arms"], ["Arm10"])
        corrected_targets = proxy.call_args_list[-1].args[3]
        np.testing.assert_allclose(corrected_targets["Arm14"], [.01]*6)


@unittest.skipIf(torch is None, "CPU Torch environment unavailable")
class DriveRestorationTests(unittest.TestCase):
    def env(self, raises=False):
        class NativeArm:
            joint_names = [f"joint{i+1}" for i in range(6)]
            def __init__(self):
                self.data = types.SimpleNamespace(
                    joint_pos=torch.tensor([[.1]*6,[.2]*6]),
                    joint_vel=torch.zeros((2,6)),
                    joint_pos_target=torch.tensor([[.1]*6,[.2]*6]),
                    joint_effort_limits=torch.full((2,6),100.),
                    joint_stiffness=torch.full((2,6),1500.),
                    joint_damping=torch.full((2,6),100.))
                self.commands=[]; self.gains=[]
            def set_joint_position_target(self, goal):
                self.commands.append(goal.clone()); self.data.joint_pos_target=goal.clone()
            def write_joint_effort_limit_to_sim(self, value, ids):
                self.data.joint_effort_limits[:,ids]=torch.as_tensor(value).expand(2,6)
            def write_joint_stiffness_to_sim(self, value, ids):
                self.data.joint_stiffness[:,ids]=torch.as_tensor(value).expand(2,6)
                self.gains.append(self.data.joint_stiffness.clone())
            def write_joint_damping_to_sim(self, value, ids):
                self.data.joint_damping[:,ids]=torch.as_tensor(value).expand(2,6)
        env=controller.SettledTransitShots();env.n=2;env.steps=0;env.dt=1/240
        env.preparation_enabled=np.array([True,False])
        env.arms={n:NativeArm() for n in ("Arm10","Arm14")}
        env.transit_limits=torch.full((6,),200.)
        env.set_balls=lambda *args:None;env.assert_buckets_fixed=lambda:None
        def tick():
            env.steps+=1
            if raises and env.steps==3:raise RuntimeError("injected simulator failure")
            for a in env.arms.values():
                a.data.joint_pos=a.data.joint_pos_target.clone()
        env.tick=tick
        return env
    def assert_original(self, env):
        for a in env.arms.values():
            torch.testing.assert_close(a.data.joint_effort_limits,torch.full((2,6),100.))
            torch.testing.assert_close(a.data.joint_stiffness,torch.full((2,6),1500.))
            torch.testing.assert_close(a.data.joint_damping,torch.full((2,6),100.))
    def test_original_gains_restored_and_inactive_goals_hold_measured_pose(self):
        env=self.env()
        targets={n:torch.tensor([[.4]*6,[8.]*6]) for n in env.arms}
        env.approach_endpoints(targets)
        self.assert_original(env)
        self.assertEqual(env.transit_ready_after_drive,[True,True])
        self.assertEqual(env.transit_history[-1]["phase"],"restored_original_hold")
        for a in env.arms.values():
            self.assertGreater(len(a.commands),10)
            self.assertTrue(all(torch.allclose(x[1],torch.full((6,),.2)) for x in a.commands))
            self.assertTrue(torch.allclose(a.commands[-1][0],torch.full((6,),.4)))
            # Includes strictly intermediate stiffness values during restoration.
            self.assertTrue(any(bool(((g>1500)&(g<6000)).all()) for g in a.gains))
    def test_tick_failure_restores_original_drives_before_propagation(self):
        env=self.env(raises=True)
        with self.assertRaisesRegex(RuntimeError,"injected simulator failure"):
            env.approach_endpoints({n:torch.ones((2,6)) for n in env.arms})
        self.assert_original(env)

if __name__ == "__main__": unittest.main()
