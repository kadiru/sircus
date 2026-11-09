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
    def abs(self): return Array(np.abs(self.value))
    def amax(self, dim): return Array(self.value.max(axis=dim))
    def cpu(self): return self
    def numpy(self): return self.value
    def tolist(self): return self.value.tolist()
    def __getitem__(self, key): return Array(self.value[key])
    def __setitem__(self, key, value): self.value[key] = value.value if isinstance(value, Array) else value

class Arm:
    joint_names = [f"joint{i+1}" for i in range(6)]
    def __init__(self):
        self.data = types.SimpleNamespace(joint_pos=Array(np.zeros((1, 6))),default_joint_pos=Array(np.zeros((1,6))))
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
        self.assertEqual(len(calls), 160)
        calls.clear()
        env._wait_stable(None, None, 1, records, np.array([True]))
        self.assertEqual(len(calls), 32)
        spec=controller.protocol()
        self.assertEqual(spec["initial_stabilization_max_s"],40.)
        self.assertEqual(spec["correction_stabilization_max_s"],8.)
        self.assertEqual(spec["total_stabilization_max_s"],64.)
    def test_quiet_gross_joint_lag_uses_full_budget_until_feedback_is_eligible(self):
        env=controller.SettledTransitShots();calls=[]
        gross=metric(position_error_mm=3.,arms={n:dict(position_rad=[0.]*6,target_rad=[.03]*6) for n in ("Arm10","Arm14")})
        def unchanged(*args):
            calls.append(None)
            return [gross],None,None
        env._measure=unchanged
        env._wait_stable(None,None,0,[[]],np.array([True]))
        self.assertEqual(len(calls),160)
        calls.clear()
        env._wait_stable(None,None,1,[[]],np.array([True]))
        self.assertEqual(len(calls),32)
        calls.clear()
        def recovers(*args):
            calls.append(None)
            m=gross if len(calls)<20 else metric(position_error_mm=3.)
            return [m],None,None
        env._measure=recovers
        env._wait_stable(None,None,0,[[]],np.array([True]))
        self.assertEqual(len(calls),20)
        self.assertTrue(controller.quiet(gross))
        self.assertFalse(controller.feedback_ready(gross))

    def test_setup_does_not_write_endpoint_state_or_reset_articulations(self):
        tree = ast.parse(Path(controller.__file__).read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SettledTransitShots")
        attrs = [n.func.attr for n in ast.walk(cls) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
        for forbidden in ["write_joint_state_to_sim", "write_root_state_to_sim", "reset"]: self.assertNotIn(forbidden, attrs)

class AllArticulationVelocityTests(unittest.TestCase):
    def test_fast_finger_prevents_release_and_feedback_while_six_arm_axes_are_quiet(self):
        env=controller.SettledTransitShots();env.n=1;env.steps=10;env.dt=1/240
        env.origins=np.zeros((1,3));env.flange_idx=0;env.arms={}
        for name,count in [("Arm10",8),("Arm14",6)]:
            velocity=np.zeros((1,count))
            if name=="Arm10":velocity[0,7]=.02
            env.arms[name]=types.SimpleNamespace(
                joint_names=[f"joint{i+1}" for i in range(6)]+(["finger_left_joint","finger_right_joint"] if count==8 else []),
                data=types.SimpleNamespace(joint_pos=Array(np.zeros((1,count))),
                    joint_vel=Array(velocity),joint_pos_target=Array(np.zeros((1,count))),
                    body_pos_w=Array(np.zeros((1,1,3)))))
        env.tick=lambda:setattr(env,"steps",env.steps+1)
        env.board_poses=lambda:(np.zeros((1,3)),np.array([[0.,0.,1.]]))
        records=[[]]
        metrics,_,_=env._measure([np.eye(4)],[np.eye(4)],0,records)
        m=metrics[0]
        self.assertAlmostEqual(m["joint_speed_rad_s"],.02)
        self.assertFalse(controller.release_ready(m))
        self.assertFalse(controller.quiet(m))
        self.assertFalse(controller.feedback_ready(m))
        # Geometry/FK/lag intentionally remain six arm coordinates, while the
        # release speed gate must include the two separate gripper coordinates.
        self.assertEqual(len(m["arms"]["Arm10"]["position_rad"]),6)
        self.assertEqual(len(m["arms"]["Arm10"]["velocity_rad_s"]),6)
        self.assertEqual(m["measurement_start_step"],10)
        self.assertEqual(m["physics_step"],70)
        self.assertEqual(len(m["all_articulation_joint_velocity"]["Arm10"]),8)
        self.assertEqual(len(m["all_articulation_joint_velocity"]["Arm14"]),6)
        self.assertEqual(m["all_articulation_joint_velocity"]["Arm10"][-1],.02)
        self.assertEqual(m["articulation_joint_names"]["Arm10"][-2:],["finger_left_joint","finger_right_joint"])


class FeedbackSelectionTests(unittest.TestCase):
    def env(self, metrics):
        env = controller.SettledTransitShots(); env.n = 1
        env.active_mask = np.array([True]); env.trial_mode = "train"
        env.arms = {n:Arm() for n in ("Arm10", "Arm14")}
        env.base_cells = [dict(id="cell",distance_m=.4)]; env.spec = {"cells":[]}
        env.home_readback={"path_eligible":[True]}
        env.cfg = dict(arms=dict(base_spacing_m=1.), board=dict(tool_offset_m=.1, mount_yaw_rad=0.))
        env.continuous_links = {n:[] for n in env.arms}; env.ik_cache = {}
        env.approach_endpoints = lambda q: None; env.assert_buckets_fixed = lambda: None
        env._wait_stable = mock.Mock(side_effect=[([m], np.array([[1.,0.,.1]]), np.array([[0.,0.,1.]])) for m in metrics])
        env._ik = mock.Mock(return_value=dict(solved=True, q=[.02]*6))
        return env
    def run_prepare(self, env):
        fit = dict(accepted=True, ik={n:dict(q=[.01]*6) for n in env.arms})
        endpoints = {n:np.eye(4) for n in env.arms}
        prepared=dict(qualification=fit,path=dict(accepted=True,reasons=[]))
        with mock.patch("feasibility_contract.dynamic_cell", side_effect=lambda c,a:c.copy()), mock.patch("expanded_pose_menu.endpoint_targets", return_value=endpoints), mock.patch("joint_feasibility.preflight", return_value=prepared), mock.patch("continuous_feasibility.approximate_path_check", return_value=dict(accepted=True,reasons=[])) as proxy:
            result = env.prepare([{} for _ in range(env.n)])
        return result, proxy
    def test_blocked_home_skips_ik_and_marks_all_arm_targets_for_actual_pose_holding(self):
        env=self.env([metric()]);env.home_readback={"path_eligible":[False]}
        recorded=[]
        env.approach_endpoints=lambda goals:recorded.append(env.preparation_enabled.copy())
        endpoints={n:np.eye(4) for n in env.arms}
        with mock.patch("feasibility_contract.dynamic_cell",side_effect=lambda c,a:c.copy()), mock.patch("expanded_pose_menu.endpoint_targets",return_value=endpoints), mock.patch("joint_feasibility.preflight") as qualify, mock.patch("continuous_feasibility.approximate_path_check") as path:
            result=env.prepare([{}])
        self.assertEqual(result[-1],["common_home_path_proxy_rejected"])
        self.assertEqual(env.qualifications,[dict(accepted=False,reason="common_home_path_proxy_rejected")])
        self.assertEqual(len(recorded),1)
        np.testing.assert_array_equal(recorded[0],[False])
        qualify.assert_not_called();path.assert_not_called();env._ik.assert_not_called()

    def test_final_bucket_separation_uses_recomputed_xy_distance_at_exact_boundary(self):
        for distance,expected in [(np.nextafter(.28,0.),"board_bucket_separation"),(.28,None)]:
            with self.subTest(distance=distance):
                env=self.env([metric()])
                env.base_cells=[dict(id="cell",bucket_xy_m=[0.,0.],distance_m=.4)]
                action=dict(board_x_m=float(distance),board_y_m=0.,radius_m=.28)
                fit=dict(accepted=True,ik={n:dict(q=[.01]*6) for n in env.arms})
                with mock.patch("expanded_pose_menu.endpoint_targets",return_value={n:np.eye(4) for n in env.arms}), mock.patch("joint_feasibility.preflight",return_value=dict(qualification=fit,path=dict(accepted=True,reasons=[]))):
                    result=env.prepare([action])
                self.assertEqual(result[-1],[expected])
                self.assertEqual(env.spec["cells"][0]["distance_m"],distance)
                self.assertEqual(action["radius_m"],.28)

    def test_under_limit_request_preserves_first_rejection_cause(self):
        for reason in ("inactive_split","no_geometry_support","common_home_path_proxy_rejected",
                       "endpoint_ik_unresolved_Arm10","setup_motion_not_quiet"):
            with self.subTest(reason=reason):
                env=self.env([metric(position_error_mm=3.,position_drift_mm=.2)]*4)
                env.base_cells=[dict(id="cell",distance_m=.1)]
                if reason=="inactive_split":env.active_mask[:]=False
                elif reason=="no_geometry_support":env.support_missing=[True]
                elif reason=="common_home_path_proxy_rejected":env.home_readback["path_eligible"]=[False]
                fit=(dict(accepted=False,reason=reason) if reason.startswith("endpoint_ik") else
                     dict(accepted=True,ik={n:dict(q=[.01]*6) for n in env.arms}))
                with mock.patch("feasibility_contract.dynamic_cell",side_effect=lambda c,a:c.copy()), mock.patch("expanded_pose_menu.endpoint_targets",return_value={n:np.eye(4) for n in env.arms}), mock.patch("joint_feasibility.preflight",return_value=dict(qualification=fit,path=dict(accepted=True,reasons=[]))), mock.patch("continuous_feasibility.approximate_path_check",return_value=dict(accepted=True,reasons=[])):
                    result=env.prepare([{}])
                self.assertEqual(result[-1],[reason])
                env._ik.assert_not_called()

    def test_preflight_consumes_actual_state_and_path_decline_holds_both_arms(self):
        env=self.env([metric()])
        env.arms["Arm10"].data.joint_pos=Array([[.012]*6])
        env.arms["Arm14"].data.joint_pos=Array([[.023]*6])
        fit=dict(accepted=True,reason=None,ik={n:dict(q=[.1]*6) for n in env.arms})
        prepared=dict(qualification=fit,path=dict(accepted=False,reasons=["inter_arm_proxy_overlap"]),
                      accepted=False,release_authorized=False)
        captured=[]
        env.approach_endpoints=lambda q:captured.append(env.preparation_enabled.copy())
        action={"id":"unchanged"}
        with mock.patch("feasibility_contract.dynamic_cell",side_effect=lambda c,a:c.copy()), \
             mock.patch("expanded_pose_menu.endpoint_targets",return_value={n:np.eye(4) for n in env.arms}), \
             mock.patch("joint_feasibility.preflight",return_value=prepared) as check:
            result=env.prepare([action])
        self.assertEqual(result[-1],["inter_arm_proxy_overlap"])
        self.assertIs(check.call_args.args[2],action)
        np.testing.assert_array_equal(check.call_args.args[4]["Arm10"],[.012]*6)
        np.testing.assert_array_equal(check.call_args.args[4]["Arm14"],[.023]*6)
        np.testing.assert_array_equal(captured[0],[False])
        self.assertIs(env.preflight_records[0],prepared)
        env._ik.assert_not_called()

    def test_preflight_admission_does_not_bypass_measured_release_guards(self):
        env=self.env([metric(position_error_mm=3.,position_drift_mm=.2)]*4)
        result,_=self.run_prepare(env)
        self.assertEqual(result[-1],["setup_motion_not_quiet"])
        self.assertEqual(env._wait_stable.call_count,4)
        self.assertTrue(env.preparation_enabled[0])
        self.assertIsNotNone(env.preflight_records[0])

    def test_moving_or_grossly_lagging_arm_does_not_receive_bias_correction(self):
        for m, reason in [(metric(position_error_mm=3.,position_drift_mm=.2), "setup_motion_not_quiet"), (metric(position_error_mm=3., arms={n:dict(position_rad=[0.]*6,target_rad=[.03]*6) for n in ("Arm10","Arm14")}), "setup_transit_incomplete")]:
            with self.subTest(reason=reason):
                env = self.env([m]*4); result, _ = self.run_prepare(env)
                self.assertEqual(result[-1], [reason]); env._ik.assert_not_called()
    def test_pending_motion_recovers_during_other_cells_feedback_without_early_bias(self):
        env=self.env([metric()]);env.n=2;env.active_mask=np.array([True,True])
        env.base_cells=[dict(id="pending",distance_m=.4),dict(id="other",distance_m=.4)]
        env.home_readback={"path_eligible":[True,True]}
        for a in env.arms.values():
            a.data.joint_pos=Array(np.zeros((2,6)))
            a.data.default_joint_pos=Array(np.zeros((2,6)))
        windows=[
            [metric(position_error_mm=3.,position_drift_mm=.2),metric(position_error_mm=3.)],
            [metric(position_error_mm=3.),metric()],
            [metric(),metric()],
        ]
        call=[0]
        def wait(*args):
            i=call[0];call[0]+=1
            # Iteration0 corrects only the other quiet cell. Pending cell must
            # survive, then receive its first correction after it becomes quiet.
            self.assertEqual(env._ik.call_count,[0,1,2][i])
            return windows[i],np.tile([1.,0.,.1],(2,1)),np.tile([0.,0.,1.],(2,1))
        env._wait_stable=mock.Mock(side_effect=wait)
        result,_=self.run_prepare(env)
        self.assertEqual(result[-1],[None,None])
        self.assertEqual([(x["cell_id"],x["iteration"]) for x in env.correction_history],[("other",0),("pending",1)])
        self.assertEqual(env._wait_stable.call_count,3)

    def test_active_tool_targets_use_configured_default_not_measured_home_drift(self):
        env=self.env([metric()])
        a=env.arms["Arm10"]
        a.joint_names=a.joint_names+["finger_left_joint","finger_right_joint"]
        a.data.joint_pos=Array([[0.]*6+[.021,.022]])
        a.data.default_joint_pos=Array([[0.]*6+[.025,.025]])
        captured=[]
        env.approach_endpoints=lambda goals:captured.append(goals["Arm10"].value.copy())
        result,_=self.run_prepare(env)
        self.assertEqual(result[-1],[None])
        self.assertEqual(len(captured),1)
        np.testing.assert_allclose(captured[0][0,:6],[.01]*6)
        np.testing.assert_allclose(captured[0][0,6:],[.025,.025])
        np.testing.assert_allclose(a.data.joint_pos.value[0,6:],[.021,.022])

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
                    default_joint_pos=torch.tensor([[.1]*6,[.2]*6]),
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
    def test_eight_dof_active_tool_defaults_and_inactive_entire_pose_are_preserved(self):
        env=self.env();a=env.arms["Arm10"]
        a.joint_names=a.joint_names+["finger_left_joint","finger_right_joint"]
        a.data.joint_pos=torch.tensor([[.1]*6+[.021,.022],[.2]*6+[.031,.032]])
        a.data.joint_pos_target=a.data.joint_pos.clone()
        a.data.joint_vel=torch.zeros((2,8))
        a.data.default_joint_pos=torch.tensor([[0.]*6+[.025,.025],[0.]*6+[.025,.025]])
        for key,extra in [("joint_effort_limits",30.),("joint_stiffness",1000.),("joint_damping",50.)]:
            setattr(a.data,key,torch.cat([getattr(a.data,key),torch.full((2,2),extra)],dim=1))
        targets={n:torch.tensor([[.4]*6,[8.]*6]) for n in env.arms}
        targets["Arm10"]=torch.tensor([[.4]*6+[.025,.025],[8.]*6+[.025,.025]])
        expected_inactive=a.data.joint_pos[1].clone()
        env.approach_endpoints(targets)
        self.assertTrue(all(torch.allclose(x[1],expected_inactive) for x in a.commands))
        torch.testing.assert_close(a.commands[-1][0,6:],torch.tensor([.025,.025]))
        a.data.joint_vel[0,7]=-.02
        _,speeds=env._joint_status(targets)
        self.assertAlmostEqual(float(speeds["Arm10"][0]),.02)

        for key,expected in [("joint_effort_limits",30.),("joint_stiffness",1000.),("joint_damping",50.)]:
            torch.testing.assert_close(getattr(a.data,key)[:,6:],torch.full((2,2),expected))

    def test_tick_failure_restores_original_drives_before_propagation(self):
        env=self.env(raises=True)
        with self.assertRaisesRegex(RuntimeError,"injected simulator failure"):
            env.approach_endpoints({n:torch.ones((2,6)) for n in env.arms})
        self.assert_original(env)

if __name__ == "__main__": unittest.main()
