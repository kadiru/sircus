"""Mock-only regression checks for live-transit admission and target holding."""

import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ISAAC = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ISAAC))
import continuous_pilot as pilot
import continuous_policy as policy


class CPUView:
    def __init__(self,value): self.value=np.asarray(value)
    def __getitem__(self,key): return CPUView(self.value[key])
    def cpu(self): return self
    def numpy(self): return self.value


class AdmissionGuardTests(unittest.TestCase):
    def test_rejected_path_and_inactive_layout_are_disabled_before_parent_prepare(self):
        env = pilot.ContinuousShots()
        env.home_readback={"path_eligible":[True,True,True]}
        env.n=3; env.active_mask=np.array([True,True,False]); env.trial_mode="train"
        env.home_readback=dict(path_eligible=[True]*3)
        env.arms={n:types.SimpleNamespace(data=types.SimpleNamespace(joint_pos=CPUView(np.zeros((3,6)))))
                  for n in ("Arm10","Arm14")}
        env.cfg={}; env.base_cells=[dict(id=str(i)) for i in range(3)]
        env.continuous_links={}; env.ik_cache={}
        accepted=dict(accepted=True,ik={n:dict(q=[.1]*6) for n in env.arms})
        rejected=dict(accepted=False,reason="endpoint_ik_unresolved_Arm10")
        parent_calls=[]

        class Parent:
            @staticmethod
            def prepare(current,actions):
                parent_calls.append(current.preparation_enabled.copy())
                return np.zeros((3,3)),np.zeros((3,3)),[[],[],[]],["parent_idle"]*3

        with mock.patch.dict(sys.modules,{"feasibility_runtime":types.SimpleNamespace(FeasibilityShots=Parent)}), \
             mock.patch.object(pilot,"qualify_action",side_effect=[accepted,rejected]), \
             mock.patch("expanded_pose_menu.endpoint_targets",return_value={n:np.eye(4) for n in env.arms}), \
             mock.patch("continuous_feasibility.approximate_path_check",return_value=dict(accepted=False,reasons=["inter_arm_proxy_overlap"])):
            _,_,_,invalid=env.prepare([{}, {}, {}])
        self.assertEqual(len(parent_calls),1)
        np.testing.assert_array_equal(parent_calls[0],[False,False,False])
        self.assertEqual(invalid,["inter_arm_proxy_overlap","endpoint_ik_unresolved_Arm10","inactive_split"])


@unittest.skipIf(policy.torch is None,"PyTorch CPU environment is unavailable")
class DriveGuardTests(unittest.TestCase):
    def test_disabled_targets_hold_measured_start_while_eligible_targets_use_live_drives(self):
        torch=policy.torch

        class Arm:
            joint_names=[f"joint{i+1}" for i in range(6)]
            def __init__(self):
                self.data=types.SimpleNamespace(joint_pos=torch.tensor([[.1]*6,[.2]*6]),
                    joint_effort_limits=torch.ones((2,6)),joint_stiffness=torch.ones((2,6)),
                    joint_damping=torch.ones((2,6)))
                self.commands=[]
            def write_joint_effort_limit_to_sim(self,*args): pass
            def write_joint_stiffness_to_sim(self,*args): pass
            def write_joint_damping_to_sim(self,*args): pass
            def set_joint_position_target(self,goal):
                self.commands.append(goal.clone()); self.data.joint_pos=goal.clone()

        env=pilot.ContinuousShots(); env.n=2; env.dt=1/240
        env.preparation_enabled=np.array([True,False])
        env.arms={n:Arm() for n in ("Arm10","Arm14")}
        env.transit_limits=torch.ones(6)
        env.set_balls=lambda *args:None; env.tick=lambda:None; env.assert_buckets_fixed=lambda:None
        targets={n:torch.tensor([[.4]*6,[8.]*6]) for n in env.arms}
        env.approach_endpoints(targets)
        for arm in env.arms.values():
            self.assertGreater(len(arm.commands),1)
            self.assertTrue(all(torch.allclose(command[1],torch.tensor([.2]*6)) for command in arm.commands))
            self.assertTrue(torch.allclose(arm.commands[-1][0],torch.tensor([.4]*6)))
        self.assertEqual(env.transit_ready_after_drive,[True,True])


if __name__ == "__main__": unittest.main()
