import copy
import math
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from sircus.mjlab import fixed35_broad_verify as V
from sircus.mjlab.benchmark_verify import EvidenceError


def chain():
    origins = np.repeat(np.eye(4)[None], 6, axis=0)
    origins[0, :3, 3] = [0, 0, .364]
    origins[2, :3, 3] = [.390, 0, 0]
    origins[3, :3, 3] = [.150, .426, 0]
    origins[5, :3, 3] = [0, -.09, 0]
    origins[5, :3, :3] = Rotation.from_euler('x', 1.5708).as_matrix()
    axes = np.tile([0., 0., 1.], (6, 1))
    return origins, axes, np.full(6, -math.pi), np.full(6, math.pi)


SCENE = dict(board=dict(mount_yaw_rad=0., tool_offset_m=.1),
             arms=dict(base_spacing_m=1.33731), ball=dict(radius_m=.0375))


class ReachTests(unittest.TestCase):
    def test_triangle_bound_matches_literal_link_lengths(self):
        target = np.eye(4)
        target[:3, 3] = [.845410611892873, 0, .454]
        result = V.chain_reach(chain(), target)
        self.assertAlmostEqual(result['radius_m'], .8416370223974114, places=14)
        self.assertAlmostEqual(result['transverse_m'], 3.3058845929745396e-7, places=15)
        self.assertAlmostEqual(result['tolerance_allowance_m'], .00013599999975975714, places=14)
        self.assertFalse(result['passed'])

    def test_pose_tolerance_is_included_not_used_to_relax_native_gate(self):
        target = np.eye(4)
        radius = .390+math.hypot(.150, .426)
        target[:3, 3] = [radius+.00010, 0, .454]
        self.assertTrue(V.chain_reach(chain(), target)['passed'])
        target[0, 3] = radius+.00020
        self.assertFalse(V.chain_reach(chain(), target)['passed'])

    def test_inside_bound_does_not_claim_joint_limit_feasibility(self):
        c = list(chain())
        c[2] = np.zeros(6)
        c[3] = np.ones(6)*.0001
        target = np.eye(4)
        target[:3, 3] = [.7, 0, .454]
        result = V.chain_reach(tuple(c), target)
        self.assertTrue(result['passed'])
        self.assertNotIn('ik_passed', result)

    def test_conditional_gap_shrinks_for_farther_and_higher_dropper(self):
        c = (chain(), chain())
        low = V.independent_gap_interval(c, SCENE, [1.24, .25], [.8, 0, .2], 28)
        high = V.independent_gap_interval(c, SCENE, [1.24, .25], [.8, 0, .4], 28)
        nearer = V.independent_gap_interval(c, SCENE, [1.24, .25], [.7, 0, .2], 28)
        self.assertEqual(low[0], .05)
        self.assertLess(low[1], .204)
        self.assertGreater(low[1], .202)
        self.assertIsNone(high)
        self.assertGreater(nearer[1], low[1])

    def test_negative_radicand_cannot_produce_an_allowed_gap(self):
        self.assertIsNone(V.independent_gap_interval((chain(), chain()), SCENE,
            [1.24, .25], [1.0, 0., .12], 28))

    def test_terminal_axis_assumption_is_explicitly_verified(self):
        c = list(chain())
        c[1] = c[1].copy()
        c[1][5] = [1, 0, 0]
        with self.assertRaises(EvidenceError):
            V.chain_reach(tuple(c), np.eye(4))


class ProtocolTests(unittest.TestCase):
    def test_independent_plan_matches_without_using_producer_builder(self):
        from unittest.mock import patch
        from sircus.mjlab import fixed35_broad_contract as C
        from sircus.mjlab.tests.test_fixed35_broad_contract import sources
        source=sources();before=copy.deepcopy(source)
        cap=lambda *args,**kw:.12
        reach=lambda *args:{'passed':True,'fixture':True}
        expected=C.make_plan(source,{},'assets',cap_fn=cap,reach_fn=reach,per_cell=24,raw_limit=200)
        with patch.object(C,'make_plan',side_effect=AssertionError('producer planner forbidden')):
            actual=V.independent_plan(source,{},'assets',cap_fn=cap,reach_fn=reach,per_cell=24,raw_limit=200)
        self.assertEqual(actual,expected);self.assertEqual(source,before)
        self.assertEqual(len(actual['rows']),72)
        self.assertEqual({r['proposal_family'] for r in actual['rows']},set(V.FAMILIES))

    def test_independent_plan_records_finite_exhaustion_and_reach_declines(self):
        from sircus.mjlab.tests.test_fixed35_broad_contract import sources
        out=V.independent_plan(sources(),{},'assets',cap_fn=lambda *a,**kw:None,
            reach_fn=lambda *a:None,per_cell=24,raw_limit=12)
        self.assertEqual(out['rows'],[])
        self.assertTrue(all(r['raw_considered']==12 and not r['proposal_limit_reached'] for r in out['cells']))

    def test_structural_comparison_refuses_new_keys_and_changed_membership(self):
        expected=dict(value=.2,passed=True,hash='a'*64)
        V.check_tree(dict(value=.2+1e-14,passed=True,hash='a'*64),expected,V.AuditChecks(),'roundoff')
        for actual in (dict(expected,passed=False),dict(expected,extra=True),dict(expected,hash='b'*64),dict(expected,value=.20001)):
            with self.assertRaises(EvidenceError):V.check_tree(actual,expected,V.AuditChecks(),'tamper')

    def test_literal_native_budgets_and_no_training(self):
        from sircus.mjlab import fixed35_broad_contract as C
        V.protocol_checks(C.protocol(),V.AuditChecks())
        p=C.protocol();p['substeps']=128
        with self.assertRaises(EvidenceError):V.protocol_checks(p,V.AuditChecks())


class BranchTests(unittest.TestCase):
    def test_wrapped_q14_clones_do_not_consume_distinct_slots(self):
        q=[[0.]*6,[2*math.pi]+[0.]*5,[.02]+[0.]*5,[.2]+[0.]*5]
        p=dict(q=q,valid=[True]*4,residual=[0.,.1,.2,.3])
        self.assertEqual(V.distinct_indices(p),[0,3])

    def test_proximal_branch_precedes_wrist_only_and_dropper_order_is_separate(self):
        first=dict(q=[[0.]*6,[.1]+[0.]*5],valid=[True,True],residual=[0.,.1])
        second=dict(q=[[0.]*6,[0.]*5+[2.],[.2]+[0.]*5],valid=[True]*3,residual=[0.,.1,.2])
        drop,board=V.branch_orders([first,second],dict(q10=[0.]*6,q14=[0.]*6))
        self.assertEqual(drop,[0,1]);self.assertEqual(board[0],2)
        self.assertEqual(set(board),{0,1,2})

    def test_connection_binds_complete_pool_and_slot_not_only_values(self):
        from sircus.mjlab.benchmark_verify import digest
        pool=dict(pool_key='pool',bucket_xy_m=[1.,0.],values=[.6,0.,.2,28.,.1],endpoint_attempts={},path_attempts=[{'passed':True}],timing={},release_behavior={},behavior_key='behavior',
            admitted=[dict(q10=[.1]*6,q14=[.2]*6,endpoint={},path_attempt_index=0)])
        anchor=dict(file='anchor.json',sha256='a'*64,anchor_values=[0]*5,anchor_derivation={})
        home=dict(sha256='h'*64)
        first=V.expected_connection(pool,anchor,home,'pool.json','1'*64,0)
        second=V.expected_connection(pool,anchor,home,'pool.json','2'*64,0)
        self.assertNotEqual(first['broad_key'],second['broad_key'])
        self.assertEqual(first['q10'],pool['admitted'][0]['q10'])
        self.assertEqual(first['broad_key'],digest(dict(version=V.BROAD_VERSION,pool_key='pool',broad_pool_sha256='1'*64,broad_branch_slot=0)))


if __name__ == '__main__':unittest.main()
