"""Canonical grid and balanced, explicitly historical imitation supervision."""
import copy
from collections import Counter
import unittest
import numpy as np
from sircus.mjlab import fixed35_imitation_contract as C
from sircus.mjlab.tests.test_corrected_imitation_contract import fixture


def yaw_group():
    _, rows = fixture()
    for r in rows:
        r.update(adapter_directory='native-yaw',diagnostic_robot_contact_steps=0)
    return rows


class ContractTests(unittest.TestCase):
    def test_canonical35_are_exact_fixed_points_with_no_jitter(self):
        first=C.fixed_grid();self.assertEqual(first,C.fixed_grid())
        expected=[[float(x),float(y)] for y in np.linspace(-.65,.25,5) for x in np.linspace(.1,1.24,7)]
        self.assertEqual([r['bucket_xy_m'] for r in first],expected)
        self.assertEqual(len(first),35)
        self.assertEqual(Counter(tuple(r['grid_cell']) for r in first),Counter({(x,y):1 for y in range(5) for x in range(7)}))
        self.assertTrue(all(r['evaluation_scope']=='historical_grid_coverage' for r in first))

    def test_one_recipe_per_context_not_three_repeat_labels(self):
        group=yaw_group();before=copy.deepcopy(group);teacher=C.yaw_teacher(group,'far')
        self.assertEqual(group,before);self.assertEqual(teacher['role'],'train')
        self.assertEqual(teacher['id'],'fixed35-yaw-teacher-far')
        self.assertEqual(len(teacher['correction_repeat_ids']),3)
        np.testing.assert_allclose(teacher['normalized_residual'],[.25,-.25,.2,.125,1/12])
        self.assertEqual(teacher['source_correction_record_sha256'],C.digest(group[0]))

    def test_missing_failed_or_mutated_repeat_refused(self):
        for mutate in (lambda r:r.pop(),lambda r:r[2].update(route_success=False),
                       lambda r:r[1].update(q14=[.5]*6),lambda r:r[1].update(adapter_directory='native'),
                       lambda r:r[1].update(diagnostic_robot_contact_steps=1)):
            group=yaw_group();mutate(group)
            with self.assertRaises(ValueError):C.yaw_teacher(group,'far')

    def test_supervision_outside_support_is_never_clipped(self):
        group=yaw_group()
        for r in group:r['values'][0]=10.
        with self.assertRaisesRegex(ValueError,'support'):C.yaw_teacher(group,'far')

    def test_scope_budget_and_orientation_are_explicit(self):
        p=C.protocol()
        self.assertEqual((p['test_layouts'],p['test_requests'],p['control_requests'],p['maximum_requests']),(35,70,18,88))
        self.assertEqual((p['base_teacher_count'],p['new_teacher_count'],p['teacher_count']),(97,2,99))
        self.assertEqual((p['actor_steps'],p['critic_steps'],p['substeps']),(800,0,64))
        self.assertFalse(p['bucket_perturbations']);self.assertFalse(p['noise'])
        self.assertIn('not held out',p['evaluation_scope']);self.assertIn('historical',p['teacher_semantics'])
        self.assertEqual(C.METHODS,('fixed35_bc','current_bc'))

if __name__=='__main__':unittest.main()
