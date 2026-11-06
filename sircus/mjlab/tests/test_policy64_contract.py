"""Fixed split, attempt-budget and TRAIN-only teacher-selection contracts."""

import copy
import json
from pathlib import Path
import unittest

import numpy as np

from sircus.mjlab import policy64_contract as contract


def shot(identity, context='context-a', *, quality=1, distance=.02, role='train', **changes):
    row = dict(id=identity, context_id=context, role=role, bucket_xy_m=[.5, -.2],
               native_release=True, route_success=True, route_quality=quality,
               invalid=None, invalid_physics=False, board_bounce=True,
               closest_opening_distance_m=distance, external_contact_steps=0,
               max_retention_s=1., single_bounce_success=True)
    row.update(changes)
    return row


class SplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[3]
        cls.parent = json.loads((root/'baselines/mjlab-calibrated-benchmark-20261104/splits.json').read_text())
        cls.historical = json.loads((root/'baselines/mjlab-sweep-rl-benchmark-20261103/splits.json').read_text())

    def test_frozen_35_12_50_are_disjoint_and_exclude_all_historical_coordinates_by_5mm(self):
        parent, historical = copy.deepcopy(self.parent), copy.deepcopy(self.historical)
        pools, excluded = contract.make_splits(parent, historical)
        self.assertEqual([len(pools[k]) for k in ('train','validation','test')], [35,12,50])
        self.assertEqual((pools, excluded), contract.make_splits(parent, historical))
        self.assertEqual(parent, self.parent); self.assertEqual(historical, self.historical)
        self.assertEqual([(r['id'],r['bucket_xy_m']) for r in pools['train']],
                         [(r['id'],r['bucket_xy_m']) for r in parent['train']])
        expected = {tuple(r['bucket_xy_m']) for source in (parent,historical)
                    for role in ('train','validation','test') for r in source[role]}
        self.assertEqual({tuple(p) for p in excluded}, expected)
        fresh = np.asarray([r['bucket_xy_m'] for role in ('validation','test') for r in pools[role]])
        previous = np.asarray(excluded)
        self.assertGreaterEqual(float(np.linalg.norm(fresh[:,None]-previous[None,:],axis=-1).min()), .005)
        pairwise = np.linalg.norm(fresh[:,None]-fresh[None,:],axis=-1)
        np.fill_diagonal(pairwise, np.inf)
        self.assertGreaterEqual(float(pairwise.min()), .005)
        all_rows = sum((pools[k] for k in ('train','validation','test')), [])
        self.assertEqual(len({r['id'] for r in all_rows}),97)
        self.assertEqual(len({tuple(r['bucket_xy_m']) for r in all_rows}),97)
        for role in ('train','validation','test'):
            self.assertTrue(all(r['split']==role for r in pools[role]))
        self.assertTrue(np.all(fresh >= [.1,-.65])); self.assertTrue(np.all(fresh <= [1.24,.25]))

    def test_seeded_1024_schedule_has_balanced_permutations_and_exact_attempt_budget(self):
        result = contract.schedule()
        self.assertEqual(result,contract.schedule()); self.assertEqual(len(result),1024)
        self.assertTrue(all(type(i) is int and 0<=i<35 for i in result))
        for start in range(0,1015,35):
            self.assertEqual(sorted(result[start:start+35]),list(range(35)))
        counts = np.bincount(result,minlength=35)
        self.assertEqual(sorted(set(counts.tolist())),[29,30])
        expected_first = np.random.default_rng(20261511).permutation(35).tolist()
        self.assertEqual(result[:35],expected_first)
        p=contract.protocol()
        self.assertEqual(p['train_requests'],1024); self.assertEqual(p['updates']*p['batch_size'],1024)
        self.assertEqual(p['maximum_fresh_requests'],1284)
        self.assertEqual(sum(p[k] for k in ('train_requests','validation_requests','test_requests','control_requests')),1284)
        self.assertEqual(p['substeps'],64); self.assertFalse(p['noise'])
        self.assertEqual(p['milestones'],[0,256,512,1024])
        self.assertEqual(p['test_methods'],['prior','nearest','bc','ppo'])


class TeacherTests(unittest.TestCase):
    def test_teacher_prefers_clean_then_calibrated_reward_then_stable_id(self):
        rows = [shot('rim-high-reward',quality=1,distance=0.),
                shot('clean-z',quality=2,distance=1.),
                shot('clean-low',context='context-b',quality=2,distance=.1),
                shot('clean-z-tie',context='context-b',quality=2,distance=.01),
                shot('clean-a-tie',context='context-b',quality=2,distance=.01)]
        # The clean preference is deliberately stronger than this shaped-reward difference.
        self.assertGreater(contract.reward(rows[0]),contract.reward(rows[1]))
        before=copy.deepcopy(rows)
        self.assertEqual([r['id'] for r in contract.select_teachers(rows)],['clean-z','clean-a-tie'])
        self.assertEqual(rows,before)
        self.assertEqual([r['id'] for r in contract.select_teachers(list(reversed(rows)))],['clean-z','clean-a-tie'])

    def test_only_actual_valid_captures_can_teach_and_held_out_labels_are_rejected(self):
        rows=[shot('valid'),shot('not-released',context='b',native_release=False),
              shot('miss',context='c',route_success=False),shot('geometry-reject',context='d',invalid='endpoint'),
              shot('physics-reject',context='e',invalid_physics=True)]
        self.assertEqual([r['id'] for r in contract.select_teachers(rows)],['valid'])
        self.assertEqual(contract.select_teachers([]),[])
        for role in ('validation','test','control'):
            with self.subTest(role=role),self.assertRaisesRegex(ValueError,'Held-out'):
                contract.select_teachers(rows+[shot('leak',role=role,route_success=False)])

    def test_nearest_teacher_uses_context_then_teacher_id_for_equal_distance_not_reward(self):
        context=dict(id='new-test',bucket_xy_m=[.5,-.2])
        rows=[shot('a',context='z-context',distance=0.),shot('z',context='a-context',distance=.1),
              shot('a',context='a-context',distance=1.)]
        rows[0]['bucket_xy_m']=[.4,-.2]
        rows[1]['bucket_xy_m']=rows[2]['bucket_xy_m']=[.6,-.2]
        before=copy.deepcopy(rows)
        self.assertEqual(contract.nearest_teacher(context,rows)['context_id'],'a-context')
        self.assertEqual(contract.nearest_teacher(context,rows)['id'],'a')
        self.assertEqual(contract.nearest_teacher(context,list(reversed(rows)))['id'],'a')
        closer=shot('closer',context='z-context'); closer['bucket_xy_m']=[.5,-.19]
        self.assertEqual(contract.nearest_teacher(context,rows+[closer])['id'],'closer')
        self.assertEqual(rows,before)


if __name__=='__main__': unittest.main()
