"""CPU finite-protocol checks; fixture outcomes are synthetic only."""
import copy
import unittest

import numpy as np

from sircus.mjlab import clean650_contract as p


def row(i=0,quality=2,step=64):
    return dict(id=f'source-{i}',native_release=True,invalid=None,invalid_physics=False,route_success=quality>0,
        route_quality=quality,whole_ball_entry=quality>0,external_contact_steps=0,minimum_mouth_clearance_mm=2. if quality==2 else 0.,
        max_retention_s=1.,closest_opening_distance_m=.02,values=[.47,-.27,.38,28.,.3868],bucket_xy_m=[-.03,-.69],q10=[0.]*6,q14=[0.]*6,
        substeps=step,role='verification',repeat_index=i,pair_id=f'pair-{i}',scalar_score_audit_passed=True,rim_band_contact=quality==1)


class CleanContractTests(unittest.TestCase):
    def test_exact_360_grid_plus_original_and_finite_conditional_budget(self):
        seed=dict(row(),source_record_id='old650',source_record_sha256='a'*64)
        grid=p.coarse(seed)
        self.assertEqual(len(grid),361);self.assertEqual(len({g['id'] for g in grid}),361)
        self.assertEqual(grid[-1]['values'],seed['values'])
        refined=p.refinement([row(i) for i in range(3)])
        self.assertEqual(len(refined),81)
        self.assertEqual(sum([12,361,len(refined),5*6,12]),p.MAX_REQUESTS)
        self.assertEqual(refined[13]['values'],seed['values'])
        self.assertNotIn('q10',refined[0])

    def test_selection_prefers_clean_then_capture_and_retains_invalid_in_stats(self):
        data=[row(0,0),row(1,1),row(2,2),dict(row(3,2),invalid_physics=True),dict(row(4,2),native_release=False)]
        self.assertEqual([r['id'] for r in p.ranked(data,5)],['source-2','source-1','source-0'])
        totals=p.statistics(data);self.assertEqual(totals['attempts'],5);self.assertEqual(totals['clean'],1)
        self.assertEqual(totals['captures'],2);self.assertEqual(totals['rim_assisted'],1);self.assertEqual(totals['geometry_rejections'],1)

    def test_verification_preserves_source_joints_and_strict_clean_qualification(self):
        source=row();group=p.verification([source])[0]
        self.assertTrue(group['freeze_joints']);self.assertEqual(group['q14'],source['q14'])
        rows=[];positions={}
        for step in (64,128):
            for i in range(3):
                r=row(i,2,step);r['id']=f'{step}-{i}';rows.append(r)
                positions[r['id']]=np.zeros((round(.350*240*step)+1,3))
        self.assertTrue(p.qualification(rows,positions,controls_passed=True,source_clean=True)['qualified'])
        self.assertFalse(p.qualification(rows,positions,controls_passed=False,source_clean=True)['qualified'])
        self.assertFalse(p.qualification(rows,positions,controls_passed=True,source_clean=False)['qualified'])
        changed=copy.deepcopy(rows);changed[-1]['route_quality']=1
        self.assertFalse(p.qualification(changed,positions,controls_passed=True,source_clean=True)['qualified'])

    def test_control_requires_clean_positive_at_both_steps_and_actual_negative_misses(self):
        rows=[]
        for step in (64,128):
            for role,quality in [('positive_control',2),('negative_control',0)]:
                for i in range(3):rows.append(dict(row(i,quality,step),role=role))
        self.assertEqual(p.control_gate(rows)['status'],'passed')
        rows[0]['route_quality']=1
        self.assertFalse(p.control_gate(rows)['positive_control'])
        rows[0]['route_quality']=2;rows[-1]['native_release']=False
        self.assertEqual(p.control_gate(rows)['status'],'failed')

    def test_verification_uses_distinct_targets_without_rounding_actions(self):
        a=row(0);b=row(1);b['values'][2]+=1e-16;c=row(2);c['values'][4]+=.02
        chosen=p.verification_selection([a,b,c],5)
        self.assertEqual([r['id'] for r in chosen],['source-0','source-2'])
        self.assertIs(chosen[1],c)


if __name__=='__main__':unittest.main()
