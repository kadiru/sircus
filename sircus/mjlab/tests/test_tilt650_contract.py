"""Synthetic protocol tests; no native-flight claims or GPU access."""
import copy
import unittest

from sircus.mjlab import tilt650_contract as p, xy650_contract as xy


def record(step, index, quality=1):
    return dict(id=f'r-{step}-{index}', pair_id=f'pair-{index}', substeps=step,
                route_success=quality > 0, route_quality=quality,
                native_release=True, invalid=None, invalid_physics=False,
                scalar_score_audit_passed=True, max_retention_s=1.)


def positions(rows, drift=0.):
    return {r['id']: {round(.35*240*r['substeps']): [drift if r['substeps']==128 else 0., 0., 0.]} for r in rows}


class TiltContractTests(unittest.TestCase):
    def test_literal_budget_and_no_adaptive_phase(self):
        settings = p.settings()
        self.assertEqual(settings['max_requests'], 54)
        self.assertEqual(settings['verification_requests'], 5*3*2)
        self.assertEqual(settings['control_requests'], 2*3*2*2)
        self.assertEqual(settings['source_ids'], list(p.SOURCE_IDS))

    def test_rim_parent_is_not_falsely_promoted_to_clean(self):
        rows = [record(s, i) for s in p.STEPS for i in range(3)]
        result = p.qualification(rows, positions(rows), controls_passed=True, parent_source_capture=True)
        self.assertEqual(result['status'], 'capture_only')
        self.assertTrue(result['capture_qualified'])
        self.assertFalse(result['qualified'])
        self.assertFalse(result['clean_qualified'])

    def test_fresh_clean_rows_are_evaluated_without_relabeling_parent(self):
        rows = [record(s, i, 2) for s in p.STEPS for i in range(3)]
        result = p.qualification(rows, positions(rows), controls_passed=True, parent_source_capture=True)
        self.assertTrue(result['qualified'])
        self.assertTrue(result['clean_qualified'])
        self.assertTrue(result['capture_qualified'])

    def test_resolution_failure_and_control_failure_cannot_qualify(self):
        rows = [record(s, i, 2 if s==64 else 0) for s in p.STEPS for i in range(3)]
        self.assertFalse(p.qualification(rows, positions(rows), controls_passed=True, parent_source_capture=True)['capture_qualified'])
        for key in ('controls_passed', 'parent_source_capture'):
            args=dict(controls_passed=True, parent_source_capture=True);args[key]=False
            good=[record(s,i,2) for s in p.STEPS for i in range(3)]
            self.assertFalse(p.qualification(good,positions(good),**args)['qualified'])

    def test_missing_physics_or_oracle_and_excess_drift_fail(self):
        rows = [record(s, i, 2) for s in p.STEPS for i in range(3)]
        for key, value in [('invalid_physics',True),('scalar_score_audit_passed',False),('native_release',False)]:
            data=copy.deepcopy(rows);data[0][key]=value
            self.assertFalse(p.qualification(data,positions(data),controls_passed=True,parent_source_capture=True)['qualified'])
        self.assertFalse(p.qualification(rows,positions(rows,.00501),controls_passed=True,parent_source_capture=True)['qualified'])
        self.assertEqual(p.qualification(rows[:3],positions(rows),controls_passed=True,parent_source_capture=True)['status'],'pending')

    def test_exact_frozen_joint_lineage_and_plan(self):
        sources=[]
        for i,ident in enumerate(p.SOURCE_IDS):
            sources.append(dict(id=ident,values=[.47,-.21,.38,30.,.41],bucket_xy_m=[-.03,-.63],
                                translation_m=[0.,.06],q10=[i/10]*6,q14=[i/20]*6,
                                query_fingerprint='a'*64,route_quality=1))
        controls=[dict(id=n,role=r,phase='controls',query_kind='historical_control',translation_m=[0.,0.],
                       source_record_id=n,source_record_sha256='b'*64,values=[.47,-.21,.38,28.,.4],bucket_xy_m=[-.03,-.63],q10=[0.]*6,q14=[.1]*6)
                  for n,r in [('positive-control','positive_control'),('negative-control','negative_control')]]
        groups=p.groups(sources,controls,'c'*64)
        for i,g in enumerate(groups):g.update(eligible=True,geometry_row=i)
        plan=(xy.plan(groups[:2],'controls_pre',p.STEPS,3)+xy.plan(groups[2:],'verification',p.STEPS,3)
              +xy.plan(groups[:2],'controls_post',p.STEPS,3))
        self.assertEqual(len(plan),54);self.assertEqual(len({r['id'] for r in plan}),54)
        for g,src in zip(groups[2:],sources):
            self.assertTrue(g['freeze_joints']);self.assertEqual(g['q10'],src['q10']);self.assertEqual(g['q14'],src['q14'])
            self.assertEqual(g['source_record_id'],src['id']);self.assertEqual(g['parent_route_quality'],1)
            self.assertEqual(g['source_query_fingerprint'],src['query_fingerprint'])


if __name__=='__main__':
    unittest.main()
