"""Synthetic bounded protocol/identity tests; no GPU or native evidence."""
import copy
import unittest

from sircus.mjlab import impact650_contract as p
from sircus.mjlab.tests.test_tilt650_contract import record,positions


def sources():
    result=[]
    for i,(ident,tilt,gap) in enumerate(zip(p.SOURCE_IDS,[27.5,30.,30.],[.3918,.41,.39])):
        result.append(dict(id=ident,values=[.47,-.21,.38,tilt,gap],bucket_xy_m=[-.03,-.63],
            translation_m=[0.,.06],q10=[i/10]*6,q14=[i/20]*6,query_fingerprint='a'*64,
            native_release=True,invalid=None,invalid_physics=False,route_success=True,route_quality=1))
    return result


class ImpactContractTests(unittest.TestCase):
    def test_grid_exact135_and_inclusive189_budget(self):
        groups=p.coarse(sources());self.assertEqual(len(groups),135)
        self.assertEqual(len({g['id'] for g in groups}),135)
        self.assertEqual(len({p.target_key(g) for g in groups}),135)
        self.assertEqual(p.settings()['max_requests'],135+24+5*3*2)
        self.assertLessEqual(p.settings()['max_requests'],p.settings()['user_request_cap'])
        for family in range(3):
            cohort=[g for g in groups if g['family_id']==f'family-{family:02d}']
            self.assertEqual(len(cohort),45)
            self.assertEqual([0.,0.],cohort[22]['impact_offset_m'])
            self.assertEqual(cohort[0]['values'],sources()[family]['values'])
            self.assertEqual(cohort[-1]['q14'],sources()[family]['q14'])

    def test_new_action_identity_binds_offset_and_native_release(self):
        g=p.coarse(sources())[0];p.bind([g],'b'*64);before=g['query_fingerprint']
        h=copy.deepcopy(g);h['impact_offset_m'][0]+=.001;p.bind([h],'b'*64)
        self.assertNotEqual(before,h['query_fingerprint'])
        h=dict(g,substeps=64,repeat_index=0,native_initial_q10=[0.]*6,native_initial_q14=[0.]*6,native_initial_ball_center_m=[.4,-.2,.8])
        before=p.execution_fingerprint(h);h['native_initial_ball_center_m'][0]+=.001
        self.assertNotEqual(before,p.execution_fingerprint(h))
        self.assertNotEqual(p.scene_fingerprint({'scene_fingerprint':'b'*64}),'b'*64)

    def test_selection_guarantees_family_diversity_before_global_fill(self):
        rows=[]
        for i,g in enumerate(p.coarse(sources())):
            if i not in [0,1,2,45,90]:continue
            rows.append(dict(g,id=f'source-{i:03d}',native_release=True,invalid=None,invalid_physics=False,
                route_success=i<90,route_quality=2 if i<45 else 1 if i<90 else 0))
        chosen=p.select(rows)
        self.assertEqual([r['family_id'] for r in chosen[:3]],['family-00','family-01','family-02'])
        self.assertEqual(len(chosen),5)
        rows[0]['invalid_physics']=True
        self.assertNotIn(rows[0]['id'],[r['id'] for r in p.select(rows)])
        self.assertEqual(len(p.select(rows[1:3])),2)

    def test_verification_freezes_actual_source_offset_joints_and_quality(self):
        rows=p.coarse(sources())[:5]
        for i,r in enumerate(rows):r.update(query_fingerprint='c'*64,route_quality=i%3)
        groups=p.verification(rows)
        for g,r in zip(groups,rows):
            self.assertTrue(g['freeze_joints']);self.assertEqual(g['q10'],r['q10']);self.assertEqual(g['q14'],r['q14'])
            self.assertEqual(g['impact_offset_m'],r['impact_offset_m']);self.assertEqual(g['source_route_quality'],r['route_quality'])

    def test_fresh_qualification_cannot_invent_source_capture(self):
        rows=[record(s,i,2) for s in p.STEPS for i in range(3)]
        self.assertTrue(p.qualification(rows,positions(rows),controls_passed=True,source_capture=True)['clean_qualified'])
        value=p.qualification(rows,positions(rows),controls_passed=True,source_capture=False)
        self.assertFalse(value['qualified']);self.assertFalse(value['source_capture'])
        self.assertNotIn('parent_source_capture',value)


if __name__=='__main__':unittest.main()
