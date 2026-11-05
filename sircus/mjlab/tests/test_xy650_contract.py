"""Literal XY protocol and identity checks; fixtures are synthetic only."""
import copy
import math
import unittest

from sircus.mjlab import xy650_contract as p
from sircus.mjlab.tests.test_clean650_contract import row


class XYContractTests(unittest.TestCase):
    def seed(self):
        r=row();r['values'][:2]=[.4708757793012148,-.2778934014509746]
        r['bucket_xy_m']=[-.030306598129941298,-.6917970430784122]
        r.update(source_record_id='old650',source_record_sha256='a'*64)
        return r

    def test_literal_unique_grid_translated_references_original_and_budget(self):
        seed=self.seed();groups=p.coarse(seed)
        self.assertEqual(len(groups),253);self.assertEqual(len({g['id'] for g in groups}),253)
        self.assertEqual(len({tuple(g['values']+g['bucket_xy_m']) for g in groups}),253)
        self.assertEqual(groups[0]['translation_m'],[-.01,.04]);self.assertEqual(groups[242]['translation_m'],[.01,.08])
        self.assertEqual(groups[0]['values'][2:],[.36,26.,.39]);self.assertEqual(groups[242]['values'][2:],[.40,30.,.43])
        self.assertEqual(groups[-1]['values'],seed['values']);self.assertEqual(groups[-1]['bucket_xy_m'],seed['bucket_xy_m'])
        for g in groups:
            dx,dy=g['translation_m']
            self.assertAlmostEqual(math.dist(g['values'][:2],g['bucket_xy_m']),.65,places=14)
            for i,delta in enumerate((dx,dy)):
                self.assertAlmostEqual(g['values'][i]-seed['values'][i],delta,places=14)
                self.assertAlmostEqual(g['bucket_xy_m'][i]-seed['bucket_xy_m'][i],delta,places=14)
        self.assertEqual(p.settings()['max_requests'],253+81+30+24)

    def test_unique_selection_and_precontact_diagnostic_ranking(self):
        data=[]
        for i in range(4):
            r=row(i,0);r['values'][4]+=.01*i
            r['first_flight']=dict(available=True,first_crossing=dict(time_s=.5),mouth_clearance_m=.01*i,closest_opening_distance_m=.2)
            data.append(r)
        # Old post-table proximity must not rescue the worse first flight.
        data[0]['closest_opening_distance_m']=0.
        duplicate=copy.deepcopy(data[-1]);duplicate['id']='source-duplicate'
        self.assertEqual([r['id'] for r in p.select_unique(data+[duplicate],3)],['source-3','source-2','source-1'])
        data[0].update(route_success=True,route_quality=2)
        self.assertEqual(p.ranked(data,1)[0]['id'],'source-0')

    def test_query_scene_source_and_execution_identities_bind_actual_branches(self):
        g=p.coarse(self.seed())[0];p.bind([g],'b'*64);first=g['query_fingerprint']
        h=copy.deepcopy(g);h['values'][0]+=.001;p.bind([h],'b'*64);self.assertNotEqual(first,h['query_fingerprint'])
        h=copy.deepcopy(g);h['source_record_id']='other';p.bind([h],'b'*64);self.assertNotEqual(first,h['query_fingerprint'])
        r=dict(g,id='exec-01',substeps=64,repeat_index=0,q10=[0.]*6,q14=[0.]*6,native_initial_q10=[0.]*6,native_initial_q14=[0.]*6)
        before=p.execution_fingerprint(r);r['native_initial_q14'][0]=.001
        self.assertNotEqual(before,p.execution_fingerprint(r))
        self.assertNotEqual(p.scene_fingerprint({}, {}, {'asset':'a'}),p.scene_fingerprint({}, {}, {'asset':'b'}))

    def test_refinement_and_verification_preserve_selected_translation_and_exact_joints(self):
        seed=dict(self.seed(),translation_m=[0.,.06],query_fingerprint='f'*64)
        refinements=p.refinement([seed]);self.assertEqual(len(refinements),27)
        self.assertEqual(refinements[13]['values'],seed['values'])
        self.assertTrue(all(g['translation_m']==seed['translation_m'] and g['bucket_xy_m']==seed['bucket_xy_m'] for g in refinements))
        g=p.verification([seed])[0]
        self.assertTrue(g['freeze_joints']);self.assertEqual(g['q10'],seed['q10']);self.assertEqual(g['source_query_fingerprint'],'f'*64)


if __name__=='__main__':unittest.main()
