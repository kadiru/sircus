"""Local seeded exploration cannot lose source coverage or rewrite provenance."""
import copy
import unittest
from sircus.mjlab import fixed35_sixhour_contract as C
from sircus.mjlab.tests.test_fixed35_bouncer_contract import fixture,score_fields

class ContractTests(unittest.TestCase):
    def plan(self,limit=500):
        d=fixture();sources=d['numeric_sources'];confirmed=[]
        for c in (C.CASE_IDS[0],C.CASE_IDS[2]):
            r=copy.deepcopy(next(r for r in sources if r['context_id']==c))
            r.update(id='confirmed-'+c,**score_fields(True));confirmed.append(r)
        kw=dict(interval_fn=lambda *a,**k:(.05,.65),reach_fn=lambda *a,**k:dict(passed=True))
        return d,confirmed,kw,C.make_plan(sources,confirmed,{},None,limit=limit,**kw)
    def test_all_four_anchors_and_both_neighbors_represented_without_parent_relabel(self):
        d,confirmed,kw,p=self.plan();again=C.make_plan(d['numeric_sources'],confirmed,{},None,limit=500,**kw)
        self.assertEqual(p,again);self.assertEqual(len(p['rows']),500)
        seeds=[r for r in d['numeric_sources'] if r['context_id']==C.TARGET]+confirmed
        self.assertEqual(set(p['seed_counts']),{r['id'] for r in seeds})
        self.assertEqual(set(p['families']),set(C.FAMILIES))
        parents={r['id']:r for r in d['numeric_sources']}
        for r in p['rows']:
            self.assertEqual(r['context_id'],C.TARGET)
            self.assertEqual(parents[r['parent_row_id']]['context_id'],C.TARGET)
            self.assertEqual(r['parent_row_sha256'],C.digest(parents[r['parent_row_id']]))
            self.assertTrue(all(lo<=v<=hi for lo,v,hi in zip(C.LOW,r['values'],C.HIGH)))
    def test_windows_are_local_only_and_preservation_ids_unique_across_variants(self):
        d,confirmed,kw,_=self.plan(20)
        target=copy.deepcopy(next(r for r in d['numeric_sources'] if r['context_id']==C.TARGET));target.update(**score_fields(True))
        p=C.make_window_plan(d['numeric_sources'],confirmed+[target],{},None,limit=100,**kw)
        self.assertEqual({r['context_id'] for r in p['rows']},set(C.CASE_IDS))
        self.assertLessEqual(set(p['families']),{'axis','local'})
        variant=copy.deepcopy(confirmed[0]);variant.update(id='variant');variant['values'][2]+=.01
        saved=C.preserved_plan(d['preserved']+confirmed+[variant]);self.assertEqual(len(saved),len({r['id'] for r in saved}))
        self.assertTrue(all(r['exact_parent_action'] for r in saved))
    def test_reach_rejection_never_clips_a_proposal(self):
        d,confirmed,_,_=self.plan(1)
        p=C.make_plan(d['numeric_sources'],confirmed,{},None,limit=20,
            interval_fn=lambda *a,**k:(.05,.050000000001),reach_fn=lambda *a,**k:dict(passed=True))
        self.assertEqual(p['rows'],[]);self.assertGreater(p['geometric_declines']['gap_outside_reach_interval'],0)
    def test_exact_command_identity_includes_joints_not_labels(self):
        r=dict(context_id='c',values=[1]*5,q10=[0]*6,q14=[1]*6)
        self.assertEqual(C.command_key(r),C.command_key(dict(r,id='another')))
        self.assertNotEqual(C.command_key(r),C.command_key(dict(r,q14=[2]*6)))
    def test_repeat_keeps_prepared_proof(self):
        _,_,_,p=self.plan(1);q=p['rows'][0]
        d=dict(q,**score_fields(True),prepared_request=dict(q,prepared_connection_sha256='proof'),q10=[1]*6,q14=[2]*6,connection_sha256='proof')
        r=C.repeat_plan(d,0)
        self.assertEqual(len(r),3)
        self.assertTrue(all(x['prepared_connection_sha256']=='proof' and x['selected_q14']==[2]*6 for x in r))

if __name__=='__main__':unittest.main()
