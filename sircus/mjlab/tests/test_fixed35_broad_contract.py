import copy
import unittest
from sircus.mjlab import fixed35_broad_contract as C
from sircus.mjlab.table_correction_native import META


def sources():
    result=[]
    for i,context in enumerate(C.CASE_IDS):
        r={k:None for k in META};r.update(id='hint-'+str(i),context_id=context,
            bucket_xy_m=[[.86,.025],[1.05,.025],[1.24,.25]][i],values=[.65,0.,.35,28.,.2])
        result.append(r)
    return result


class BroadProtocolTests(unittest.TestCase):
    def test_native_budget_distinguishes_repeats_controls_and_refusals(self):
        p=C.protocol();self.assertEqual(sum(p['search_native_quotas'].values()),1000)
        self.assertEqual(1000+p['repeat_max']+p['preserved']+p['controls'],1075)
        self.assertFalse(p['training']);self.assertEqual(p['substeps'],64)
    def test_deterministic_conditional_gaps_without_source_mutation(self):
        source=sources();old=copy.deepcopy(source)
        cap=lambda *a,**kw:.12
        reach=lambda *a:{'passed':True,'fixture':True}
        a=C.make_plan(source,{},'assets',cap_fn=cap,reach_fn=reach,per_cell=12,raw_limit=200)
        b=C.make_plan(source,{},'assets',cap_fn=cap,reach_fn=reach,per_cell=12,raw_limit=200)
        self.assertEqual(a,b);self.assertEqual(source,old);self.assertEqual(len(a['rows']),36)
        for r in a['rows']:
            self.assertGreaterEqual(r['values'][4],.05);self.assertLessEqual(r['values'][4],.12)
            self.assertFalse(r['exact_parent_action']);self.assertIn(r['proposal_family'],C.FAMILIES)
        self.assertEqual(len({(r['context_id'],tuple(r['values'])) for r in a['rows']}),36)
    def test_exhaustion_is_explicit_not_clipped_or_relabelled(self):
        out=C.make_plan(sources(),{},'assets',cap_fn=lambda *a,**kw:None,
                        reach_fn=lambda *a:None,per_cell=2,raw_limit=3)
        self.assertFalse(out['rows']);self.assertTrue(all(x['raw_considered']==3 and not x['proposal_limit_reached'] for x in out['cells']))
    def test_geometry_filter_cannot_be_ignored(self):
        out=C.make_plan(sources(),{},'assets',cap_fn=lambda *a,**kw:.2,
                        reach_fn=lambda *a:{'passed':False},per_cell=2,raw_limit=10)
        self.assertFalse(out['rows'])


if __name__=='__main__':unittest.main()
