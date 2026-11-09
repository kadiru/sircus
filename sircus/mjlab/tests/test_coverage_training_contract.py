import unittest
import numpy as np
from sircus.mjlab import coverage_training_contract as C

class Contract(unittest.TestCase):
    def original(self):
        return [dict(id=f'old{i}',bucket_xy_m=[.1+(i%7)*.19,-.65+(i//7)*.225]) for i in range(35)]
    def test_split_stratification_and_separation(self):
        old=self.original();ex=[r['bucket_xy_m'] for r in old]+[[1.1095395651982425,-.22146373203137287]]
        splits,exclusions=C.make_splits(old,ex)
        self.assertEqual([len(splits[k]) for k in ['train','validation','test']],[105,35,70])
        again,_=C.make_splits(old,ex);self.assertEqual(again,splits)
        fresh=splits['train'][35:]+splits['validation']+splits['test'];known=np.asarray(exclusions)
        for r in fresh:
            self.assertGreaterEqual(np.linalg.norm(known-r['bucket_xy_m'],axis=1).min(),.005)
            known=np.vstack((known,r['bucket_xy_m']))
        for role,cohort,n in [('train',splits['train'][35:],2),('validation',splits['validation'],1),('test',splits['test'],2)]:
            from collections import Counter
            bins=Counter(tuple(r['grid_cell']) for r in cohort);self.assertEqual(len(bins),35);self.assertEqual(set(bins.values()),{n})
    def test_schedule_balanced_seeded(self):
        for seed in C.SEEDS:
            a=C.schedule(seed);self.assertEqual(len(a),512);self.assertEqual(sorted(a[:105]),list(range(105)));self.assertEqual(a,C.schedule(seed))
        self.assertNotEqual(C.schedule(C.SEEDS[0]),C.schedule(C.SEEDS[1]))
    def test_search_exact_axes_no_clipping(self):
        context=dict(id='train-a',split='train')
        r,meta=C.search_residual(context,1);self.assertAlmostEqual(r[3],-.25)
        for i in range(12):
            r,m=C.search_residual(context,i);self.assertTrue(np.isfinite(r).all());self.assertLessEqual(np.abs(r).max(),1)
        with self.assertRaises(ValueError):C.search_residual(dict(id='test',split='test'),0)
    def test_refinement_train_membership(self):
        c=dict(id='train-a',split='train');best=dict(id='sample',role='train',context_id='train-a',normalized_residual=[0,0,0,.25,0])
        r,m=C.search_residual(c,10,best);self.assertEqual(m['source_training_ids'],['sample'])
        r2,_=C.search_residual(c,10,best);self.assertTrue(np.array_equal(r,r2))
        for key,value in [('role','validation'),('context_id','different')]:
            with self.assertRaises(ValueError):C.search_residual(c,10,dict(best,**{key:value}))
    def test_request_budget_and_limits(self):
        p=C.protocol();self.assertEqual(p['search_requests']+p['train_requests']+p['validation_requests']+p['test_requests']+p['control_requests'],2757)
        self.assertLess(p['search_limit_s'],p['train_cut_s']);self.assertLess(p['train_cut_s'],p['acquired_slot_limit_s']);self.assertLess(p['acquired_slot_limit_s'],p['global_limit_s'])
        self.assertEqual(p['test_requests'],70*4)
    def test_version_preserves_old_contract(self):
        from sircus.mjlab import table_training_contract as old
        self.assertNotEqual(C.VERSION,old.VERSION);self.assertEqual(old.protocol()['maximum_fresh_requests'],766)

if __name__=='__main__':unittest.main()
