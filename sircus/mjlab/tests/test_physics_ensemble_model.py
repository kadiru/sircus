"""Small synthetic tests; no simulator outcome or generalization claims."""
import copy
from pathlib import Path
import tempfile
import unittest
import numpy as np
from sircus.mjlab import physics_ensemble_contract as C
try:
    import torch
    from sircus.mjlab import physics_ensemble_model as M
except ImportError:
    torch=None


class ContractTests(unittest.TestCase):
    def test_fresh_disjoint_splits_deterministic_and_failure_denominators(self):
        old=dict(exclusions=[[.1,-.2]],splits={k:[dict(bucket_xy_m=[.2+i*.1,0.])] for i,k in enumerate(('train','validation','test'))})
        a=C.make_splits(old);self.assertEqual(a,C.make_splits(old))
        self.assertEqual([len(a['splits'][k]) for k in ('validation','test')],[24,60])
        used=a['excluded'][:]
        for role in ('validation','test'):
            for r in a['splits'][role]:
                self.assertGreaterEqual(min(np.linalg.norm(np.asarray(x)-r['bucket_xy_m']) for x in used),.01)
                used.append(r['bucket_xy_m'])
        self.assertEqual(C.protocol()['maximum_fresh_requests'],264)
        r=[dict(native_release=False,route_success=False,invalid_physics=False),
           dict(native_release=True,route_success=True,invalid_physics=False,route_quality=2),
           dict(native_release=True,route_success=False,invalid_physics=False),
           dict(native_release=True,route_success=False,invalid_physics=True)]
        self.assertEqual(C.counts(r),dict(requested=4,native_releases=3,captures=1,clean=1,rejected=1,native_misses=1,invalid=1))


@unittest.skipIf(torch is None,'Optional pinned Torch environment required')
class ModelTests(unittest.TestCase):
    def test_invalid_and_missing_targets_stay_masked_and_heldout_cannot_fit(self):
        reject=dict(native_release=False,invalid='geometry',invalid_physics=False)
        y,mask,_=M.target(reject,None,{},{});self.assertEqual(y,[0.]*9);self.assertEqual(mask,[1.]+[0.]*8)
        bad=dict(native_release=True,invalid_physics=True)
        y,mask,_=M.target(bad,None,{},{});self.assertEqual(y[0],1.);self.assertEqual(mask,[1.]+[0.]*8)
        with self.assertRaises(ValueError):M.build_dataset([dict(id='test',role='test')],'.',{},{} )
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):M.fit([dict(role='validation')],d,'a'*64)
            self.assertFalse((Path(d)/'ensemble.pt').exists())

    def test_fit_changes_weights_reduces_loss_and_roundtrips_exactly(self):
        rng=np.random.default_rng(2);data=[]
        for i in range(24):
            x=rng.uniform(-.6,.6,7);positive=float(x[0]>0)
            y=[1.,1.,positive,positive,.5,float(np.log1p(.1/.1)),float(x[0]),float(x[1]),float(x[2])]
            data.append(dict(id=f'train{i}',role='train',x=x.tolist(),y=y,mask=[1.]*9))
        with tempfile.TemporaryDirectory() as d:
            artifact=M.fit(data,d,'a'*64);p=Path(d)/artifact['path']
            self.assertEqual(artifact['sha256'],C.sha(p));state=torch.load(p,weights_only=True)
            for member in range(5):
                curves=[r for r in artifact['curves'] if r['member']==member]
                self.assertLess(curves[-1]['full_train_loss'],curves[0]['full_train_loss'])
                self.assertFalse(torch.equal(state['members'][member]['0.weight'],state['initial_members'][member]['0.weight']))
                self.assertEqual(state['bootstrap_indices'][member],np.random.default_rng(C.SEEDS[member]).integers(0,24,24).tolist())
            a=M.load(p);pred=a.predict_inputs([data[0]['x']]);self.assertEqual(pred,M.load(p).predict_inputs([data[0]['x']]))
            self.assertGreaterEqual(pred[0]['capture_probability'],0.);self.assertLessEqual(pred[0]['capture_probability'],1.)
            self.assertGreaterEqual(pred[0]['aperture_distance_m'],0.)
            before=p.read_bytes()
            with self.assertRaises(ValueError):M.fit(data,d,'a'*64)
            self.assertEqual(before,p.read_bytes())

    def test_candidate_ranking_has_fixed_bank_and_no_outcome_dependency(self):
        anchor=dict(values=[.5,-.2,.3,30.,.3]);context=dict(id='new',bucket_xy_m=[.6,-.4])
        teacher=dict(id='train',role='train',context_id='train',bucket_xy_m=[.65,-.4],values=anchor['values'],
            effective_source_hint=anchor,native_release=True,route_success=True,invalid_physics=False,route_quality=2,learning_reward=4.)
        items=M.candidates(context,anchor,anchor['values'],anchor['values'],[teacher])
        self.assertEqual(len(items),16);self.assertEqual(items,M.candidates(context,anchor,anchor['values'],anchor['values'],[teacher]))
        for r in items:self.assertTrue((np.abs(r['normalized_residual'])<=1).all())
        class Fake:
            def predict_inputs(self,x):return [dict(ranking_score=float(i)) for i in range(len(x))]
        rank=M.rank(Fake(),context,anchor,items)
        self.assertEqual(rank['order'],list(reversed(range(16))));self.assertFalse(rank['ball_flight_search_at_inference'])
        changed=copy.deepcopy(context);changed['actual_capture']=False
        self.assertEqual(rank,M.rank(Fake(),changed,anchor,items))


if __name__=='__main__':unittest.main()
