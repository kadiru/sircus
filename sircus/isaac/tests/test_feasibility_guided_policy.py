"""Unit-only guided residual tests; no simulated capture claims."""

import copy
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import continuous_policy as old
import feasibility_guided_policy as policy
import feasibility_guided_support as support
from continuous_feasibility import HIGH, LOW, canonical_hash, decode_action


def signed(value):
    value=copy.deepcopy(value)
    value.pop("fingerprint",None)
    return dict(value,fingerprint=canonical_hash(value))


def fixture():
    cells=[dict(id=f"cell-{i}", bucket_xy_m=[i*.01, -.2]) for i in range(35)]
    rows=[]
    for i,cell in enumerate(cells):
        action=decode_action(cell,[0,.4,.35,.24,math.radians(29)],str(i))
        center=np.array([action[k] for k in support.ACTION_NAMES])
        rows.append(dict(cell_id=cell['id'],cell_index=i,bucket_xy_m=cell['bucket_xy_m'],
                         status='supported',anchor_action=action,
                         anchor_ik={n:dict(q=[0.]*6) for n in ('Arm10','Arm14')},
                         low=(center-support.BASE_HALF_WIDTH).tolist(),
                         high=(center+support.BASE_HALF_WIDTH).tolist()))
    value=signed(dict(schema=support.SCHEMA,source_fingerprint='a'*64,
                      feasibility_fingerprint='b'*64,policy_geometry_only=True,
                      action_names=list(support.ACTION_NAMES),global_low=LOW.tolist(),global_high=HIGH.tolist(),cells=rows))
    return cells,value


class SupportTests(unittest.TestCase):
    def test_affine_decode_is_continuous_and_context_bound(self):
        cells,value=fixture()
        raw=np.zeros((35,5))
        middle=support.physical_actions(raw,value,cells)
        tiny=support.physical_actions(raw+1e-6,value,cells)
        lo=support.physical_actions(raw-1,value,cells)
        hi=support.physical_actions(raw+1,value,cells)
        self.assertTrue((tiny>middle).all())
        np.testing.assert_allclose(lo,[r['low'] for r in value['cells']])
        np.testing.assert_allclose(hi,[r['high'] for r in value['cells']])
        np.testing.assert_allclose(middle,(lo+hi)/2)
        self.assertTrue((np.abs(tiny-middle)<1e-6).all())
        changed=copy.deepcopy(cells);changed[0]['bucket_xy_m'][0]+=.001
        with self.assertRaises(ValueError):support.physical_actions(raw,value,changed)
        with self.assertRaises(ValueError):support.physical_actions(raw+2,value,cells)
        with self.assertRaises(ValueError):support.physical_actions(raw[:3],value,cells)

    def test_missing_geometry_is_explicit_and_fallback_not_a_feasibility_claim(self):
        cells,value=fixture()
        row=value['cells'][1]
        row.update(status='unresolved',anchor_action=None,anchor_ik=None,low=LOW.tolist(),high=HIGH.tolist())
        value=signed(value)
        support.validate_support(value,cells)
        seeds=support.ik_seeds(value,cells)
        self.assertIsNone(seeds[1])
        self.assertIsNotNone(seeds[0])
        seeds[0]['Arm10']['q'][0]=99
        self.assertEqual(value['cells'][0]['anchor_ik']['Arm10']['q'][0],0)

    def test_tamper_and_invalid_boxes_are_rejected(self):
        cells,value=fixture()
        bad=copy.deepcopy(value);bad['cells'][0]['low'][2]=.3
        with self.assertRaises(ValueError):support.validate_support(bad)
        for edit in ('coordinates','bounds','anchor','status','global'):
            bad=copy.deepcopy(value)
            if edit=='coordinates':bad['cells'][0]['bucket_xy_m']=[99,99]
            if edit=='bounds':bad['cells'][0]['low'][0]=-99
            if edit=='anchor':bad['cells'][0]['anchor_ik']=None
            if edit=='status':bad['cells'][0]['status']='guaranteed'
            if edit=='global':bad['global_high'][0]=99
            with self.subTest(edit=edit),self.assertRaises(ValueError):support.validate_support(signed(bad),cells)

    def _local(self, accept, path_accept=lambda q:True):
        cell=dict(id='cell',bucket_xy_m=[.67,.3])
        action=decode_action(cell,[0,.4,.35,.24,math.radians(29)],'anchor')
        fits={n:dict(q=[0.]*6) for n in ('Arm10','Arm14')}
        home={n:np.zeros(6) for n in fits}
        options=[dict(action=action,ik=fits,proposal=4)]
        with patch.object(support,'_links',return_value={}),patch.object(support,'_home',return_value=home),\
             patch.object(support,'qualify_action',side_effect=lambda *args,**kw:dict(accepted=accept(),ik=fits)),\
             patch.object(support,'approximate_path_check',side_effect=lambda *args,**kw:dict(accepted=path_accept(args),reasons=[])):
            return support._cell_support(({},cell,0,options,'unused'))

    def test_local_probe_uses_largest_predeclared_passing_box(self):
        row=self._local(lambda:True)
        self.assertEqual(row['selected_width_scale'],1)
        self.assertEqual(row['calibration'][0]['requests'],16)
        self.assertEqual(row['calibration'][0]['accepted'],16)
        self.assertTrue(row['qualification_target_met'])
        self.assertEqual(row['anchor_proposal'],4)
        self.assertEqual(row['status'],'supported')

    def test_shrink_keeps_every_failed_probe_and_stops_at_threshold(self):
        calls=[0]
        def accept():
            calls[0]+=1
            return calls[0]>32
        row=self._local(accept)
        self.assertEqual(row['selected_width_scale'],.25)
        self.assertEqual([x['accepted'] for x in row['calibration']],[0,0,16])
        self.assertEqual(sum(x['requests'] for x in row['calibration']),48)
        self.assertEqual(len([r for x in row['calibration'] for r in x['records'] if not r['accepted']]),32)

    def test_thin_domain_fallback_stays_below_target_explicitly(self):
        row=self._local(lambda:False)
        self.assertEqual(row['selected_width_scale'],.125)
        self.assertFalse(row['qualification_target_met'])
        self.assertEqual(row['reason'],'finite_probe_acceptance_below_target')
        self.assertEqual(len(row['calibration']),4)
        self.assertEqual(sum(x['requests'] for x in row['calibration']),64)

    def test_anchor_path_rejection_keeps_cell_unresolved(self):
        row=self._local(lambda:True,lambda q:False)
        self.assertEqual(row['status'],'unresolved')
        self.assertEqual(row['calibration'],[])
        self.assertIsNone(row['anchor_action'])
        self.assertEqual(len(row['anchor_checks']),1)

    def test_prior_prefers_declared_geometry_without_reading_reward(self):
        cell=dict(bucket_xy_m=[.67,.3])
        home={n:np.zeros(6) for n in ('Arm10','Arm14')}
        fit={n:dict(q=[0.]*6) for n in home}
        center=decode_action(cell,support.prior_values(cell),'x')
        away=decode_action(cell,[.5,.8,.5,.45,math.radians(45)],'y')
        before=support.anchor_score(cell,center,fit,home)
        center.update(reward=-999,captures=0,outcome='miss')
        self.assertEqual(before,support.anchor_score(cell,center,fit,home))
        self.assertLess(before,support.anchor_score(cell,away,fit,home))
        far={n:dict(q=[3.]*6) for n in home}
        self.assertGreater(support.anchor_score(cell,center,far,home),before)


@unittest.skipIf(old.torch is None,'PyTorch CPU environment unavailable')
class GuidedPolicyTests(unittest.TestCase):
    def setUp(self):
        self.torch=old.torch;self.torch.set_num_threads(1);self.torch.manual_seed(20261112)
        self.policy=policy.GuidedPolicy()
        self.obs=self.torch.zeros((128,5));self.obs[:,2]=.65
        self.bindings=dict(study_fingerprint='a'*64,geometry_fingerprint='b'*64,
                           domain_fingerprint='c'*64,support_fingerprint='d'*64)

    def test_guided_fresh_mean_zero_and_unchanged_architecture_budget(self):
        distribution=self.policy.distribution(self.obs)
        self.torch.testing.assert_close(distribution.loc,self.torch.zeros((128,5)))
        self.torch.testing.assert_close(distribution.scale,self.torch.tensor(old.INITIAL_LOGSTD).exp().expand(128,5))
        self.assertEqual(sum(x.numel() for x in self.policy.parameters()),sum(x.numel() for x in old.ContinuousPolicy().parameters()))
        self.assertNotEqual(policy.POLICY_VERSION,old.POLICY_VERSION)
        self.assertNotEqual(policy.ARCHITECTURE,old.ARCHITECTURE)

    def test_likelihood_is_exact_for_saved_residual_latent(self):
        model=self.policy.double();obs=self.obs.double()
        latent=self.torch.linspace(-2,2,128*5,dtype=self.torch.float64).reshape(128,5)
        expected=(model.distribution(obs).log_prob(latent)-self.torch.log1p(-latent.tanh().square())).sum(-1)
        self.torch.testing.assert_close(model.log_prob(obs,latent),expected,atol=1e-10,rtol=1e-10)

    def test_ppo_uses_all_samples_including_rejected_attempts(self):
        drawn=policy.sample(self.policy,self.obs,self.torch.Generator().manual_seed(12))
        returns=drawn['actions_normalized'][:,3].clone();returns[::3]=-1
        optim=self.torch.optim.Adam(self.policy.parameters(),lr=.0003)
        change=policy.update(self.policy,optim,self.obs,drawn['latent'],drawn['logp'],returns,old.PPO_DEFAULTS)
        self.assertEqual(change['samples'],128)
        self.assertNotEqual(change['actor_before'],change['actor_after'])
        self.assertEqual(len(change['epochs']),4)

    def test_checkpoint_binds_support_and_preserves_rng(self):
        payload=policy.checkpoint_payload(self.policy,None,**self.bindings,update_count=0,attempted_shots=0,seed=20261112)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'guided.pt';self.torch.save(payload,path)
            rng=self.torch.random.get_rng_state().clone()
            loaded,saved=policy.load_checkpoint(path,**self.bindings,expected_updates=0,expected_attempted_shots=0)
            self.torch.testing.assert_close(rng,self.torch.random.get_rng_state())
            self.assertEqual(policy.weight_hash(loaded),policy.weight_hash(self.policy))
            self.assertEqual(saved['support_fingerprint'],'d'*64)
            wrong=dict(self.bindings,support_fingerprint='e'*64)
            with self.assertRaises(ValueError):policy.load_checkpoint(path,**wrong)
            with self.assertRaises(ValueError):old.load_checkpoint(path,**{k:v for k,v in self.bindings.items() if k!='support_fingerprint'})
            changed=copy.deepcopy(payload);changed['architecture']=old.ARCHITECTURE
            self.torch.save(changed,path)
            with self.assertRaises(ValueError):policy.load_checkpoint(path,**self.bindings)

    def test_old_policy_cannot_be_saved_as_guided(self):
        with self.assertRaises(ValueError):policy.checkpoint_payload(old.ContinuousPolicy(),None,**self.bindings,update_count=0,attempted_shots=0,seed=1)


if __name__=='__main__':
    unittest.main()
