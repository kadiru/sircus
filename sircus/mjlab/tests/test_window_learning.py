"""Terminal SAC and TRAIN-only supervised gap inference, without native shots."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from sircus.mjlab.window_learning import SuccessModel, normalized_contexts, normalized_gaps

HAVE_TORCH = importlib.util.find_spec("torch") is not None
if HAVE_TORCH:
    import torch
    from sircus.mjlab.window_learning import SACAgent, terminal_target, squashed_gaussian_log_prob


def toy_rows():
    result=[]
    for height in (.2,.4):
        for tilt in (24.,48.):
            target=.12+.6*height
            for gap in np.linspace(.02,.65,25):
                result.append(dict(role="train",context=[height,tilt],gap_m=float(gap),
                    route_success=bool(abs(gap-target)<.065),reward=float(1-(gap-target)**2)))
    return result


class WindowNormalizationTests(unittest.TestCase):
    def test_units_bounds_and_invalid_shapes(self):
        np.testing.assert_allclose(normalized_contexts([[.04,8],[.56,64]]),[[-1,-1],[1,1]])
        np.testing.assert_allclose(normalized_gaps([.02,.65]),[-1,1])
        for value in ([[.03,30]],[[.2,70]],[[float('nan'),30]],[[.2]],[[.2,30,40]]):
            with self.assertRaises(ValueError):normalized_contexts(value)
        for value in ([.01],[.7],[float('inf')]):
            with self.assertRaises(ValueError):normalized_gaps(value)
        with self.assertRaises(ValueError):normalized_gaps([.1],2)


class SuccessModelTests(unittest.TestCase):
    def test_actual_supervised_fit_predicts_an_unobserved_context(self):
        model=SuccessModel(seed=22).fit(toy_rows())
        gaps=model.propose([[.3,36]])
        self.assertTrue(.24<gaps[0]<.36,gaps)
        high=model.predict([[.3,36]],[float(gaps[0])])[0]
        low=model.predict([[.3,36]],[.62])[0]
        self.assertGreater(high,low+.2)
        self.assertEqual(model.fit_log['rows'],100)
        self.assertFalse(model.fit_log['constant_labels'])
        before=model.coef_.copy();model.propose([[.3,36]])
        np.testing.assert_array_equal(before,model.coef_)

    def test_all_positive_and_negative_labels_are_finite_defined_priors(self):
        for label in (False,True):
            rows=[dict(role='train',context=[.3,36],gap_m=float(gap),route_success=label) for gap in [.1,.3,.5]]
            model=SuccessModel().fit(rows)
            curve=model.curve([[.3,36]])
            self.assertEqual(curve['probability'].shape,(1,129))
            self.assertTrue(np.isfinite(curve['probability']).all())
            self.assertTrue(((curve['probability']>0)&(curve['probability']<1)).all())
            self.assertAlmostEqual(model.propose([[.3,36]])[0],.335)

    def test_train_split_guard_and_malformed_observations(self):
        for override in ({'role':'validation'},{'split':'test','role':'test'},
                         {'split':'test','role':'train'}, {'reward':float('nan')},
                         {'route_success':None},{'gap_m':float('nan')}):
            row=dict(role='train',context=[.3,36],gap_m=.3,route_success=True);row.update(override)
            with self.assertRaises(ValueError):SuccessModel().fit([row])
        with self.assertRaises(ValueError):SuccessModel().fit([])
        with self.assertRaises(ValueError):SuccessModel().propose([[.3,36]])

    def test_native_record_adapter_and_checkpoint_identity(self):
        rows=[dict(role='train',values=[.47,-.28,.3,36,float(gap)],route_success=bool(.2<gap<.4)) for gap in np.linspace(.02,.65,20)]
        model=SuccessModel(seed=3).fit(rows)
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'surrogate.json';model.save(path,'contract-a')
            same=SuccessModel.load(path,'contract-a')
            np.testing.assert_array_equal(model.propose([[.3,36]]),same.propose([[.3,36]]))
            with self.assertRaises(ValueError):SuccessModel.load(path,'contract-b')
            value=json.loads(path.read_text());value['coef'][0]+=1;path.write_text(json.dumps(value))
            with self.assertRaises(ValueError):SuccessModel.load(path,'contract-a')


@unittest.skipUnless(HAVE_TORCH,"SAC tests require the pinned Ubuntu Torch environment")
class TerminalSACTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_terminal_targets_ignore_arbitrary_next_value_and_detach(self):
        reward=torch.tensor([[2.],[-1.]],requires_grad=True)
        a=terminal_target(reward,next_values=torch.full_like(reward,1e9))
        b=terminal_target(reward,next_values=torch.full_like(reward,-1e9))
        torch.testing.assert_close(a,reward);torch.testing.assert_close(a,b)
        self.assertFalse(a.requires_grad)

    def test_normalized_tanh_density_includes_stable_jacobian(self):
        z=torch.tensor([[-4.],[0.],[4.]],dtype=torch.float64)
        mean=torch.zeros_like(z);std=torch.zeros_like(z)
        actual=squashed_gaussian_log_prob(z,mean,std)
        expected=torch.distributions.Normal(mean,torch.ones_like(z)).log_prob(z)-torch.log1p(-z.tanh()**2)
        torch.testing.assert_close(actual,expected,atol=1e-10,rtol=1e-10)

    def test_reproducible_bounded_actions_and_immutable_mean_inference(self):
        a=SACAgent(seed=9);b=SACAgent(seed=9)
        context=np.tile([.3,36],(100,1))
        first=a.propose(context,stochastic=True);second=b.propose(context,stochastic=True)
        np.testing.assert_array_equal(first,second)
        self.assertTrue(((first>=.02)&(first<=.65)).all())
        before=a.weight_hash();one=a.propose(context);two=a.propose(context)
        np.testing.assert_array_equal(one,two);self.assertEqual(a.weight_hash(),before)

    def test_replay_is_bounded_and_rejects_held_out_or_nonfinite_data(self):
        a=SACAgent(seed=1,capacity=3)
        a.add(np.tile([.3,36],(4,1)),[.1,.2,.3,.4],[0,1,2,3],roles=['train']*4)
        self.assertEqual(a.replay_size,3);self.assertEqual(a.total_seen,4)
        np.testing.assert_allclose(a.rewards[:,0],[1,2,3])
        for kwargs in (dict(roles=['test']),dict(roles=['validation'])):
            with self.assertRaises(ValueError):a.add([[.3,36]],[.3],[1],**kwargs)
        with self.assertRaises(ValueError):a.add([[.3,36]],[.3],[float('nan')])
        with self.assertRaises(ValueError):a.add([[.3,36]],[.3],[1e100])
        self.assertEqual(a.total_seen,4)

    def test_actual_actor_critic_updates_improve_a_terminal_quadratic(self):
        a=SACAgent(seed=13,alpha=.005,learning_rate=.002)
        rng=np.random.default_rng(14);gaps=rng.uniform(.02,.65,512)
        context=np.tile([.3,36],(512,1));rewards=1-((gaps-.55)/.315)**2
        a.add(context,gaps,rewards,roles=['train']*512)
        before=a.propose([[.3,36]])[0];log=a.update(steps=500,batch_size=128)
        after=a.propose([[.3,36]])[0]
        self.assertNotEqual(log['before'],log['after']);self.assertEqual(log['gradient_steps'],500)
        self.assertLess(abs(after-.55),abs(before-.55)*.65)
        self.assertTrue(np.isfinite([log['critic_loss'],log['actor_loss'],log['normalized_action_entropy']]).all())

    def test_checkpoint_fingerprint_and_exact_rng_resume(self):
        a=SACAgent(seed=17);a.add([[.2,24],[.4,48]],[.2,.5],[0,1]);a.update(steps=3,batch_size=2)
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'sac.pt';a.save(path,'contract-a',include_replay=True)
            with self.assertRaises(ValueError):SACAgent.load(path,'contract-b')
            b=SACAgent.load(path,'contract-a')
            np.testing.assert_array_equal(a.propose([[.3,36]],True),b.propose([[.3,36]],True))
            a.update(steps=2,batch_size=2);b.update(steps=2,batch_size=2)
            self.assertEqual(a.weight_hash(),b.weight_hash())
            with self.assertRaises(ValueError):a.save(path,'contract-a')


if __name__=='__main__':unittest.main()
