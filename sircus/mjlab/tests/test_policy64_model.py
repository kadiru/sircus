"""Real CPU behavior-cloning checks; skipped when pinned Torch is unavailable."""

import copy
import importlib.util
import unittest

import numpy as np


@unittest.skipUnless(importlib.util.find_spec('torch'),'Pinned Torch model checks run on Ubuntu CPU')
class BootstrapTests(unittest.TestCase):
    def fixture(self):
        rng=np.random.default_rng(31)
        x=rng.uniform(-1.,1.,(16,9))
        y=.35*np.tanh(x[:,:5]+.25*x[:,4:9])
        rows=[dict(id=f'train-{i:02d}',role='train',context_id=f'context-{i:02d}',
                   observation=x[i].tolist(),normalized_residual=y[i].tolist(),
                   learning_reward=float(.8+.3*x[i,0]-.2*x[i,8])) for i in range(len(x))]
        return rows

    def test_seeded_initialization_is_residual_nine_feature_policy(self):
        import torch
        from sircus.mjlab.policy64_model import initialized
        from sircus.mjlab.learning_policy import weight_hash
        a,b,c=initialized(7),initialized(7),initialized(8)
        self.assertEqual(weight_hash(a),weight_hash(b)); self.assertNotEqual(weight_hash(a),weight_hash(c))
        with torch.no_grad():
            self.assertEqual(tuple(a.actor(torch.zeros(3,9)).shape),(3,5))
            self.assertTrue(torch.equal(a.actor(torch.zeros(3,9)),torch.zeros(3,5)))
            self.assertTrue(torch.equal(a.logstd,torch.full((5,),-3.)))

    def test_warmstart_fits_known_bounded_residuals_and_train_critic_without_mutating_data(self):
        import torch
        from sircus.mjlab.policy64_model import initialized,bootstrap
        from sircus.mjlab.learning_policy import weight_hash
        torch.set_num_threads(1)
        rows=self.fixture(); teachers=rows[:12]; before=copy.deepcopy(rows)
        model=initialized(17)
        result=bootstrap(model,rows,teachers,actor_steps=160,critic_steps=100)
        self.assertNotEqual(result['before'],result['after']); self.assertEqual(result['after'],weight_hash(model))
        self.assertLess(result['final_actor_loss'],result['initial_actor_loss']*.1)
        self.assertLess(result['final_actor_loss'],.005)
        self.assertLess(result['final_critic_loss'],result['initial_critic_loss']*.1)
        self.assertEqual(result['actor_teacher_ids'],[r['id'] for r in teachers])
        self.assertEqual(result['critic_row_count'],16); self.assertEqual(result['actor_steps'],160)
        self.assertEqual(result['critic_steps'],100); self.assertEqual(rows,before)
        self.assertTrue(torch.equal(model.logstd,torch.full((5,),-3.)))
        with torch.no_grad():
            x=torch.tensor([r['observation'] for r in teachers],dtype=torch.float32)
            actual=model.actor(x).tanh().numpy()
        np.testing.assert_allclose(actual,np.asarray([r['normalized_residual'] for r in teachers]),atol=.12,rtol=0.)

    def test_heldout_rows_or_teachers_and_empty_bootstrap_fail_before_weight_changes(self):
        from sircus.mjlab.policy64_model import initialized,bootstrap
        from sircus.mjlab.learning_policy import weight_hash
        rows=self.fixture(); model=initialized(7); original=weight_hash(model)
        for role in ('validation','test','control'):
            for where in ('rows','teachers'):
                a,b=copy.deepcopy(rows),copy.deepcopy(rows[:4])
                (a if where=='rows' else b)[0]['role']=role
                with self.subTest(role=role,where=where),self.assertRaisesRegex(ValueError,'TRAIN'):
                    bootstrap(model,a,b,actor_steps=1,critic_steps=1)
                self.assertEqual(weight_hash(model),original)
        for a,b in (([],rows[:4]),(rows,[])):
            with self.assertRaisesRegex(ValueError,'TRAIN'):bootstrap(model,a,b,actor_steps=1,critic_steps=1)
            self.assertEqual(weight_hash(model),original)

    def test_teacher_wrong_feature_width_and_nonfinite_target_fail_before_updates(self):
        from sircus.mjlab.policy64_model import initialized,bootstrap
        from sircus.mjlab.learning_policy import weight_hash
        rows=self.fixture(); model=initialized(7); original=weight_hash(model)
        for key,value in (('observation',[0.]*10),('normalized_residual',[0.]*4),
                          ('observation',[float('nan')]*9),('normalized_residual',[float('inf')]*5)):
            teachers=copy.deepcopy(rows[:1]);teachers[0][key]=value
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):
                bootstrap(model,rows,teachers,actor_steps=1,critic_steps=1)
            self.assertEqual(weight_hash(model),original)


if __name__=='__main__': unittest.main()
