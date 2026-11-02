"""Stable starts, independently stopped branches and strict continuation gates."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from sircus.mjlab.stable_ik import StableBatchIK, normalize_known, target_digest
from sircus.mjlab.stable_geometry import known_array, known_request_match, ordered_pairs, select_pair

try:
    import torch
except ImportError:
    torch = None


class PureStableTests(unittest.TestCase):
    def test_query_identity_exact_role_separated_and_signed_zero_canonical(self):
        a = np.eye(4)
        b = a.copy(); b[0, 3] = -0.
        self.assertEqual(target_digest(a), target_digest(b))
        self.assertNotEqual(target_digest(a, role="a"), target_digest(a, role="b"))
        b[0, 3] = 1e-12
        self.assertNotEqual(target_digest(a), target_digest(b))

    def test_absent_hints_are_explicit_and_partial_hints_rejected(self):
        self.assertTrue(np.isnan(normalize_known(None, 2)).all())
        self.assertEqual(known_array(np.zeros((2, 6)), (2, 1)).shape, (2, 1, 6))
        for a in (np.array([[0, 0, np.nan, 0, 0, 0]]), np.full((1, 6), np.inf)):
            with self.assertRaises(ValueError): normalize_known(a, 1)
            with self.assertRaises(ValueError): known_array(a, (1, 1))

    def fixture(self):
        return dict(q=np.array([[0., 0, 0, 0, 0, 0], [1., 0, 0, 0, 0, 0]]),
                    valid=np.array([True, True]), residual=np.array([1e-9, 5e-5]))

    def test_old_request_identity_is_separate_from_loose_pose_acceptance(self):
        xy=np.array([[.7,-.4]]);v=np.array([[[.6,-.1,.3,25,.2]]])
        self.assertTrue(known_request_match(xy,v,v,xy)[0,0])
        jitter=v.copy();jitter[0,0,2]+=1e-7
        self.assertFalse(known_request_match(xy,jitter,v,xy)[0,0])
        with self.assertRaises(ValueError):known_request_match(xy,v,v,None)

    def test_collision_on_lowest_residual_branch_does_not_discard_alternative(self):
        first, second = self.fixture(), self.fixture()
        gate = lambda a, b: dict(reason=4 if b[0] == 0 else 0, pairs=[])
        i, j, _, tried = select_pair(first, second, gate)
        self.assertEqual((i, j), (0, 1))
        self.assertEqual([x["reason"] for x in tried], [4, 4, 0])

    def test_exact_valid_known_pair_has_priority_without_residual_optimization(self):
        first, second = self.fixture(), self.fixture()
        order = ordered_pairs(first, second, known10=first["q"][1], known14=second["q"][1])
        self.assertEqual(order[0], (1, 1))
        calls = []
        def gate(a,b): calls.append((a.copy(),b.copy())); return dict(reason=0,pairs=[])
        i,j,_,tried=select_pair(first,second,gate,known10=first["q"][1],known14=second["q"][1])
        self.assertEqual((i,j), (1,1)); self.assertEqual(len(tried),1)
        np.testing.assert_array_equal(calls[0][0],first["q"][1])

    def test_invalid_known_branch_cannot_bypass_native_gate(self):
        first, second=self.fixture(),self.fixture()
        i,j,checked,tried=select_pair(first,second,lambda a,b:dict(reason=5,pairs=[]),
            known10=first["q"][1],known14=second["q"][1])
        self.assertIsNone(checked); self.assertIsNone(i);self.assertIsNone(j)
        self.assertEqual(len(tried),4)


@unittest.skipIf(torch is None, "Pinned Ubuntu CPU Torch required")
class TorchStableTests(unittest.TestCase):
    def setUp(self):
        from sircus.mjlab.tests.test_learning_ik import fixture
        self.tmp=tempfile.TemporaryDirectory();p=Path(self.tmp.name)/"arm.urdf";fixture(p)
        self.solver=StableBatchIK.from_urdf(p,role="test")
        self.q=np.random.default_rng(129).uniform(-1,1,(3,6))
        self.targets=self.solver.fk(self.q).numpy()

    def tearDown(self): self.tmp.cleanup()

    def test_starts_are_independent_of_global_rng_order_and_batch(self):
        start,_,ids=self.solver.starts_for(self.targets,starts=5)
        torch.manual_seed(999);np.random.seed(999)
        perm=np.array([2,0,1]);other,_,oid=self.solver.starts_for(self.targets[perm],starts=5)
        np.testing.assert_array_equal(start.numpy()[perm],other.numpy())
        self.assertEqual([ids[i] for i in perm],oid)
        one,_,_=self.solver.starts_for(self.targets[1:2],starts=5)
        np.testing.assert_array_equal(start.numpy()[1],one.numpy()[0])

    def test_results_do_not_depend_on_hard_neighbours_or_chunking(self):
        target=self.targets.copy();target[2,0,3]=100
        full=self.solver.solve_candidates(target,initial=self.q+.05,starts=4,iterations=50)
        perm=np.array([2,0,1]);mixed=self.solver.solve_candidates(target[perm],initial=(self.q+.05)[perm],starts=4,iterations=50)
        np.testing.assert_array_equal(full['valid'].numpy()[perm],mixed['valid'].numpy())
        np.testing.assert_allclose(full['q'].numpy()[perm],mixed['q'].numpy(),atol=1e-12,rtol=0)
        for i in range(2):
            one=self.solver.solve_candidates(target[i:i+1],initial=self.q[i:i+1]+.05,starts=4,iterations=50)
            np.testing.assert_array_equal(full['valid'].numpy()[i],one['valid'].numpy()[0])
            np.testing.assert_allclose(full['q'].numpy()[i],one['q'].numpy()[0],atol=1e-12,rtol=0)

    def test_valid_known_branch_is_kept_exactly_with_unreachable_neighbour(self):
        target=self.targets.copy();target[2,0,3]=100
        out=self.solver.solve_candidates(target,known=self.q,starts=4,iterations=60)
        self.assertTrue(bool(out['valid'][:2,0].all()))
        np.testing.assert_array_equal(out['q'].numpy()[:2,0],self.q[:2])
        np.testing.assert_array_equal(out['converged_at'].numpy()[:2,0],0)
        self.assertFalse(bool(out['valid'][2].any()))

    def test_unavailable_or_outside_limit_hint_cannot_be_claimed_exact(self):
        out=self.solver.solve_candidates(self.targets[:1],known=np.full((1,6),np.nan),starts=2,iterations=5)
        self.assertFalse(bool(out['valid'][0,0]))
        known=self.q[:1].copy();known[0,0]=99
        out=self.solver.solve_candidates(self.targets[:1],known=known,starts=2,iterations=5)
        self.assertFalse(np.array_equal(out['q'].numpy()[0,0],known[0]))

    def test_small_nonzero_jitter_is_polished_not_erased_by_acceptance_tolerance(self):
        target=self.targets[:1].copy();target[0,0,3]+=2e-5
        out=self.solver.solve_candidates(target,known=self.q[:1],starts=2,iterations=80)
        self.assertTrue(bool(out['valid'][0,0]))
        self.assertFalse(np.array_equal(out['q'].numpy()[0,0],self.q[0]))
        self.assertLess(float(out['residual'][0,0]),1e-8)

    def test_provenance_matched_nominal_can_preserve_historical_valid_residual(self):
        target=self.targets[:1].copy();target[0,0,3]+=2e-5
        out=self.solver.solve_candidates(target,known=self.q[:1],preserve_known=[True],starts=2,iterations=80)
        self.assertTrue(bool(out['valid'][0,0]))
        np.testing.assert_array_equal(out['q'].numpy()[0,0],self.q[0])
        self.assertGreater(float(out['residual'][0,0]),1e-8)
        self.assertLess(float(out['residual'][0,0]),1e-4)


if __name__ == '__main__': unittest.main()
