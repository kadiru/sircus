import unittest
from pathlib import Path
import numpy as np
from sircus.mjlab.expanded_actions import project,checked_bounds,WIDE_BOUNDS
from sircus.mjlab.learning_contract import project as original, constraints,LOW,HIGH


class ActionRangeTests(unittest.TestCase):
    def test_original_projection_identical(self):
        rng=np.random.default_rng(401)
        v=rng.uniform(LOW-.05,HIGH+.05,(50,5));locks=constraints(50)
        locks["board"][::3]=[.6,-.1,.3];locks["dropper"][::7]=[.6,-.1,.7]
        a,am=original(v,locks);b,bm=project(v,locks)
        np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(am,bm)

    def test_wide_explicit_not_default(self):
        v=np.array([[.6,-.4,.14,50.,.55]])
        self.assertFalse(project(v,constraints(1))[1][0])
        self.assertTrue(project(v,constraints(1),bounds=WIDE_BOUNDS)[1][0])

    def test_bad_bounds_rejected(self):
        for b in ({'low':[0], 'high':[1]}, {'low':[0,0,0,-1,0], 'high':[1,1,1,91,1]}):
            with self.assertRaises(ValueError):checked_bounds(b)

    def test_native_integration_identical(self):
        root=Path(__file__).resolve().parents[1]
        old=(root/'learning_env.py').read_text();old=old[old.index('    def run(self,'):]
        new=(root/'expanded_env.py').read_text();new=new[new.index('    def run(self,'):]
        self.assertEqual(old,new.replace('self.project(values, locks,','project(values, locks,'))
