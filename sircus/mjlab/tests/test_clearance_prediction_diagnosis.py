import copy
import unittest

import numpy as np

from sircus.mjlab import arm_clearance_distance as distance
from sircus.mjlab import arm_clearance_selector as old
from sircus.mjlab import clearance_prediction_diagnosis as diagnosis


class DiagnosisCertificateTests(unittest.TestCase):
    def setUp(self):
        self.colliders = [dict(name='finger',
            planes=old.box_planes([0, 0, 0], [1, 1, 1]),
            vertices=distance.box_vertices([0, 0, 0], [1, 1, 1]))]
        self.radius = .07
        self.origin = [1.06, 1.06, 0]
        directions = lambda _, points: np.tile([2**-.5, 2**-.5, 0], (len(points), 1))
        self.proof = distance.certify_path(self.origin, [0, 0, 0], .01, [0, 0, 0],
            self.colliders, self.radius, directions)
        self.points = np.tile(self.origin, (self.proof['samples'], 1))

    def audit(self, proof):
        checks = diagnosis.Checks()
        diagnosis.independent_support_check(self.points, self.colliders, self.radius,
            self.proof['between_sample_allowance_m'], proof, checks, 'fixture')
        return checks

    def test_independent_bound_accepts_complete_valid_corner_proof(self):
        checks = self.audit(self.proof)
        self.assertTrue(self.proof['passed'])
        self.assertGreater(len(checks.rows), 4)
        self.assertTrue(all(row['passed'] for row in checks.rows))

    def test_forged_positive_clearance_is_rejected(self):
        proof = copy.deepcopy(self.proof)
        proof['minimum_clearance_lower_bound_m'] += .001
        with self.assertRaisesRegex(ValueError, 'reconstructed bound'):
            self.audit(proof)

    def test_duplicate_or_nonunit_certificate_is_rejected(self):
        proof = copy.deepcopy(self.proof)
        proof['refinements'].append(copy.deepcopy(proof['refinements'][0]))
        with self.assertRaisesRegex(ValueError, 'unique refined'):
            self.audit(proof)
        proof = copy.deepcopy(self.proof)
        proof['refinements'][0]['support_normals'][0] = [2, 0, 0]
        with self.assertRaisesRegex(ValueError, 'unit witness'):
            self.audit(proof)

    def test_false_failure_label_cannot_replace_numeric_pass(self):
        proof = copy.deepcopy(self.proof)
        proof['passed'] = False
        with self.assertRaisesRegex(ValueError, 'unchanged two-mm'):
            self.audit(proof)


if __name__ == '__main__': unittest.main()
