"""Protocol and independent accounting agree on synthetic CPU fixtures only."""
import copy
import unittest

import numpy as np

from sircus.mjlab import replay650_contract as protocol, replay650_verify as verifier
from sircus.mjlab.tests.test_replay650_verify import group_fixture, rows


class ReplayContractTests(unittest.TestCase):
    def test_literal_frozen_plans_match_independent_verifier(self):
        _, _, groups, _ = group_fixture()
        initial = protocol.plan(groups[:3], 'replay')
        window = protocol.plan(groups[3:], 'window')
        self.assertEqual(initial, verifier.expected_plan(groups[:3], 'replay'))
        self.assertEqual(window, verifier.expected_plan(groups[3:], 'window'))
        self.assertEqual((len(initial), len(window)), (18, 30))
        self.assertEqual(len({r['id'] for r in initial+window}), protocol.MAX_REQUESTS)
        initial[0]['q10'][0] = 1.
        self.assertEqual(groups[0]['q10'][0], 0.)

    def test_exact_calibrated_gate_and_failure_modes_match(self):
        data = rows()
        for r in data:
            r['phase'] = 'replay'
        def positions(delta):
            result = {r['id']: np.zeros((round(.350*240*r['substeps'])+1, 3)) for r in data}
            for r in data:
                if r['role'] == 'candidate' and r['substeps'] == 128:
                    result[r['id']][-1, 0] = delta
            return result
        for delta in (0., .005, np.nextafter(.005, np.inf)):
            self.assertEqual(protocol.replay_gate(data, positions(delta)), verifier.replay_gate(data, delta))
        for field, value in (('route_success', False), ('native_release', False),
                             ('invalid_physics', True), ('scalar_score_audit_passed', False)):
            changed = copy.deepcopy(data)
            next(r for r in changed if r['role'] == 'candidate')[field] = value
            gate = protocol.replay_gate(changed, positions(0.))
            self.assertFalse(gate['window_opened'])
            self.assertEqual(gate, verifier.replay_gate(changed, 0.))

    def test_window_rows_cannot_help_a_failed_replay_pass(self):
        data = rows()
        for r in data:
            r['phase'] = 'replay'
        changed = copy.deepcopy(data)
        next(r for r in changed if r['role'] == 'candidate')['route_success'] = False
        successful_window = [dict(r, id='window-'+r['id'], phase='window') for r in data]
        locations = {r['id']: np.zeros((round(.350*240*r['substeps'])+1, 3)) for r in data}
        self.assertEqual(protocol.replay_gate(changed, locations), protocol.replay_gate(changed+successful_window, locations))


if __name__ == '__main__':
    unittest.main()
