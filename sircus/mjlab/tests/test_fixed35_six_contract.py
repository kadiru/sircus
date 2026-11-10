"""Immutable29 recipe sources and bounded coupled-pose/distinct-branch proposals."""
import copy
from pathlib import Path
import unittest

import numpy as np

from sircus.mjlab import fixed35_six_contract as C
from sircus.mjlab import fixed35_recovery_contract as R
from sircus.mjlab.tests.test_fixed35_recovery_contract import fixture as parent_fixture, score_fields


def fixture():
    old = parent_fixture()
    rows = []
    for source, request in zip(old['preserved'], R.preserved_plan(old['preserved'])):
        rows.append(dict(copy.deepcopy(source), **request))
    cases = []
    for source in old['cases']+old['blocked']:
        if source['context_id'] in C.CASE_IDS+C.BLOCKED_IDS:
            cases.append(dict(context_id=source['context_id'], confirmed_recipe=None))
            continue
        discovery = dict(copy.deepcopy(source), **R.request(source,
            identity='recovered-'+source['context_id'], phase='search', method='recovery_search', proposal_index=0))
        discovery.update(score_fields(True))
        discovery['values'][4] += .01
        discovery['command_values'] = copy.deepcopy(discovery['values'])
        rows.append(discovery)
        for request in R.repeat_plan(discovery, 0):
            rows.append(dict(copy.deepcopy(discovery), **request))
        cases.append(dict(context_id=source['context_id'], confirmed_recipe=copy.deepcopy(discovery)))
    report = dict(cases=cases)
    original_rows = old['current']+old['refit']
    result = C.cohort(original_rows, rows, report)
    return dict(result, original_rows=original_rows, recovery_rows=rows, recovery_report=report)


class ContractTests(unittest.TestCase):
    def test_source_cohort_has29_successes_four_search_cases_and_two_cpu_only(self):
        data = fixture()
        self.assertEqual(len(data['preserved']), 29)
        self.assertEqual(len({r['context_id'] for r in data['preserved']}), 29)
        self.assertEqual(tuple(r['context_id'] for r in data['cases']), C.CASE_IDS)
        self.assertEqual(tuple(r['context_id'] for r in data['blocked']), C.BLOCKED_IDS)
        self.assertEqual(C.protocol()['maximum_requests'], 768+36+29+18)
        self.assertEqual(C.protocol()['maximum_requests'], 851)
        self.assertFalse(C.protocol()['training'])
        self.assertFalse(C.protocol()['ui_changes'])
        self.assertFalse(C.protocol()['bucket_perturbations'])

    def test_captured_discovery_without_three_exact_confirmations_cannot_be_preserved(self):
        for mutation in ('missing', 'miss', 'branch', 'hash'):
            data = fixture(); rows = data['recovery_rows']
            repeat = next(r for r in rows if r['phase'] == 'repeat')
            if mutation == 'missing':
                rows.remove(repeat)
            elif mutation == 'miss':
                repeat['route_success'] = False
            elif mutation == 'branch':
                repeat['q10'] = [99.]*6
            else:
                repeat['selected_from_sha256'] = 'wrong'
            with self.assertRaises(ValueError):
                C.cohort(data['original_rows'], rows, data['recovery_report'])

    def test_changed_preserved_parent_or_missing_original_cannot_enter_cohort(self):
        for kind in ('branch', 'parent_hash', 'capture', 'grid'):
            data = fixture(); first = data['recovery_rows'][0]
            if kind == 'branch':
                first['q14'] = [99.]*6
            elif kind == 'parent_hash':
                first['parent_row_sha256'] = 'wrong'
            elif kind == 'capture':
                first['route_success'] = False
            else:
                data['original_rows'][0]['bucket_xy_m'][0] += .001
            with self.assertRaises(ValueError):
                C.cohort(data['original_rows'], data['recovery_rows'], data['recovery_report'])

    def test96_unique_numeric_actions_with_paired_branch_slots_and_frozen_buckets(self):
        data = fixture(); before = copy.deepcopy(data)
        plan = C.make_plan(data['cases'], data['refit'], data['preserved'])
        self.assertEqual(len(plan), 768)
        self.assertEqual(data, before)
        self.assertEqual(len({r['id'] for r in plan}), 768)
        for case in data['cases']:
            subset = [r for r in plan if r['context_id'] == case['context_id']]
            self.assertEqual(len({tuple(r['values']) for r in subset}), 96)
            for index in range(96):
                pair = subset[index*2:index*2+2]
                self.assertEqual([r['ik_branch_slot'] for r in pair], [0, 1])
                self.assertEqual(pair[0]['values'], pair[1]['values'])
                self.assertTrue(all(r['bucket_xy_m'] == case['bucket_xy_m'] for r in pair))
                self.assertTrue(all(r['parent_row_sha256'] == C.digest(case) for r in pair))
                self.assertTrue(all(r['values'] == r['command_values'] for r in pair))
                self.assertTrue(all(r['exact_parent_action'] is False for r in pair))
        self.assertFalse(set(C.BLOCKED_IDS) & {r['context_id'] for r in plan})
        for index in range(192):
            group = plan[index*4:index*4+4]
            self.assertEqual([r['context_id'] for r in group], list(C.CASE_IDS))
            self.assertTrue(all(r['proposal_index'] == index for r in group))

    def test_new_coupled_geometry_is_not_duplicate_old80_or_clipped(self):
        data = fixture(); case = data['cases'][0]
        proposals = C.proposals(case, data['refit'], data['preserved'])
        old = R.proposals(case, data['refit'], data['preserved'])
        old_values = {tuple(r['values']) for r in old}
        self.assertGreater(sum(tuple(r['values']) not in old_values for r in proposals), 80)
        coupled = [r for r in proposals if r['kind'] == 'coupled_polar_height_tilt_gap']
        self.assertGreater(len(coupled), 60)
        for row in proposals:
            v = np.asarray(row['values'])
            self.assertTrue((v >= C.LOW).all() and (v <= C.HIGH).all())
            self.assertNotIn('clipped', row)
            if 'radius_m' in row:
                self.assertAlmostEqual(np.linalg.norm(v[:2]-case['bucket_xy_m']), row['radius_m'])
        self.assertGreater(np.ptp([r['radius_m'] for r in coupled]), .5)
        self.assertGreater(np.ptp([r['values'][2] for r in coupled]), .35)
        self.assertGreater(np.ptp([r['values'][3] for r in coupled]), 35.)
        self.assertGreater(np.ptp([r['values'][4] for r in coupled]), .4)

    def test_edge_actions_are_omitted_instead_of_silently_clipped(self):
        data = fixture(); source = copy.deepcopy(data['cases'][0])
        source['values'] = [.1, -.51, .05, 2., .01]
        proposals = C.proposals(source, data['refit'], data['preserved'])
        self.assertEqual(len(proposals), 96)
        self.assertFalse(any(r['kind'] == 'original_failed_action' for r in proposals))
        self.assertNotIn(np.clip(source['values'], C.LOW, C.HIGH).tolist(), [r['values'] for r in proposals])

    def test_exact_preservation_has_no_new_branch_request_and_is_immutable(self):
        data = fixture(); before = copy.deepcopy(data['preserved'])
        requests = C.preserved_plan(data['preserved'])
        for request, source in zip(requests, data['preserved']):
            self.assertTrue(request['exact_parent_action'])
            self.assertNotIn('ik_branch_slot', request)
            self.assertEqual(request['values'], source['values'])
            self.assertEqual(request['parent_row_sha256'], C.digest(source))
        requests[0]['values'][0] = 999
        self.assertEqual(data['preserved'], before)
        with self.assertRaises(ValueError):
            C.preserved_plan(data['preserved'][:-1])

    def test_repeat_keeps_selected_branch_and_requires_three_distinct_complete_rows(self):
        data = fixture(); request = C.make_plan(data['cases'], data['refit'], data['preserved'])[4]
        discovery = dict(request, **score_fields(True), connection_sha256='c'*64, q10=[.1]*6, q14=[.2]*6)
        repeats = C.repeat_plan(discovery, 0)
        self.assertEqual(len(repeats), 3)
        self.assertTrue(all(r['ik_branch_slot'] == 1 and r['selected_q10'] == discovery['q10'] and
                            r['selected_q14'] == discovery['q14'] and r['selected_from_sha256'] == C.digest(discovery)
                            for r in repeats))
        observed = [dict(r, **score_fields(True), q10=discovery['q10'], q14=discovery['q14'],
                         connection_sha256=discovery['connection_sha256']) for r in repeats]
        self.assertEqual(C.confirmation_groups(observed, discovery['context_id'])[0]['status'], 'confirmed')
        observed[-1] = copy.deepcopy(observed[0])
        self.assertEqual(C.confirmation_groups(observed, discovery['context_id'])[0]['status'], 'failed')
        with self.assertRaises(ValueError):
            C.repeat_plan(dict(discovery, invalid_physics=True), 0)

    def test_blocked_and_time_truncated_cases_never_claim_complete_search(self):
        data = fixture()
        outcomes = C.case_results([], data['cases'], data['blocked'], finished=True, budget_truncated=True)
        self.assertEqual(sum(r['status'] == 'fixed_base_clearance_blocked' for r in outcomes), 2)
        self.assertEqual(sum(r['status'] == 'budget_truncated' for r in outcomes), 4)
        self.assertTrue(all(r['search_requests'] == 0 and r['confirmed_recipe'] is None for r in outcomes))

    def test_checked_in_rule_matches_frozen_protocol(self):
        self.assertEqual(C.read(Path(C.__file__).with_name('fixed35_six_rule.json'))['protocol'], C.protocol())


if __name__ == '__main__':
    unittest.main()
