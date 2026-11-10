"""Independent reconstruction rejects changed poses, omitted branches and stale repeats."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from sircus.mjlab import fixed35_six_contract as producer
from sircus.mjlab import fixed35_six_schedule_verify as verify
from sircus.mjlab.tests.test_fixed35_six_contract import fixture, score_fields
from sircus.mjlab.replay650_verify import AuditChecks
from sircus.mjlab.benchmark_verify import EvidenceError


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True)+'\n')


def evidence(directory, *, discovery=True, repeats=True, cutoff=False):
    data = fixture()
    plan = producer.make_plan(data['cases'], data['refit'], data['preserved'])
    write(directory/'search-plan.json', dict(rows=plan, frozen_before_outcomes=True, maximum_requests=768))
    write(directory/'preserved-plan.json', dict(rows=producer.preserved_plan(data['preserved']), frozen_before_outcomes=True))
    (directory/'repeat-plans').mkdir()
    rows, boundaries, timings = [], [], []
    clock = 0.
    sources = {r['id']: r for r in data['preserved']+data['failures']}

    def ledger():
        return ''.join(json.dumps(r, sort_keys=True)+'\n' for r in rows).encode()

    def prefix():
        return hashlib.sha256(ledger()).hexdigest() if rows else None

    def boundary(phase, **extra):
        boundaries.append(dict(phase=phase, acquired_elapsed_s=clock, rows_count=len(rows),
                               rows_sha256=prefix(), **copy.deepcopy(extra)))

    def batch(phase, requests, success=True):
        nonlocal clock
        start = clock; clock += .1; observed = []
        for request in requests:
            row = dict(copy.deepcopy(request), **score_fields(success))
            original = sources.get(row.get('parent_row_id'))
            row.update(connection_file=row.get('selected_connection_file', 'fixed-yaw-connections/mock.json'),
                       connection_sha256=row.get('selected_connection_sha256', 'a'*64),
                       q10=copy.deepcopy(row.get('selected_q10', original['q10'] if original else [0.]*6)),
                       q14=copy.deepcopy(row.get('selected_q14', original['q14'] if original else [0.]*6)),
                       native_cache_sha256=hashlib.sha256(phase.encode()).hexdigest())
            observed.append(row)
        rows.extend(observed)
        timings.append(dict(phase=phase, acquired_start_s=start, acquired_end_s=clock))
        return observed

    def controls(phase):
        batch(phase, [dict(id=phase+str(i), phase=phase) for i in range(9)])
        boundary(phase+'-complete')

    boundary('inputs-frozen')
    controls('controls_pre')
    batch('preserved', producer.preserved_plan(data['preserved']))
    boundary('preserved-complete')
    active = list(producer.CASE_IDS); stopped = {}; groups = {c: 0 for c in active}
    for index in range(192):
        active = [c for c in active if c not in stopped]
        if not active:
            break
        requests = [r for r in plan if r['proposal_index'] == index and r['context_id'] in active]
        boundary('search-round-frozen', proposal_index=index, contexts=active)
        discovered = batch('search', [dict(r, search_plan_sha256=verify.sha(directory/'search-plan.json')) for r in requests], discovery)
        if discovery:
            selected = [r for d in discovered for r in producer.repeat_plan(d, groups[d['context_id']])]
            name = f'repeat-plans/round-{index:02d}.json'
            write(directory/name, dict(proposal_index=index, rows=selected,
                selected=[dict(id=r['id'], sha256=verify.digest(r)) for r in discovered],
                rows_count_before=len(rows), rows_sha256_before=prefix(), frozen_before_repeat_outcomes=True))
            boundary('repeat-plan-frozen', proposal_index=index, path=name, sha256=verify.sha(directory/name))
            batch('repeat', [dict(r, repeat_plan_file=name, repeat_plan_sha256=verify.sha(directory/name)) for r in selected], repeats)
            for d in discovered:
                context = d['context_id']; groups[context] += 1
                if repeats:
                    stopped[context] = 'confirmed'
                elif groups[context] == 3:
                    stopped[context] = 'confirmation_limit'
        if cutoff:
            clock = 5401.
        boundary('search-round-complete', proposal_index=index, stopped=stopped)
        if cutoff:
            break
    reason = ('search_time_limit' if cutoff else 'all_searchable_cases_confirmed' if len(stopped) == 4 and repeats and discovery
              else 'finite_candidate_or_confirmation_caps')
    report = dict(budget_truncated=cutoff, completion_reason=reason)
    boundary('search-complete', completion_reason=reason, budget_truncated=cutoff)
    controls('controls_post')
    (directory/'rows.jsonl').write_bytes(ledger())
    for name, content in (('phase-boundaries.jsonl', boundaries), ('timing.jsonl', timings)):
        (directory/name).write_text(''.join(json.dumps(r, sort_keys=True)+'\n' for r in content))
    return data, plan, rows, report


class ScheduleTests(unittest.TestCase):
    def test_literal_independent_plan_matches_current_producer_without_importing_it(self):
        data = fixture()
        wanted = producer.make_plan(data['cases'], data['refit'], data['preserved'])
        with patch.object(producer, 'make_plan', side_effect=AssertionError('producer planner called')):
            self.assertEqual(verify.independent_plan(data), wanted)
        self.assertEqual(verify.independent_preserved(data), producer.preserved_plan(data['preserved']))
        source = Path(verify.__file__).read_text()
        self.assertNotIn('import fixed35_six_contract', source)
        self.assertNotIn('import fixed35_six_study', source)

    def test_every_plan_field_and_pair_order_are_bound(self):
        for tamper in (None, 'action', 'slot', 'skip', 'preserved'):
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp); data, _, _, _ = evidence(path)
                if tamper:
                    filename = 'preserved-plan.json' if tamper == 'preserved' else 'search-plan.json'
                    packet = json.loads((path/filename).read_text())
                    if tamper == 'action': packet['rows'][0]['values'][4] += .001
                    elif tamper == 'slot': packet['rows'][0]['ik_branch_slot'] = 1
                    elif tamper == 'skip': packet['rows'].pop()
                    else: packet['rows'][0]['exact_parent_action'] = False
                    write(path/filename, packet)
                    with self.assertRaises(EvidenceError): verify.audit_plan(path, data, AuditChecks())
                else:
                    self.assertEqual(len(verify.audit_plan(path, data, AuditChecks())), 768)

    def test_complete_confirmation_failed_groups_and_cutoff_are_independently_accounted(self):
        for options, expected in (({}, (63, 4, 12)),
                                  ({'repeats': False}, (95, 12, 36)),
                                  ({'discovery': False}, (815, 768, 0)),
                                  ({'discovery': False, 'cutoff': True}, (51, 4, 0))):
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp); data, plan, rows, report = evidence(path, **options)
                result = verify.audit_schedule(path, rows, report, data, plan, AuditChecks())
                self.assertEqual((result['total_requests'], result['search_requests'], result['repeat_requests']), expected)
                self.assertEqual(verify.independent_cases(rows, data, finished=True, budget_truncated=report['budget_truncated']),
                                 producer.case_results(rows, data['cases'], data['blocked'], finished=True, budget_truncated=report['budget_truncated']))

    def test_false_early_stop_and_fake_time_truncation_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp); data, plan, rows, report = evidence(path, discovery=False, cutoff=True)
            report.update(budget_truncated=False, completion_reason='finite_candidate_or_confirmation_caps')
            with self.assertRaises(EvidenceError): verify.audit_schedule(path, rows, report, data, plan, AuditChecks())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp); data, plan, rows, report = evidence(path, discovery=False, cutoff=True)
            boundaries = [json.loads(r) for r in (path/'phase-boundaries.jsonl').read_text().splitlines()]
            for row in boundaries:
                if row['acquired_elapsed_s'] >= 5400: row['acquired_elapsed_s'] -= 5000
            write_lines = lambda name, content: (path/name).write_text(''.join(json.dumps(r)+'\n' for r in content))
            write_lines('phase-boundaries.jsonl', boundaries)
            with self.assertRaises(EvidenceError): verify.audit_schedule(path, rows, report, data, plan, AuditChecks())

    def test_changed_confirmation_branch_or_cached_discovery_cannot_pass(self):
        for key in ('ik_branch_slot', 'q10', 'native_cache_sha256'):
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp); data, plan, rows, report = evidence(path)
                repeated = next(r for r in rows if r['phase'] == 'repeat')
                if key == 'ik_branch_slot': repeated[key] = 1
                elif key == 'q10': repeated[key] = [999.]*6
                else: repeated[key] = next(r['native_cache_sha256'] for r in rows if r['phase'] == 'search')
                with self.assertRaises(EvidenceError): verify.audit_schedule(path, rows, report, data, plan, AuditChecks())

    def test_extra_blocked_request_or_repeat_file_cannot_be_hidden(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp); data, plan, rows, report = evidence(path)
            write(path/'repeat-plans/extra.json', {})
            with self.assertRaises(EvidenceError): verify.audit_schedule(path, rows, report, data, plan, AuditChecks())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp); data, plan, rows, report = evidence(path)
            rows.append(dict(rows[-1], id='hidden', context_id=producer.BLOCKED_IDS[0]))
            with self.assertRaises(EvidenceError): verify.audit_schedule(path, rows, report, data, plan, AuditChecks())


if __name__ == '__main__':
    unittest.main()
