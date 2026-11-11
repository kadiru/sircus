"""Small synthetic ledgers exercise independent accounting without physics/GPU."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from sircus.mjlab import fixed35_broad_schedule_verify as V
from sircus.mjlab.benchmark_verify import Checks, EvidenceError, digest, sha


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(root, *, proposals=5, capture=False, decline=False, cutoff=None):
    for name in ('requests', 'preparation-requests', 'prepared', 'repeat-plans', 'native-yaw'):
        (root/name).mkdir()
    plan = [dict(id=f'proposal-{c}-{i}', context_id=c, phase='search', method='broad_search',
                 values=[.6 + i*.0001, -.1, .3, 20., .1], bucket_xy_m=[1., .1])
            for c in V.CASE_IDS for i in range(proposals)]
    write(root/'search-plan.json', dict(rows=plan))
    kept = [dict(id=f'kept-{i}', phase='preserved', method='preserved') for i in range(30)]
    rows, boundaries, timings, commits, preparations, raw = [], [], [], [], [], []
    clock = 0.

    def prefix():
        return hashlib.sha256(b''.join(raw)).hexdigest() if raw else None

    def boundary(phase, **extra):
        nonlocal clock
        clock += 1
        boundaries.append(dict(phase=phase, rows_count=len(rows), rows_sha256=prefix(), acquired_elapsed_s=clock, **extra))

    def batch(expected, phase):
        nonlocal clock
        result = []
        for offset in range(0, len(expected), 32):
            chunk = expected[offset:offset+32]
            packet = f'requests/batch-{len(commits):05d}.json'
            write(root/packet, dict(phase=phase, rows=chunk, pre_outcome=True))
            start = clock + 1
            clock += 2
            cache = f'cache-{len(commits):05d}.bin'
            (root/'native-yaw'/cache).write_text(packet)
            group = []
            for request in chunk:
                success = phase == 'preserved' or (capture and phase in ('search', 'repeat'))
                row = dict(request, native_release=True, route_success=success, invalid=None, invalid_physics=None,
                    q10=request.get('selected_q10', [0.]*6), q14=request.get('selected_q14', [0.]*6),
                    connection_sha256=request.get('prepared_connection_sha256', 'connection'),
                    request_packet=packet, request_packet_sha256=sha(root/packet),
                    adapter_directory='native-yaw', native_cache_file=cache, native_cache_sha256=sha(root/'native-yaw'/cache),
                    scoring_version='sircus-table-assisted-native-capture-v3',
                    route_version='sircus-table-assisted-board-route-v3', reward_version='sircus-table-assisted-shot-reward-v3',
                    board_bounce=True, closest_opening_distance_m=.1, forbidden_contact_steps=0,
                    max_retention_s=float(success), route_quality=2 if success else 0,
                    single_board_bounce_success=success)
                if phase in ('search', 'repeat'):
                    row['prepared_request'] = copy.deepcopy(request)
                rows.append(row)
                raw.append((json.dumps(row)+'\n').encode())
                group.append(row)
            timings.append(dict(phase=phase, request_packet=packet, request_packet_sha256=sha(root/packet),
                acquired_start_s=start, acquired_end_s=clock))
            commits.append(dict(packet=packet, packet_sha256=sha(root/packet), row_ids=[r['id'] for r in group],
                observed_rows_sha256=digest(group), rows_count=len(rows), rows_sha256=prefix(),
                native_files={'native-yaw/'+cache: sha(root/'native-yaw'/cache)}))
            result.extend(group)
        return result

    def controls(phase):
        batch([dict(id=f'{phase}-{i}', phase=phase, method='control') for i in range(9)], phase)
        boundary(phase+'-complete')

    boundary('inputs-frozen')
    controls('controls_pre')
    batch(kept, 'preserved')
    boundary('preserved-complete')
    cursors = dict.fromkeys(V.CASE_IDS, 0)
    counts = dict.fromkeys(V.CASE_IDS, 0)
    groups = {c: [] for c in V.CASE_IDS}
    round_number = 0
    while counts != V.QUOTAS:
        if cutoff is not None and round_number == cutoff:
            clock = 36000
            break
        wanted = []
        for _ in range(4):
            for c in V.CASE_IDS:
                if counts[c] < V.QUOTAS[c] and cursors[c] < proposals:
                    wanted.append(next(r for r in plan if r['id'] == f'proposal-{c}-{cursors[c]}'))
                    cursors[c] += 1
        if not wanted:
            break
        packet = f'preparation-requests/batch-{round_number:05d}.json'
        prepared_file = f'prepared/batch-{round_number:05d}.json'
        write(root/packet, dict(rows=wanted, search_plan_sha256=sha(root/'search-plan.json'), pre_outcome=True))
        admitted, records = [], []
        for request in wanted:
            admitted_here = []
            for slot in range(0 if decline else 2):
                value = dict(request, id=request['id']+f'-b{slot}', proposal_id=request['id'], native_prepared=True,
                    selected_q10=[0.]*6, selected_q14=[slot*.1]*6, prepared_connection_sha256=request['id']+str(slot))
                admitted_here.append(value)
                admitted.append(value)
            records.append(dict(proposal_id=request['id'], proposal_sha256=digest(request), context_id=request['context_id'],
                values=request['values'], bucket_xy_m=request['bucket_xy_m'], admitted=len(admitted_here),
                admitted_request_ids=[r['id'] for r in admitted_here], admitted_request_sha256=[digest(r) for r in admitted_here],
                native_releases=0, preparation_only=True, reason='finite_ik_unresolved' if decline else None))
        selected, omitted = [], []
        used = dict(counts)
        for r in admitted:
            if used[r['context_id']] < V.QUOTAS[r['context_id']]:
                selected.append(r)
                used[r['context_id']] += 1
            else:
                omitted.append(r['id'])
        write(root/prepared_file, dict(requests=selected, preparations=records, not_executed_due_quota=omitted,
            preparation_request_file=packet, preparation_request_sha256=sha(root/packet), frozen_before_native_outcomes=True))
        preparations.extend(dict(r, preparation_request_file=packet, preparation_request_sha256=sha(root/packet),
                                 prepared_file=prepared_file, prepared_sha256=sha(root/prepared_file)) for r in records)
        clock += 2
        boundary('preparation-committed', round=round_number, request_packet=packet, prepared_file=prepared_file,
                 prepared_sha256=sha(root/prepared_file), elapsed_preparation_s=2.)
        observed = batch([dict(r, prepared_batch_file=prepared_file, prepared_batch_sha256=sha(root/prepared_file)) for r in selected], 'search')
        discoveries = []
        for row in observed:
            c = row['context_id']
            counts[c] += 1
            if row['route_success'] and len(groups[c]) < 3 and row['values'] not in groups[c]:
                group = len(groups[c])
                groups[c].append(row['values'])
                discoveries.append((row, group))
        if discoveries:
            repeated = [r for source, group in discoveries for r in V.independent_repeat(source, group)]
            path = f'repeat-plans/group-{round_number:05d}.json'
            write(root/path, dict(rows=repeated, selected=[dict(id=r['id'], sha256=digest(r)) for r, _ in discoveries],
                rows_count_before=len(rows), rows_sha256_before=prefix(), frozen_before_repeat_outcomes=True))
            batch([dict(r, repeat_plan_file=path, repeat_plan_sha256=sha(root/path)) for r in repeated], 'repeat')
        round_number += 1
        boundary('search-round-complete', round=round_number, cursors=copy.deepcopy(cursors))
    done = counts == V.QUOTAS
    truncated = cutoff is not None and round_number == cutoff and not done
    reason = '1000_fresh_search_drops_completed' if done else 'search_time_limit' if truncated else 'finite_reachable_proposal_pool_exhausted'
    boundary('search-complete', reason=reason, cursors=cursors)
    controls('controls_post')
    report = dict(status='completed' if done else 'incomplete', completion_reason=reason, budget_truncated=truncated,
        deadlines=dict(effective_worker_limit_s=43200, search_limit_s=36000), fresh_search_releases=counts,
        cases=V.independent_cases(rows))
    (root/'rows.jsonl').write_bytes(b''.join(raw))
    for name, values in (('phase-boundaries.jsonl', boundaries), ('timing.jsonl', timings),
                         ('packet-commits.jsonl', commits), ('preparations.jsonl', preparations)):
        (root/name).write_text(''.join(json.dumps(r)+'\n' for r in values))
    return rows, report, {}, dict(rows=plan), kept


def check(root, parts):
    return V.audit_schedule(root, *parts, V.independent_repeat, Checks())


class BroadScheduleTests(unittest.TestCase):
    def test_native_quota_is_exact_and_not_candidate_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = check(root, fixture(root, proposals=170))
            self.assertEqual(result['fresh_search_releases'], V.QUOTAS)
            self.assertEqual(result['total_requests'], 1048)
            self.assertEqual(result['preparation_count'], 504)

    def test_distinct_numeric_confirmation_once_per_setting_search_continues(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parts = fixture(root, proposals=5, capture=True)
            result = check(root, parts)
            self.assertEqual(result['repeat_requests'], 27)
            self.assertEqual(sum(result['fresh_search_releases'].values()), 30)
            self.assertEqual(len(parts[1]['cases'][0]['confirmed_recipes']), 3)

    def test_preparation_declines_never_count_as_releases(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = check(root, fixture(root, proposals=5, decline=True))
            self.assertEqual(result['preparation_declines'], 15)
            self.assertEqual(result['fresh_search_releases'], dict.fromkeys(V.CASE_IDS, 0))
            self.assertEqual(result['total_requests'], 48)

    def test_honest_timed_prefix_is_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = check(root, fixture(root, proposals=10, cutoff=1))
            self.assertTrue(result['budget_truncated'])
            self.assertFalse(result['search_quota_completed'])

    def test_false_complete_missing_native_outcome_and_wrong_quota_are_rejected(self):
        for mutation in ('status', 'count', 'missing'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                parts = list(fixture(root))
                if mutation == 'status': parts[1]['status'] = 'completed'
                elif mutation == 'count': parts[1]['fresh_search_releases'][V.CASE_IDS[0]] += 1
                else: parts[0].pop(40)
                with self.assertRaises(EvidenceError): check(root, parts)

    def test_unaccounted_packets_and_corrupt_native_archives_are_rejected(self):
        for mutation in ('extra', 'archive', 'receipt'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                parts = fixture(root)
                if mutation == 'extra': write(root/'prepared/extra.json', {})
                elif mutation == 'archive': (root/'native-yaw/cache-00002.bin').write_bytes(b'changed')
                else:
                    values = (root/'packet-commits.jsonl').read_text().splitlines()
                    row = json.loads(values[2]); row['rows_sha256'] = 'wrong'; values[2] = json.dumps(row)
                    (root/'packet-commits.jsonl').write_text('\n'.join(values)+'\n')
                with self.assertRaises(EvidenceError): check(root, parts)

    def test_preparation_order_and_native_classification_cannot_be_relabelled(self):
        for mutation in ('order', 'release'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                parts = fixture(root)
                if mutation == 'order':
                    path = root/'preparation-requests/batch-00000.json'
                    value = json.loads(path.read_text()); value['rows'].reverse(); write(path, value)
                else:
                    parts[0][39]['native_release'] = False
                    (root/'rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in parts[0]))
                with self.assertRaises(EvidenceError): check(root, parts)

    def test_confirmation_requires_identical_joint_and_connection(self):
        row = dict(context_id=V.CASE_IDS[0], phase='repeat', confirmation_index=0, native_release=True,
            route_success=True, selected_from_id='source', selected_from_sha256='sha', values=[1]*5,
            bucket_xy_m=[1, 2], q10=[0]*6, q14=[1]*6, connection_sha256='connection')
        rows = [dict(row, id=f'repeat{i}', repeat=i) for i in range(3)]
        self.assertEqual(V.independent_groups(rows, V.CASE_IDS[0])[0]['status'], 'confirmed')
        rows[-1]['q14'] = [2]*6
        self.assertEqual(V.independent_groups(rows, V.CASE_IDS[0])[0]['status'], 'failed')


if __name__ == '__main__': unittest.main()
