"""Small synthetic ledgers exercise independent accounting without physics/GPU."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from sircus.mjlab import fixed35_sixhour_schedule_verify as V
from sircus.mjlab.benchmark_verify import Checks, EvidenceError, digest, sha


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(root, *, proposals=5, capture=False, decline=False, cutoff=None, switch=False, early=False, historical=False, branches=2, window_cutoff=False):
    for name in ('requests', 'preparation-requests', 'prepared', 'repeat-plans', 'window-plans', 'native-yaw'):
        (root/name).mkdir()
    plan = [dict(id=f'proposal-{c}-{i}', context_id=c, phase='search', method='broad_search',
                 values=[.6 + i*.0001, -.1, .3, 20., .1], bucket_xy_m=[1., .1])
            for c in (V.TARGET,) for i in range(proposals)]
    write(root/'search-plan.json', dict(rows=plan))
    kept = [dict(id=f'kept-{i}', phase='preserved', method='preserved') for i in range(36)]
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
    counts = dict.fromkeys(V.CASE_IDS, 0)
    groups = {c: [] for c in V.CASE_IDS}
    round_number, cursor, stage = 0, 0, 'primary'
    active_plan = dict(rows=plan)
    plan_file = 'search-plan.json'
    all_prepared, history_rows, seen = {}, [], set()
    while sum(counts.values()) < 1200:
        if cutoff is not None and round_number == cutoff:
            clock = 16200
            break
        if switch and round_number == 1:
            clock = 9000 if early else 16201 if window_cutoff else 10801
            plan_file = 'window-plans/plan.json'
            window_rows = [dict(id=f'window-{c}-{i}', context_id=c, phase='search', method='sixhour_search',
                values=[.75+i*.001, -.1, .3, 20., .1], bucket_xy_m=[1., .1]) for c in V.CASE_IDS for i in range(2)]
            active_plan = dict(rows=window_rows, limit=6000-len(preparations), selected_after_rows_count=len(rows),
                selected_after_rows_sha256=prefix(), acquired_elapsed_s=clock, posthoc_development=True)
            write(root/plan_file, active_plan)
            boundary('window-plan-frozen', plan_file=plan_file, plan_sha256=sha(root/plan_file))
            cursor, stage = 0, 'window'
            if window_cutoff: break
        wanted = active_plan['rows'][cursor:cursor+12]
        cursor += len(wanted)
        if not wanted: break
        packet = f'preparation-requests/batch-{round_number:05d}.json'
        prepared_file = f'prepared/batch-{round_number:05d}.json'
        write(root/packet, dict(rows=wanted, search_plan_file=plan_file, search_plan_sha256=sha(root/plan_file), pre_outcome=True))
        admitted, records = [], []
        for request in wanted:
            admitted_here = []
            for slot in range(0 if decline else branches):
                value = dict(request, id=request['id']+f'-b{slot}', proposal_id=request['id'], native_prepared=True,
                    selected_q10=[0.]*6, selected_q14=[slot*.1]*6, prepared_connection_sha256=request['id']+str(slot))
                admitted_here.append(value)
                admitted.append(value)
                all_prepared[value['id']] = value
                if historical and not history_rows:
                    old = dict(value, native_release=True)
                    history_rows.append(old)
                    seen.add(V.command_key(old))
            records.append(dict(proposal_id=request['id'], proposal_sha256=digest(request), context_id=request['context_id'],
                values=request['values'], bucket_xy_m=request['bucket_xy_m'], admitted=len(admitted_here),
                admitted_request_ids=[r['id'] for r in admitted_here], admitted_request_sha256=[digest(r) for r in admitted_here],
                native_releases=0, preparation_only=True, reason='finite_ik_unresolved' if decline else None))
        selected, omitted = [], []
        selected_keys = set(seen)
        capacity = 1200-sum(counts.values())
        for r in admitted:
            key = V.command_key(r)
            reason = 'previously_flown_exact_command' if key in selected_keys else 'native_release_limit' if len(selected)>=capacity else None
            if reason: omitted.append(dict(id=r['id'],command_key=key,reason=reason))
            else:
                selected.append(r)
                selected_keys.add(key)
        write(root/prepared_file, dict(requests=selected, preparations=records, not_executed=omitted,
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
            seen.add(V.command_key(row))
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
        boundary('search-round-complete', round=round_number, cursor=cursor, search_stage=stage)
    capped = sum(counts.values()) == 1200
    truncated = (cutoff is not None and round_number == cutoff and not capped) or window_cutoff
    reason = 'search_native_limit' if capped else 'search_time_limit' if truncated else 'finite_reachable_proposal_pool_exhausted'
    boundary('search-complete', reason=reason, cursor=cursor, search_stage=stage)
    controls('controls_post')
    report = dict(status='completed', completion_reason=reason, budget_truncated=truncated,
        deadlines=dict(effective_worker_limit_s=18000, search_limit_s=16200), fresh_search_releases=counts,
        cases=V.independent_cases(rows))
    (root/'rows.jsonl').write_bytes(b''.join(raw))
    for name, values in (('phase-boundaries.jsonl', boundaries), ('timing.jsonl', timings),
                         ('packet-commits.jsonl', commits), ('preparations.jsonl', preparations)):
        (root/name).write_text(''.join(json.dumps(r)+'\n' for r in values))
    return rows, report, dict(prepared_requests=all_prepared, broad_rows=history_rows), dict(rows=plan), kept


def check(root, parts):
    return V.audit_schedule(root, *parts, V.independent_repeat, Checks())


class SixhourScheduleTests(unittest.TestCase):
    def test_full1200_cap_is_not_preparation_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); result=check(root,fixture(root,proposals=155,branches=8))
            self.assertEqual(sum(result['fresh_search_releases'].values()),1200)
            self.assertEqual(result['preparation_count'],155)
            self.assertEqual(result['total_requests'],1254)
            self.assertEqual(result['completion_reason'],'search_native_limit')
    def test_historical_duplicate_is_omitted_without_spending_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); result=check(root,fixture(root,historical=True))
            self.assertEqual(sum(result['fresh_search_releases'].values()),9)
            skipped=json.loads((root/'prepared/batch-00000.json').read_text())['not_executed']
            self.assertEqual(skipped[0]['reason'],'previously_flown_exact_command')
    def test_windows_require_confirmed_target_and_elapsed_three_hours(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); result=check(root,fixture(root,proposals=20,capture=True,switch=True))
            self.assertTrue(result['window_frozen']);self.assertEqual(result['final_stage'],'window')
            self.assertEqual(result['repeat_requests'],21)
        for options in (dict(capture=False),dict(capture=True,early=True)):
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp)
                with self.assertRaises(EvidenceError):check(root,fixture(root,proposals=20,switch=True,**options))
    def test_window_generation_crossing_cutoff_leaves_frozen_unexecuted_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); result=check(root,fixture(root,proposals=20,capture=True,switch=True,window_cutoff=True))
            self.assertTrue(result['window_frozen']);self.assertTrue(result['budget_truncated'])
            self.assertEqual(result['preparation_count'],12)
    def test_declines_and_time_bounded_completion_do_not_promise1200(self):
        for options in (dict(decline=True),dict(proposals=20,cutoff=1)):
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);result=check(root,fixture(root,**options))
                self.assertLess(sum(result['fresh_search_releases'].values()),1200)
    def test_tampered_selected_packet_or_historical_duplicate_reason_fails(self):
        for field in ('requests','not_executed'):
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);parts=fixture(root,historical=True)
                path=root/'prepared/batch-00000.json';value=json.loads(path.read_text());value[field]=[];write(path,value)
                with self.assertRaises(EvidenceError):check(root,parts)
    def test_extra_window_file_or_bad_native_receipt_fails(self):
        for option in ('extra','receipt'):
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);parts=fixture(root)
                if option=='extra':write(root/'window-plans/unused.json',{})
                else:(root/'native-yaw/cache-00002.bin').write_text('changed')
                with self.assertRaises(EvidenceError):check(root,parts)
    def test_window_cannot_claim_more_than_remaining_numeric_capacity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);parts=fixture(root,proposals=20,capture=True,switch=True)
            path=root/'window-plans/plan.json';value=json.loads(path.read_text());value['limit']=6000;write(path,value)
            with self.assertRaises(EvidenceError):check(root,parts)
    def test_ninth_bouncer_branch_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with self.assertRaises(EvidenceError):check(root,fixture(root,branches=9))


if __name__=='__main__':unittest.main()
