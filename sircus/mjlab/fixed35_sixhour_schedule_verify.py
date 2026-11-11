"""Independently reconstruct the six-hour staged search and native accounting.

No producer planner, selector or study module is imported. Numeric proposals and
native admission proofs are authenticated separately by the main auditor.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

from . import benchmark_verify as common
from . import fixed35_recovery_verify as old
from .benchmark_verify import digest, read, sha, contained

CASE_IDS = ('fixed35-dev-grid-03-04', 'fixed35-dev-grid-03-05', 'fixed35-dev-grid-04-06')
TARGET = CASE_IDS[1]
NUMERIC_LIMIT = 6000
NATIVE_LIMIT = 1200
PRIMARY_LIMIT_S = 10800
captured, statistics = old.captured, old.statistics


def independent_repeat(discovery, group):
    result = []
    for index in range(3):
        row = copy.deepcopy(discovery['prepared_request'])
        row.update(id=f'fixed35-sixhour-repeat-g{group}-r{index}-{discovery["context_id"]}',
            phase='repeat', method='sixhour_repeat', confirmation_index=group, repeat=index,
            selected_from_id=discovery['id'], selected_from_sha256=digest(discovery),
            selected_connection_sha256=discovery['connection_sha256'],
            selected_q10=copy.deepcopy(discovery['q10']), selected_q14=copy.deepcopy(discovery['q14']))
        result.append(row)
    return result


def independent_groups(rows, context):
    result = []
    for number in range(3):
        group = [r for r in rows if r.get('phase') == 'repeat' and r['context_id'] == context
                 and r['confirmation_index'] == number]
        if not group:
            continue
        first = group[0]
        exact = (len(group) == 3 and sorted(r['repeat'] for r in group) == [0, 1, 2]
                 and len({r['id'] for r in group}) == 3 and all(all(r.get(k) == first.get(k) for k in
                 ('selected_from_id', 'selected_from_sha256', 'values', 'bucket_xy_m',
                  'q10', 'q14', 'connection_sha256')) for r in group))
        result.append(dict(group=number, selected_from_id=first['selected_from_id'], requests=len(group),
            captures=sum(captured(r) for r in group),
            status='confirmed' if exact and all(captured(r) for r in group) else 'failed' if len(group) == 3 else 'incomplete'))
    return result


def independent_cases(rows, data=None):
    result = []
    for context in CASE_IDS:
        attempts = [r for r in rows if r.get('phase') == 'search' and r['context_id'] == context]
        groups = independent_groups(rows, context)
        ids = {g['selected_from_id'] for g in groups if g['status'] == 'confirmed'}
        result.append(dict(context_id=context, search_counts=statistics(attempts),
            fresh_search_releases=sum(r.get('native_release') is True for r in attempts),
            unique_numeric_settings=len({tuple(r['values']) for r in attempts}),
            confirmed_recipes=[copy.deepcopy(r) for r in attempts if r['id'] in ids], confirmation_groups=groups))
    return result


def command_key(row):
    return digest(dict(context_id=row['context_id'], values=row['values'],
        q10=row['selected_q10'] if 'selected_q10' in row else row['q10'],
        q14=row['selected_q14'] if 'selected_q14' in row else row['q14']))


def audit_schedule(study, rows, report, data, plan, preserved, repeat_builder, checks):
    study = Path(study)
    numeric_plan = plan['rows'] if isinstance(plan, dict) else plan
    checks.require(len(numeric_plan) <= NUMERIC_LIMIT and all(r['context_id'] == TARGET for r in numeric_plan),
                   'At most6000 primary numeric proposals at03-05 only')
    checks.equal(len({r['id'] for r in numeric_plan}), len(numeric_plan), 'Unique primary numeric proposal IDs')
    all_prepared = data['prepared_requests']
    historical_keys = {command_key(r) for r in data['broad_rows'] if r.get('native_release')}
    boundaries = common.lines(study/'phase-boundaries.jsonl')
    timings = common.lines(study/'timing.jsonl')
    commits = common.lines(study/'packet-commits.jsonl')
    preparations = common.lines(study/'preparations.jsonl') if (study/'preparations.jsonl').exists() else []
    raw = (study/'rows.jsonl').read_bytes().splitlines(keepends=True)
    checks.equal(common.lines(study/'rows.jsonl'), rows, 'Complete raw native ledger supplied')
    position = boundary_index = packet_number = preparation_index = rounds = 0
    previous_time = 0.
    cursor = 0
    stage = 'primary'
    active_plan, plan_file = plan, 'search-plan.json'
    window_frozen = False
    counts = dict.fromkeys(CASE_IDS, 0)
    confirmed_numeric = {c: [] for c in CASE_IDS}
    seen_actions = set(historical_keys)
    expected_repeat_files, expected_preparation_files, expected_prepared_files = [], [], []
    deadlines = report['deadlines']
    worker_limit = deadlines['effective_worker_limit_s']
    search_limit = deadlines['search_limit_s']
    checks.require(60 < worker_limit <= 18000 and 0 < search_limit <= 16200,
                   'Finite worker/search budgets')

    def prefix(count):
        return hashlib.sha256(b''.join(raw[:count])).hexdigest() if count else None

    def boundary(phase, **extra):
        nonlocal boundary_index, previous_time
        checks.require(boundary_index < len(boundaries), phase+'/boundary exists')
        row = boundaries[boundary_index]
        boundary_index += 1
        checks.equal((row['phase'], row['rows_count'], row['rows_sha256']),
                     (phase, position, prefix(position)), phase+'/immutable native prefix')
        checks.require(previous_time <= row['acquired_elapsed_s'] < worker_limit,
                       phase+'/bounded ordered time')
        previous_time = row['acquired_elapsed_s']
        for key, value in extra.items():
            checks.equal(row[key], value, phase+'/'+key)
        return row

    def execute(expected, phase):
        nonlocal position, packet_number, previous_time
        result = []
        for offset in range(0, len(expected), 32):
            wanted = expected[offset:offset+32]
            packet = f'requests/batch-{packet_number:05d}.json'
            checks.require(packet_number < len(timings) and packet_number < len(commits), phase+'/packet receipts exist')
            checks.equal(read(study/packet), dict(phase=phase, rows=wanted, pre_outcome=True), packet+'/frozen exact requests')
            packet_sha = sha(study/packet)
            timing = timings[packet_number]
            checks.equal((timing['phase'], timing['request_packet'], timing['request_packet_sha256']),
                         (phase, packet, packet_sha), packet+'/timing binding')
            checks.require(previous_time <= timing['acquired_start_s'] <= timing['acquired_end_s'] < worker_limit,
                           packet+'/bounded ordered execution')
            observed = rows[position:position+len(wanted)]
            checks.equal(len(observed), len(wanted), packet+'/all outcomes exist')
            for request, row in zip(wanted, observed):
                for key, value in request.items():
                    checks.equal(row.get(key), value, row['id']+'/request/'+key)
                checks.equal((row['request_packet'], row['request_packet_sha256']), (packet, packet_sha), row['id']+'/packet binding')
                if phase in ('search', 'repeat'):
                    checks.require(row.get('native_release') is True, row['id']+'/actually released after preparation')
                    checks.equal(row['prepared_request'], request, row['id']+'/prepared request retained')
                    for key in ('q10', 'q14'):
                        checks.equal(row[key], request['selected_'+key], row['id']+'/frozen joints/'+key)
                if phase == 'search':
                    context = row['context_id']
                    identity = command_key(row)
                    checks.require(identity not in seen_actions, row['id']+'/no duplicate5D+joints top-up')
                    seen_actions.add(identity)
                    counts[context] += 1
                    checks.require(sum(counts.values()) <= NATIVE_LIMIT, 'Native1200 search cap not exceeded')
            position += len(observed)
            caches = {r['adapter_directory']+'/'+r['native_cache_file']: r['native_cache_sha256'] for r in observed if r.get('native_release')}
            for name, expected_sha in caches.items():
                checks.equal(sha(contained(study, name)), expected_sha, packet+'/native archive/'+name)
            checks.equal(commits[packet_number], dict(packet=packet, packet_sha256=packet_sha,
                row_ids=[r['id'] for r in observed], observed_rows_sha256=digest(observed),
                rows_count=position, rows_sha256=prefix(position), native_files=caches), packet+'/atomic outcome receipt')
            previous_time = timing['acquired_end_s']
            result.extend(observed)
            packet_number += 1
        return result

    def controls(phase):
        # Main auditor independently checks all original control action fields.
        path = study/f'requests/batch-{packet_number:05d}.json'
        expected = read(path)['rows']
        checks.equal(len(expected), 9, phase+'/9 original control repeats')
        checks.require(all(r['phase'] == phase and r['method'] == 'control' for r in expected), phase+'/control-only block')
        execute(expected, phase)
        boundary(phase+'-complete')

    boundary('inputs-frozen')
    controls('controls_pre')
    checks.require(len(preserved) >= 30, 'All authenticated existing recipes preserved')
    kept = execute(preserved, 'preserved')
    checks.require(all(captured(r) for r in kept), 'All preserved recipes recapture before exploration')
    boundary('preserved-complete')

    expected_window_files = []
    while boundary_index < len(boundaries) and boundaries[boundary_index]['phase'] in ('preparation-committed', 'window-plan-frozen'):
        checks.require(sum(counts.values()) < NATIVE_LIMIT and preparation_index < NUMERIC_LIMIT,
                       'No new preparation after native/numeric cap')
        if boundaries[boundary_index]['phase'] == 'window-plan-frozen':
            checks.require(not window_frozen and stage == 'primary', 'Only one forward stage transition')
            target = next(c for c in independent_cases(rows[:position]) if c['context_id'] == TARGET)
            checks.require(bool(target['confirmed_recipes']), 'No window switch without fresh03-05 confirmation')
            plan_file = 'window-plans/plan.json'
            active_plan = read(study/plan_file)
            checks.equal((active_plan['selected_after_rows_count'], active_plan['selected_after_rows_sha256'], active_plan['posthoc_development']),
                         (position, prefix(position), True), 'Window plan uses only prior outcomes')
            checks.equal(active_plan['limit'], NUMERIC_LIMIT-preparation_index, 'Window plan consumes only remaining numeric capacity')
            checks.require(len(active_plan['rows']) <= NUMERIC_LIMIT-preparation_index, 'Window proposal cap')
            checks.require(all(r['context_id'] in CASE_IDS for r in active_plan['rows']), 'Only3 fixed window cells')
            checks.equal(len({r['id'] for r in active_plan['rows']}), len(active_plan['rows']), 'Unique window proposal IDs')
            checks.require(previous_time <= active_plan['acquired_elapsed_s'] and active_plan['acquired_elapsed_s'] >= PRIMARY_LIMIT_S,
                           'Conditional window plan frozen after3 acquired hours')
            switch = boundary('window-plan-frozen', plan_file=plan_file, plan_sha256=sha(study/plan_file))
            checks.require(active_plan['acquired_elapsed_s'] <= switch['acquired_elapsed_s'], 'Window freeze precedes boundary')
            expected_window_files.append(plan_file)
            cursor, stage, window_frozen = 0, 'window', True
        elif stage == 'primary' and previous_time >= PRIMARY_LIMIT_S:
            target = next(c for c in independent_cases(rows[:position]) if c['context_id'] == TARGET)
            checks.require(not target['confirmed_recipes'], 'Confirmed primary must switch to windows after3h')
        if boundaries[boundary_index]['phase'] == 'search-complete':
            break
        expected = active_plan['rows'][cursor:cursor+min(12, NUMERIC_LIMIT-preparation_index)]
        cursor += len(expected)
        checks.require(bool(expected), 'Each round consumes remaining frozen numeric candidates')
        packet = f'preparation-requests/batch-{rounds:05d}.json'
        prepared_path = f'prepared/batch-{rounds:05d}.json'
        checks.equal(read(study/packet), dict(rows=expected, search_plan_file=plan_file, search_plan_sha256=sha(study/plan_file), pre_outcome=True),
                     packet+'/literal next12 stage proposals')
        frozen = read(study/prepared_path)
        checks.equal((frozen['preparation_request_file'], frozen['preparation_request_sha256'], frozen['frozen_before_native_outcomes']),
                     (packet, sha(study/packet), True), prepared_path+'/frozen before flight')
        prepared = frozen['preparations']
        checks.equal(len(prepared), len(expected), prepared_path+'/every preparation recorded including declines')
        admitted = []
        for wanted, record in zip(expected, prepared):
            checks.equal((record['proposal_id'], record['proposal_sha256'], record['context_id'], record['values'], record['bucket_xy_m']),
                         (wanted['id'], digest(wanted), wanted['context_id'], wanted['values'], wanted['bucket_xy_m']), wanted['id']+'/numeric preparation binding')
            ids = record['admitted_request_ids']
            checks.require(0 <= len(ids) <= 8, wanted['id']+'/at most8 admitted bouncer branches')
            checks.equal(ids, [wanted['id']+'-b'+str(i) for i in range(len(ids))], wanted['id']+'/ordered branch slots')
            checks.equal((record['admitted'], record['native_releases'], record['preparation_only']), (len(ids), 0, True), wanted['id']+'/preparation is not a release')
            checks.equal(len(record['admitted_request_sha256']), len(ids), wanted['id']+'/all admitted requests hashed')
            for identity, value in zip(ids, record['admitted_request_sha256']):
                checks.require(identity in all_prepared, identity+'/independently reconstructed admitted request')
                checks.equal(digest(all_prepared[identity]), value, identity+'/exact prepared packet hash')
                admitted.append(all_prepared[identity])
            appended = dict(record, preparation_request_file=packet, preparation_request_sha256=sha(study/packet),
                            prepared_file=prepared_path, prepared_sha256=sha(study/prepared_path))
            checks.require(preparation_index < len(preparations), wanted['id']+'/append-only preparation record exists')
            checks.equal(preparations[preparation_index], appended, wanted['id']+'/immutable preparation receipt')
            preparation_index += 1
        selected, unused, selected_keys = [], [], set(seen_actions)
        capacity = NATIVE_LIMIT-sum(counts.values())
        for request in admitted:
            key = command_key(request)
            reason = ('previously_flown_exact_command' if key in selected_keys else
                      'native_release_limit' if len(selected) >= capacity else None)
            if reason:
                unused.append(dict(id=request['id'], command_key=key, reason=reason))
            else:
                selected.append(request)
                selected_keys.add(key)
        checks.equal(frozen['requests'], selected, prepared_path+'/duplicate and cap filtering only')
        checks.equal(frozen['not_executed'], unused, prepared_path+'/every omitted admitted branch explained')
        start_bound = previous_time
        frozen_boundary = boundary('preparation-committed', round=rounds, request_packet=packet,
                                   prepared_file=prepared_path, prepared_sha256=sha(study/prepared_path))
        prep_start = frozen_boundary['acquired_elapsed_s']-frozen_boundary['elapsed_preparation_s']
        checks.require(start_bound <= prep_start <= frozen_boundary['acquired_elapsed_s'] and prep_start < search_limit,
                       prepared_path+'/preparation begins before finite search cutoff')
        observed = execute([dict(r, prepared_batch_file=prepared_path, prepared_batch_sha256=sha(study/prepared_path)) for r in selected], 'search')
        discoveries = []
        for row in observed:
            context, numeric = row['context_id'], tuple(row['values'])
            if captured(row) and len(confirmed_numeric[context]) < 3 and numeric not in confirmed_numeric[context]:
                group = len(confirmed_numeric[context])
                confirmed_numeric[context].append(numeric)
                discoveries.append((row, group))
        if discoveries:
            name = f'repeat-plans/group-{rounds:05d}.json'
            repeated = [r for source, group in discoveries for r in independent_repeat(source, group)]
            if repeat_builder is not None:
                checks.equal([r for source, group in discoveries for r in repeat_builder(source, group)], repeated,
                             name+'/independently reconstructed repeat builder')
            checks.equal(read(study/name), dict(rows=repeated, selected=[dict(id=r['id'], sha256=digest(r)) for r, _ in discoveries],
                rows_count_before=position, rows_sha256_before=prefix(position), frozen_before_repeat_outcomes=True), name+'/first3 distinct numeric captures only')
            measured = execute([dict(r, repeat_plan_file=name, repeat_plan_sha256=sha(study/name)) for r in repeated], 'repeat')
            for source, _ in discoveries:
                group = [r for r in measured if r['selected_from_id'] == source['id']]
                checks.equal([r['repeat'] for r in group], [0, 1, 2], source['id']+'/3 fresh confirmations')
                for row in group:
                    checks.require(row['native_cache_sha256'] != source['native_cache_sha256'], row['id']+'/fresh native trajectory')
                    for key in ('values', 'bucket_xy_m', 'q10', 'q14', 'connection_sha256'):
                        checks.equal(row[key], source[key], row['id']+'/unchanged discovery/'+key)
            expected_repeat_files.append(name)
        rounds += 1
        boundary('search-round-complete', round=rounds, cursor=cursor, search_stage=stage)
        expected_preparation_files.append(packet)
        expected_prepared_files.append(prepared_path)

    if sum(counts.values()) >= NATIVE_LIMIT:
        reason, truncated = 'search_native_limit', False
    elif report['budget_truncated']:
        reason, truncated = 'search_time_limit', True
    elif preparation_index >= NUMERIC_LIMIT:
        reason, truncated = 'numeric_proposal_limit', False
    else:
        checks.require(cursor >= len(active_plan['rows']), 'No unexplained missing active proposal')
        reason, truncated = 'finite_reachable_proposal_pool_exhausted', False
    checks.equal((report['completion_reason'], report['budget_truncated']), (reason, truncated), 'Honest bounded completion reason')
    end = boundary('search-complete', reason=reason, cursor=cursor, search_stage=stage)
    if truncated:
        checks.require(end['acquired_elapsed_s'] >= search_limit, 'Time cutoff observed before budget shortfall')
    controls('controls_post')
    checks.equal((position, packet_number, boundary_index, preparation_index),
                 (len(rows), len(timings), len(boundaries), len(preparations)), 'No omitted or extra native/preparation/timing records')
    checks.equal(len(commits), packet_number, 'Every packet has exactly one receipt')
    for directory, expected in (('requests', [f'requests/batch-{i:05d}.json' for i in range(packet_number)]),
            ('preparation-requests', expected_preparation_files), ('prepared', expected_prepared_files), ('repeat-plans', expected_repeat_files), ('window-plans', expected_window_files)):
        checks.equal(sorted(str(p.relative_to(study)) for p in (study/directory).glob('*.json')), sorted(expected), directory+'/no unaccounted packet files')
    checks.require(len(rows) <= 1200+27+18+len(preserved) and sum(r['phase'] == 'repeat' for r in rows) <= 27, 'Bounded actual native releases including controls and confirmations')
    checks.equal(len({r['id'] for r in rows}), len(rows), 'Unique native request IDs')
    checks.equal(report['status'], 'completed', 'Completed bounded study is not a promise of1200 drops')
    checks.equal(report['fresh_search_releases'], counts, 'Per-cell actual search release counters')
    checks.equal(report['cases'], independent_cases(rows), 'Independent case summaries and confirmed recipes')
    return dict(total_requests=len(rows), fresh_search_releases=counts,
        preparation_count=len(preparations), preparation_rounds=rounds,
        preparation_declines=sum(not r['admitted_request_ids'] for r in preparations),
        repeat_requests=sum(r['phase'] == 'repeat' for r in rows), preserved_requests=len(preserved), control_requests=18,
        maximum_search_releases=1200, maximum_numeric_preparations=6000,
        budget_truncated=truncated, completion_reason=reason, final_stage=stage, window_frozen=window_frozen)
