"""Independently reconstruct broad preparation, flight and confirmation ordering.

No producer planner, selector or study module is imported. Numeric proposals and
native admission proofs are authenticated separately by the main auditor.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import numpy as np

from . import benchmark_verify as common
from . import fixed35_recovery_verify as old
from .benchmark_verify import digest, read, sha, contained

CASE_IDS = ('fixed35-dev-grid-03-04', 'fixed35-dev-grid-03-05', 'fixed35-dev-grid-04-06')
QUOTAS = dict(zip(CASE_IDS, (334, 333, 333)))
captured, statistics = old.captured, old.statistics


def independent_repeat(discovery, group):
    result = []
    for index in range(3):
        row = copy.deepcopy(discovery['prepared_request'])
        row.update(id=f'fixed35-broad-repeat-g{group}-r{index}-{discovery["context_id"]}',
            phase='repeat', method='broad_repeat', confirmation_index=group, repeat=index,
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
        native = [r for r in attempts if r.get('native_release')]
        groups = independent_groups(rows, context)
        ids = {g['selected_from_id'] for g in groups if g['status'] == 'confirmed'}
        joint_bins = {tuple(np.rint(((np.asarray(r['q14']) + np.pi) % (2*np.pi) - np.pi) / .05).astype(int).tolist()) for r in native}
        result.append(dict(context_id=context, quota=QUOTAS[context], search_counts=statistics(attempts),
            fresh_search_releases=len(native), unique_numeric_settings=len({tuple(r['values']) for r in native}),
            exact_unique_joint_pairs=len({tuple(r['q10'] + r['q14']) for r in native}),
            distinct_q14_quantized_50mrad=len(joint_bins),
            confirmed_recipes=[copy.deepcopy(r) for r in attempts if r['id'] in ids], confirmation_groups=groups))
    return result


def audit_schedule(study, rows, report, data, plan, preserved, repeat_builder, checks):
    study = Path(study)
    numeric_plan = plan['rows'] if isinstance(plan, dict) else plan
    queues = {c: [r for r in numeric_plan if r['context_id'] == c] for c in CASE_IDS}
    checks.require(all(len(q) <= 4000 for q in queues.values()), 'At most4000 numeric proposals per cell')
    checks.equal(sum(map(len, queues.values())), len(numeric_plan), 'Only the3 authorized fixed cells')
    checks.equal(len({r['id'] for r in numeric_plan}), len(numeric_plan), 'Unique numeric proposal IDs')
    boundaries = common.lines(study/'phase-boundaries.jsonl')
    timings = common.lines(study/'timing.jsonl')
    commits = common.lines(study/'packet-commits.jsonl')
    preparations = common.lines(study/'preparations.jsonl') if (study/'preparations.jsonl').exists() else []
    raw = (study/'rows.jsonl').read_bytes().splitlines(keepends=True)
    checks.equal(common.lines(study/'rows.jsonl'), rows, 'Complete raw native ledger supplied')
    position = boundary_index = packet_number = preparation_index = rounds = 0
    previous_time = 0.
    cursors = dict.fromkeys(CASE_IDS, 0)
    counts = dict.fromkeys(CASE_IDS, 0)
    confirmed_numeric = {c: [] for c in CASE_IDS}
    seen_actions = set()
    expected_repeat_files, expected_preparation_files, expected_prepared_files = [], [], []
    deadlines = report['deadlines']
    worker_limit = deadlines['effective_worker_limit_s']
    search_limit = deadlines['search_limit_s']
    checks.require(60 < worker_limit <= 43200 and 0 < search_limit <= 36000,
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
                    identity = (context, tuple(row['values']), tuple(row['q10']), tuple(row['q14']))
                    checks.require(identity not in seen_actions, row['id']+'/no duplicate5D+joints top-up')
                    seen_actions.add(identity)
                    counts[context] += 1
                    checks.require(counts[context] <= QUOTAS[context], context+'/native quota not exceeded')
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
    checks.equal(len(preserved), 30, '30 authenticated existing recipes preserved')
    kept = execute(preserved, 'preserved')
    checks.require(all(captured(r) for r in kept), 'All30 preserved recipes recapture before exploration')
    boundary('preserved-complete')

    while boundary_index < len(boundaries) and boundaries[boundary_index]['phase'] == 'preparation-committed':
        checks.require(counts != QUOTAS, 'No new preparation after1000 completed releases')
        expected = []
        for _ in range(4):
            for context in CASE_IDS:
                if counts[context] < QUOTAS[context] and cursors[context] < len(queues[context]):
                    expected.append(queues[context][cursors[context]])
                    cursors[context] += 1
        checks.require(bool(expected), 'Each round consumes remaining frozen numeric candidates')
        packet = f'preparation-requests/batch-{rounds:05d}.json'
        prepared_path = f'prepared/batch-{rounds:05d}.json'
        checks.equal(read(study/packet), dict(rows=expected, search_plan_sha256=sha(study/'search-plan.json'), pre_outcome=True),
                     packet+'/4 proposals per active cell in literal round-robin order')
        frozen = read(study/prepared_path)
        checks.equal((frozen['preparation_request_file'], frozen['preparation_request_sha256'], frozen['frozen_before_native_outcomes']),
                     (packet, sha(study/packet), True), prepared_path+'/frozen before flight')
        prepared = frozen['preparations']
        checks.equal(len(prepared), len(expected), prepared_path+'/every preparation recorded including declines')
        admitted_ids, admitted_hashes = [], {}
        used = dict(counts)
        selected_ids, omitted_ids = [], []
        for wanted, record in zip(expected, prepared):
            checks.equal((record['proposal_id'], record['proposal_sha256'], record['context_id'], record['values'], record['bucket_xy_m']),
                         (wanted['id'], digest(wanted), wanted['context_id'], wanted['values'], wanted['bucket_xy_m']), wanted['id']+'/numeric preparation binding')
            ids = record['admitted_request_ids']
            checks.require(0 <= len(ids) <= 2, wanted['id']+'/at most2 admitted bouncer branches')
            checks.equal(ids, [wanted['id']+'-b'+str(i) for i in range(len(ids))], wanted['id']+'/ordered branch slots')
            checks.equal((record['admitted'], record['native_releases'], record['preparation_only']), (len(ids), 0, True), wanted['id']+'/preparation is not a release')
            checks.equal(len(record['admitted_request_sha256']), len(ids), wanted['id']+'/all admitted requests hashed')
            for identity, value in zip(ids, record['admitted_request_sha256']):
                checks.require(identity not in admitted_hashes, identity+'/unique admitted ID')
                admitted_ids.append(identity)
                admitted_hashes[identity] = value
                context = wanted['context_id']
                if used[context] < QUOTAS[context]:
                    selected_ids.append(identity)
                    used[context] += 1
                else:
                    omitted_ids.append(identity)
            appended = dict(record, preparation_request_file=packet, preparation_request_sha256=sha(study/packet),
                            prepared_file=prepared_path, prepared_sha256=sha(study/prepared_path))
            checks.require(preparation_index < len(preparations), wanted['id']+'/append-only preparation record exists')
            checks.equal(preparations[preparation_index], appended, wanted['id']+'/immutable preparation receipt')
            preparation_index += 1
        selected = frozen['requests']
        checks.equal([r['id'] for r in selected], selected_ids, prepared_path+'/quota trim only; no outcome filtering')
        checks.equal(frozen['not_executed_due_quota'], omitted_ids, prepared_path+'/all omitted admitted branches explained by quota')
        for request in selected:
            checks.equal(digest(request), admitted_hashes[request['id']], request['id']+'/admitted request bytes unchanged')
        start_bound = previous_time
        frozen_boundary = boundary('preparation-committed', round=rounds, request_packet=packet,
                                   prepared_file=prepared_path, prepared_sha256=sha(study/prepared_path))
        prep_start = frozen_boundary['acquired_elapsed_s'] - frozen_boundary['elapsed_preparation_s']
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
        boundary('search-round-complete', round=rounds, cursors=copy.deepcopy(cursors))
        expected_preparation_files.append(packet)
        expected_prepared_files.append(prepared_path)

    done = counts == QUOTAS
    remaining = any(counts[c] < QUOTAS[c] and cursors[c] < len(queues[c]) for c in CASE_IDS)
    if done:
        reason, truncated = '1000_fresh_search_drops_completed', False
    elif report['budget_truncated']:
        reason, truncated = 'search_time_limit', True
    else:
        checks.require(not remaining, 'Incomplete finite pool must be exhausted, not silently stopped')
        reason, truncated = 'finite_reachable_proposal_pool_exhausted', False
    checks.equal((report['completion_reason'], report['budget_truncated']), (reason, truncated), 'Honest completion reason')
    end = boundary('search-complete', reason=reason, cursors=cursors)
    if truncated:
        checks.require(end['acquired_elapsed_s'] >= search_limit, 'Time cutoff observed before claiming budget shortfall')
    controls('controls_post')
    checks.equal((position, packet_number, boundary_index, preparation_index),
                 (len(rows), len(timings), len(boundaries), len(preparations)), 'No omitted or extra native/preparation/timing records')
    checks.equal(len(commits), packet_number, 'Every packet has exactly one receipt')
    for directory, expected in (('requests', [f'requests/batch-{i:05d}.json' for i in range(packet_number)]),
            ('preparation-requests', expected_preparation_files), ('prepared', expected_prepared_files), ('repeat-plans', expected_repeat_files)):
        checks.equal(sorted(str(p.relative_to(study)) for p in (study/directory).glob('*.json')), sorted(expected), directory+'/no unaccounted packet files')
    checks.require(len(rows) <= 1075 and sum(r['phase'] == 'repeat' for r in rows) <= 27, 'Bounded actual native releases including controls and confirmations')
    checks.equal(len({r['id'] for r in rows}), len(rows), 'Unique native request IDs')
    checks.equal(report['status'], 'completed' if done else 'incomplete', 'Completed means all1000 search drops')
    checks.equal(report['fresh_search_releases'], counts, 'Per-cell actual search release counters')
    checks.equal(report['cases'], independent_cases(rows), 'Independent case summaries and confirmed recipes')
    return dict(total_requests=len(rows), fresh_search_releases=counts, search_quota_completed=done,
        preparation_count=len(preparations), preparation_rounds=rounds,
        preparation_declines=sum(not r['admitted_request_ids'] for r in preparations),
        repeat_requests=sum(r['phase'] == 'repeat' for r in rows), preserved_requests=30, control_requests=18,
        maximum_native_releases=1075, budget_truncated=truncated, completion_reason=reason)
