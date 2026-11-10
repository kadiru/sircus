"""Independent CPU reconstruction of fixed-six numeric plans and execution order.

This module imports no producer planner, search rule, scorer or scheduler.
"""
from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path

import numpy as np

from . import benchmark_verify as common
from . import fixed35_recovery_verify as old
from .benchmark_verify import EvidenceError, digest, read, sha

META = old.META
captured = old.captured
statistics = old.statistics
request = old.request
LOW = (.30, -.50, .12, 8., .05)
HIGH = (1.04, .35, .60, 60., .65)
CASE_IDS = ('fixed35-dev-grid-01-05', 'fixed35-dev-grid-03-04',
            'fixed35-dev-grid-03-05', 'fixed35-dev-grid-04-06')


def digit_fraction(number, radix):
    value = 0.
    weight = 1.
    while number > 0:
        digit = number % radix
        number //= radix
        weight /= radix
        value += digit*weight
    return value


def independent_proposals(source, alternatives, preserved):
    base = list(map(float, source['values']))
    bucket = np.asarray(source['bucket_xy_m'], dtype=float)
    delta = np.asarray(base[:2])-bucket
    distance = float(np.linalg.norm(delta))
    angle0 = math.atan2(delta[1], delta[0])
    planned, seen = [], set()

    def offer(values, kind, ids=None, **derivation):
        candidate = list(map(float, values))
        key = tuple(candidate)
        if len(candidate) != 5 or not all(math.isfinite(x) for x in candidate):
            raise EvidenceError('Nonfinite independent candidate')
        if any(x < lo or x > hi for x, lo, hi in zip(candidate, LOW, HIGH)) or key in seen:
            return
        seen.add(key)
        planned.append(dict(values=candidate, kind=kind, parent_source_ids=ids or [source['id']], **derivation))

    offer(base, 'original_failed_action')
    same_layout = [r for r in alternatives if r['context_id'] == source['context_id']]
    if len(same_layout) != 1:
        raise EvidenceError('One original alternate action required')
    offer(same_layout[0]['values'], 'historical_refit_action', [same_layout[0]['id']])
    nearby = sorted(preserved, key=lambda r: (float(np.linalg.norm(np.asarray(r['bucket_xy_m'])-bucket)), r['id']))[:6]
    if len(nearby) != 6:
        raise EvidenceError('Six preserved neighbor recipes required')
    for row in nearby:
        offer(row['values'], 'neighbor_recipe_at_fixed_bucket', [row['id']])
    for delta_radius in (-.12, -.06, .06, .12):
        for delta_degrees in (-30., -15., 15., 30.):
            radius = distance+delta_radius
            if radius <= 0:
                continue
            angle = angle0+math.radians(delta_degrees)
            xy = bucket+radius*np.asarray([math.cos(angle), math.sin(angle)])
            offer([*xy, *base[2:]], 'local_polar_xy', radius_m=radius, azimuth_rad=angle,
                  radius_offset_m=delta_radius, azimuth_offset_deg=delta_degrees)
    for n in range(1, 10001):
        if len(planned) == 96:
            break
        unit = [digit_fraction(n, radix) for radix in (2, 3, 5, 7, 11)]
        radius = .18+.72*unit[0]
        angle = angle0+math.radians(-100.+200.*unit[1])
        xy = bucket+radius*np.asarray([math.cos(angle), math.sin(angle)])
        action = [*xy, .14+.42*unit[2], 10.+44.*unit[3], .08+.52*unit[4]]
        offer(action, 'coupled_polar_height_tilt_gap', radical_inverse_index=n,
              radius_m=radius, azimuth_rad=angle, unit_coordinates=unit)
    if len(planned) != 96:
        raise EvidenceError('Independent bounded96-action construction exhausted')
    return [dict(row, numeric_proposal_index=index) for index, row in enumerate(planned)]


def independent_plan(data):
    if tuple(r['context_id'] for r in data['cases']) != CASE_IDS:
        raise EvidenceError('Exactly four original search contexts required')
    actions = {r['context_id']: independent_proposals(r, data['refit'], data['preserved']) for r in data['cases']}
    result = []
    for index in range(96):
        for branch in (0, 1):
            for source in data['cases']:
                proposal = actions[source['context_id']][index]
                result.append(request(source,
                    f'fixed35-six-search-p{index:02d}-b{branch}-{source["context_id"]}', 'search', 'six_search',
                    proposal['values'], proposal_index=2*index+branch, numeric_proposal_index=index,
                    ik_branch_slot=branch, derivation=copy.deepcopy(proposal), exact_parent_action=False))
    return result


def independent_preserved(data):
    return [request(r, 'fixed35-six-preserved-'+r['context_id'], 'preserved', 'preserved_recipe',
                    exact_parent_action=True) for r in data['preserved']]


def independent_repeat(discovery, group):
    if not captured(discovery) or group not in range(3) or discovery.get('ik_branch_slot') not in (0, 1):
        raise EvidenceError('Captured distinct-branch discovery required for confirmation')
    result = []
    for repeat in range(3):
        row = request(discovery, f'fixed35-six-repeat-c{group}-r{repeat}-{discovery["context_id"]}',
            'repeat', 'six_repeat', confirmation_index=group, repeat=repeat,
            selected_from_id=discovery['id'], selected_from_sha256=digest(discovery),
            proposal_index=discovery['proposal_index'], numeric_proposal_index=discovery['numeric_proposal_index'],
            ik_branch_slot=discovery['ik_branch_slot'], selected_connection_sha256=discovery['connection_sha256'],
            selected_q10=copy.deepcopy(discovery['q10']), selected_q14=copy.deepcopy(discovery['q14']))
        row.update(parent_row_id=discovery['parent_row_id'], parent_row_sha256=discovery['parent_row_sha256'])
        result.append(row)
    return result


def audit_plan(study, data, checks):
    study = Path(study)
    expected = independent_plan(data)
    checks.equal(read(study/'search-plan.json'),
        dict(rows=expected, frozen_before_outcomes=True, maximum_requests=768),
        'All768 pose/branch requests independently reconstructed before outcomes')
    checks.equal(read(study/'preserved-plan.json'),
        dict(rows=independent_preserved(data), frozen_before_outcomes=True),
        'All29 immutable preservation requests independently reconstructed')
    checks.equal(len({r['id'] for r in expected}), 768, 'No duplicate search request identities')
    for source in data['cases']:
        group = [r for r in expected if r['context_id'] == source['context_id']]
        checks.equal(len({tuple(r['values']) for r in group}), 96, source['context_id']+'/96 distinct numeric actions')
        for index in range(96):
            pair = group[2*index:2*index+2]
            checks.equal([r['ik_branch_slot'] for r in pair], [0, 1], source['context_id']+'/two branch slots/'+str(index))
            checks.equal(pair[0]['values'], pair[1]['values'], source['context_id']+'/matched branch geometry/'+str(index))
        for row in group:
            checks.equal(row['bucket_xy_m'], source['bucket_xy_m'], row['id']+'/unchanged bucket')
            checks.equal(row['values'], row['command_values'], row['id']+'/no hidden action substitution')
            checks.require(all(lo <= value <= hi for value, lo, hi in zip(row['values'], LOW, HIGH)),
                           row['id']+'/admissible un-clipped numeric proposal')
    return expected


def independent_groups(rows, context):
    result = []
    for number in range(3):
        group = [r for r in rows if r.get('phase') == 'repeat' and r['context_id'] == context and r['confirmation_index'] == number]
        if not group:
            continue
        first = group[0]
        consistent = (len(group) == 3 and sorted(r['repeat'] for r in group) == [0, 1, 2] and
                      len({r['id'] for r in group}) == 3 and all(all(r.get(key) == first.get(key) for key in
                      ('selected_from_id', 'selected_from_sha256', 'values', 'bucket_xy_m', 'ik_branch_slot',
                       'q10', 'q14', 'connection_sha256')) for r in group))
        successes = sum(captured(r) for r in group)
        result.append(dict(confirmation_index=number, selected_from_id=first['selected_from_id'],
            selected_from_sha256=first['selected_from_sha256'], values=first['values'], bucket_xy_m=first['bucket_xy_m'],
            ik_branch_slot=first['ik_branch_slot'], requests=len(group), captures=successes,
            record_ids=[r['id'] for r in group],
            status='confirmed' if consistent and successes == 3 else 'failed' if len(group) == 3 else 'incomplete'))
    return result


def independent_cases(rows, data, *, finished=False, budget_truncated=False):
    result = []
    for parent in sorted(data['failures'], key=lambda r: r['context_id']):
        attempts = [r for r in rows if r.get('phase') == 'search' and r['context_id'] == parent['context_id']]
        confirmations = independent_groups(rows, parent['context_id'])
        success = next((r for r in confirmations if r['status'] == 'confirmed'), None)
        status = ('fixed_base_clearance_blocked' if parent in data['blocked'] else 'confirmed' if success else
                  'confirmation_limit' if len(confirmations) == 3 else 'search_exhausted' if len(attempts) == 192 else
                  'budget_truncated' if finished and budget_truncated else 'in_progress')
        recipe = next((r for r in attempts if success and r['id'] == success['selected_from_id']), None)
        result.append(dict(context_id=parent['context_id'], bucket_xy_m=parent['bucket_xy_m'], parent_row_id=parent['id'],
            parent_row_sha256=digest(parent), status=status, search_requests=len(attempts),
            search_captures=sum(captured(r) for r in attempts), search_counts=statistics(attempts),
            confirmation_groups=confirmations, confirmed_recipe=copy.deepcopy(recipe)))
    return result


def audit_schedule(study, rows, report, data, plan, checks):
    """All adaptive decisions are reconstructed from already-recorded outcomes."""
    study = Path(study)
    boundaries = common.lines(study/'phase-boundaries.jsonl')
    raw = (study/'rows.jsonl').read_bytes().splitlines(keepends=True)
    timings = common.lines(study/'timing.jsonl')
    boundary_index = timing_index = position = 0
    previous_time = 0.

    def prefix(count):
        return hashlib.sha256(b''.join(raw[:count])).hexdigest() if count else None

    def boundary(phase, **extra):
        nonlocal boundary_index, previous_time
        checks.require(boundary_index < len(boundaries), phase+'/required boundary exists')
        row = boundaries[boundary_index]
        boundary_index += 1
        checks.equal((row['phase'], row['rows_count'], row['rows_sha256']),
                     (phase, position, prefix(position)), phase+'/immutable evidence prefix')
        checks.require(previous_time <= row['acquired_elapsed_s'] < 7200, phase+'/ordered bounded boundary time')
        previous_time = row['acquired_elapsed_s']
        for key, value in extra.items():
            checks.equal(row[key], value, phase+'/boundary '+key)
        return row

    def batch(phase, expected):
        nonlocal position, timing_index, previous_time
        group = rows[position:position+len(expected)]
        checks.equal(len(group), len(expected), phase+'/all requested outcomes present')
        checks.require(timing_index < len(timings), phase+'/execution timing exists')
        timing = timings[timing_index]
        timing_index += 1
        checks.equal(timing['phase'], phase, phase+'/timed phase')
        checks.require(previous_time <= timing['acquired_start_s'] <= timing['acquired_end_s'] < 7200,
                       phase+'/executed after freeze and inside7200s')
        for row, wanted in zip(group, expected):
            for key, value in wanted.items():
                checks.equal(row.get(key), value, row['id']+'/exact frozen request/'+key)
        previous_time = timing['acquired_end_s']
        position += len(group)
        return group

    boundary('inputs-frozen')
    batch('controls_pre', [dict(phase='controls_pre')]*9)
    boundary('controls_pre-complete')
    batch('preserved', independent_preserved(data))
    boundary('preserved-complete')
    pools = {c['context_id']: [r for r in plan if r['context_id'] == c['context_id']] for c in data['cases']}
    stopped = {}
    counts = {key: 0 for key in pools}
    rounds = 0
    repeat_files = []
    for round_index in range(192):
        active = [key for key in pools if key not in stopped]
        if not active:
            break
        checks.require(position < len(rows), 'Post controls must remain after search')
        if rows[position]['phase'] == 'controls_post':
            break
        frozen = boundary('search-round-frozen', proposal_index=round_index, contexts=active)
        checks.require(frozen['acquired_elapsed_s'] < 5400, 'No new round after5400s search cutoff')
        expected = [dict(pools[key][round_index], search_plan_sha256=sha(study/'search-plan.json')) for key in active]
        discoveries = batch('search', expected)
        selected = [r for r in discoveries if captured(r)]
        if selected:
            name = f'repeat-plans/round-{round_index:02d}.json'
            expected = [r for d in selected for r in independent_repeat(d, counts[d['context_id']])]
            wanted = dict(proposal_index=round_index, rows=expected,
                selected=[dict(id=r['id'], sha256=digest(r)) for r in selected], rows_count_before=position,
                rows_sha256_before=prefix(position), frozen_before_repeat_outcomes=True)
            checks.equal(read(study/name), wanted, name+'/fresh repeat plan follows exactly the selected discoveries')
            boundary('repeat-plan-frozen', proposal_index=round_index, path=name, sha256=sha(study/name))
            repeat_files.append(name)
            measured = batch('repeat', [dict(r, repeat_plan_file=name, repeat_plan_sha256=sha(study/name)) for r in expected])
            for selected_row in selected:
                key = selected_row['context_id']
                group = [r for r in measured if r['selected_from_id'] == selected_row['id']]
                checks.equal([r['repeat'] for r in group], [0, 1, 2], key+'/exactly3 fresh confirmations')
                for row in group:
                    for field in (*META, 'values', 'bucket_xy_m', 'ik_branch_slot', 'q10', 'q14',
                                  'connection_file', 'connection_sha256'):
                        checks.equal(row[field], selected_row[field], row['id']+'/unchanged discovered pose and branch/'+field)
                    checks.require(row['native_cache_sha256'] != selected_row['native_cache_sha256'],
                                   row['id']+'/fresh native trajectory, not reused discovery')
                counts[key] += 1
                if all(captured(r) for r in group):
                    stopped[key] = 'confirmed'
                elif counts[key] == 3:
                    stopped[key] = 'confirmation_limit'
        boundary('search-round-complete', proposal_index=round_index, stopped=stopped)
        rounds = round_index+1
    truncated = report['budget_truncated']
    checks.require(type(truncated) is bool, 'Explicit budget truncation flag')
    if truncated:
        checks.require(rounds < 192 and len(stopped) < 4, 'Truncated run has remaining proposals')
        reason = 'search_time_limit'
    else:
        checks.require(rounds == 192 or len(stopped) == 4, 'No unexplained missing active case/branch slot')
        reason = ('all_searchable_cases_confirmed' if len(stopped) == 4 and set(stopped.values()) == {'confirmed'}
                  else 'finite_candidate_or_confirmation_caps')
    checks.equal(report['completion_reason'], reason, 'Completion reason independently reconstructed')
    end = boundary('search-complete', completion_reason=reason, budget_truncated=truncated)
    if truncated:
        checks.require(end['acquired_elapsed_s'] >= 5400, 'Truncation follows actual search cutoff')
    batch('controls_post', [dict(phase='controls_post')]*9)
    boundary('controls_post-complete')
    checks.equal((position, boundary_index, timing_index), (len(rows), len(boundaries), len(timings)),
                 'No extra or omitted requests, freezes or timing batches')
    checks.equal(sorted(str(p.relative_to(study)) for p in (study/'repeat-plans').glob('*.json')),
                 sorted(repeat_files), 'Every stored confirmation plan accounted')
    checks.require(len(rows) <= 851 and len({r['id'] for r in rows}) == len(rows), 'All unique requests within851 cap')
    return dict(total_requests=len(rows), preserved_requests=29, control_requests=18, search_rounds=rounds,
        search_requests=sum(r['phase'] == 'search' for r in rows), repeat_requests=sum(r['phase'] == 'repeat' for r in rows),
        maximum_requests=851, budget_truncated=truncated, stopped=stopped)
