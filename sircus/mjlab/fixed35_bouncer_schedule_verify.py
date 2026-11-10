"""Independent bounded replay schedule audit; imports no producer selector."""
from __future__ import annotations
import copy
import hashlib
from pathlib import Path
from . import benchmark_verify as common
from . import fixed35_recovery_verify as old
from .benchmark_verify import digest, read, sha
META = old.META
captured = old.captured
statistics = old.statistics

def audit_schedule(study, rows, report, data, plan, preserved, repeat_builder, checks):
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
        checks.require(previous_time <= row['acquired_elapsed_s'] < 3600, phase+'/ordered bounded boundary time')
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
        checks.require(previous_time <= timing['acquired_start_s'] <= timing['acquired_end_s'] < 3600,
                       phase+'/executed after freeze and inside3600s')
        for row, wanted in zip(group, expected):
            for key, value in wanted.items():
                checks.equal(row.get(key), value, row['id']+'/exact frozen request/'+key)
        previous_time = timing['acquired_end_s']
        position += len(group)
        return group

    boundary('inputs-frozen')
    batch('controls_pre', [dict(phase='controls_pre')]*9)
    boundary('controls_pre-complete')
    checks.equal(len(preserved), 30, 'All30 prior recipes replayed')
    batch('preserved', preserved)
    boundary('preserved-complete')
    pools = {c['context_id']: [r for r in plan if r['context_id'] == c['context_id']] for c in data['cases']}
    stopped = {}
    counts = {key: 0 for key in pools}
    rounds = 0
    repeat_files = []
    checks.require(len(plan) <= 24 and len({r['id'] for r in plan}) == len(plan), 'Unique bounded saved-candidate plan')
    maximum_rounds = max((len(v) for v in pools.values()), default=0)
    for round_index in range(maximum_rounds):
        active = [key for key in pools if key not in stopped and round_index < len(pools[key])]
        if not active:
            break
        checks.require(position < len(rows), 'Post controls must remain after search')
        if rows[position]['phase'] == 'controls_post':
            break
        frozen = boundary('search-round-frozen', proposal_index=round_index, contexts=active)
        checks.require(frozen['acquired_elapsed_s'] < 2400, 'No new round after2400s search cutoff')
        expected = [dict(pools[key][round_index], search_plan_sha256=sha(study/'search-plan.json')) for key in active]
        discoveries = batch('search', expected)
        selected = [r for r in discoveries if captured(r)]
        if selected:
            name = f'repeat-plans/round-{round_index:02d}.json'
            expected = [r for d in selected for r in repeat_builder(d, counts[d['context_id']])]
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
                    for field in (*META, 'values', 'bucket_xy_m', 'candidate_index', 'candidate_pool_sha256', 'q10', 'q14',
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
        checks.require(any(c not in stopped and rounds < len(pool) for c, pool in pools.items()), 'Truncated run has remaining proposals')
        reason = 'search_time_limit'
    else:
        checks.require(all(c in stopped or rounds >= len(pool) for c, pool in pools.items()), 'No unexplained missing active saved candidate')
        reason = ('all_searchable_cases_confirmed' if len(stopped) == len(pools) and set(stopped.values()) == {'confirmed'}
                  else 'finite_candidate_or_confirmation_caps')
    checks.equal(report['completion_reason'], reason, 'Completion reason independently reconstructed')
    end = boundary('search-complete', completion_reason=reason, budget_truncated=truncated)
    if truncated:
        checks.require(end['acquired_elapsed_s'] >= 2400, 'Truncation follows actual search cutoff')
    batch('controls_post', [dict(phase='controls_post')]*9)
    boundary('controls_post-complete')
    checks.equal((position, boundary_index, timing_index), (len(rows), len(boundaries), len(timings)),
                 'No extra or omitted requests, freezes or timing batches')
    checks.equal(sorted(str(p.relative_to(study)) for p in (study/'repeat-plans').glob('*.json')),
                 sorted(repeat_files), 'Every stored confirmation plan accounted')
    checks.require(len(rows) <= 99 and len({r['id'] for r in rows}) == len(rows), 'All unique requests within99 cap')
    return dict(total_requests=len(rows), preserved_requests=30, control_requests=18, search_rounds=rounds,
        search_requests=sum(r['phase'] == 'search' for r in rows), repeat_requests=sum(r['phase'] == 'repeat' for r in rows),
        maximum_requests=99, budget_truncated=truncated, stopped=stopped)


def independent_groups(rows, context):
    result = []
    for number in range(3):
        group = [r for r in rows if r.get('phase') == 'repeat' and r['context_id'] == context and r['confirmation_index'] == number]
        if not group:
            continue
        first = group[0]
        consistent = (len(group) == 3 and sorted(r['repeat'] for r in group) == [0, 1, 2] and
                      len({r['id'] for r in group}) == 3 and all(all(r.get(key) == first.get(key) for key in
                      ('selected_from_id', 'selected_from_sha256', 'values', 'bucket_xy_m', 'candidate_sha256',
                       'q10', 'q14', 'connection_sha256')) for r in group))
        successes = sum(captured(r) for r in group)
        result.append(dict(confirmation_index=number, selected_from_id=first['selected_from_id'],
            selected_from_sha256=first['selected_from_sha256'], values=first['values'], bucket_xy_m=first['bucket_xy_m'],
            candidate_sha256=first['candidate_sha256'], requests=len(group), captures=successes,
            record_ids=[r['id'] for r in group],
            status='confirmed' if consistent and successes == 3 else 'failed' if len(group) == 3 else 'incomplete'))
    return result


def independent_cases(rows, data, plan, *, finished=False, budget_truncated=False):
    result = []
    for parent in sorted(data['failures'], key=lambda r: r['context_id']):
        attempts = [r for r in rows if r.get('phase') == 'search' and r['context_id'] == parent['context_id']]
        capacity = sum(r['context_id'] == parent['context_id'] for r in plan)
        confirmations = independent_groups(rows, parent['context_id'])
        success = next((r for r in confirmations if r['status'] == 'confirmed'), None)
        status = ('fixed_base_clearance_blocked' if parent in data['blocked'] else 'confirmed' if success else
                  'confirmation_limit' if len(confirmations) == 3 else 'search_exhausted' if len(attempts) == capacity else
                  'budget_truncated' if finished and budget_truncated else 'in_progress')
        recipe = next((r for r in attempts if success and r['id'] == success['selected_from_id']), None)
        result.append(dict(context_id=parent['context_id'], bucket_xy_m=parent['bucket_xy_m'], parent_row_id=parent['id'],
            parent_row_sha256=digest(parent), status=status, planned_search_requests=capacity, search_requests=len(attempts),
            search_captures=sum(captured(r) for r in attempts), search_counts=statistics(attempts),
            confirmation_groups=confirmations, confirmed_recipe=copy.deepcopy(recipe)))
    return result
