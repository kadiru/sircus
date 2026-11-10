"""Frozen pose/IK-branch search on the six remaining fixed-grid locations."""
from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np

from .benchmark_verify import digest, read, sha
from .corrected_imitation_contract import authenticate_study, require
from .fixed35_imitation_contract import fixed_grid, ROTATION
from .fixed35_recovery_contract import request, diagnose_failures
from .table_training_contract import captured, clean, statistics
from .table_assisted_score import reward

VERSION = 'sircus-fixed35-six-v1'
AUDIT_VERSION = 'sircus-fixed35-six-audit-v1'
INPUT_KEYS = ('parent-study', 'parent-audit', 'original-study', 'original-audit', 'reference', 'assets')
METHODS = ('preserved_recipe', 'six_search', 'six_repeat')
LOW = np.asarray([.30, -.50, .12, 8., .05])
HIGH = np.asarray([1.04, .35, .60, 60., .65])
PARAMETERS = ('board_x_m', 'board_y_m', 'board_height_m', 'tilt_deg', 'drop_gap_m')
CASE_IDS = ('fixed35-dev-grid-01-05', 'fixed35-dev-grid-03-04',
            'fixed35-dev-grid-03-05', 'fixed35-dev-grid-04-06')
BLOCKED_IDS = ('fixed35-dev-grid-03-00', 'fixed35-dev-grid-03-06')
NUMERIC_PROPOSALS = 96
BRANCH_SLOTS = (0, 1)
SEARCH_REQUESTS = len(CASE_IDS) * NUMERIC_PROPOSALS * len(BRANCH_SLOTS)
MAXIMUM_REQUESTS = SEARCH_REQUESTS + 36 + 29 + 18
SCOPE = (
    'Bounded development search on four unresolved historical fixed-grid locations, '
    'with two fixed-base clearance cases receiving CPU geometry audit only. '
    'Twenty-nine successful recipes keep their exact actions and admitted joints. '
    'Ninety-six unique numeric proposals per searched location each offer two distinct '
    'IK branch slots; bucket positions, bucket-facing yaw and native64/v3 stay fixed. '
    'A discovered recipe needs three fresh nominal captures. No policy fitting, '
    'promotion, new positions, noise, 128-step qualification, transit learning, '
    'hardware or UI changes.'
)


def protocol():
    return dict(
        substeps=64, methods=list(METHODS), grid_positions=35, preserved_recipes=29,
        historical_policy_captures=24, development_failures=6, fixed_base_audits=2,
        search_layouts=4, numeric_proposals_per_layout=NUMERIC_PROPOSALS,
        branch_slots=list(BRANCH_SLOTS), requests_per_layout=192,
        search_requests=SEARCH_REQUESTS, repeat_per_selected=3,
        candidates_per_layout=3, repeat_requests_max=36, preserved_requests=29,
        control_requests=18, maximum_requests=MAXIMUM_REQUESTS,
        noise=False, bucket_perturbations=False, training=False, actor_steps=0,
        critic_steps=0, ui_changes=False, worker_limit_s=7200, search_limit_s=5400,
        queue_limit_s=900, global_limit_s=9900, audit_reserve_s=1800,
        storage_reserve_bytes=8*1024**3, rotation=ROTATION,
        bounds=dict(low=LOW.tolist(), high=HIGH.tolist()),
        numeric_plan=dict(
            anchors='Unique original, historical refit, and six nearest preserved recipes',
            local_polar='Parent radius offsets -120,-60,+60,+120mm crossed with azimuth offsets -30,-15,+15,+30deg',
            coupled_polar='Deterministic radical-inverse bases 2,3,5,7,11; radius180..900mm, parent azimuth +/-100deg, height140..560mm, tilt10..54deg, gap80..600mm',
            bounds='Reject out-of-bounds proposals; never clip. Deduplicate exact five-number actions before branch expansion.',
            frozen=True),
        search_order='For each numeric proposal, branch slot0 then1; one request per active layout in each round. Stop a case after3/3 fresh confirmations or3 failed confirmation groups.',
        confirmation='Replay the exact selected connection and both joint vectors three fresh times; no re-solving or numeric adjustment.',
        geometry='Two blocked cells receive CPU geometry diagnostics only; no guard relaxation or automatic admission.',
        preservation='Replay24 original successes and5 confirmed recovery recipes with source-bound unchanged action and joint branch.',
        evaluation_scope='Historical fixed-grid recipe development, not held out or a new policy score',
        scope=SCOPE)


def _bind_native(study, row, *, trace=False):
    for prefix in ('context_resolution', 'connection'):
        require(sha(study['bound']('native-yaw/'+row[prefix+'_file'])) == row[prefix+'_sha256'],
                'Authenticated native proof differs')
    connection = read(study['bound']('native-yaw/'+row['connection_file']))
    require(connection.get('release_behavior', {}).get('version') ==
            'sircus-continuous-bucket-facing-release-v1', 'Fixed bucket-facing yaw proof required')
    if trace and row.get('native_release'):
        require(sha(study['bound']('native-yaw/'+row['native_cache_file'])) == row['native_cache_sha256'],
                'Authenticated native trajectory differs')


def cohort(original_rows, recovery_rows, recovery_report):
    """Join the two authenticated studies without replacing any historical row."""
    current = sorted((r for r in original_rows if r.get('phase') == 'test' and r.get('method') == 'current_bc'),
                     key=lambda r: r['context_id'])
    refit = sorted((r for r in original_rows if r.get('phase') == 'test' and r.get('method') == 'fixed35_bc'),
                   key=lambda r: r['context_id'])
    expected = {r['id']: r['bucket_xy_m'] for r in fixed_grid()}
    require(len(current) == len(refit) == 35 and
            all({r['context_id']: r['bucket_xy_m'] for r in group} == expected for group in (current, refit)),
            'Exact original35 current/refit rows required')
    old_successes = [r for r in current if captured(r)]
    require(len(old_successes) == 24, 'Original current24 captures required')
    replay = [r for r in recovery_rows if r.get('phase') == 'preserved']
    require(len(replay) == 24 and all(captured(r) for r in replay), 'Parent recovery preserved24 must all capture')
    by_original = {r['id']: r for r in current}
    for r in replay:
        parent = by_original.get(r['parent_row_id'])
        require(parent is not None and captured(parent) and r['parent_row_sha256'] == digest(parent),
                'Preserved original source binding differs')
        require(all(r[k] == parent[k] for k in ('values', 'bucket_xy_m', 'q10', 'q14', 'connection_sha256')),
                'Preserved original action/joints differ')
    by_id = {r['id']: r for r in recovery_rows}
    require(len(by_id) == len(recovery_rows), 'Unique recovery evidence rows required')
    confirmed = []
    for c in recovery_report['cases']:
        recipe = c.get('confirmed_recipe')
        if recipe is None:
            continue
        source = by_id.get(recipe['id'])
        require(source == recipe and source.get('phase') == 'search' and captured(source),
                'Confirmed recipe must equal authenticated captured discovery')
        repeats = sorted((r for r in recovery_rows if r.get('phase') == 'repeat' and
                          r.get('selected_from_id') == source['id']), key=lambda r: r['repeat'])
        require(len(repeats) == 3 and [r['repeat'] for r in repeats] == [0, 1, 2] and
                all(captured(r) for r in repeats), 'Each recovery recipe needs three captured fresh repeats')
        for r in repeats:
            require(r['selected_from_sha256'] == digest(source) and all(r[k] == source[k]
                    for k in ('values', 'bucket_xy_m', 'q10', 'q14', 'connection_sha256')),
                    'Recovery confirmation source/action/joints differ')
        confirmed.append(source)
    require(len(confirmed) == 5, 'Exactly five authenticated recovery recipes required')
    preserved = sorted(old_successes+confirmed, key=lambda r: r['context_id'])
    require(len({r['context_id'] for r in preserved}) == 29, 'Exactly29 distinct preserved layouts required')
    failed = [r for r in current if r['context_id'] not in {s['context_id'] for s in preserved}]
    cases = [r for r in failed if r['context_id'] in CASE_IDS]
    blocked = [r for r in failed if r['context_id'] in BLOCKED_IDS]
    require(tuple(r['context_id'] for r in cases) == CASE_IDS and
            tuple(r['context_id'] for r in blocked) == BLOCKED_IDS and len(failed) == 6,
            'Published four search/two blocked cohort differs')
    require(all(r.get('invalid') == 'fixed_base_clearance_rejected' and not r['native_release']
                for r in blocked), 'Only previously bound fixed-base clearance cases stay blocked')
    return dict(current=current, refit=refit, preserved=preserved, old_successes=old_successes,
                recovered=confirmed, cases=cases, blocked=blocked, failures=failed, grid=fixed_grid())


def load_inputs(a, root):
    public = read(Path(root)/'baselines/mjlab-fixed35-recovery-20261109/summary.json')
    original_public = read(Path(root)/'baselines/mjlab-fixed35-imitation-20261109/summary.json')
    parent = authenticate_study(a.parent_study, a.parent_audit,
        dict(report_sha256=public['report_sha256'], proof_sha256=public['audit']['proof_sha256']), checks_count=141041)
    original = authenticate_study(a.original_study, a.original_audit,
        dict(report_sha256=original_public['report_sha256'], proof_sha256=original_public['audit']['proof_sha256']), checks_count=40692)
    require(len(parent['rows']) == 441 and len(original['rows']) == 88, 'Complete published441/88 evidence required')
    require(parent['spec']['source_bindings']['parent']['report_sha256'] == original_public['report_sha256'],
            'Recovery original study lineage differs')
    out = cohort(original['rows'], parent['rows'], parent['report'])
    source_roots = {}
    for row in out['current']+out['refit']:
        _bind_native(original, row, trace=row in out['failures'])
    for row in out['preserved']+out['failures']:
        source = parent if row in out['recovered'] else original
        _bind_native(source, row, trace=row in out['recovered'])
        source_roots[row['id']] = source['directory']/'native-yaw'
    for key, name in (('prior', 'prior.json'), ('old_prior', 'old-prior.json'),
                      ('controls', 'controls.json'), ('control_replays', 'control-replays.json'),
                      ('calibration', 'calibration.json'), ('home', 'setup-home.json')):
        out[key] = read(parent['bound'](name))
        require(out[key] == read(original['bound'](name)), 'Pinned parent input differs: '+name)
    require(len(out['prior']['records']) == 19 and len(out['old_prior']['records']) == 17,
            'Unchanged19/17 geometry priors required')
    for name in ('verification.json', 'native-yaw/effective-options.json', 'native-yaw/native-compiled-geometry.json',
                 'control-parent/native/effective-options.json', 'control-parent/native/native-compiled-geometry.json'):
        require(sha(parent['bound'](name)) == sha(original['bound'](name)), 'Parent native options/geometry differ: '+name)
    for row in out['control_replays']:
        for prefix in ('context_resolution', 'connection'):
            require(sha(parent['bound']('control-parent/native/'+row[prefix+'_file'])) == row[prefix+'_sha256'],
                    'Control native proof differs')
    artifacts = copy.deepcopy(original['report']['artifacts'])
    for value in artifacts.values():
        require(sha(original['bound'](value['path'])) == value['sha256'], 'Original model checkpoint differs')
    require(all(parent['spec']['identity'][k] == original['spec']['identity'][k]
                for k in ('scene_sha256', 'asset_sha256')) and parent['spec']['packages'] == original['spec']['packages'],
            'Source scene/assets/packages differ')
    out.update(parent=parent['spec'], parent_report=parent['report'], artifacts=artifacts,
        source_roots=source_roots, sources=out['preserved']+out['failures'],
        diagnoses=diagnose_failures(Path(a.original_study), out['failures']),
        bindings=dict(parent=parent['bindings'], original=original['bindings'],
            recipe_source={r['id']: dict(study='recovery' if r in out['recovered'] else 'original', row_sha256=digest(r))
                           for r in out['preserved']+out['failures']},
            scope='Original policy24/35 and verified recipe coverage29/35 remain historical; new cases are development'))
    return out


def radical_inverse(index, base):
    result, scale = 0., 1.
    while index:
        index, digit = divmod(index, base)
        scale /= base
        result += digit*scale
    return result


def proposals(source, refit, successes):
    """96 unique actions; two branch slots are expanded only after freezing actions."""
    parent = np.asarray(source['values'], float)
    bucket = np.asarray(source['bucket_xy_m'], float)
    direction = parent[:2]-bucket
    radius = float(np.linalg.norm(direction))
    require(parent.shape == (5,) and bucket.shape == (2,) and np.isfinite(parent).all() and radius > 0.,
            'Finite parent and nonzero board-to-bucket distance required')
    azimuth = math.atan2(direction[1], direction[0])
    items, seen = [], set()

    def add(values, kind, lineage=(), **extra):
        raw = np.asarray(values, float)
        require(raw.shape == (5,) and np.isfinite(raw).all(), 'Finite five-value candidate required')
        if np.any(raw < LOW) or np.any(raw > HIGH) or tuple(raw) in seen:
            return False
        seen.add(tuple(raw))
        items.append(dict(values=raw.tolist(), kind=kind,
                          parent_source_ids=list(lineage) or [source['id']], **extra))
        return True

    add(parent, 'original_failed_action')
    alternate = [r for r in refit if r['context_id'] == source['context_id']]
    require(len(alternate) == 1, 'Unique historical refit action required')
    add(alternate[0]['values'], 'historical_refit_action', [alternate[0]['id']])
    nearest = sorted(successes, key=lambda r: (float(np.linalg.norm(np.asarray(r['bucket_xy_m'])-bucket)), r['id']))[:6]
    require(len(nearest) == 6, 'Six source-bound successful neighbors required')
    for row in nearest:
        add(row['values'], 'neighbor_recipe_at_fixed_bucket', [row['id']])
    for offset_m in (-.12, -.06, .06, .12):
        for offset_deg in (-30., -15., 15., 30.):
            r = radius+offset_m
            if r <= 0.:
                continue
            angle = azimuth+math.radians(offset_deg)
            values = parent.copy()
            values[:2] = bucket+r*np.asarray([math.cos(angle), math.sin(angle)])
            add(values, 'local_polar_xy', radius_m=r, azimuth_rad=angle,
                radius_offset_m=offset_m, azimuth_offset_deg=offset_deg)
    for index in range(1, 10001):
        if len(items) == NUMERIC_PROPOSALS:
            break
        coordinates = [radical_inverse(index, base) for base in (2, 3, 5, 7, 11)]
        r = .18+.72*coordinates[0]
        angle = azimuth+math.radians(-100.+200.*coordinates[1])
        xy = bucket+r*np.asarray([math.cos(angle), math.sin(angle)])
        values = [*xy, .14+.42*coordinates[2], 10.+44.*coordinates[3], .08+.52*coordinates[4]]
        add(values, 'coupled_polar_height_tilt_gap', radical_inverse_index=index,
            radius_m=r, azimuth_rad=angle, unit_coordinates=coordinates)
    require(len(items) == NUMERIC_PROPOSALS, 'Bounded unique proposal generation exhausted')
    for index, row in enumerate(items):
        row['numeric_proposal_index'] = index
    return items


def make_plan(cases, refit, successes):
    require(tuple(r['context_id'] for r in cases) == CASE_IDS, 'Exactly four ordered searchable cases required')
    pools = {r['context_id']: proposals(r, refit, successes) for r in cases}
    plan = []
    for index in range(NUMERIC_PROPOSALS):
        for slot in BRANCH_SLOTS:
            for source in cases:
                proposal = pools[source['context_id']][index]
                plan.append(request(source,
                    identity=f'fixed35-six-search-p{index:02d}-b{slot}-{source["context_id"]}',
                    phase='search', method='six_search', values=proposal['values'],
                    proposal_index=index*2+slot, numeric_proposal_index=index,
                    ik_branch_slot=slot, derivation=copy.deepcopy(proposal), exact_parent_action=False))
    require(len(plan) == SEARCH_REQUESTS and len({r['id'] for r in plan}) == SEARCH_REQUESTS,
            'Exactly768 distinct frozen search requests required')
    return plan


def preserved_plan(successes):
    require(len(successes) == 29 and len({r['context_id'] for r in successes}) == 29 and
            all(captured(r) for r in successes), 'Exactly29 captured source recipes required')
    return [request(row, identity='fixed35-six-preserved-'+row['context_id'], phase='preserved',
                    method='preserved_recipe', exact_parent_action=True) for row in successes]


def repeat_plan(discovery, confirmation_index):
    require(captured(discovery) and confirmation_index in range(3) and
            discovery.get('ik_branch_slot') in BRANCH_SLOTS, 'Captured discovery and valid branch/group required')
    result = []
    for repeat in range(3):
        row = request(discovery,
            identity=f'fixed35-six-repeat-c{confirmation_index}-r{repeat}-{discovery["context_id"]}',
            phase='repeat', method='six_repeat', confirmation_index=confirmation_index, repeat=repeat,
            selected_from_id=discovery['id'], selected_from_sha256=digest(discovery),
            proposal_index=discovery['proposal_index'], numeric_proposal_index=discovery['numeric_proposal_index'],
            ik_branch_slot=discovery['ik_branch_slot'], selected_connection_sha256=discovery['connection_sha256'],
            selected_q10=copy.deepcopy(discovery['q10']), selected_q14=copy.deepcopy(discovery['q14']))
        row.update(parent_row_id=discovery['parent_row_id'], parent_row_sha256=discovery['parent_row_sha256'])
        result.append(row)
    return result


def confirmation_groups(rows, context_id):
    groups = []
    for index in range(3):
        selected = [r for r in rows if r.get('phase') == 'repeat' and r['context_id'] == context_id and r['confirmation_index'] == index]
        if not selected:
            continue
        first = selected[0]
        exact = (len(selected) == 3 and sorted(r['repeat'] for r in selected) == [0, 1, 2] and
                 len({r['id'] for r in selected}) == 3 and all(all(r.get(k) == first.get(k) for k in
                 ('selected_from_id', 'selected_from_sha256', 'values', 'bucket_xy_m', 'ik_branch_slot',
                  'q10', 'q14', 'connection_sha256')) for r in selected))
        status = ('confirmed' if exact and all(captured(r) for r in selected) else
                  'failed' if len(selected) == 3 else 'incomplete')
        groups.append(dict(confirmation_index=index, selected_from_id=first['selected_from_id'],
            selected_from_sha256=first['selected_from_sha256'], values=first['values'], bucket_xy_m=first['bucket_xy_m'],
            ik_branch_slot=first['ik_branch_slot'], requests=len(selected), captures=sum(captured(r) for r in selected),
            record_ids=[r['id'] for r in selected], status=status))
    return groups


def case_results(rows, cases, blocked, *, finished=False, budget_truncated=False):
    results = []
    for parent in sorted(cases+blocked, key=lambda r: r['context_id']):
        search = [r for r in rows if r.get('phase') == 'search' and r['context_id'] == parent['context_id']]
        groups = confirmation_groups(rows, parent['context_id'])
        confirmed = next((g for g in groups if g['status'] == 'confirmed'), None)
        status = ('fixed_base_clearance_blocked' if parent in blocked else 'confirmed' if confirmed else
                  'confirmation_limit' if len(groups) == 3 else 'search_exhausted' if len(search) == 192 else
                  'budget_truncated' if finished and budget_truncated else 'in_progress')
        recipe = next((r for r in search if confirmed and r['id'] == confirmed['selected_from_id']), None)
        results.append(dict(context_id=parent['context_id'], bucket_xy_m=parent['bucket_xy_m'],
            parent_row_id=parent['id'], parent_row_sha256=digest(parent), status=status,
            search_requests=len(search), search_captures=sum(captured(r) for r in search),
            search_counts=statistics(search), confirmation_groups=groups, confirmed_recipe=copy.deepcopy(recipe)))
    return results
