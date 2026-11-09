"""Independent CPU audit for the opt-in table-assisted fixed64 training pilot.

This checker never initializes a GPU. It reconstructs TRAIN membership, PPO
updates and frozen inference, audits unchanged geometry, independently rescoring
both historical-v2 and new-v3 labels from each fresh dense native trace.
"""
from __future__ import annotations
import argparse
from collections import Counter
import copy
import json
import math
from pathlib import Path
import time
import numpy as np
from . import benchmark_verify as common
from . import calibrated_benchmark_verify as calibrated
from . import policy64_verify as old
from . import replay650_verify as replay
from . import table_assisted_score_oracle as oracle
from .benchmark_verify import EvidenceError, contained, digest, read, sha

VERSION = "sircus-table-assisted-training-audit-v1"
STUDY_VERSION = "sircus-table-assisted-training-v1"
SEED = 20261511
VAL_SEED, TEST_SEED = 20262203, 20262204
MILESTONES = (0, 256, 512)
METHODS = ("old_bc", "old_ppo", "bc", "ppo")
CONTROL_ROLES = ("direct_control", "table_control", "negative_control")
RESIDUAL = np.asarray([.04, .04, .05, 8., .12])
MAX_REQUESTS = 766
AuditChecks = replay.AuditChecks
bootstrap_replay = old.bootstrap_replay
independent_policy = old.independent_policy

def captured(row):
    return bool(row.get('native_release') and row.get('route_success')
                and not row.get('invalid') and not row.get('invalid_physics'))


def statistics(rows):
    return dict(attempts=len(rows), captures=sum(captured(r) for r in rows),
        native_releases=sum(bool(r.get('native_release')) for r in rows),
        invalid=sum(bool(r.get('invalid')) for r in rows),
        clean=sum(captured(r) and r.get('route_quality') == 2 for r in rows),
        mean_reward=float(np.mean([oracle.reward(r) for r in rows])) if rows else 0.,
        direct_captures=sum(captured(r) and r.get('direct_capture') is True for r in rows),
        table_assisted_captures=sum(captured(r) and r.get('table_assisted_capture') is True for r in rows),
        setup_rejections=sum(not r.get('native_release') for r in rows),
        native_misses=sum(bool(r.get('native_release')) and not captured(r) for r in rows),
        forbidden_contact_shots=sum(bool(r.get('forbidden_contact_steps')) for r in rows),
        invalid_physics=sum(bool(r.get('invalid_physics')) for r in rows))


def best_teachers(rows):
    groups = {}
    for row in rows:
        if row.get('role') != 'train':
            raise EvidenceError('Held-out or control outcomes cannot become actor teachers')
        if captured(row):
            groups.setdefault(row['context_id'], []).append(row)
    return [min(groups[k], key=lambda r: (-int(r.get('route_quality') == 2), -float(r['learning_reward']), r['id']))
            for k in sorted(groups)]


def protocol_checks(protocol, checks):
    expected = dict(substeps=64, seed=SEED, train_layouts=35, validation_layouts=12,
        test_layouts=50, train_requests=512, batch_size=32, updates=16,
        milestones=list(MILESTONES), test_methods=list(METHODS), test_repeats=1,
        validation_requests=36, test_requests=200, control_requests=18,
        maximum_fresh_requests=MAX_REQUESTS, noise=False, actor_steps=800,
        critic_steps=400, bootstrap_lr=.001, ppo_lr=.0003, initial_logstd=-3.,
        ppo_epochs=6, clip=.2, entropy=.001, gradient_norm=1.,
        validation_seed=VAL_SEED, test_seed=TEST_SEED, minimum_split_distance_m=.005,
        scoring_version=oracle.VERSION, route_version=oracle.ROUTE_VERSION,
        reward_version=oracle.REWARD_VERSION, acquired_slot_limit_s=7200)
    for key, value in expected.items():
        checks.equal(protocol.get(key), value, 'Literal v3 pilot protocol/' + key)


def audit_sequence(rows, selected_samples, old_samples, checks):
    expected = [("controls_pre", role) for role in CONTROL_ROLES for _ in range(3)]
    expected += [("validation", "validation")] * 12
    for offset in range(0, 512, 32):
        expected += [("train", "train")] * 32
        if offset + 32 in MILESTONES:
            expected += [("validation", "validation")] * 12
    expected += [("test", "test")] * 200
    expected += [("controls_post", role) for role in CONTROL_ROLES for _ in range(3)]
    checks.equal(len(rows), MAX_REQUESTS, 'Exactly766 paid requests, including all declines')
    checks.equal([(r['phase'], r['role']) for r in rows], expected, 'Control/TRAIN/validation/TEST chronology')
    checks.equal(len({r['id'] for r in rows}), len(rows), 'Unique native request identities')
    tests = [r for r in rows if r['role'] == 'test']
    checks.equal([r['method'] for r in tests], [m for m in METHODS for _ in range(50)], 'Four paired frozen methods')
    samples = [old_samples['old_bc'], old_samples['old_ppo'], 0, selected_samples]
    checks.equal([r['samples'] for r in tests], [n for n in samples for _ in range(50)], 'Only prescribed immutable model prefixes')


def regenerated_splits(parent_splits, exclusions):
    train = [dict(id=c['id'], split='train', bucket_xy_m=copy.deepcopy(c['bucket_xy_m']))
             for c in parent_splits['train']]
    used = np.asarray(sorted({tuple(map(float, xy)) for xy in exclusions}), dtype=float)
    if used.ndim != 2 or used.shape[1] != 2 or not np.isfinite(used).all():
        raise EvidenceError('Frozen finite historical coordinate exclusions required')
    result = dict(train=train)
    for role, count, seed in (('validation', 12, VAL_SEED), ('test', 50, TEST_SEED)):
        rng, rows = np.random.default_rng(seed), []
        for _ in range(100000):
            if len(rows) == count:
                break
            xy = rng.uniform([.1, -.65], [1.24, .25])
            if np.linalg.norm(used - xy, axis=1).min() < .005:
                continue
            rows.append(dict(id=f'table-training-{role}-{len(rows):02d}', split=role, bucket_xy_m=xy.tolist()))
            used = np.vstack((used, xy))
        if len(rows) != count:
            raise EvidenceError('Could not reproduce fresh coordinate manifest')
        result[role] = rows
    return result


def check_rows(rows, checks):
    for row in rows:
        label = row['id']
        values = np.asarray(row['values'], dtype=float)
        checks.require(values.shape == (5,) and np.isfinite(values).all(), label + '/finite exact action')
        checks.equal(row['values'], row['command_values'], label + '/no substituted action')
        checks.require(row.get('action_substituted') is False, label + '/exact-action admission preserved')
        checks.equal(row['perturbation_mixed_units'], [0.] * 5, label + '/noise deferred')
        checks.numeric(row['learning_reward'], oracle.reward(row), label + '/independent v3 terminal reward', 1e-8)
        if row.get('native_release'):
            checks.equal(row['scoring_version'], oracle.VERSION, label + '/explicit native v3 scorer')
            checks.equal(row['route_version'], oracle.ROUTE_VERSION, label + '/explicit native v3 route')
            checks.equal(row['reward_version'], oracle.REWARD_VERSION, label + '/explicit native v3 reward')
            checks.equal(row['contact_protocol'], dict(substeps=64, board_mode='articulated', gravity_mode='passive_only'), label + '/unchanged64 native contact protocol')
            checks.require(row.get('scalar_score_audit_passed') is True, label + '/online oracle agreement')
            checks.equal(row['score_adapter_version'], 'sircus-table-assisted-held-endpoint-v3', label + '/explicit opt-in scoring adapter')
        else:
            checks.require(bool(row.get('invalid')) and not captured(row), label + '/setup refusal retained in denominator')
        if captured(row):
            checks.require(row.get('board_bounce') and row.get('whole_ball_entry')
                and row.get('max_retention_s', 0) >= 1 - 1e-9 and not row.get('forbidden_contact_steps'), label + '/complete permitted route')
            checks.equal(int(row.get('direct_capture')) + int(row.get('table_assisted_capture')), 1, label + '/direct and table capture partition')


def rescore_rows(directory, rows, scene, calibration, trace_spec, assets, checks):
    """Audit original physics evidence under v2, then independently score v3.

    The inherited v2 checker retains native clock, contact geometry, tangency,
    measured pose and dense-array association validation. Only a copied view of
    labels is restored for that checker; original files remain untouched.
    """
    old_rows = []
    for row in rows:
        value = copy.deepcopy(row)
        if row.get('native_release'):
            value.update(row['historical_v2_score'])
        old_rows.append(value)
    traces = calibrated.rescore_rows(directory, old_rows, scene, calibration, checks,
        'Unchanged native evidence and historical v2 labels', trace_spec, assets=assets)
    files = {}
    for row in rows:
        if row.get('native_release'):
            files.setdefault(row['native_cache_file'], []).append(row)
    for name, members in files.items():
        with np.load(contained(directory, name), allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files if key != 'record_ids'}
        for row in members:
            index = row['native_cache_row']
            trace = {key: value[:, index] if value.ndim >= 2 and key != 'time_s' else value for key, value in arrays.items()}
            score = oracle.score_trace(scene, row['bucket_xy_m'], row['board_normal'], calibration, trace)
            check_native_score(row, score, checks)
    return traces


def check_native_score(row, score, checks):
    for key, value in score.items():
        checks.require(key in row, row['id'] + '/complete oracle field/' + key)
        if key == 'invalid':
            # The unchanged adapter turns a native scorer's physics reason into
            # its explicit execution-level reason. It never clears invalidity.
            checks.equal(row[key], 'native_physics_outside_calibrated_domain' if score['invalid_physics'] else None,
                         row['id'] + '/adapter preserves oracle physics rejection')
        elif isinstance(value, (float, np.floating)):
            checks.numeric(row[key], value, row['id'] + '/independent dense v3 score/' + key, 1e-8)
        else:
            checks.equal(row[key], value, row['id'] + '/independent dense v3 score/' + key)


def control_gate(rows):
    groups = [[r for r in rows if r['role'] == role] for role in CONTROL_ROLES]
    direct, table, negative = groups
    valid = all(r.get('native_release') is True and not r.get('invalid')
        and not r.get('invalid_physics') and r.get('scalar_score_audit_passed') is True
        and r.get('contact_protocol', {}).get('substeps') == 64 for r in rows)
    good = (len(rows) == 9 and all(len(g) == 3 for g in groups) and valid
        and all(captured(r) and r.get('direct_capture') is True and r.get('table_contact_steps') == 0
                and r.get('historical_v2_score', {}).get('route_success') is True for r in direct)
        and all(captured(r) and r.get('table_assisted_capture') is True and r.get('table_contact_steps', 0) > 0
                and r.get('historical_v2_score', {}).get('route_success') is False for r in table)
        and all(not captured(r) and not r.get('whole_ball_entry')
                and r.get('historical_v2_score', {}).get('route_success') is False for r in negative))
    return dict(status='passed' if good else 'failed', counts=statistics(rows),
        direct_captures=sum(captured(r) for r in direct),
        table_assisted_captures=sum(captured(r) for r in table),
        negative_captures=sum(captured(r) for r in negative))


def audit_controls(rows, sources, checks):
    result = {}
    for phase, key in (('controls_pre', 'pre'), ('controls_post', 'post')):
        group = [r for r in rows if r['phase'] == phase]
        checks.equal(len(group), 9, phase + '/three fixed controls repeated three times')
        for index, (source, role) in enumerate(zip(sources, CONTROL_ROLES)):
            for repeat in range(3):
                row = group[index * 3 + repeat]
                label = f'{phase}-{role}-r{repeat}'
                checks.equal(row['id'], label, label + '/literal control ID')
                checks.equal((row['role'], row['method'], row['repeat']), (role, 'control', repeat), label + '/control role')
                checks.equal((row['control_source_id'], row['control_source_sha256']), (source['id'], digest(source)), label + '/full source row provenance')
                checks.equal(row['values'], source['values'], label + '/unchanged requested action')
                checks.equal(row['bucket_xy_m'], source['bucket_xy_m'], label + '/unchanged bucket')
        result[key] = control_gate(group)
        checks.equal(result[key]['status'], 'passed', phase + '/direct, table-assisted and miss controls all pass')
    return result


def audit_prediction_packet(packet, rows, contexts, record, state, checks):
    role, method, samples = (packet[k] for k in ('role', 'method', 'samples'))
    checks.require(method in METHODS, 'Known frozen method')
    checks.equal(packet['artifact'], record, 'One frozen artifact per prediction packet')
    checks.require(packet.get('selection_or_learning') is False and packet.get('one_prediction_per_layout') is True,
                   'One direct inference per position without held-out learning')
    checks.equal(len(packet['rows']), len(contexts), 'Every layout predicted exactly once')
    checks.equal(len(rows), len(contexts), 'Every prediction attempted exactly once')
    for prediction, row, context in zip(packet['rows'], rows, contexts):
        label = f'{role}-{method}-{samples:04d}-{context["id"]}'
        checks.equal(row['id'], label, label + '/literal trial identity')
        checks.equal((row['role'], row['phase'], row['method'], row['samples']), (role, role, method, samples), label + '/phase and frozen prefix')
        checks.equal(prediction['context_id'], context['id'], label + '/preselected context')
        checks.equal(prediction['bucket_xy_m'], context['bucket_xy_m'], label + '/preselected coordinate')
        for key, value in prediction.items():
            checks.equal(row.get(key), value, label + '/prediction survives observed outcome/' + key)
        checks.equal(prediction['artifact_sha256'], record['sha256'], label + '/frozen artifact binding')
        checks.equal(row['frozen_artifact_sha256'], record['sha256'], label + '/same model used at execution')
        checks.require(prediction.get('ball_flight_search_at_inference') is False
            and prediction.get('selection_or_learning') is False and prediction.get('geometry_search_at_inference') is True,
            label + '/geometry-only resolution is separate from ball-flight inference')
        checks.equal(prediction['source_training_ids'], [], label + '/no teacher lookup inserted into inference')
        latent = common.network_forward(state['policy'], prediction['observation'])
        residual = np.tanh(latent.astype(float))
        checks.numeric(prediction['latent_mean'], latent, label + '/independent actor output', 2e-6)
        checks.numeric(prediction['normalized_residual'], residual, label + '/independent residual', 2e-6)
        checks.numeric(prediction['command_values'], np.asarray(prediction['effective_source_hint']['values']) + residual * RESIDUAL,
                       label + '/actual actor request without flight search', 2e-6)
        checks.equal(row['values'], row['command_values'], label + '/no projection or replacement')


def audit_predictions(directory, rows, report, spec, milestones, checks):
    curves = []
    for samples in MILESTONES:
        member = milestones[samples]
        packet = read(directory / f'predictions/validation-ppo-{samples:04d}.json')
        cohort = [r for r in rows if r['role'] == 'validation' and r['samples'] == samples]
        checks.equal((packet['role'], packet['method'], packet['samples']), ('validation', 'ppo', samples), 'Prescheduled validation checkpoint')
        audit_prediction_packet(packet, cohort, spec['splits']['validation'], member['record'], member['state'], checks)
        curves.append(dict(samples=samples, validation=statistics(cohort), artifact=member['record']))
    checks.equal(report['curves'], curves, 'All validation outcomes remain visible')
    best = max(curves, key=lambda p: (p['validation']['captures'], p['validation']['mean_reward'], -p['samples']))
    selected = milestones[best['samples']]['record']
    checks.equal(report['selected'], selected, 'Checkpoint chosen only by validation')
    checks.equal(report['artifacts']['selected'], selected, 'Published selected model is exact chosen prefix')
    checks.equal(read(directory / 'selected.json'), dict(selected=selected, rule=spec['protocol']['selection'], validation=best['validation']),
                 'Pretest model selection receipt')
    plan = read(directory / 'test-plan.json')
    checks.equal(plan['selected'], selected, 'Whole paired TEST plan binds selected model')
    checks.require(plan['frozen_before_outcomes'] is True, 'All200 test predictions frozen before outcomes')
    checks.equal([p['method'] for p in plan['methods']], list(METHODS), 'Literal paired methods and ordering')
    comparisons, positions = [], []
    for method, packet in zip(METHODS, plan['methods']):
        if method.startswith('old_'):
            record = spec['old_artifacts'][method]
            checks.equal(report['artifacts'][method], record, method + '/same declared historical artifact')
            checks.equal(sha(contained(directory, record['path'])), record['sha256'], method + '/historical checkpoint copied byte-for-byte')
            state = common.load_checkpoint(contained(directory, record['path']), checks, method)
            checks.equal(state['weight_sha256'], record['weight_sha256'], method + '/historical exact weights')
            checks.equal(state['samples'], record['samples'], method + '/historical prefix')
        else:
            member = milestones[best['samples'] if method == 'ppo' else 0]
            record, state = member['record'], member['state']
        checks.equal((packet['role'], packet['samples']), ('test', record['samples']), method + '/prescribed frozen prefix')
        checks.equal(read(directory / f'predictions/test-{method}-{record["samples"]:04d}.json'), packet, method + '/same preoutcome packet')
        cohort = [r for r in rows if r['role'] == 'test' and r['method'] == method]
        audit_prediction_packet(packet, cohort, spec['splits']['test'], record, state, checks)
        comparisons.append(dict(method=method, counts=statistics(cohort), samples=record['samples'], artifact_sha256=record['sha256']))
        for row in cohort:
            positions.append(dict(context_id=row['context_id'], bucket_xy_m=row['bucket_xy_m'], method=method, values=row['values'],
                board_bucket_distance_m=float(np.linalg.norm(np.asarray(row['values'][:2]) - row['bucket_xy_m'])),
                capture=captured(row), clean=captured(row) and row.get('route_quality') == 2,
                direct_capture=bool(row.get('direct_capture')), table_assisted_capture=bool(row.get('table_assisted_capture')),
                route_class=row.get('route_class'), native_release=row['native_release'], invalid=row.get('invalid'), invalid_physics=row.get('invalid_physics')))
    checks.equal(report['test_comparison'], comparisons, 'Paired test metrics retain all requested positions')
    checks.equal(report['test_positions'], positions, 'Every per-position result is authentic')
    return selected, comparisons

def artifact(directory, record, contract_fp, checks, label, *, samples=None):
    path = contained(directory, record["path"])
    checks.equal(sha(path), record["sha256"], label+"/immutable checkpoint bytes")
    state = common.load_checkpoint(path, checks, label)
    checks.equal(state["contract_fingerprint"], contract_fp, label+"/exact study")
    checks.equal((state["seed"], state["method"]), (SEED, "ppo"), label+"/fixed learner seed/method")
    checks.equal(state["samples"], record["samples"], label+"/artifact training-prefix identity")
    if samples is not None:
        checks.equal(state["samples"], samples, label+"/literal training-prefix size")
    checks.equal(record["weight_sha256"], state["weight_sha256"], label+"/published exact weight identity")
    return state


def audit_training(directory, rows, report, spec, dataset, teachers, checks):
    """Audit source-only BC and all16 real PPO updates from fresh TRAIN rows."""
    import torch
    torch.set_num_threads(2)
    initial = artifact(directory, report["artifacts"]["initial"], spec["fingerprint"], checks, "Initial", samples=0)
    bc = artifact(directory, report["artifacts"]["bc"], spec["fingerprint"], checks, "TRAIN-imitation model", samples=0)
    checks.equal(report["artifacts"]["initial"]["path"], "checkpoints/initial.pt", "Fixed seeded initializer path")
    checks.equal(report["artifacts"]["bc"]["path"], "checkpoints/milestone-0000.pt", "BC is the zero-fresh-request checkpoint")
    checks.require(not initial["optimizer"]["state"] and not bc["optimizer"]["state"], "Fresh PPO Adam state after two separate imitation fits")
    fit = read(directory/"bootstrap/fit.json")
    fit_result = bootstrap_replay(initial["policy"], bc["policy"], dataset, teachers, fit, checks)
    train = [row for row in rows if row["role"] == "train"]
    checks.equal(len(train), 512, "Exactly512 fresh TRAIN requests including all rejections")
    checks.equal([row["train_index"] for row in train], list(range(512)), "Complete ordered fresh TRAIN prefix")
    schedule = calibrated.expected_schedule(SEED, 512)
    checks.equal(read(directory/"schedule.json"), schedule, "Seeded repeated balanced TRAIN-layout schedule")
    contexts = spec["splits"]["train"]
    updates = common.lines(directory/"updates.jsonl")
    checks.equal(len(updates), 16, "All16 PPO updates persisted")
    previous = bc
    milestones = {0: dict(record=report["artifacts"]["bc"], state=bc)}
    for index, update in enumerate(updates):
        offset = index*32;label = f"PPO update{index:02d}"
        cohort = train[offset:offset+32]
        checks.equal(update["index"], index, label+"/ordered update index")
        checks.equal(update["train_ids"], [row["id"] for row in cohort], label+"/only current32 TRAIN outcomes")
        before = artifact(directory, update["behavior"], spec["fingerprint"], checks, label+" behavior", samples=offset)
        after = artifact(directory, update["after_artifact"], spec["fingerprint"], checks, label+" after", samples=offset+32)
        checks.equal(update["behavior"]["path"], f"behavior/update-{index:04d}.pt", label+"/literal behavior path")
        checks.equal(update["after_artifact"]["path"], f"after/update-{index:04d}.pt", label+"/literal after path")
        checks.equal(before["weight_sha256"], previous["weight_sha256"], label+"/continuous model chain")
        checks.equal(common.tensor_object_digest(before["optimizer"]), common.tensor_object_digest(previous["optimizer"]), label+"/continuous Adam moments and steps")
        checks.equal((update["before"], update["after"]), (before["weight_sha256"], after["weight_sha256"]), label+"/actual changed-state hashes")
        checks.equal(update["buffer"], f"buffers/update-{index:04d}.npz", label+"/literal TRAIN buffer path")
        path = contained(directory, update["buffer"])
        checks.equal(sha(path), update["buffer_sha256"], label+"/exact frozen update buffer")
        with np.load(path, allow_pickle=False) as archive:
            checks.equal(set(archive.files), {"observation", "latent", "old_logp", "reward"}, label+"/no extra held-out buffer input")
            buffer = {key: archive[key].copy() for key in archive.files}
        for local, row in enumerate(cohort):
            ident = f"table-training-train-{offset+local:05d}"
            context = contexts[schedule[offset+local]]
            checks.equal(row["id"], ident, label+"/literal request ID")
            checks.equal((row["method"], row["seed"], row["phase"], row["samples"]), ("ppo", SEED, "train", offset+local+1), ident+"/fresh learner identity")
            checks.equal(row["context_id"], context["id"], ident+"/TRAIN-only scheduled context")
            checks.equal(row["bucket_xy_m"], context["bucket_xy_m"], ident+"/TRAIN-only bucket coordinate")
            checks.equal(row["source_training_ids"], [], ident+"/no label-derived teacher action injected into PPO sampling")
            checks.equal((row["behavior_checkpoint"], row["behavior_checkpoint_sha256"]),
                         (update["behavior"]["path"], update["behavior"]["sha256"]), ident+"/exact sampling actor before flight")
        for key, row_key in (("observation", "observation"), ("latent", "latent"), ("old_logp", "old_logp"), ("reward", "learning_reward")):
            expected = np.asarray([row[row_key] for row in cohort], dtype=np.float32)
            checks.numeric(buffer[key], expected, label+"/buffer contains only current TRAIN/"+key, 0)
        common.check_behavior_likelihood(before["policy"], buffer["observation"], buffer["latent"], buffer["old_logp"], checks, label)
        residual = np.tanh(np.asarray(buffer["latent"], dtype=float))
        checks.numeric([row["normalized_residual"] for row in cohort], residual, label+"/sampled bounded residual", 1e-12)
        for local, row in enumerate(cohort):
            expected = np.asarray(row["effective_source_hint"]["values"])+residual[local]*RESIDUAL
            checks.numeric(row["values"], expected, row["id"]+"/exact sampled continuous request", 1e-12)
        replayed = common.replay_ppo_update(before["policy"], before["optimizer"], after["policy"],
                    dict(obs=buffer["observation"], latent=buffer["latent"], old_logp=buffer["old_logp"], reward=buffer["reward"]), checks, label)
        checks.equal(len(update["epochs"]), 6, label+"/six actual PPO epochs")
        checks.numeric([epoch["loss"] for epoch in update["epochs"]], replayed["losses"], label+"/independent epoch losses", 2e-6)
        checks.require(after["weight_sha256"] != before["weight_sha256"], label+"/actual model update, not unchanged fake training")
        for state in after["optimizer"]["state"].values():
            checks.numeric(float(state["step"]), 6*(index+1), label+"/actual cumulative Adam step", 0)
        checks.numeric(common.numpy_state(after["policy"])["logstd"], np.clip(common.numpy_state(after["policy"])["logstd"], -3., -.7), label+"/bounded exploration scale", 0)
        previous = after
        samples = offset+32
        if samples in MILESTONES:
            relative = f"checkpoints/milestone-{samples:04d}.pt"
            state = common.load_checkpoint(directory/relative, checks, f"Milestone{samples}")
            record = dict(path=relative, sha256=sha(directory/relative), weight_sha256=state["weight_sha256"], samples=samples)
            loaded = artifact(directory, record, spec["fingerprint"], checks, f"Milestone{samples}", samples=samples)
            checks.equal(loaded["weight_sha256"], after["weight_sha256"], f"Milestone{samples}/frozen exact TRAIN prefix")
            checks.equal(common.tensor_object_digest(loaded["optimizer"]), common.tensor_object_digest(after["optimizer"]), f"Milestone{samples}/same optimizer checkpoint")
            milestones[samples] = dict(record=record, state=loaded)
    checks.equal((report["train_requests"], report["updates"]), (512, 16), "Reported actual fresh-learning budget")
    return milestones, fit_result


def authenticate_sources(source, parent, parent_audit, policy, policy_audit, rescore, checks):
    """Authenticate old proofs; inspect only TRAIN labels for fitting ancestry."""
    policy_public = read(source / 'baselines/mjlab-policy64-20261105/summary.json')
    provenance = policy_public['provenance']
    for path, expected in ((policy / 'report.json', provenance['report_sha256']),
            (policy / 'contract.json', provenance['contract_sha256']),
            (policy / 'evidence-manifest.json', provenance['evidence_manifest_sha256']),
            (policy_audit, provenance['independent_audit_sha256'])):
        checks.equal(sha(path), expected, 'Audited historical policy input/' + path.name)
    policy_spec, policy_proof = read(policy / 'contract.json'), read(policy_audit)
    checks.equal(policy_proof.get('status'), 'passed', 'Historical policy proof passed')
    checks.equal(policy_proof.get('failures'), [], 'Historical policy has no failed checks')
    checks.equal(policy_spec['fingerprint'], provenance['contract_fingerprint'], 'Known old policy contract')
    parent_spec, parent_rows = old.audit_parent(parent, parent_audit, policy_spec['source_bindings'], checks)
    policy_manifest = read(policy / 'evidence-manifest.json')
    checks.equal(sha(policy / 'rows.jsonl'), policy_manifest['rows.jsonl'], 'All old policy rows are immutable')
    policy_rows = [r for r in common.lines(policy / 'rows.jsonl') if r.get('role') == 'train']
    checks.equal(len(policy_rows), 1024, 'All1024 old policy TRAIN outcomes retained')
    checks.equal([r['train_index'] for r in policy_rows], list(range(1024)), 'Complete old policy TRAIN prefix')
    contexts = {c['id']: c['bucket_xy_m'] for c in parent_spec['splits']['train']}
    checks.equal(policy_spec['splits']['train'], [dict(id=c['id'], split='train', bucket_xy_m=c['bucket_xy_m']) for c in parent_spec['splits']['train']],
                 'Both original sources use the same35 TRAIN layouts')
    for row in policy_rows:
        checks.equal(row['bucket_xy_m'], contexts.get(row['context_id']), row['id'] + '/old TRAIN context and coordinate')
        checks.equal(row['method'], 'ppo', row['id'] + '/old TRAIN method')
        checks.equal(row['seed'], SEED, row['id'] + '/old TRAIN seed')
        checks.equal(row['phase'], 'train', row['id'] + '/old TRAIN phase')
    from .prepare_table_assisted_scoring import compact
    # This proof reader authenticates a derivative, not the producer's new
    # training-data join. The join is independently reconstructed below.
    verified = compact(rescore)
    public_rescore = read(source / 'baselines/mjlab-table-assisted-scoring-20261108/results.json')
    for key in ('summary_sha256', 'manifest_sha256', 'rule_sha256'):
        checks.equal(verified[key], public_rescore[key], 'Published completed v3 rescore/' + key)
    derivative = [r for r in common.lines(rescore / 'records.jsonl')
                  if r.get('status') in ('scored', 'rejected') and r.get('phase') == 'train']
    by_key = {}
    for row in derivative:
        key = (row['source_study'], row['id'])
        checks.require(key not in by_key, 'Unique derivative TRAIN identity')
        by_key[key] = row
    rows = [(parent.name, r) for r in parent_rows] + [(policy.name, r) for r in policy_rows]
    return parent_spec, policy_spec, rows, by_key, policy_public


def audit_bootstrap_inputs(directory, source_rows, derivative, spec, checks):
    dataset, teachers = read(directory / 'bootstrap/dataset.json'), read(directory / 'bootstrap/teachers.json')
    checks.equal(len(dataset), 7168, 'All7168 historical TRAIN attempts retained')
    checks.equal(len({r['id'] for r in dataset}), 7168, 'Unique fitted TRAIN identities')
    checks.equal([r['id'] for r in dataset], [r['id'] for _, r in source_rows], 'Exact source order and membership')
    gains = 0
    for reduced, (study_name, raw) in zip(dataset, source_rows):
        label = raw['id']
        checks.equal(reduced['role'], 'train', label + '/no held-out labels in fitting')
        checks.equal(reduced['source_record_sha256'], digest(raw), label + '/full original record hash')
        for key in ('id', 'context_id', 'bucket_xy_m', 'observation', 'normalized_residual', 'values', 'effective_source_hint', 'native_release', 'invalid_physics'):
            checks.equal(reduced[key], raw[key], label + '/actual source feature/' + key)
        checks.numeric(raw['observation'], calibrated.expected_observation(raw['bucket_xy_m'], raw['effective_source_hint']), label + '/geometry-only original observation', 0)
        residual = np.asarray(raw['normalized_residual'], dtype=float)
        checks.require(residual.shape == (5,) and np.isfinite(residual).all() and (np.abs(residual) <= 1 + 1e-8).all(), label + '/valid residual bounds')
        checks.numeric(raw['values'], np.asarray(raw['effective_source_hint']['values']) + residual * RESIDUAL, label + '/actual historical requested action', 2e-6)
        item = derivative.get((study_name, label))
        checks.require(item is not None, label + '/exact compatible rescore row')
        checks.equal(item['original']['values'], raw['values'], label + '/rescore exact action')
        checks.equal(item['original']['bucket_xy_m'], raw['bucket_xy_m'], label + '/rescore exact bucket')
        if raw.get('native_release'):
            checks.equal(item['status'], 'scored', label + '/native v3 evidence required')
            checks.equal((item['native_sha256'], item['native_row']), (raw['native_cache_sha256'], raw['native_cache_row']), label + '/same dense archive and column')
            checks.equal(raw['contact_protocol']['substeps'], 64, label + '/only64 source actions')
            scored = item['new_score']
            for key, value in scored.items():
                if key in reduced:
                    checks.equal(reduced[key], value, label + '/verified derivative v3 field/' + key)
            checks.equal(bool(reduced['route_success']), bool(scored['route_success']), label + '/new route label')
            checks.numeric(reduced['learning_reward'], oracle.reward(scored), label + '/independent new TRAIN reward', 1e-8)
            gains += int(scored['route_success'] and not raw['route_success'])
        else:
            checks.equal(item['status'], 'rejected', label + '/original failed setup remains in fitting')
            checks.require(not reduced['route_success'], label + '/no invented capture')
            checks.numeric(reduced['learning_reward'], -1., label + '/unchanged setup penalty', 0)
    checks.equal(gains, 115, 'Exactly115 newly accepted TRAIN captures; no held-out gains fitted')
    checks.equal(teachers, best_teachers(dataset), 'One independent best teacher per TRAIN context')
    checks.equal(read(directory / 'bootstrap/source-bindings.json'), spec['source_bindings'], 'Published source bindings unchanged')
    return dataset, teachers


def audit_coordinate_inventory(directory, source, history_root, parent_splits, spec, checks):
    """Re-read only the frozen inventoried files; never add this study's outputs."""
    inventory = read(directory / 'historical-coordinate-bindings.json')
    checks.equal(sha(directory / 'historical-coordinate-bindings.json'), spec['historical_coordinate_bindings_sha256'],
                 'Historical coordinate inventory frozen before new study output exists')
    coordinates = set()
    coordinate_names = {'bucket_xy_m', 'bucket_xy', 'exclusions', 'excluded', 'coordinates', 'excluded_coordinates'}
    def collect(item, enabled=False):
        if enabled and isinstance(item, (list, tuple)) and len(item) == 2:
            if all(type(v) in (int, float) and math.isfinite(v) for v in item):
                coordinates.add(tuple(map(float, item)))
                return
        if isinstance(item, dict):
            for key, value in item.items():
                collect(value, key in coordinate_names)
        elif isinstance(item, list):
            for value in item:
                collect(value, enabled)
    for name, expected in inventory['files_sha256'].items():
        prefix, relative = name.split('/', 1)
        checks.require(prefix in ('output', 'baselines'), 'Known coordinate-inventory source')
        root = history_root if prefix == 'output' else source / 'baselines'
        path = contained(root, relative)
        checks.require(not path.resolve().is_relative_to(directory.resolve()), 'Own outputs cannot leak into historical inventory')
        checks.equal(sha(path), expected, 'Frozen coordinate source unchanged/' + name)
        if path.suffix == '.jsonl':
            for item in common.lines(path):
                collect(item)
        else:
            collect(read(path), 'exclusion' in path.name)
    checks.equal(inventory['coordinates'], [list(x) for x in sorted(coordinates)], 'All inventoried coordinates independent of outcome or IK')
    coordinates.update(tuple(c['bucket_xy_m']) for c in parent_splits['train'])
    excluded = [list(x) for x in sorted(coordinates)]
    checks.equal(read(directory / 'exclusions.json'), excluded, 'All historical coordinates excluded')
    checks.equal(spec['exclusions'], excluded, 'Contract binds same unfiltered exclusion set')
    expected = regenerated_splits(parent_splits, excluded)
    checks.equal(read(directory / 'splits.json'), expected, 'Seeded fresh layouts exactly reproduced')
    checks.equal(spec['splits'], expected, 'Predeclared disjoint35/12/50 layout contract')
    return dict(files=len(inventory['files_sha256']), historical_coordinates=len(excluded), novel_validation=12, novel_test=50)


def audit(study, source, assets, *, parent_study, parent_audit, policy_study,
          policy_audit, rescore_dir, reference, history_root=None):
    started, checks = time.monotonic(), AuditChecks()
    study, source, assets, parent_study, parent_audit, policy_study, policy_audit, rescore_dir, reference = map(Path,
        (study, source, assets, parent_study, parent_audit, policy_study, policy_audit, rescore_dir, reference))
    history_root = Path(history_root) if history_root else parent_study.parent
    manifest = replay.check_manifest(study, checks)
    spec, report = read(study / 'contract.json'), read(study / 'report.json')
    unsigned = dict(spec); fingerprint = unsigned.pop('fingerprint', None)
    checks.equal(digest(unsigned), fingerprint, 'Complete opt-in v3 experiment fingerprint')
    checks.equal((spec['version'], report['version']), (STUDY_VERSION, STUDY_VERSION), 'Explicit new study version')
    checks.equal((report['status'], report['phase']), ('completed', 'completed'), 'Finite requested study completed')
    checks.equal(report['contract_fingerprint'], fingerprint, 'Results bind actual new contract')
    checks.equal(spec['evidence_kind'], 'native_fixed64_table_assisted_retraining', 'Actual native retraining evidence')
    checks.equal(report['evidence_sha256'], manifest, 'Report binds all original study artifacts')
    checks.equal(report['protocol'], spec['protocol'], 'Frozen protocol unchanged')
    protocol_checks(spec['protocol'], checks)
    replay.check_source_files(source, spec['source_sha256'], checks, 'Actual producer source snapshot')
    own = 'sircus/mjlab/table_training_verify.py'
    checks.equal(sha(Path(__file__)), spec['source_sha256'][own], 'Independent checker same frozen source version')
    checks.equal(sha(Path(oracle.__file__)), spec['source_sha256']['sircus/mjlab/table_assisted_score_oracle.py'], 'Actual independent v3 oracle source')
    parent, policy_parent, raw_sources, derived, policy_public = authenticate_sources(source, parent_study,
        parent_audit, policy_study, policy_audit, rescore_dir, checks)
    old.audit_runtime_reference(source, reference, spec, parent, assets, checks)
    dataset, teachers = audit_bootstrap_inputs(study, raw_sources, derived, spec, checks)
    summary = spec['bootstrap_summary']
    summary_fields = dict(requests=7168, native_releases=sum(bool(r['native_release']) for r in dataset),
        captures=sum(captured(r) for r in dataset), clean=sum(captured(r) and r['route_quality'] == 2 for r in dataset),
        layouts=35, successful_layouts=len(teachers), teacher_count=len(teachers), newly_credited_training_count=115,
        table_teacher_count=sum(bool(r.get('table_assisted_capture')) for r in teachers),
        table_assisted_teachers=sum(bool(r.get('table_assisted_capture')) for r in teachers), newly_credited_captures=115,
        newly_credited_teacher_count=sum(bool(r.get('newly_credited_capture')) for r in teachers))
    old_map = {r['context_id']: r['id'] for r in old.best_teachers([r for _, r in raw_sources[:6144]])}
    changes = [dict(context_id=r['context_id'], old_id=old_map.get(r['context_id']), new_id=r['id'],
                    table_assisted=bool(r.get('table_assisted_capture'))) for r in teachers if old_map.get(r['context_id']) != r['id']]
    summary_fields.update(teacher_changes=changes, teacher_change_count=len(changes), changed_teacher_count=len(changes), original_teacher_count=len(old_map))
    for key, value in summary_fields.items():
        checks.equal(summary[key], value, 'Actual fitting-data change/' + key)
    checks.equal(report['bootstrap_summary'], summary, 'Reported actor/critic data counts are separate')
    for name, key in (('bootstrap/dataset.json', 'bootstrap_dataset_sha256'), ('bootstrap/teachers.json', 'teachers_sha256'), ('controls.json', 'controls_sha256')):
        checks.equal(sha(study / name), spec[key], 'Pretraining input file binding/' + name)
    controls = [min((r for r in dataset if captured(r) and r.get('direct_capture')), key=lambda r: (-int(r['route_quality'] == 2), -r['learning_reward'], r['id'])),
        min((r for r in dataset if captured(r) and r.get('table_assisted_capture')), key=lambda r: r['id']),
        min((r for r in dataset if r.get('native_release') and not captured(r) and not r.get('invalid') and not r.get('invalid_physics') and not r.get('whole_ball_entry')), key=lambda r: r['id'])]
    checks.equal(read(study / 'controls.json'), controls, 'Fixed controls chosen strictly from historical TRAIN')
    for method, name in (('old_bc', 'bc'), ('old_ppo', 'selected')):
        record = spec['old_artifacts'][method]
        original = read(policy_study / 'report.json')['artifacts'][name]
        checks.equal(record['path'], 'checkpoints/' + method + '.pt', method + '/local frozen copy path')
        checks.equal(record['historical_relative_path'], original['path'], method + '/actual old artifact path')
        for key in ('sha256', 'weight_sha256', 'samples'):
            checks.equal(record[key], original[key], method + '/old artifact identity/' + key)
        checks.equal(sha(contained(policy_study, original['path'])), record['sha256'], method + '/original checkpoint unchanged')
        checks.equal(record['sha256'], policy_public['checkpoints'][name]['sha256'], method + '/published historical model')
    checks.equal(read(study / 'old-artifacts.json'), spec['old_artifacts'], 'Old models explicitly bound to new test')
    inventory = audit_coordinate_inventory(study, source, history_root, parent['splits'], spec, checks)
    prior = read(study / 'prior.json')
    checks.equal(prior, read(parent_study / 'prior.json'), 'Original geometry-only prior unchanged')
    checks.equal(digest(prior), spec['prior_sha256'], 'Same fixed geometric source identity')
    calibration = replay.check_calibration(study, spec['identity']['calibration'], checks)
    home = spec['identity']['setup']['home']
    for name in ('calibration.json', 'verification.json', 'setup-home.json'):
        checks.equal(sha(study / name), sha(parent_study / name), 'Unchanged calibration/home bytes/' + name)
    rows = replay.read_records(study / 'rows.jsonl')
    check_rows(rows, checks)
    setup = calibrated.check_setup_evidence(study, rows, prior, home, checks)
    old.audit_native_options(study / 'native', spec['scene'], calibration, assets, checks)
    for row in rows:
        if row.get('native_release'):
            checks.equal((row['effective_options_file'], row['effective_options_sha256']),
                         ('effective-options.json', sha(study / 'native/effective-options.json')), row['id'] + '/allocated64 options binding')
    traces = rescore_rows(study / 'native', rows, spec['scene'], calibration, spec['identity']['trace_schema'], assets, checks)
    milestones, fit = audit_training(study, rows, report, spec, dataset, teachers, checks)
    selected, comparisons = audit_predictions(study, rows, report, spec, milestones, checks)
    audit_sequence(rows, selected['samples'], {m: spec['old_artifacts'][m]['samples'] for m in ('old_bc', 'old_ppo')}, checks)
    controls_result = audit_controls(rows, controls, checks)
    checks.equal(report['control_gates'], controls_result, 'Actual before/after control results')
    checks.equal(report['counts'], statistics(rows), 'All requested attempts and failures in headline metrics')
    audit_trace_examples(study, report, rows, checks)
    checks.equal(report['gate'], dict(status='passed', controls_passed=True, complete=True, substeps=64,
        scope='Protocol/controls completion, not a128convergence or minimum success-rate gate'), 'Completion is not convergence or policy qualification')
    return dict(version=VERSION, status='passed', full_checks_verified=True, checks_count=checks.count, checks=checks.entries, failures=[],
        contract_fingerprint=fingerprint, report_sha256=sha(study / 'report.json'),
        evidence_manifest_sha256=sha(study / 'evidence-manifest.json'), auditor_sha256=sha(Path(__file__)),
        producer_source_sha256=spec['source_sha256'], requested_trials=len(rows), native_releases=len(traces),
        bootstrap_summary=summary, selected=selected, test_comparison=comparisons, control_gates=controls_result,
        setup_evidence=setup, coordinate_inventory=inventory, bootstrap_replay=fit,
        elapsed_s=time.monotonic() - started, maximum_numeric_error=checks.numeric_error,
        limitations=['Fixed64 held-endpoint simulator evidence; no driven transit, grasping, perception, noise or physical validation.',
                     'One prediction per method at50 fresh positions; this is not a repeated-reliability estimate.',
                     'Data/reward adaptation ablation with unequal historical exposure, not equal-budget algorithm superiority.',
                     'Finite IK failure declines one requested recipe, not every possible recipe at that bucket.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('study', 'source', 'assets', 'parent-study', 'parent-audit', 'policy-study', 'policy-audit', 'rescore-dir', 'reference', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--history-root', type=Path)
    a = parser.parse_args()
    if a.output.exists() or a.output.resolve().is_relative_to(a.study.resolve()):
        raise ValueError('New audit output must be outside immutable original study')
    try:
        result = audit(a.study, a.source, a.assets, parent_study=a.parent_study, parent_audit=a.parent_audit,
            policy_study=a.policy_study, policy_audit=a.policy_audit, rescore_dir=a.rescore_dir,
            reference=a.reference, history_root=a.history_root)
    except Exception as error:
        result = dict(version=VERSION, status='failed', failures=[type(error).__name__ + ': ' + str(error)])
        a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        raise
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: result[k] for k in ('status', 'checks_count', 'requested_trials', 'native_releases', 'selected')}))


def audit_trace_examples(directory, report, rows, checks):
    expected = []
    for method in METHODS:
        for success in (True, False):
            cohort = [r for r in rows if r['role'] == 'test' and r['method'] == method and r.get('native_release') and captured(r) == success]
            if cohort:
                expected.append(cohort[0])
    checks.equal([r['id'] for r in report['trace_examples']], [r['id'] for r in expected], 'Displayed examples use literal first capture/miss per method')
    for trace, row in zip(report['trace_examples'], expected):
        for key in ('values', 'bucket_xy_m', 'board_center_m', 'board_normal', 'trajectory_m', 'trajectory_dt_s',
                    'whole_ball_entry', 'max_retention_s', 'invalid', 'invalid_physics', 'historical_v2_score',
                    'native_cache_sha256', 'native_cache_file', 'native_cache_row', 'route_class'):
            checks.equal(trace[key], row[key], row['id'] + '/actual UI trace field/' + key)
        checks.equal(trace['capture'], captured(row), row['id'] + '/actual UI capture')
        checks.equal(trace['table_assisted_capture'], bool(row.get('table_assisted_capture')), row['id'] + '/actual UI route')
        with np.load(contained(directory / 'native', row['native_cache_file']), allow_pickle=False) as archive:
            dense = archive['position_m'][256::256, row['native_cache_row']]
        checks.numeric(trace['trajectory_m'], dense, row['id'] + '/UI trajectory is actual dense data', 0)
        checks.numeric(trace['trajectory_dt_s'], 1 / 60, row['id'] + '/UI trace cadence', 0)



if __name__ == '__main__':
    main()
