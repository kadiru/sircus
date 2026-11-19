"""Adversarial CPU checks for the v3 retraining auditor."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from sircus.mjlab import table_training_verify as audit
from sircus.mjlab import table_training_contract as contract
from sircus.mjlab.benchmark_verify import Checks, EvidenceError


def row(ident='r', *, capture=True, table=False, forbidden=0):
    result = dict(id=ident, role='train', context_id='train-a', native_release=True,
        invalid=None, invalid_physics=False, route_success=capture, route_quality=2 if capture else 0,
        board_bounce=True, whole_ball_entry=capture, max_retention_s=1. if capture else 0.,
        closest_opening_distance_m=.02, single_board_bounce_success=capture,
        forbidden_contact_steps=forbidden, table_contact_steps=3 if table else 0,
        direct_capture=capture and not table, table_assisted_capture=capture and table,
        scoring_version=audit.oracle.VERSION, route_version=audit.oracle.ROUTE_VERSION,
        reward_version=audit.oracle.REWARD_VERSION, scalar_score_audit_passed=True,
        historical_v2_score=dict(route_success=capture and not table), contact_protocol=dict(substeps=64))
    result['learning_reward'] = audit.oracle.reward(result)
    return result


class AuditTests(unittest.TestCase):
    def test_literal_protocol_detects_wrong_scoring_budget_and_methods(self):
        audit.protocol_checks(contract.protocol(), Checks())
        for key, value in (('substeps', 128), ('train_requests', 1024), ('maximum_fresh_requests', 765),
                           ('noise', True), ('test_methods', ['bc', 'ppo'])):
            changed = contract.protocol(); changed[key] = value
            with self.assertRaises(EvidenceError):
                audit.protocol_checks(changed, Checks())

    def test_counts_preserve_rejections_and_separate_table_routes(self):
        data = [row('direct'), row('table', table=True), row('miss', capture=False, forbidden=2),
                dict(id='rejected', native_release=False, invalid='ik_unresolved', route_success=False)]
        self.assertEqual(audit.statistics(data), contract.statistics(data))
        counts = audit.statistics(data)
        self.assertEqual((counts['attempts'], counts['captures'], counts['setup_rejections'], counts['native_misses']), (4, 2, 1, 1))
        self.assertEqual((counts['direct_captures'], counts['table_assisted_captures'], counts['forbidden_contact_shots']), (1, 1, 1))

    def test_teacher_rank_uses_new_reward_and_rejects_heldout(self):
        direct, table = row('z-direct'), row('a-table', table=True)
        self.assertEqual(audit.best_teachers([direct, table]), contract.select_teachers([direct, table]))
        self.assertEqual(audit.best_teachers([direct, table])[0]['id'], 'a-table')
        table['role'] = 'test'
        with self.assertRaisesRegex(EvidenceError, 'Held-out'):
            audit.best_teachers([table])

    def test_native_physics_rejection_normalization_never_clears_failure(self):
        score = dict(invalid='uncalibrated_contact_dynamics', invalid_physics=True, route_success=False, reward=-1.)
        native = dict(score, id='invalid', invalid='native_physics_outside_calibrated_domain')
        audit.check_native_score(native, score, Checks())
        native['invalid'] = None
        with self.assertRaisesRegex(EvidenceError, 'preserves'):
            audit.check_native_score(native, score, Checks())
        score = dict(invalid=None, invalid_physics=False, route_success=True, reward=4.5)
        audit.check_native_score(dict(score, id='valid'), score, Checks())

    def test_split_seed_and_full_exclusion_regeneration(self):
        parent = {'train': [dict(id=f'train-{i}', bucket_xy_m=[.2 + i * .01, -.4]) for i in range(35)]}
        exclusions = [r['bucket_xy_m'] for r in parent['train']] + [[.7, -.5]]
        expected, frozen = contract.make_splits(parent, exclusions)
        self.assertEqual(audit.regenerated_splits(parent, frozen), expected)
        new = np.array([r['bucket_xy_m'] for role in ('validation', 'test') for r in expected[role]])
        self.assertTrue((np.linalg.norm(new[:, None] - np.array(frozen)[None], axis=-1) >= .005).all())

    def test_frozen_inventory_preserves_root_exclusion_only_coordinates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); history = root/'output'; study = root/'new'; source = root/'source'
            history.mkdir(); study.mkdir(); (source/'baselines').mkdir(parents=True)
            parent = {'train': [dict(id=f'train-{i}', bucket_xy_m=[.2+i*.01,-.4]) for i in range(35)]}
            historical = [[.987,-.333]]
            (history/'exclusions.json').write_text(json.dumps(historical))
            inventory = contract.collect_exclusions(history, source)
            splits, excluded = contract.make_splits(parent, inventory['coordinates'])
            for name, value in [('historical-coordinate-bindings.json', inventory), ('splits.json', splits), ('exclusions.json', excluded)]:
                (study/name).write_text(json.dumps(value))
            spec = dict(historical_coordinate_bindings_sha256=audit.sha(study/'historical-coordinate-bindings.json'),
                        splits=splits, exclusions=excluded)
            actual = audit.audit_coordinate_inventory(study, source, history, parent, spec, Checks())
            self.assertEqual(actual['historical_coordinates'], 36)
            (history/'exclusions.json').write_text('[]')
            with self.assertRaisesRegex(EvidenceError, 'unchanged'):
                audit.audit_coordinate_inventory(study, source, history, parent, spec, Checks())

    def test_control_gate_distinguishes_real_table_capture_and_miss(self):
        rows = []
        for role, capture, table in [('direct_control', True, False), ('table_control', True, True), ('negative_control', False, False)]:
            for i in range(3):
                r = row(f'{role}-{i}', capture=capture, table=table); r['role'] = role; rows.append(r)
        self.assertEqual(audit.control_gate(rows)['status'], 'passed')
        changed = copy.deepcopy(rows); changed[3]['historical_v2_score']['route_success'] = True
        self.assertEqual(audit.control_gate(changed)['status'], 'failed')
        changed = copy.deepcopy(rows); changed[-1]['native_release'] = False
        self.assertEqual(audit.control_gate(changed)['status'], 'failed')

    def test_sequence_counts_all512_train_and_every_frozen_test(self):
        rows = []
        def add(phase, role, count, **extra):
            for _ in range(count): rows.append(dict(id=str(len(rows)), phase=phase, role=role, **extra))
        for role in audit.CONTROL_ROLES: add('controls_pre', role, 3)
        add('validation', 'validation', 12)
        for offset in range(0, 512, 32):
            add('train', 'train', 32)
            if offset + 32 in (256, 512): add('validation', 'validation', 12)
        for method, samples in [('old_bc', 0), ('old_ppo', 1024), ('bc', 0), ('ppo', 256)]:
            add('test', 'test', 50, method=method, samples=samples)
        for role in audit.CONTROL_ROLES: add('controls_post', role, 3)
        audit.audit_sequence(rows, 256, {'old_bc': 0, 'old_ppo': 1024}, Checks())
        with self.assertRaises(EvidenceError):
            audit.audit_sequence(rows[:-1], 256, {'old_bc': 0, 'old_ppo': 1024}, Checks())
        changed = copy.deepcopy(rows); changed[25]['role'] = 'test'
        with self.assertRaises(EvidenceError):
            audit.audit_sequence(changed, 256, {'old_bc': 0, 'old_ppo': 1024}, Checks())

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'CPU Torch required')
    def test_frozen_inference_reconstructs_all_four_method_outputs(self):
        from sircus.mjlab.policy64_model import initialized
        state = {'policy': initialized(audit.SEED).state_dict()}
        context = dict(id='fresh-1', bucket_xy_m=[.6, -.2])
        anchor = dict(values=[.2, -.3, .3, 28., .35])
        obs = [.1] * 9
        mean = audit.common.network_forward(state['policy'], obs)
        for method in audit.METHODS:
            artifact = dict(path='model.pt', sha256='a' * 64, samples=0)
            pred = dict(context_id=context['id'], bucket_xy_m=context['bucket_xy_m'], method=method, samples=0,
                artifact_sha256=artifact['sha256'], source_training_ids=[], observation=obs,
                latent_mean=mean.tolist(), normalized_residual=np.tanh(mean.astype(float)).tolist(),
                command_values=anchor['values'], effective_source_hint=anchor, ball_flight_search_at_inference=False,
                geometry_search_at_inference=True, selection_or_learning=False)
            packet = dict(role='test', method=method, samples=0, artifact=artifact, rows=[pred],
                          selection_or_learning=False, one_prediction_per_layout=True)
            actual = dict(pred, id=f'test-{method}-0000-fresh-1', role='test', phase='test',
                          frozen_artifact_sha256=artifact['sha256'], values=anchor['values'])
            audit.audit_prediction_packet(packet, [actual], [context], artifact, state, Checks())
            bad = copy.deepcopy(packet); bad['rows'][0]['latent_mean'][0] += .1
            bad_row = dict(actual, latent_mean=bad['rows'][0]['latent_mean'])
            with self.assertRaisesRegex(EvidenceError, 'actor output'):
                audit.audit_prediction_packet(bad, [bad_row], [context], artifact, state, Checks())


if __name__ == '__main__':
    unittest.main()
