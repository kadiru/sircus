"""Synthetic guard readbacks only: no simulator, optimizer or real shots."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import candidate_continuation_native_guards as subject
from continuous_feasibility import canonical_hash, decode_action
from continuous_native_audit import expected_empty
from expanded_pose_menu import endpoint_targets
from joint_feasibility import preflight
from prepare_assets import chain


def synthetic_guard_packet():
    """Return independently reconstructible, all-declined 35-slot mock data."""
    source = Path(__file__).resolve().parents[1]
    scene = json.loads((source / 'scene.json').read_text())
    cells = [dict(id=f'train-{i:03d}', split='train', bucket_xy_m=[.27+i*.001, -.1]) for i in range(35)]
    actions = [decode_action(c, [0., .4, .35, .24, .5], 'synthetic-'+c['id']) for c in cells]
    xml = '<robot>' + ''.join(
        f'<joint name="joint{i}" type="revolute"><origin xyz="0 0 {0.3 if i == 1 else 0}" rpy="0 0 0"/>'
        '<axis xyz="0 0 1"/><limit lower="-6.28" upper="6.28"/></joint>' for i in range(1, 7)) + '</robot>'
    links = {n: chain(ET.fromstring(xml)) for n in subject.ARMS}
    starts = {n: [0.]*6 for n in subject.ARMS}
    arrays = {n: np.zeros((35, 6)).tolist() for n in subject.ARMS}
    state = {n: {k: copy.deepcopy(arrays[n]) for k in ('position_rad', 'velocity_rad_s', 'target_rad')}
             for n in subject.ARMS}
    gains = {n: {k: np.ones((35, 6)).tolist() for k in ('effort', 'stiffness', 'damping')} for n in subject.ARMS}
    transit = [dict(phase='start', step=0, state=copy.deepcopy(state), original_gains=copy.deepcopy(gains)),
               dict(phase='tail', step=60, state=copy.deepcopy(state)),
               dict(phase='restored_original_hold', step=300, state=copy.deepcopy(state), restored_gains=copy.deepcopy(gains))]
    histories, rows, records = [], [], []
    for cell, action in zip(cells, actions):
        targets = endpoint_targets(scene, cell, action)
        center, normal = targets['board_world'][:3, 3].tolist(), targets['board_normal_world'].tolist()
        drop = targets['Arm10'][:3, 3].tolist()
        drift = float(np.rad2deg(np.arccos(np.clip(np.dot(normal, normal), -1, 1))))
        history = [dict(iteration=0, measurement_start_step=300+j*60, physics_step=360+j*60,
            board_center_m=center, board_normal=normal, dropper_flange_m=drop,
            previous_board_center_m=center, previous_board_normal=normal, previous_dropper_flange_m=drop,
            position_error_mm=0., normal_error_deg=drift, dropper_position_error_mm=0.,
            position_drift_mm=0., normal_drift_deg=drift, dropper_drift_mm=0., joint_speed_rad_s=0.,
            arms={n: dict(position_rad=[0.]*6, velocity_rad_s=[0.]*6, target_rad=[0.]*6) for n in subject.ARMS},
            articulation_joint_names=dict(Arm10=[f'joint{k}' for k in range(1, 7)]+['finger_left_joint', 'finger_right_joint'],
                                          Arm14=[f'joint{k}' for k in range(1, 7)]),
            all_articulation_joint_velocity=dict(Arm10=[0.]*8, Arm14=[0.]*6)) for j in range(7)]
        record = preflight(scene, cell, action, links, starts)
        assert record['reason'] == 'outer_link_length_bound'
        histories.append(history); records.append(record)
        rows.append(dict(action, **expected_empty(scene, cell, 1/240), cell_id=cell['id'], split='train',
            invalid=record['reason'], qualification=record['qualification'], path_proxy=record['path'],
            path_rejection=None, settled=history[-1], settled_board_center_m=center,
            settled_board_normal=normal, scoring_board_normal=normal,
            release_ball_center_m=targets['release_center_m'].tolist()))
    packet = dict(synthetic=True, cells=cells, actions=actions, rows=rows, enabled=[True]*35,
        proposal_ik_seeds=[None]*35, preflight_records=records,
        native_binding=dict(canonical_home_rad=starts),
        home=dict(protocol='canonical-episode-home-reset-before-visible-drive-v1', episode_reset=True,
            path_eligible=[True]*35, path_checks=[dict(accepted=True) for _ in range(35)], physics_step=0,
            target_joint_rad=copy.deepcopy(arrays), measured_joint_rad=copy.deepcopy(arrays),
            measured_joint_velocity_rad_s=copy.deepcopy(arrays)),
        release_step=720, score_end_step=720, setup_controller=subject.protocol(),
        settling_history=histories, transit_history=transit, correction_history=[])
    return packet, scene, cells, links, [None]*35


class GuardAuditTests(unittest.TestCase):
    def setUp(self):
        self.packet, self.scene, self.cells, self.links, self.seeds = synthetic_guard_packet()

    def verify(self):
        return subject.verify_guards(self.packet, self.scene, self.cells, self.links, self.seeds)

    def test_synthetic_all_rejected_evidence_is_independently_verified_without_solver(self):
        before = copy.deepcopy(self.packet)
        with patch('robust_ik.solve', side_effect=AssertionError('No optimization allowed')):
            result = self.verify()
        self.assertEqual(result['initial_geometry']['initial_geometry_path_declined'], 35)
        self.assertEqual(result['settlement']['measured_windows'], 245)
        self.assertTrue(result['settlement']['full_articulation_velocity_gate_verified'])
        self.assertEqual(self.packet, before)

    def test_guard_and_score_auditors_compose_on_same_synthetic_packet(self):
        from candidate_continuation_native_score import PACKET_SCHEMA, verify_score
        self.packet.update(schema=PACKET_SCHEMA, mode='train', request_ids=[f'train-{i:06d}' for i in range(35)],
            choices=[dict(request_id=f'train-{i:06d}', context_id=self.cells[i]['id'], action=self.packet['actions'][i])
                     for i in range(35)], rewards=[-1.]*35, reward_features=[None]*35,
            origins_m=np.zeros((35, 3)).tolist(), physics=None)
        before = copy.deepcopy(self.packet)
        with tempfile.TemporaryDirectory() as directory, patch('robust_ik.solve',
                side_effect=AssertionError('No optimization allowed')):
            guards = self.verify()
            scores = verify_score(directory, self.packet, self.scene)
        self.assertEqual(scores['requests'], 35)
        self.assertEqual(scores['rejections'], 35)
        self.assertEqual(scores['releases'], 0)
        self.assertEqual(scores['captures'], 0)
        self.assertEqual(scores['physical']['native_steps'], 0)
        self.assertTrue(guards['rejections']['first_cause_rejections_reconstructed'])
        self.assertEqual(self.packet, before)

    def test_shortened_or_nonboolean_slot_coverage_fails(self):
        for key, change in [('rows', lambda x: x.pop()), ('enabled', lambda x: x.__setitem__(0, 1)),
                            ('proposal_ik_seeds', lambda x: x.pop())]:
            with self.subTest(key=key):
                original = copy.deepcopy(self.packet[key]); change(self.packet[key])
                with self.assertRaises(ValueError): self.verify()
                self.packet[key] = original

    def test_canonical_home_and_transit_start_are_bound(self):
        mutations = [lambda p: p['home']['target_joint_rad']['Arm10'][0].__setitem__(0, .1),
                     lambda p: p['transit_history'][0]['state']['Arm10']['position_rad'][0].__setitem__(0, .1),
                     lambda p: p['home'].__setitem__('episode_reset', False)]
        for mutate in mutations:
            original = copy.deepcopy(self.packet); mutate(self.packet)
            with self.assertRaises(ValueError): self.verify()
            self.packet = original

    def test_preflight_action_start_targets_and_checksum_tamper_fail(self):
        for key in ('action', 'measured_starts_rad', 'targets', 'fingerprint'):
            with self.subTest(key=key):
                original = copy.deepcopy(self.packet)
                record = self.packet['preflight_records'][0]
                if key == 'action': record['action']['gap_m'] += .01
                elif key == 'measured_starts_rad': record['binding'][key]['Arm10'][0] += .01
                elif key == 'targets': record[key]['Arm10'][0][3] += .01
                else: record[key] = '0'*64
                if key != 'fingerprint':
                    unsigned = copy.deepcopy(record); unsigned.pop('fingerprint')
                    record['fingerprint'] = canonical_hash(unsigned)
                with self.assertRaises(ValueError): self.verify()
                self.packet = original

    def test_finger_velocity_cannot_hide_behind_six_quiet_rotary_axes(self):
        self.packet['settling_history'][0][-1]['all_articulation_joint_velocity']['Arm10'][-1] = .006
        with self.assertRaisesRegex(ValueError, 'joint-speed gate'): self.verify()

    def test_articulation_joint_inventory_and_fk_velocity_association_are_checked(self):
        for mutate in (lambda m: m['articulation_joint_names']['Arm10'].pop(),
                       lambda m: m['all_articulation_joint_velocity']['Arm14'].__setitem__(0, .001)):
            original = copy.deepcopy(self.packet); mutate(self.packet['settling_history'][0][0])
            with self.assertRaises(ValueError): self.verify()
            self.packet = original

    def test_gain_restore_clock_drift_and_release_gate_tamper_fail(self):
        mutations = [lambda p: p['transit_history'][-1]['restored_gains']['Arm14']['effort'][0].__setitem__(0, 2.),
                     lambda p: p['settling_history'][1][2].__setitem__('physics_step', 1),
                     lambda p: p['settling_history'][0][0].__setitem__('position_drift_mm', 1.),
                     lambda p: p.__setitem__('release_step', 719)]
        for mutate in mutations:
            original = copy.deepcopy(self.packet); mutate(self.packet)
            with self.assertRaises(ValueError): self.verify()
            self.packet = original

    def test_unqualified_hidden_target_change_is_rejected(self):
        self.packet['settling_history'][0][-1]['arms']['Arm10']['target_rad'][0] = .001
        with self.assertRaisesRegex(ValueError, 'without a qualified correction'): self.verify()

    def test_correction_on_ineligible_cell_is_rejected(self):
        self.packet['correction_history'] = [dict(cell_id=self.cells[0]['id'], iteration=0,
            measured=self.packet['settling_history'][0][-1], arms=[], fits={}, invalid=None)]
        with self.assertRaises(ValueError): self.verify()

    def correction_fixture(self, *, failed_ik=False):
        packet = copy.deepcopy(self.packet)
        row = packet['rows'][0]
        row['qualification'] = dict(accepted=True, ik={n: dict(q=[0.]*6) for n in subject.ARMS})
        row['path_proxy'] = dict(accepted=True)
        first = copy.deepcopy(packet['settling_history'][0][-1])
        first['dropper_flange_m'][0] -= .003
        second = copy.deepcopy(first); second['iteration'] = 1
        packet['settling_history'][0] = [first, second]
        target = endpoint_targets(self.scene, self.cells[0], packet['actions'][0])['Arm10'].copy()
        target[0, 3] += .003
        origin = target.copy()
        if failed_ik: origin[0, 3] += .001
        links = copy.deepcopy(self.links)
        links['Arm10'] = [(origin, np.array([0., 0., 1.]), -6.28, 6.28)] + [
            (np.eye(4), np.array([0., 0., 1.]), -6.28, 6.28) for _ in range(5)]
        fit = dict(q=[0.]*6, residual=.001 if failed_ik else 0.,
            position_error_mm=1. if failed_ik else 0., orientation_error_deg=0.,
            solved=not failed_ik, attempts=[])
        result = 'settling_correction_ik' if failed_ik else (
            None if subject.approximate_path_check(self.scene, links,
                {n: [0.]*6 for n in subject.ARMS}, {n: [0.]*6 for n in subject.ARMS},
                cell=self.cells[0], samples=16)['accepted'] else 'settling_correction_path_proxy')
        packet['rows'][0]['invalid'] = result
        packet['correction_history'] = [dict(cell_id=self.cells[0]['id'], iteration=0,
            measured=first, arms=['Arm10'], fits=dict(Arm10=fit), invalid=result)]
        return packet, links

    def test_saved_correction_fk_and_sampled_path_recomputed_without_ik(self):
        packet, links = self.correction_fixture()
        with patch('robust_ik.solve', side_effect=AssertionError('No optimization allowed')):
            receipt = subject._verify_corrections(packet, self.scene, self.cells, links)
        self.assertEqual(receipt['saved_ik_fits_recomputed'], 1)
        self.assertEqual(receipt['sampled_paths_recomputed'], 1)
        packet['correction_history'][0]['fits']['Arm10']['position_error_mm'] = 3.
        with self.assertRaisesRegex(ValueError, 'independent FK'):
            subject._verify_corrections(packet, self.scene, self.cells, links)

    def test_failed_correction_ik_short_circuits_path_and_cannot_be_admitted(self):
        packet, links = self.correction_fixture(failed_ik=True)
        with patch.object(subject, 'approximate_path_check', side_effect=AssertionError('IK failure must short circuit')):
            receipt = subject._verify_corrections(packet, self.scene, self.cells, links)
        self.assertEqual(receipt['sampled_paths_recomputed'], 0)
        packet['correction_history'][0]['invalid'] = None
        with self.assertRaisesRegex(ValueError, 'Failed correction IK was admitted'):
            subject._verify_corrections(packet, self.scene, self.cells, links)

    def test_rejection_label_cannot_be_changed_after_valid_geometry_reconstruction(self):
        self.packet['rows'][0]['invalid'] = 'fabricated_guard_reason'
        with self.assertRaisesRegex(ValueError, 'first cause changed'): self.verify()

    def test_partial_cohort_keeps_inactive_guards_without_counting_requests(self):
        for i in (33, 34):
            self.packet['enabled'][i] = False
            self.packet['preflight_records'][i] = None
            self.packet['rows'][i].update(invalid='inactive_split',
                qualification=dict(accepted=False, reason='inactive_split'), path_proxy=None)
        receipt = self.verify()
        self.assertEqual(receipt['initial_geometry']['initial_geometry_path_declined'], 33)
        self.assertEqual(receipt['settlement']['measured_windows'], 245)

    def test_inactive_slot_must_not_fabricate_preflight(self):
        self.packet['enabled'][0] = False
        with self.assertRaisesRegex(ValueError, 'Inactive slot'): self.verify()


if __name__ == '__main__':
    unittest.main()
