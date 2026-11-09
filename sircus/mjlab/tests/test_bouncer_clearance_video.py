import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from sircus.mjlab import bouncer_clearance_video as V


RECORDS = [("bouncer-clearance-baseline-case-00-r0", "Original 00"),
           ("bouncer-clearance-selected-case-00-r0", "Selected 00"),
           ("bouncer-clearance-selected-case-18-r0", "Selected 18"),
           ("bouncer-clearance-controls-pre-table-r0", "Reference control")]

def write(path, value):
    path.write_text(json.dumps(value))


class TimelineTests(unittest.TestCase):
    def bundles(self):
        times = np.arange(61441) / 15360
        result = []
        for i, (identifier, label) in enumerate(RECORDS):
            terminal = [17679, 17171, 58700, 32012][i]
            events = {'board': 3300 / 15360}
            if i < 2:
                events['robot'] = 7191 / 15360
            if i == 3:
                events.update(whole_ball_entry=16652 / 15360, capture=32012 / 15360)
            row = dict(id=identifier, context_id=str(i), route_success=i == 3, whole_ball_entry=i == 3,
                       values=[.3, -.4, .46, 28., .25], bucket_xy_m=[.8, -.4], native_cache_row=i,
                       native_initial_q10=[0.] * 6, native_initial_q14=[0.] * 6)
            result.append(dict(row=row, time_s=times, label=label, events=events,
                               scored_until_s=float(times[terminal]), record_sha256='a' * 64, cache_sha256='b' * 64))
        return result

    def test_timing_exact_contact_freezes_and_terminal_are_recorded(self):
        bundles = self.bundles()
        frames, clips = V.frame_plan(bundles)
        self.assertTrue(0 < len(frames) / V.FPS <= 120)
        self.assertEqual(clips[0]['start_s'], 0)
        self.assertEqual(clips[-1]['end_s'], len(frames) / V.FPS)
        self.assertEqual(len([f for f in frames if f['segment'] == 'robot_contact_hold']), 42)
        for i, bundle in enumerate(bundles):
            part = [f for f in frames if f['clip_index'] == i]
            self.assertEqual(part[-1]['source_time_s'], bundle['scored_until_s'])
            for frame in part:
                self.assertEqual(frame['source_time_s'], bundle['time_s'][frame['source_tick']])
                self.assertLessEqual(frame['source_time_s'], bundle['scored_until_s'])
                self.assertEqual(frame['video_time_s'], frame['frame'] / 30)
            normal = [f for f in part if f['segment'] == 'normal']
            self.assertTrue(all(b['source_tick'] - a['source_tick'] == 512 for a, b in zip(normal, normal[1:])))
            slow = [f for f in part if f['segment'] == 'slow']
            self.assertTrue(all(b['source_tick'] - a['source_tick'] == 128 for a, b in zip(slow, slow[1:])))
            if i < 2:
                hold = [f for f in part if f['segment'] == 'robot_contact_hold']
                self.assertTrue(all(f['source_tick'] == 7191 for f in hold))
                self.assertTrue(all(a['source_tick'] <= b['source_tick'] for a, b in zip(
                    [f for f in part if f['segment'] in ('slow', 'robot_contact_hold')],
                    [f for f in part if f['segment'] in ('slow', 'robot_contact_hold')][1:])) )
        self.assertIn('one-second capture', V.stage(frames[-1], bundles[-1]))
        self.assertIn('no whole-ball entry', V.stage([f for f in frames if f['clip_index'] == 0][-1], bundles[0]))

    def test_contact_threshold_and_score_terminal(self):
        times = np.arange(11) / 100
        forces = np.zeros((10, 6))
        forces[2, 0] = .01
        forces[3, 0] = .0101
        forces[4, 4] = 3.
        forces[7, 3] = 2.
        events = V.first_events(times, forces, dict(physics_steps=7))
        self.assertEqual(events, {'board': .03, 'robot': .04})
        with self.assertRaises(ValueError):
            V.first_events(times, forces, dict(physics_steps=11))

    def test_software_renderer_only(self):
        self.assertTrue(V.software_backend('llvmpipe (LLVM)', 'Mesa', '')['software_only'])
        for renderer, cuda in (('NVIDIA RTX4090', ''), ('NVIDIA RTX4070', ''), ('llvmpipe', '0')):
            with self.assertRaises(ValueError):
                V.software_backend(renderer, 'vendor', cuda)

    def test_render_refuses_visible_gpu_before_loading_any_evidence(self):
        with patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': '0'}):
            with patch.object(V, 'load_recordings') as loader:
                with self.assertRaises(ValueError):
                    V.render(SimpleNamespace())
                loader.assert_not_called()


class IntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.study = self.root / 'study'
        (self.study / 'native').mkdir(parents=True)
        assets = self.root / 'assets'
        assets.mkdir()
        for name in ('dropper.urdf', 'board_arm.urdf'):
            (assets / name).write_text('<robot name="mock"/>')
        write(assets / 'poses.json', {})
        repository = Path(V.__file__).resolve().parents[2]
        contract = dict(grid_protocol={'substeps': 64}, scene={'synthetic_fixture': True}, packages={'mujoco': '3.11.0'},
                        identity=dict(scene_sha256=V.digest({'synthetic_fixture': True}),
                                      asset_sha256={'assets/' + p.name: V.sha(p) for p in assets.iterdir()},
                                      source_sha256={n: V.sha(repository / n) for n in V.SOURCE_MODULES},
                                      trace_schema={'native_dt_s': 1 / 15360}))
        contract['fingerprint'] = V.digest(contract)
        self.contract = contract
        write(self.study / 'contract.json', contract)
        write(self.study / 'admission-matrix.json', [])
        times = np.arange(61441) / 15360
        positions = np.zeros((len(times), 4, 3))
        positions[:, :, 2] = .8
        forces = np.zeros((len(times) - 1, 4, 6))
        forces[3300, :, 0] = 1.
        forces[7191, :2, 4] = 1.
        forces[9500, 3, 2] = 1.
        cache = self.study / 'native' / 'native-0000.npz'
        np.savez_compressed(cache, record_ids=[v[0] for v in RECORDS], time_s=times, position_m=positions, forces_N=forces)
        self.rows = []
        for i, (identifier, _) in enumerate(RECORDS):
            row = dict(id=identifier, context_id=str(i), native_release=True, invalid_physics=False,
                       contact_protocol={'substeps': 64}, native_cache_file=cache.name,
                       native_cache_sha256=V.sha(cache), native_cache_row=i,
                       native_initial_ball_center_m=positions[0, i].tolist(), native_initial_q10=[0.] * 6,
                       native_initial_q14=[0.] * 6, physics_steps=32012 if i == 3 else 58700,
                       route_success=i == 3, whole_ball_entry=i == 3, table_assisted_capture=i == 3,
                       values=[.3, -.4, .46, 28., .25], bucket_xy_m=[.8, -.4])
            if i == 3:
                row.update(whole_ball_entry_time_s=16652 / 15360, capture_time_s=32012 / 15360)
            self.rows.append(row)
        (self.study / 'rows.jsonl').write_text('\n'.join(json.dumps(row) for row in self.rows) + '\n')
        manifest = {name: V.sha(self.study / name) for name in ('contract.json', 'rows.jsonl', 'admission-matrix.json', 'native/native-0000.npz')}
        write(self.study / 'evidence-manifest.json', manifest)
        report = dict(version=V.STUDY_VERSION, status='completed', phase='completed', contract_fingerprint=contract['fingerprint'], evidence_sha256=manifest)
        write(self.study / 'report.json', report)
        audit = dict(version=V.AUDIT, status='passed', full_checks_verified=True, native_checks_verified=True, failures=[],
                     checks_count=1, checks=[{'passed': True}], contract_fingerprint=contract['fingerprint'],
                     report_sha256=V.sha(self.study / 'report.json'), evidence_manifest_sha256=V.sha(self.study / 'evidence-manifest.json'),
                     admission_matrix_sha256=V.sha(self.study / 'admission-matrix.json'))
        write(self.root / 'audit.json', audit)
        provenance = {name + '_sha256': V.sha(self.study / filename) for name, filename in
                      [('contract', 'contract.json'), ('report', 'report.json'), ('rows', 'rows.jsonl'),
                       ('evidence_manifest', 'evidence-manifest.json'), ('admission_matrix', 'admission-matrix.json')]}
        provenance.update(audit_sha256=V.sha(self.root / 'audit.json'), contract_fingerprint=contract['fingerprint'])
        write(self.root / 'summary.json', {'provenance': provenance})
        write(self.root / 'records.json', [dict(id=identifier, label=label) for identifier,label in RECORDS])
        self.args = SimpleNamespace(records=self.root / 'records.json', study=self.study, audit=self.root / 'audit.json', summary=self.root / 'summary.json', assets=assets)

    def tearDown(self):
        self.temp.cleanup()

    def test_all_selected_sources_authenticate_without_physics(self):
        bundles, shared = V.load_recordings(self.args)
        self.assertEqual([b['row']['id'] for b in bundles], [r[0] for r in RECORDS])
        self.assertEqual(bundles[0]['events']['robot'], 7191 / 15360)
        self.assertIn('capture', bundles[-1]['events'])
        self.assertEqual(shared['scene_sha256'], self.contract['identity']['scene_sha256'])

    def test_mutated_archive_is_rejected(self):
        with (self.study / 'native/native-0000.npz').open('ab') as stream:
            stream.write(b'tampered')
        with self.assertRaisesRegex(ValueError, 'archive changed'):
            V.load_recordings(self.args)

    def test_mutated_rows_are_rejected(self):
        with (self.study / 'rows.jsonl').open('a') as stream:
            stream.write('\n')
        with self.assertRaisesRegex(ValueError, 'Evidence bytes differ: rows'):
            V.load_recordings(self.args)

    def test_asset_changes_are_rejected(self):
        (self.args.assets / 'poses.json').write_text('{"changed":true}')
        with self.assertRaisesRegex(ValueError, 'assets changed'):
            V.load_recordings(self.args)

    def test_failed_native_audit_cannot_be_accepted_by_rehashing_summary(self):
        audit = V.read(self.args.audit)
        audit['native_checks_verified'] = False
        write(self.args.audit, audit)
        summary = V.read(self.args.summary)
        summary['provenance']['audit_sha256'] = V.sha(self.args.audit)
        write(self.args.summary, summary)
        with self.assertRaisesRegex(ValueError, 'native audit'):
            V.load_recordings(self.args)

    def test_archive_id_checked_in_addition_to_column(self):
        original = np.load
        class WrongIDs:
            def __init__(self, *args, **kwargs):
                self.archive = original(*args, **kwargs)
            def __enter__(self):
                return self
            def __exit__(self, *_):
                self.archive.close()
            def __getitem__(self, name):
                return np.array(['wrong'] * 4) if name == 'record_ids' else self.archive[name]
        with patch.object(V.np, 'load', WrongIDs):
            with self.assertRaisesRegex(ValueError, 'Archive identity differs'):
                V.load_recordings(self.args)


if __name__ == '__main__':
    unittest.main()
