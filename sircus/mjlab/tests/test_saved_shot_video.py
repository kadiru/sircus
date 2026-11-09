import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

from sircus.mjlab import saved_shot_video as V


def write(path, value):
    path.write_text(json.dumps(value))


class TimelineTests(unittest.TestCase):
    def test_frame_plan_uses_actual_ticks_and_exact_declared_speeds(self):
        times = np.arange(61441) / 15360
        plan = V.frame_plan(times)
        self.assertEqual(len(plan), 450)
        normal = [p for p in plan if p['segment'] == 'normal']
        slow = [p for p in plan if p['segment'] == 'slow']
        self.assertEqual([p['source_tick'] for p in normal], list(range(0, 38400, 512)))
        self.assertEqual([p['source_tick'] for p in slow], list(range(0, 38400, 128)))
        self.assertTrue(all(p['source_time_s'] == times[p['source_tick']] for p in plan))
        self.assertTrue(all(p['source_tick'] == 38400 for p in plan if p['playback_speed'] == 0))
        self.assertEqual(plan[-1]['video_time_s'], 449 / 30)

    def test_nearest_tick_never_interpolates_or_extrapolates(self):
        times = np.array([0., .25, .5])
        self.assertEqual(V.nearest_tick(times, .125), 0)
        self.assertEqual(V.nearest_tick(times, .26), 1)
        self.assertEqual(V.nearest_tick(times, .5), 2)
        for value in (-.1, .6, float('nan')):
            with self.assertRaises(ValueError):
                V.nearest_tick(times, value)

    def test_software_guard_rejects_either_gpu_and_exposed_cuda(self):
        self.assertTrue(V.software_backend('llvmpipe (LLVM 15.0.7)', 'Mesa', '')['software_only'])
        for renderer, cuda in (('NVIDIA GeForce RTX 4070', ''), ('NVIDIA GeForce RTX 4090', ''),
                               ('llvmpipe', 'GPU-any')):
            with self.assertRaises(ValueError):
                V.software_backend(renderer, 'vendor', cuda)


class IntegrityTests(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory)
        study = root / 'study'; study.mkdir()
        assets = root / 'assets'; assets.mkdir()
        for name in ('dropper.urdf', 'board_arm.urdf'):
            (assets / name).write_text('<robot><link name="root"/></robot>')
        write(assets / 'poses.json', {})
        scene = {'fixture': 'native geometry identity'}
        scene_file = root / 'scene.json'; write(scene_file, {'scene': scene})
        calibration = study / 'calibration.json'; write(calibration, {'fixture': 'original calibration'})
        repository = Path(V.__file__).resolve().parents[2]
        contract = dict(identity=dict(scene_sha256=V.digest(scene),
            asset_sha256={str(p.relative_to(root)): V.sha(p) for p in assets.iterdir()},
            source_sha256={name: V.sha(repository / name) for name in V.SOURCE_MODULES},
            trace_schema={'native_dt_s': 1/15360}), packages={'mujoco': '3.11.0'})
        contract['fingerprint'] = V.digest(contract)
        write(study / 'contract.json', contract)
        cache = root / 'native-0008.npz'
        times = np.arange(61441) / 15360
        positions = np.zeros((len(times), 1, 3)); positions[0, 0] = [.6, -.2, .54]
        np.savez_compressed(cache, time_s=times, position_m=positions, record_ids=np.array(['historical-shot']))
        score = dict(success=True, complete_success=False, route_success=False, board_bounce=True,
            whole_ball_entry=True, max_retention_s=1., invalid_physics=False)
        row = dict(score, id='historical-shot', native_cache_file=cache.name, native_cache_row=0,
            native_cache_sha256=V.sha(cache), native_release=True, contact_protocol={'substeps': 64},
            native_initial_ball_center_m=[.6, -.2, .54], native_initial_q10=[0.]*6, native_initial_q14=[.1]*6,
            bucket_xy_m=[1.05, -.65], board_center_m=[.6, -.2, .4])
        metadata = root / 'shots.jsonl'; metadata.write_text(json.dumps(row)+'\n')
        evidence = dict(version='sircus-table-assisted-history-example-v1', status='verified_historical_example',
            shot=dict(id=row['id'], release_ball_center_m=row['native_initial_ball_center_m'],
                      bucket_xy_m=row['bucket_xy_m'], settled_board_center_m=row['board_center_m']),
            events={'capture_s': 2.1456}, original_score=score,
            provenance=dict(original_contract_sha256=V.sha(study/'contract.json'),
                original_calibration_file_sha256=V.sha(calibration),
                metadata={'sha256': V.sha(metadata), 'line': 1},
                native_archive={'sha256': V.sha(cache), 'row': 0}))
        evidence_file = root / 'evidence.json'; write(evidence_file, evidence)
        return SimpleNamespace(study=study, metadata=metadata, cache=cache, evidence=evidence_file,
                               scene=scene_file, assets=assets), evidence, row

    def test_valid_trace_retains_historical_failure_and_actual_samples(self):
        with tempfile.TemporaryDirectory() as root:
            args, evidence, row = self.fixture(root)
            result = V.load_recording(args)
            self.assertFalse(result['row']['route_success'])
            self.assertTrue(result['row']['success'])
            self.assertEqual(result['row'], row)
            np.testing.assert_array_equal(result['positions'][0], [.6, -.2, .54])

    def test_changed_cache_metadata_scene_and_asset_are_rejected_before_render(self):
        for field in ('cache', 'metadata', 'scene', 'assets'):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as root:
                args, _, _ = self.fixture(root)
                target = args.assets / 'poses.json' if field == 'assets' else getattr(args, field)
                target.write_bytes(target.read_bytes() + b' ')
                if field == 'scene':
                    write(target, {'scene': {'fixture': 'different geometry'}})
                with self.assertRaises(ValueError):
                    V.load_recording(args)

    def test_metadata_id_and_archive_id_must_match_not_just_valid_file_hashes(self):
        with tempfile.TemporaryDirectory() as root:
            args, evidence, row = self.fixture(root)
            row['id'] = 'another-shot'
            args.metadata.write_text(json.dumps(row)+'\n')
            evidence['shot']['id'] = row['id']
            evidence['provenance']['metadata']['sha256'] = V.sha(args.metadata)
            write(args.evidence, evidence)
            with self.assertRaisesRegex(ValueError, 'Archive record identity'):
                V.load_recording(args)


if __name__ == '__main__':
    unittest.main()
