"""Synthetic publication checks; no native simulator, robot or service is used."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from sircus.mjlab.prepare_tilt650_report import AUDIT_VERSION, KEY, VERSION, compact, full_audit_verified, prepare, watch


class Tilt650PublisherTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.study = self.root/'study'
        self.study.mkdir()
        self.base, self.output = self.root/'base.json', self.root/'display.json'
        self.old = {'active_policy':'preserved', 'xy650_study':{'original':'xy'}, 'clean650_study':{'original':'raised'}, 'previous_clean650_studies':[{'original':'v1'}], 'calibrated650_replay':{'unchanged':True}}
        self.base.write_text(json.dumps(self.old))
        geometry = dict(id='tilt30-00', group_id='tilt30-00', phase='verification', source_id='coarse-64-xy-grid-0043-r0', translation_m=[-.01,.06], board_xy_m=[-.01,.06], bucket_xy_m=[.64,.06], board_height_m=.38, tilt_deg=30, gap_m=.41, release_center_height_m=.8275, distance_m=.65, parent_route_quality=1, q10=[1]*6)
        stat = dict(attempts=3, native_releases=3, captures=3, clean=0, rim_assisted=3, invalid=0, geometry_rejections=0, whole_entries=3, max_retention_s=1)
        qualification = dict(status='capture_only', qualified=False, capture_qualified=True, clean_qualified=False, capture_both=True, clean_both=False, physics_valid=True, oracle_agreement=True, max_drift_m=.002, parent_source_capture=True)
        self.raw = dict(version=VERSION, status='completed', phase='completed', contract_fingerprint='new-tilt', settings=dict(source_count=5, source_ids=['coarse-64-xy-grid-0043-r0'], repeats_per_resolution=3, substeps=[64,128], control_requests=24, verification_requests=30, max_requests=54, minimum_clearance_m=.005, max_drift_350ms_m=.005), counts=dict(stat, phase_counts={'verification':6}, substep_counts={'64':3,'128':3}), gate=dict(status='capture_only', controls_pre_passed=True, controls_post_passed=True, qualified_settings=0, capture_qualified_settings=1, more_ppo_started=False), groups=[dict(geometry, by_substeps={'64':stat,'128':stat})], controls=[dict(phase='controls_pre', group_id='positive', role='positive_control', substeps=64, stats=stat)], top_settings=[], verification=[dict(geometry, by_substeps={'64':stat,'128':stat}, qualification=qualification, sensitivity=dict(paired_attempts=3, capture_agreement=3, nominal_position_delta_350ms_max_m=.002))], trace_examples=[dict(geometry, record_id='native-128', substeps=128, position_m=[[-.01,.06,.8275],[.64,.06,.1]], invalid=None, invalid_physics=False, route_success=True, route_quality=1, whole_ball_entry=True, max_retention_s=1, scalar_score_audit_passed=True, cache_file='native/actual.npz', cache_sha256='a'*64)], source_sha256={'sircus/mjlab/tilt650_contract.py':'b'*64, '/private/host':'c'*64, '../escape':'d'*64})
        self.audit = dict(version=AUDIT_VERSION, status='passed', contract_fingerprint='new-tilt', checks_count=1, checks=[{'passed':True}], failures=[])

    def tearDown(self):
        self.tmp.cleanup()

    def write(self):
        report, manifest = self.study/'report.json', self.study/'evidence-manifest.json'
        report.write_text(json.dumps(self.raw))
        manifest.write_text(json.dumps({'native/actual.npz':'a'*64}))
        self.audit.update(report_sha256=hashlib.sha256(report.read_bytes()).hexdigest(), evidence_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())
        (self.study/'audit-tilt650.json').write_text(json.dumps(self.audit))

    def test_additive_bounded_metadata_separate_capture_and_clean_no_history_changes(self):
        self.write()
        before = {p.name:p.read_bytes() for p in self.study.iterdir()}
        result = prepare(self.base, self.study, self.output)
        value = result[KEY]
        self.assertEqual({k:result[k] for k in self.old}, self.old)
        self.assertEqual(value['settings']['max_requests'],54)
        self.assertEqual(value['settings']['verification_requests'],30)
        self.assertEqual(value['settings']['source_count'],5)
        self.assertEqual(value['gate']['capture_qualified_settings'],1)
        self.assertEqual(value['gate']['qualified_settings'],0)
        row = value['verification'][0]
        self.assertEqual(row['translation_m'],[-.01,.06])
        self.assertEqual(row['parent_route_quality'],1)
        self.assertTrue(row['qualification']['capture_qualified'])
        self.assertFalse(row['qualification']['clean_qualified'])
        self.assertTrue(row['qualification']['parent_source_capture'])
        self.assertEqual(value['controls'][0]['statistics']['rim_assisted'],3)
        self.assertTrue(value['audit']['full_checks_verified'])
        self.assertTrue(value['audit']['bound_files_verified'])
        self.assertEqual(value['source_sha256'],{'sircus/mjlab/tilt650_contract.py':'b'*64})
        self.assertNotIn('q10',json.dumps(value))
        self.assertEqual({p.name:p.read_bytes() for p in self.study.iterdir()},before)

    def test_exact_pair_completed_phase_full_nonempty_strict_checks_and_identity_required(self):
        self.assertTrue(full_audit_verified(self.raw,self.audit))
        mutations = [lambda r,a:r.update(status='running'), lambda r,a:r.update(version='sircus-xy650-shift-study-v1'), lambda r,a:r.update(phase='verify128'), lambda r,a:a.update(version='sircus-xy650-independent-audit-v2'), lambda r,a:a.update(contract_fingerprint='other'), lambda r,a:a.update(checks=[]), lambda r,a:a.update(checks_count=True), lambda r,a:a.update(checks_count=0), lambda r,a:a.update(checks_count=2), lambda r,a:a.update(checks=[{'passed':1}]), lambda r,a:a.update(failures=['failed'])]
        for mutate in mutations:
            raw,audit = copy.deepcopy(self.raw),copy.deepcopy(self.audit)
            mutate(raw,audit)
            self.assertFalse(full_audit_verified(raw,audit))

    def test_post_audit_byte_changes_and_missing_proof_cannot_verify(self):
        for name in ('report.json','evidence-manifest.json','audit-tilt650.json'):
            self.write()
            path = self.study/name
            if name == 'audit-tilt650.json': path.unlink()
            else: path.write_bytes(path.read_bytes()+b'\n')
            self.assertFalse(prepare(self.base,self.study,self.output)[KEY]['audit']['full_checks_verified'])
        self.write()
        self.audit.pop('report_sha256')
        (self.study/'audit-tilt650.json').write_text(json.dumps(self.audit))
        self.assertFalse(prepare(self.base,self.study,self.output)[KEY]['audit']['full_checks_verified'])

    def test_external_proof_supported_without_mutating_original_evidence(self):
        self.write()
        external = self.root/'independent.json'
        external.write_bytes((self.study/'audit-tilt650.json').read_bytes())
        (self.study/'audit-tilt650.json').unlink()
        before = {p.name:p.read_bytes() for p in self.study.iterdir()}
        proof_before = external.read_bytes()
        self.assertFalse(prepare(self.base,self.study,self.output)[KEY]['audit']['full_checks_verified'])
        self.assertTrue(prepare(self.base,self.study,self.output,audit_path=external)[KEY]['audit']['full_checks_verified'])
        self.assertEqual({p.name:p.read_bytes() for p in self.study.iterdir()},before)
        self.assertEqual(external.read_bytes(),proof_before)
        with self.assertRaises(ValueError): prepare(self.base,self.study,external,audit_path=external)
        (self.study/'report.json').write_bytes((self.study/'report.json').read_bytes()+b'\n')
        self.assertFalse(prepare(self.base,self.study,self.output,audit_path=external)[KEY]['audit']['full_checks_verified'])

    def test_missing_counts_and_invalid_traces_stay_unknown(self):
        raw = dict(version=VERSION,status='running',counts={'attempts':0},verification=[{'id':'pending'}],trace_examples=[{'position_m':[[0,0,float('nan')],[0,0,0]]}])
        value = compact(raw)
        self.assertNotIn('captures',value['counts'])
        self.assertNotIn('capture_qualified',value['verification'][0]['qualification'])
        self.assertEqual(value['trace_examples'],[])
        raw = copy.deepcopy(self.raw)
        raw['trace_examples'][0].update(cache_file='../escape',cache_sha256='bad')
        trace = compact(raw)['trace_examples'][0]
        self.assertNotIn('cache_file',trace)
        self.assertNotIn('cache_sha256',trace)

    def test_output_and_watch_guards(self):
        self.assertIsNone(prepare(self.base,self.study,self.output)[KEY])
        self.write()
        for output in (self.base,self.study/'display.json'):
            with self.assertRaises(ValueError): prepare(self.base,self.study,output)
        self.raw['counts']['attempts'] = float('nan')
        self.write()
        with self.assertRaises(ValueError): prepare(self.base,self.study,self.output)
        for seconds in (-1,float('nan'),30001):
            with self.assertRaises(ValueError): watch(self.base,self.study,self.output,seconds)


if __name__ == '__main__':
    unittest.main()
