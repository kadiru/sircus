import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from sircus.mjlab.prepare_replay650_report import KEY, VERSION, compact, full_audit_verified, prepare, watch


class Replay650DisplayTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.study=self.root/'study';self.study.mkdir()
        self.base,self.output=self.root/'base.json',self.root/'display.json'
        self.old=dict(active_policy='existing',lowboard_evaluation={'native_releases':208,'captures':0},shallow_evaluation={'attempts':4,'captures':0},distance_evaluation={'old':'unchanged'})
        self.base.write_text(json.dumps(self.old))
        geometry=dict(board_height_m=.38,tilt_deg=28.,gap_m=.3868,release_center_height_m=.8043,board_xy_m=[0,0],bucket_xy_m=[.65,0],distance_m=.65)
        self.raw=dict(version=VERSION,status='completed',phase='completed',contract_fingerprint='replay',max_requests=48,
            settings=dict(candidate=geometry,controls={'positive_control':dict(id='positive-control',role='positive_control',q10=[1]*6,source_record_sha256='a'*64),
                'negative_control':dict(id='negative-control',role='negative_control',q14=[1]*6,source_record_sha256='b'*64)},substeps=[64,128],gaps_m=[.3768,.3808,.3848,.3868,.3888]),
            counts=dict(attempts=18,native_releases=18,captures=12,invalid=0,clean=3,whole_entries=15,max_retention_s=1.2),
            gate=dict(status='passed',positive_control=True,negative_control=True,candidate=True,physics_valid=True,oracle_agreement=True,max_drift_m=.002,window_opened=True),
            calibration=dict(score_config_sha256='b'*64,calibration_sha256='c'*64,verification_sha256='d'*64),
            groups=[dict(id='candidate-650',role='candidate',**geometry,q14=[1]*6,eligible=True,
                by_substeps={'64':dict(attempts=3,native_releases=3,captures=3,invalid=0,clean=1,whole_entries=3,max_retention_s=1.2)},
                sensitivity=dict(paired_attempts=3,capture_agreement=3,nominal_position_delta_350ms_max_m=.002))],
            trace_examples=[dict(record_id='actual',group_id='candidate-650',role='candidate',**geometry,substeps=64,route_success=True,whole_ball_entry=True,
                max_retention_s=1.2,invalid='',invalid_physics=False,scalar_score_audit_passed=True,external_contact_steps=0,
                position_m=[[0,0,.8043],[.65,0,.1]],cache_file='native/trace.npz',cache_sha256='e'*64,q10=[1]*6)],
            source_sha256={'sircus/mjlab/replay650_contract.py':'f'*64,'/absolute/host':'a'*64,'../outside':'a'*64})
        self.audit=dict(status='passed',contract_fingerprint='replay',checks_count=1,checks=[{'passed':True}],failures=[])

    def tearDown(self):self.tmp.cleanup()

    def write(self):
        report=self.study/'report.json';manifest=self.study/'evidence-manifest.json'
        report.write_text(json.dumps(self.raw));manifest.write_text(json.dumps({'native/trace.npz':'e'*64}))
        self.audit.update(report_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),evidence_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())
        (self.study/'audit-replay650.json').write_text(json.dumps(self.audit))

    def test_additive_output_preserves_histories_exact_values_and_observation_identity(self):
        self.write();before={p.name:p.read_bytes() for p in self.study.iterdir()}
        result=prepare(self.base,self.study,self.output);value=result[KEY]
        self.assertEqual({k:result[k] for k in self.old},self.old)
        self.assertEqual(value['settings']['candidate']['gap_m'],.3868)
        self.assertEqual(set(value['settings']['controls']),{'positive_control','negative_control'})
        self.assertNotIn('q10',value['settings']['controls']['positive_control'])
        self.assertNotIn('q14',value['settings']['controls']['negative_control'])
        self.assertEqual(value['settings']['controls']['positive_control']['source_record_sha256'],'a'*64)
        self.assertNotIn('q14',value['groups'][0]);self.assertNotIn('q10',value['trace_examples'][0])
        self.assertEqual(value['groups'][0]['by_substeps']['64']['whole_entries'],3)
        self.assertEqual(value['counts']['attempts'],18)
        self.assertEqual(value['calibration'],self.raw['calibration'])
        trace=value['trace_examples'][0]
        self.assertEqual(trace['position_m'],self.raw['trace_examples'][0]['position_m'])
        self.assertEqual(trace['group_id'],'candidate-650');self.assertEqual(trace['cache_file'],'native/trace.npz')
        self.assertFalse(trace['invalid_physics']);self.assertTrue(trace['scalar_score_audit_passed'])
        self.assertTrue(value['audit']['full_checks_verified'])
        self.assertTrue(value['audit']['bound_files_verified'])
        self.assertEqual(value['audit']['report_sha256'],value['audit']['observed_report_sha256'])
        self.assertEqual(value['audit']['evidence_manifest_sha256'],value['audit']['observed_evidence_manifest_sha256'])
        self.assertEqual(value['audit']['source_sha256'],hashlib.sha256(before['audit-replay650.json']).hexdigest())
        self.assertEqual({p.name:p.read_bytes() for p in self.study.iterdir()},before)
        self.assertEqual(value['source_sha256'],{'sircus/mjlab/replay650_contract.py':'f'*64})

    def test_postaudit_report_or_manifest_changes_and_missing_bindings_are_unverified(self):
        for kind in ('changed-report','changed-manifest','missing-manifest','missing-report-hash','missing-manifest-hash','wrong-report-hash'):
            with self.subTest(kind=kind):
                self.write()
                if kind=='changed-report':
                    (self.study/'report.json').write_bytes((self.study/'report.json').read_bytes()+b'\n')
                if kind=='changed-manifest':
                    (self.study/'evidence-manifest.json').write_bytes((self.study/'evidence-manifest.json').read_bytes()+b'\n')
                if kind=='missing-manifest':(self.study/'evidence-manifest.json').unlink()
                audit=copy.deepcopy(self.audit)
                if kind=='missing-report-hash':audit.pop('report_sha256')
                if kind=='missing-manifest-hash':audit.pop('evidence_manifest_sha256')
                if kind=='wrong-report-hash':audit['report_sha256']='0'*64
                (self.study/'audit-replay650.json').write_text(json.dumps(audit))
                before={p.name:p.read_bytes() for p in self.study.iterdir()}
                value=prepare(self.base,self.study,self.output)[KEY]
                self.assertFalse(value['audit']['bound_files_verified'])
                self.assertFalse(value['audit']['full_checks_verified'])
                self.assertIn('mismatched',value['audit']['verification_reason'])
                self.assertEqual({p.name:p.read_bytes() for p in self.study.iterdir()},before)

    def test_legacy_control_list_is_still_compacted_without_raw_joints(self):
        raw=copy.deepcopy(self.raw);raw['settings']['controls']=list(raw['settings']['controls'].values())
        value=compact(raw)['settings']['controls']
        self.assertIsInstance(value,list);self.assertEqual(len(value),2);self.assertNotIn('q10',value[0]);self.assertNotIn('q14',value[1])

    def test_matching_complete_full_audit_is_required_and_embedded_flags_are_ignored(self):
        self.assertTrue(full_audit_verified(self.raw,self.audit))
        changes=[lambda r,a:r.update(version='old'),lambda r,a:r.update(status='running'),lambda r,a:r.update(phase='replay-128'),
            lambda r,a:a.update(contract_fingerprint='old'),lambda r,a:a.update(checks=[]),lambda r,a:a.update(checks_count=0),
            lambda r,a:a.update(checks_count=True),lambda r,a:a.update(checks_count=1.0),lambda r,a:a.update(checks_count=2),
            lambda r,a:a.update(checks=[{'passed':'yes'}]),lambda r,a:a.update(checks=[{'passed':1}]),lambda r,a:a.update(failures=[{}])]
        for change in changes:
            raw,audit=copy.deepcopy(self.raw),copy.deepcopy(self.audit);change(raw,audit);self.assertFalse(full_audit_verified(raw,audit))
        self.raw['audit']=dict(self.audit,full_checks_verified=True);(self.study/'report.json').write_text(json.dumps(self.raw))
        self.assertFalse(prepare(self.base,self.study,self.output)[KEY]['audit']['full_checks_verified'])

    def test_missing_measurements_and_booleans_stay_missing_and_no_trace_is_reconstructed(self):
        raw=dict(version=VERSION,gate={'status':'pending'},groups=[dict(id='g',role='candidate',by_substeps={'64':{'attempts':0}})],
            trace_examples=[dict(record_id='bad',position_m=[[0,0,float('nan')],[0,0,0]])])
        value=compact(raw);self.assertNotIn('physics_valid',value['gate']);self.assertNotIn('captures',value['groups'][0]['by_substeps']['64'])
        self.assertNotIn('native_releases',value['groups'][0]['by_substeps']['64']);self.assertEqual(value['trace_examples'],[])
        raw=copy.deepcopy(self.raw);raw['trace_examples'][0].update(cache_file='../outside.npz',cache_sha256='not-a-hash')
        trace=compact(raw)['trace_examples'][0];self.assertNotIn('cache_file',trace);self.assertNotIn('cache_sha256',trace)

    def test_optional_queued_output_write_guards_and_bounded_watch(self):
        self.assertIsNone(prepare(self.base,self.study,self.output)[KEY])
        self.raw.update(status='waiting_for_4090',phase='device_queue',queue={'wait_s':30,'reason':'shared lock'});self.write()
        value=prepare(self.base,self.study,self.output)[KEY];self.assertEqual(value['queue']['wait_s'],30);self.assertFalse(value['audit']['full_checks_verified'])
        for output in (self.base,self.study/'display.json'):
            with self.assertRaises(ValueError):prepare(self.base,self.study,output)
        before=self.output.read_bytes();self.raw['counts']['attempts']=float('nan');self.write()
        with self.assertRaises(ValueError):prepare(self.base,self.study,self.output)
        self.assertEqual(self.output.read_bytes(),before)
        for seconds in (-1,float('nan'),30001):
            with self.assertRaises(ValueError):watch(self.base,self.study,self.output,seconds)
        with mock.patch('sircus.mjlab.prepare_replay650_report.prepare',return_value={KEY:{'status':'failed'}}):
            self.assertEqual(watch(self.base,self.study,self.output,10)[KEY]['status'],'failed')


if __name__=='__main__':unittest.main()
