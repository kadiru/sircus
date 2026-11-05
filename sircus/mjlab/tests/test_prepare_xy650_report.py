"""Synthetic CPU publication tests; no simulator or live evidence is launched."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from sircus.mjlab.prepare_xy650_report import AUDIT_VERSION,KEY,VERSION,compact,full_audit_verified,prepare,watch

class XY650PublisherTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.study=self.root/'study';self.study.mkdir();self.base=self.root/'base.json';self.output=self.root/'display.json'
        self.old={'active_policy':'preserved','clean650_study':{'fingerprint':'raised-original','captures':0},'previous_clean650_studies':[{'fingerprint':'v1-original'}],'calibrated650_replay':{'unchanged':True}};self.base.write_text(json.dumps(self.old))
        g=dict(id='g',group_id='g',phase='verification',translation_m=[.01,.06],board_height_m=.38,tilt_deg=28,gap_m=.41,release_center_height_m=.8275,distance_m=.65,board_xy_m=[.01,.06],bucket_xy_m=[.66,.06],q10=[1]*6,query_fingerprint='q'*64,execution_fingerprint='e'*64)
        stat=dict(attempts=3,native_releases=3,captures=3,clean=3,rim_assisted=0,invalid=0,geometry_rejections=0,whole_entries=3,max_retention_s=1)
        diagnostic=dict(version='sircus-xy650-first-flight-aperture-v1',available=True,start_index=2,end_index=8,start_time_s=.2,end_time_s=.8,cutoff_reason='bucket_contact',contact_channels=['bucket'],crossing_plane='ball_center_at_rim',first_crossing=dict(time_s=.6,interval_index=5,fraction=.3,xy_m=[.66,.06],q14=[1]*6),mouth_clearance_m=.02,closest_opening_distance_m=.03,sample_count=7,q10=[1]*6,private_token='do not expose')
        self.raw=dict(version=VERSION,status='completed',phase='completed',contract_fingerprint='fp',settings=dict(max_requests=388,coarse_requests=253,translations=dict(dx_m=[-.01,0,.01],dy_m=[.04,.06,.08],rule='translated-board-and-bucket-fixed650-bearing-v1'),grid=dict(heights_m=[.36,.38,.4],tilts_deg=[26,28,30],gaps_m=[.39,.41,.43],reference=[.38,28,.3868])),counts=dict(stat,phase_counts={'verification':6}),gate=dict(status='passed',controls_pre_passed=True,controls_post_passed=True,qualified_settings=1,more_ppo_started=False),
          groups=[dict(g,by_substeps={'64':stat,'128':stat})],controls=[dict(g,phase='controls_pre',role='positive_control',substeps=64,stats=stat)],top_settings=[dict(g,phase='coarse',stats=stat,first_flight=diagnostic)],verification=[dict(g,by_substeps={'64':stat,'128':stat},qualification=dict(status='qualified',qualified=True,clean_both=True,physics_valid=True,oracle_agreement=True),sensitivity=dict(paired_attempts=3,capture_agreement=3,nominal_position_delta_350ms_max_m=.002))],
          trace_examples=[dict(g,record_id='actual',substeps=128,position_m=[[.01,.06,.8275],[.66,.06,.1]],invalid=None,invalid_physics=False,route_success=True,route_quality=2,whole_ball_entry=True,max_retention_s=1,scalar_score_audit_passed=True,first_flight=diagnostic,cache_file='native/actual.npz',cache_sha256='a'*64)],source_sha256={'sircus/mjlab/xy650_contract.py':'b'*64,'/private/host':'c'*64,'../escape':'d'*64})
        self.audit=dict(version=AUDIT_VERSION,status='passed',contract_fingerprint='fp',checks_count=1,checks=[{'passed':True}],failures=[])
    def tearDown(self):self.tmp.cleanup()
    def write(self):
        report=self.study/'report.json';manifest=self.study/'evidence-manifest.json';report.write_text(json.dumps(self.raw));manifest.write_text(json.dumps({'native/actual.npz':'a'*64}));self.audit.update(report_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),evidence_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest());(self.study/'audit-xy650.json').write_text(json.dumps(self.audit))
    def test_additive_xy_metadata_and_firstflight_without_raw_joints_or_history_changes(self):
        self.write();before={p.name:p.read_bytes() for p in self.study.iterdir()};result=prepare(self.base,self.study,self.output);v=result[KEY]
        self.assertEqual({k:result[k] for k in self.old},self.old);self.assertEqual(v['settings']['max_requests'],388);self.assertEqual(v['settings']['translations']['dy_m'],[.04,.06,.08]);self.assertEqual(v['controls'][0]['substeps'],64)
        self.assertEqual(v['top_settings'][0]['translation_m'],[.01,.06]);self.assertEqual(v['top_settings'][0]['query_fingerprint'],'q'*64);self.assertEqual(v['top_settings'][0]['first_flight']['first_crossing']['xy_m'],[.66,.06]);self.assertEqual(v['trace_examples'][0]['first_flight']['mouth_clearance_m'],.02)
        self.assertTrue(v['audit']['full_checks_verified']);self.assertTrue(v['audit']['bound_files_verified']);self.assertEqual(v['source_sha256'],{'sircus/mjlab/xy650_contract.py':'b'*64});self.assertEqual({p.name:p.read_bytes() for p in self.study.iterdir()},before)
        encoded=json.dumps(v);self.assertNotIn('q10',encoded);self.assertNotIn('q14',encoded);self.assertNotIn('private_token',encoded);self.assertNotIn('do not expose',encoded);self.assertNotIn('previous_clean650_studies',v)
    def test_exact_pair_complete_full_checks_and_fingerprint_required(self):
        self.assertTrue(full_audit_verified(self.raw,self.audit))
        for change in (lambda r,a:r.update(status='running'),lambda r,a:r.update(version='sircus-clean650-height-tilt-gap-v1'),lambda r,a:r.update(phase='verify128'),lambda r,a:a.update(version='sircus-clean650-independent-audit-v1'),lambda r,a:a.update(contract_fingerprint='other'),lambda r,a:a.update(checks=[]),lambda r,a:a.update(checks_count=True),lambda r,a:a.update(checks_count=0),lambda r,a:a.update(checks_count=2),lambda r,a:a.update(checks=[{'passed':1}]),lambda r,a:a.update(failures=['failed'])):
            raw,audit=copy.deepcopy(self.raw),copy.deepcopy(self.audit);change(raw,audit);self.assertFalse(full_audit_verified(raw,audit))
    def test_postaudit_byte_tampering_and_missing_bindings_fail_closed(self):
        for kind in ('report','manifest','missingmanifest','missinghash','wronghash','missingaudit'):
            self.write()
            if kind=='report':(self.study/'report.json').write_bytes((self.study/'report.json').read_bytes()+b'\n')
            if kind=='manifest':(self.study/'evidence-manifest.json').write_bytes((self.study/'evidence-manifest.json').read_bytes()+b'\n')
            if kind=='missingmanifest':(self.study/'evidence-manifest.json').unlink()
            if kind=='missinghash':self.audit.pop('report_sha256')
            if kind=='wronghash':self.audit['report_sha256']='0'*64
            if kind in ('missinghash','wronghash'):(self.study/'audit-xy650.json').write_text(json.dumps(self.audit))
            if kind=='missingaudit':(self.study/'audit-xy650.json').unlink()
            self.assertFalse(prepare(self.base,self.study,self.output)[KEY]['audit']['full_checks_verified'])
    def test_missing_diagnostic_counts_and_null_crossing_remain_explicit(self):
        raw=dict(version=VERSION,counts={'attempts':0},top_settings=[{'id':'x','first_flight':{'version':'sircus-xy650-first-flight-aperture-v1','available':True,'first_crossing':None,'mouth_clearance_m':None}}],trace_examples=[{'position_m':[[0,0,float('nan')],[0,0,0]]}]);v=compact(raw)
        self.assertNotIn('captures',v['counts']);self.assertIsNone(v['top_settings'][0]['first_flight']['first_crossing']);self.assertIsNone(v['top_settings'][0]['first_flight']['mouth_clearance_m']);self.assertEqual(v['trace_examples'],[])
        raw=copy.deepcopy(self.raw);raw['trace_examples'][0].update(cache_file='../escape',cache_sha256='bad');t=compact(raw)['trace_examples'][0];self.assertNotIn('cache_file',t);self.assertNotIn('cache_sha256',t)
    def test_separate_output_and_bounded_watch(self):
        self.assertIsNone(prepare(self.base,self.study,self.output)[KEY]);self.write()
        for output in (self.base,self.study/'display.json'):
            with self.assertRaises(ValueError):prepare(self.base,self.study,output)
        self.raw['counts']['attempts']=float('nan');self.write()
        with self.assertRaises(ValueError):prepare(self.base,self.study,self.output)
        for value in (-1,float('nan'),30001):
            with self.assertRaises(ValueError):watch(self.base,self.study,self.output,value)
if __name__=='__main__':unittest.main()
