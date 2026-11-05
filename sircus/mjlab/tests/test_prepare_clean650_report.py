"""Read-only publication bindings; fixtures are synthetic CPU data only."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from sircus.mjlab.prepare_clean650_report import AUDIT_VERSION,FINE_VERSION,FINE_AUDIT_VERSION,RAISED_VERSION,RAISED_AUDIT_VERSION,HISTORY_KEY,KEY,VERSION,compact,full_audit_verified,merge_display,prepare,watch

class Clean650PublisherTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.study=self.root/'study';self.study.mkdir();self.base=self.root/'base.json';self.output=self.root/'display.json'
        self.old={'active_policy':'preserved','calibrated650_replay':{'captures':3,'outcome':'historical'},'shallow_evaluation':{'untouched':True}};self.base.write_text(json.dumps(self.old))
        g=dict(id='g',group_id='g',phase='verification',board_height_m=.32,tilt_deg=24,gap_m=.46,release_center_height_m=.8175,distance_m=.65,board_xy_m=[0,0],bucket_xy_m=[.65,0],q10=[1]*6)
        stat=dict(attempts=3,native_releases=3,captures=3,clean=3,rim_assisted=0,invalid=0,whole_entries=3,max_retention_s=1)
        self.raw=dict(version=VERSION,status='completed',phase='completed',contract_fingerprint='fp',settings=dict(max_requests=496,grid=dict(heights_m=[.22,.26],tilts_deg=[16,20],gaps_m=[.2,.24],reference=dict(g,q14=[1]*6))),
          counts=dict(stat,phase_counts={'verification':6}),gate=dict(status='passed',controls_pre_passed=True,controls_post_passed=True,qualified_settings=1,more_ppo_started=False),
          groups=[dict(g,by_substeps={'64':stat,'128':stat})],controls=[dict(g,phase='controls_pre',role='positive_control',substeps=64,stats=stat)],top_settings=[dict(g,phase='coarse',stats=stat)],
          verification=[dict(g,by_substeps={'64':stat,'128':stat},qualification=dict(status='qualified',qualified=True,clean_both=True,physics_valid=True,oracle_agreement=True),sensitivity=dict(paired_attempts=3,capture_agreement=3,nominal_position_delta_350ms_max_m=.002))],
          trace_examples=[dict(g,record_id='actual',substeps=128,position_m=[[0,0,.8175],[.65,0,.1]],invalid=None,invalid_physics=False,route_success=True,route_quality=2,whole_ball_entry=True,max_retention_s=1,scalar_score_audit_passed=True,cache_file='native/actual.npz',cache_sha256='a'*64)],
          source_sha256={'sircus/mjlab/clean650_contract.py':'b'*64,'/host/secret':'c'*64,'../escape':'d'*64})
        self.audit=dict(version=AUDIT_VERSION,status='passed',contract_fingerprint='fp',checks_count=1,checks=[{'passed':True}],failures=[])
    def tearDown(self):self.tmp.cleanup()
    def write(self):
        report=self.study/'report.json';manifest=self.study/'evidence-manifest.json';report.write_text(json.dumps(self.raw));manifest.write_text(json.dumps({'native/actual.npz':'a'*64}));self.audit.update(report_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),evidence_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest());(self.study/'audit-clean650.json').write_text(json.dumps(self.audit))
    def test_additive_canonical_schema_preserves_counts_but_excludes_raw_joint_values(self):
        self.write();before={p.name:p.read_bytes() for p in self.study.iterdir()};result=prepare(self.base,self.study,self.output);v=result[KEY]
        self.assertEqual({k:result[k] for k in self.old},self.old);self.assertEqual(v['settings']['max_requests'],496);self.assertEqual(v['controls'][0]['substeps'],64);self.assertEqual(v['controls'][0]['statistics']['captures'],3)
        self.assertEqual(v['top_settings'][0]['statistics']['clean'],3);self.assertTrue(v['verification'][0]['qualification']['clean_both']);self.assertTrue(v['gate']['controls_post_passed'])
        self.assertTrue(v['audit']['full_checks_verified']);self.assertTrue(v['audit']['bound_files_verified']);self.assertEqual(v['audit']['source_sha256'],hashlib.sha256(before['audit-clean650.json']).hexdigest());self.assertEqual(v['source_sha256'],{'sircus/mjlab/clean650_contract.py':'b'*64})
        self.assertEqual({p.name:p.read_bytes() for p in self.study.iterdir()},before)
        encoded=json.dumps(v);self.assertNotIn('q10',encoded);self.assertNotIn('q14',encoded);self.assertEqual(v['trace_examples'][0]['position_m'],self.raw['trace_examples'][0]['position_m'])
    def test_exact_completed_version_full_check_list_and_fingerprint_required(self):
        self.assertTrue(full_audit_verified(self.raw,self.audit))
        for change in (lambda r,a:r.update(status='running'),lambda r,a:r.update(version='historical'),lambda r,a:r.update(phase='verify128'),lambda r,a:a.update(version='older'),lambda r,a:a.update(contract_fingerprint='other'),lambda r,a:a.update(checks=[]),lambda r,a:a.update(checks_count=True),lambda r,a:a.update(checks_count=0),lambda r,a:a.update(checks_count=2),lambda r,a:a.update(checks=[{'passed':1}]),lambda r,a:a.update(failures=['failed'])):
            raw,audit=copy.deepcopy(self.raw),copy.deepcopy(self.audit);change(raw,audit);self.assertFalse(full_audit_verified(raw,audit))
    def test_report_manifest_byte_tampering_and_missing_audit_bindings_fail_closed(self):
        for kind in ('report','manifest','missingmanifest','missinghash','wronghash','missingaudit'):
            self.write()
            if kind=='report':(self.study/'report.json').write_bytes((self.study/'report.json').read_bytes()+b'\n')
            if kind=='manifest':(self.study/'evidence-manifest.json').write_bytes((self.study/'evidence-manifest.json').read_bytes()+b'\n')
            if kind=='missingmanifest':(self.study/'evidence-manifest.json').unlink()
            if kind=='missinghash':self.audit.pop('report_sha256')
            if kind=='wronghash':self.audit['report_sha256']='0'*64
            if kind in ('missinghash','wronghash'):(self.study/'audit-clean650.json').write_text(json.dumps(self.audit))
            if kind=='missingaudit':(self.study/'audit-clean650.json').unlink()
            self.assertFalse(prepare(self.base,self.study,self.output)[KEY]['audit']['full_checks_verified'])
    def test_missing_counts_stay_missing_and_invalid_trace_is_not_reconstructed(self):
        raw=dict(version=VERSION,counts={'attempts':0},verification=[{'id':'x','by_substeps':{'64':{'attempts':0}}}],trace_examples=[{'position_m':[[0,0,float('nan')],[0,0,0]]}]);v=compact(raw)
        self.assertNotIn('captures',v['counts']);self.assertNotIn('native_releases',v['verification'][0]['by_substeps']['64']);self.assertEqual(v['trace_examples'],[])
        raw=copy.deepcopy(self.raw);raw['trace_examples'][0].update(cache_file='../escape',cache_sha256='bad');t=compact(raw)['trace_examples'][0];self.assertNotIn('cache_file',t);self.assertNotIn('cache_sha256',t)
    def test_explicit_profile_and_auditor_pairing(self):
        for version,expected in ((VERSION,AUDIT_VERSION),(FINE_VERSION,FINE_AUDIT_VERSION),(RAISED_VERSION,RAISED_AUDIT_VERSION)):
            raw=dict(self.raw,version=version)
            for auditor in (AUDIT_VERSION,FINE_AUDIT_VERSION,RAISED_AUDIT_VERSION,"unregistered-auditor"):
                self.assertEqual(full_audit_verified(raw,dict(self.audit,version=auditor)),auditor==expected)
        self.assertFalse(full_audit_verified(dict(self.raw,version="unregistered-profile"),self.audit))

    def test_previous_v1_survives_first_queue_fine_publication_and_refresh_without_duplicates(self):
        self.raw['counts'].update(attempts=496,native_releases=155,clean=12)
        self.raw['gate'].update(qualified_settings=0,status='failed')
        self.write();published=prepare(self.base,self.study,self.output);before=copy.deepcopy(published)
        queued=compact(dict(version=FINE_VERSION,status='waiting_for_4090',phase='device_queue'))
        waiting=merge_display(published,queued)
        self.assertEqual(published,before)
        self.assertEqual(len(waiting[HISTORY_KEY]),1)
        old=waiting[HISTORY_KEY][0]
        self.assertEqual(old['version'],VERSION);self.assertEqual(old['counts']['attempts'],496);self.assertEqual(old['counts']['native_releases'],155);self.assertEqual(old['gate']['qualified_settings'],0)
        self.assertEqual(waiting[KEY][HISTORY_KEY],waiting[HISTORY_KEY]);self.assertNotIn('groups',old);self.assertNotIn('position_m',json.dumps(old))
        self.assertTrue(old['audit']['full_checks_verified'])
        self.base.write_text(json.dumps(waiting))
        self.raw.update(version=FINE_VERSION,contract_fingerprint='fine')
        self.audit.update(version=FINE_AUDIT_VERSION,contract_fingerprint='fine')
        self.write();fine=prepare(self.base,self.study,self.output)
        self.assertTrue(fine[KEY]['audit']['full_checks_verified']);self.assertEqual(fine[KEY]['version'],FINE_VERSION)
        self.assertEqual(len(fine[HISTORY_KEY]),1);self.assertEqual(fine[HISTORY_KEY][0],old)
        self.base.write_text(json.dumps(fine));again=prepare(self.base,self.study,self.output)
        self.assertEqual(again[HISTORY_KEY],fine[HISTORY_KEY])
        self.assertEqual({k:again[k] for k in self.old},self.old)

    def test_both_prior_profiles_survive_raised_queue_completion_and_repeated_refresh(self):
        self.write();original=prepare(self.base,self.study,self.output)
        self.base.write_text(json.dumps(original))
        self.raw.update(version=FINE_VERSION,contract_fingerprint='fine')
        self.audit.update(version=FINE_AUDIT_VERSION,contract_fingerprint='fine')
        self.write();fine=prepare(self.base,self.study,self.output)
        before=copy.deepcopy(fine)
        waiting=merge_display(fine,compact(dict(version=RAISED_VERSION,status='waiting_for_4090',phase='device_queue')))
        self.assertEqual(fine,before)
        self.assertEqual([h['version'] for h in waiting[HISTORY_KEY]],[VERSION,FINE_VERSION])
        self.assertEqual(len({h['contract_fingerprint'] for h in waiting[HISTORY_KEY]}),2)
        self.assertTrue(all(h['audit']['full_checks_verified'] for h in waiting[HISTORY_KEY]))
        self.base.write_text(json.dumps(waiting))
        self.raw.update(version=RAISED_VERSION,contract_fingerprint='raised')
        self.audit.update(version=RAISED_AUDIT_VERSION,contract_fingerprint='raised')
        self.write();raised=prepare(self.base,self.study,self.output)
        self.assertTrue(raised[KEY]['audit']['full_checks_verified'])
        self.assertEqual(raised[HISTORY_KEY],waiting[HISTORY_KEY])
        self.base.write_text(json.dumps(raised));again=prepare(self.base,self.study,self.output)
        self.assertEqual(again[HISTORY_KEY],raised[HISTORY_KEY])
        self.assertEqual(again[KEY][HISTORY_KEY],raised[HISTORY_KEY])
        self.assertNotIn('q10',json.dumps(again[HISTORY_KEY]))

    def test_history_is_sanitized_and_cannot_inherit_an_incompatible_checker_badge(self):
        old=dict(self.raw,audit=dict(self.audit,version=FINE_AUDIT_VERSION,full_checks_verified=True,bound_files_verified=True),q10=[1]*6,token='never retained')
        base=dict(self.old,clean650_study=old,previous_clean650_studies=[old,old,dict(old,version='unknown')])
        result=merge_display(base,compact(dict(version=FINE_VERSION,status='running')))
        self.assertEqual(len(result[HISTORY_KEY]),1)
        history=result[HISTORY_KEY][0];self.assertFalse(history['audit']['full_checks_verified'])
        self.assertNotIn('q10',json.dumps(history));self.assertNotIn('never retained',json.dumps(history))

    def test_separate_output_and_bounded_watch_guards(self):
        self.assertIsNone(prepare(self.base,self.study,self.output)[KEY]);self.write()
        for output in (self.base,self.study/'display.json'):
            with self.assertRaises(ValueError):prepare(self.base,self.study,output)
        self.raw['counts']['attempts']=float('nan');self.write()
        with self.assertRaises(ValueError):prepare(self.base,self.study,self.output)
        for value in (-1,float('nan'),30001):
            with self.assertRaises(ValueError):watch(self.base,self.study,self.output,value)

if __name__=='__main__':unittest.main()
