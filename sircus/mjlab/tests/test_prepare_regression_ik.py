import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from sircus.mjlab import prepare_regression_ik as P


def write(path,value):path.write_text(json.dumps(value))

def row(i,release=True,capture=False):
    return dict(record_id=f'r{i}',context_id=f'c{i}',values=[.5,0,.4,25,.2],bucket_xy_m=[.7,0],native_release=release,capture=capture,invalid_physics=False,
        trace=dict(time_s=[j/100 for j in range(241)],position_m=[[.5+j/1000,0,.7-j/1000] for j in range(241)]) if release else None,
        events=[dict(category='board',time_s=.321)])


def diagnosis():
    d=dict(version=P.DIAGNOSIS,status='completed',source=dict(checks_count=283251,full_checks_verified=True,native_checks_verified=True,report_sha256='a'*64,audit_sha256='b'*64,contract_fingerprint='c'*64),
        losses=[dict(context_id=(P.CASES[i] if i<2 else f'c{i}'),parent_bc=row(i+100,capture=True),recovery_bc=row(i,release=False)) for i in range(5)],
        setup_rejections=[row(i,release=False) for i in range(16)],flight_misses=[row(i) for i in range(17)])
    d['diagnosis_fingerprint']=P.digest(d)
    return d


class Publication(unittest.TestCase):
    def setUp(self):
        mocked=patch.object(P,'comparison_recordings',return_value=[]);mocked.start();self.addCleanup(mocked.stop)
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.study=self.root/'study';self.study.mkdir();self.diag=self.root/'diagnosis.json';write(self.diag,diagnosis())
        self.html=self.root/'index.html';self.original='<html><body><div id="history">retained</div></body></html>';self.html.write_text(self.original)
        self.css=self.root/'style.css';self.style='/* historical */ body {color:white}\n';self.css.write_text(self.style)
        self.snapshot=self.root/'snapshot.json';self.audit=self.root/'audit.json'

    def compact(self):return P.compact(self.study,self.diag,audit_path=self.audit,now=123)

    def test_native_samples_are_subset_not_interpolated_and_stay_bounded(self):
        r=row(1);out=P.compact_row(r);self.assertLessEqual(len(out['trace']['time_s']),100)
        self.assertEqual(out['trace']['time_s'][0],0);self.assertEqual(out['trace']['time_s'][-1],2.4)
        for t,p in zip(out['trace']['time_s'],out['trace']['position_m']):self.assertEqual(p,r['trace']['position_m'][r['trace']['time_s'].index(t)])
        self.assertLess(len(P.encoded(self.compact()).encode()),P.MAX_SNAPSHOT_BYTES)
        r['native_release']=False
        with self.assertRaisesRegex(ValueError,'Unflown'):P.compact_row(r)

    def test_diagnosis_hash_native_audit_and_exact_groups_required(self):
        d=diagnosis();d['losses'].pop();write(self.diag,d)
        with self.assertRaises(ValueError):self.compact()
        d['diagnosis_fingerprint']=P.digest({k:v for k,v in d.items() if k!='diagnosis_fingerprint'});write(self.diag,d)
        with self.assertRaisesRegex(ValueError,'five-loss'):self.compact()
        d=diagnosis();d['source']['native_checks_verified']=False;d['diagnosis_fingerprint']=P.digest({k:v for k,v in d.items() if k!='diagnosis_fingerprint'});write(self.diag,d)
        with self.assertRaisesRegex(ValueError,'native audit'):self.compact()

    def completed(self):
        contract=dict(version=P.STUDY,protocol=dict(substeps=64,variants=list(P.VARIANTS)));contract['fingerprint']=P.digest(contract);write(self.study/'contract.json',contract)
        raw=dict(version=P.STUDY,status='completed',phase='completed',expected_requests=66,contract_fingerprint=contract['fingerprint'],rule=dict(diagnosis_sha256=P.sha(self.diag),mode='matched_parent_coordinate_blocks_and_interpolation',contexts=list(P.CASES)),counts=dict(attempts=66),cases=[],control_gates={'pre':{'status':'passed'},'post':{'status':'passed'}})
        raw['evidence_sha256']={};write(self.study/'evidence-manifest.json',{});write(self.study/'report.json',raw);write(self.audit,dict(evidence_manifest_sha256=P.sha(self.study/'evidence-manifest.json'),version=P.AUDIT,status='passed',full_checks_verified=True,native_checks_verified=True,failures=[],checks=[dict(passed=True)],checks_count=1,report_sha256=P.sha(self.study/'report.json'),contract_fingerprint=contract['fingerprint']))

    def test_audit_requires_current_bytes_all_checks_and_native_verification(self):
        self.completed();self.assertEqual(self.compact()['correction']['audit']['status'],'passed')
        p=self.study/'report.json';p.write_bytes(p.read_bytes()+b' ');self.assertEqual(self.compact()['correction']['audit']['status'],'unverified')
        self.completed();a=P.read(self.audit);a['native_checks_verified']=False;write(self.audit,a);self.assertEqual(self.compact()['correction']['audit']['status'],'unverified')

    def test_preserves_every_byte_outside_own_markers_and_escapes_html_css(self):
        raw=self.compact();raw['note']='*/</script><script>bad</script>'
        html=P.embed_html(self.original,raw);css=P.embed_css(self.style,raw)
        self.assertEqual(html.split(P.START)[0]+html.split(P.END)[1],self.original)
        self.assertEqual(css.split(P.CSS_START)[0]+css.split(P.CSS_END)[1],self.style)
        self.assertNotIn('*/</script>',html);self.assertNotIn('*/</script>',css)
        self.assertEqual(P.embed_html(html,raw),html);self.assertEqual(P.embed_css(css,raw),css)
        with self.assertRaisesRegex(ValueError,'Ambiguous'):P.embed_css(css+P.CSS_START,raw)
        with self.assertRaisesRegex(ValueError,'Reversed'):P.embed_html(P.END+P.START,raw)

    def test_refuses_changed_surface_or_source_target_before_writing(self):
        args=dict(snapshot_path=self.snapshot,target_html=self.html,target_css=self.css)
        with self.assertRaisesRegex(ValueError,'HTML changed'):P.refresh(self.study,self.diag,**args,expected_html_sha256='0'*64)
        self.assertFalse(self.snapshot.exists());self.assertEqual(self.css.read_text(),self.style)
        with self.assertRaisesRegex(ValueError,'CSS changed'):P.refresh(self.study,self.diag,**args,expected_css_sha256='0'*64)
        args['snapshot_path']=self.diag
        with self.assertRaisesRegex(ValueError,'immutable'):P.refresh(self.study,self.diag,**args)

    def test_filters_only_two_setup_losses_and_rejects_wrong_plan(self):
        out=self.compact()
        self.assertEqual([x['context_id'] for x in out['diagnosis']['losses']],list(P.CASES))
        self.assertNotIn('flight_misses',out['diagnosis']);self.assertNotIn('setup_rejections',out['diagnosis'])
        self.completed();raw=P.read(self.study/'report.json');raw['rule']['mode']='unrelated';write(self.study/'report.json',raw)
        with self.assertRaisesRegex(ValueError,'eight-variant'):self.compact()

    def test_preserves_prior_panel_and_embedded_video_byte_for_byte(self):
        old='<html><body><!--RECOVERY_REGRESSION_START--><video src="data:video/mp4;base64,AAAA"></video><!--RECOVERY_REGRESSION_END--></body></html>'
        new=P.embed_html(old,self.compact())
        self.assertEqual(new.split(P.START)[0]+new.split(P.END)[1],old)
        self.assertEqual(new.count('data:video/mp4;base64,AAAA'),1)

    def test_pending_display_does_not_invent_correction(self):
        out=P.refresh(self.study,self.diag,snapshot_path=self.snapshot,target_html=self.html,target_css=self.css,expected_html_sha256=P.sha(self.html),expected_css_sha256=P.sha(self.css),now=123)
        self.assertEqual(out['correction']['status'],'pending');self.assertEqual(out['correction']['cases'],[]);self.assertFalse(out['publication']['terminal'])
        self.assertEqual(P.read(self.snapshot),out)

    def test_compact_baseline_is_audit_bound_and_excludes_raw_traces(self):
        with self.assertRaisesRegex(ValueError,'verified'):P.baseline_summary(self.compact())
        self.completed();out=self.compact();out['correction']['recordings']=[dict(row(1,release=False),id='refused',variant='blend25',setup_proof=dict(valid_ik_branches=[0,8],reason='finite_ik_unresolved'))]
        summary=P.baseline_summary(out)
        self.assertNotIn('recordings',summary)
        self.assertEqual(summary['first_repeat_admissions'][0]['setup_proof']['valid_ik_branches'],[0,8])
        self.assertNotIn('trace',json.dumps(summary))
        out['correction']['counts']['attempts']=65
        with self.assertRaisesRegex(ValueError,'66'):P.baseline_summary(out)

    def test_actual_worker_plan_and_case_results_are_publication_compatible(self):
        # CPU schema smoke only. Synthetic rows are not a native evidence audit.
        from sircus.mjlab import regression_ik_study as S
        from sircus.mjlab.tests.test_regression_ik_verify import observed, fixtures
        _,_,plan,rows=observed()  # Uses the real N.make_plan, independently built inputs.
        self.assertEqual(len(plan),66)
        for record in rows:record['setup_proof']=dict(valid_ik_branches=[0,8] if not record['native_release'] else [6,8],reason=record['invalid'])
        contract=dict(version=P.STUDY,protocol=S.protocol());contract['fingerprint']=P.digest(contract);write(self.study/'contract.json',contract)
        rule=fixtures()[2];rule['diagnosis_sha256']=P.sha(self.diag)
        raw=dict(version=P.STUDY,status='completed',phase='completed',expected_requests=66,contract_fingerprint=contract['fingerprint'],rule=rule,
            counts=S.summarize(rows),counts_by_variant={v:S.summarize([r for r in rows if r['variant']==v]) for v in P.VARIANTS},
            cases=S.case_results(plan,rows),control_gates={'pre':{'status':'passed'},'post':{'status':'passed'}},evidence_sha256={})
        write(self.study/'evidence-manifest.json',{});write(self.study/'report.json',raw)
        write(self.audit,dict(evidence_manifest_sha256=P.sha(self.study/'evidence-manifest.json'),version=P.AUDIT,status='passed',full_checks_verified=True,native_checks_verified=True,failures=[],checks=[dict(passed=True,label='Synthetic schema fixture only')],checks_count=1,report_sha256=P.sha(self.study/'report.json'),contract_fingerprint=contract['fingerprint']))
        samples=[P.compact_row(r) for r in rows if r['phase']=='diagnostic' and r['repeat']==0]
        with patch.object(P,'comparison_recordings',return_value=samples):out=self.compact()
        self.assertEqual(out['correction']['counts']['attempts'],66)
        self.assertEqual(out['correction']['counts']['cached_refusal_requests'],6)
        self.assertEqual(len(out['correction']['recordings']),16)
        self.assertEqual(out['correction']['recordings'][1]['setup_proof']['valid_ik_branches'],[0,8])
        self.assertFalse(out['correction']['recordings'][1]['native_release'])
        for case,source in zip(out['correction']['cases'],raw['cases']):
            self.assertEqual(case['variant_values'],source['variant_values'])
            self.assertEqual(list(case['variants']),list(P.VARIANTS))
        self.assertEqual(len(P.baseline_summary(out)['first_repeat_admissions']),16)
        self.assertLess(len(P.encoded(out).encode()),P.MAX_SNAPSHOT_BYTES)

class RecordedSamples(unittest.TestCase):
    def test_all_sixteen_records_keep_declines_unflown_and_native_metadata(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);(root/'native').mkdir();write(root/'report.json',{'test':'fixture only'})
            members=[(context,variant) for context in P.CASES for variant in P.VARIANTS]
            times=np.arange(241)/15360;positions=np.zeros((241,16,3));positions[:,:,2]=.7
            positions[:,:,0]=np.arange(241)[:,None]/10000;forces=np.zeros((240,16,6));forces[60,:,0]=1
            path=root/'native'/'samples.npz';np.savez(path,time_s=times,position_m=positions,forces_N=forces,record_ids=np.asarray([f'r{i}' for i in range(16)]))
            rows=[dict(id=f'r{i}',phase='diagnostic',repeat=0,context_id=context,variant=variant,native_release=variant!='recovery_exact',route_success=False,adapter_directory='native',native_cache_file='samples.npz',native_cache_row=i,native_cache_sha256=P.sha(path),physics_steps=240,native_initial_ball_center_m=positions[0,i].tolist(),realization_comparison=dict(basis='matched_successful_parent',q14_equal_to_source=False,q14_max_abs_delta_rad=.1)) for i,(context,variant) in enumerate(members)]
            ledger=root/'rows.jsonl';ledger.write_text('\n'.join(json.dumps(row) for row in rows))
            raw=dict(evidence_sha256={'rows.jsonl':P.sha(ledger),'native/samples.npz':P.sha(path)})
            P._RECORDINGS_CACHE.clear();out=P.comparison_recordings(root,raw)
            self.assertEqual(len(out),16)
            for record in out:
                self.assertFalse(record['realization_comparison']['q14_equal_to_source'])
                self.assertEqual(record['realization_comparison']['basis'],'matched_successful_parent')
                if record['native_release']:
                    self.assertLess(len(record['trace']['time_s']),100)
                    for t,point in zip(record['trace']['time_s'],record['trace']['position_m']):self.assertEqual(point,positions[list(times).index(t),0].tolist())
                else:self.assertNotIn('trace',record)
            P._RECORDINGS_CACHE.clear();rows[0]['context_id']='foreign';ledger.write_text('\n'.join(json.dumps(row) for row in rows));raw['evidence_sha256']['rows.jsonl']=P.sha(ledger)
            with self.assertRaisesRegex(ValueError,'sixteen first-repeat'):P.comparison_recordings(root,raw)

if __name__=='__main__':unittest.main()
