import copy
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from sircus.mjlab import prepare_fixed35_recovery as P


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,allow_nan=False))


class RecoveryPublisherTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.study=self.root/'study';self.study.mkdir();self.report=self.study/'report.json';self.audit=self.root/'audit.json';self.pipeline=self.root/'pipeline.json'
        public=json.loads((Path(P.__file__).parents[2]/'baselines/mjlab-fixed35-imitation-20261109/summary.json').read_text())
        parent=[]
        for r in public['test_positions']:
            if r['method']!='current_bc':continue
            item=copy.deepcopy(r);item['id']=item.pop('record_id');item['route_success']=item.pop('capture');parent.append(item)
        self.preserved=[r for r in parent if P.captured(r)]
        failures=[r for r in parent if not P.captured(r)]
        self.blocked=[r for r in failures if r.get('invalid')=='fixed_base_clearance_rejected']
        self.cases=[r for r in failures if r not in self.blocked]
        self.protocol=dict(substeps=64,maximum_requests=843,noise=False,bucket_perturbations=False,training=False,grid_positions=35,parent_captures=24,development_failures=11,fixed_base_skips=2,search_layouts=9,repeat_per_selected=3,candidates_per_layout=3,preserved_requests=24,control_requests=18)
        self.spec=dict(version=P.VERSION,protocol=self.protocol);self.spec['fingerprint']=P.digest(self.spec)
        self.rows=[]
        for phase in ('controls_pre','controls_post'):
            for i in range(9):self.rows.append(dict(id=f'{phase}-{i}',context_id=f'control-{i}',phase=phase,native_release=True,route_success=i<6,values=[.6,-.2,.4,28.,.1],bucket_xy_m=[1.,-.2],invalid_physics=False))
        for r in self.preserved:
            row=copy.deepcopy(r);row.update(id='preserved-'+r['id'],phase='preserved');self.rows.append(row)
        discoveries=[]
        for i,r in enumerate(self.cases):
            row=copy.deepcopy(r);row.update(id=f'search-{i}',phase='search',route_success=i<2,native_release=True,invalid=None,invalid_physics=False);self.rows.append(row);discoveries.append(row)
        for selected_index,selected in enumerate(discoveries[:2]):
            for repeat in range(3):
                row=copy.deepcopy(selected);row.update(id=f'repeat-{selected_index}-{repeat}',phase='repeat',repeat=repeat,confirmation_index=0,selected_from_id=selected['id'],selected_from_sha256=P.digest(selected),route_success=selected_index==0 or repeat<2);self.rows.append(row)
        selected=copy.deepcopy(discoveries[1]);selected.update(id='search-1-second',values=list(selected['values']));selected['values'][4]+=.01;self.rows.append(selected)
        for repeat in range(3):
            row=copy.deepcopy(selected);row.update(id=f'repeat-second-{repeat}',phase='repeat',repeat=repeat,confirmation_index=1,selected_from_id=selected['id'],selected_from_sha256=P.digest(selected),route_success=True);self.rows.append(row)
        write(self.study/'contract.json',self.spec)
        for name,rows in [('cases',self.cases),('preserved',self.preserved),('blocked',self.blocked)]:write(self.study/f'bootstrap/{name}.json',rows)
        write(self.study/'diagnoses.json',[dict(context_id=r['context_id'],category='fixed_base_blocked' if r in self.blocked else 'no_bucket_contact') for r in failures])
        self.save_rows()
        self.raw=dict(version=P.VERSION,status='completed',phase='completed',protocol=self.protocol,contract_fingerprint=self.spec['fingerprint'],expected_requests=843,counts=P.statistics(self.rows),budget_truncated=False,completion_reason='cases_completed',control_gates={k:dict(status='passed') for k in ('pre','post')},cases=[dict(context_id=r['context_id'],status='confirmed' if r in self.cases[:2] else 'search_exhausted') for r in failures])
        self.bind()
        write(self.pipeline,dict(status='completed',finished_unix_s=100))
        self.html=self.root/'index.html';self.css=self.root/'style.css';self.snapshot=self.root/'snapshot.json'
        self.html.write_text('<html><body><!--OLD-->history<video src="old.mp4"></video></body></html>');self.css.write_text('body{color:white}/*historical*/')

    def save_rows(self):
        (self.study/'rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in self.rows))

    def bind(self):
        self.manifest={str(p.relative_to(self.study)):P.sha(p) for p in self.study.rglob('*') if p.is_file() and p.name not in ('report.json','evidence-manifest.json')}
        self.raw['evidence_sha256']=self.manifest;write(self.study/'evidence-manifest.json',self.manifest);write(self.report,self.raw)
        write(self.audit,dict(version=P.AUDIT,status='passed',full_checks_verified=True,native_checks_verified=True,checks_count=1,checks=[dict(passed=True)],failures=[],report_sha256=P.sha(self.report),contract_fingerprint=self.spec['fingerprint'],evidence_manifest_sha256=P.sha(self.study/'evidence-manifest.json')))

    def compact(self):return P.compact(self.study,audit_path=self.audit,pipeline_path=self.pipeline,now=150)

    def test_recovery_groups_include_failed_confirmations_and_separate_preservation(self):
        s=self.compact();self.assertEqual(len(s['cases']),11);self.assertEqual(s['confirmed_layouts'],2);self.assertEqual(sum(c['blocked'] for c in s['cases']),2)
        case=next(c for c in s['cases'] if c['context_id']==self.cases[1]['context_id']);self.assertEqual(len(case['confirmation_groups']),2);self.assertEqual(sum(r['capture'] for r in case['confirmation_groups'][0]['repeats']),2)
        self.assertEqual(len(s['preserved_recipes']),24);self.assertEqual(s['preserved_replay_counts']['captures'],24);self.assertEqual(s['expected_requests'],len(self.rows));self.assertEqual(s['request_cap'],843)
        self.assertEqual(P.baseline_summary(s)['confirmed_layouts'],2);self.assertNotIn('publication',P.baseline_summary(s));self.assertLess(len(json.dumps(s)),P.MAX_SNAPSHOT_BYTES)

    def test_no_outcomes_before_bound_native_audit(self):
        self.audit.unlink();s=self.compact();self.assertEqual(s['status'],'awaiting_audit');self.assertEqual(s['cases'],[]);self.assertEqual(s['preserved_replays'],[]);self.assertEqual(s['counts'],{'attempts':len(self.rows)})
        with self.assertRaises(ValueError):P.baseline_summary(s)

    def test_corrupt_trace_ledger_or_manifest_is_not_accepted(self):
        with (self.study/'rows.jsonl').open('a') as f:f.write(' ')
        with self.assertRaisesRegex(ValueError,'hash differs'):self.compact()
        self.save_rows();self.bind();write(self.study/'evidence-manifest.json',{});self.assertEqual(self.compact()['audit']['status'],'unverified')

    def test_all_three_matching_repeats_are_required(self):
        for edit in (lambda rows:rows.pop(next(i for i,r in enumerate(rows) if r.get('phase')=='repeat')),lambda rows:next(r for r in rows if r.get('phase')=='repeat').update(values=[1,2,3,4,5]),lambda rows:next(r for r in rows if r.get('phase')=='repeat').update(selected_from_sha256='bad')):
            rows=copy.deepcopy(self.rows);edit(rows)
            with self.assertRaises(ValueError):P.build_cases(self.cases+self.blocked,self.preserved,self.blocked,rows)

    def test_no_blocked_layout_search_and_no_changed_bucket(self):
        rows=copy.deepcopy(self.rows);r=copy.deepcopy(self.blocked[0]);r.update(id='forbidden-search',phase='search');rows.append(r)
        with self.assertRaisesRegex(ValueError,'Blocked'):P.build_cases(self.cases+self.blocked,self.preserved,self.blocked,rows)
        rows=copy.deepcopy(self.rows);next(r for r in rows if r['phase']=='search')['bucket_xy_m']=[9,9]
        with self.assertRaisesRegex(ValueError,'displacement'):P.build_cases(self.cases+self.blocked,self.preserved,self.blocked,rows)

    def test_preserved_recipe_changes_are_rejected(self):
        rows=copy.deepcopy(self.rows);next(r for r in rows if r['phase']=='preserved')['values']=[1,2,3,4,5]
        with self.assertRaisesRegex(ValueError,'exact original'):P.build_cases(self.cases+self.blocked,self.preserved,self.blocked,rows)

    def test_invalid_discovery_never_qualifies_for_repeats(self):
        rows=copy.deepcopy(self.rows);next(r for r in rows if r['id']=='search-0')['invalid_physics']=True
        with self.assertRaisesRegex(ValueError,'captured discovery'):P.build_cases(self.cases+self.blocked,self.preserved,self.blocked,rows)

    def test_false_audit_control_or_protocol_never_completes(self):
        self.raw['control_gates']['post']['status']='failed';self.bind()
        with self.assertRaisesRegex(ValueError,'control'):self.compact()
        self.raw['control_gates']['post']['status']='passed';self.bind();a=P.read(self.audit);a['native_checks_verified']=False;write(self.audit,a);self.assertEqual(self.compact()['cases'],[])
        self.bind();self.raw['protocol']=dict(self.protocol,substeps=128);write(self.report,self.raw)
        with self.assertRaises(ValueError):self.compact()

    def test_markup_is_additive_escaped_and_idempotent(self):
        s=self.compact();s['phase']='</script>*/<script>bad()</script>'
        oldh,oldc=self.html.read_text(),self.css.read_text();h=P.embed_html(oldh,s);c=P.embed_css(oldc,s)
        self.assertIn('old.mp4',h);self.assertNotIn('<script>bad()',h);self.assertNotIn('*/<script>',c)
        self.assertEqual(P.embed_html(h,s),h);self.assertEqual(P.embed_css(c,s),c)
        self.assertEqual(h[:h.index(P.START)]+h[h.index(P.END)+len(P.END):],oldh)
        self.assertEqual(c[:c.index(P.CSS_START)]+c[c.index(P.CSS_END)+len(P.CSS_END):],oldc)

    def test_refresh_rejects_concurrent_edits_foreign_snapshot_and_input_paths(self):
        args=dict(snapshot_path=self.snapshot,target_html=self.html,target_css=self.css,audit_path=self.audit,pipeline_path=self.pipeline)
        with self.assertRaisesRegex(ValueError,'HTML changed'):P.refresh(self.study,**args,expected_html_sha256='0'*64)
        out=P.refresh(self.study,**args,expected_html_sha256=P.sha(self.html),expected_css_sha256=P.sha(self.css));self.assertEqual(out['audit']['status'],'passed')
        write(self.snapshot,dict(version='foreign'))
        with self.assertRaisesRegex(ValueError,'another study'):P.refresh(self.study,**args)
        with self.assertRaises(ValueError):P.guard_targets(self.study,(self.audit,),(self.report,))

    def test_pending_and_stopped_progress_remains_explicit(self):
        self.report.unlink();write(self.pipeline,dict(status='running'));self.assertEqual(self.compact()['status'],'running')
        write(self.pipeline,dict(status='failed',finished_unix_s=100));out=self.compact();self.assertEqual(out['status'],'stopped');self.assertEqual(out['cases'],[])

    @unittest.skipUnless(shutil.which('node'),'Browser schema interoperability requiresNode')
    def test_browser_verifies_actual_compact(self):
        uri=(Path(P.__file__).parent/'web/fixed35-recovery.js').resolve().as_uri()
        code='import {verified} from '+json.dumps(uri)+';let text="";for await(const chunk of process.stdin)text+=chunk;if(!verified(JSON.parse(text)))process.exit(2);'
        r=subprocess.run(['node','--input-type=module','--eval',code],input=json.dumps(self.compact()),capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stderr)

if __name__=='__main__':unittest.main()
