import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from sircus.mjlab import prepare_clearance_horizon_report as P

H='a'*64
def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,allow_nan=False)+'\n')
def counts(group):
    native=sum(r['native_release'] for r in group);capture=sum(r['capture'] for r in group)
    return dict(requested=len(group),native_releases=native,captures=capture,clean=sum(r['clean'] for r in group),rejected=len(group)-native,native_misses=native-capture,invalid=0)
def native_rows():
    rows=[]
    for method in P.METHODS:
        for repeat in range(3):
            release=method=='short_window'
            rows.append(dict(record_id=f'{method}-{repeat}',context_id=P.CONTEXT,repeat=repeat,method=method,bucket_xy_m=[.5,-.2],values=[.6,-.1,.3,30.,.1],native_release=release,capture=False,clean=False,invalid=False,clearance_rejected=not release,robot_contact_before_terminal=True if release else None,robot_contact_after_terminal=False if release else None,table_contact_before_terminal=False if release else None,table_contact_after_terminal=True if release else None,bucket_retained=False if release else None))
    return rows

class HorizonPublisherTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.study=self.root/'study';self.audit=self.root/'audit.json';self.display=self.root/'interface.json'
        rows=native_rows();groups=[dict(method=m,counts=counts([r for r in rows if r['method']==m])) for m in P.METHODS];ctrl=dict(requested=6,native_releases=6,captures=3,clean=3,rejected=0,native_misses=3,invalid=0)
        total={k:sum(g['counts'][k] for g in groups)+2*ctrl[k] for k in P.COUNT_FIELDS}
        dump(self.study/'prediction-plan.json',dict(requests=list(range(18))))
        dump(self.study/'selection-plan.json',dict(context_id=P.CONTEXT))
        protocol=dict(methods=list(P.METHODS),substeps=64,new_training_requests=0,maximum_requests=18)
        spec=dict(version=P.STUDY_VERSION,methods=list(P.METHODS),context_id=P.CONTEXT,repeats=3,substeps=64,protocol=protocol,maximum_requests=18,training=False,noise=False,diagnostic_source_sha256={'study.py':H},native_source_sha256={'native.py':'b'*64},selection_plan_sha256=P.sha(self.study/'selection-plan.json'));spec['fingerprint']=P.digest(spec)
        dump(self.study/'contract.json',spec)
        manifest={p.name:P.sha(p) for p in self.study.iterdir()};dump(self.study/'evidence-manifest.json',manifest)
        windows=[dict(method=m,passed=m=='short_window',horizon_min_s=.32 if m=='short_window' else .39,horizon_max_s=.33 if m=='short_window' else .40,minimum_clearance_lower_bound_m=.012 if m=='short_window' else -.036,required_clearance_m=.002) for m in P.METHODS]
        raw=dict(version=P.STUDY_VERSION,status='completed',phase='completed',protocol=protocol,substeps=64,training=False,plan_sha256=P.sha(self.study/'prediction-plan.json'),contract_fingerprint=spec['fingerprint'],counts=total,comparisons=groups,rows=rows,windows=windows,control_gates={p:dict(status='passed',counts=ctrl) for p in ('pre','post')},evidence_sha256=manifest)
        dump(self.study/'report.json',raw)
        proof=dict(version=P.AUDIT_VERSION,status='passed',failures=[],checks_count=1,checks=[dict(label='Matched native evidence',passed=True)],report_sha256=P.sha(self.study/'report.json'),evidence_manifest_sha256=P.sha(self.study/'evidence-manifest.json'),plan_sha256=P.sha(self.study/'prediction-plan.json'),contract_fingerprint=spec['fingerprint'],requested_trials=18,training_requests=0,producer_source_sha256=spec['diagnostic_source_sha256'],native_source_sha256=spec['native_source_sha256'],auditor_sha256=H,comparisons=groups,rows=rows,windows=windows,control_gates=raw['control_gates'],counts=total,native_releases=total['native_releases'])
        dump(self.audit,proof);dump(self.display,dict(historical=dict(unchanged=True)))
        original=P.sha
        self.sha_patch=patch.object(P,'sha',side_effect=lambda path:H if Path(path).name=='horizon_window_verify.py' else original(path));self.sha_patch.start();self.addCleanup(self.sha_patch.stop)
    def compact(self):return P.compact(self.study,self.audit)
    def test_verified_scalar_publication_keeps_abstentions_and_unknown_contacts(self):
        out=self.compact();self.assertEqual(out['status'],'verified');self.assertEqual(len(out['rows']),6)
        self.assertEqual(out['counts']['requested'],18);self.assertEqual(out['comparisons'][0]['counts']['captures'],0);self.assertEqual(out['comparisons'][1]['counts']['rejected'],3)
        self.assertIn('not a capture',out['recommendation']);self.assertEqual(out['windows'][1]['minimum_clearance_lower_bound_m'],-.036)
        self.assertTrue(out['rows'][0]['table_contact_after_terminal']);self.assertIsNone(out['rows'][3]['robot_contact_before_terminal']);self.assertIsNone(out['rows'][3]['bucket_retained'])
    def test_pending_and_failed_audit_never_publish_partial_counts(self):
        out=P.compact(self.study,None);self.assertEqual(out['status'],'awaiting_audit');self.assertEqual(out['rows'],[]);self.assertEqual(out['windows'],[]);self.assertNotIn('counts',out)
        dump(self.audit,dict(status='failed'));out=self.compact();self.assertEqual(out['status'],'audit_failed');self.assertEqual(out['comparisons'],[])
        self.assertEqual(P.compact(None)['status'],'pending')
    def test_incomplete_or_unbound_native_proof_rejected(self):
        before=P.read(self.audit)
        changes=(lambda p:p.update(report_sha256=H),lambda p:p.update(requested_trials=17),lambda p:p.update(training_requests=1),lambda p:p.update(producer_source_sha256={}),lambda p:p.update(native_source_sha256={}),lambda p:p.update(auditor_sha256='f'*64),lambda p:p.update(checks_count=2),lambda p:p['checks'][0].update(passed=False),lambda p:p.update(failures=['bad']),lambda p:p['windows'][1].update(passed=True))
        for change in changes:
            bad=copy.deepcopy(before);change(bad);dump(self.audit,bad)
            with self.subTest(change=change):
                with self.assertRaises(ValueError):self.compact()
        dump(self.audit,before)
    def test_mutated_or_unmanifested_evidence_rejected(self):
        dump(self.study/'unexpected.json',dict(extra=True))
        with self.assertRaises(ValueError):self.compact()
        (self.study/'unexpected.json').unlink();dump(self.study/'prediction-plan.json',dict(requests=[]))
        with self.assertRaises(ValueError):self.compact()
    def test_native_rows_cannot_drop_abstentions_or_relabel_observations(self):
        raw=P.read(self.study/'report.json');proof=P.read(self.audit)
        changes=(lambda p:p['rows'].pop(),lambda p:p['rows'][1].update(record_id=p['rows'][0]['record_id']),lambda p:p['rows'][3].update(robot_contact_before_terminal=False),lambda p:p['rows'][3].update(bucket_retained=False),lambda p:p['rows'][3]['values'].__setitem__(4,.9),lambda p:p['control_gates']['post'].update(status='failed'),lambda p:p['comparisons'][0]['counts'].update(captures=1))
        for change in changes:
            bad=copy.deepcopy(proof);change(bad)
            with self.subTest(change=change):
                with self.assertRaises(ValueError):P.validate_native(raw,bad)
    def test_unknown_released_observations_remain_unknown(self):
        raw=P.read(self.study/'report.json');proof=P.read(self.audit)
        for r in (raw['rows'][0],proof['rows'][0]):r.update(robot_contact_before_terminal=None,bucket_retained=None)
        rows=P.validate_native(raw,proof);self.assertIsNone(rows[0]['robot_contact_before_terminal']);self.assertIsNone(rows[0]['bucket_retained'])
    def test_publish_preserves_history_and_evidence_and_marks_pipeline_failure(self):
        before=(self.study/'report.json').read_bytes();out=P.publish(self.display,self.study,self.audit);doc=P.read(self.display)
        self.assertEqual(doc['historical'],dict(unchanged=True));self.assertEqual(doc[P.KEY],out);self.assertEqual(before,(self.study/'report.json').read_bytes())
        out=P.publish(self.display,self.study,self.audit,pipeline=dict(status='failed'));self.assertEqual(out['status'],'failed');self.assertEqual(out['rows'],[]);self.assertEqual(out['windows'],[])
        with self.assertRaises(ValueError):P.publish(self.study/'display.json',self.study,self.audit)
        with self.assertRaises(ValueError):P.publish(self.audit,self.study,self.audit)

if __name__=='__main__':unittest.main()
