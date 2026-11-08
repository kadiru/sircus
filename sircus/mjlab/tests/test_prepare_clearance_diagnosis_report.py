import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from sircus.mjlab import prepare_clearance_diagnosis_report as P

H='a'*64

def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,allow_nan=False)+'\n')

def counts(group):
    native=sum(r['native_release'] for r in group);capture=sum(r['capture'] for r in group)
    return dict(requested=len(group),native_releases=native,captures=capture,clean=sum(r['clean'] for r in group),rejected=len(group)-native,native_misses=native-capture,invalid=0)

def native_rows():
    rows=[]
    for method in P.METHODS:
        for case in range(6):
            for repeat in range(3):
                release=case in (1,5) if method=='original' else case!=0;capture=method=='corrected' and 2<=case<=4
                rows.append(dict(record_id=f'{method}-{case}-{repeat}',context_id=f'case{case}',repeat=repeat,method=method,bucket_xy_m=[.5+case*.05,-.2],values=[.6,-.1,.3,30.,.1],native_release=release,capture=capture,clean=capture and case==3,invalid=False,clearance_rejected=not release,robot_contact_before_terminal=not capture if release else None,robot_contact_after_terminal=False if release else None))
    return rows

class PublisherTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.dr=self.root/'diagnosis/report.json';self.da=self.root/'diagnosis-audit.json';self.study=self.root/'native-study';self.na=self.root/'native-audit.json';self.display=self.root/'interface.json'
        raw=dict(version=P.DIAGNOSIS_VERSION,status='completed',source=dict(audit_sha256=P.PARENT_AUDIT_SHA256,diagnostic_source_sha256=P.sha(Path(P.__file__).with_name('clearance_prediction_diagnosis.py'))),finger_cases=[dict(context_id='case07',native_capture=True,native_robot_contact=False,old_lower_bound_m=-.0036,corrected_lower_bound_m=.004541,observed_first_flight_sample_clearance_m=.017086,observed_distance_scope='Saved first-flight centers with held joints; not continuous measured clearance',closest_geom='Arm10/finger',diagnosis='Conservative corner bound')],horizon_cases=[dict(context_id='case01',predicted_horizon_min_s=.321,predicted_horizon_max_s=.327,observed_first_contact_s=.335,first_contact_channels=['robot'],observed_robot_contact_s=.335,diagnosis='After predicted horizon')])
        dump(self.dr,raw);dump(self.da,dict(version=P.DIAGNOSIS_AUDIT_VERSION,status='passed',failures=[],checks_count=1,checks=[dict(label='Exact derivative evidence',passed=True)],report_sha256=P.sha(self.dr),source_audit_sha256=P.PARENT_AUDIT_SHA256,auditor_source_sha256=raw['source']['diagnostic_source_sha256']))
        rows=native_rows();groups=[dict(method=m,counts=counts([r for r in rows if r['method']==m])) for m in P.METHODS];ctrl=dict(requested=6,native_releases=6,captures=3,clean=3,rejected=0,native_misses=3,invalid=0)
        total={k:sum(g['counts'][k] for g in groups)+2*ctrl[k] for k in P.COUNT_FIELDS}
        dump(self.study/'prediction-plan.json',dict(requests=list(range(48))))
        dump(self.study/'selection-plan.json',dict(contexts=list(P.CONTEXTS)))
        protocol=dict(methods=list(P.METHODS),substeps=64,new_training_requests=0,maximum_requests=48)
        spec=dict(version=P.STUDY_VERSION,methods=list(P.METHODS),contexts=list(P.CONTEXTS),repeats=3,substeps=64,protocol=protocol,maximum_requests=48,training=False,noise=False,diagnostic_source_sha256={'study.py':H},native_source_sha256={'native.py':'b'*64},selection_plan_sha256=P.sha(self.study/'selection-plan.json'));spec['fingerprint']=P.digest(spec)
        dump(self.study/'contract.json',spec)
        manifest={p.name:P.sha(p) for p in self.study.iterdir()};dump(self.study/'evidence-manifest.json',manifest)
        raw=dict(version=P.STUDY_VERSION,status='completed',phase='completed',protocol=protocol,substeps=64,training=False,plan_sha256=P.sha(self.study/'prediction-plan.json'),contract_fingerprint=spec['fingerprint'],counts=total,comparisons=groups,rows=rows,control_gates={p:dict(status='passed',counts=ctrl) for p in ('pre','post')},evidence_sha256=manifest)
        dump(self.study/'report.json',raw)
        proof=dict(version=P.AUDIT_VERSION,status='passed',failures=[],checks_count=1,checks=[dict(label='Matched native evidence',passed=True)],report_sha256=P.sha(self.study/'report.json'),evidence_manifest_sha256=P.sha(self.study/'evidence-manifest.json'),prediction_plan_sha256=P.sha(self.study/'prediction-plan.json'),contract_fingerprint=spec['fingerprint'],requested_trials=48,training_requests=0,producer_source_sha256=spec['diagnostic_source_sha256'],native_source_sha256=spec['native_source_sha256'],auditor_sha256=H,comparisons=groups,rows=rows,control_gates=raw['control_gates'],counts=total,source_parent=dict(audit_sha256=P.PARENT_AUDIT_SHA256),native_releases=total['native_releases'])
        dump(self.na,proof);dump(self.display,dict(historical=dict(unchanged=True)))
        original=P.sha
        self.sha_patch=patch.object(P,'sha',side_effect=lambda path:H if Path(path).name=='clearance_correction_verify.py' else original(path));self.sha_patch.start();self.addCleanup(self.sha_patch.stop)
    def compact(self):return P.compact(self.dr,self.da,self.study,self.na)
    def test_verified_combined_scalar_publication_and_exact_denominators(self):
        out=self.compact();self.assertEqual(out['status'],'verified');self.assertEqual(len(out['correction']['rows']),36)
        self.assertEqual(out['correction']['counts']['requested'],48);self.assertEqual(out['correction']['comparisons'][0]['counts']['captures'],0);self.assertEqual(out['correction']['comparisons'][1]['counts']['captures'],9)
        self.assertIn('new frozen held-out',out['recommendation']);self.assertFalse('source_hint' in json.dumps(out));self.assertEqual(out['diagnosis']['horizon_cases'][0]['robot_contact_delay_s'],.335);self.assertEqual(out['diagnosis']['finger_cases'][0]['observed_first_flight_sample_clearance_m'],.017086)
    def test_diagnosis_authentication_rejects_changed_source_or_partial_proof(self):
        for modify in (lambda p:p.update(source_audit_sha256=H),lambda p:p.update(checks_count=2),lambda p:p['checks'][0].update(passed=False),lambda p:p.update(report_sha256=H),lambda p:p.update(auditor_source_sha256=H)):
            before=P.read(self.da);bad=copy.deepcopy(before);modify(bad);dump(self.da,bad)
            with self.assertRaises(ValueError):P.diagnosis_compact(self.dr,self.da)
            dump(self.da,before)
    def test_pending_and_failed_native_audit_do_not_publish_interim_results(self):
        value=P.compact(self.dr,self.da,self.study,None);self.assertEqual(value['diagnosis']['status'],'verified');self.assertEqual(value['correction']['status'],'awaiting_audit');self.assertEqual(value['correction']['rows'],[])
        dump(self.na,dict(status='failed'));value=self.compact();self.assertEqual(value['correction']['status'],'audit_failed');self.assertEqual(value['correction']['comparisons'],[])
    def test_native_proof_and_frozen_bytes_cannot_be_replaced(self):
        before=P.read(self.na)
        for modify in (lambda p:p.update(report_sha256=H),lambda p:p.update(requested_trials=47),lambda p:p.update(training_requests=1),lambda p:p.update(producer_source_sha256={}),lambda p:p.update(native_source_sha256={}),lambda p:p.update(auditor_sha256='f'*64)):
            bad=copy.deepcopy(before);modify(bad);dump(self.na,bad)
            with self.assertRaises(ValueError):self.compact()
        dump(self.na,before);dump(self.study/'prediction-plan.json',dict(requests=[]))
        with self.assertRaises(ValueError):self.compact()
    def test_native_accounting_cannot_drop_abstentions_or_relabel_measurements(self):
        raw=P.read(self.study/'report.json');proof=P.read(self.na)
        for change in (lambda p:p['rows'].pop(),lambda p:p['rows'][1].update(record_id=p['rows'][0]['record_id']),lambda p:p['rows'][0].update(robot_contact_before_terminal=False),lambda p:p['rows'][18]['bucket_xy_m'].__setitem__(0,.9),lambda p:p['control_gates']['post'].update(status='failed'),lambda p:p['comparisons'][0]['counts'].update(captures=1)):
            bad=copy.deepcopy(proof);change(bad)
            with self.assertRaises(ValueError):P.validate_native(raw,bad)
    def test_unknown_contact_and_link_name_remain_unknown(self):
        d=P.read(self.dr);d['finger_cases'][0].update(closest_geom=None,native_robot_contact=None);dump(self.dr,d);a=P.read(self.da);a['report_sha256']=P.sha(self.dr);dump(self.da,a)
        result=P.diagnosis_compact(self.dr,self.da);self.assertIsNone(result['finger_cases'][0]['native_robot_contact']);self.assertIsNone(result['finger_cases'][0]['closest_geom'])
    def test_publish_preserves_existing_data_and_evidence_and_marks_pipeline_failure(self):
        old=self.dr.read_bytes();result=P.publish(self.display,self.dr,self.da,self.study,self.na);display=P.read(self.display)
        self.assertEqual(display['historical'],dict(unchanged=True));self.assertEqual(display[P.KEY],result);self.assertEqual(old,self.dr.read_bytes())
        result=P.publish(self.display,self.dr,self.da,self.study,self.na,pipeline=dict(status='failed'));self.assertEqual(result['correction']['status'],'failed');self.assertEqual(result['correction']['rows'],[])
        with self.assertRaises(ValueError):P.publish(self.study/'display.json',self.dr,self.da,self.study,self.na)
        with self.assertRaises(ValueError):P.publish(self.dr,self.dr,self.da,self.study,self.na)
    def test_incomplete_diagnosis_cannot_authorize_verified_correction(self):
        with self.assertRaises(ValueError):P.compact(self.dr,None,self.study,self.na)

if __name__=='__main__':unittest.main()
