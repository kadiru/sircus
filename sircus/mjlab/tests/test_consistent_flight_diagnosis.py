"""Synthetic audit/trajectory controls; never invoke engine or CUDA."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from sircus.mjlab import consistent_flight_diagnosis as d
from sircus.mjlab.tests.test_calibrated_score import config

ROOT=Path(__file__).resolve().parents[3]


def write(path,value):path.write_text(json.dumps(value,sort_keys=True,allow_nan=False)+'\n')

def fixture(root):
    study=root/'study';study.mkdir();audit=root/'audit.json'
    spec=dict(version='sircus-consistent-flight-v1',source_sha256={'source.py':'a'*64},native_source_sha256={'native.py':'b'*64},
        protocol=dict(substeps=64,methods=list(d.METHODS),test_layouts=30,maximum_fresh_requests=132,control_requests=12,
                      new_train_requests=0,noise=False,no_test_flight_search=True,model_weights_unchanged=True,candidates=16,maximum_geometry_checks=4))
    spec['fingerprint']=d.digest(spec);write(study/'contract.json',spec)
    rows=[]
    def add(role,phase,method,context,xy):
        success=role=='positive_control'
        rows.append(dict(id='row-'+str(len(rows)),role=role,phase=phase,method=method,context_id=context,bucket_xy_m=xy,
                         values=[.5,-.2,.3,30.,.2],native_release=True,route_success=success,route_quality=1 if success else 0,
                         invalid=None,invalid_physics=False,whole_ball_entry=success,scalar_score_audit_passed=True,action_substituted=False))
    for phase in ('controls_pre','controls_post'):
        if phase=='controls_post':
            for m in d.METHODS:
                for i in range(30):add('test','test',m,'context-'+str(i),[.2+i*.01,-.3])
        for role in ('positive_control','negative_control'):
            for i in range(3):add(role,phase,'control',role,[.8,-.2])
    packet=dict(frozen_before_any_native_outcome=True,native_planning_queries=0,frozen_unix_s=5.,
                requests=[{k:r[k] for k in ('id','role','phase','method','context_id','bucket_xy_m','values')} for r in rows])
    write(study/'prediction-plan.json',packet)
    (study/'records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    comparisons=[dict(method=m,counts=d.statistics([r for r in rows if r['role']=='test' and r['method']==m])) for m in d.METHODS]
    report=dict(version=spec['version'],status='completed',phase='completed',contract_fingerprint=spec['fingerprint'],
        source_sha256=spec['source_sha256'],protocol=spec['protocol'],counts=d.statistics(rows),comparisons=comparisons,
        acquired_unix_s=6.,control_gates={k:d.control_gate([r for r in rows if r['phase']==phase]) for k,phase in [('pre','controls_pre'),('post','controls_post')]})
    proof=dict(version='sircus-consistent-flight-audit-v1',status='passed',failures=[],checks=[dict(label='synthetic fixture only',passed=True)],
        checks_count=1,contract_fingerprint=spec['fingerprint'],requested_trials=132,training_requests=0,native_releases=132,
        producer_source_sha256=spec['source_sha256'],native_source_sha256=spec['native_source_sha256'],comparisons=comparisons)
    write(study/'report.json',report);write(audit,proof)
    rebind(study,audit)
    return study,audit


def rebind(study,audit):
    # Explicitly synthetic fake trusted signatures: isolate verifier invariants.
    manifest={str(p.relative_to(study)):d.sha(p) for p in study.rglob('*') if p.is_file() and p.name not in ('report.json','evidence-manifest.json')}
    write(study/'evidence-manifest.json',manifest)
    report=d.read(study/'report.json');report['evidence_sha256']=manifest;report['prediction_plan_sha256']=d.sha(study/'prediction-plan.json');write(study/'report.json',report)
    proof=d.read(audit);proof.update(report_sha256=d.sha(study/'report.json'),evidence_manifest_sha256=d.sha(study/'evidence-manifest.json'));write(audit,proof)


def pins(study,audit):return dict(report=d.sha(study/'report.json'),manifest=d.sha(study/'evidence-manifest.json'),audit=d.sha(audit))


class AuditBindingTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.study,self.audit=fixture(self.root)
    def tearDown(self):self.temp.cleanup()
    def verify(self):
        with patch.object(d,'SOURCE_HASHES',pins(self.study,self.audit)):
            return d.verify_inputs(self.study,self.audit)
    def test_valid_synthetic_bound_fixture_returns_seven_elements(self):
        found=self.verify();self.assertEqual(len(found),7);self.assertEqual(len(found[-1]),132)
        self.assertEqual(found[-2]['requests'][0]['id'],'row-0')
    def test_actual_source_pins_reject_synthetic_proof(self):
        with self.assertRaises(ValueError):d.verify_inputs(self.study,self.audit)
    def test_changed_record_bytes_fail_bound_manifest(self):
        p=self.study/'records.jsonl';p.write_text(p.read_text()+'\n')
        with self.assertRaises(ValueError):self.verify()
    def test_changed_dense_cache_bytes_fail_even_when_json_is_unchanged(self):
        native=self.study/'native';native.mkdir()
        path=native/'dense.npz';np.savez(path,position_m=np.zeros((3,1,3)))
        rebind(self.study,self.audit)
        self.verify()
        np.savez(path,position_m=np.ones((3,1,3)))
        with self.assertRaises(ValueError):self.verify()
    def test_boolean_like_pass_does_not_replace_explicit_true(self):
        proof=d.read(self.audit);proof['checks'][0]['passed']=1;write(self.audit,proof)
        with self.assertRaises(ValueError):self.verify()
    def test_empty_or_incomplete_audit_is_rejected(self):
        for entries,count in [([],0),([dict(label='test',passed=True)],2)]:
            proof=d.read(self.audit);proof.update(checks=entries,checks_count=count);write(self.audit,proof)
            with self.subTest(count=count),self.assertRaises(ValueError):self.verify()
    def test_extra_unmanifested_file_cannot_be_ignored(self):
        (self.study/'later-edit.json').write_text('{}')
        with self.assertRaises(ValueError):self.verify()
    def test_plan_after_acquisition_rejected_even_with_rebound_hashes(self):
        packet=d.read(self.study/'prediction-plan.json');packet['frozen_unix_s']=8.;write(self.study/'prediction-plan.json',packet);rebind(self.study,self.audit)
        with self.assertRaises(ValueError):self.verify()
    def test_changed_executed_action_not_hidden_by_new_manifest(self):
        p=self.study/'records.jsonl';rows=[json.loads(l) for l in p.read_text().splitlines()];rows[10]['values'][4]+=.01
        p.write_text(''.join(json.dumps(r)+'\n' for r in rows));rebind(self.study,self.audit)
        with self.assertRaises(ValueError):self.verify()
    def test_unsafe_manifest_reference_rejected(self):
        p=self.study/'evidence-manifest.json';m=d.read(p);m['../escape']='c'*64;write(p,m)
        proof=d.read(self.audit);proof['evidence_manifest_sha256']=d.sha(p);write(self.audit,proof)
        with self.assertRaises(ValueError):self.verify()
    def test_build_does_not_overwrite_source_or_existing_outputs(self):
        for path in (self.study/'new-diagnosis',self.study):
            with self.subTest(path=path),self.assertRaises(ValueError):d.build(self.study,self.audit,path)


class TrajectoryDiagnosisTests(unittest.TestCase):
    def setUp(self):
        self.scene=json.loads((ROOT/'sircus/isaac/scene.json').read_text());self.cfg=config(self.scene)
        values=[.5,-.2,.3,30.,.2];xy=[.9,-.2]
        self.launch=d.nominal_origin(values,xy,d.geometry(self.scene,self.cfg))
        self.row=dict(id='synthetic',values=values,bucket_xy_m=xy,board_normal=self.launch['board_normal'],
            native_release=True,invalid_physics=False,invalid=None,native_cache_sha256='a'*64,
            model_prediction=dict(rebound_velocity_m_s=[1.1,.02,.45]))
        self.times=np.arange(6)*.01;start=1
        self.origin=np.array(self.launch['origin_m'])+[-.001,0.,.0006]
        self.velocity=np.array([1.,0.,.5]);g=np.array([0.,0.,-9.81]);t=self.times-self.times[start]
        self.arrays=dict(position_m=self.origin+t[:,None]*self.velocity+.5*t[:,None]**2*g,
                         velocity_m_s=self.velocity+t[:,None]*g,time_s=self.times,forces_N=np.zeros((5,6)))
        self.metric=dict(available=True,reason=None,start_index=1,end_index=4,start_time_s=.01,end_time_s=.04,
                         cutoff_reason='contact',cutoff_contact_channels=['bucket'],first_crossing=None,mouth_clearance_m=None,
                         closest_opening_distance_m=.1,sample_count=4)
    def analyze(self,metric=None):
        with patch.object(d,'first_flight',return_value=metric or self.metric):
            return d.analyze_trace(self.row,self.scene,self.cfg,self.arrays,[0.,0.,-9.81])
    def test_nominal_origin_observed_state_and_prediction_are_distinct(self):
        before=copy.deepcopy(self.row);r=self.analyze();o=r['observed']
        np.testing.assert_allclose(o['nominal_origin_minus_observed_rebound_mm'],[1.,0.,-.6],atol=1e-10)
        np.testing.assert_allclose(o['rebound_velocity_m_s'],[1.,0.,.5],atol=1e-10)
        np.testing.assert_allclose(o['velocity_error_m_s'],[.1,.02,-.05],atol=1e-10)
        self.assertLess(o['observed_velocity_ballistic_error_mm'],1e-9)
        self.assertGreater(o['velocity_head_endpoint_error_mm'],1.)
        self.assertIsNone(o['rim_crossing']);self.assertIsNone(o['mouth_clearance_m'])
        self.assertEqual(self.row,before)
    def test_curves_end_before_later_contact_and_have_literal_provenance(self):
        r=self.analyze();self.assertEqual(len(r['trace']['time_s']),4)
        self.assertAlmostEqual(r['trace']['time_s'][-1],.03)
        np.testing.assert_array_equal(r['trace']['position_m'][-1],self.arrays['position_m'][4])
        self.assertEqual(len(r['forecasts']),3)
        self.assertTrue(all(f['basis'].endswith('_gravity_no_contacts') for f in r['forecasts']))
        self.assertNotIn('capture',r)
    def test_unavailable_rebound_stays_unknown(self):
        r=self.analyze(dict(available=False,reason='no_board_rebound'))
        self.assertIsNone(r['trace']);self.assertEqual(r['forecasts'],[])
        self.assertEqual(r['observed']['missing_reason'],'no_board_rebound')
    def test_malformed_dense_shape_rejected_by_real_cpu_first_flight_oracle(self):
        arrays=copy.deepcopy(self.arrays);arrays['velocity_m_s']=np.zeros((5,3))
        with self.assertRaisesRegex(ValueError,'dense trace shape'):
            d.analyze_trace(self.row,self.scene,self.cfg,arrays,[0.,0.,-9.81])
    def test_nonfinite_dense_state_rejected_by_real_cpu_first_flight_oracle(self):
        arrays=copy.deepcopy(self.arrays);arrays['position_m'][2,0]=np.nan
        with self.assertRaisesRegex(ValueError,'Nonfinite'):
            d.analyze_trace(self.row,self.scene,self.cfg,arrays,[0.,0.,-9.81])
    def test_bucket_retention_without_valid_route_is_not_called_no_entry(self):
        o=dict(native_release=True,capture=False,bucket_capture=True,whole_entry=True,retention_s=1.)
        label=d.classification(o)
        self.assertIn('one-second bucket retention',label);self.assertIn('board-route scoring failed',label)
        self.assertNotIn('No whole entry',label)
    def test_examples_preserve_retained_but_failed_route_without_hardcoded_case_id(self):
        rows=[dict(id=str(i),context_id=str(i),prediction=dict(capture_score=1.-i*.1),
                   observed=dict(native_release=True,capture=False,bucket_capture=i==5)) for i in range(6)]
        rows.append(dict(id='reject',context_id='r',observed=dict(native_release=False,capture=False)))
        result=d.choose_cases(rows,{m:dict(losses=[]) for m in ('bc','ppo','ensemble')})
        self.assertIn('5',result);self.assertIn('reject',result);self.assertLessEqual(len(result),6)
    def test_paired_gains_and_losses_count_same_layouts(self):
        rows=[]
        for m in d.METHODS:
            for i in range(3):rows.append(dict(method=m,context_id=str(i),observed=dict(capture=(i==0 if m=='consistent' else i==1))))
        out=d.paired_results(rows)
        for value in out.values():
            self.assertEqual(value['wins'],['0']);self.assertEqual(value['losses'],['1']);self.assertEqual(value['both_fail'],['2'])
    def test_paired_missing_layout_raises(self):
        with self.assertRaises(ValueError):d.paired_results([dict(method='consistent',context_id='a',observed=dict(capture=False))])
    def test_import_is_cpu_only_without_engine_or_torch(self):
        cmd='import sys;from sircus.mjlab import consistent_flight_diagnosis;assert not any(n in sys.modules for n in ("torch","mujoco","warp","mjlab"))'
        subprocess.run([sys.executable,'-c',cmd],cwd=ROOT,check=True,capture_output=True)


if __name__=='__main__':unittest.main()
