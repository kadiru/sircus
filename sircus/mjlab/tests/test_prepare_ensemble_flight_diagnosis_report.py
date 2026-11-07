"""Strict additive publication binds derived checks and original source IDs."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from sircus.mjlab import prepare_ensemble_flight_diagnosis_report as publisher


CHECK_VERSION='sircus-ensemble-flight-diagnosis-checks-v1'
CODE_SHA='a'*64


def payload():
    proof=dict(version=CHECK_VERSION,status='passed',checks_count=1,
        checks=[dict(label='independent measured diagnostic parity',passed=True)],
        source_report_sha256=publisher.REPORT_SHA,source_audit_sha256=publisher.AUDIT_SHA,
        diagnosis_source_sha256=CODE_SHA)
    raw=dict(version=publisher.VERSION,status='completed',
        source=dict(verified=True,report_sha256=publisher.REPORT_SHA,
            audit_sha256=publisher.AUDIT_SHA,contract_fingerprint=publisher.SOURCE_FP,
            native_requests_added=0,original_native_audit_checks=897304),
        audit=dict(version=CHECK_VERSION,status='passed',checks_count=1,
            full_checks_verified=True),
        processing=dict(source_sha256=CODE_SHA,cuda_used=False,training=False,new_native_requests=0),
        cases=[dict(id='case-29',observed=dict(capture=False,whole_entry=False),
            trace=dict(position_m=[[0.,0.,.3],[.1,0.,.4]],time_s=[0.,.1]),
            forecasts=[dict(basis='observed_rebound_origin_model_velocity_gravity_no_contacts',
                            position_m=[[0.,0.,.3],[.09,0.,.4]])])],
        summary=dict(requested=60,native_releases=49,captures=24,rejected=11,native_misses=25))
    return raw,proof


def write_json(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def bind(folder,raw,proof,*,summary=None,manifest=None):
    folder.mkdir(exist_ok=True)
    write_json(folder/'checks.json',proof)
    # This is the same non-circular binding direction used by the producer.
    raw=copy.deepcopy(raw)
    raw['audit']['source_sha256']=publisher.sha(folder/'checks.json')
    write_json(folder/'diagnosis.json',raw)
    if summary is None:
        summary=copy.deepcopy(raw)
        for case in summary['cases']:
            case.pop('trace');case.pop('forecasts')
    write_json(folder/'summary.json',summary)
    if manifest is None:
        manifest={name:publisher.sha(folder/name) for name in ('diagnosis.json','summary.json','checks.json')}
    write_json(folder/'checksums.json',manifest)
    return raw


class PublisherTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.base=self.root/'base.json';self.folder=self.root/'diagnosis';self.output=self.root/'derived.json'
        self.base_value={'physics_ensemble_study':{'status':'completed'},'old_panel':{'untouched':[1,2]}}
        write_json(self.base,self.base_value)

    def publish(self,raw=None,proof=None,**kwargs):
        default_raw,default_proof=payload()
        bind(self.folder,default_raw if raw is None else raw,default_proof if proof is None else proof,**kwargs)
        return publisher.prepare(self.base,self.folder,self.output)

    def assert_denied(self,raw=None,proof=None,**kwargs):
        before=self.base.read_bytes()
        with self.assertRaises(ValueError):self.publish(raw,proof,**kwargs)
        self.assertEqual(self.base.read_bytes(),before)
        self.assertFalse(self.output.exists())

    def test_valid_packet_adds_only_panel_and_preserves_original_bytes(self):
        result=self.publish();expected=copy.deepcopy(self.base_value)
        actual=copy.deepcopy(result);panel=actual.pop(publisher.KEY)
        self.assertEqual(actual,expected)
        self.assertEqual(json.loads(self.output.read_text()),result)
        self.assertEqual(json.loads(self.base.read_text()),expected)
        self.assertEqual(panel['cases'][0]['observed']['capture'],False)
        self.assertIn('trace',panel['cases'][0]);self.assertIn('forecasts',panel['cases'][0])
        self.assertEqual(panel['source']['report_sha256'],publisher.REPORT_SHA)
        self.assertFalse(self.output.with_suffix('.json.tmp').exists())

    def test_every_payload_file_is_byte_bound_not_merely_named(self):
        for name in ('diagnosis.json','summary.json','checks.json'):
            with self.subTest(name=name):
                raw,proof=payload();bind(self.folder,raw,proof)
                with (self.folder/name).open('a') as stream:stream.write(' ')
                with self.assertRaisesRegex(ValueError,'Diagnostic bytes changed'):
                    publisher.prepare(self.base,self.folder,self.output)
                self.assertFalse(self.output.exists())

    def test_complete_manifest_and_base_object_are_required(self):
        for manifest in ({},{'diagnosis.json':'a'*64},{'diagnosis.json':'a'*64,'summary.json':'b'*64,'checks.json':'c'*64,'extra.json':'d'*64}):
            with self.subTest(manifest=manifest):self.assert_denied(manifest=manifest)
        raw,proof=payload();bind(self.folder,raw,proof);write_json(self.base,[])
        with self.assertRaisesRegex(ValueError,'Complete bound'):publisher.prepare(self.base,self.folder,self.output)

    def test_pending_wrong_versions_and_nonliteral_proof_pass_flags_fail(self):
        changes=[('version','other-version'),('status','running')]
        for key,value in changes:
            raw,proof=payload();raw[key]=value
            with self.subTest(raw_key=key):self.assert_denied(raw,proof)
        mutations=[lambda p:p.update(version='other-checks'),lambda p:p.update(status='pending'),
            lambda p:p.update(checks=[]),lambda p:p.update(checks_count=0),
            lambda p:p.update(checks_count=True),lambda p:p['checks'][0].update(passed=False),
            lambda p:p['checks'][0].update(passed=1),lambda p:p['checks'][0].update(passed='true')]
        for edit in mutations:
            raw,proof=payload();edit(proof)
            with self.subTest(edit=edit):self.assert_denied(raw,proof)

    def test_original_report_audit_fingerprint_and_counter_bindings_fail_closed(self):
        source_edits=[('report_sha256','b'*64),('audit_sha256','b'*64),('contract_fingerprint','b'*64),
            ('verified',1),('verified',False),('native_requests_added',1),
            ('native_requests_added',False),('original_native_audit_checks',1)]
        for key,value in source_edits:
            raw,proof=payload();raw['source'][key]=value
            with self.subTest(key=key,value=value):self.assert_denied(raw,proof)
        for key in ('source_report_sha256','source_audit_sha256'):
            raw,proof=payload();proof[key]='b'*64
            with self.subTest(proof_key=key):self.assert_denied(raw,proof)

    def test_derived_audit_version_status_count_and_literal_flags_are_required(self):
        for key,value in [('version','other-audit'),('status','failed'),('checks_count',2),
                          ('full_checks_verified',1),('full_checks_verified','true'),('full_checks_verified',False)]:
            raw,proof=payload();raw['audit'][key]=value
            with self.subTest(key=key,value=value):self.assert_denied(raw,proof)

    def test_missing_short_or_mismatched_source_identity_does_not_authenticate(self):
        edits=[lambda r,p:(r['processing'].pop('source_sha256'),p.pop('diagnosis_source_sha256')),
               lambda r,p:(r['processing'].update(source_sha256='short'),p.update(diagnosis_source_sha256='short')),
               lambda r,p:p.update(diagnosis_source_sha256='b'*64)]
        for edit in edits:
            raw,proof=payload();edit(raw,proof)
            with self.subTest(edit=edit):self.assert_denied(raw,proof)

    def test_publication_cannot_silently_claim_no_training_or_cuda_when_processing_disagrees(self):
        for key,value in [('cuda_used',True),('cuda_used',0),('training',True),
                          ('training',0),('new_native_requests',1),('new_native_requests',False)]:
            raw,proof=payload();raw['processing'][key]=value
            with self.subTest(key=key,value=value):self.assert_denied(raw,proof)

    def test_summary_must_be_exact_trace_stripped_packet(self):
        raw,proof=payload();summary=copy.deepcopy(raw)
        # bind() updates audit-byte identity; make a correct stripped template first.
        bound=bind(self.folder,raw,proof);summary=copy.deepcopy(bound)
        for case in summary['cases']:case.pop('trace');case.pop('forecasts')
        summary['cases'][0]['observed']['capture']=True
        bind(self.folder,raw,proof,summary=summary)
        with self.assertRaisesRegex(ValueError,'Sanitized summary differs'):
            publisher.prepare(self.base,self.folder,self.output)
        self.assertFalse(self.output.exists())

    def test_inplace_base_and_nested_evidence_outputs_are_refused_without_changes(self):
        raw,proof=payload();bind(self.folder,raw,proof)
        before={p:p.read_bytes() for p in (self.base,*self.folder.iterdir())}
        for output in (self.base,self.folder/'diagnosis.json',self.folder/'nested'/'display.json'):
            with self.subTest(output=output),self.assertRaisesRegex(ValueError,'Separate derived display'):
                publisher.prepare(self.base,self.folder,output)
        self.assertEqual({p:p.read_bytes() for p in before},before)
        self.assertFalse((self.folder/'nested').exists())

    def test_publisher_has_only_standard_library_imports(self):
        script=('import sys; from sircus.mjlab import prepare_ensemble_flight_diagnosis_report; '
                'assert not any(x in sys.modules for x in ("torch","numpy","warp","mujoco","mjlab"))')
        completed=subprocess.run([sys.executable,'-c',script],capture_output=True,text=True)
        self.assertEqual(completed.returncode,0,completed.stderr)


if __name__=='__main__':unittest.main()
