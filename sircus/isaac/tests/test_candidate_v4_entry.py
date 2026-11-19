"""Entrypoint and independent CPU boundary tests with fabricated components."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock,patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import candidate_v4_entry as subject
import candidate_v4_inputs as inputs_module
import candidate_v4_native as native
import candidate_v4_application as application
import candidate_v4_watchdog as watchdog
from candidate_continuation_ledger import canonical_hash
from test_candidate_v4_evaluation import fixture


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


class EntryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.session='a'*32;self.argv=['--manifest',str(self.root/'manifest.json'),'--expected-manifest-sha256','b'*64,
            '--output',str(self.root),'--session',self.session]
        self.fake_inputs=dict(plan=dict(synthetic=False))
        self.load=patch.object(inputs_module,'production_inputs',return_value=self.fake_inputs)
        self.freeze=patch.object(inputs_module,'frozen_verifier',return_value=Mock())
        self.modelreq=patch.object(inputs_module,'model_request',return_value=dict(fake_request=True))
        self.construct=patch.object(native,'ProductionComponents',return_value=object())
        self.pipeline=patch.object(application,'run_application',return_value=dict(fake_complete=True))
        self.writer=patch.object(watchdog,'ProgressWriter',return_value=object())
        self.auditor=patch.object(subject,'audit_application',return_value=dict(fake_audit=True))
        for p in (self.load,self.freeze,self.modelreq,self.construct,self.pipeline,self.writer,self.auditor):p.start();self.addCleanup(p.stop)
        self.env=patch.dict(os.environ,{},clear=True);self.env.start();self.addCleanup(self.env.stop)
    def test_no_mode_and_both_modes_fail_before_inputs_or_components(self):
        for flags in ([],['--execute-approved','--audit-only']):
            with self.assertRaisesRegex(ValueError,'explicit future'):subject.main(self.argv+flags)
        inputs_module.production_inputs.assert_not_called();native.ProductionComponents.assert_not_called()
    def test_missing_native_flags_or_cookie_never_construct_app(self):
        for flags in (['--execute-approved'],['--execute-approved','--headless'],
            ['--execute-approved','--headless','--enable_cameras','--device','cuda:1'],
            ['--execute-approved','--headless','--enable_cameras']):
            with self.assertRaises(ValueError):subject.main(self.argv+flags)
        inputs_module.production_inputs.assert_not_called();native.ProductionComponents.assert_not_called()
    def test_explicit_future_native_mode_only_wires_fake_new_components(self):
        os.environ[watchdog.COOKIE]=self.session
        r=subject.main(self.argv+['--execute-approved','--headless','--enable_cameras'])
        self.assertEqual(r,dict(fake_complete=True));native.ProductionComponents.assert_called_once()
        self.assertTrue(native.ProductionComponents.call_args.kwargs['approved'])
        self.assertFalse(native.ProductionComponents.call_args.kwargs['synthetic'])
        application.run_application.assert_called_once();subject.audit_application.assert_not_called()
    def test_audit_requires_hidden_cuda_and_external_receipt_hash(self):
        with self.assertRaisesRegex(ValueError,'hide CUDA'):subject.main(self.argv+['--audit-only'])
        os.environ['CUDA_VISIBLE_DEVICES']=''
        with self.assertRaisesRegex(ValueError,'External'):subject.main(self.argv+['--audit-only'])
        inputs_module.production_inputs.assert_not_called()
    def test_audit_only_never_constructs_app_or_publisher(self):
        os.environ['CUDA_VISIBLE_DEVICES']=''
        r=subject.main(self.argv+['--audit-only','--expected-audit-request-sha256','c'*64])
        self.assertEqual(r,dict(fake_audit=True));subject.audit_application.assert_called_once()
        native.ProductionComponents.assert_not_called();watchdog.ProgressWriter.assert_not_called();application.run_application.assert_not_called()
    def test_malformed_session_and_missing_output_do_not_create_files(self):
        argv=list(self.argv);argv[-1]='bad'
        with self.assertRaisesRegex(ValueError,'session'):subject.main(argv+['--audit-only'])
        argv=list(self.argv);argv[argv.index('--output')+1]=str(self.root/'absent')
        with self.assertRaisesRegex(ValueError,'Supervisor-owned'):subject.main(argv+['--audit-only'])
        self.assertFalse((self.root/'absent').exists())
    def test_native_entry_rejects_synthetic_payload(self):
        self.fake_inputs['plan']['synthetic']=True;os.environ[watchdog.COOKIE]=self.session
        with self.assertRaisesRegex(ValueError,'disguise synthetic'):subject.main(self.argv+['--execute-approved','--headless','--enable_cameras'])
        native.ProductionComponents.assert_not_called()


class AuditCallbackTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        for name in ('evaluation-result.json','native-binding.json','model-check.json','native-setup/startup/receipt.json','evaluation-plan.json'):
            save(self.root/name,{})
        save(self.root/'evaluation-audit.json',dict(status='fabricated_reader_result'))
        self.inputs=dict(plan=dict(fingerprint='a'*64),execution=dict(source_sha256='b'*64),
            evaluation_manifest_sha256='c'*64,evaluation_manifest_path=self.root/'manifest.json',
            model_receipt_sha256=subject.sha(self.root/'model-check.json'))
    def test_future_cpu_child_command_is_fixed_hidden_and_bounded(self):
        fake=Mock();callback=subject.audit_callback(self.inputs,'d'*32,run=fake)
        fake.assert_not_called()
        r=callback(self.root,self.inputs['plan'],{})
        self.assertEqual(r['status'],'fabricated_reader_result');fake.assert_called_once()
        args,kw=fake.call_args
        self.assertIn('--audit-only',args[0]);self.assertNotIn('--execute-approved',args[0])
        self.assertEqual(kw['env']['CUDA_VISIBLE_DEVICES'],'');self.assertEqual(kw['timeout'],600);self.assertTrue(kw['check'])
        with self.assertRaises(FileExistsError):callback(self.root,self.inputs['plan'],{})
        self.assertEqual(fake.call_count,1)

    def test_changed_pre_app_model_receipt_never_starts_audit_child(self):
        fake=Mock();callback=subject.audit_callback(self.inputs,'d'*32,run=fake)
        save(self.root/'model-check.json',dict(changed=True))
        with self.assertRaisesRegex(ValueError,'pre-App freeze'):callback(self.root,self.inputs['plan'],{})
        fake.assert_not_called();self.assertFalse((self.root/'audit-request.json').exists())


class CombinedAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.plan=fixture()
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.session='a'*32
        self.summary=dict(synthetic=True,original_assessment_completed=False,journal_head_sha256='d'*64)
        self.proof=dict(status='passed_synthetic',signature='frozen commands',fingerprint='old',request_sha256='old request')
        for name,v in [('evaluation-plan.json',self.plan),('evaluation-result.json',self.summary),('native-binding.json',{}),
            ('native-setup/startup/receipt.json',{}),('model-check.json',self.proof)]:save(self.root/name,v)
        self.inputs=dict(plan=copy.deepcopy(self.plan),execution=dict(source_sha256='b'*64),evaluation_manifest_sha256='c'*64,
            paths={k:self.root/(k+'.json') for k in ('dropper_urdf','board_urdf','poses','camera')},
            evaluation_paths={k:self.root/(k+'.pt') for k in ('parent_checkpoint','final_checkpoint')},
            values=dict(scene={},source_contract=dict(cells=[])),train=dict(cells=[]),model_request_base=dict(mock_request=True))
        req=dict(schema=subject.REQUEST_SCHEMA,session=self.session,plan_fingerprint=self.plan['fingerprint'],source_sha256='b'*64,
            manifest_sha256='c'*64,**{role+'_sha256':subject.sha(self.root/name) for role,name in
                dict(evaluation_result='evaluation-result.json',native_binding='native-binding.json',model_receipt='model-check.json',
                startup_receipt='native-setup/startup/receipt.json',plan='evaluation-plan.json').items()})
        req['fingerprint']=canonical_hash(req);save(self.root/'audit-request.json',req)
        self.request_sha=subject.sha(self.root/'audit-request.json')
        def model(request):
            result=dict(self.proof,fingerprint='new',request_sha256='different output path request')
            save(Path(request['receipt_path']),result);return result
        def native_check(*args,**kwargs):
            r=dict(status='passed_synthetic',all_components_passed=True,synthetic=True,
                plan_fingerprint=self.plan['fingerprint'],journal_head_sha256='d'*64)
            r['fingerprint']=canonical_hash(r);return r
        self.checks=dict(model=Mock(side_effect=model),model_validate=Mock(side_effect=lambda r,p:r),native=Mock(side_effect=native_check))
        self.env=patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':''});self.env.start();self.addCleanup(self.env.stop)
    def run_audit(self):
        return subject.audit_application(self.root,self.inputs,self.session,expected_request_sha256=self.request_sha,checks=self.checks)
    def test_cpu_recheck_then_independent_native_reader_preserve_old_and_new_proof_scopes(self):
        r=self.run_audit();self.assertEqual(r['status'],'passed_synthetic');self.assertFalse(r['full_native_audit'])
        self.assertFalse(r['original_assessment_completed']);self.checks['model'].assert_called_once();self.checks['native'].assert_called_once()
        kw=self.checks['native'].call_args.kwargs
        self.assertEqual(kw['expected_result_sha256'],subject.sha(self.root/'evaluation-result.json'))
        self.assertEqual(kw['expected_head_sha256'],'d'*64)
        self.assertEqual(kw['model_receipt_path'],self.root/'model-check.json')
        self.assertEqual(subject.read(self.root/'evaluation-audit.json'),r)
    def test_model_proof_disagreement_stops_before_physics(self):
        def changed(request):
            r=dict(self.proof,signature='different commands');save(Path(request['receipt_path']),r);return r
        self.checks['model'].side_effect=changed
        with self.assertRaisesRegex(ValueError,'reconstruction changed'):self.run_audit()
        self.checks['native'].assert_not_called();self.assertFalse((self.root/'evaluation-audit.json').exists())
    def test_changed_durable_model_receipt_rejected_before_any_model_check(self):
        save(self.root/'model-check.json',dict(changed=True))
        with self.assertRaisesRegex(ValueError,'input bytes'):self.run_audit()
        self.checks['model'].assert_not_called()
    def test_real_audit_cannot_accept_injected_readers(self):
        self.inputs['plan']['synthetic']=False
        self.summary['synthetic']=False;save(self.root/'evaluation-result.json',self.summary)
        save(self.root/'evaluation-plan.json',self.inputs['plan'])
        req=subject.read(self.root/'audit-request.json');req['evaluation_result_sha256']=subject.sha(self.root/'evaluation-result.json');req['plan_sha256']=subject.sha(self.root/'evaluation-plan.json')
        req['fingerprint']=canonical_hash({k:v for k,v in req.items() if k!='fingerprint'});save(self.root/'audit-request.json',req);self.request_sha=subject.sha(self.root/'audit-request.json')
        with self.assertRaisesRegex(ValueError,'cannot replace independent'):self.run_audit()
        self.checks['model'].assert_not_called()
    def test_audit_only_hidden_cuda_and_failed_native_checks_do_not_emit_completion(self):
        os.environ['CUDA_VISIBLE_DEVICES']='0'
        with self.assertRaisesRegex(ValueError,'hide CUDA'):self.run_audit()
        os.environ['CUDA_VISIBLE_DEVICES']=''
        self.checks['native'].side_effect=ValueError('synthetic corrupted physics')
        with self.assertRaisesRegex(ValueError,'corrupted physics'):self.run_audit()
        self.assertFalse((self.root/'evaluation-audit.json').exists())


if __name__=='__main__':unittest.main()
