"""Control-file and import boundary tests; no simulator or saved-data audit."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
import io
from types import ModuleType
from unittest import mock

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
import location_comparison_recovery as entry


class EntryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.control = self.root/'control'; self.control.mkdir()
        self.producer = self.root/'producer'; self.producer.mkdir()
        self.manifest_path = self.control/'manifest.json'
        self.manifest = dict(fingerprint='a'*64, output=str(self.root/'native'),
            audit_output=str(self.root/'original-audit'), source_commit='b'*40)
        self.manifest_path.write_text(json.dumps(self.manifest))
        sources = ('location_comparison_recovery.py','location_comparison_recovery_verify.py',
                   'location_comparison_recovery_watchdog.py')
        self.request = dict(schema='sircus-location-comparison-recovery-request-v1',
            producer_source=str(self.producer), execute_approved=True, native_application_starts=0,
            optimizer_steps=0, new_shots=0, recovery_session_id='1'*32, native_session_id='2'*32,
            audit_output=str(self.root/'new-audit'), native_output=self.manifest['output'],
            recovery_source_sha256={name:entry.sha(SOURCE/name) for name in sources},
            manifest_path=str(self.manifest_path),manifest_sha256=entry.sha(self.manifest_path),
            manifest_fingerprint='a'*64, source_commit='b'*40)
        for field in ('inventory','retirement_proof'):
            path = self.control/(field+'.json'); path.write_text('{}')
            self.request[field+'_path'] = str(path)
            self.request[field+'_sha256'] = entry.sha(path)
        self.path = self.control/'request.json'

    def save(self, request=None):
        value = copy.deepcopy(self.request if request is None else request)
        value.pop('fingerprint',None)
        value['fingerprint'] = hashlib.sha256(json.dumps(value,sort_keys=True,
            separators=(',',':'),allow_nan=False).encode()).hexdigest()
        self.path.write_text(json.dumps(value))
        return entry.sha(self.path)

    def load(self, checksum=None):
        return entry.load_control(self.path, checksum or self.save(), self.producer)

    def test_valid_control_is_read_only(self):
        value, manifest = self.load()
        self.assertEqual(manifest,self.manifest)
        self.assertFalse(Path(value['audit_output']).exists())

    def test_external_pin_and_internal_fingerprint_both_required(self):
        pin = self.save()
        value = json.loads(self.path.read_text()); value['new_shots']=1
        self.path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError,'request bytes'):
            self.load(pin)
        with self.assertRaisesRegex(ValueError,'fingerprint'):
            self.load(entry.sha(self.path))

    def test_forbidden_scope_and_overlapping_destinations_fail(self):
        for field, wrong in [('execute_approved',False),('new_shots',1),('optimizer_steps',1),
            ('native_application_starts',1),('recovery_session_id','2'*32),
            ('audit_output',str(self.root/'native'/'audit')),('audit_output',str(self.control/'audit')),
            ('audit_output',str(SOURCE/'audit')),
            ('producer_source',str(self.root/'other'))]:
            with self.subTest(field=field,wrong=wrong):
                value=copy.deepcopy(self.request);value[field]=wrong
                with self.assertRaises(ValueError): self.load(self.save(value))

    def test_missing_module_pin_and_changed_retirement_fail(self):
        value=copy.deepcopy(self.request);value['recovery_source_sha256'].pop('location_comparison_recovery.py')
        with self.assertRaisesRegex(ValueError,'source binding'): self.load(self.save(value))
        pin=self.save();Path(self.request['retirement_proof_path']).write_text('{"changed":true}')
        with self.assertRaisesRegex(ValueError,'retirement_proof'):self.load(pin)

    def test_original_audit_may_not_be_overwritten(self):
        Path(self.manifest['audit_output']).mkdir()
        with self.assertRaisesRegex(ValueError,'Original audit destination'):self.load()

    def test_symlink_and_duplicate_json_fail(self):
        self.path.symlink_to(self.manifest_path)
        with self.assertRaisesRegex(ValueError,'Noncanonical'):entry.canonical(self.path)
        self.path.unlink(); self.path.write_text('{"a":1,"a":2}')
        with self.assertRaisesRegex(ValueError,'Duplicate'):entry.read(self.path)

    def test_import_does_not_load_numerics_simulator_or_execute(self):
        code="import sys;sys.path.insert(0,sys.argv[1]);import location_comparison_recovery;assert not any(x.split('.')[0] in {'torch','numpy','omni','isaacsim','isaaclab'} for x in sys.modules)"
        subprocess.run([sys.executable,'-c',code,str(SOURCE)],check=True,capture_output=True)

    def test_simulator_import_boundary_and_cuda_environment_fail(self):
        guard=entry.NoSimulator()
        for name in ('isaacsim','isaaclab.app','omni.kit'):
            with self.assertRaisesRegex(RuntimeError,'Simulator import forbidden'):guard.find_spec(name)
        with mock.patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'0','NVIDIA_VISIBLE_DEVICES':'void'}):
            with self.assertRaisesRegex(ValueError,'CPU-only'):entry.install_cpu_boundary()

    def test_original_source_is_checked_before_import(self):
        path=self.producer/'location_comparison_verify.py'
        path.write_text("raise RuntimeError('Untrusted helper executed')")
        manifest=dict(source_sha256={path.name:'0'*64})
        with self.assertRaisesRegex(ValueError,'Original source changed'):
            entry.verify_producer_source(manifest,self.producer)
        manifest['source_sha256'][path.name]=entry.sha(path)
        entry.verify_producer_source(manifest,self.producer)
        (self.producer/'unexpected.py').write_text('')
        with self.assertRaisesRegex(ValueError,'membership'):
            entry.verify_producer_source(manifest,self.producer)

    def test_unlisted_recovery_helper_is_rejected(self):
        directory=self.root/'code';directory.mkdir()
        (directory/'entry.py').write_text('')
        with mock.patch.object(entry,'__file__',str(directory/'entry.py')):
            entry.verify_recovery_membership(dict(recovery_source_sha256={'entry.py':'pin'}))
            (directory/'unexpected.py').write_text('')
            with self.assertRaisesRegex(ValueError,'Unexpected helper'):
                entry.verify_recovery_membership(dict(recovery_source_sha256={'entry.py':'pin'}))


class RoutingTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve();self.output=self.root/'audit'
        self.events=[];self.writes=[]
        self.request=dict(audit_output=str(self.output),native_output=str(self.root/'native'),
            comparison_fingerprint='c',recovery_session_id='1'*32,fingerprint='f',storage_root=str(self.root))
        self.manifest=dict(files={'comparison':{'path':'plan','sha256':'pin'}})
        self.modules={}
        for name in ('candidate_continuation_journal','location_comparison_verify',
                'location_comparison_recovery_verify','location_comparison_recovery_watchdog'):
            self.modules[name]=ModuleType(name)
        self.modules['candidate_continuation_journal'].write_new_json=lambda p,v:self.writes.append((p,v))
        self.modules['location_comparison_verify'].read=lambda p:{'fingerprint':'c'}
        self.modules['location_comparison_verify'].verify_inputs=lambda *a:self.events.append('inputs')
        self.modules['location_comparison_recovery_verify'].audit_recovery=lambda *a:self.events.append('science') or {'verified':'mock'}
        self.modules['location_comparison_recovery_verify'].verify_recovery_exit=lambda *a:({}, {})
        self.modules['location_comparison_recovery_verify'].verify_scientific_body=lambda:self.events.append('parity')
        self.modules['location_comparison_recovery_verify'].verify_inventory=lambda *a:{'mock':True}
        self.modules['location_comparison_recovery_verify'].verify_journal=lambda *a:{'mock':True}
        self.modules['location_comparison_recovery_watchdog'].validate_child_owner=lambda *a:self.events.append('owner')
        self.modules['location_comparison_recovery_watchdog'].supervise_recovery=lambda *a,**kw:self.events.append('supervisor') or {'independent_report_verified':True}
        self.modules['location_comparison_recovery_watchdog'].LinuxRecoveryOps=lambda:mock.Mock(
            free_bytes=lambda p:10*2**30, prepare=lambda:None,resources=lambda:{'mock':True})
        self.modules['location_comparison_recovery_watchdog'].LIMITS={'min_free_bytes':6*2**30}

    def run_mode(self,mode):
        with ExitStack() as stack:
            stack.enter_context(mock.patch.dict(sys.modules,self.modules))
            stack.enter_context(mock.patch.object(sys,'path',list(sys.path)))
            stack.enter_context(mock.patch.object(entry,'load_control',return_value=(self.request,self.manifest)))
            stack.enter_context(mock.patch.object(entry,'verify_recovery_membership',side_effect=lambda r:self.events.append('recovery source')))
            stack.enter_context(mock.patch.object(entry,'verify_producer_source',side_effect=lambda *a:self.events.append('producer source')))
            stack.enter_context(mock.patch.object(entry,'install_cpu_boundary',side_effect=lambda:self.events.append('CPU boundary')))
            stack.enter_context(mock.patch.object(entry,'verify_runtime',side_effect=lambda m:self.events.append('runtime')))
            stack.enter_context(mock.patch.object(entry,'sha',return_value='pin'))
            stack.enter_context(mock.patch.object(entry,'canonical',side_effect=Path))
            stack.enter_context(redirect_stdout(io.StringIO()))
            return entry.main(['--request',str(self.root/'request'),'--expected-request-sha256','pin',
                '--producer-source',str(self.root/'producer'),mode])

    def test_preflight_checks_sources_cpu_runtime_without_start_or_write(self):
        self.assertEqual(self.run_mode('--preflight-only'),0)
        self.assertEqual(self.events[:3],['recovery source','producer source','CPU boundary'])
        self.assertIn('runtime',self.events);self.assertIn('inputs',self.events)
        self.assertNotIn('science',self.events);self.assertNotIn('supervisor',self.events)
        self.assertEqual(self.writes,[]);self.assertFalse(self.output.exists())

    def test_approved_entry_dispatches_only_supervisor(self):
        self.assertEqual(self.run_mode('--execute-approved'),0)
        self.assertIn('supervisor',self.events);self.assertNotIn('science',self.events)
        self.assertEqual(self.writes,[])

    def test_child_requires_owner_then_science_before_report_write(self):
        self.assertEqual(self.run_mode('--audit-child'),0)
        self.assertLess(self.events.index('owner'),self.events.index('science'))
        self.assertEqual(self.events.count('owner'),2)
        self.assertEqual([p.name for p,v in self.writes],['recovery-audit.json'])

    def test_owner_failure_writes_no_report(self):
        self.modules['location_comparison_recovery_watchdog'].validate_child_owner=mock.Mock(side_effect=ValueError('owner mismatch'))
        with self.assertRaisesRegex(ValueError,'owner mismatch'):self.run_mode('--audit-child')
        self.assertEqual(self.writes,[]);self.assertNotIn('science',self.events)

    def test_scientific_failure_preserves_separate_failure_receipt_only(self):
        self.modules['location_comparison_recovery_verify'].audit_recovery=mock.Mock(side_effect=ValueError('physics mismatch'))
        with self.assertRaisesRegex(ValueError,'physics mismatch'):self.run_mode('--audit-child')
        self.assertEqual([p.name for p,v in self.writes],['recovery-audit-failure.json'])
        self.assertFalse(self.writes[0][1]['clean_shutdown'])


if __name__ == '__main__':
    unittest.main()
