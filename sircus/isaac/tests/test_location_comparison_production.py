"""Fake application/process/policy only. No simulator, learning or new shots."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import location_comparison_production as production
from location_comparison_bounds import Bounds, BoundsExceeded
from location_comparison_mock import FakeClock, MockEngine
from test_location_comparison_runtime import synthetic_plan


class FakeEngine(MockEngine):
    def propose(self,name,cells,deterministic):
        # Fake-clock batches are well under .5 seconds. Reservations and phase
        # changes must nevertheless reach the watchdog before sampling.
        progress = production.read(self.directory.parent/'comparison-progress.json')['snapshot']
        count = len(cells)
        if not hasattr(self,'proposed'):
            self.proposed = {'original_usable':0,'broader':0,'assessment':0}
        key = name if cells[0]['split'] == 'train' else 'assessment'
        self.proposed[key] += count
        if key == 'assessment':
            assert progress['assessment_requested'] == self.proposed[key]
        else:
            assert progress['active_phase'] == name
            assert progress['train_requested'][name] == self.proposed[key]
        return super().propose(name,cells,deterministic)
    def close(self):
        self.closed = True


class FakeServices:
    synthetic = True
    def __init__(self,clock,fail=None):
        self.clock,self.fail = clock,fail
        self.calls = []
        self.engine = None
        self.app = SimpleNamespace(close=self.close)
    def step(self,name):
        self.calls.append(name)
        self.clock.advance(.1)
        if self.fail == name:
            raise RuntimeError('synthetic '+name+' failure')
    def preflight(self,manifest,experiment):
        self.step('preflight')
        return dict(status='passed',synthetic=True)
    def launch_app(self):
        self.step('launch')
        return self.app
    def create_scene(self,manifest,output):
        self.step('scene')
        return object()
    def startup(self,env,output,guard):
        self.step('startup')
        guard()
        return dict(status='passed',synthetic=True)
    def check_resources(self):
        return True
    def make_engine(self,env,manifest,experiment,output,plan,guard):
        self.step('engine')
        self.engine = FakeEngine(self.clock,output/'synthetic-checkpoints')
        return self.engine
    def close(self):
        self.step('close')


class LifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = synthetic_plan()
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name).resolve()
        self.clock = FakeClock()
        self.services = FakeServices(self.clock)
        self.manifest = dict(fingerprint='b'*64)
    def execute(self,**kwargs):
        options = dict(started_monotonic=99.,services=self.services,
            owner_check=lambda:True,clock=self.clock,
            resource_check=lambda:dict(free_storage_bytes=10*1024**3),
            integrity_check=lambda:True,synthetic=True)
        options.update(kwargs)
        return production.execute_child(self.manifest,{},self.plan,self.output,'a'*32,**options)
    def test_full_fake_application_closes_before_completion_and_remains_synthetic(self):
        result = self.execute()
        self.assertEqual(result['status'],'synthetic_complete_pending_audit',result)
        self.assertTrue(result['app_closed'])
        self.assertEqual(self.services.calls,['preflight','launch','scene','startup','engine','close'])
        trace = production.read(self.output/'comparison-trace.json')
        finish = trace['events'][-1]['payload']
        self.assertEqual(finish['counts']['assessment']['completed'],600)
        for arm in ('original_usable','broader'):
            self.assertEqual(finish['counts']['train'][arm]['completed'],8960)
        progress = production.read(self.output/'comparison-progress.json')
        self.assertFalse(progress['running'])
        self.assertEqual(progress['status'],'runner_complete_pending_audit')
        self.assertEqual(progress['snapshot']['started_monotonic'],99.)
        self.assertGreaterEqual(progress['snapshot']['phase_elapsed_s']['shared'],1.)
        self.assertFalse((self.output/'audit.json').exists())
    def test_changed_inputs_or_owner_block_before_app_start(self):
        for condition in ('integrity_check','owner_check'):
            with self.subTest(condition=condition):
                target = self.output/condition
                target.mkdir()
                with mock.patch.object(self,'output',target):
                    result = self.execute(**{condition:lambda:False})
                self.assertEqual(result['status'],'incomplete_no_retry')
                self.assertEqual(self.services.calls,[])
    def test_failed_preflight_never_spends_application_start(self):
        self.services.fail = 'preflight'
        result = self.execute()
        self.assertEqual(result['status'],'incomplete_no_retry')
        self.assertEqual(self.services.calls,['preflight'])
    def test_failed_startup_closes_once_without_policy_or_requests(self):
        self.services.fail = 'startup'
        result = self.execute()
        self.assertEqual(result['status'],'incomplete_no_retry')
        self.assertEqual(self.services.calls,['preflight','launch','scene','startup','close'])
        self.assertIsNone(self.services.engine)
        self.assertFalse((self.output/'events').exists())
    def test_close_failure_cannot_publish_finished(self):
        self.services.fail = 'close'
        # Fail runner immediately, so no full fabricated run is repeated here.
        with mock.patch.object(FakeEngine,'restore',side_effect=ValueError('synthetic restore')):
            result = self.execute()
        self.assertEqual(result['status'],'incomplete_no_retry')
        self.assertFalse(result['app_closed'])
        self.assertTrue(production.read(self.output/'comparison-progress.json')['running'])
    def test_existing_start_marker_cannot_retry(self):
        (self.output/'child-start.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'One start'):
            self.execute()
        self.assertEqual(self.services.calls,[])
    def test_native_mode_does_not_accept_fake_services_or_callbacks(self):
        with self.assertRaisesRegex(ValueError,'cannot be substituted'):
            self.execute(synthetic=False,execute_approved=True)
        with self.assertRaisesRegex(ValueError,'explicit fake'):
            self.execute(services=None)
        self.assertEqual(self.services.calls,[])
    def test_storage_and_startup_elapsed_are_checked_before_model(self):
        with self.assertRaises(BoundsExceeded):
            self.execute(resource_check=lambda:dict(free_storage_bytes=5*1024**3))
        self.assertEqual(self.services.calls,[])
    def test_default_cli_cannot_execute(self):
        with mock.patch('sys.stderr'), self.assertRaises(SystemExit):
            production.main([])
        with mock.patch.object(production,'load_inputs') as load:
            with self.assertRaisesRegex(ValueError,'Execution is off'):
                production.main(['--manifest','/synthetic/input.json','--expected-manifest-sha256','b'*64,
                    '--session-id','a'*32,'--native-child'])
            load.assert_not_called()


class InputTests(unittest.TestCase):
    def test_load_rejects_changed_bytes_before_native_imports(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp).resolve()/'manifest.json'
            file.write_text('{}')
            with mock.patch.object(production,'validate_inputs') as check:
                with self.assertRaisesRegex(ValueError,'Manifest bytes changed'):
                    production.load_inputs(file,'f'*64)
                check.assert_not_called()
    def test_json_rejects_duplicates_and_nonfinite_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp)/'manifest.json'
            for raw in ('{"x":1,"x":2}','{"x":NaN}'):
                file.write_text(raw)
                with self.assertRaises(ValueError):
                    production.read(file)
    def test_signed_manifest_requires_new_schema_and_all_roles(self):
        with self.assertRaisesRegex(ValueError,'production manifest'):
            production.validate_inputs({'schema':'legacy'})
        m = dict(schema=production.SCHEMA,runtime=production.RUNTIME,files={})
        m['fingerprint'] = production.digest(m)
        with self.assertRaisesRegex(ValueError,'Missing independent'):
            production.validate_inputs(m)
    def test_asset_inventory_rejects_linked_subdirectories_and_special_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            assets = root/'assets'
            assets.mkdir()
            (assets/'mesh.bin').write_bytes(b'synthetic')
            self.assertEqual(production.asset_inventory(assets),{'mesh.bin'})
            other = root/'other'
            other.mkdir()
            (assets/'linked-meshes').symlink_to(other,target_is_directory=True)
            with self.assertRaisesRegex(ValueError,'Linked or special'):
                production.asset_inventory(assets)
            (assets/'linked-meshes').unlink()
            production.os.mkfifo(assets/'not-a-file')
            with self.assertRaisesRegex(ValueError,'Linked or special'):
                production.asset_inventory(assets)
    def test_both_outputs_must_be_fresh_before_starting_native_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            manifest = dict(output=str(root/'native'),audit_output=str(root/'audit'))
            self.assertTrue(production.fresh_outputs(manifest))
            (root/'audit').mkdir()
            with self.assertRaisesRegex(ValueError,'Both native and audit'):
                production.fresh_outputs(manifest)
    def test_unsafe_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            target = root/'target'
            target.write_text('test')
            link = root/'link'
            link.symlink_to(target)
            with self.assertRaisesRegex(ValueError,'linked'):
                production.canonical(link)


class StartClockTests(unittest.TestCase):
    def test_bounds_include_child_startup_delay(self):
        plan = LifecycleTests.plan if hasattr(LifecycleTests,'plan') else synthetic_plan()
        clock = FakeClock()
        bounds = Bounds(plan,clock,lambda:dict(free_storage_bytes=10*1024**3),started_monotonic=90.)
        self.assertEqual(bounds.snapshot()['phase_elapsed_s']['shared'],10.)
        for value in (101.,-1.,float('nan'),True):
            with self.assertRaises(ValueError):
                Bounds(plan,clock,lambda:dict(free_storage_bytes=10*1024**3),started_monotonic=value)


class OwnershipAndEnvelopeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.native = self.root/'native-output'
        self.native.mkdir()
        self.auditdir = self.root/'audit-output'
        self.auditdir.mkdir()
        self.session = 'a'*32
        self.manifest = dict(fingerprint='b'*64,output=str(self.native),audit_output=str(self.auditdir))
        self.plan = dict(fingerprint='c'*64)
    def write(self,path,value):
        path.write_text(json.dumps(value,sort_keys=True))
    def process_signed(self,value):
        return dict(value,fingerprint=production.process_digest(value))
    def fixtures(self):
        trace = dict(mode='native',events=[])
        self.write(self.native/'comparison-trace.json',trace)
        child = dict(status='native_complete_pending_audit',synthetic=False,app_closed=True,
            session_id=self.session,runtime_fingerprint=self.manifest['fingerprint'],
            comparison_fingerprint=self.plan['fingerprint'],
            trace_sha256=production.file_sha(self.native/'comparison-trace.json'))
        self.write(self.native/'child-exit.json',child)
        native_exit = self.process_signed(dict(status='native_exit_zero_independent_audit_required',
            session_id=self.session,runtime_fingerprint=self.manifest['fingerprint'],
            comparison_fingerprint=self.plan['fingerprint'],lock_release_safe=True,
            exit_code=0,native_application_starts=1))
        self.write(self.native/'comparison-watchdog-exit.json',native_exit)
        report = dict(status='native_verified',synthetic=False,manifest_fingerprint=self.manifest['fingerprint'],
            comparison_fingerprint=self.plan['fingerprint'],trace_canonical_sha256=production.digest(trace),
            checkpoint_bytes_independently_verified=True,model_likelihoods_independently_verified=True,
            native_physics_verified=True)
        report['fingerprint'] = production.digest(report)
        return trace,child,native_exit,report
    def test_exact_audit_envelope_binds_the_verified_report_and_child_exit(self):
        _,_,native_exit,report = self.fixtures()
        value = production.audit_envelope(self.manifest,self.plan,self.session,native_exit,report)
        self.write(self.auditdir/'audit.json',value)
        self.assertTrue(production.verify_report(self.auditdir,native_exit['fingerprint'],
            self.manifest,self.plan,self.session))
        value['trace_sha256'] = 'd'*64
        self.write(self.auditdir/'audit.json',value)
        with self.assertRaisesRegex(ValueError,'envelope changed'):
            production.verify_report(self.auditdir,native_exit['fingerprint'],self.manifest,self.plan,self.session)
    def test_flags_alone_or_synthetic_audit_cannot_certify_native_work(self):
        _,_,native_exit,report = self.fixtures()
        for key,value in (('synthetic',True),('native_physics_verified',False),
                          ('manifest_fingerprint','d'*64),('trace_canonical_sha256','d'*64)):
            changed = dict(report,**{key:value})
            changed['fingerprint'] = production.digest({k:v for k,v in changed.items() if k != 'fingerprint'})
            with self.subTest(key=key), self.assertRaisesRegex(ValueError,'audit binding'):
                production.audit_envelope(self.manifest,self.plan,self.session,native_exit,changed)
    def test_changed_trace_or_incomplete_child_cannot_publish_complete(self):
        _,child,native_exit,report = self.fixtures()
        child['app_closed'] = False
        self.write(self.native/'child-exit.json',child)
        with self.assertRaisesRegex(ValueError,'Child completion'):
            production.audit_envelope(self.manifest,self.plan,self.session,native_exit,report)
    def test_audit_owner_requires_live_owned_ancestry_and_exact_deadline(self):
        from location_comparison_watchdog import AUDIT_SCHEMA
        _,_,native_exit,_ = self.fixtures()
        owner = self.process_signed(dict(schema=AUDIT_SCHEMA,session_id=self.session,
            runtime_fingerprint=self.manifest['fingerprint'],comparison_fingerprint=self.plan['fingerprint'],
            native_exit_fingerprint=native_exit['fingerprint'],start_monotonic=100.,deadline_monotonic=7300.,
            cuda_visible_devices='',owner={},native_owner={}))
        ownerpath = self.auditdir/'comparison-audit-owner.json'
        self.write(ownerpath,owner)
        env = dict(CUDA_VISIBLE_DEVICES='',SIRCUS_COMPARISON_SESSION=self.session,
            SIRCUS_COMPARISON_RUNTIME_FINGERPRINT=self.manifest['fingerprint'],
            SIRCUS_COMPARISON_FINGERPRINT=self.plan['fingerprint'],SIRCUS_COMPARISON_OWNER_FILE=str(ownerpath),
            SIRCUS_COMPARISON_NATIVE_EXIT_FINGERPRINT=native_exit['fingerprint'],
            SIRCUS_COMPARISON_STARTED_MONOTONIC='100',SIRCUS_COMPARISON_DEADLINE_MONOTONIC='7300')
        with mock.patch.dict(production.os.environ,env), \
             mock.patch('seed22_continuation.validate_owner',return_value=True) as ancestry:
            self.assertEqual(production.validate_audit_owner(self.manifest,self.plan,self.session,clock=lambda:101),native_exit)
            ancestry.assert_called_once()
            with self.assertRaisesRegex(ValueError,'deadline'):
                production.validate_audit_owner(self.manifest,self.plan,self.session,clock=lambda:7300)
            ancestry.return_value = False
            with self.assertRaisesRegex(ValueError,'owner changed'):
                production.validate_audit_owner(self.manifest,self.plan,self.session,clock=lambda:102)
    def test_handshake_waits_only_for_owned_parent_and_is_bounded(self):
        clock = FakeClock()
        target = self.root/'owner.json'
        def publish(delay):
            clock.advance(delay)
            target.write_text('{}')
        production.await_owner(target,clock=clock,sleep=publish,parent=lambda:5)
        target.unlink()
        with self.assertRaisesRegex(ValueError,'Watchdog exited'):
            production.await_owner(target,clock=clock,sleep=clock.advance,parent=lambda:1)
        with self.assertRaisesRegex(ValueError,'timed out'):
            production.await_owner(target,clock=clock,sleep=clock.advance,parent=lambda:5)


class NativeStartupRouteTests(unittest.TestCase):
    def test_real_startup_entry_uses_observer_and_independent_reader_then_restores_tick(self):
        calls = []
        tick = lambda: calls.append('tick')
        env = SimpleNamespace(tick=tick,cfg={},spec=dict(cells=[]),origins=[])
        class Recorder:
            def __init__(self,got,directory):
                self.env = got
            def run(self):
                calls.append('recorder')
                self.env.tick()
                return dict(status='completed')
        audit = mock.Mock(return_value=dict(status='passed'))
        module = SimpleNamespace(StartupRecorder=Recorder,audit_startup=audit)
        with mock.patch.dict(sys.modules,{'candidate_continuation_scene':module}):
            value = production.NativeServices().startup(env,Path('/synthetic'),lambda:True)
            self.assertEqual(value['status'],'passed')
            self.assertEqual(calls,['recorder','tick'])
            self.assertIs(env.tick,tick)
            audit.assert_called_once()
            with self.assertRaisesRegex(ValueError,'Startup guard'):
                production.NativeServices().startup(env,Path('/synthetic'),lambda:False)
            self.assertIs(env.tick,tick)


if __name__ == '__main__':
    unittest.main()
