"""Exercise actual HTTP dispatch in memory, with no Viam or physical adapter."""
import io
import json
from pathlib import Path
import sys
import types
import tempfile
import unittest
from unittest.mock import Mock, patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import server

class WorkflowRoutes(unittest.TestCase):
    def setUp(self):
        self.operations = Mock(); self.training = Mock()
        self.operations.status.return_value = {'hardware_enabled': False, 'phase': 'ready_to_scan'}
        self.operations.action.return_value = {'hardware_enabled': False, 'phase': 'scanning'}
        self.training.status.return_value = {'phase': 'idle'}
        self.training.start.return_value = {'phase': 'running'}
        self.training.cancel.return_value = {'phase': 'canceled'}
        self.training.artifact.return_value = {'schema': 'checkpoint'}
        self.native = Mock()
        self.scope = patch.multiple(server, OFFLINE=True, PHYSICAL=None, BRIDGE=self.native,
                                    OPERATIONS=self.operations, TRAINING_JOBS=self.training,
                                    WORKFLOW_TOKEN='TEST_ONLY_WORKFLOW')
        self.scope.start(); self.addCleanup(self.scope.stop)

    def request(self, path, payload=None, *, token='TEST_ONLY_WORKFLOW', host='127.0.0.1:18766',
                origin='http://127.0.0.1:18766', raw=None, length=None):
        h = server.Handler.__new__(server.Handler); h.path=path
        h.server=types.SimpleNamespace(server_address=('127.0.0.1',18766))
        h.headers={'Host':host,'Origin':origin,'Content-Type':'application/json','X-Sircus-Workflow-Token':token}
        h.send=Mock()
        if payload is None and raw is None: h.do_GET()
        else:
            data=raw if raw is not None else json.dumps(payload).encode()
            h.headers['Content-Length']=str(len(data) if length is None else length)
            h.rfile=io.BytesIO(data);h.do_POST()
        args=h.send.call_args.args
        return args[0], args[1] if len(args)>1 else 200

    def test_guided_and_training_paths_never_use_native_bridge(self):
        self.assertEqual(self.request('/api/operations/status')[0]['action_token'],'TEST_ONLY_WORKFLOW')
        self.assertEqual(self.request('/api/operations/action',{'action':'scan_bucket'})[1],202)
        self.assertEqual(self.request('/api/training/start',{'seed':42})[1],202)
        self.assertEqual(self.request('/api/training/cancel',{})[1],202)
        self.assertEqual(self.native.mock_calls,[])

    def test_hardware_or_cloud_mode_cannot_enable_workflow_even_with_valid_token(self):
        for changes in ({'OFFLINE':False},{'PHYSICAL':Mock()},{'OPERATIONS':None}):
            with patch.multiple(server,**changes):
                self.assertEqual(self.request('/api/operations/action',{'action':'grip_drop'})[1],503)
                self.assertEqual(self.request('/api/training/start',{})[1],503)
        self.operations.action.assert_not_called();self.training.start.assert_not_called()
        self.assertEqual(self.native.mock_calls,[])

    def test_token_host_and_origin_rejected_before_any_action(self):
        for kwargs in ({'token':''},{'host':'evil.example:18766'},{'origin':'https://evil.example'}):
            self.assertEqual(self.request('/api/operations/action',{'action':'grip_drop'},**kwargs)[1],403)
        self.assertEqual(self.request('/api/operations/status',host='evil.example')[1],403)
        self.operations.action.assert_not_called()

    def test_unknown_paths_nonfinite_and_oversized_inputs_rejected(self):
        self.assertEqual(self.request('/api/operations/hardware',{})[1],404)
        self.assertEqual(self.request('/api/operations/action',raw=b'{"seed":NaN}')[1],400)
        self.assertEqual(self.request('/api/operations/action',[],length=1000001)[1],413)
        self.assertEqual(self.request('/api/operations/action',[])[1],400)
        self.assertEqual(self.request('/api/training/cancel',{'run_id':'not-allowed'})[1],400)
        self.operations.action.assert_not_called();self.training.cancel.assert_not_called()

    def test_artifact_query_is_explicit_and_delegates_safe_identifiers(self):
        self.assertEqual(self.request('/api/training/artifact?run_id=run-1&name=checkpoint.json')[1],200)
        self.training.artifact.assert_called_once_with('run-1','checkpoint.json')
        for query in ('name=x','run_id=a&name=x&name=y','run_id=a&name=x&path=/etc/passwd'):
            self.assertEqual(self.request('/api/training/artifact?'+query)[1],400)
        self.assertEqual(self.training.artifact.call_count,1)

    def test_busy_and_failure_do_not_fallback(self):
        self.operations.action.side_effect=RuntimeError('Work already in progress')
        self.assertEqual(self.request('/api/operations/action',{'action':'plan'})[1],409)
        self.operations.action.side_effect=Exception('private details not to echo')
        value,code=self.request('/api/operations/action',{'action':'plan'})
        self.assertEqual(code,503);self.assertNotIn('private details',json.dumps(value))
        self.assertEqual(self.native.mock_calls,[])

class StudyResultsBoundary(unittest.TestCase):
    setUp = WorkflowRoutes.setUp
    request = WorkflowRoutes.request
    def test_report_and_readiness_are_read_only_offline_resources(self):
        report = {'schema':'sircus-learning-challenge-report-v1', 'data_origin':'synthetic',
                  'physically_verified':False, 'learning_curve':[], 'status':'complete',
                  'profile_id':'front-negative-y-v1', 'native_paths_checked':False}
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'report.json';path.write_text(json.dumps(report))
            with patch.object(server,'STUDY_RESULTS_PATH',path):
                self.assertEqual(self.request('/api/study/results'),(report,200))
                self.assertEqual(self.request('/api/study/results',host='untrusted.test')[1],403)
                state,code=self.request('/api/deployment/status')
                self.assertEqual(code,200);self.assertEqual(state['mode'],'blocked')
                self.assertFalse(state['hardware_enabled'])
                with patch.object(server,'OFFLINE',False):
                    self.assertEqual(self.request('/api/study/results')[1],503)
                    self.assertEqual(self.request('/api/deployment/status')[1],503)
                for changes in ({'physically_verified':True}, {'status':'running'},
                                {'profile_id':'historical-positive-y-v1'}, {'native_paths_checked':True}):
                    path.write_text(json.dumps({**report,**changes}))
                    self.assertEqual(self.request('/api/study/results')[1],400)
        self.operations.action.assert_not_called();self.training.start.assert_not_called()
        self.assertEqual(self.native.mock_calls,[])

    def test_report_rejects_missing_symlink_nonfinite_and_oversize(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'report.json'
            for contents in ('{"value":NaN}', '{}'):
                path.write_text(contents)
                with self.assertRaises(ValueError):server.load_study_results(path)
            link=Path(folder)/'link.json';link.symlink_to(path)
            with self.assertRaises(FileNotFoundError):server.load_study_results(link)
            with self.assertRaises(FileNotFoundError):server.load_study_results(Path(folder)/'missing')
            with path.open('wb') as handle:handle.truncate(32_000_001)
            with self.assertRaisesRegex(ValueError,'limit'):server.load_study_results(path)


class WorkflowShutdown(unittest.TestCase):
    def test_sigterm_cleans_owned_workers_only_in_operations_mode(self):
        operations=Mock();training=Mock();httpd=Mock();handlers={}
        def register(signum,handler):handlers[signum]=handler
        def terminate():handlers[server.signal.SIGTERM](server.signal.SIGTERM,None)
        httpd.serve_forever.side_effect=terminate
        with patch.multiple(server, OPERATIONS=None, TRAINING_JOBS=None, PHYSICAL=None, BRIDGE=None), \
             patch.object(sys,'argv',['server.py','--offline','--physical-mode','disabled','--operations']), \
             patch.object(server,'ThreadingHTTPServer',return_value=httpd), \
             patch.object(server.signal,'signal',side_effect=register), \
             patch('sircus.operations.service.GuidedOperations',return_value=operations), \
             patch('sircus.operations.training.TrainingJobs',return_value=training), \
             patch.object(server,'ViamBridge') as cloud, patch.object(server,'PhysicalBridge') as physical, \
             patch('builtins.print'):
            with self.assertRaises(SystemExit):server.main()
            operations.close.assert_called_once_with();training.close.assert_called_once_with()
            httpd.server_close.assert_called_once_with()
            cloud.assert_not_called();physical.assert_not_called()
        self.assertEqual(set(handlers),{server.signal.SIGTERM})

    def test_legacy_offline_server_keeps_existing_signal_behavior(self):
        httpd=Mock()
        with patch.multiple(server, OPERATIONS=None, TRAINING_JOBS=None, PHYSICAL=None, BRIDGE=None), \
             patch.object(sys,'argv',['server.py','--offline']), \
             patch.object(server,'ThreadingHTTPServer',return_value=httpd), \
             patch.object(server.signal,'signal') as install, patch('builtins.print'):
            server.main()
            install.assert_not_called()
            httpd.serve_forever.assert_called_once_with()
            httpd.server_close.assert_called_once_with()

if __name__ == '__main__': unittest.main()
