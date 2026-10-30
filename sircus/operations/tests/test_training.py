"""Headless manager regressions. Fake children cannot access hardware or network."""
import copy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from sircus.operations.training import TrainingJobs, configuration


class FakeProcess:
    def __init__(self, code=0, stdout='', stderr=''):
        self.returncode = code
        self.stdout, self.stderr = io.StringIO(stdout), io.StringIO(stderr)
        self.pid = 0  # Tests replace the group stopper; never signal any real PID.
    def poll(self): return self.returncode
    def wait(self, timeout=None):
        if self.returncode is None: raise subprocess.TimeoutExpired('fake', timeout)
        return self.returncode


def complete_files(command):
    output = Path(command[command.index('--output') + 1]); output.mkdir()
    checkpoint = {'checkpoint_id': 'synthetic-test-checkpoint'}
    raw = json.dumps(checkpoint).encode(); (output/'checkpoint.json').write_bytes(raw)
    summary = dict(status='complete', checkpoint_id=checkpoint['checkpoint_id'],
                   checkpoint_file_sha256=hashlib.sha256(raw).hexdigest(),
                   checkpoint_unchanged_after_test=True, split_manifest_unchanged_after_test=True,
                   physically_verified=False, test={'frozen_policy': {'captures': 0, 'total': 2}})
    (output/'summary.json').write_text(json.dumps(summary))
    (output/'episodes.json').write_text(json.dumps({'synthetic_test_fixture': True}))
    return output


def wait_done(manager):
    deadline = time.monotonic()+3
    while manager.status()['phase'] in ('starting','running','canceling'):
        if time.monotonic() > deadline: raise AssertionError('fake job did not stop')
        time.sleep(.01)
    return manager.status()


class TrainingManagerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
    def tearDown(self): self.directory.cleanup()
    def manager(self, factory, **kwargs):
        manager = TrainingJobs(self.root, _popen=factory, **kwargs)
        manager._stop_process = lambda child: setattr(child, 'returncode', -15)
        return manager

    def test_numeric_limits_and_unknown_input(self):
        self.assertEqual(configuration({})['train_scenes'], 4)
        for body in ({'seed': True}, {'seed': -1}, {'train_scenes': 3}, {'test_scenes': 33},
                     {'command': 'shell'}, {'output': '/tmp/override'},
                     {'train_scenes':64,'teacher_budget':32}):
            with self.subTest(body=body), self.assertRaises(ValueError): configuration(body)

    def test_complete_job_uses_fixed_process_and_scrubbed_environment(self):
        seen=[]
        def factory(command, **kwargs):
            seen.append((command, kwargs)); complete_files(command)
            return FakeProcess(stdout='{"phase":"teacher","completed":1,"total":4}\n')
        manager=self.manager(factory)
        with patch.dict('os.environ', {'SIRCUS_AUDIO_TOKEN':'TEST_SECRET','VIAM_API_KEY':'TEST_SECRET'}):
            first=manager.start({}); state=wait_done(manager)
        self.assertEqual(state['phase'],'complete')
        self.assertEqual(state['progress']['phase'],'teacher')
        self.assertEqual(state['run_id'],first['run_id'])
        self.assertTrue(seen[0][0][1].endswith('/sircus/two-arm/train_scene_study.py'))
        self.assertNotIn('SIRCUS_AUDIO_TOKEN',seen[0][1]['env'])
        self.assertNotIn('VIAM_API_KEY',seen[0][1]['env'])
        self.assertTrue(seen[0][1]['start_new_session'])
        self.assertNotIn('shell',seen[0][1])
        self.assertEqual(manager.artifact(state['run_id'],'checkpoint.json')['checkpoint_id'],
                         'synthetic-test-checkpoint')

    def test_relative_output_root_is_absolute_before_child_changes_cwd(self):
        import os
        relative = Path(os.path.relpath(self.root, Path.cwd()))
        def factory(command, **kwargs):
            output = Path(command[command.index('--output')+1])
            self.assertTrue(output.is_absolute())
            self.assertEqual(output.parent, self.root.resolve())
            self.assertNotEqual(Path(kwargs['cwd']), output.parent)
            complete_files(command)
            return FakeProcess()
        manager = TrainingJobs(relative, profile='front', _popen=factory)
        self.addCleanup(manager.close)
        manager.start({})
        state = wait_done(manager)
        self.assertEqual(state['phase'], 'complete')
        self.assertEqual(manager.artifact(state['run_id'], 'checkpoint.json')['checkpoint_id'],
                         'synthetic-test-checkpoint')

    def test_cancel_rejects_concurrent_start_and_preserves_partial_evidence(self):
        children=[]
        def factory(command, **kwargs):
            out=Path(command[command.index('--output')+1]);out.mkdir()
            (out/'episodes.json').write_text('{}')
            child=FakeProcess(None,stdout='{"phase":"teacher"}\n');children.append(child);return child
        manager=self.manager(factory)
        first=manager.start({})
        with self.assertRaisesRegex(ValueError,'already active'):manager.start({})
        manager.cancel(); state=wait_done(manager)
        self.assertEqual(state['phase'],'canceled')
        self.assertIsNone(state['summary']);self.assertEqual(state['artifacts'],[])
        self.assertEqual(children[0].returncode,-15)
        self.assertTrue((self.root/first['run_id']/'episodes.json').exists())
        with self.assertRaisesRegex(ValueError,'incomplete'):manager.artifact(first['run_id'],'episodes.json')

    def test_close_waits_for_owned_child_and_leaves_no_selectable_checkpoint(self):
        child=FakeProcess(None)
        manager=self.manager(lambda *a,**k:child)
        manager.start({});result=manager.close()
        self.assertEqual(child.returncode,-15)
        self.assertEqual(result['phase'],'canceled')
        self.assertFalse(manager._thread.is_alive())
        self.assertEqual(result['artifacts'],[])

    def test_cancel_during_final_artifact_hashing_wins_over_completion(self):
        def factory(command, **kwargs):complete_files(command);return FakeProcess()
        manager=self.manager(factory)
        original=manager._list_artifacts
        def cancel_at_finish(run_id):
            artifacts=original(run_id);manager.cancel();return artifacts
        manager._list_artifacts=cancel_at_finish
        manager.start({});state=wait_done(manager)
        self.assertEqual(state['phase'],'canceled')
        self.assertIsNone(state['summary']);self.assertEqual(state['artifacts'],[])
        with self.assertRaises(ValueError):manager.artifact(state['run_id'],'checkpoint.json')

    def test_failure_never_promotes_partial_checkpoint_or_leaks_stderr(self):
        def factory(command, **kwargs):
            complete_files(command)
            return FakeProcess(2,stderr='PRIVATE_TEST_DIAGNOSTIC /host/location\n')
        manager=self.manager(factory);first=manager.start({});state=wait_done(manager)
        self.assertEqual(state['phase'],'failed');self.assertIsNone(state['summary'])
        self.assertNotIn('PRIVATE_TEST_DIAGNOSTIC',json.dumps(state))
        with self.assertRaises(ValueError):manager.artifact(first['run_id'],'checkpoint.json')
        self.assertIn('PRIVATE_TEST_DIAGNOSTIC',(self.root/f"{first['run_id']}.stderr.log").read_text())

    def test_timeout_stops_only_owned_job(self):
        child=FakeProcess(None)
        manager=self.manager(lambda *a,**k:child,_max_seconds=.03)
        manager.start({});state=wait_done(manager)
        self.assertEqual(state['phase'],'failed');self.assertIn('time limit',state['error'])
        self.assertEqual(child.returncode,-15)

    def test_output_limit_terminates_job_without_serving_checkpoint(self):
        child=FakeProcess(None,stdout='x'*65_537)
        manager=self.manager(lambda *a,**k:child)
        manager.start({});state=wait_done(manager)
        self.assertEqual(state['phase'],'failed');self.assertIn('output',state['error'])
        self.assertEqual(child.returncode,-15)

    def test_artifacts_require_safe_names_completed_manifest_and_original_bytes(self):
        def factory(command, **kwargs):complete_files(command);return FakeProcess()
        manager=self.manager(factory);manager.start({});state=wait_done(manager);rid=state['run_id']
        for run,name in [('../outside','checkpoint.json'),(rid,'../summary.json'),(rid,'stdout.log')]:
            with self.assertRaises(ValueError):manager.artifact(run,name)
        (self.root/rid/'checkpoint.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'changed'):manager.artifact(rid,'checkpoint.json')
        target=self.root/rid/'summary.json';target.unlink();target.symlink_to(self.root/rid/'episodes.json')
        with self.assertRaisesRegex(ValueError,'unavailable'):manager.artifact(rid,'summary.json')

    def test_metadata_storage_failure_does_not_leave_job_active(self):
        manager=self.manager(lambda *a,**k:FakeProcess())
        manager._persist=lambda: (_ for _ in ()).throw(OSError('disk full'))
        with self.assertRaisesRegex(ValueError,'metadata'):manager.start({})
        self.assertEqual(manager.status()['phase'],'failed')
        def factory(command, **kwargs):complete_files(command);return FakeProcess()
        manager=self.manager(factory)
        original=manager._persist
        def fail_completed():
            if manager._state['phase']=='complete':raise OSError('disk full')
            original()
        manager._persist=fail_completed
        manager.start({});state=wait_done(manager)
        self.assertEqual(state['phase'],'failed');self.assertEqual(state['artifacts'],[])
        with self.assertRaises(ValueError):manager.artifact(state['run_id'],'checkpoint.json')

    def test_diagnostic_disk_failure_does_not_claim_completion(self):
        def factory(command, **kwargs):complete_files(command);return FakeProcess()
        manager=self.manager(factory)
        original=Path.write_text
        def fail_logs(path, *args, **kwargs):
            if path.suffix=='.log':raise OSError('disk full')
            return original(path,*args,**kwargs)
        with patch.object(Path,'write_text',fail_logs):
            manager.start({});state=wait_done(manager)
        self.assertEqual(state['phase'],'failed');self.assertEqual(state['artifacts'],[])
        self.assertIn('persist',state['error'])

    def test_bad_completion_hash_is_not_complete(self):
        def factory(command, **kwargs):
            output=complete_files(command);(output/'checkpoint.json').write_text('{}')
            return FakeProcess()
        manager=self.manager(factory);manager.start({});self.assertEqual(wait_done(manager)['phase'],'failed')

    def test_progress_retention_and_repeated_jobs_are_bounded_and_isolated(self):
        def factory(command, **kwargs):
            complete_files(command)
            return FakeProcess(stdout=''.join(json.dumps({'phase':'teacher','completed':n})+'\n' for n in range(100)))
        manager=self.manager(factory);manager.start({});first=wait_done(manager)
        self.assertEqual(len(first['recent_progress']),64)
        self.assertEqual(first['progress']['completed'],99)
        manager.start({'seed':123});second=wait_done(manager)
        self.assertNotEqual(first['run_id'],second['run_id'])
        self.assertEqual(manager.artifact(first['run_id'],'summary.json')['status'],'complete')
        self.assertEqual(manager.cancel()['phase'],'complete')


if __name__=='__main__':unittest.main()
