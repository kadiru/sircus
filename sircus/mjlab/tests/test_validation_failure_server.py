"""Read-only reader checks; no native simulation, sockets or hardware."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from sircus.mjlab import validation_failure_server as server
from sircus.mjlab.interface import RequestError


class ValidationFailureServerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / 'source'; self.source.mkdir()
        self.output = self.root / 'diagnostic'; self.output.mkdir()
        (self.output / 'replays').mkdir()
        for name in ('report.json', 'manifest.json'):
            (self.source / name).write_text('{}')
        self.audit_path = self.root / 'audit.json'
        self.audit = dict(contract_fingerprint='a'*64, report_sha256=server.sha(self.source/'report.json'),
            study_manifest_sha256=server.sha(self.source/'manifest.json'), source_sha256={'producer':'b'*64},
            auditor_sha256='c'*64, auditor_dependencies={'dependency':'d'*64})
        self.audit_path.write_text(json.dumps(self.audit))
        self.replay_id = 'validation-0123456789abcdef'
        self.replay = dict(version='sircus-recorded-validation-replay-v1',id=self.replay_id, role='validation',
            source=dict(self.audit,audit_sha256=server.sha(self.audit_path)),source_row_id='original-validation-id',
            result={'request_id':'original-validation-id','method':'validation_recorded_replay'})
        self.replay_path = self.output/'replays'/(self.replay_id+'.json')
        self.replay_path.write_text(json.dumps(self.replay))
        self.value = dict(version=server.VERSION, status='passed', synthetic_fixture=False,
            source=dict(self.audit, audit_sha256=server.sha(self.audit_path)),
            representatives=[dict(id=self.replay_id,source_row_id='original-validation-id',recorded_replay_available=True,sha256=server.sha(self.replay_path))])
        self.state = object.__new__(server.ValidationFailureState)
        self.state.benchmark = self.source; self.state.benchmark_audit = self.audit_path
        self.state.benchmark_contract = {'fingerprint':'a'*64}
        self.state.diagnostic_dir = self.output
        self.state.benchmark_report = Mock(return_value=dict(status='completed',test_disclosure='verified',audit={'status':'passed'}))
        self.write_report()

    def tearDown(self):
        self.temp.cleanup()

    def write_report(self):
        self.value.pop('fingerprint',None)
        self.value['fingerprint'] = server.digest(self.value)
        path=self.output/'report.json'; path.write_text(json.dumps(self.value))
        self.state.diagnostic_sha256=server.sha(path)

    def test_reads_pinned_report_and_validation_replay(self):
        self.assertEqual(self.state.diagnostic_report(),self.value)
        self.assertEqual(self.state.validation_replay(self.replay_id),self.replay['result'])

    def test_incomplete_or_unaudited_original_stays_hidden(self):
        for value in (dict(status='running',test_disclosure='verified',audit={'status':'passed'}),
                      dict(status='completed',test_disclosure='hidden_until_completed_and_audited',audit={'status':'passed'}),
                      dict(status='completed',test_disclosure='verified',audit={'status':'pending'})):
            self.state.benchmark_report.return_value=value
            with self.subTest(value=value),self.assertRaises(RequestError):self.state.diagnostic_report()

    def test_original_or_derived_bytes_changed(self):
        for path in (self.source/'report.json',self.source/'manifest.json',self.audit_path,self.output/'report.json'):
            original=path.read_bytes();path.write_bytes(original+b' ')
            with self.subTest(path=path),self.assertRaises(RequestError):self.state.diagnostic_report()
            path.write_bytes(original)

    def test_changed_fingerprint_or_source_cannot_be_served(self):
        original=copy.deepcopy(self.value)
        for key in self.audit:
            self.value=copy.deepcopy(original);self.value['source'][key]='f'*64;self.write_report()
            with self.subTest(key=key),self.assertRaises(RequestError):self.state.diagnostic_report()
        self.value=copy.deepcopy(original);self.write_report()
        self.value['fingerprint']='f'*64
        path=self.output/'report.json';path.write_text(json.dumps(self.value));self.state.diagnostic_sha256=server.sha(path)
        with self.assertRaises(RequestError):self.state.diagnostic_report()

    def test_rejects_synthetic_or_wrong_version_diagnostic(self):
        original=copy.deepcopy(self.value)
        for change in ({'status':'running'},{'synthetic_fixture':True},{'version':'old'}):
            self.value=dict(copy.deepcopy(original),**change);self.write_report()
            with self.subTest(change=change),self.assertRaises(RequestError):self.state.diagnostic_report()

    def test_replay_unknown_path_tampering_and_nonvalidation_refused(self):
        for identity in ('../../report','unknown-12345678',self.replay_id+'/test'):
            with self.subTest(identity=identity),self.assertRaises(RequestError):self.state.validation_replay(identity)
        self.replay_path.write_text('{}')
        with self.assertRaises(RequestError):self.state.validation_replay(self.replay_id)
        for role in ('train','test'):
            self.replay['role']=role;self.replay_path.write_text(json.dumps(self.replay))
            self.value['representatives'][0]['sha256']=server.sha(self.replay_path);self.write_report()
            with self.subTest(role=role),self.assertRaises(RequestError):self.state.validation_replay(self.replay_id)

    def test_all_post_commands_rejected_without_state_calls(self):
        state=Mock()
        handler=object.__new__(server.validation_handler(state));handler._json=Mock()
        handler.do_POST()
        self.assertEqual(handler._json.call_args.args[0],405)
        self.assertEqual(state.mock_calls,[])

    def test_reader_state_must_be_new_and_outside_all_evidence(self):
        for state in (self.source,self.source/'new',self.output/'new',self.root):
            with self.subTest(state=state),self.assertRaises(ValueError):
                server.validate_state_directory(state,[self.source,self.output])
        server.validate_state_directory(self.root/'fresh',[self.source,self.output])

    def test_malformed_metadata_fails_closed(self):
        original=copy.deepcopy(self.value)
        for change in ({'source':None},{'representatives':None},{'representatives':[None]}):
            self.value=dict(copy.deepcopy(original),**change);self.write_report()
            with self.subTest(change=change),self.assertRaises(RequestError):self.state.diagnostic_report()

    def test_rejected_request_has_no_fabricated_native_replay(self):
        self.value['representatives'][0].pop('sha256')
        self.value['representatives'][0]['recorded_replay_available']=False
        self.write_report()
        self.assertEqual(self.state.diagnostic_report(),self.value)
        with self.assertRaises(RequestError) as caught:self.state.validation_replay(self.replay_id)
        self.assertEqual(caught.exception.status,404)


if __name__=='__main__':unittest.main()
