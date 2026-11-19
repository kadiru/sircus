"""Show routes are audio-only and never accept device credentials or robot actions."""
import io
import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock,patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import server

class ShowAPI(unittest.TestCase):
    def setUp(self):
        self.audio=Mock();self.audio.status.return_value={'config':{'balls':[]},'capabilities':{'contact_detection':False}}
        self.ops=Mock();self.ops.status.return_value={'phase':'complete','busy':False,'trial_id':'finished-1',
           'hardware_enabled':False,'result':{'result':{'synthetic':'trusted fixture'}}}
        self.patch=patch.multiple(server,SHOW_ENABLED=True,SHOW_AUDIO=self.audio,SHOW_TOKEN='TEST_SHOW',
            OFFLINE=True,PHYSICAL=None,OPERATIONS=self.ops,TRAINING_JOBS=Mock())
        self.patch.start();self.addCleanup(self.patch.stop)
    def request(self,path='/api/show/action',payload=None,*,token='TEST_SHOW',host='127.0.0.1:18769',origin='http://127.0.0.1:18769',raw=None):
        h=server.Handler.__new__(server.Handler);h.path=path
        h.server=types.SimpleNamespace(server_address=('127.0.0.1',18769));h.send=Mock()
        h.headers={'Host':host,'Origin':origin,'Content-Type':'application/json','X-Sircus-Show-Token':token}
        if payload is None and raw is None:h.do_GET()
        else:
            data=raw if raw is not None else json.dumps(payload).encode();h.headers['Content-Length']=str(len(data));h.rfile=io.BytesIO(data);h.do_POST()
        a=h.send.call_args.args;return a[0],a[1] if len(a)>1 else 200
    def test_status_distinguishes_mock_physical_and_numerical(self):
        r,code=self.request('/api/show/status');self.assertEqual(code,200)
        self.assertEqual(r['modes']['deterministic']['mode'],'disabled');self.assertTrue(r['modes']['simulator']['available'])
        self.assertEqual(set(r['modes']), {'deterministic','simulator','trained','mock'})
        self.assertTrue(r['modes']['mock']['available'])
        self.assertEqual(r['modes']['mock']['planner'], 'facing_search')
        self.assertEqual(r['modes']['trained']['planner'], 'facing_learned')
        self.assertFalse(r['numerical_busy'])
        self.ops.status.return_value['busy']=True
        self.assertTrue(self.request('/api/show/status')[0]['numerical_busy'])
        with patch.object(server,'PHYSICAL',types.SimpleNamespace(mode='hardware')):
            r,code=self.request('/api/show/status');self.assertTrue(r['modes']['deterministic']['available'])
            self.assertFalse(r['modes']['trained']['available']);self.assertFalse(r['modes']['simulator']['available'])
            self.assertFalse(r['modes']['mock']['available']);self.assertFalse(r['numerical_busy'])
    def test_host_origin_token_before_actions(self):
        for opts in [{'token':''},{'host':'evil.invalid'},{'origin':'https://evil.invalid'}]:
            self.assertEqual(self.request(payload={'action':'stop_audio_replay'},**opts)[1],403)
        self.assertEqual(self.audio.mock_calls,[])
    def test_no_manual_impacts_or_arbitrary_events(self):
        for action in ['preview_contact','manual_impact','emit','move_arm','open_gripper']:
            self.assertEqual(self.request(payload={'action':action})[1],400)
        self.assertEqual(self.audio.mock_calls,[])
    def test_configure_and_cancel_never_call_robot_or_worker(self):
        self.assertEqual(self.request(payload={'action':'configure_audio','config':{'fixture':True}})[1],202)
        self.audio.configure.assert_called_once_with({'fixture':True})
        self.assertEqual(self.request(payload={'action':'stop_audio_replay'})[1],202)
        self.audio.cancel_replay.assert_called_once();self.ops.action.assert_not_called()
    def test_replay_uses_only_current_server_owned_synthetic_result(self):
        self.assertEqual(self.request(payload={'action':'replay_simulation','trial_id':'finished-1','ball_id':'ball-pink'})[1],202)
        self.audio.start_replay.assert_called_once_with({'synthetic':'trusted fixture'},'finished-1','ball-pink')
        self.ops.action.assert_not_called()
        for changes in [{'trial_id':'old-trial'},{'result':{'fake':'input'}}]:
            self.assertEqual(self.request(payload={'action':'replay_simulation','trial_id':'finished-1','ball_id':'ball-pink',**changes})[1],400)
        self.assertEqual(self.audio.start_replay.call_count,1)
    def test_hardware_busy_missing_and_foreign_provenance_rejected(self):
        with patch.object(server,'PHYSICAL',types.SimpleNamespace(mode='hardware')):
            self.assertEqual(self.request(payload={'action':'replay_simulation','trial_id':'finished-1','ball_id':'ball-pink'})[1],400)
        self.ops.status.return_value['busy']=True
        self.assertEqual(self.request(payload={'action':'replay_simulation','trial_id':'finished-1','ball_id':'ball-pink'})[1],400)
        self.ops.status.return_value.update(busy=False,hardware_enabled=True)
        self.assertEqual(self.request(payload={'action':'replay_simulation','trial_id':'finished-1','ball_id':'ball-pink'})[1],400)
        self.audio.start_replay.assert_not_called()
    def test_audio_failure_is_sanitized_and_never_robot_fallback(self):
        self.audio.configure.side_effect=RuntimeError('private token never echo')
        r,code=self.request(payload={'action':'configure_audio','config':{}})
        self.assertEqual(code,503);self.assertNotIn('private',json.dumps(r));self.ops.action.assert_not_called()
        with patch.object(server,'SHOW_ENABLED',False):self.assertEqual(self.request('/api/show/status')[1],503)
        self.assertEqual(self.request(raw=b'{"action":NaN}')[1],400)
        self.assertEqual(self.request(raw=b' '*24001)[1],413)
if __name__=='__main__':unittest.main()
