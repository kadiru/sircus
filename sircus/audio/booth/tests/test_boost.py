"""Loudness control with fake native messages; no audio service or playback."""
import io
import json
from pathlib import Path
import sys
import types
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from bridge import Controller,Handler,MAX_BOOST_DB,ROOT


class Backend:
    def __init__(self):self.commands=[];self.failure=False
    def send(self,**command):
        if self.failure:raise RuntimeError('private backend diagnostic')
        self.commands.append(command)


class BoostTests(unittest.TestCase):
    def setUp(self):
        self.backend=Backend();self.controller=Controller(self.backend)

    def ready(self):
        self.controller.native({'type':'ready','audio_ready':True,'midi_status':[],
                                'peak_limiter_ready':True,'boost_db':0.})

    def ack(self,value,ready=True):
        self.controller.native({'type':'boost','boost_db':value,'peak_limiter_ready':ready})

    def post(self,payload):
        raw=json.dumps(payload).encode();reply=[]
        handler=object.__new__(Handler)
        handler.server=types.SimpleNamespace(c=self.controller)
        handler.path='/api/boost';handler.headers={'Content-Type':'application/json','Content-Length':str(len(raw))}
        handler.rfile=io.BytesIO(raw);handler.authorized=lambda:True
        handler.reply=lambda value,status=200:reply.append((status,value))
        handler.do_POST()
        return reply[0]

    def test_defaults_normal_off_and_does_not_send_commands(self):
        status=self.controller.status()
        self.assertEqual(status['boost_db'],0.)
        self.assertEqual(status['boost_requested_db'],0.)
        self.assertFalse(status['boost_pending'])
        self.assertFalse(status['peak_limiter_ready'])
        self.assertFalse(status['fx_enabled'])
        self.assertEqual(self.backend.commands,[])

    def test_request_preserves_mute_until_native_ack(self):
        self.ready()
        status,result=self.post({'boost_db':MAX_BOOST_DB})
        self.assertEqual(status,200)
        self.assertEqual(self.backend.commands,[{'cmd':'boost','value':MAX_BOOST_DB}])
        self.assertEqual(result['boost_db'],0.)
        self.assertEqual(result['boost_requested_db'],MAX_BOOST_DB)
        self.assertTrue(result['boost_pending']);self.assertFalse(result['fx_enabled'])
        self.ack(MAX_BOOST_DB)
        result=self.controller.status()
        self.assertEqual(result['boost_db'],MAX_BOOST_DB)
        self.assertFalse(result['boost_pending']);self.assertFalse(result['fx_enabled'])
        self.assertFalse(any(command['cmd'] in ('enable','play') for command in self.backend.commands))

    def test_bad_input_rejected_without_command_or_state_change(self):
        self.ready()
        for value in (None,True,False,'3',float('nan'),float('inf'),-1,MAX_BOOST_DB+.0001):
            with self.subTest(value=value):
                status,result=self.post({'boost_db':value})
                self.assertEqual(status,400);self.assertIn('error',result)
                self.assertEqual(self.controller.status()['boost_db'],0.)
                self.assertFalse(self.controller.boost_pending)
        self.assertEqual(self.post({'boost_db':3,'volume':1})[0],400)
        self.assertEqual(self.backend.commands,[])

    def test_missing_backend_audio_or_limiter_blocks_positive_gain(self):
        self.assertEqual(self.post({'boost_db':3})[0],400)
        self.controller.ready=True
        self.assertEqual(self.post({'boost_db':3})[0],400)
        self.controller.peak_limiter_ready=True;self.controller.backend=None
        self.assertEqual(self.post({'boost_db':3})[0],400)
        self.assertEqual(self.backend.commands,[])
        self.assertFalse(self.controller.boost_pending)

    def test_return_to_normal_requires_ready_path_and_keeps_volume(self):
        volume=self.controller.volume
        self.assertEqual(self.post({'boost_db':0})[0],400)
        self.assertEqual(self.backend.commands,[])
        self.ready()
        status,result=self.post({'boost_db':0})
        self.assertEqual(status,200);self.assertEqual(result['volume'],volume)
        self.assertEqual(self.backend.commands,[{'cmd':'boost','value':0.}])
        self.ack(0,True)
        self.assertFalse(self.controller.boost_pending);self.assertFalse(self.controller.enabled)

    def test_logic_route_reports_unsupported_without_mutation(self):
        self.ready();self.controller.settings('route',{'route':'logic','commissioning_acknowledged':True})
        before=list(self.backend.commands)
        status,result=self.post({'boost_db':3})
        self.assertEqual(status,400);self.assertIn('native',result['error'])
        self.assertEqual(self.backend.commands,before)
        self.assertEqual(self.controller.boost_db,0.)

    def test_backend_failure_cannot_claim_gain_and_hides_diagnostic(self):
        self.ready();self.backend.failure=True
        status,result=self.post({'boost_db':6})
        self.assertEqual(status,503)
        self.assertNotIn('private',result['error'])
        self.assertEqual(self.controller.boost_db,0.)
        self.assertEqual(self.controller.boost_requested_db,0.)
        self.assertFalse(self.controller.boost_pending)

    def test_limiter_failure_cannot_claim_positive_boost(self):
        self.ready();self.controller.settings('boost',{'boost_db':6});self.ack(6)
        self.controller.enable(True)
        self.controller.native({'type':'meter','peak':0,'audio_ready':False,
                                'peak_limiter_ready':False,'boost_db':6})
        state=self.controller.status()
        self.assertFalse(state['fx_enabled']);self.assertFalse(state['audio_ready'])
        self.assertFalse(state['peak_limiter_ready']);self.assertEqual(state['boost_db'],0.)
        self.assertFalse(state['boost_pending'])
        self.assertIn('boost not verified',state['last_error'])
        self.assertEqual(self.backend.commands[-1],{'cmd':'panic'})

    def test_native_exit_and_restart_clear_confirmation(self):
        self.ready();self.controller.settings('boost',{'boost_db':3});self.ack(3)
        self.controller.native({'type':'native_exit','message':'test exit'})
        state=self.controller.status()
        self.assertEqual(state['boost_db'],0.);self.assertFalse(state['peak_limiter_ready'])
        self.assertFalse(state['fx_enabled']);self.assertFalse(state['boost_pending'])
        self.ready();self.assertEqual(self.controller.boost_requested_db,0.)

    def test_mute_and_midi_reconnect_do_not_enable_or_forget_applied_native_gain(self):
        self.ready();self.controller.settings('boost',{'boost_db':3});self.ack(3)
        self.controller.enable(True);self.controller.enable(False)
        self.controller.native({'type':'connection','connected':False})
        self.controller.native({'type':'connection','connected':True})
        self.assertFalse(self.controller.enabled);self.assertEqual(self.controller.boost_db,3.)
        self.assertEqual(self.controller.boost_requested_db,3.)

    def test_old_ack_keeps_new_request_pending(self):
        self.ready();self.controller.settings('boost',{'boost_db':3})
        self.controller.settings('boost',{'boost_db':6})
        self.ack(3)
        self.assertEqual(self.controller.boost_db,3.);self.assertTrue(self.controller.boost_pending)
        self.ack(6);self.assertEqual(self.controller.boost_db,6.);self.assertFalse(self.controller.boost_pending)

    def test_static_boost_control_module_is_available_as_javascript(self):
        handler=object.__new__(Handler);headers=[];statuses=[]
        handler.path='/boost-controls.mjs';handler.authorized=lambda:True
        handler.send_response=statuses.append
        handler.send_header=lambda name,value:headers.append((name,value))
        handler.end_headers=lambda:None;handler.wfile=io.BytesIO()
        handler.do_GET()
        self.assertEqual(statuses,[200])
        self.assertIn(('Content-Type','text/javascript'),headers)
        self.assertEqual(handler.wfile.getvalue(),(ROOT/'web/boost-controls.mjs').read_bytes())


if __name__=='__main__':unittest.main()
