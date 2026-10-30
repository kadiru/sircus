import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from bridge import Controller
from sircus.audio.show import default_config
from sircus.audio.contract import AudioEvent

class FakeNative:
    def __init__(self):self.commands=[]
    def send(self,**value):self.commands.append(value)

class MultiBallTests(unittest.TestCase):
    def setUp(self):
        self.now=10.;self.native=FakeNative();self.c=Controller(self.native,clock=lambda:self.now);self.c.ready=True
        self.config=default_config()
        for i,b in enumerate(self.config['balls']):b['midi_binding']={'input_id':100+i,'channel':1,'note':0}
        self.c.configure_show(self.config);self.c.enable(True)
    def event(self,kind,trial,ball,t=1.,**kw):
        e=AudioEvent(kw.pop('event_id',kind+str(t)),trial,'operator',kind,'unknown',t,'host_monotonic_s','operator',
                     'Explicit test action','operator_action',ball_id=ball,**kw)
        return self.c.audio_event(e)
    def plays(self):return [x for x in self.native.commands if x['cmd']=='play']
    def test_four_inputs_explicit_and_reconnect_mutes(self):
        self.assertIn({'cmd':'inputs','values':[100,101,102,103]},self.native.commands)
        self.c.native({'type':'connection','connected':True,'connected_inputs':[100,101]})
        self.assertFalse(self.c.enabled)
    def test_prior_pink_impact_keeps_identity_does_not_stop_purple_drums(self):
        self.event('release','pink-trial','ball-pink')
        self.event('carry_start','purple-trial','ball-purple',2.)
        before=len(self.native.commands)
        accepted=self.c.midi({'input':100,'status':144,'data':[0,90]})
        self.assertTrue(accepted['accepted']);self.assertEqual(self.c.transport_trial,'purple-trial')
        self.assertNotIn('stop_transport',[x['cmd'] for x in self.native.commands[before:]])
        self.assertEqual(self.plays()[-1]['sound'],self.config['balls'][0]['sounds']['unknown'])
        log=[x for x in self.c.monitor if x['kind']=='event'][-1]
        self.assertEqual(log['trial_id'],'pink-trial');self.assertEqual(log['ball_id'],'ball-pink')
        self.assertEqual(log['contact_object'],'unknown');self.assertEqual(log['evidence'],'midi_gesture')
    def test_bound_four_releases_and_unbound_source_rejected(self):
        for i,b in enumerate(self.config['balls']):
            self.event('release','trial'+str(i),b['ball_id'],float(i))
        for i in range(4):self.assertTrue(self.c.midi({'input':100+i,'status':144,'data':[0,90]})['accepted'])
        self.assertFalse(self.c.midi({'input':999,'status':144,'data':[0,90]})['accepted'])
        self.assertEqual(len(self.plays()),4)
    def test_duplicate_delayed_cancel_and_expiry(self):
        self.assertTrue(self.event('release','t','ball-pink',2.)['accepted'])
        self.assertFalse(self.event('release','t','ball-pink',2.)['accepted'])
        self.assertFalse(self.event('carry_start','t','ball-pink',1.)['accepted'])
        self.c.stop('t');self.assertFalse(self.c.midi({'input':100,'status':144,'data':[0,90]})['accepted'])
        self.event('release','fresh','ball-pink',3.);self.now+=31
        self.assertFalse(self.c.midi({'input':100,'status':144,'data':[0,90]})['accepted'])
    def test_cancel_prior_ball_does_not_stop_current_transport(self):
        self.event('release','old','ball-pink')
        self.event('carry_start','current','ball-purple',2.)
        before=len(self.native.commands);self.c.stop('old')
        commands=self.native.commands[before:]
        self.assertEqual(self.c.transport_trial,'current')
        self.assertNotIn('stop_all',[c['cmd'] for c in commands])
        self.assertNotIn('stop_transport',[c['cmd'] for c in commands])
        self.assertIn({'cmd':'stop_trial','trial_id':'old'},commands)
        self.assertFalse(self.c.midi({'input':100,'status':144,'data':[0,90]})['accepted'])
        before=len(self.native.commands);self.c.stop('old');self.c.stop('not-arrived')
        self.assertNotIn('stop_all',[c['cmd'] for c in self.native.commands[before:]])
        self.assertEqual(self.c.transport_trial,'current')
    def test_repeated_release_does_not_extend_midi_window(self):
        self.event('release','t','ball-pink',1.)
        self.now+=29
        self.assertFalse(self.event('release','t','ball-pink',2.)['accepted'])
        self.now+=2
        self.assertIsNone(self.c.show_router.midi_identity(100,1,0))
    def test_unknown_outcome_and_fx_off_are_silent(self):
        self.event('release','t','ball-pink');before=len(self.plays());self.event('outcome','t','ball-pink',2.)
        self.assertEqual(len(self.plays()),before)
        self.c.enable(False);self.c.midi({'input':100,'status':144,'data':[0,90]})
        self.assertEqual(len(self.plays()),before);self.assertEqual(self.native.commands[-1]['cmd'],'panic')

if __name__=='__main__':unittest.main()
