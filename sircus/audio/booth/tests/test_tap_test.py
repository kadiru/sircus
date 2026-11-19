import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from bridge import Controller
from sircus.audio.show import default_config
from sircus.audio.contract import AudioEvent

class Native:
    def __init__(self):self.commands=[]
    def send(self,**kw):self.commands.append(kw)

class Timer:
    def __init__(self,seconds,callback):self.seconds=seconds;self.callback=callback;self.cancelled=False
    def start(self):pass
    def cancel(self):self.cancelled=True

class TapTest(unittest.TestCase):
    def setUp(self):
        self.now=100.;self.native=Native();self.c=Controller(self.native,clock=lambda:self.now,timer_factory=Timer)
        self.c.ready=True;self.cfg=default_config()
        for i,b in enumerate(self.cfg['balls']):b['midi_binding']={'input_id':i+100,'channel':1,'note':0}
        self.c.configure_show(self.cfg)
        self.c.native({'type':'connection','connected':True,'selected_inputs':[100,101,102,103],'connected_inputs':[100,101,102,103]})
        self.seq=0
    def command(self,**kw):
        self.seq+=1
        return {'action':'start','control_client_id':'tester','control_sequence':self.seq,
                'control_epoch':self.c.control_epoch,'show_config_digest':self.c.show_readiness()['config_digest'],**kw}
    def start(self,**kw):return self.c.control_tap_test(self.command(**kw))
    def tap(self,port=100,status=144,data=None,source='hardware'):
        return self.c.midi({'input':port,'status':status,'data':data or [0,90]},source)
    def plays(self):return [c for c in self.native.commands if c['cmd']=='play']
    def test_four_balls_count_and_play_without_fabricating_release(self):
        s=self.start();self.assertEqual(s['tap_test']['duration_s'],15);self.assertFalse(self.plays())
        for i in range(4):self.assertTrue(self.tap(100+i)['accepted'])
        self.assertEqual([b['taps'] for b in self.c.status()['tap_test']['balls']],[1]*4)
        self.assertEqual([p['sound'] for p in self.plays()],[b['sounds']['unknown'] for b in self.cfg['balls']])
        self.assertEqual(self.c.show_router.trials,{})
        events=[r['event'] for r in self.c.monitor if r['kind']=='tap_test_impact']
        self.assertTrue(all(e['source']=='sensor' and e['evidence']=='midi_gesture' and e['contact_object']=='unknown' and e['outcome']=='unknown' for e in events))
        self.assertEqual(len({e['event_id'] for e in events}),4)
    def test_unmapped_nonimpact_and_synthetic_packets_do_not_pass(self):
        self.start()
        for args in ({'port':999},{'data':[1,90]},{'data':[0,0]},{'status':128},{'status':176},{'source':'mock'}):self.tap(**args)
        self.assertFalse(self.plays());self.assertEqual(sum(b['taps'] for b in self.c.status()['tap_test']['balls']),0)
    def test_server_timer_mutes_without_browser_polling(self):
        self.start(duration_s=2);timer=self.c.tap_timer;self.tap();self.now+=2;timer.callback()
        self.assertFalse(self.c.enabled);self.assertFalse(self.c.tap_test['active']);self.assertTrue(timer.cancelled)
        before=len(self.plays());self.tap();self.assertEqual(len(self.plays()),before)
    def test_late_packet_cannot_extend_deadline(self):
        self.start();self.now+=15;self.tap();self.assertFalse(self.plays());self.assertFalse(self.c.enabled)
    def test_explicit_stop_and_delayed_start_are_ordered(self):
        old=self.command();self.c.control_tap_test({'action':'stop','control_client_id':'tester','control_sequence':2})
        self.assertFalse(self.c.control_tap_test(old)['accepted']);self.assertFalse(self.c.enabled)
    def test_duplicate_start_does_not_reset_timer_or_counts(self):
        cmd=self.command();self.c.control_tap_test(cmd);timer=self.c.tap_timer;self.tap();self.now+=5
        self.assertFalse(self.c.control_tap_test(cmd)['accepted']);self.assertIs(self.c.tap_timer,timer)
        self.assertEqual(self.c.status()['tap_test']['remaining_s'],10)
    def test_replaced_timer_does_not_mute_new_test(self):
        self.start();old=self.c.tap_timer;self.tap();s=self.start(duration_s=30);old.callback()
        self.assertTrue(self.c.enabled);self.assertTrue(self.c.tap_test['active']);self.assertEqual(s['tap_test']['balls'][0]['taps'],0)
    def test_reconnect_cancels_and_requires_fresh_explicit_start(self):
        self.start();self.tap();self.c.native({'type':'connection','connected':True,'selected_inputs':[100,101,102,103],'connected_inputs':[101,102,103]})
        s=self.c.status();self.assertFalse(s['fx_enabled']);self.assertFalse(s['tap_test']['active']);self.assertFalse(s['tap_test']['balls'][0]['connected'])
        self.start();self.assertFalse(self.tap()['accepted']);self.assertTrue(self.tap(101)['accepted'])
    def test_new_round_cancels_test_without_old_timer_muting_round(self):
        self.start();old=self.c.tap_timer
        ev=AudioEvent('carry-1','round-1','operator','carry_start','unknown',self.now,'host_monotonic_s','operator','Explicit fake carry action','operator_action',ball_id='ball-pink')
        self.assertTrue(self.c.audio_event(ev)['accepted']);old.callback()
        self.assertFalse(self.c.tap_test['active']);self.assertTrue(self.c.enabled);self.assertTrue(self.c.transport)
        self.assertEqual(self.plays()[-1]['trial_id'],'round-1')
        with self.assertRaisesRegex(ValueError,'Wait for active'):self.start()
    def test_active_release_window_blocks_start(self):
        self.c.enable(True)
        ev=AudioEvent('r','round','operator','release','unknown',self.now,'host_monotonic_s','operator','Explicit release','operator_action',ball_id='ball-pink')
        self.c.audio_event(ev)
        with self.assertRaisesRegex(ValueError,'Wait for active'):self.start()
    def test_unavailable_and_invalid_duration_fail_without_enabling(self):
        for value in (0,121,1.5,True,float('nan')):
            with self.assertRaises(ValueError):self.start(duration_s=value)
        self.c.ready=False
        with self.assertRaises(ValueError):self.start()
        self.assertFalse(self.c.enabled);self.assertFalse(self.plays())
    def test_changed_bindings_and_stale_epoch_rejected(self):
        cmd=self.command();self.c.disable()
        with self.assertRaisesRegex(ValueError,'stale'):self.c.control_tap_test(cmd)
        cmd=self.command(show_config_digest='0'*64)
        with self.assertRaisesRegex(ValueError,'mapping changed'):self.c.control_tap_test(cmd)
    def test_dedup_preserves_existing_threshold(self):
        self.c.mapping['exact_duplicate_ms']=20;self.start();self.tap();self.tap()
        self.assertEqual(len(self.plays()),1);self.now+=.03;self.tap();self.assertEqual(len(self.plays()),2)
    def test_stop_playback_error_and_mapping_changes_end_test(self):
        for cancel in (lambda:self.c.stop(),lambda:self.c.native({'type':'error','message':'fake error'}),lambda:self.c.settings('mapping',{'min_velocity':2})):
            self.start();cancel();self.assertFalse(self.c.tap_test['active']);self.assertFalse(self.c.enabled)
    def test_status_timeout_fallback(self):
        self.start();self.now+=16;self.assertFalse(self.c.status()['tap_test']['active']);self.assertFalse(self.c.enabled)

if __name__=='__main__':unittest.main()
