import copy
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from sircus.audio.contract import AudioEvent
from sircus.audio.show import BallRouter, ShowAudio, default_config, validate_config
from sircus.audio.dashboard import AudioObserver
from sircus.audio.sources import versioned_simulation_events, VERSIONED_SCORING, VERSIONED_PHYSICS


def event(kind='release',trial='pink-1',ball='ball-pink',t=1.,contact='unknown',**kw):
    return AudioEvent(kw.pop('event_id',kind+str(t)),trial,'operator',kind,contact,t,'host_monotonic_s','test-clock',
                      'Explicit test operator action, not physical evidence','operator_action',ball_id=ball,**kw)


def result():
    return dict(scoring_version=VERSIONED_SCORING,physics_version=VERSIONED_PHYSICS,episode_kind='complete_shot',
                executed=True,contacts=[{'t':.2,'object':'board'},{'t':.4,'object':'bucket-floor'}],duration_s=.8,
                success=False,outcome='no_entry',truncated=False)


class ShowTests(unittest.TestCase):
    def test_four_defaults_unbound_and_duplicate_mapping_rejected(self):
        cfg=default_config();self.assertEqual([b['label'] for b in cfg['balls']],['Pink','Purple','Green','Gray'])
        self.assertTrue(all(b['midi_binding'] is None for b in cfg['balls']))
        cfg['balls'][0]['midi_binding']=cfg['balls'][1]['midi_binding']={'input_id':12,'channel':1,'note':0}
        with self.assertRaises(ValueError):validate_config(cfg)

    def test_old_ball_identity_retained_while_next_ball_carries(self):
        now=[10.];r=BallRouter(clock=lambda:now[0]);cfg=default_config()
        cfg['balls'][0]['midi_binding']={'input_id':12,'channel':1,'note':0};r.configure(cfg)
        r.route(event());r.route(event('carry_start','purple-1','ball-purple',t=2.))
        self.assertEqual(r.midi_identity(12,1,0),('pink-1','ball-pink'))
        self.assertIsNone(r.midi_identity(13,1,0))
        unknown=AudioEvent('tap','pink-1','sensor','impact','unknown',3.,'host_monotonic_s','midi',
                           'Mapped endpoint; impact object unknown','midi_gesture',ball_id='ball-pink')
        routed=r.route(unknown);self.assertEqual(routed.contact_object,'unknown')
        self.assertEqual(routed.sound_id,cfg['balls'][0]['sounds']['unknown'])
        now[0]=23.;self.assertIsNone(r.midi_identity(12,1,0))

    def test_retired_identity_cannot_be_revived_or_relabelled(self):
        r=BallRouter();r.route(event());r.cancel('pink-1')
        with self.assertRaises(ValueError):r.route(event(t=2.))
        r.route(event(trial='new'))
        with self.assertRaises(ValueError):r.route(event(trial='new',ball='ball-green',t=3.))

    def test_source_timestamps_are_per_clock_and_delayed_events_rejected(self):
        r=BallRouter();r.route(event(t=3.))
        with self.assertRaises(ValueError):r.route(event('impact',t=2.))
        with self.assertRaises(ValueError):AudioEvent('bad','x','sensor','impact','board',1.,'host_monotonic_s','clock','No board inference','midi_gesture',ball_id='ball-pink')

    def test_new_fields_roundtrip_without_changing_legacy_shape(self):
        legacy=event(ball=None).to_dict();self.assertNotIn('ball_id',legacy);self.assertNotIn('source_event_id',legacy)
        e=event(source_trial_id='original',source_event_id='original-event')
        self.assertEqual(AudioEvent.from_dict(e.to_dict()),e)

    def test_versioned_converter_rejects_legacy_and_inconsistent_capture(self):
        events=versioned_simulation_events(result(),'trial')
        self.assertEqual([e.contact_object for e in events[:-1]],['board','bucket-floor'])
        self.assertTrue(all(e.bounce_index is None and e.source=='simulation' for e in events))
        self.assertEqual(events[-1].outcome,'missed')
        invalid=result();invalid.update(success=True,outcome='captured')
        with self.assertRaises(ValueError):versioned_simulation_events(invalid,'trial')
        invalid=result();invalid['scoring_version']='legacy'
        with self.assertRaises(ValueError):versioned_simulation_events(invalid,'trial')
        uncertain=result();uncertain['outcome']='model_limit'
        self.assertEqual(versioned_simulation_events(uncertain,'trial')[-1].outcome,'unknown')

    def test_published_complete_shots_convert_and_invalid_actions_do_not(self):
        report=Path(__file__).resolve().parents[3]/'baselines/front-bucket-20261030/report.json'
        if not report.exists():self.skipTest('Published front study absent')
        converted=0
        for i,row in enumerate(json.loads(report.read_text())['replays']):
            value=row['result']
            if value['executed'] is False:
                with self.assertRaises(ValueError):versioned_simulation_events(value,'source-'+str(i))
                continue
            events=versioned_simulation_events(value,'source-'+str(i));converted+=1
            self.assertEqual(events[-1].outcome,'captured' if value['success'] else 'missed')
            self.assertTrue(all(e.bounce_index is None for e in events))
        self.assertGreater(converted,0)

    def test_repeated_replay_uses_new_playback_id_retains_source_and_mute_cancels(self):
        observer=AudioObserver(environ={'SIRCUS_AUDIO_URL':'http://127.0.0.1:8766'},start_worker=False)
        show=ShowAudio(observer);observer._state.update(enabled=True,requested_enabled=True,show_audio_capable=True)
        first=show.start_replay(result(),'same-source','ball-pink');time.sleep(.03)
        second=show.start_replay(result(),'same-source','ball-pink');time.sleep(.03)
        self.assertNotEqual(first['replay_trial_id'],second['replay_trial_id'])
        self.assertIn(first['replay_trial_id'],observer._retired)
        emitted=[x['event'] for x in observer._queue if x['op']=='emit']
        self.assertTrue(emitted);self.assertTrue(all(x['source_trial_id']=='same-source' for x in emitted))
        observer.control({'action':'fx','enabled':False,'client_id':'test','sequence':1})
        self.assertIsNone(show.status()['replay_trial_id']);self.assertFalse(observer.snapshot()['enabled'])
        show.close();observer.close()

    def test_remote_mute_or_failure_cancels_future_replay_cues(self):
        for failure in ('remote','transport'):
            observer=AudioObserver(environ={'SIRCUS_AUDIO_URL':'http://127.0.0.1:8766'},start_worker=False)
            show=ShowAudio(observer);observer._state.update(enabled=True,requested_enabled=True,show_audio_capable=True)
            show.start_replay(result(),'source','ball-green')
            if failure=='remote':observer._accept_status({'revision':observer._revision,'status':{'enabled':False,'requested_enabled':False}})
            else:observer._error()
            self.assertIsNone(show.status()['replay_trial_id'])
            observer.close()
    def test_robot_observer_hooks_never_wait_for_show_configuration_lock(self):
        observer=AudioObserver(environ={'SIRCUS_AUDIO_URL':'http://127.0.0.1:8766'},start_worker=False)
        show=ShowAudio(observer);observer._state.update(enabled=True,requested_enabled=True,show_audio_capable=True)
        held=threading.Event();release=threading.Event()
        def hold():
            with show.lock:held.set();release.wait(2)
        thread=threading.Thread(target=hold);thread.start();self.assertTrue(held.wait(1))
        try:
            start=time.monotonic();observer.emit(event().to_dict());observer.stop('pink-1')
            self.assertLess(time.monotonic()-start,.1)
            self.assertIn('pink-1',observer._retired)
            self.assertFalse(any(x['op']=='emit' for x in observer._queue))
        finally:release.set();thread.join(1)
        observer.emit(event(t=3.).to_dict())
        self.assertFalse(any(x['op']=='emit' for x in observer._queue))
        observer.close()

    def test_disabled_fx_never_leaves_a_pending_schedule_for_later_enable(self):
        observer=AudioObserver(environ={},start_worker=False);show=ShowAudio(observer)
        status=show.start_replay(result(),'source','ball-gray')
        self.assertFalse(status['replay_running']);self.assertIsNone(status['replay_trial_id'])
        observer.close()

    def test_unavailable_old_service_drops_optional_extended_cue(self):
        observer=AudioObserver(environ={'SIRCUS_AUDIO_URL':'http://127.0.0.1:8766'},start_worker=False)
        show=ShowAudio(observer);observer._state.update(enabled=True,requested_enabled=True)
        observer.emit(event().to_dict())
        self.assertFalse(any(x['op']=='emit' for x in observer._queue))
        self.assertIn('protocol',observer.snapshot()['error'])
        show.close();observer.close()

    def test_configuration_persists_without_credentials_and_mutes(self):
        with tempfile.TemporaryDirectory() as tmp:
            observer=AudioObserver(environ={},start_worker=False);show=ShowAudio(observer,config_path=Path(tmp)/'show.json')
            cfg=default_config();cfg['balls'][0]['label']='My Pink Ball';show.configure(cfg)
            self.assertEqual(show.status()['config']['balls'][0]['label'],'My Pink Ball')
            self.assertNotIn('token',(Path(tmp)/'show.json').read_text());self.assertFalse(observer.snapshot()['enabled'])
            show.close();observer.close()


if __name__=='__main__':unittest.main()
