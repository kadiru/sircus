"""Silent observer tests. No Node/audio service, sockets, Viam, or hardware."""
import json
import time
import unittest
from unittest.mock import Mock

from sircus.audio.dashboard import AudioObserver


def event(eid='release', trial='trial-1'):
    return dict(event_id=eid, trial_id=trial, source='operator', event_type='release',
                contact_object='unknown', timestamp=10, time_basis='host_monotonic_s',
                clock_id='test-session', provenance='explicit mock operator action',
                evidence='operator_action')


class AudioObserverTests(unittest.TestCase):
    def make(self, **kwargs):
        return AudioObserver(environ={'SIRCUS_AUDIO_URL': 'http://192.168.1.50:8766',
                                     'SIRCUS_AUDIO_TOKEN': 'TEST_ONLY_PRIVATE'},
                             start_worker=False, **kwargs)

    def enable(self, observer):
        self.control(observer, {'action': 'fx', 'enabled': True})
        observer._take()
        observer._accept_status({'revision': observer._revision,
                                 'status': {'enabled': True, 'pending': False,
                                            'connection': 'connected', 'error': None}})

    def control(self, observer, payload):
        return observer.control({**payload, 'client_id': 'mock-session',
                                 'sequence': observer._clients.get('mock-session', (0,))[0]+1})

    def test_default_off_and_no_configuration_no_process(self):
        factory = Mock(side_effect=AssertionError('process must not start'))
        observer = AudioObserver(environ={}, process_factory=factory)
        observer.emit(event()); observer.stop('trial-1')
        self.assertFalse(observer.snapshot()['enabled'])
        self.assertEqual(len(observer._queue), 0)
        factory.assert_not_called()

    def test_configuration_is_private_and_never_public(self):
        observer = self.make()
        self.assertTrue(observer.snapshot()['configured'])
        self.assertNotIn('192.168', json.dumps(observer.snapshot()))
        self.assertNotIn('TEST_ONLY_PRIVATE', json.dumps(observer.snapshot()))
        for url in ('http://0.0.0.0:8766', 'http://8.8.8.8:8766',
                    'http://192.168.1.50:8766/?token=private', 'http://name.invalid:8766'):
            obs = AudioObserver(environ={'SIRCUS_AUDIO_URL': url}, start_worker=False)
            self.assertFalse(obs.snapshot()['configured'])

    def test_emit_is_immediate_snapshot_and_bounded(self):
        observer = self.make(max_queue=2); self.enable(observer)
        source = event()
        before = time.monotonic()
        observer.emit(source); source['event_id'] = 'mutated'
        observer.emit(event('second')); observer.emit(event('third'))
        self.assertLess(time.monotonic()-before, .1)
        self.assertEqual(len(observer._queue), 2)
        row = observer._take()['event']
        self.assertEqual(row['event_id'], 'release')
        self.assertEqual(row['evidence'], 'operator_action')
        self.assertFalse(row['confirmed'])
        self.assertIn('queue full', observer.snapshot()['error'])

    def test_stop_prioritizes_over_full_queue_and_rejects_late_trial(self):
        observer = self.make(max_queue=2); self.enable(observer)
        observer.emit(event('first')); observer.emit(event('second'))
        before = time.monotonic(); observer.stop('trial-1')
        self.assertLess(time.monotonic()-before, .1)
        self.assertEqual(observer._take()['op'], 'stop')
        observer.emit(event('late'))
        self.assertEqual(len(observer._queue), 0)
        observer.emit(event('new', 'trial-2'))
        self.assertEqual(len(observer._queue), 1)

    def test_stale_queue_expires_without_changing_source_timestamp(self):
        now = [100.]
        observer = self.make(clock=lambda: now[0]); self.enable(observer)
        observer.emit(event()); now[0] += 1.1
        self.assertEqual(observer._take(), {})
        observer.emit(event('fresh'))
        self.assertEqual(observer._take()['event']['timestamp'], 10.)

    def test_disable_drops_cues_and_rejects_stale_enable_status(self):
        observer = self.make(); self.enable(observer)
        old_revision = observer._revision
        observer.emit(event())
        self.control(observer, {'action': 'fx', 'enabled': False})
        observer._accept_status({'revision': old_revision, 'status': {'enabled': True}})
        self.assertFalse(observer.snapshot()['enabled'])
        self.assertFalse(observer.snapshot()['mute_confirmed'])
        self.assertEqual(observer._take()['op'], 'fx')
        observer.emit(event('late'))
        self.assertEqual(len(observer._queue), 0)

    def test_unknown_outcomes_silent(self):
        observer = self.make(); self.enable(observer)
        row = event(); row.update(event_type='outcome', evidence='operator_label', outcome='unknown')
        observer.emit(row)
        self.assertEqual(len(observer._queue), 0)

    def test_unavailable_worker_and_raw_errors_are_sanitized(self):
        observer = self.make()
        observer._factory = Mock(side_effect=RuntimeError('TEST_ONLY_PRIVATE http://private-host'))
        before = time.monotonic(); observer.emit(event())
        self.assertLess(time.monotonic()-before, .1)
        observer._run()
        self.assertEqual(observer.snapshot()['connection'], 'error')
        self.assertNotIn('TEST_ONLY_PRIVATE', json.dumps(observer.snapshot()))
        self.assertNotIn('private-host', json.dumps(observer.snapshot()))
        observer._accept_status({'revision': 0, 'status': {'connection': 'error', 'error': 'TEST_ONLY_PRIVATE'}})
        self.assertNotIn('TEST_ONLY_PRIVATE', json.dumps(observer.snapshot()))

    def test_browser_cannot_submit_events_or_credentials(self):
        observer = self.make()
        for payload in ({'action': 'emit', 'event': event()}, {'action': 'fx', 'enabled': True, 'token': 'x'},
                        {'action': 'volume', 'volume': float('nan')}, {'action': 'volume', 'volume': True}):
            with self.assertRaises(ValueError): self.control(observer, payload)

    def test_off_is_preserved_by_stop_and_volume_before_drain(self):
        observer = self.make(); self.enable(observer)
        self.control(observer, {'action': 'fx', 'enabled': False})
        observer.stop('trial-1')
        self.control(observer, {'action': 'volume', 'volume': .2})
        rows = [observer._take() for _ in range(3)]
        self.assertEqual([r['op'] for r in rows], ['stop', 'fx', 'volume'])
        self.assertFalse(rows[1]['enabled'])
        self.assertFalse(observer.snapshot()['requested_enabled'])

    def test_overflow_mutes_instead_of_losing_cancellation(self):
        observer = self.make(max_queue=2); self.enable(observer)
        self.control(observer, {'action': 'fx', 'enabled': False})
        observer.stop('trial-1'); observer.stop('trial-2')
        self.assertEqual(len(observer._queue), 1)
        self.assertEqual(observer._take()['op'], 'fx')
        self.assertFalse(observer.snapshot()['requested_enabled'])

    def test_browser_request_reordering_and_duplicates_do_not_reenable(self):
        observer = self.make()
        off = {'action': 'fx', 'enabled': False, 'client_id': 'client-a', 'sequence': 2}
        observer.control(off)
        observer.control({**off, 'enabled': True, 'sequence': 1})
        observer.control(off)
        self.assertFalse(observer.snapshot()['requested_enabled'])
        self.assertEqual(observer._revision, 1)
        with self.assertRaises(ValueError): observer.control({**off, 'enabled': True})
        with self.assertRaises(ValueError): observer.control({'action': 'fx', 'enabled': True})

    def test_remote_status_reconciles_intent_and_preserves_mute_warning(self):
        observer = self.make(); self.enable(observer)
        revision = observer._revision
        observer._accept_status({'revision': revision, 'status': {
            'enabled': False, 'requested_enabled': False, 'pending': False,
            'mute_confirmed': True, 'connection': 'connected', 'error': 'audio_disabled_remotely'}})
        status = observer.snapshot()
        self.assertFalse(status['enabled']); self.assertFalse(status['requested_enabled'])
        self.assertFalse(status['pending']); self.assertIn('disabled on the audio host', status['error'])
        observer.emit(event('after-remote-off'))
        self.assertEqual(len(observer._queue), 0)
        observer._accept_status({'revision': revision, 'status': {
            'enabled': False, 'requested_enabled': False, 'pending': False,
            'mute_confirmed': False, 'connection': 'connected', 'error': 'remote_mute_unconfirmed'}})
        status = observer.snapshot()
        self.assertFalse(status['requested_enabled']); self.assertFalse(status['mute_confirmed'])
        self.assertIn('unconfirmed', status['error'])
        # A status sampled before a fresh operator enable must not erase that intent.
        self.control(observer, {'action': 'fx', 'enabled': True})
        observer._accept_status({'revision': revision, 'status': {'requested_enabled': False}})
        self.assertTrue(observer.snapshot()['requested_enabled'])

    def test_readiness_is_bounded_and_sanitized(self):
        observer=self.make()
        observer._accept_status({'revision':0,'status':{'engine_ready':True,'mapping_matches':True,
            'ball_inputs':[{'ball_id':'ball-pink','bound':True,'connected':True,'input_id':123,'token':'PRIVATE'},
                           {'ball_id':'ball-purple','bound':True,'connected':False},
                           {'ball_id':'ball-green','bound':False,'connected':True},
                           {'ball_id':'ball-gray','bound':False,'connected':False}],
            'bound_ball_count':999,'connected_ball_count':999,'config_digest':'PRIVATE','endpoint_name':'PRIVATE'}})
        state=observer.snapshot()
        self.assertTrue(state['engine_ready']);self.assertTrue(state['mapping_matches'])
        self.assertEqual(state['bound_ball_count'],2);self.assertEqual(state['connected_ball_count'],1)
        self.assertNotIn('PRIVATE',json.dumps(state));self.assertNotIn('input_id',json.dumps(state))
        state['ball_inputs'][0]['connected']=False
        self.assertTrue(observer.snapshot()['ball_inputs'][0]['connected'])
        observer._accept_status({'revision':0,'status':{'mapping_matches':False,'engine_ready':None,
                                'ball_inputs':[{'ball_id':'ball-pink','bound':True,'connected':True}]}})
        self.assertEqual(observer.snapshot()['connected_ball_count'],0)
        observer._error();state=observer.snapshot()
        self.assertFalse(state['mapping_matches']);self.assertIsNone(state['engine_ready']);self.assertEqual(state['ball_inputs'],[])

    def test_ordering_history_is_bounded_and_never_silently_evicted(self):
        observer = self.make()
        for i in range(128):
            observer.control({'action': 'fx', 'enabled': False, 'client_id': f'client-{i}', 'sequence': 1})
        with self.assertRaises(ValueError):
            observer.control({'action': 'fx', 'enabled': True, 'client_id': 'client-new', 'sequence': 1})
        self.assertEqual(len(observer._clients), 128)
        self.assertFalse(observer.snapshot()['requested_enabled'])


if __name__ == '__main__':
    unittest.main()
