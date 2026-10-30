"""Audio is an optional observer; every adapter in these tests is in-memory.

Using mode=hardware here tests provenance, not hardware. No Viam import or
network/audio service is used.
"""
import asyncio
import copy
from datetime import datetime
import unittest
from unittest.mock import patch

import controller as module
import test_fixed_cycle as fixtures
from sircus.audio.contract import AudioEvent


class NeverAwait:
    def __await__(self):
        raise AssertionError('Robot lifecycle awaited audio')
        yield


class AudioSpy:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def emit(self, event):
        self.calls.append(('emit', copy.deepcopy(event)))
        if self.fail:
            raise RuntimeError('Unavailable optional audio')
        return NeverAwait()

    def stop(self, trial_id):
        self.calls.append(('stop', trial_id))
        if self.fail:
            raise RuntimeError('Unavailable optional audio')
        return NeverAwait()


class AudioObserverTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = fixtures.FixedCycleTests.asyncSetUp
    asyncTearDown = fixtures.FixedCycleTests.asyncTearDown
    teach = fixtures.FixedCycleTests.teach
    command = fixtures.FixedCycleTests.command
    drain = fixtures.FixedCycleTests.drain
    launch = fixtures.FixedCycleTests.launch
    wait_phase = fixtures.FixedCycleTests.wait_phase

    def attach(self, fail=False):
        audio = AudioSpy(fail)
        self.controller.audio = audio
        self.controller.mode = 'hardware'  # FakeAdapter remains in-memory.
        return audio

    async def cycle(self, practice=False):
        with patch.object(module, 'LOAD_WAIT_SECONDS', .01):
            await self.controller.action({'action': 'load_drop_cycle',
                'fixed_setup_confirmed': True, 'practice_mode': practice})
            await self.drain()

    async def test_complete_cycle_preserves_ids_timestamps_and_action_provenance(self):
        audio = self.attach()
        await self.teach()
        await self.cycle()
        trial = self.controller.trial
        events = [AudioEvent.from_dict(v) for k, v in audio.calls if k == 'emit']
        self.assertEqual([e.event_type for e in events],
                         ['pickup', 'carry_start', 'carry_stop', 'release'])
        audit = {e['event_id']: e for e in trial['events']}
        for event in events:
            self.assertEqual(event.trial_id, trial['id'])
            self.assertEqual(event.timestamp, datetime.fromisoformat(audit[event.event_id]['timestamp']).timestamp())
            self.assertEqual(event.source, 'operator')
            self.assertEqual(event.evidence, 'operator_action')
            self.assertFalse(event.confirmed)
            self.assertEqual(event.outcome, 'unknown')
            self.assertEqual(event.contact_object, 'unknown')
        self.assertEqual(self.controller.audio_observer_errors, 0)
        self.assertEqual(trial['status'], 'completed')

    async def test_dry_run_never_masquerades_as_hardware_audio(self):
        audio = self.attach()
        self.controller.mode = 'dry_run'
        await self.teach()
        await self.cycle()
        await self.command('save_trial', trial_id=self.controller.trial['id'], outcome='captured')
        self.assertFalse(any(k == 'emit' for k, _ in audio.calls))

    async def test_saved_known_labels_only_and_stable_duplicate_identity(self):
        audio = self.attach()
        await self.teach()
        for outcome in ('captured', 'missed', 'unsure'):
            await self.cycle()
            audio.calls.clear()
            await self.command('save_trial', trial_id=self.controller.trial['id'], outcome=outcome)
            emitted = [v for k, v in audio.calls if k == 'emit']
            self.assertEqual(len(emitted), 0 if outcome == 'unsure' else 1)
            if emitted:
                event = AudioEvent.from_dict(emitted[0])
                self.assertEqual(event.evidence, 'operator_label')
                self.assertTrue(event.confirmed)
                self.assertEqual(event.outcome, outcome)
                self.controller._audio_outcome()
                self.assertEqual(emitted[0], audio.calls[-1][1])

    async def test_audio_exception_never_blocks_release_or_robot_stop(self):
        audio = self.attach(fail=True)
        await self.teach()
        await self.cycle(practice=True)
        self.assertEqual(self.controller.trial['status'], 'completed')
        self.assertIsNone(self.controller.error)
        self.assertEqual([c[0] for c in self.adapter.calls].count('open_gripper'), 2)
        await asyncio.wait_for(self.command('stop'), .5)
        self.assertIn(('stop_dropper',), self.adapter.calls)
        self.assertGreater(self.controller.audio_observer_errors, 0)
        self.assertTrue(any(k == 'stop' for k, _ in audio.calls))

    async def test_stop_during_motion_does_not_wait_for_audio(self):
        audio = self.attach()
        await self.teach()
        self.adapter.move_gate = asyncio.Event()
        await self.launch()
        await self.adapter.move_started.wait()
        trial_id = self.controller.trial['id']
        await asyncio.wait_for(self.command('stop'), .5)
        self.assertIn(('stop', trial_id), audio.calls)
        self.assertIn(('stop_dropper',), self.adapter.calls)
        self.assertNotIn('close_gripper', [c[0] for c in self.adapter.calls])
        self.assertEqual(self.controller.trial['status'], 'aborted')

    async def test_replacement_cancels_previous_id_before_new_cues(self):
        audio = self.attach()
        await self.teach()
        await self.cycle(practice=True)
        old_id = self.controller.trial['id']
        audio.calls.clear()
        await self.cycle(practice=True)
        self.assertEqual(audio.calls[0], ('stop', old_id))
        self.assertTrue(all(v['trial_id'] == self.controller.trial['id']
                            for k, v in audio.calls if k == 'emit'))
        self.assertFalse(any(k == 'emit' and v['event_type'] == 'outcome' for k, v in audio.calls))

    async def test_failed_grab_cancels_audio_and_never_implies_outcome(self):
        audio = self.attach()
        await self.teach()
        self.adapter.close_result = False
        await self.cycle()
        self.assertEqual(self.controller.trial['status'], 'aborted')
        self.assertIn(('stop', self.controller.trial['id']), audio.calls)
        kinds = [v['event_type'] for k, v in audio.calls if k == 'emit']
        self.assertEqual(kinds, ['pickup'])  # Explicit close request, never confirmed pickup.

    async def test_lifecycle_duplicate_keeps_original_event_identity(self):
        audio = self.attach()
        await self.teach()
        await self.cycle()
        entry = next(e for e in self.controller.trial['events'] if e['command'] == 'drop' and e['event'] == 'command_sent')
        before = copy.deepcopy(next(v for k, v in audio.calls if k == 'emit' and v['event_id'] == entry['event_id']))
        self.controller._audio_lifecycle(entry)
        self.assertEqual(before, audio.calls[-1][1])


if __name__ == '__main__':
    unittest.main()
