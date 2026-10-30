import dataclasses
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest

from sircus.audio import AudioAdapter, AudioEvent
from sircus.audio.preview import write_preview
from sircus.audio.replay import demo_events, load_events, plan, schedule
from sircus.audio.score import CUES, choose, config, score
from sircus.audio.sources import odd_tap, operator_outcome, simulation_events


def event(**changes):
    value = dict(event_id="hit1", trial_id="t1", source="simulation", event_type="impact",
                 contact_object="board", timestamp=0, time_basis="simulation_elapsed_s",
                 clock_id="t1", provenance="synthetic unit test", evidence="model_prediction")
    return AudioEvent(**{**value, **changes})


class AudioTests(unittest.TestCase):
    def setUp(self):
        self.now, self.sent = 0, []
        owner = self

        class Sink:
            def send(self, payload):
                owner.sent.append(list(payload))

        self.adapter = AudioAdapter(Sink(), clock=lambda: self.now, threaded=False)
        self.adapter.start_trial("t1", "simulation")
        self.adapter.pump()
        self.sent.clear()

    def tearDown(self):
        self.adapter.close()

    def notes(self):
        return [p for p in self.sent if p[0] & 0xf0 == 0x90 and p[2]]

    def test_duplicate_is_silent_and_new_trial_can_replay(self):
        self.assertTrue(self.adapter.submit(event()))
        self.assertFalse(self.adapter.submit(event()))
        self.adapter.pump()
        self.assertEqual(len(self.notes()), 1)
        self.adapter.start_trial("t1", "simulation")
        self.assertTrue(self.adapter.submit(event()))
        self.adapter.pump()
        self.assertEqual(len(self.notes()), 2)

    def test_repeated_saved_trace_has_identical_plan(self):
        events = demo_events()
        self.assertEqual(plan(events), plan(events + events))
        changed = dataclasses.replace(events[0], timestamp=.4)
        with self.assertRaises(ValueError):
            schedule(events + [changed])

    def test_first_board_then_stable_variation(self):
        mapping = config()
        self.assertEqual(choose(event(), mapping), "boing")
        self.assertIn(choose(event(), mapping, True), mapping["bounces"])
        self.assertEqual(choose(event(), mapping, True), choose(event(), mapping, True))
        self.assertNotEqual(choose(event(contact_object="unknown"), mapping), "boing")

    def test_unknown_is_not_a_failure(self):
        value = event(event_type="outcome", outcome="unknown")
        self.adapter.submit(value)
        self.adapter.pump()
        self.assertEqual(self.notes(), [])
        self.adapter.submit(event(event_id="late"))
        self.adapter.pump()
        self.assertEqual(self.notes(), [])

    def test_outcome_requires_physical_confirmation(self):
        value = odd_tap("t1", "tap", 20, 123, "host-session")
        self.assertEqual(value.contact_object, "unknown")
        self.assertAlmostEqual(value.intensity, 20/127)
        with self.assertRaises(ValueError):
            dataclasses.replace(value, contact_object="board")
        with self.assertRaises(ValueError):
            dataclasses.replace(value, event_type="outcome", outcome="captured", confirmed=True)
        detected = dataclasses.replace(value, evidence="sensor_detection", event_type="outcome", outcome="captured")
        self.assertIsNone(choose(detected, config()))
        self.assertEqual(choose(dataclasses.replace(detected, confirmed=True), config()), "celebration")

    def test_wrong_trial_domain_and_invalid_payload_are_rejected(self):
        self.assertFalse(self.adapter.submit(event(trial_id="other")))
        self.assertFalse(self.adapter.submit(odd_tap("t1", "tap", 20, 123, "host")))
        self.assertFalse(self.adapter.submit({"bogus": True}))
        self.assertFalse(self.adapter.submit(event(), float("nan")))
        self.adapter.start_trial("t1", "physical")
        self.assertFalse(self.adapter.submit(event()))

    def test_stop_cancels_future_notes_and_releases(self):
        self.adapter.submit(event())
        self.adapter.submit(event(event_id="later"), 10)
        self.adapter.pump()
        self.assertEqual(len(self.notes()), 1)
        self.adapter.cancel("t1")
        self.now = 20
        self.adapter.pump()
        self.assertEqual(len(self.notes()), 1)
        self.assertEqual(self.adapter.pending, [])
        self.assertIn([0xbf, 120, 0], self.sent)
        self.assertFalse(self.adapter.submit(event(event_id="after-stop")))

    def test_stop_event_and_stale_stop(self):
        self.assertFalse(self.adapter.cancel("old-trial"))
        self.adapter.submit(event(event_id="stop", event_type="stop"))
        self.adapter.submit(event(event_id="later"), 1)
        self.adapter.pump()
        self.now = 2
        self.adapter.pump()
        self.assertEqual(self.notes(), [])
        self.assertIsNone(self.adapter.trial)

    def test_mute_does_not_accumulate_or_replay_events(self):
        self.adapter.submit(event(), 1)
        self.adapter.set_muted(True)
        self.adapter.submit(event(event_id="muted"), 5)
        self.adapter.set_muted(False)
        self.now = 10
        self.adapter.pump()
        self.assertEqual(self.notes(), [])
        self.assertFalse(self.adapter.submit(event(event_id="muted")))

    def test_gain_zero_is_silent(self):
        self.adapter.set_gain(0)
        self.adapter.submit(event())
        self.adapter.pump()
        self.assertEqual(self.notes(), [])

    def test_muted_stop_still_ends_trial(self):
        self.adapter.set_muted(True)
        self.adapter.submit(event(event_type="stop"))
        self.adapter.set_muted(False)
        self.assertIsNone(self.adapter.trial)
        self.assertFalse(self.adapter.submit(event(event_id="late")))

    def test_new_cue_removes_old_pitch_bends(self):
        self.adapter.submit(event())
        self.adapter.pump()
        self.now = .1
        self.adapter.submit(event(event_id="capture", event_type="outcome", outcome="captured"))
        self.adapter.pump()
        midi = [p[4] for p in self.adapter.pending if p[3] == "midi"]
        self.assertTrue(midi)
        self.assertFalse(any(p[0] & 0xf0 == 0xe0 for p in midi))

    def test_capacity_never_evicts_deduplication(self):
        self.adapter.mapping["max_events"] = 1
        self.assertTrue(self.adapter.submit(event()))
        self.adapter.pump()
        self.assertFalse(self.adapter.submit(event(event_id="overflow")))
        self.assertFalse(self.adapter.submit(event()))
        self.assertEqual(len(self.adapter.seen), 1)

    def test_queue_capacity_fails_closed(self):
        self.adapter.mapping["max_pending_messages"] = 1
        self.adapter.submit(event())
        self.adapter.pump()
        self.assertEqual(self.notes(), [])
        self.assertEqual(self.adapter.dropped, 1)

    def test_late_events_do_not_dump_old_bends(self):
        self.adapter.submit(event())
        self.now = 10
        self.adapter.pump()
        self.assertEqual(len(self.notes()), 1)
        self.assertTrue(all(p[0] >= 10 for p in self.adapter.pending))
        self.assertLess(len(self.sent), 10)

    def test_failing_output_does_not_escape(self):
        class Broken:
            def send(self, payload):
                raise OSError("output unavailable")
        self.adapter.sink = Broken()
        self.assertTrue(self.adapter.submit(event()))
        self.adapter.pump()
        self.assertGreater(self.adapter.errors, 0)
        self.assertEqual(self.adapter.last_error, "OSError")
        self.assertTrue(self.adapter.cancel())

    def test_blocked_output_does_not_hold_control_lock(self):
        entered, release = threading.Event(), threading.Event()

        class Blocked:
            def send(self, payload):
                entered.set()
                release.wait(3)

        adapter = AudioAdapter(Blocked())
        try:
            adapter.start_trial("t1", "simulation")
            self.assertTrue(entered.wait(1))
            start = time.monotonic()
            self.assertTrue(adapter.submit(event()))
            adapter.cancel()
            adapter.close()
            self.assertLess(time.monotonic() - start, .5)
        finally:
            release.set()
            adapter.worker.join(1)

    def test_contract_rejects_nonfinite_and_unknown_fields(self):
        for changes in ({"timestamp": float("nan")}, {"intensity": 1.1}, {"bounce_index": True},
                        {"schema": "future"}, {"confidence": -1}, {"time_basis": "unix_s"},
                        {"confirmed": True}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                event(**changes)
        with self.assertRaises(ValueError):
            AudioEvent.from_dict({**event().to_dict(), "extra": 1})
        self.assertEqual(AudioEvent.from_dict(event().to_dict()), event())

    def test_predictions_preserve_unknown_and_do_not_invent_bounce_indices(self):
        result = {"contacts": [{"t": .2, "object": "board"}], "retained": False,
                  "outcome": "model_limit", "duration_s": 1}
        events = simulation_events(result, "t1")
        self.assertIsNone(events[0].bounce_index)
        self.assertIsNone(events[0].intensity)
        self.assertEqual(events[-1].outcome, "unknown")
        self.assertEqual(events, simulation_events(result, "t1"))
        for raw, retained, expected in (("retained", True, "captured"), ("retained", False, "unknown"),
                                        ("miss", False, "missed"), (None, False, "unknown")):
            result.update(outcome=raw, retained=retained)
            self.assertEqual(simulation_events(result, "t1")[-1].outcome, expected)

    def test_only_saved_operator_labels_confirm_outcome(self):
        trial = dict(schema="sircus-physical-trial-v1", id="t1", mode="hardware",
                     saved=True, annotated_at="2026-10-29T12:00:00Z", outcome="captured")
        self.assertEqual(operator_outcome(trial).outcome, "captured")
        self.assertTrue(operator_outcome(trial).confirmed)
        for changes in ({"outcome": "unsure"}, {"saved": False}, {"label_status": "unlabeled"}):
            value = operator_outcome({**trial, **changes})
            self.assertEqual(value.outcome, "unknown")
            self.assertFalse(value.confirmed)
        with self.assertRaises(ValueError):
            operator_outcome({**trial, "mode": "dry_run"})

    def test_replay_rejects_empty_mixed_clocks_and_cross_domain(self):
        for events in ([], [event(), event(event_id="two", clock_id="another")],
                       [event(), odd_tap("t1", "tap", 20, 1, "t1")]):
            with self.assertRaises(ValueError):
                plan(events)

    def test_scores_have_bounded_midi_and_release_every_note(self):
        for cue in CUES:
            held = set()
            for _, packet in score(cue, intensity=1, gain=1):
                self.assertIn(packet[0], (0x8f, 0x9f, 0xef))
                self.assertTrue(all(0 <= x <= 127 for x in packet[1:]))
                if packet[0] == 0x9f:
                    held.add(packet[1])
                    self.assertLessEqual(packet[2], 100)
                if packet[0] == 0x8f:
                    held.remove(packet[1])
            self.assertEqual(held, set())

    def test_config_and_silent_preview_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"preview.html"
            write_preview(path, config())
            html = path.read_text()
            self.assertIn("Synthetic demo", html)
            self.assertNotIn("__PACKETS__", html)
            trace = Path(tmp)/"demo.jsonl"
            trace.write_text("\n".join(json.dumps(e.to_dict()) for e in demo_events()))
            self.assertEqual(load_events(trace), demo_events())
            bad = Path(tmp)/"mapping.json"
            bad.write_text(json.dumps({**config(), "channel": 17}))
            with self.assertRaises(ValueError):
                config(bad)


if __name__ == "__main__":
    unittest.main()
