"""Synthetic filesystem/accounting tests; no policy or native execution."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from candidate_continuation_journal import Journal, read_events, write_new_json
from candidate_continuation_ledger import Ledger, Outcome


class JournalTests(unittest.TestCase):
    def test_complete_new_events_are_bound_and_detached(self):
        with tempfile.TemporaryDirectory() as temporary:
            writer = Journal(Path(temporary)/"events")
            payload = {"values":[1]}
            first = writer.append("start", payload)
            payload["values"][0] = 9
            first["payload"]["values"][0] = 8
            writer.append("stop", {})
            events = read_events(writer.directory)
            self.assertEqual(events[0]["payload"], {"values":[1]})
            self.assertEqual(events[1]["previous_sha256"], events[0]["fingerprint"])
            self.assertEqual(events[-1]["fingerprint"], writer.head)
            self.assertEqual(writer.sequence, 2)

    def test_existing_directory_and_files_are_never_replaced(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"receipt.json"
            write_new_json(path, {"original":1})
            original = path.read_bytes()
            with self.assertRaises(FileExistsError): write_new_json(path, {"original":2})
            self.assertEqual(path.read_bytes(), original)
            writer = Journal(Path(temporary)/"events")
            with self.assertRaises(FileExistsError): Journal(writer.directory)

    def test_write_failure_poison_prevents_any_retry(self):
        with tempfile.TemporaryDirectory() as temporary:
            writer = Journal(Path(temporary)/"events")
            with patch("candidate_continuation_journal.os.fsync", side_effect=OSError("synthetic failure")):
                with self.assertRaises(OSError): writer.append("reserve", {})
            self.assertTrue(writer.failed)
            with self.assertRaisesRegex(RuntimeError, "cannot resume"): writer.append("retry", {})
            self.assertEqual(list(writer.directory.iterdir()), [])

    def test_published_reservation_survives_directory_sync_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            writer = Journal(Path(temporary)/"events")
            with patch("candidate_continuation_journal._sync_directory", side_effect=OSError("synthetic directory")):
                with self.assertRaises(OSError): writer.append("reserve", {"charged":35})
            self.assertTrue(writer.failed)
            self.assertEqual(read_events(writer.directory)[0]["payload"]["charged"], 35)
            with self.assertRaises(RuntimeError): writer.append("replacement", {})

    def test_nonfinite_gap_changed_chain_and_pending_files_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            writer = Journal(Path(temporary)/"events")
            writer.append("start", {})
            extra = writer.directory/".pending-crash"
            extra.write_text("preserve")
            with self.assertRaisesRegex(ValueError, "partial"): read_events(writer.directory)
            extra.unlink()  # Test-owned synthetic file only.
            writer.append("next", {})
            second = writer.directory/"event-000001.json"
            value = json.loads(second.read_text()); value["payload"] = {"changed":True}
            second.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "chain"): read_events(writer.directory)
            second.unlink()
            (writer.directory/"event-000002.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "gap"): read_events(writer.directory)
        with tempfile.TemporaryDirectory() as temporary:
            writer = Journal(Path(temporary)/"events")
            with self.assertRaises(ValueError): writer.append("bad", {"x":float("nan")})
            self.assertTrue(writer.failed)

    def test_small_rejection_batches_stop_at_ppo_buffer_boundary(self):
        ledger = Ledger.new(tuple(f"train-{i:03d}" for i in range(35)))
        for index in range(40):
            batch = ledger.next_batch()
            ledger = ledger.reserve_batch(batch)
            ledger = ledger.record_batch(tuple(Outcome(r.request_id, index != 0 or j >= 30,
                "synthetic-decline" if index == 0 and j < 30 else None)
                for j,r in enumerate(batch.requests)))
        self.assertEqual(ledger.summary()["train"]["released"], 1370)
        for expected in (30, 30, 30, 30, 20):
            batch = ledger.next_batch()
            self.assertEqual(len(batch.requests), expected)
            ledger = ledger.reserve_batch(batch).record_batch(tuple(
                Outcome(r.request_id, False, "synthetic-decline") for r in batch.requests))
        self.assertEqual(ledger.summary()["train"]["requested"], 1540)
        self.assertEqual(len(ledger.next_batch().requests), 30)


if __name__ == "__main__":
    unittest.main()
