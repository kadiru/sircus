import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from sircus.mjlab import consistent_flight_verify as audit


def requests():
    rows = []
    for role in ("positive_control", "negative_control"):
        rows.extend(dict(id=f"pre-{role}-{i}", role=role, phase="controls_pre", method="control") for i in range(3))
    for method in audit.METHODS:
        rows.extend(dict(id=f"{method}-{i}", role="test", phase="test", method=method,
                         context_id=f"context-{i}", bucket_xy_m=[i*.02, -.2]) for i in range(30))
    for role in ("positive_control", "negative_control"):
        rows.extend(dict(id=f"post-{role}-{i}", role=role, phase="controls_post", method="control") for i in range(3))
    return rows


class FrozenProtocolChecks(unittest.TestCase):
    def test_seeded_layouts_exclude_old_test_and_all_ancestors(self):
        old = dict(excluded=[[.1, -.5], [.7, .2]], splits=dict(
            validation=[dict(bucket_xy_m=[.2, .1])], test=[dict(bucket_xy_m=[.8, -.3])]))
        result = audit.expected_splits(old)
        self.assertEqual(result, audit.expected_splits(old))
        self.assertEqual(len(result["splits"]["test"]), 30)
        xy = np.asarray([r["bucket_xy_m"] for r in result["splits"]["test"]])
        excluded = np.asarray(result["excluded"])
        self.assertEqual(len(excluded), 4)
        self.assertGreaterEqual(np.linalg.norm(xy[:, None]-excluded[None], axis=2).min(), .01)
        distance = np.linalg.norm(xy[:, None]-xy[None], axis=2)
        distance += np.eye(30)*100
        self.assertGreaterEqual(distance.min(), .01)
        self.assertTrue(((xy >= [.1, -.65]) & (xy <= [1.24, .25])).all())

    def test_complete_paired_budget_passes(self):
        audit.budget_checks(requests(), audit.AuditChecks())

    def test_same_length_replaced_context_is_rejected(self):
        rows = requests()
        next(r for r in rows if r.get("method") == "consistent")["bucket_xy_m"] = [1.2, .2]
        with self.assertRaises(audit.EvidenceError):
            audit.budget_checks(rows, audit.AuditChecks())

    def test_missing_failed_request_cannot_disappear_from_denominator(self):
        with self.assertRaises(audit.EvidenceError):
            audit.budget_checks(requests()[:-1], audit.AuditChecks())

    def test_repeated_identity_is_rejected(self):
        rows = requests(); rows[-1]["id"] = rows[-2]["id"]
        with self.assertRaises(audit.EvidenceError):
            audit.budget_checks(rows, audit.AuditChecks())

    def test_model_freeze_must_precede_acquisition(self):
        packet = dict(version=audit.STUDY_VERSION, contract_fingerprint="fp", requests=requests(),
                      frozen_before_any_native_outcome=True, native_planning_queries=0, frozen_unix_s=100.)
        audit.freeze_checks(packet, dict(acquired_unix_s=101.), dict(fingerprint="fp"), audit.AuditChecks())
        for bad in (dict(acquired_unix_s=99.), dict(acquired_unix_s=True), dict(acquired_unix_s=float("nan"))):
            with self.assertRaises(audit.EvidenceError):
                audit.freeze_checks(packet, bad, dict(fingerprint="fp"), audit.AuditChecks())

    def test_boolean_zero_is_not_integer_native_query_count(self):
        packet = dict(version=audit.STUDY_VERSION, contract_fingerprint="fp", requests=requests(),
                      frozen_before_any_native_outcome=True, native_planning_queries=False, frozen_unix_s=100.)
        with self.assertRaises(audit.EvidenceError):
            audit.freeze_checks(packet, dict(acquired_unix_s=101.), dict(fingerprint="fp"), audit.AuditChecks())


class SelectionEvidenceChecks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        self.items = [dict(id=f"candidate-{i:02d}", values=[.5, -.2, .3, 30., .2]) for i in range(16)]
        self.order = list(reversed(range(16)))

    def tearDown(self):
        self.tmp.cleanup()

    def selection(self, flags):
        records = []
        for offset, flag in enumerate(flags):
            name = f"proof-{offset}.json"; path = self.root/name
            path.write_text(json.dumps(dict(passed=flag, reason=None if flag else "unresolved")))
            index = self.order[offset]
            records.append(dict(candidate_index=index, candidate_id=self.items[index]["id"],
                                passed=flag, reason=None if flag else "unresolved", connection_file=name,
                                connection_sha256=audit.sha(path)))
        chosen = self.order[next((i for i, f in enumerate(flags) if f), 0)]
        return dict(checks=records, selected_index=chosen)

    def check(self, selection):
        with patch("sircus.mjlab.calibrated_admission.verify_connection", return_value=True):
            return audit.selection_check(selection, self.order, self.items, self.root, {}, {}, audit.AuditChecks(), "test")

    def test_exact_first_admitted_candidate(self):
        result = self.check(self.selection([False, True]))
        self.assertEqual(result, dict(index=14, admitted=True, geometry_checks=2))

    def test_all_four_rejections_preserve_original_top_one(self):
        result = self.check(self.selection([False]*4))
        self.assertEqual(result, dict(index=15, admitted=False, geometry_checks=4))

    def test_rejected_prefix_cannot_stop_early(self):
        with self.assertRaises(audit.EvidenceError):
            self.check(self.selection([False, False]))

    def test_admitted_candidate_cannot_be_skipped(self):
        with self.assertRaises(audit.EvidenceError):
            self.check(self.selection([True, False]))

    def test_geometry_proof_tampering_rejected(self):
        selection = self.selection([True])
        (self.root/selection["checks"][0]["connection_file"]).write_text('{"passed":false}')
        with self.assertRaises(audit.EvidenceError):
            self.check(selection)

    def test_wrong_candidate_identity_rejected(self):
        selection = self.selection([True]); selection["checks"][0]["candidate_index"] = 2
        with self.assertRaises(audit.EvidenceError):
            self.check(selection)

    def test_prediction_order_must_cover_all_sixteen(self):
        selection = self.selection([True]); self.order[-1] = self.order[0]
        with self.assertRaises(audit.EvidenceError):
            self.check(selection)


if __name__ == "__main__":
    unittest.main()
