"""Deadline, checkpoint, accounting and evidence corruption tests without CUDA."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


@unittest.skipUnless(importlib.util.find_spec("numpy"), "Evidence checks require NumPy")
class BudgetEvidenceTests(unittest.TestCase):
    def setUp(self):
        from sircus.mjlab import planning_budget_audit as audit
        self.audit = audit

    @staticmethod
    def encoded(value):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()

    @staticmethod
    def layout(index=0):
        return dict(id=f"case-{index}", context_id=f"case-{index}", split="test", stratum="interior",
                    bucket_xy_m=[.4 + index * .1, -.3])

    def deadline_fixture(self, root, *, snapshot=True):
        native = root / "native-current_planner"
        journal = native / "budget-journals" / "case-0"
        journal.mkdir(parents=True)
        layout = self.layout()
        choice = dict(id="case-0-current_planner", context_id="case-0", method="current_planner",
                      bucket_xy_m=layout["bucket_xy_m"], intended_route=None, request=None)
        plan = {name: dict(proposals=[], admitted=[], records=[])
                for name in ("base", "fallback", "neural", "branches")}
        site = dict(layout=layout, method="current_planner", budget_s=30., deadline_reached=True,
                    status="deadline", decision_latency_s=30., latency_s=30., wall_planning_s=30.05,
                    stop_overhead_s=.04, startup_s=1., journal_directory=str(journal), choice=choice,
                    choices=[choice], plan=plan, generated={}, completed_proposal_ids=[], ranked_proposal_ids=[],
                    active_stage="no_complete_snapshot", accepted_receipts=[],
                    accepted_snapshot_file=None, accepted_snapshot_sha256=None)
        manifest = {}
        if snapshot:
            value = {key: site[key] for key in ("layout", "method", "choice", "plan", "generated",
                                               "completed_proposal_ids", "ranked_proposal_ids")}
            value["active_stage"] = "base"
            path = journal / "snapshot-00000.json"
            path.write_bytes(self.encoded(value))
            sha = self.audit.sha(path)
            receipt = dict(kind="snapshot", sequence=0, file=path.name, sha256=sha, elapsed_s=5.,
                           received_elapsed_s=5.01, choice_sha256=hashlib.sha256(self.encoded(choice)).hexdigest(),
                           completed_proposals=0)
            site.update(value, accepted_receipts=[receipt], accepted_snapshot_file=path.name,
                        accepted_snapshot_sha256=sha)
            manifest[path.relative_to(root).as_posix()] = sha
        return site, native, manifest

    def verify(self, fixture):
        return self.audit.verify_deadline(*fixture, self.audit.Checks())

    def test_deadline_keeps_only_an_immutable_received_snapshot(self):
        with TemporaryDirectory() as root:
            self.assertEqual(len(self.verify(self.deadline_fixture(Path(root)))), 1)

    def test_timeout_before_any_complete_candidate_has_no_release(self):
        with TemporaryDirectory() as root:
            fixture = self.deadline_fixture(Path(root), snapshot=False)
            self.assertEqual(self.verify(fixture), [])

    def test_receipt_after_deadline_is_rejected_even_if_candidate_finished_earlier(self):
        with TemporaryDirectory() as root:
            fixture = self.deadline_fixture(Path(root))
            fixture[0]["accepted_receipts"][0]["received_elapsed_s"] = 30.001
            with self.assertRaisesRegex(ValueError, "received before deadline"):
                self.verify(fixture)

    def test_receipt_cannot_precede_candidate_completion(self):
        with TemporaryDirectory() as root:
            fixture = self.deadline_fixture(Path(root))
            fixture[0]["accepted_receipts"][0]["elapsed_s"] = 5.02
            with self.assertRaisesRegex(ValueError, "received before deadline"):
                self.verify(fixture)

    def test_over_budget_decision_time_is_rejected(self):
        with TemporaryDirectory() as root:
            fixture = self.deadline_fixture(Path(root))
            fixture[0].update(decision_latency_s=30.01, latency_s=30.01)
            with self.assertRaisesRegex(ValueError, "bounded decision time"):
                self.verify(fixture)

    def test_published_snapshot_cannot_change_after_receipt(self):
        with TemporaryDirectory() as root:
            fixture = self.deadline_fixture(Path(root))
            path = Path(fixture[0]["journal_directory"]) / "snapshot-00000.json"
            path.write_text("{}")
            with self.assertRaisesRegex(ValueError, "published bytes unchanged"):
                self.verify(fixture)

    def test_unreceived_late_snapshot_is_not_used(self):
        with TemporaryDirectory() as root:
            fixture = self.deadline_fixture(Path(root))
            late = Path(fixture[0]["journal_directory"]) / "snapshot-00001.json"
            late.write_text('{"not": "accepted"}')
            self.assertEqual(len(self.verify(fixture)), 1)

    def test_changed_final_choice_cannot_be_smuggled_after_deadline(self):
        with TemporaryDirectory() as root:
            fixture = self.deadline_fixture(Path(root))
            fixture[0]["choice"] = dict(fixture[0]["choice"], request={"id": "late"})
            with self.assertRaisesRegex(ValueError, "selected complete snapshot field choice"):
                self.verify(fixture)

    def test_missing_final_snapshot_is_rejected(self):
        with TemporaryDirectory() as root:
            fixture = self.deadline_fixture(Path(root))
            fixture[0]["accepted_snapshot_file"] = "snapshot-00001.json"
            with self.assertRaisesRegex(ValueError, "last complete snapshot selected"):
                self.verify(fixture)

    def test_no_receipt_cannot_authorize_a_release(self):
        with TemporaryDirectory() as root:
            fixture = self.deadline_fixture(Path(root), snapshot=False)
            fixture[0]["choice"]["request"] = {"id": "unreceived"}
            with self.assertRaisesRegex(ValueError, "no unreceived candidate"):
                self.verify(fixture)

    def test_timeout_is_not_normal_worker_failure(self):
        with TemporaryDirectory() as root:
            fixture = self.deadline_fixture(Path(root))
            fixture[0]["status"] = "failed"
            with self.assertRaisesRegex(ValueError, "failures cannot masquerade"):
                self.verify(fixture)

    def test_full_completion_requires_finished_snapshot(self):
        with TemporaryDirectory() as root:
            fixture = self.deadline_fixture(Path(root))
            fixture[0].update(deadline_reached=False, status="completed")
            with self.assertRaisesRegex(ValueError, "exhausted its frozen rule"):
                self.verify(fixture)

    def test_shutdown_overhead_cannot_be_hidden_in_reported_wall_time(self):
        with TemporaryDirectory() as root:
            fixture = self.deadline_fixture(Path(root))
            fixture[0]["wall_planning_s"] = 30.001
            with self.assertRaisesRegex(ValueError, "shutdown cost"):
                self.verify(fixture)

    def test_negative_boolean_or_nonfinite_times_are_rejected(self):
        for bad in (-1., True, float("nan"), float("inf")):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "finite and nonnegative"):
                self.audit.latency([1., bad])

    def test_changed_frozen_checkpoint_is_rejected(self):
        with TemporaryDirectory() as root:
            root = Path(root)
            path = root / "neural-checkpoint.json"
            path.write_text('{"weight":1}')
            bindings = {path.name: self.audit.sha(path)}
            self.audit.verify_bindings(root, bindings, self.audit.Checks(), "frozen")
            path.write_text('{"weight":2}')
            with self.assertRaisesRegex(ValueError, "neural-checkpoint"):
                self.audit.verify_bindings(root, bindings, self.audit.Checks(), "frozen")

    def test_binding_cannot_escape_evidence_root(self):
        with TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError, "escapes"):
                self.audit.verify_bindings(Path(root), {"../checkpoint.json": "unknown"}, self.audit.Checks(), "frozen")


@unittest.skipUnless(importlib.util.find_spec("torch"), "Existing accounting imports require PyTorch")
class BudgetAccountingTests(unittest.TestCase):
    def setUp(self):
        from sircus.mjlab import planning_budget_audit as audit
        self.audit = audit

    @staticmethod
    def observed(label):
        return dict(native_release=True, invalid_physics=label == "invalid_physics",
                    route_success=label in {"direct", "table"}, shot_route=label if label in {"direct", "table"} else None,
                    experimental_route=dict(robot_assisted_capture=label == "arm_assisted_capture",
                                            robot_contact_steps=int(label == "arm_assisted_capture")))

    def fixture(self):
        layouts, slots, rows, sites = [], [], {}, []
        for index, labels in enumerate((("direct", "table"), ("flight_miss", None),
                                        ("invalid_physics", "arm_assisted_capture"), (None, "direct"))):
            layout = BudgetEvidenceTests.layout(index)
            layouts.append(layout)
            for method, label in zip(self.audit.METHODS, labels, strict=True):
                identity = layout["id"] + "-" + method
                request = None if label is None else dict(id=identity, context_id=layout["id"],
                    bucket_xy_m=layout["bucket_xy_m"], method=method, intended_route="direct",
                    values=[.4, -.3, .3, 28., .2], board_heading_offset_deg=0., board_twist_deg=0.,
                    selected_q10=[0.] * 6, selected_q14=[.1] * 6)
                slot = dict(id=identity, context_id=layout["id"], bucket_xy_m=layout["bucket_xy_m"], method=method,
                            intended_route="direct" if request is not None else None, request=request)
                slots.append(slot)
                sites.append(dict(layout=layout, method=method, choice=slot, deadline_reached=label is None,
                                  latency_s=30. if label is None else 5., startup_s=2., wall_planning_s=30.1 if label is None else 5.1,
                                  stop_overhead_s=.1))
                if request is not None:
                    rows[identity] = self.observed(label)
        return layouts, slots, rows, sites

    def test_no_candidate_invalid_and_robot_assisted_stay_in_denominator(self):
        result = self.audit.summarize(*self.fixture())
        current, neural = (result["methods"][method] for method in self.audit.METHODS)
        self.assertEqual((current["attempts"], current["releases"], current["captures"], current["capture_rate"]),
                         (4, 3, 1, .25))
        self.assertEqual((neural["attempts"], neural["captures"], neural["capture_rate"], neural["robot_contacts"]),
                         (4, 2, .5, 1))
        self.assertEqual((neural["deadline_reached"], neural["deadline_no_candidate"], neural["no_candidate"]), (1, 1, 1))
        self.assertEqual(result["paired"], dict(both=1, gained=1, lost=0, neither=2))

    def test_timeout_with_valid_choice_is_not_an_automatic_failure(self):
        fixture = self.fixture()
        fixture[3][0]["deadline_reached"] = True
        result = self.audit.summarize(*fixture)["methods"]["current_planner"]
        self.assertEqual(result["deadline_with_candidate"], 1)
        self.assertEqual(result["captures"], 1)

    def test_missing_or_extra_observation_is_rejected(self):
        for extra in (False, True):
            fixture = self.fixture()
            if extra:
                fixture[2]["unplanned"] = self.observed("direct")
            else:
                fixture[2].pop(next(iter(fixture[2])))
            with self.subTest(extra=extra), self.assertRaisesRegex(ValueError, "exactly one observation"):
                self.audit.summarize(*fixture)

    def test_wrong_bucket_or_request_identity_is_rejected(self):
        fixture = self.fixture()
        fixture[1][0]["request"]["bucket_xy_m"] = [0., 0.]
        with self.assertRaisesRegex(ValueError, "identity and bucket"):
            self.audit.summarize(*fixture)

    def test_missing_or_duplicate_site_is_rejected(self):
        fixture = self.fixture()
        fixture[3][-1] = deepcopy(fixture[3][0])
        with self.assertRaisesRegex(ValueError, "ordered preparation"):
            self.audit.summarize(*fixture)

    def test_candidate_different_from_saved_complete_choice_is_rejected(self):
        fixture = self.fixture()
        fixture[3][0]["choice"] = dict(fixture[3][0]["choice"], request=None)
        with self.assertRaisesRegex(ValueError, "exact completed planner choice"):
            self.audit.summarize(*fixture)

    def test_cold_start_and_shutdown_are_separate_from_decision_latency(self):
        result = self.audit.summarize(*self.fixture())
        self.assertEqual(result["latency"]["current_planner"]["median_s"], 5.)
        self.assertEqual(result["timing"]["current_planner"]["startup"]["median_s"], 2.)
        self.assertAlmostEqual(result["timing"]["current_planner"]["end_to_end"]["median_s"], 7.1)


if __name__ == "__main__":
    unittest.main()
