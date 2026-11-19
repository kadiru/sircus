"""Synthetic process ancestry only: no process, App, checkpoint or shot starts."""

import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import seed22_continuation as entry


SESSION = "a" * 32


def process(pid, ppid, sid, label, cookie=SESSION):
    return dict(pid=pid, ppid=ppid, sid=sid, start_token="synthetic-boot:" + label,
                cookie=cookie, state="S")


class OwnerAncestryTests(unittest.TestCase):
    def setUp(self):
        self.watchdog = process(500, 20, 500, "watchdog")
        self.anchor = process(600, 500, 600, "wrapper")
        self.child = process(601, 600, 600, "python")
        self.owner = dict(owner=copy.deepcopy(self.watchdog), native_owner=copy.deepcopy(self.anchor))
        self.processes = {item["pid"]: item for item in (self.watchdog, self.anchor, self.child)}

    def validate(self, self_pid=601):
        before = copy.deepcopy((self.owner, self.processes))
        result = entry.validate_owner(self.owner, SESSION,
            identity=lambda pid: copy.deepcopy(self.processes.get(pid)), self_pid=self_pid)
        self.assertEqual((self.owner, self.processes), before)
        return result

    def test_direct_native_child_matches_exact_watchdog_anchor(self):
        self.assertIs(self.validate(self_pid=600), True)

    def test_python_sh_wrapper_child_is_admitted_through_its_pinned_anchor(self):
        self.assertIs(self.validate(), True)

    def test_nested_wrappers_are_allowed_only_with_complete_owned_ancestry(self):
        self.processes[602] = process(602, 601, 600, "nested-python")
        self.assertIs(self.validate(self_pid=602), True)

    def test_foreign_python_session_is_rejected(self):
        self.processes[601]["sid"] = 999
        with self.assertRaisesRegex(ValueError, "ancestry/session"):
            self.validate()

    def test_foreign_cookie_on_python_or_intermediate_wrapper_is_rejected(self):
        for pid in (601, 602):
            with self.subTest(pid=pid):
                self.setUp()
                self.processes[602] = process(602, 601, 600, "nested-python")
                self.processes[pid]["cookie"] = "b" * 32
                with self.assertRaisesRegex(ValueError, "ancestry/session"):
                    self.validate(self_pid=602)

    def test_reused_anchor_pid_cannot_replace_its_pinned_start_token(self):
        self.processes[600]["start_token"] = "synthetic-boot:reused-wrapper"
        with self.assertRaisesRegex(ValueError, "wrapper identity"):
            self.validate()

    def test_anchor_reparenting_cannot_change_its_pinned_watchdog_parent(self):
        self.processes[600]["ppid"] = 999
        with self.assertRaisesRegex(ValueError, "wrapper identity"):
            self.validate()

    def test_missing_or_reused_watchdog_is_rejected_before_lineage_is_trusted(self):
        for replacement in (None, process(500, 20, 500, "replacement-watchdog")):
            with self.subTest(replacement=replacement):
                self.processes[500] = replacement
                with self.assertRaisesRegex(ValueError, "Watchdog identity"):
                    self.validate()

    def test_missing_python_or_missing_wrapper_ancestor_is_rejected(self):
        for pid in (601, 600):
            with self.subTest(pid=pid):
                self.setUp()
                self.processes.pop(pid)
                with self.assertRaisesRegex(ValueError, "ancestry/session"):
                    self.validate()

    def test_matching_cookie_on_an_unrelated_branch_is_insufficient(self):
        self.processes[601]["ppid"] = 602
        self.processes[602] = process(602, 500, 600, "unrelated-owned-child")
        with self.assertRaisesRegex(ValueError, "ancestry/session"):
            self.validate()

    def test_ancestry_loop_fails_without_unbounded_walk(self):
        self.processes[601]["ppid"] = 602
        self.processes[602] = process(602, 601, 600, "loop")
        with self.assertRaisesRegex(ValueError, "ancestry/session"):
            self.validate()

    def test_exact_sixteen_record_depth_is_allowed_but_seventeen_fails(self):
        for children, accepted in ((15, True), (16, False)):
            with self.subTest(children=children):
                self.setUp()
                for offset in range(children):
                    pid = 700 + offset
                    parent = pid + 1 if offset < children - 1 else 600
                    self.processes[pid] = process(pid, parent, 600, "depth-" + str(offset))
                if accepted:
                    self.assertIs(self.validate(self_pid=700), True)
                else:
                    with self.assertRaisesRegex(ValueError, "exceeded fixed depth"):
                        self.validate(self_pid=700)

    def test_watchdog_cookie_must_also_match_the_authorized_session(self):
        self.owner["owner"]["cookie"] = "b" * 32
        self.processes[500]["cookie"] = "b" * 32
        with self.assertRaises(ValueError):
            self.validate()


class NativeBudgetTests(unittest.TestCase):
    def fixture(self):
        spent = 1.512317179
        manifest = dict(files=dict(prior_attempt_exit_v1=dict(
            path="/synthetic/closed-v1/watchdog-exit.json", sha256="b" * 64)),
            native_wall_seconds=28800. - spent)
        receipt = dict(status="stopped_no_retry", lock_release_safe=True,
                       train_requested=0, train_completed=0, wall_s=spent)
        return manifest, receipt

    def budget(self, manifest, receipt):
        with patch.object(entry, "digest", return_value="b" * 64), \
                patch.object(entry, "read", return_value=copy.deepcopy(receipt)):
            return entry.native_budget(manifest)

    def test_default_without_prior_work_keeps_exact_eight_hour_ceiling(self):
        with patch.object(entry, "digest") as digest, patch.object(entry, "read") as read:
            self.assertEqual(entry.native_budget(dict(files={})), 28800.)
            digest.assert_not_called()
            read.assert_not_called()

    def test_bound_closed_zero_request_startup_cost_is_deducted(self):
        manifest, receipt = self.fixture()
        self.assertAlmostEqual(self.budget(manifest, receipt), 28798.487682821, places=9)

    def test_changed_prior_receipt_bytes_are_rejected_before_trusting_fields(self):
        manifest, receipt = self.fixture()
        with patch.object(entry, "digest", return_value="c" * 64), patch.object(entry, "read") as read:
            with self.assertRaisesRegex(ValueError, "Prior startup receipt changed"):
                entry.native_budget(manifest)
            read.assert_not_called()

    def test_only_closed_zero_request_safely_released_startup_receipts_are_allowed(self):
        manifest, receipt = self.fixture()
        for field, replacement in (("status", "native_exit_zero_independent_audit_required"),
                                   ("train_requested", 1), ("train_completed", 1),
                                   ("lock_release_safe", False)):
            with self.subTest(field=field):
                changed = dict(receipt, **{field: replacement})
                with self.assertRaisesRegex(ValueError, "Only closed zero-request"):
                    self.budget(manifest, changed)

    def test_nonfinite_negative_or_boolean_prior_cost_is_rejected(self):
        manifest, receipt = self.fixture()
        for cost in (float("nan"), float("inf"), -1., True, None):
            with self.subTest(cost=cost):
                with self.assertRaisesRegex(ValueError, "Invalid prior startup cost"):
                    self.budget(manifest, dict(receipt, wall_s=cost))

    def test_duration_cannot_spend_the_original_full_budget_again(self):
        manifest, receipt = self.fixture()
        for duration in (28800., manifest["native_wall_seconds"] + .0001, 28801., float("nan")):
            with self.subTest(duration=duration):
                with self.assertRaisesRegex(ValueError, "extend total native budget"):
                    self.budget(dict(manifest, native_wall_seconds=duration), receipt)

    def test_multiple_closed_startup_costs_are_all_charged(self):
        manifest, receipt = self.fixture()
        manifest["files"]["prior_attempt_exit_v2"] = dict(
            path="/synthetic/closed-v2/watchdog-exit.json", sha256="b" * 64)
        manifest["native_wall_seconds"] -= .5
        with patch.object(entry, "digest", return_value="b" * 64), \
                patch.object(entry, "read", side_effect=[receipt, dict(receipt, wall_s=.5)]):
            self.assertAlmostEqual(entry.native_budget(manifest), 28797.987682821, places=9)


if __name__ == "__main__":
    unittest.main()
