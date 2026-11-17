"""Reject forged timing, reused failed workers and post-deadline choices."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch


HAS_NUMPY = bool(importlib.util.find_spec("numpy"))
HAS_TORCH = bool(importlib.util.find_spec("torch"))


def layout(index=0):
    return dict(id=f"case-{index}", context_id=f"case-{index}", split="test", stratum="interior",
                bucket_xy_m=[.4 + index * .1, -.3])


def profile(duration=30., interrupted=True):
    stages = ("menu", "neural_order", "arm_setup_checks", "physics_ranking", "evidence_publication", "other")
    totals = dict.fromkeys(stages, 0.)
    totals.update(other=.5, arm_setup_checks=duration - .5)
    events = [dict(stage="other", elapsed_s=0.), dict(stage="arm_setup_checks", elapsed_s=.5)]
    blocks = [
        dict(stage="other", start_s=0., end_s=.5, duration_s=.5, interrupted=False),
        dict(stage="arm_setup_checks", start_s=.5, end_s=duration, duration_s=duration - .5, interrupted=interrupted),
    ]
    return dict(stage_seconds=totals, events=events, blocks=blocks, decision_observed_s=duration,
                interrupted_stage="arm_setup_checks" if interrupted else None,
                interrupted_stage_s=duration - .5 if interrupted else 0.)


@unittest.skipUnless(HAS_NUMPY, "Evidence checks require NumPy")
class FastEvidenceTests(unittest.TestCase):
    def setUp(self):
        from sircus.mjlab import fast_planning_audit as audit
        self.audit = audit

    @staticmethod
    def encoded(value):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()

    def fixture(self, root, snapshot=True):
        native = root / "native-warm_30"
        journal = native / "fast-journals" / "case-0"
        journal.mkdir(parents=True)
        position = layout()
        choice = dict(id="case-0-warm_30", context_id="case-0", method="warm_30",
                      bucket_xy_m=position["bucket_xy_m"], intended_route=None, request=None)
        plan = {name: dict(proposals=[], admitted=[], records=[])
                for name in ("base", "fallback", "neural", "branches")}
        site = dict(layout=position, method="warm_30", budget_s=30., deadline_reached=True,
                    return_reason="deadline", status="deadline", decision_latency_s=30., latency_s=30.,
                    wall_planning_s=30.05, stop_overhead_s=.04, startup_s=1.,
                    receipt_verification_s=.01, request_wall_s=31.1, journal_directory=str(journal),
                    choice=choice, choices=[choice], plan=plan, generated={}, completed_proposal_ids=[],
                    ranked_proposal_ids=[], active_stage="no_complete_snapshot", accepted_receipts=[],
                    accepted_snapshot_file=None, accepted_snapshot_sha256=None, completion_receipt=None,
                    profile=profile(), job_id="lane-0-0")
        manifest = {}
        if snapshot:
            value = {key: deepcopy(site[key]) for key in ("layout", "method", "choice", "plan", "generated",
                                                          "completed_proposal_ids", "ranked_proposal_ids")}
            value.update(active_stage="neural", profile_events_before_publication=deepcopy(site["profile"]["events"]))
            path = journal / "snapshot-00000.json"
            path.write_bytes(self.encoded(value))
            digest = self.audit.sha(path)
            receipt = dict(kind="snapshot", job_id=site["job_id"], sequence=0, file=path.name,
                           sha256=digest, elapsed_s=5., received_elapsed_s=5.01,
                           choice_sha256=hashlib.sha256(self.encoded(choice)).hexdigest(), completed_proposals=0)
            site.update(value, accepted_receipts=[receipt], accepted_snapshot_file=path.name,
                        accepted_snapshot_sha256=digest)
            manifest[path.relative_to(root).as_posix()] = digest
        return site, native, manifest

    def verify(self, fixture):
        return self.audit.verify_deadline(*fixture, self.audit.Checks())

    def test_deadline_counts_interrupted_work_and_received_snapshot(self):
        with TemporaryDirectory() as root:
            fixture = self.fixture(Path(root))
            self.assertEqual(len(self.verify(fixture)), 1)
            self.assertEqual(fixture[0]["profile"]["stage_seconds"]["arm_setup_checks"], 29.5)

    def test_no_snapshot_is_a_no_release_timeout(self):
        with TemporaryDirectory() as root:
            self.assertEqual(self.verify(self.fixture(Path(root), snapshot=False)), [])

    def test_late_snapshot_cannot_release(self):
        with TemporaryDirectory() as root:
            fixture = self.fixture(Path(root))
            fixture[0]["accepted_receipts"][0]["received_elapsed_s"] = 30.001
            with self.assertRaisesRegex(ValueError, "before deadline"):
                self.verify(fixture)

    def test_snapshot_from_previous_worker_job_is_rejected(self):
        with TemporaryDirectory() as root:
            fixture = self.fixture(Path(root))
            fixture[0]["accepted_receipts"][0]["job_id"] = "lane-0-previous"
            with self.assertRaisesRegex(ValueError, "current job"):
                self.verify(fixture)

    def test_changed_publication_rejected(self):
        with TemporaryDirectory() as root:
            fixture = self.fixture(Path(root))
            (Path(fixture[0]["journal_directory"]) / "snapshot-00000.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "published bytes"):
                self.verify(fixture)

    def test_choice_cannot_change_after_publication(self):
        with TemporaryDirectory() as root:
            fixture = self.fixture(Path(root))
            fixture[0]["choice"] = dict(fixture[0]["choice"], request={"id": "late"})
            with self.assertRaisesRegex(ValueError, "snapshot field choice"):
                self.verify(fixture)

    def test_hidden_startup_or_verification_time_is_rejected(self):
        with TemporaryDirectory() as root:
            fixture = self.fixture(Path(root))
            fixture[0]["request_wall_s"] = 30.1
            with self.assertRaisesRegex(ValueError, "startup, planning and verification"):
                self.verify(fixture)

    def test_normal_completion_requires_done_receipt(self):
        with TemporaryDirectory() as root:
            fixture = self.fixture(Path(root))
            fixture[0].update(status="completed", deadline_reached=False, return_reason="completed")
            with self.assertRaisesRegex(ValueError, "completion receipt required"):
                self.verify(fixture)

    def test_overlapping_or_missing_profile_intervals_rejected(self):
        for start in (.4, .6):
            with self.subTest(start=start):
                site = dict(layout=layout(), method="warm_30", decision_latency_s=30.,
                            deadline_reached=True, profile=profile())
                site["profile"]["blocks"][1]["start_s"] = start
                with self.assertRaises(ValueError):
                    self.audit.verify_profile(site, self.audit.Checks())

    def test_unfinished_stage_cannot_disappear_from_profile(self):
        site = dict(layout=layout(), method="warm_30", decision_latency_s=30.,
                    deadline_reached=True, profile=profile())
        site["profile"]["stage_seconds"]["arm_setup_checks"] = 0.
        with self.assertRaisesRegex(ValueError, "exclusive sum"):
            self.audit.verify_profile(site, self.audit.Checks())

    def test_interrupted_time_cannot_be_reported_as_zero(self):
        site = dict(layout=layout(), method="warm_30", decision_latency_s=30.,
                    deadline_reached=True, profile=profile())
        site["profile"]["interrupted_stage_s"] = 0.
        with self.assertRaisesRegex(ValueError, "unfinished work"):
            self.audit.verify_profile(site, self.audit.Checks())

    def test_snapshot_profiling_prefix_cannot_be_rewritten(self):
        with TemporaryDirectory() as root:
            fixture = self.fixture(Path(root))
            fixture[0]["profile"]["events"][1]["stage"] = "physics_ranking"
            with self.assertRaisesRegex(ValueError, "immutable profiling prefix"):
                self.verify(fixture)

    def test_boolean_nonfinite_and_negative_latency_rejected(self):
        for value in (True, float("nan"), float("inf"), -1.):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.audit.latency([value])

    def test_latency_has_explicit_mean_and_count(self):
        result = self.audit.latency([2., 4., 6.])
        self.assertEqual((result["positions"], result["mean_s"], result["aggregate_s"]), (3, 4., 12.))


@unittest.skipUnless(HAS_NUMPY, "Evidence checks require NumPy")
class WorkerLifecycleTests(unittest.TestCase):
    def setUp(self):
        from sircus.mjlab import fast_planning_audit as audit
        self.audit = audit

    def fixture(self):
        layouts = [layout(index) for index in range(3)]
        order = [row["id"] for row in layouts]
        random.Random(3).shuffle(order)
        spec = dict(methods=list(self.audit.METHODS), processing_seed=3, processing_order=order)
        sites = []
        for method_index, method in enumerate(self.audit.METHODS):
            generation, index = 0, 0
            for step, identity in enumerate(order):
                reused = method != "cold_30" and step == 1
                if step:
                    generation, index = (generation, index + 1) if reused else (generation + 1, 0)
                pid = 100 + method_index * 10 + generation
                sites.append(dict(layout=next(row for row in layouts if row["id"] == identity), method=method,
                    worker_pool_id=method, worker_generation=generation, worker_job_index=index,
                    worker_pid=pid, worker_reused=reused,
                    prior_worker_status=None if step == 0 else "completed" if step == 1 else "deadline",
                    status="deadline" if step == 1 else "completed", startup_s=0. if reused else 1.,
                    job_id=f"{method}-{generation}-{index}",
                    readiness_receipt=None if reused else dict(kind="ready", pid=pid)))
        return sites, layouts, spec

    def verify(self, value):
        self.audit.verify_workers(*value, self.audit.Checks())

    def test_reuse_after_completion_and_respawn_after_deadline(self):
        self.verify(self.fixture())

    def test_timed_out_worker_cannot_be_reused(self):
        fixture = self.fixture()
        site = next(row for row in fixture[0] if row["method"] == "warm_30" and row["prior_worker_status"] == "deadline")
        site.update(worker_reused=True, readiness_receipt=None)
        with self.assertRaisesRegex(ValueError, "reuse only after clean"):
            self.verify(fixture)

    def test_new_pid_cannot_claim_initialized_worker_reuse(self):
        fixture = self.fixture()
        site = next(row for row in fixture[0] if row["method"] == "warm_30" and row["worker_reused"])
        site["worker_pid"] += 1000
        with self.assertRaisesRegex(ValueError, "same initialized process"):
            self.verify(fixture)

    def test_methods_cannot_share_worker_pool(self):
        fixture = self.fixture()
        for site in fixture[0]:
            if site["method"] == "warm_30":
                site["worker_pool_id"] = "cold_30"
                site["job_id"] = f"cold_30-{site['worker_generation']}-{site['worker_job_index']}"
        with self.assertRaises(ValueError):
            self.verify(fixture)

    def test_unfrozen_processing_order_is_rejected(self):
        fixture = self.fixture()
        fixture[2]["processing_order"].reverse()
        with self.assertRaisesRegex(ValueError, "seeded processing order"):
            self.verify(fixture)


@unittest.skipUnless(HAS_TORCH, "Existing selection helpers require PyTorch")
class EarlyStopAndAccountingTests(unittest.TestCase):
    def setUp(self):
        from sircus.mjlab import fast_planning_audit as audit
        self.audit = audit

    def early_fixture(self):
        site = dict(layout=layout(), method="warm_early", return_reason="early_good_shot", deadline_reached=False)
        snapshots = [dict(completed_proposal_ids=[f"candidate-{j}" for j in range(i + 1)],
                          choice=dict(request={"id": f"candidate-{i}"}, prediction=dict(score_m=score)))
                     for i, score in enumerate((-.01, .003, .006))]
        return site, snapshots

    def test_first_checked_five_mm_candidate_is_eligible(self):
        self.audit.verify_early_rule(*self.early_fixture(), self.audit.Checks())

    def test_later_better_candidate_cannot_replace_first_qualifying_one(self):
        site, snapshots = self.early_fixture()
        snapshots.append(dict(completed_proposal_ids=[f"candidate-{j}" for j in range(4)],
                              choice=dict(request={"id": "later"}, prediction=dict(score_m=.04))))
        with self.assertRaisesRegex(ValueError, "no later checks"):
            self.audit.verify_early_rule(site, snapshots, self.audit.Checks())

    def test_missing_per_candidate_eligibility_check_rejected(self):
        site, snapshots = self.early_fixture()
        with self.assertRaisesRegex(ValueError, "after each completed candidate"):
            self.audit.verify_early_rule(site, snapshots[1:], self.audit.Checks())

    def test_claimed_early_return_below_frozen_margin_rejected(self):
        site, snapshots = self.early_fixture()
        snapshots[-1]["choice"]["prediction"]["score_m"] = .0049
        with self.assertRaisesRegex(ValueError, "positive margin"):
            self.audit.verify_early_rule(site, snapshots, self.audit.Checks())

    def test_late_done_after_good_snapshot_stays_a_deadline(self):
        site, snapshots = self.early_fixture()
        site.update(deadline_reached=True, return_reason="deadline")
        self.audit.verify_early_rule(site, snapshots, self.audit.Checks())

    def test_claimed_physics_score_is_independently_reconstructed(self):
        from contextlib import ExitStack
        from sircus.mjlab.neural_shot_plan import _choice

        class Ranker:
            def __init__(self, *args):
                pass

            def predict(self, request):
                return dict(score_m=.006)

            def order(self, rows):
                return list(rows)

        position = layout()
        menu = [dict(id=f"candidate-{i}") for i in range(4)]
        admitted = dict(id="candidate-0-b0", proposal_id="candidate-0", intended_route="direct",
                        values=[.4, -.3, .3, 28., .2], bucket_xy_m=position["bucket_xy_m"],
                        board_heading_offset_deg=0., board_twist_deg=0.,
                        selected_q10=[0.] * 6, selected_q14=[.1] * 6)
        sites = []
        for method in self.audit.METHODS:
            count = 1 if method == "warm_early" else 4
            plan = {name: dict(proposals=[], admitted=[], records=[])
                    for name in ("base", "fallback", "neural", "branches")}
            plan["neural"] = dict(proposals=menu[:count], admitted=[admitted], records=[{}] * count)
            choice = _choice(position, [admitted], Ranker(), count, [row["id"] for row in menu])
            identity = position["id"] + "-" + method
            choice.update(id=identity, method=method, predicted_flight_risk=False)
            choice["request"].update(id=identity, method=method)
            sites.append(dict(layout=position, method=method, plan=plan, choice=choice,
                completed_proposal_ids=[row["id"] for row in menu[:count]],
                ranked_proposal_ids=[row["id"] for row in menu], generated=dict(menu=menu, neural=menu),
                active_stage="early_good_shot" if method == "warm_early" else "finished",
                return_reason="early_good_shot" if method == "warm_early" else "completed",
                status="completed", deadline_reached=False))
        sites[0]["choice"]["prediction"]["score_m"] = .9
        with TemporaryDirectory() as root, ExitStack() as patches:
            root = Path(root)
            for name, value in (("plans.json", sites), ("shot-library-rows.json", dict(original=[], additions=[])),
                                ("route-models.json", {}), ("neural-checkpoint.json", {}), ("route-history.json", {}),
                                ("calibration.json", dict(score_config={}))):
                (root / name).write_text(json.dumps(value))
            for site in sites:
                directory = root / "preparation" / site["method"]
                directory.mkdir(parents=True)
                (directory / (position["id"] + ".json")).write_text(json.dumps(site))
            for name, value in (
                ("sircus.mjlab.intergrid_library_plan.libraries", lambda *args: {}),
                ("sircus.mjlab.neural_shot_plan.menu", lambda *args: menu),
                ("sircus.mjlab.neural_shot_model.order", lambda *args: menu),
                ("sircus.mjlab.intergrid_flight_ranker.FlightRanker", Ranker),
                ("sircus.mjlab.intergrid_proposer_audit.audit_preparation", lambda *args: None),
                ("sircus.mjlab.clearance_expansion_audit.audit_branches", lambda *args: None),
                ("sircus.mjlab.fast_planning_audit.verify_deadline", lambda site, *args: [site]),
                ("sircus.mjlab.fast_planning_audit.verify_workers", lambda *args: None),
            ):
                patches.enter_context(patch(name, value))
            with self.assertRaisesRegex(ValueError, "command independently reconstructed"):
                self.audit.verify_plans(root, [position],
                    dict(scene={}, flight_ranking=dict(rule={}), planning_budget_s=30., early_score_m=.005),
                    None, {}, {}, {}, self.audit.Checks())

    def accounting_fixture(self):
        layouts, slots, rows, sites = [], [], {}, []
        for index, labels in enumerate((("direct", "table", "direct"), ("flight_miss", None, "direct"),
                                        ("invalid_physics", "arm_assisted_capture", None), (None, "direct", "flight_miss"))):
            position = layout(index)
            layouts.append(position)
            for method, label in zip(self.audit.METHODS, labels, strict=True):
                identity = position["id"] + "-" + method
                request = None if label is None else dict(id=identity, context_id=position["id"],
                    bucket_xy_m=position["bucket_xy_m"], method=method, intended_route="direct",
                    values=[.4, -.3, .3, 28., .2], board_heading_offset_deg=0., board_twist_deg=0.,
                    selected_q10=[0.] * 6, selected_q14=[.1] * 6)
                slot = dict(id=identity, context_id=position["id"], bucket_xy_m=position["bucket_xy_m"],
                            method=method, intended_route="direct" if request is not None else None, request=request)
                slots.append(slot)
                timeout = label is None
                duration = 30. if timeout else 5.
                sites.append(dict(layout=position, method=method, choice=slot, deadline_reached=timeout,
                    return_reason="deadline" if timeout else "early_good_shot" if method == "warm_early" else "completed",
                    latency_s=duration, startup_s=2., wall_planning_s=duration + .1, stop_overhead_s=.1,
                    receipt_verification_s=.01, request_wall_s=duration + 2.2,
                    worker_reused=False, profile=profile(duration, timeout)))
                if request is not None:
                    rows[identity] = dict(native_release=True, invalid_physics=label == "invalid_physics",
                        route_success=label in {"direct", "table"}, shot_route=label if label in {"direct", "table"} else None,
                        experimental_route=dict(robot_assisted_capture=label == "arm_assisted_capture",
                                                robot_contact_steps=int(label == "arm_assisted_capture")))
        return layouts, slots, rows, sites

    def test_all_three_methods_keep_no_shot_and_invalid_attempts_in_denominator(self):
        result = self.audit.summarize(*self.accounting_fixture())
        self.assertEqual([result["methods"][key]["captures"] for key in self.audit.METHODS], [1, 2, 2])
        self.assertEqual([result["methods"][key]["attempts"] for key in self.audit.METHODS], [4, 4, 4])
        self.assertEqual(result["paired"]["warm_vs_cold"], dict(both=1, gained=1, lost=0, neither=2))
        self.assertEqual(result["paired"]["early_vs_warm"], dict(both=1, gained=1, lost=1, neither=1))
        self.assertEqual(result["methods"]["cold_30"]["early_return"], 0)
        self.assertEqual(result["methods"]["warm_early"]["early_return"], 3)
        self.assertAlmostEqual(result["timing"]["cold_30"]["request"]["mean_s"], 13.45)
        self.assertEqual(result["profile"]["cold_30"]["interrupted_requests"], 1)


if __name__ == "__main__":
    unittest.main()
