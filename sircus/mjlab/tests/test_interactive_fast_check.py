"""Acceptance accounting and crash-safe one-drop behavior, without GPU access."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from sircus.mjlab import interactive_fast_check as subject


def frozen():
    return dict(geometry={"scene_sha256": "1" * 64}, system_source={"audit_sha256": "2" * 64},
                fast_source={"fast_study_report_sha256": "3" * 64})


def preview(xy, method):
    provenance = dict(scene_sha256="1" * 64, study_audit_sha256="2" * 64, simulation_only=True,
                      substeps=64, required_retention_s=1., request_sha256="4" * 64,
                      scoring_version="fixture-v3")
    if method == "fast":
        provenance.update(frozen()["fast_source"])
    return dict(bucket_xy_m=xy, status="feasible", planner=method,
                candidate_pool="neural_checked" if method == "fast" else "base", route_intent="direct",
                configuration=dict(board_center_m=[.5, -.2, .3], board_normal=[0., 0., 1.],
                                   board_quaternion_wxyz=[1., 0., 0., 0.], release_center_m=[.5, -.2, .6],
                                   q10=[0.] * 6, q14=[0.] * 6, link_positions10=[[0.] * 3] * 2,
                                   link_positions14=[[0.] * 3] * 2, tilt_deg=0., gap_m=.3),
                timing=dict(planning_wall_s=2., worker_reused=True), provenance=provenance)


class Client:
    def __init__(self, *, fail_drop=False, no_setup=False):
        self.plan_count = self.drop_count = self.rejected_count = 0
        self.job = None
        self.fail_drop = fail_drop
        self.no_setup = no_setup

    def request(self, path="", *, body=None, expected=200):
        if path == "/plan":
            self.plan_count += 1
            xy, method = body["bucket_xy_m"], body["planner"]
            self.job = dict(id=f"job-{self.plan_count}", state="no_setup" if self.no_setup else "ready",
                            bucket_xy_m=xy, planner=method)
            if not self.no_setup:
                self.job["preview"] = preview(xy, method)
            else:
                self.job["preview"] = dict(preview(xy, method), status="unresolved", configuration=None,
                                           candidate_pool=None, route_intent=None)
            return copy.deepcopy(self.job)
        if path.endswith("/drop"):
            if path.split("/")[1] != self.job["id"]:
                assert expected == 404
                self.rejected_count += 1
                return {}
            if self.job["state"] == "completed":
                assert expected == 409
                self.rejected_count += 1
                return None
            assert self.job["state"] == "ready" and expected == 202
            self.drop_count += 1
            p = self.job["preview"]
            result = dict(p, status="completed", outcome="direct", capture=True, retention_s=1.,
                          trajectory=[dict(time=0., ball=p["configuration"]["release_center_m"]),
                                      dict(time=1., ball=[.6, -.2, .1])],
                          scalar_score_audit_passed=True, scoring_version="fixture-v3",
                          provenance=dict(p["provenance"], native_cache_sha256="5" * 64,
                                          native_release_id="native-" + self.job["id"]))
            self.job.update(state="completed", result=result)
            if self.fail_drop:
                self.fail_drop = False
                raise ConnectionError("Response lost after accepted release")
            return copy.deepcopy(self.job)
        assert path == "/" + self.job["id"]
        return copy.deepcopy(self.job)


class AcceptanceTests(unittest.TestCase):
    def row(self, method="fast"):
        return dict(case="01", planner=method, bucket_xy_m=[.7, -.3], region="interior", state="pending")

    def test_realistic_success_verifies_preview_result_and_duplicate_guard(self):
        row, client = self.row(), Client()
        subject.check_row(client, row, frozen(), lambda: None, interval=.001)
        self.assertEqual(row["state"], "completed")
        self.assertTrue(row["duplicate_rejected"])
        self.assertEqual(client.drop_count, 1)
        subject.check_row(client, row, frozen(), lambda: None)
        self.assertEqual(client.drop_count, 1)

    def test_lost_drop_response_resumes_by_polling_without_second_release(self):
        row, client, saved = self.row(), Client(fail_drop=True), []
        persist = lambda: saved.append(copy.deepcopy(row))
        with self.assertRaises(ConnectionError):
            subject.check_row(client, row, frozen(), persist)
        self.assertEqual(saved[-1]["state"], "drop_intent")
        resumed = saved[-1]
        subject.check_row(client, resumed, frozen(), persist)
        self.assertEqual(resumed["state"], "completed")
        self.assertEqual(client.drop_count, 1)
        self.assertEqual(client.plan_count, 1)

    def test_crash_before_drop_post_does_not_resubmit_saved_intent(self):
        row, client = self.row(), Client()
        def persist():
            if row["state"] == "drop_intent":
                raise RuntimeError("Simulated process death after durable intent")
        with self.assertRaises(RuntimeError):
            subject.check_row(client, row, frozen(), persist)
        self.assertEqual(client.drop_count, 0)
        subject.check_row(client, row, frozen(), lambda: None)
        self.assertEqual(row["state"], "unreleased_uncertain")
        self.assertEqual(client.drop_count, 0)

    def test_unknown_plan_post_is_never_automatically_repeated(self):
        row, client = self.row(), Client()
        row["state"] = "plan_intent"
        with self.assertRaisesRegex(RuntimeError, "unknown"):
            subject.check_row(client, row, frozen(), lambda: None)
        self.assertEqual(client.plan_count, 0)

    def test_no_setup_stays_in_denominator_without_drop(self):
        row, client = self.row(), Client(no_setup=True)
        subject.check_row(client, row, frozen(), lambda: None)
        counts, _ = subject.summarize([row])
        self.assertEqual((counts["fast"]["positions"], counts["fast"]["no_setup"], client.drop_count), (10, 1, 0))

    def test_stale_plan_rejected_before_next_release(self):
        row, client = self.row(), Client()
        subject.check_row(client, row, frozen(), lambda: None, stale_id="previous-job")
        self.assertTrue(row["stale_rejected"])
        self.assertEqual(client.rejected_count, 2)

    def test_bad_preview_provenance_blocks_release(self):
        row, client = self.row(), Client()
        bad = frozen()
        bad["geometry"]["scene_sha256"] = "9" * 64
        with self.assertRaisesRegex(ValueError, "frozen audited"):
            subject.check_row(client, row, bad, lambda: None)
        self.assertEqual(client.drop_count, 0)

    def test_gpu_wait_failure_keeps_ready_plan_and_does_not_drop(self):
        row, client = self.row(), Client()
        def unavailable(*_):
            raise TimeoutError("Protected GPU job still running")
        with self.assertRaises(TimeoutError):
            subject.check_row(client, row, frozen(), lambda: None, before_drop=unavailable)
        self.assertEqual(row["state"], "ready")
        subject.check_row(client, row, frozen(), lambda: None)
        self.assertEqual((client.plan_count, client.drop_count), (1, 1))

    def test_resume_archives_old_error_and_releases_saved_plan_without_replanning(self):
        row, client = self.row(), Client()
        row.update(state="ready", plan_id="saved", preview=preview(row["bucket_xy_m"], "fast"))
        client.job = dict(id="saved", state="ready", preview=row["preview"],
                          bucket_xy_m=row["bucket_xy_m"], planner=row["planner"])
        report = dict(status="failed", error="Previous GPU wait expired",
                      updated_at_unix=123., results=[row])
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(base_url="http://localhost", output=Path(directory),
                                   display_report=None, gpu_wait_s=0, timeout_s=1, poll_s=.001)
            with patch.object(subject, "load_frozen", return_value=(frozen(), report)), \
                 patch.object(subject, "Client", return_value=client), \
                 patch.object(subject, "audit", return_value=report):
                subject.run(args)
        self.assertNotIn("error", report)
        self.assertEqual(report["interruptions"], [
            dict(error="Previous GPU wait expired", observed_at_unix=123.)])
        self.assertEqual(row["state"], "completed")
        self.assertEqual((client.plan_count, client.drop_count), (0, 1))

    def test_bucket_coordinates_only_and_deterministic_alternating_order(self):
        data = dict(bucket_xy_m=[.1, .2], q10=[3, 4], child=[dict(bucket_xy_m=[.3, .4])],
                    excluded_coordinates_m=[[.5, .6]])
        self.assertEqual(subject.coordinates(data), {(.1, .2), (.3, .4), (.5, .6)})
        positions = [dict(case=str(i), bucket_xy_m=[i / 10, 0]) for i in range(10)]
        rows = subject.ordered_rows(positions)
        self.assertEqual([r["planner"] for r in rows[:4]], ["fast", "classic", "classic", "fast"])
        self.assertEqual(len({(r["case"], r["planner"]) for r in rows}), 20)

    def test_final_accounting_includes_failures_and_paired_direction(self):
        rows = subject.ordered_rows([dict(case=str(i), bucket_xy_m=[i / 10, 0]) for i in range(10)])
        for row in rows:
            row["state"] = "no_setup"
        rows[0].update(state="completed", outcome="direct", planning_s=2., preview=preview([0, 0], "fast"))
        rows[1].update(state="failed")
        comparison, paired = subject.summarize(rows)
        for method in subject.METHODS:
            own = comparison[method]
            self.assertEqual(own["releases"] + own["no_setup"] + own["failed"], 10)
        self.assertEqual(paired, dict(both=0, gained=1, lost=0, neither=9))
        self.assertEqual(comparison["fast"]["worker_reused"], 1)

    def test_publish_preserves_other_dashboard_results(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            display = path / "display.json"
            display.write_text('{"old_study":{"capture":99}}')
            args = SimpleNamespace(output=path, display_report=display)
            report = dict(status="running", results=[self.row()])
            subject.publish(args, report)
            value = json.loads(display.read_text())
            self.assertEqual(value["old_study"], {"capture": 99})
            self.assertNotIn("results", value["interactive_fast_acceptance"])
            self.assertEqual(value["interactive_fast_acceptance"]["progress"]["planned"], 20)

    def test_matched_plan_result_pose_tampering_is_rejected(self):
        row, client = self.row(), Client()
        subject.check_row(client, row, frozen(), lambda: None)
        row["result"]["configuration"] = dict(row["result"]["configuration"], gap_m=.9)
        with self.assertRaisesRegex(ValueError, "accepted preview"):
            subject.verify_result(row["result"], row["preview"], row["bucket_xy_m"])

    def test_fast_archive_binds_all_fifty_queries_and_historical_exclusions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            values = [dict(bucket_xy_m=[i / 100, .1]) for i in range(50)]
            excluded = [[.9, -.2], [.8, -.4]]
            layouts = dict(layouts=values, provenance=dict(layouts_sha256=subject._digest(values),
                excluded_coordinates_m=excluded, excluded_coordinates_sha256=subject._digest(excluded)))
            contract = dict(version="sircus-fast-planning-study-v1", physics_substeps=64)
            contract["fingerprint"] = subject._digest(contract)
            files = {"contract.json": contract, "layouts.json": layouts, "calibration.json": {},
                     "neural-checkpoint.json": {}, "choices.json": {},
                     "report.json": dict(status="completed", comparison={"captures": 35})}
            for name, value in files.items():
                subject.save(root / name, value)
            subject.save(root / "evidence-manifest.json", {name: subject.checksum(root / name) for name in files})
            receipt = dict(status="passed", version="sircus-fast-planning-independent-audit-v1",
                           report_sha256=subject.checksum(root / "report.json"),
                           evidence_manifest_sha256=subject.checksum(root / "evidence-manifest.json"),
                           comparison=files["report.json"]["comparison"])
            subject.save(root / "audit.json", receipt)
            known, _, _ = subject.fast_inventory(root, root / "audit.json")
            self.assertEqual(len(known), 52)
            self.assertIn((.9, -.2), known)
            layouts["layouts"][0]["bucket_xy_m"] = [.999, .1]
            subject.save(root / "layouts.json", layouts)
            with self.assertRaisesRegex(ValueError, "Audited fast input changed"):
                subject.fast_inventory(root, root / "audit.json")


if __name__ == "__main__":
    unittest.main()
