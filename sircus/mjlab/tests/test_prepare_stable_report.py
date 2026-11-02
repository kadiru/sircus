"""The new report view is optional, bounded and never rewrites source studies."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from sircus.mjlab.prepare_stable_report import prepare, watch, confirmation_summaries


class StableReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.base, self.study, self.output = self.root / "base.json", self.root / "study", self.root / "view.json"
        self.value = dict(status="completed", frozen_evaluation={"captures": 65}, expansion_evaluation={"training": {"status": "completed", "captures": 188}})
        self.base.write_text(json.dumps(self.value))

    def tearDown(self):
        self.temp.cleanup()

    def build(self):
        return prepare(self.base, self.study, self.output)

    def test_missing_study_is_optional_and_old_evidence_is_identical(self):
        result = self.build()
        self.assertIsNone(result["stable_ik_evaluation"])
        self.assertEqual({k: result[k] for k in self.value}, self.value)
        self.assertEqual(json.loads(self.base.read_text()), self.value)
        self.assertFalse(self.study.exists())

    def test_compaction_keeps_actual_counts_and_denominators_without_raw_joints(self):
        self.study.mkdir()
        raw = dict(status="completed", phase="completed", no_training=True,
            native_search_attempts=100, confirmation_attempts=10, final_confirmation_attempts=20,
            solver_audit={"comparison": [{"method": "stable_seeded", "accepted": 6, "queries": 6}], "q10": ["raw"]},
            cells=[dict(id="old-test-08", bucket_xy_m=[1.2,-.15], observed_assisted_entries=50,
                search={"attempts": 100, "captures": 2, "rows": ["raw"]},
                promotion={"source_trial_id": "stable-shot-1", "q14": ["raw"],
                    "confirmation": {"captures": 8,"attempts": 10},
                    "final_confirmation": {"captures": 16,"attempts": 20}})])
        source = self.study / "report.json"
        source.write_text(json.dumps(raw));before = source.read_bytes()
        result = self.build()["stable_ik_evaluation"]
        self.assertEqual(result["cells"][0]["promotion"]["final_confirmation"]["attempts"],20)
        self.assertEqual(result["cells"][0]["observed_assisted_entries"],50)
        self.assertNotIn('"q10"',json.dumps(result));self.assertNotIn('"q14"',json.dumps(result))
        self.assertEqual(result["verification"]["status"], "not_available")
        (self.study / "audit-stable.json").write_text(json.dumps({"passed": True,"checks":[{"passed":True}]}))
        self.assertEqual(self.build()["stable_ik_evaluation"]["verification"],dict(status="passed",checks=1))
        self.assertEqual(source.read_bytes(), before)

    def test_protected_output_paths_rejected(self):
        for output in (self.base,self.study / "report.json",self.study / "contract.json",self.study / "nested" / "view.json"):
            with self.assertRaises(ValueError):prepare(self.base,self.study,output)

    def test_failed_final_is_visible_without_promotion_and_keeps_its_own_screen(self):
        self.study.mkdir()
        report = dict(status="completed", phase="completed", no_training=True,
            cells=[dict(id="old-test-21",bucket_xy_m=[.16,-.63],promotion=None,
                search=dict(attempts=1024,captures=20),observed_assisted_entries=100)])
        (self.study/"report.json").write_text(json.dumps(report))
        records=[]
        def group(phase,source,n,captures,invalid=0):
            for i in range(n):
                records.append(dict(id=f"shot-{len(records)}",phase=phase,target_id="old-test-21",
                    confirmation_source_id=source,route_success=i<captures,
                    invalid="rejected_endpoint" if captures <= i < captures+invalid else None))
        group("confirmation","higher-screen-not-final",10,9)
        group("confirmation","final-tested-source",10,8)
        group("final_confirmation","final-tested-source",20,13,3)
        (self.study/"shots.jsonl").write_text("".join(json.dumps(r)+"\n" for r in records))
        before = {p:p.read_bytes() for p in self.study.iterdir()}
        value=self.build()["stable_ik_evaluation"];cell=value["cells"][0]
        self.assertIsNone(cell["promotion"])
        self.assertEqual(cell["best_screening"]["source_trial_id"],"final-tested-source")
        self.assertEqual((cell["best_screening"]["captures"],cell["best_screening"]["attempts"]),(8,10))
        self.assertEqual((cell["final_confirmation"]["captures"],cell["final_confirmation"]["attempts"]),(13,20))
        self.assertEqual(cell["final_confirmation"]["invalid"],3)
        self.assertEqual(cell["final_confirmation"]["status"],"failed")
        self.assertEqual(cell["best_screening"]["status"],"passed")
        self.assertEqual(value["confirmation_records_status"],"available")
        for path,raw in before.items():self.assertEqual(path.read_bytes(),raw)

    def test_best_screening_without_final_and_incomplete_final_stays_pending(self):
        def group(phase,source,n,captures):
            return [dict(phase=phase,target_id="t",confirmation_source_id=source,
                route_success=i<captures,invalid=None) for i in range(n)]
        rows=group("confirmation","low",10,7)+group("confirmation","best",10,9)
        result=confirmation_summaries(rows)["t"]
        self.assertEqual(result["best_screening"]["source_trial_id"],"best")
        self.assertIsNone(result["final_confirmation"])
        rows+=group("final_confirmation","low",12,10)
        result=confirmation_summaries(rows)["t"]
        self.assertEqual(result["best_screening"]["source_trial_id"],"low")
        self.assertEqual(result["final_confirmation"]["attempts"],12)
        self.assertEqual(result["final_confirmation"]["status"],"pending")

    def test_incomplete_jsonl_never_silently_shrinks_confirmation_denominator(self):
        self.study.mkdir();(self.study/"report.json").write_text(json.dumps(dict(status="completed",cells=[dict(id="t")])))
        (self.study/"shots.jsonl").write_text('{"phase":"confirmation"}\n{"phase":')
        value=self.build()["stable_ik_evaluation"]
        self.assertEqual(value["confirmation_records_status"],"not_available")
        self.assertNotIn("best_screening",value["cells"][0])

    def test_watcher_bounds_and_stops_only_after_completed_audited_study(self):
        waiting={"stable_ik_evaluation":{"status":"completed","verification":{"status":"not_available"}}}
        done={"stable_ik_evaluation":{"status":"completed","verification":{"status":"passed"}}}
        with mock.patch('sircus.mjlab.prepare_stable_report.prepare',side_effect=[waiting,done]) as build, mock.patch('sircus.mjlab.prepare_stable_report.time.monotonic',side_effect=[0,0]), mock.patch('sircus.mjlab.prepare_stable_report.time.sleep') as sleep:
            self.assertEqual(watch(self.base,self.study,self.output,10),done)
            self.assertEqual(build.call_count,2);sleep.assert_called_once_with(5)
        for seconds in (-1,1801,float('nan')):
            with self.assertRaises(ValueError):watch(self.base,self.study,self.output,seconds)


if __name__ == "__main__":unittest.main()
