"""Mixed native/rejected request provenance and failure accounting without a GPU."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from sircus.mjlab.calibrated_adapter import CalibratedAdapter
from sircus.mjlab.calibrated_admission import GeometryResolver
from sircus.mjlab.tests.test_calibrated_admission import FakePlanner, FakeSolver, home, source
from sircus.mjlab.tests.test_calibrated_score import config


class BoundSolver(FakeSolver):
    def solve(self, bucket, values, hint):
        if values[0] > 1:
            return dict(passed=False, reason="action_bounds_rejected", candidates=[])
        return super().solve(bucket, values, hint)


class AdapterTests(unittest.TestCase):
    def build(self, path, *, blocked=False, corrupt=False, invalid=False):
        scene = json.loads((Path(__file__).resolve().parents[3]/"sircus/isaac/scene.json").read_text())
        h, sources = home(scene), [source("a")]
        solver, planner = BoundSolver(), FakePlanner(h, blocked)
        resolver = GeometryResolver({"scene": scene}, path, Path(path)/"native", sources,
                                    home=h, solver=solver, planner=planner)
        calls = []
        def native(requests):
            calls.append(copy.deepcopy(requests))
            rows = [dict(r, native_release=True, invalid_physics=invalid,
                         route_success=True, success=True, complete_success=True, route_quality=2)
                    for r in requests]
            if corrupt:
                rows[0]["values"] = [.7, -.2, .3, 30., .2]
            arrays = dict(position_m=np.zeros((3, len(rows), 3)), time_s=np.arange(3)*.001)
            return rows, arrays, {"native_rollout_s": .01}
        adapter = CalibratedAdapter({"scene": scene}, path, Path(path)/"native", prior_records=sources,
                                    calibration=config(scene), home=h, resolver=resolver, native_executor=native)
        return adapter, calls

    def request(self, adapter, ident, values):
        bucket = [.5, -.4]
        record, _ = adapter.resolve_context({"bucket_xy_m": bucket})
        return dict(id=ident, bucket_xy_m=bucket, values=values,
                    source_teacher_id=record["source_hint"]["id"],
                    context_resolution_file=record["file"], context_resolution_sha256=record["sha256"],
                    context_home_sha256=record["home_sha256"], context_layout_eligible=record["layout_eligible"],
                    context_anchor_admitted=record["anchor_admitted"], context_hard_reject=record["hard_reject"],
                    source_hint=record["source_hint"], effective_source_hint=record["effective_source_hint"])

    def test_mixed_rejects_preserve_ids_order_and_raw_column_mapping(self):
        with tempfile.TemporaryDirectory() as path:
            adapter, calls = self.build(path)
            requests = [self.request(adapter, str(i), [x, -.2, .3, 30., .2]) for i,x in enumerate([.61, 10., .62])]
            before = copy.deepcopy(requests)
            rows, timing = adapter.evaluate(requests, "train")
            self.assertEqual(requests, before)
            self.assertEqual([r["id"] for r in rows], ["0", "1", "2"])
            self.assertEqual([r["native_release"] for r in rows], [True, False, True])
            self.assertEqual(timing["requested"], 3); self.assertEqual(timing["native_releases"], 2)
            self.assertEqual([r["id"] for r in calls[0]], ["0", "2"])
            self.assertEqual([rows[i]["native_cache_row"] for i in (0,2)], [0,1])
            with np.load(adapter.directory/rows[0]["native_cache_file"]) as raw:
                self.assertEqual(raw["record_ids"].tolist(), ["0", "2"])
            saved = [json.loads(line) for line in (adapter.directory/"shots.jsonl").read_text().splitlines()]
            self.assertEqual([r["id"] for r in saved], ["0", "1", "2"])
            self.assertTrue(all(r["action_substituted"] is False for r in saved))

    def test_all_rejected_requests_do_not_initialize_native(self):
        with tempfile.TemporaryDirectory() as path:
            adapter, calls = self.build(path, blocked=True)
            rows, timing = adapter.evaluate([self.request(adapter,"blocked",[.61,-.2,.3,30.,.2])], "test")
            self.assertFalse(rows[0]["native_release"]); self.assertEqual(calls, [])
            self.assertEqual(timing["requested"], 1); self.assertEqual(timing["native_releases"], 0)

    def test_invalid_physics_is_a_charged_native_failure(self):
        with tempfile.TemporaryDirectory() as path:
            adapter, _ = self.build(path, invalid=True)
            rows, timing = adapter.evaluate([self.request(adapter,"bad-physics",[.61,-.2,.3,30.,.2])], "test")
            self.assertEqual(timing["native_releases"], 1)
            self.assertTrue(rows[0]["native_release"]); self.assertFalse(rows[0]["route_success"])
            self.assertFalse(rows[0]["success"]); self.assertEqual(rows[0]["route_quality"], 0)

    def test_changed_identity_duplicate_id_and_action_replacement_are_rejected(self):
        with tempfile.TemporaryDirectory() as path:
            adapter, _ = self.build(path)
            request = self.request(adapter,"changed",[.61,-.2,.3,30.,.2])
            request["context_hard_reject"] = True
            with self.assertRaisesRegex(ValueError,"status"): adapter.evaluate([request], "test")
        with tempfile.TemporaryDirectory() as path:
            adapter, _ = self.build(path)
            request = self.request(adapter,"duplicate",[.61,-.2,.3,30.,.2])
            with self.assertRaisesRegex(ValueError,"Unique"): adapter.evaluate([request,request], "test")
        with tempfile.TemporaryDirectory() as path:
            adapter, _ = self.build(path, corrupt=True)
            with self.assertRaisesRegex(RuntimeError,"association"):
                adapter.evaluate([self.request(adapter,"substituted",[.61,-.2,.3,30.,.2])], "test")


if __name__ == "__main__":
    unittest.main()
