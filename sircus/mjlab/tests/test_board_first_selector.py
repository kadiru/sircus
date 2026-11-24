"""Opt-in relabeling preserves old evidence and the common ranking protocol."""
from copy import deepcopy
import importlib.util
import unittest
import tempfile
import json
from pathlib import Path

from sircus.mjlab import neural_shot_model as historical
from sircus.mjlab import board_first_selector_model as model
from sircus.mjlab.board_first_score import VERSION
from sircus.mjlab.board_first_selector_plan import permitted_contact_score
from sircus.mjlab.board_first_selector_study import summarize, GPU_CAP_S
from sircus.mjlab.tests.test_neural_shot_model import record


def revised(row, captured=True, invalid=False):
    result = deepcopy(row)
    result["board_first_score"] = dict(scoring_version=VERSION, route_success=captured, invalid_physics=invalid)
    return result


class BoardFirstSelectorTests(unittest.TestCase):
    def test_empty_resume_never_accepts_partial_native_evidence(self):
        from sircus.mjlab.board_first_empty_resume import require_empty
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path/"report.json").write_text(json.dumps(dict(phase="prepared",controls={},progress=dict(completed_releases=0))))
            require_empty(path)
            (path/"rows.jsonl").write_text("")
            with self.assertRaises(ValueError):require_empty(path)
            (path/"rows.jsonl").unlink()
            (path/"native-0000.npz").write_bytes(b"not-even-a-valid-archive")
            with self.assertRaises(ValueError):require_empty(path)

    def test_empty_resume_source_migration_is_narrow(self):
        from sircus.mjlab.board_first_empty_resume import source_changes
        file = "sircus/mjlab/board_first_selector_study.py"
        original = {file:"old", "sircus/mjlab/board_first_score.py":"unchanged"}
        self.assertEqual(source_changes(original,dict(original,**{file:"new"}))["changed"],[file])
        changed = dict(original,**{file:"new", "sircus/mjlab/board_first_score.py":"different-scoring"})
        with self.assertRaises(ValueError):source_changes(original,changed)

    def test_gpu_deadline_budget_is_an_alarm_compatible_integer(self):
        # signal.alarm rejects floats before the guarded native block begins.
        # Exercise the real deadline context without importing CUDA.
        self.assertIs(type(GPU_CAP_S), int)
        self.assertEqual(GPU_CAP_S, 3600)
        try:
            from sircus.mjlab.intergrid_system_study import gpu_deadline
        except ModuleNotFoundError:
            return  # The strict type assertion still runs on the light Mac env.
        with gpu_deadline(GPU_CAP_S):
            pass

    def test_robot_assisted_label_is_positive_only_for_new_task(self):
        row = record("robot", (.6, -.4), kind="arm_assisted_capture")
        new = revised(row)
        self.assertEqual(historical.outcome(row), "arm_assisted_capture")
        self.assertEqual(model.outcome(new), "unclassified_capture")
        self.assertEqual(historical._encoded([row], "train")[0]["labels"], [1., 0.])
        self.assertEqual(historical._encoded([new], "train", labeler=model.outcome)[0]["labels"], [1., 1.])
        self.assertEqual(new["observation"], row["observation"])

    def test_preboard_contact_can_remove_old_capture_without_changing_original(self):
        row = record("table-before", (.6, -.4), kind="table")
        new = revised(row, False)
        self.assertEqual(historical.outcome(new), "table")
        self.assertEqual(model.outcome(new), "flight_miss")
        self.assertEqual(model.label_changes([new])["counts"], dict(removed=1))

    def test_setup_and_invalid_masks_stay_unchanged(self):
        rows = [record("setup", (.6, -.4), kind="setup_rejected"),
                revised(record("invalid", (.7, -.4), kind="invalid_physics"), False, True)]
        encoded = historical._encoded(rows, "train", labeler=model.outcome)
        self.assertEqual([r["masks"] for r in encoded], [[1., 0.], [1., 0.]])
        rows[0]["board_first_score"] = dict(scoring_version=VERSION)
        with self.assertRaises(ValueError):
            model.outcome(rows[0])

    def test_revised_labels_require_matching_native_command_and_forbid_test(self):
        source = record("shot", (.6, -.4))
        source["observation"].update(bucket_xy_m=[.6, -.4], values=source["request"]["values"],
                                    native_cache_sha256="abc", native_cache_row=1)
        original = deepcopy(source)
        item = dict(original=source["observation"], status="scored", new_score=dict(scoring_version=VERSION,
                    route_success=True, invalid_physics=False), native_sha256="abc", native_row=1)
        result = model.attach_scores([source], {"shot": item})
        self.assertEqual(source, original)
        self.assertEqual(result[0]["observation"], original["observation"])
        changed = deepcopy(item)
        changed["original"]["bucket_xy_m"] = [.61, -.4]
        with self.assertRaises(ValueError):
            model.attach_scores([source], {"shot": changed})
        source["split"] = "test"
        with self.assertRaises(ValueError):
            model.attach_scores([source], {"shot": item})

    def test_arm_clearance_is_diagnostic_in_new_ranking(self):
        prior = dict(score_m=-.2, direct=dict(rim_clearance_m=.05, arm_clearance_m=-.2),
                     table=dict(rim_clearance_m=.02, arm_clearance_m=.1), predicted_route="table")
        new = permitted_contact_score(prior)
        self.assertEqual(new["score_m"], .05)
        self.assertEqual(new["predicted_route"], "direct")
        self.assertEqual(new["direct"]["arm_clearance_m"], -.2)
        self.assertEqual(prior["score_m"], -.2)
        self.assertIsNone(permitted_contact_score(dict(score_m=None, direct=None, table=None))["score_m"])
        beyond = dict(score_m=None, direct=dict(rim_clearance_m=.05, arm_clearance_m=None), table=None)
        self.assertIsNone(permitted_contact_score(beyond)["score_m"])
        table = dict(score_m=-.1, direct=None, table=dict(rim_clearance_m=.05, arm_clearance_m=-.1))
        self.assertEqual(permitted_contact_score(table, table_edge_margin_m=.01)["score_m"], .01)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "CPU fit requires torch")
    def test_custom_training_is_frozen_and_versioned_without_changing_architecture(self):
        train = [revised(record("train-good", (.5, -.4), kind="arm_assisted_capture")),
                 revised(record("train-bad", (.5, -.4), kind="flight_miss", tilt=45.), False)]
        val = [revised(record("val-good", (.8, -.4), "validation", "arm_assisted_capture")),
               revised(record("val-bad", (.8, -.4), "validation", "flight_miss", tilt=45.), False)]
        result = model.fit(train, val, epochs=(2,))
        checkpoint = result["checkpoint"]
        self.assertEqual(checkpoint["architecture"], [44, 64, 32, 2])
        self.assertEqual(checkpoint["label_version"], model.VERSION)
        historical.predict(checkpoint, train[0]["request"])
        val[0]["split"] = "test"
        with self.assertRaises(ValueError):
            model.fit(train, val, epochs=(2,))

    def test_comparison_denominator_includes_refused_setup(self):
        layouts = [dict(id="a", bucket_xy_m=[.6, -.4]), dict(id="b", bucket_xy_m=[.8, -.4])]
        slots, rows, plans = [], [], []
        for layout in layouts:
            for method in ("original", "retrained"):
                identity = layout["id"] + method
                released = not (layout["id"] == "b" and method == "original")
                slots.append(dict(id=identity, context_id=layout["id"], method=method, request={} if released else None))
                plans.append(dict(method=method, decision_latency_s=2.))
                if released:
                    rows.append(dict(id=identity, adapter_directory="native-"+method,
                                     board_first_scalar_audit_passed=True,
                                     board_first_score=dict(route_success=True, invalid_physics=False)))
        result = summarize(layouts, slots, rows, plans)
        self.assertEqual(result["methods"]["original"]["capture_rate"], .5)
        self.assertEqual(result["methods"]["retrained"]["capture_rate"], 1.)
        self.assertEqual(result["paired"]["retrained_only"], 1)
        with self.assertRaises(ValueError):
            summarize(layouts, slots, rows[:-1], plans)


if __name__ == "__main__":
    unittest.main()
