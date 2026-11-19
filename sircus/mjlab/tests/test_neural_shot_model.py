"""No outcome leakage, positional weighting, label integrity and portable inference."""
from copy import deepcopy
import importlib.util
import json
import math
import unittest

from sircus.mjlab import neural_shot_model as model


def request(identifier="shot", *, xy=(.6, -.4), route="direct", tilt=28.):
    return dict(id=identifier, bucket_xy_m=list(xy), values=[.35, -.35, .38, tilt, .38],
                intended_route=route, board_heading_offset_deg=0., board_twist_deg=90.,
                seed_q10=[.1] * 6, seed_q14=[-.2] * 6)


def record(identifier, xy, split="train", kind="direct", tilt=28.):
    proposal = request(identifier, xy=xy, route="table" if kind == "table" else "direct", tilt=tilt)
    native = None if kind == "setup_rejected" else dict(
        id=identifier, native_release=True, invalid_physics=kind == "invalid_physics",
        route_success=kind in {"direct", "table"}, shot_route=kind if kind in {"direct", "table"} else None,
        experimental_route=dict(robot_assisted_capture=kind == "arm_assisted_capture"),
    )
    return dict(request=proposal, setup_ok=native is not None, observation=native, split=split)


def checkpoint():
    layers, width = [], len(model.FEATURE_NAMES)
    for size in model.ARCHITECTURE:
        layers.append(dict(weight=[[0.] * width for _ in range(size)], bias=[0.] * size))
        width = size
    return dict(version=model.VERSION, feature_version=model.FEATURE_VERSION,
                feature_names=list(model.FEATURE_NAMES), heads=list(model.HEADS),
                architecture=[len(model.FEATURE_NAMES), *model.ARCHITECTURE], activation="relu", layers=layers)


class NeuralShotModelTests(unittest.TestCase):
    def test_features_ignore_solved_joints_outcome_and_ids(self):
        original = request()
        contaminated = dict(original, selected_q10=[999.] * 6, selected_q14=[999.] * 6,
                            route_success=True, score_m=100., native_release=True,
                            id="different", context_id="hidden-test-case", observation={"capture": True})
        self.assertEqual(model.feature(original), model.feature(contaminated))
        self.assertEqual(len(model.feature(original)), len(model.FEATURE_NAMES))
        self.assertFalse(any("selected" in field for field in model.FEATURE_NAMES))

    def test_missing_seed_is_distinct_and_heading_wraps(self):
        original = request()
        wrapped = dict(original, board_heading_offset_deg=360., board_twist_deg=450.)
        for left, right in zip(model.feature(original), model.feature(wrapped), strict=True):
            self.assertAlmostEqual(left, right)
        missing = dict(original, seed_q10=None, seed_q14=None)
        self.assertNotEqual(model.feature(original), model.feature(missing))
        for arm in (10, 14):
            index = model.FEATURE_NAMES.index(f"source_seed_arm{arm}_missing")
            self.assertEqual(model.feature(missing)[index], 1.)

    def test_rejects_nonfinite_or_wrong_features(self):
        for changed in (dict(values=[.1, .2, .3, float("nan"), .5]), dict(seed_q10=[0.] * 5),
                        dict(bucket_xy_m=[True, .3]), dict(intended_route="robot")):
            with self.assertRaises(ValueError):
                model.feature(dict(request(), **changed))

    def test_invalid_is_not_capture_negative_and_robot_assisted_stays_separate(self):
        kinds = ("direct", "table", "flight_miss", "setup_rejected", "invalid_physics", "arm_assisted_capture")
        rows = [record(str(index), (.5, -.4), kind=kind) for index, kind in enumerate(kinds)]
        encoded = model._encoded(rows, "train")
        self.assertEqual([row["outcome"] for row in encoded], list(kinds))
        self.assertEqual(encoded[3]["masks"], [1., 0.])
        self.assertEqual(encoded[4]["masks"], [1., 0.])
        self.assertEqual(encoded[5]["masks"], [1., 1.])
        self.assertEqual(encoded[5]["labels"], [1., 0.])

    def test_every_eligible_position_gets_equal_loss_weight(self):
        rows = [record("first", (.5, -.4)), record("second", (.6, -.4)),
                record("third", (.6, -.4), kind="flight_miss"),
                record("invalid", (.7, -.4), kind="invalid_physics")]
        data = model._encoded(rows, "train")
        setup = model._weights(data, 0)
        capture = model._weights(data, 1)
        self.assertAlmostEqual(sum(setup), 1.)
        self.assertAlmostEqual(sum(capture), 1.)
        self.assertAlmostEqual(setup[0], setup[1] + setup[2])
        self.assertEqual(capture, [.5, .25, .25, 0.])

    def test_unreleased_admitted_proposal_is_setup_only_not_a_failed_drop(self):
        row = record("prepared", (.5, -.4), "validation")
        row["observation"] = None
        data = model._encoded([row], "validation")
        self.assertEqual(data[0]["outcome"], "setup_only")
        self.assertEqual(data[0]["labels"], [1., 0.])
        self.assertEqual(data[0]["masks"], [1., 0.])
        result = model._evaluate(checkpoint(), data)
        self.assertEqual(result["top_proposal_capture_unknown"], 1)
        self.assertEqual(result["first_feasible_capture_unknown"], 1)
        self.assertIsNone(result["cases"][0]["top_capture"])

    def test_portable_json_inference_and_stable_ties(self):
        frozen = json.loads(json.dumps(checkpoint(), allow_nan=False))
        first, second = request("first"), request("second", xy=(.8, -.5))
        self.assertEqual(model.predict(frozen, first), dict(setup_probability=.5,
                         capture_given_setup_probability=.5, capture_probability=.25))
        self.assertEqual(model.order(frozen, [first, second]), [first, second])
        with self.assertRaises(ValueError):
            model.order(frozen, [first, first])
        frozen["layers"][1]["weight"][0].pop()
        with self.assertRaises(ValueError):
            model.predict(frozen, first)

    def test_schema_and_split_contamination_fail_before_torch_import(self):
        train = [record("train", (.5, -.4))]
        val = [record("val", (.5, -.4), "validation")]
        with self.assertRaisesRegex(ValueError, "disjoint"):
            model.fit(train, val)
        val[0]["request"]["bucket_xy_m"] = [.8, -.5]
        val[0]["split"] = "test"
        with self.assertRaisesRegex(ValueError, "validation"):
            model.fit(train, val)
        contradictory = record("bad", (.5, -.4))
        contradictory["observation"]["experimental_route"]["robot_assisted_capture"] = True
        with self.assertRaisesRegex(ValueError, "conflict"):
            model.outcome(contradictory)

    def test_invalid_physics_counts_as_failure_for_validation_selection(self):
        rows = [record("invalid", (.5, -.4), "validation", "invalid_physics"),
                record("capture", (.5, -.4), "validation", "direct")]
        result = model._evaluate(checkpoint(), model._encoded(rows, "validation"))
        self.assertEqual(result["positions"], 1)
        self.assertEqual(result["top_proposal_captures"], 0)
        self.assertEqual(result["first_feasible_captures"], 0)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "CPU fit requires pinned Ubuntu torch")
    def test_training_learns_nonlinear_geometry_and_roundtrips_without_refitting(self):
        # Captures in two disconnected tilt ranges; a linear tilt classifier
        # cannot separate them from the middle range.
        training, validation = [], []
        for split, data, start in (("train", training, .4), ("validation", validation, .8)):
            for site in range(6):
                for index, tilt in enumerate((10., 20., 30., 40., 50., 60.)):
                    kind = "direct" if tilt < 25 or tilt > 45 else "flight_miss"
                    data.append(record(f"{split}-{site}-{index}", (start + .01 * site, -.4),
                                       split, kind, tilt=tilt))
        result = model.fit(training, validation, seed=17, epochs=model.DEFAULT_EPOCHS)
        frozen = json.loads(json.dumps(result["checkpoint"], allow_nan=False))
        result2 = model.fit(training, validation, seed=17, epochs=model.DEFAULT_EPOCHS)
        self.assertEqual(frozen, result2["checkpoint"])
        scores = [model.predict(frozen, request(tilt=tilt))["capture_probability"] for tilt in (15., 35., 55.)]
        self.assertGreater(min(scores[0], scores[2]), scores[1] + .2)
        self.assertEqual(frozen["training_positions"], 6)
        self.assertEqual(frozen["validation_positions"], 6)
        self.assertEqual(frozen["device"], "cpu")
        self.assertEqual(result["validation"]["selected"]["top_proposal_captures"], 6)
        before = deepcopy(frozen)
        model.predict(frozen, dict(request(), observation={"route_success": False}))
        self.assertEqual(frozen, before)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "CPU fit requires pinned Ubuntu torch")
    def test_validation_labels_select_epochs_but_never_change_training_gradients(self):
        train = [record("train-good", (.5, -.4), tilt=15.),
                 record("train-bad", (.5, -.4), kind="flight_miss", tilt=45.)]
        val = [record("val-good", (.8, -.4), "validation", tilt=15.),
               record("val-bad", (.8, -.4), "validation", kind="flight_miss", tilt=45.)]
        changed = deepcopy(val)
        for row in changed:
            native = row["observation"]
            native["route_success"] = not native["route_success"]
            native["shot_route"] = "direct" if native["route_success"] else None
        first = model.fit(train, val, seed=3, epochs=(20,))
        second = model.fit(train, changed, seed=3, epochs=(20,))
        self.assertEqual(first["checkpoint"]["layers"], second["checkpoint"]["layers"])
        self.assertNotEqual(first["checkpoint"]["validation_sha256"], second["checkpoint"]["validation_sha256"])

    @unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("mujoco"),
                         "Historical outcome contract requires native environment")
    def test_outcome_contract_matches_existing_scoring_summary(self):
        from sircus.mjlab.intergrid_matched_route_study import outcome

        for kind in ("direct", "table", "flight_miss", "setup_rejected", "invalid_physics", "arm_assisted_capture"):
            row = record(kind, (.5, -.4), kind=kind)
            self.assertEqual(model.outcome(row), outcome(row["observation"]))


if __name__ == "__main__":
    unittest.main()
