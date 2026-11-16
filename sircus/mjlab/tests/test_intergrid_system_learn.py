"""Check that held-out selections and frozen inference use the same pre-release model."""

import json
import unittest

from sircus.mjlab.intergrid_system_learn import FOLDS, fit_ranker, learned_order


def sample():
    cases, predictions, observations = [], [], {}
    for index in range(2 * FOLDS):
        identity = f"site-{index}"
        xy = [.45 + index * .01, -.5]
        requests, scores = [], []
        for route in ("direct", "table"):
            identifier = identity + "-" + route
            requests.append(dict(
                id=identifier, bucket_xy_m=xy, values=[.4, -.3, .35, 28., .32],
                board_heading_offset_deg=0., board_twist_deg=0.,
                selected_q10=[0.] * 6, selected_q14=[0.] * 6, intended_route=route,
            ))
            scores.append(dict(
                id=identifier, score_m=.01 if route == "direct" else .009,
                direct=dict(rim_clearance_m=.02, arm_clearance_m=.03),
                table=None, predicted_route="direct",
            ))
            observations[identifier] = dict(
                id=identifier, native_release=True, invalid_physics=False, route_success=route == "table",
            )
        cases.append(dict(layout=dict(id=identity, bucket_xy_m=xy), requests=requests))
        predictions.append(scores)
    return cases, predictions, observations


class SystemLearningTests(unittest.TestCase):
    def test_promotion_uses_only_held_out_site_choices(self):
        cases, predictions, labels = sample()
        result = fit_ranker(cases, predictions, labels)
        validation = result["validation"]
        self.assertEqual(validation["learned_captures"], len(cases))
        self.assertEqual(validation["analytic_captures"], 0)
        self.assertEqual(len(validation["cases"]), len(cases))
        self.assertEqual({row["training_positions"] for row in validation["folds"]}, {len(cases) - 2})
        frozen = json.loads(json.dumps(result["checkpoint"], allow_nan=False))
        self.assertEqual(frozen, fit_ranker(cases, predictions, labels)["checkpoint"])
        menu = cases[0]["requests"]
        self.assertEqual(learned_order(menu, predictions[0], frozen), list(reversed(menu)))

    def test_empty_menu_stays_in_denominator_and_invalid_physics_provides_no_label(self):
        cases, predictions, labels = sample()
        for row in labels.values():
            row.update(route_success=True, invalid_physics=True)
        cases.append(dict(layout=dict(id="empty", bucket_xy_m=[.6, -.5]), requests=[]))
        predictions.append([])
        result = fit_ranker(cases, predictions, labels)
        validation = result["validation"]
        self.assertIsNone(result["checkpoint"])
        self.assertEqual(validation["positions"], len(cases))
        self.assertEqual(validation["learned_captures"], 0)
        self.assertEqual(validation["invalid_physics"], len(labels))
        self.assertTrue(all(row["training_pairs"] == 0 for row in validation["folds"]))
        empty = next(row for row in validation["cases"] if row["id"] == "empty")
        self.assertIsNone(empty["learned_id"])
        self.assertIsNone(empty["analytic_id"])

    def test_fixed_menu_rejects_unmatched_or_contaminated_training_evidence(self):
        cases, predictions, labels = sample()
        with self.assertRaisesRegex(ValueError, "must match admitted candidate IDs"):
            fit_ranker(cases, [list(reversed(predictions[0])), *predictions[1:]], labels)
        labels["extra-evaluation-row"] = dict(
            id="extra-evaluation-row", native_release=True, invalid_physics=False, route_success=True,
        )
        with self.assertRaisesRegex(ValueError, "cover every admitted menu exactly once"):
            fit_ranker(cases, predictions, labels)
