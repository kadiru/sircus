"""Verify held-out choice and accounting without starting a simulator."""

import importlib.util
import unittest

from sircus.mjlab import intergrid_ranker
from sircus.mjlab.table_assisted_score import ROUTE_VERSION


def recipe(identifier, xy, gap):
    return dict(id=identifier, bucket_xy_m=list(xy), values=[.63, -.10, .38, 28., gap],
                q10=[.1] * 6, q14=[.2] * 6, board_heading_offset_deg=0.,
                board_twist_deg=0.)


def released(route, *, invalid=False):
    return dict(invalid_physics=invalid, experimental_route={"route_class": route})


@unittest.skipUnless(importlib.util.find_spec("torch"), "Study imports require PyTorch")
class IntergridSelectionTests(unittest.TestCase):
    def test_both_methods_share_the_menu_and_nearest_decline_stays_a_failure(self):
        from sircus.mjlab import intergrid_study as study

        layout = dict(id="intergrid-test-00", bucket_xy_m=[.5, -.4], stratum="grid_perimeter")
        xy = [(.52 + index * .04, -.40) for index in range(5)]
        library = {
            position: [recipe(f"recipe-{index}-{shot}", position, .10 + .03 * index + .001 * shot)
                       for shot in range(3)]
            for index, position in reversed(list(enumerate(xy)))
        }
        proposals = study.proposals_at(layout, library)
        self.assertEqual([row["training_recipe_id"] for row in proposals],
                         [f"recipe-{index}-{shot}" for index in range(4) for shot in range(2)])
        self.assertTrue(all(row["bucket_xy_m"] == layout["bucket_xy_m"] and row["role"] == "test"
                            for row in proposals))
        admitted = [dict(row, proposal_id=row["id"]) for row in proposals]
        scores = [.20, .85, .80, .12, .17, .09, .60, .70]
        nearest, learned = study.select(layout, admitted, proposals, scores)
        self.assertEqual(nearest["request"]["proposal_id"], proposals[0]["id"])
        self.assertEqual(learned["request"]["proposal_id"], proposals[1]["id"])
        self.assertEqual(learned["score"], .85)
        self.assertTrue(nearest["first_known_admitted"])

        nearest, learned = study.select(layout, admitted[1:], proposals, scores[1:])
        self.assertIsNone(nearest["request"])
        self.assertFalse(nearest["first_known_admitted"])
        self.assertEqual(learned["request"]["proposal_id"], proposals[1]["id"])
        self.assertEqual([slot["method"] for slot in (nearest, learned)], ["nearest", "learned"])
        self.assertEqual(len({slot["id"] for slot in (nearest, learned)}), 2)
        with self.assertRaisesRegex(ValueError, "scores must match"):
            study.select(layout, admitted, proposals, scores[:-1])

    def test_paired_counts_keep_rejections_invalid_and_arm_assisted_results(self):
        from sircus.mjlab import intergrid_study as study

        slots, observations = [], {}
        cases = (
            ("rectangle_center", released("standard_capture"), released("standard_capture")),
            ("grid_perimeter", None, released("standard_capture")),
            ("grid_perimeter", released("standard_capture"), released("standard_capture", invalid=True)),
            ("near_fixed_base", released("robot_assisted_capture"), released("not_captured")),
        )
        for index, (stratum, nearest, learned) in enumerate(cases):
            layout = dict(id=f"test-{index}", bucket_xy_m=[.53, -.43 + index * .03], stratum=stratum)
            proposals = [dict(id=f"candidate-{offset}") for offset in range(2)]
            admitted = [dict(proposal_id=row["id"]) for row in proposals[int(nearest is None):]]
            pair = study.select(layout, admitted, proposals, [.3, .8][-len(admitted):])
            for slot, observed in zip(pair, (nearest, learned), strict=True):
                if observed is not None:
                    observations[slot["id"]] = observed
            slots.extend(pair)

        result = study.compare(slots, observations)
        self.assertEqual(result["paired"], dict(both=1, nearest_only=1, learned_only=1, neither=1))
        self.assertEqual(result["methods"], dict(
            nearest=dict(standard_capture=2, setup_rejected=1, robot_assisted_capture=1),
            learned=dict(standard_capture=2, invalid_physics=1, not_captured=1),
        ))
        self.assertEqual(result["strata"]["grid_perimeter"], dict(
            nearest=dict(setup_rejected=1, standard_capture=1),
            learned=dict(standard_capture=1, invalid_physics=1),
        ))
        with self.assertRaisesRegex(ValueError, "releases differ"):
            study.compare(slots, dict(observations, unexpected=released("not_captured")))
        with self.assertRaisesRegex(ValueError, "one paired result"):
            study.compare([slots[1], slots[0], *slots[2:]], observations)


class IntergridTrainingWeightTests(unittest.TestCase):
    def test_unequal_example_counts_get_equal_position_weight(self):
        rows = []
        positions = sorted(intergrid_ranker.TRAIN_POSITIONS)
        self.assertEqual(len(positions), 33)
        for index, xy in enumerate(positions):
            for attempt in range(1 + index % 4):
                row = recipe(f"r-{index:02d}-{attempt}", xy, .2 + attempt * .001)
                rows.append(dict(row, native_release=True, invalid_physics=False, invalid=None,
                                 route_success=(index + attempt) % 2 == 0, route_version=ROUTE_VERSION))
        conflict = dict(rows[0], id="duplicate-outcomes", label_conflict=True)
        matrix, labels, weights, groups, _, excluded = intergrid_ranker._training_data([*rows, conflict])
        self.assertEqual((len(matrix), len(labels), excluded), (len(rows), len(rows), 1))
        self.assertAlmostEqual(float(weights.sum()), 1.)
        offset = 0
        for index, group in enumerate(groups):
            count = 1 + index % 4
            self.assertEqual(group["count"], count)
            self.assertAlmostEqual(float(weights[offset:offset + count].sum()), 1 / 33)
            offset += count


if __name__ == "__main__":
    unittest.main()
