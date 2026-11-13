"""Check intermediate collection and frozen comparisons without native simulation."""

import importlib.util
import unittest
from unittest.mock import patch

from sircus.mjlab.cell1050_followup_plan import geometry_key
from sircus.mjlab.intergrid_proposer_plan import COLLECTION_LIMIT, collection_proposals


def layout(split, index=0):
    return dict(id=f"{split}-{index}", split=split, stratum="rectangle_center",
                bucket_xy_m=[.49 + .002 * index, -.45])


def library():
    result = {}
    for index in reversed(range(5)):
        xy = (.51 + .04 * index, -.45)
        result[xy] = [
            dict(id=f"source-{index}-{shot}", bucket_xy_m=list(xy),
                 values=[.63, -.10, .38, 28., .15 + .02 * index + .001 * shot],
                 board_heading_offset_deg=0., board_twist_deg=0., q10=[.1] * 6, q14=[.2] * 6)
            for shot in range(3)
        ]
    return result


class IntergridCollectionTests(unittest.TestCase):
    def test_only_train_layouts_get_a_bounded_collection_menu(self):
        known = library()
        train = layout("train")
        proposals = collection_proposals(train, known)
        self.assertEqual(len(proposals), COLLECTION_LIMIT)
        self.assertEqual(len({geometry_key(row) for row in proposals}), len(proposals))
        self.assertEqual(proposals, collection_proposals(train, known))
        self.assertTrue(all(row["bucket_xy_m"] == train["bucket_xy_m"]
                            and row["phase"] == row["role"] == "train" for row in proposals))
        with self.assertRaisesRegex(ValueError, "only allowed at TRAIN"):
            collection_proposals(layout("test"), known)


@unittest.skipUnless(importlib.util.find_spec("torch"), "Study imports require PyTorch")
class IntergridProposerComparisonTests(unittest.TestCase):
    def test_baseline_and_model_share_ordered_sources_and_learned_commands_are_novel(self):
        from sircus.mjlab import intergrid_proposer, intergrid_proposer_study as study

        test = layout("test")
        known = library()
        checkpoint = {"fingerprint": "frozen-fixture"}
        sources = [row for _, group in sorted(known.items())[:4] for row in group[:2]]
        unchanged = dict(sources[0], id="unchanged")
        novel = dict(sources[1], id="new-a", values=[.63, -.10, .381, 28., .20])
        duplicate = dict(novel, id="repeat-a")
        second = dict(sources[2], id="new-b", values=[.63, -.10, .382, 28., .21])
        saved = {geometry_key(row) for group in known.values() for row in group}
        with patch.object(intergrid_proposer, "propose_candidates",
                          return_value=[unchanged, novel, duplicate, second]) as predict:
            menu = study.test_proposals(test, known, checkpoint, saved)
        predict.assert_called_once_with(checkpoint, test["bucket_xy_m"], sources, context_id=test["id"])
        self.assertEqual([row["training_recipe_id"] for row in menu[:8]], [row["id"] for row in sources])
        self.assertTrue(all(row["method"] == "nearest_feasible" for row in menu[:8]))
        self.assertEqual([row["id"] for row in menu[8:]], ["new-a", "new-b"])
        self.assertTrue(all(row["method"] == "learned" and row["role"] == row["phase"] == "test"
                            and geometry_key(row) not in saved for row in menu[8:]))

    def test_first_feasible_uses_proposal_order_and_counts_every_frozen_choice(self):
        from sircus.mjlab import intergrid_proposer_study as study

        proposed = [dict(id=f"{method}-{index}", method=method)
                    for method in study.METHODS for index in range(3)]
        admitted = [
            dict(row, proposal_id=row["id"], route_success=row["id"].endswith("2"))
            for row in reversed(proposed) if not row["id"].endswith("0")
        ]
        both = study.choose(layout("test"), proposed, admitted)
        only_learned = study.choose(layout("test", 1), proposed,
                                    [row for row in admitted if row["method"] == "learned"])
        only_baseline = study.choose(layout("test", 2), proposed,
                                     [row for row in admitted if row["method"] == "nearest_feasible"])
        self.assertEqual([slot["request"]["proposal_id"] for slot in both], ["nearest_feasible-1", "learned-1"])
        self.assertIsNone(only_learned[0]["request"])
        self.assertIsNone(only_baseline[1]["request"])

        def flight(route, *, invalid=False):
            return dict(invalid_physics=invalid, experimental_route={"route_class": route})

        observations = {
            both[0]["id"]: flight("robot_assisted_capture"),
            both[1]["id"]: flight("standard_capture", invalid=True),
            only_learned[1]["id"]: flight("standard_capture"),
            only_baseline[0]["id"]: flight("standard_capture"),
        }
        slots = [*both, *only_learned, *only_baseline]
        result = study.summarize(slots, observations)
        self.assertEqual(result["methods"], dict(
            nearest_feasible=dict(robot_assisted_capture=1, setup_rejected=1, standard_capture=1),
            learned=dict(invalid_physics=1, standard_capture=1, setup_rejected=1),
        ))
        self.assertEqual(result["paired"], dict(neither=1, learned_only=1, nearest_only=1))
        self.assertEqual(result["both_released"], dict(neither=1))
        with self.assertRaisesRegex(ValueError, "one observation"):
            study.summarize([*slots, *both], observations)


if __name__ == "__main__":
    unittest.main()
