"""Exercise the matched route fits without the native simulator."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from sircus.mjlab import intergrid_route_models as models


POSITIONS = ((.38, -.32), (.48, -.31), (.58, -.33), (.68, -.32), (.78, -.31))


def example(route, index, position, slope):
    x, y = position
    return dict(
        id=f"{route}-{index}", bucket_xy_m=[x, y],
        values=[.55 + slope * (x - .5), -.18, .40, 28., .20 - slope * (y + .3)],
        q10=[.1] * 6, q14=[.2] * 6, shot_route=route,
        route_version=models.ROUTE_VERSION, route_success=True, native_release=True,
        invalid=None, invalid_physics=False, phase="train",
    )


class IntergridRouteModelsTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            example(route, index, position, slope)
            for route, slope in (("direct", .30), ("table", -.30))
            for index, position in enumerate(POSITIONS)
        ]
        self.rows.append(example("sparse", 0, POSITIONS[0], .1))
        self.rows[0]["intended_route"] = "table"
        self.checkpoint = models.fit_route_models(self.rows, seed=33)

    def test_common_pairs_are_deterministic_and_balanced_by_measured_route(self):
        checkpoint = self.checkpoint
        self.assertEqual(checkpoint, models.fit_route_models(reversed(self.rows), seed=33))
        self.assertEqual(checkpoint["routes"], ["direct", "table"])
        self.assertEqual(checkpoint["training_captures"], 10)
        self.assertEqual(checkpoint["training_pairs"], 40)
        self.assertEqual(checkpoint["unsupported_routes"]["sparse"]["paired_locations"], 0)
        for route, support in checkpoint["route_support"].items():
            with self.subTest(route=route):
                self.assertEqual(support["captures"], 5)
                self.assertEqual(support["pairs"], 20)
                self.assertEqual(support["loss_weight"], .5)
                self.assertAlmostEqual(sum(row["joint_loss_weight"] for row in support["locations"]), .5)
                self.assertEqual({row["conditional_loss_weight"] for row in support["locations"]}, {.2})

    def test_proposals_keep_source_order_and_recover_distinct_route_changes(self):
        menu = [dict(self.rows[0], intended_route="direct"), self.rows[5]]
        changes = {}
        for model in models.MODELS:
            with self.subTest(model=model):
                proposals = models.propose_route_candidates(
                    self.checkpoint, model, [.53, -.35], menu, context_id="fixture",
                )
                self.assertEqual([row["source_index"] for row in proposals], [0, 1])
                self.assertEqual([row["intended_route"] for row in proposals], ["direct", "table"])
                self.assertEqual([row["training_recipe_id"] for row in proposals], [row["id"] for row in menu])
                self.assertEqual([row["seed_q14"] for row in proposals], [[.2] * 6] * 2)
                changes[model] = [row["predicted_change"][0] for row in proposals]
                self.assertEqual(models.propose_route_candidates(
                    self.checkpoint, model, POSITIONS[0], menu,
                ), [])
        for model in ("conditioned", "specialist"):
            self.assertGreater(changes[model][0], .03)
            self.assertLess(changes[model][1], -.03)
            for pooled, routed in zip(changes["pooled"], changes[model], strict=True):
                self.assertLess(abs(pooled), abs(routed))

    def test_training_and_inference_guards(self):
        with self.assertRaisesRegex(ValueError, "lacks matched recipes"):
            models.fit_route_models(self.rows, seed=33, routes=("direct", "table", "sparse"))
        held_out = dict(self.rows[0], id="held-out", role="test")
        with self.assertRaisesRegex(ValueError, "held-out row"):
            models.fit_route_models([*self.rows, held_out], seed=33)
        with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": ""}):
            with self.assertRaisesRegex(ValueError, "4090"):
                models.fit_route_models(self.rows, seed=33, device="cuda:0")
        with self.assertRaisesRegex(ValueError, "one to eight"):
            models.propose_route_candidates(self.checkpoint, "pooled", [.53, -.35], self.rows, limit=9)

    def test_checkpoint_rejects_mutation_or_overwrite(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "routes.json"
            self.assertEqual(models.save_checkpoint(path, self.checkpoint), self.checkpoint["fingerprint"])
            self.assertEqual(models.load_checkpoint(path), self.checkpoint)
            with self.assertRaises(FileExistsError):
                models.save_checkpoint(path, self.checkpoint)
            changed = json.loads(path.read_text())
            changed["coefficients"]["pooled"][0][0] += 1
            path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                models.load_checkpoint(path)


if __name__ == "__main__":
    unittest.main()
