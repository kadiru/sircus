"""CPU planner contracts; proposals are not native simulation evidence."""

import copy
import json
import math
from pathlib import Path
import unittest

import numpy as np

from sircus.mjlab import sweep_contract as c


ROOT = Path(__file__).resolve().parents[3]


class SweepContractTests(unittest.TestCase):
    def setUp(self):
        self.scene = json.loads((ROOT / "sircus/isaac/scene.json").read_text())

    def request(self, **kwargs):
        return dict(request_id="first", mode="search", bucket_xy=[0.67, -0.5], **kwargs)

    def test_arbitrary_bucket_whole_footprint_validation(self):
        bounds = c.bucket_bounds(self.scene)
        np.testing.assert_allclose(bounds["x"], [-0.13, 1.47])
        np.testing.assert_allclose(bounds["y"], [-0.775, 0.275])
        for xy in [[-0.13, -0.775], [1.47, 0.275], [0.93, -0.233]]:
            value = c.validate_request(dict(self.request(), bucket_xy=xy), self.scene)
            self.assertEqual(value["bucket_xy"], xy)
        for xy in [[-0.131, 0], [0, 0.276], [float("nan"), 0], [True, 0]]:
            with self.assertRaises(ValueError):
                c.validate_request(dict(self.request(), bucket_xy=xy), self.scene)

    def test_explicit_search_budgets_and_policy_independence(self):
        request = c.validate_request(self.request(), self.scene)
        self.assertEqual(request["search_budget"], 64)
        self.assertEqual(request["cpu_candidates"], 256)
        for extra in [
            dict(search_budget=129),
            dict(search_budget=True),
            dict(cpu_candidates=1025),
            dict(cpu_candidates=16),
            dict(mode="policy"),
            dict(checkpoint="ignored"),
        ]:
            with self.assertRaises(ValueError):
                c.validate_request(dict(self.request(), **extra), self.scene)

    def test_structural_constraints_are_explicit(self):
        with self.assertRaisesRegex(ValueError, "XY conflict"):
            c.validate_request(
                self.request(
                    constraints=dict(
                        board_position=[0.6, -0.1, 0.3],
                        dropper_position=[0.61, -0.1, 0.7],
                    )
                ),
                self.scene,
            )
        with self.assertRaisesRegex(ValueError, "release height"):
            c.validate_request(
                self.request(constraints=dict(dropper_position=[0.6, -0.1, 3.0])),
                self.scene,
            )
        self.assertEqual(
            c.validate_request(
                self.request(constraints=dict(board_position=[0.6, -0.1, 0.3])),
                self.scene,
            )["constraints"]["board_position"],
            [0.6, -0.1, 0.3],
        )

    def test_exact_cache_identity_binds_all_executed_inputs(self):
        hashes = {"mesh": "a" * 64}
        sources = {"worker": "b" * 64}
        request = self.request()
        key = c.request_key(self.scene, hashes, request, source_hashes=sources)
        self.assertEqual(
            key,
            c.request_key(
                self.scene,
                hashes,
                dict(request, request_id="other"),
                source_hashes=sources,
            ),
        )
        changed = dict(request, bucket_xy=[0.67 + 1e-12, -0.5])
        self.assertNotEqual(
            key, c.request_key(self.scene, hashes, changed, source_hashes=sources)
        )
        self.assertNotEqual(
            key,
            c.request_key(
                self.scene,
                hashes,
                dict(request, search_budget=32),
                source_hashes=sources,
            ),
        )
        self.assertNotEqual(
            key,
            c.request_key(
                self.scene, {"mesh": "c" * 64}, request, source_hashes=sources
            ),
        )
        self.assertNotEqual(
            key,
            c.request_key(
                self.scene, hashes, request, source_hashes={"worker": "c" * 64}
            ),
        )
        physics = dict(c.PHYSICS, substeps=32)
        self.assertNotEqual(
            key,
            c.request_key(
                self.scene, hashes, request, source_hashes=sources, physics=physics
            ),
        )
        scene = copy.deepcopy(self.scene)
        scene["ball"]["mass_kg"] += 0.001
        self.assertNotEqual(
            key, c.request_key(scene, hashes, request, source_hashes=sources)
        )
        with self.assertRaises(ValueError):
            c.request_key(self.scene, hashes, request, source_hashes={})

    def test_inverse_gap_matches_sphere_contact_and_descending_preentry(self):
        gap = c.inverse_gap(self.scene, 0.32, 0.56, 28.0)
        self.assertAlmostEqual(gap, 0.449720701444664, places=12)
        radius = self.scene["ball"]["radius_m"]
        theta = math.radians(28)
        impact = 0.32 + (radius + 0.0025) / math.cos(theta)
        speed = math.sqrt(2 * 9.81 * (0.32 + gap + radius - impact))
        vx = 1.65 * speed * math.sin(theta) * math.cos(theta)
        vz = speed * (1.65 * math.cos(theta) ** 2 - 1)
        t = 0.56 / vx
        self.assertAlmostEqual(impact + vz * t - 0.5 * 9.81 * t * t, 0.183 + radius)
        self.assertLess(vz - 9.81 * t, 0)
        for args in [
            (0.1, 5, 60),
            (0.32, -0.1, 28),
            (0.32, 0.56, 0),
            (0.32, 0.56, float("nan")),
        ]:
            self.assertIsNone(c.inverse_gap(self.scene, *args))

    def test_reproducible_bounded_full_action_family(self):
        first, meta = c.proposals(self.scene, [0.67, -0.5], limit=256, seed=42)
        second, meta2 = c.proposals(self.scene, [0.67, -0.5], limit=256, seed=42)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(meta, meta2)
        self.assertEqual(first.shape, (256, 5))
        self.assertEqual(len(set(map(tuple, first))), 256)
        self.assertTrue((first >= c.BOUNDS["low"]).all())
        self.assertTrue((first <= c.BOUNDS["high"]).all())
        self.assertGreater(float(np.ptp(first[:, 2])), 0.44)
        self.assertGreater(float(np.ptp(first[:, 3])), 48)
        self.assertEqual(
            {row["method"] for row in meta},
            {"broad_space_filling", "inverse_ballistic_ring"},
        )
        self.assertEqual(
            c.NAMES, ("board_x_m", "board_y_m", "board_height_m", "tilt_deg", "gap_m")
        )

    def test_hint_provenance_is_geometry_only_never_success(self):
        hint = dict(
            id="native-source",
            bucket_xy_m=[0.67, -0.5],
            values=[0.67, -0.1, 0.32, 28, 0.45],
            q10=[0.1] * 6,
            q14=[0.2] * 6,
            route_success=True,
            reward=100.0,
            qualified=True,
        )
        values, meta = c.proposals(self.scene, [0.67, -0.5], limit=64, hints=[hint])
        np.testing.assert_array_equal(values[0], hint["values"])
        self.assertEqual(meta[0]["method"], "exact_source_geometry")
        self.assertEqual(meta[0]["source_hint"]["q10"], hint["q10"])
        self.assertEqual(
            sum(np.allclose(v, hint["values"], atol=1e-15, rtol=0) for v in values), 1
        )
        for row in meta:
            for key in ["route_success", "reward", "qualified", "success"]:
                self.assertNotIn(key, row)
                self.assertNotIn(key, row.get("source_hint", {}))
        translated, translated_meta = c.proposals(
            self.scene, [0.69, -0.51], limit=64, hints=[hint]
        )
        self.assertFalse(
            any(row["method"] == "exact_source_geometry" for row in translated_meta)
        )
        self.assertTrue(
            any(row["method"] == "relative_source_geometry" for row in translated_meta)
        )

    def test_out_of_bound_hints_not_clipped_to_look_admissible(self):
        hint = dict(
            id="outside", bucket_xy_m=[0.67, -0.5], values=[0.67, -0.1, 9.0, 28, 0.45]
        )
        values, meta = c.proposals(self.scene, [0.67, -0.5], limit=32, hints=[hint])
        self.assertFalse(any(row["method"].endswith("source_geometry") for row in meta))
        self.assertEqual(values.shape, (32, 5))

    def test_nearby_query_leads_with_all_translated_source_geometries(self):
        hints = [
            dict(
                id=f"source-{i}",
                bucket_xy_m=[0.48, -0.65],
                values=[0.50 + i * 0.002, -0.10, 0.32, 28, 0.45],
                q10=[0.1 + i * 0.001] * 6,
                q14=[0.2] * 6,
                route_success=True,
            )
            for i in range(17)
        ]
        target = np.array([0.49, -0.64])
        values, meta = c.proposals(self.scene, target, limit=64, hints=hints)
        for i, hint in enumerate(hints):
            expected = np.asarray(hint["values"], float)
            expected[:2] = target + (
                np.asarray(hint["values"][:2]) - hint["bucket_xy_m"]
            )
            np.testing.assert_array_equal(values[i], expected)
            self.assertEqual(meta[i]["source_id"], hint["id"])
            self.assertEqual(meta[i]["method"], "relative_source_geometry")
            self.assertEqual(meta[i]["rotation_deg"], 0.0)
            # This is a continuation seed for another absolute request, not an
            # exact old joint branch or an inherited success certificate.
            self.assertEqual(meta[i]["source_hint"]["values"], hint["values"])
            self.assertEqual(meta[i]["source_hint"]["bucket_xy_m"], hint["bucket_xy_m"])
            self.assertEqual(meta[i]["source_hint"]["q10"], hint["q10"])
            self.assertFalse(
                np.array_equal(target, meta[i]["source_hint"]["bucket_xy_m"])
            )
            self.assertNotIn("route_success", meta[i])
        self.assertTrue(any(m["method"] == "broad_space_filling" for m in meta[17:]))
        self.assertTrue(any(m["method"] == "inverse_ballistic_ring" for m in meta[17:]))

    def test_release_lock_is_constructed_exactly_not_clipped(self):
        lock = [0.61, -0.1, 0.75]
        values, _ = c.proposals(
            self.scene, [0.7, -0.5], limit=64, constraints=dict(dropper_position=lock)
        )
        np.testing.assert_array_equal(
            values[:, :2], np.tile(lock[:2], (len(values), 1))
        )
        np.testing.assert_allclose(
            values[:, 2] + values[:, 4] + 0.0375, lock[2], atol=1e-15, rtol=0
        )
        self.assertGreater(float(np.ptp(values[:, 2])), 0.2)

    def test_rank_never_promotes_invalid_or_unreleased_capture_flag(self):
        good = dict(
            native_release=True, invalid=None, route_success=True, route_quality=2
        )
        fake = dict(
            native_release=False, invalid=None, route_success=True, route_quality=2
        )
        missed = dict(
            native_release=True, invalid=None, route_success=False, board_bounce=True
        )
        self.assertGreater(c.rank_row(good), c.rank_row(fake))
        self.assertGreater(c.rank_row(missed), c.rank_row(fake))


if __name__ == "__main__":
    unittest.main()
