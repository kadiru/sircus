"""Full-record equivalence checks for the opt-in broad-phase implementation."""
import json
import os
from pathlib import Path
from types import MethodType, SimpleNamespace
import unittest

import numpy as np

from sircus.mjlab.arm_check_speed import install_vectorized_gate
from sircus.mjlab.transit_audit import GeometricGate, audit_segment

try:
    import mujoco
except ImportError:
    mujoco = None


def fixture(centers, *, pairs=None, distances=None, bounds=None, contacts=(), inconsistent=()):
    """Small collision fixture with observable narrow-phase call ordering."""
    centers = np.asarray(centers, dtype=float)
    count = len(centers)
    gate = GeometricGate.__new__(GeometricGate)
    gate.clearance = .005
    gate.names = [f"geometry-{i}" for i in range(count)]
    gate.body_names = [f"body-{i}" for i in range(count)]
    gate.moving_geoms = list(range(count))
    gate.carried_exclusions = []
    gate.pairs = list(pairs) if pairs is not None else [(i, j) for i in range(count) for j in range(i+1, count)]
    gate.native = SimpleNamespace(limits10=np.tile([-3., 3.], (6, 1)),
        limits14=np.tile([-3., 3.], (6, 1)), board=0, flange=1)
    gate.m = SimpleNamespace(ngeom=count, geom_size=np.asarray([[.1, .1, .01]]),
        geom=lambda name: SimpleNamespace(id=0))
    gate.d = SimpleNamespace(geom_xmat=np.tile(np.eye(3).ravel(), (count, 1)),
        geom_xpos=centers.copy(), xpos=np.asarray([[0., 0., .3], [0., 0., .9]]),
        xmat=np.tile(np.eye(3).ravel(), (2, 1)),
        contact=[SimpleNamespace(geom=pair, dist=distance) for pair, distance in contacts], ncon=len(contacts))
    gate.local_points = [np.zeros((1, 3)) for _ in range(count)]
    gate.local_axes = [np.eye(3) for _ in range(count)]
    gate.state = lambda q: None
    gate.native_calls = []
    def native_distance(m, d, i, j, limit, witness):
        gate.native_calls.append((i, j))
        distance = (distances or {}).get((i, j), .25)
        witness[:] = [0., 0., 0., abs(distance), 0., 0.]
        if (i, j) in inconsistent:
            witness[3] += .01
        return distance
    gate.mj = SimpleNamespace(mj_geomDistance=native_distance)
    if bounds is not None:
        gate.separation_lower_bound = lambda i, j: bounds.get((i, j), 0.)
    return gate


class BroadPhaseTests(unittest.TestCase):
    def equivalent(self, factory, q=None, corridor=None):
        q = np.zeros(12) if q is None else q
        original, optimized = factory(), factory()
        install_vectorized_gate(optimized)
        expected = original.check(q, corridor=corridor)
        actual = optimized.check(q, corridor=corridor)
        self.assertEqual(actual, expected)
        self.assertEqual(optimized.native_calls, original.native_calls)
        return actual

    def test_clear_pass_and_minimum_tie_keep_original_pair_order(self):
        result = self.equivalent(lambda: fixture([[0, 0, 0], [.02, 0, 0], [0, .02, 0]],
            pairs=[(0, 2), (0, 1), (1, 2)]))
        self.assertTrue(result["passed"])
        self.assertEqual(result["closest_pair"], ["geometry-0", "geometry-2"])

    def test_threshold_neighbors_use_original_scalar_decision(self):
        for value in (np.nextafter(.005, 0), .005, np.nextafter(.005, 1), .004, .006):
            with self.subTest(value=value):
                self.equivalent(lambda: fixture([[0, 0, 0], [value, 0, 0]],
                    distances={(0, 1): value}, bounds={(0, 1): value}))
        # Three-axis norms may round differently in scalar vs batched reduction.
        direction = np.asarray([1., 2., 3.]); direction /= np.linalg.norm(direction)
        for value in np.linspace(.005 - 1e-16, .005 + 1e-16, 31):
            self.equivalent(lambda: fixture([[0, 0, 0], direction * value],
                distances={(0, 1): 0.}, bounds={}))

    def test_unresolved_pairs_witness_fallback_and_native_order(self):
        result = self.equivalent(lambda: fixture([[0, 0, 0], [.001, 0, 0], [.002, 0, 0]],
            pairs=[(1, 2), (0, 2), (0, 1)],
            distances={(1, 2): .001, (0, 2): .003, (0, 1): 0.},
            bounds={}, inconsistent=[(0, 2)]))
        self.assertFalse(result["passed"])
        self.assertEqual(result["inconsistent_distance_queries"], 1)
        self.assertEqual([r["pair"] for r in result["unresolved_clearance_pairs"]],
            [["geometry-1", "geometry-2"], ["geometry-0", "geometry-2"], ["geometry-0", "geometry-1"]])

    def test_sat_can_prove_clearance_without_changing_contact_checks(self):
        args = dict(distances={(0, 1): 0.}, bounds={(0, 1): .006})
        result = self.equivalent(lambda: fixture([[0, 0, 0], [.001, 0, 0]], **args))
        self.assertTrue(result["passed"])
        result = self.equivalent(lambda: fixture([[0, 0, 0], [.001, 0, 0]],
            contacts=[((0, 1), -.001)], **args))
        self.assertEqual(result["reason"], "clearance")
        self.assertEqual(result["margin_contacts"][0]["distance_m"], -.001)

    def test_contact_exclusions_and_nonmoving_pairs_preserved(self):
        def create():
            gate = fixture([[0, 0, 0], [.02, 0, 0], [.04, 0, 0]],
                contacts=[((0, 1), -.01), ((1, 2), -.02)])
            gate.carried_exclusions = [["body-0", "body-1"]]
            gate.moving_geoms = [0]
            return gate
        self.assertTrue(self.equivalent(create)["passed"])

    def test_joint_and_corridor_rules_unchanged(self):
        factory = lambda: fixture([[0, 0, 0], [.02, 0, 0]])
        for q in (np.ones(12)*4, np.full(12, np.nan), np.full(12, np.inf), np.zeros(11)):
            self.assertEqual(self.equivalent(factory, q)["reason"], "joint_limits_or_nonfinite")
        for corridor in ({"board_top_max_m": .3}, {"dropper_flange_min_m": .91}):
            self.assertEqual(self.equivalent(factory, corridor=corridor)["reason"], "corridor")
        self.assertTrue(self.equivalent(factory, corridor={"board_top_max_m": .31,
            "dropper_flange_min_m": .9})["passed"])

    def test_one_meter_sentinel_empty_pairs_and_random_minima(self):
        for value in (1., np.nextafter(1., 0), np.nextafter(1., 2), 2.):
            self.equivalent(lambda: fixture([[0, 0, 0], [value, 0, 0]]))
        self.equivalent(lambda: fixture([[0, 0, 0], [.02, 0, 0]], pairs=[]))
        rng = np.random.default_rng(2701)
        for _ in range(20):
            centers = rng.uniform(-.5, .5, (30, 3))
            self.equivalent(lambda: fixture(centers))

    def test_installer_is_local_idempotent_and_handles_pair_changes(self):
        gate = fixture([[0, 0, 0], [.02, 0, 0], [.04, 0, 0]])
        untouched = fixture([[0, 0, 0], [.02, 0, 0]])
        self.assertIs(install_vectorized_gate(gate), gate)
        self.assertIs(install_vectorized_gate(gate), gate)
        self.assertIs(untouched.check.__func__, GeometricGate.check)
        gate.pairs[:] = [(0, 2)]
        self.assertEqual(gate.check(np.zeros(12)), GeometricGate.check(gate, np.zeros(12)))
        untouched.check = MethodType(lambda self, q: {}, untouched)
        with self.assertRaises(TypeError):
            install_vectorized_gate(untouched)

    def test_mid_path_failure_and_reverse_sampling_are_identical(self):
        def create():
            gate = fixture([[0, 0, 0], [.02, 0, 0]], distances={(0, 1): 0.}, bounds={})
            gate.state = lambda q: gate.d.geom_xpos.__setitem__((1, 0), abs(q[0] - .05))
            return gate
        a, b = np.zeros(12), np.r_[.1, np.zeros(11)]
        for start, end in ((a, b), (b, a)):
            original, optimized = create(), install_vectorized_gate(create())
            self.assertEqual(audit_segment(original, start, end), audit_segment(optimized, start, end))


@unittest.skipUnless(mujoco is not None and os.environ.get("SIRCUS_IK_ASSETS"),
    "Optional pinned MuJoCo/native assets required")
class NativeBroadPhaseTests(unittest.TestCase):
    def test_native_pass_rejection_and_sampled_paths_match_full_records(self):
        root = Path(__file__).resolve().parents[3]
        scene = json.loads(Path(os.environ.get("SIRCUS_TRANSIT_SCENE", root / "sircus/isaac/scene.json")).read_text())
        catalogue = json.loads(Path(os.environ.get("SIRCUS_TRANSIT_CATALOGUE",
            root / "baselines/mjlab-ik-stability-20261102/candidate-catalogue.json")).read_text())
        rows = {row["target"]: row for row in catalogue["candidates"] if row.get("values")}
        original = GeometricGate(scene, os.environ["SIRCUS_IK_ASSETS"], rows["old-test-21"]["bucket_xy_m"])
        optimized = install_vectorized_gate(GeometricGate(scene, os.environ["SIRCUS_IK_ASSETS"],
            rows["old-test-21"]["bucket_xy_m"]))
        for name in ("old-test-21", "old-test-39"):
            row = rows[name]
            original.bucket_xy = optimized.bucket_xy = np.asarray(row["bucket_xy_m"])
            q = np.r_[row["q10"], row["q14"]]
            self.assertEqual(original.check(q), optimized.check(q))
            target = q + np.linspace(-.01, .01, 12)
            for a, b in ((q, target), (target, q)):
                self.assertEqual(audit_segment(original, a, b), audit_segment(optimized, a, b))


if __name__ == "__main__":
    unittest.main()
