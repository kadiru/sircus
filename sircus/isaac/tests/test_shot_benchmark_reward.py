"""Ordered synthetic scoring traces; these are not simulated shot evidence."""

import copy
import hashlib
import math
import pathlib
import sys
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import farm_contract
from board_route import BoardRoute
from entry_quality import EntryQuality
from rl_contract import ShotScore
import shot_benchmark_reward as subject


DT = 1/240
SCENE = {
    "simulation": {"physics_dt_s": DT},
    "ball": {"radius_m": .0375},
    "bucket": {
        "center_xy_m": [0., 0.], "inner_top_diameter_m": .177,
        "inner_bottom_diameter_m": .150, "assumed_interior_depth_m": .175,
        "assumed_bottom_thickness_m": .008, "segments": 64,
        "reported_height_m": .175, "reported_height_status": "unconfirmed",
    },
}
TOP = .183
NORMAL = [0., 0., 1.]


def sample(position, *, board=0., bucket=0., table=0., floor=0., robot=0.,
           unattributed=0., vertical_speed=0.):
    return (position, [0., 0., vertical_speed],
            [board, bucket, table, floor, robot, unattributed])


def bounced(distance=.4):
    return [sample([distance, 0., TOP+.1], board=1., vertical_speed=-1.),
            sample([distance, 0., TOP+.1], vertical_speed=1.)]


def mouth(distance=0.):
    return sample([distance, 0., TOP-SCENE["ball"]["radius_m"]])


def capture(*, rim=False):
    return (bounced() + [sample([0., 0., TOP], bucket=.4 if rim else 0.)]
            + [sample([0., 0., .08]) for _ in range(245)])


def traces(samples):
    return tuple(np.asarray([row[i] for row in samples], dtype=float) for i in range(3))


def native_row(samples):
    """Independent use of unchanged native scorer, quality and route contracts."""
    score = ShotScore(copy.deepcopy(SCENE["bucket"]), SCENE["ball"]["radius_m"])
    route, quality = BoardRoute(DT), EntryQuality(SCENE["bucket"], SCENE["ball"]["radius_m"])
    for position, velocity, forces in samples:
        score.step(position, forces[0], DT, velocity[2])
        quality.step(position, forces[1], score.entered)
        route.step(dict(zip(subject.FORCE_NAMES, forces)))
        if score.success or position[2] < -.2:
            break
    row = score.result()
    row.update(quality.result(row["success"]))
    row.update(route.result(row["success"], row["quality_score"]))
    row["invalid"] = None
    return row


def evidence(samples):
    return subject.build_features(SCENE, [0., 0.], *traces(samples), DT, NORMAL)


def evaluate(samples):
    row, packet = native_row(samples), evidence(samples)
    return subject.reward(row, packet), row, packet


class RewardTests(unittest.TestCase):
    def test_near_and_far_bounced_misses_are_graded_without_changing_capture(self):
        near, near_row, near_packet = evaluate(bounced()+[mouth(.1)])
        far, far_row, far_packet = evaluate(bounced()+[mouth(.8)])
        self.assertGreater(near, far)
        for value, row in ((near, near_row), (far, far_row)):
            self.assertGreaterEqual(value, -.7)
            self.assertLessEqual(value, 0.)
            self.assertFalse(row["route_success"])
        self.assertAlmostEqual(near_packet["closest_opening_distance_preexternal_m"], .1)
        self.assertGreater(far_packet["closest_opening_distance_preexternal_m"], .4)
        expected = -.6+.45*math.exp(-.1/.2)
        self.assertAlmostEqual(near, expected)

    def test_post_table_near_bucket_retention_cannot_improve_miss_reward(self):
        prefix = bounced(.6)+[mouth(.5)]
        external = sample([.5, 0., TOP-.0375], table=2.)
        far_samples = prefix+[external]+[mouth(.7) for _ in range(245)]
        near_samples = prefix+[external]+capture()[2:]
        far, far_row, far_packet = evaluate(far_samples)
        near, near_row, near_packet = evaluate(near_samples)
        self.assertGreater(far_row["closest_opening_distance_m"], near_row["closest_opening_distance_m"])
        self.assertFalse(far_row["success"])
        self.assertTrue(near_row["success"])
        self.assertFalse(near_row["route_success"])
        for key in ("prefix_steps", "first_external_step", "board_bounce_preexternal",
                    "closest_opening_distance_preexternal_m", "max_retention_preexternal_s"):
            self.assertEqual(near_packet[key], far_packet[key])
        self.assertEqual(near_packet["first_external_step"], 4)
        self.assertEqual(near_packet["prefix_steps"], 3)
        self.assertEqual(near, far)
        self.assertLessEqual(near, -.1)

    def test_first_external_sample_entirely_excluded_even_if_board_contact(self):
        samples = [sample([.4, 0., .4]), sample([.1, 0., TOP], board=1., table=1.)]
        samples += [sample([0., 0., TOP], vertical_speed=1.)]+capture()[2:]
        value, row, packet = evaluate(samples)
        self.assertTrue(row["board_bounce"])
        self.assertEqual(packet["first_external_step"], 2)
        self.assertEqual(packet["prefix_steps"], 1)
        self.assertFalse(packet["board_bounce_preexternal"])
        self.assertIsNone(packet["closest_opening_distance_preexternal_m"])
        self.assertEqual(value, -.8)

    def test_external_before_board_has_no_bounce_progress(self):
        value, row, packet = evaluate([sample([.4, 0., .3], robot=.1)]+capture())
        self.assertTrue(row["board_bounce"])
        self.assertEqual(packet["prefix_steps"], 0)
        self.assertEqual(value, -.8)

    def test_bucket_rim_contact_is_allowed_and_only_clean_bonus_differs(self):
        clean, clean_row, clean_packet = evaluate(capture())
        assisted, assisted_row, assisted_packet = evaluate(capture(rim=True))
        self.assertTrue(clean_row["route_success"])
        self.assertTrue(assisted_row["route_success"])
        self.assertEqual(clean_row["route_quality"], 2)
        self.assertEqual(assisted_row["route_quality"], 1)
        self.assertEqual(clean, 1.6)
        self.assertEqual(assisted, 1.5)
        self.assertIsNone(clean_packet["first_external_step"])
        self.assertIsNone(assisted_packet["first_external_step"])

    def test_success_termination_ignores_later_full_batch_tail_contacts(self):
        samples = capture()+[sample([0., 0., .08], table=10.) for _ in range(20)]
        value, row, packet = evaluate(samples)
        self.assertEqual(value, 1.6)
        self.assertEqual(row["external_contact_steps"], 0)
        self.assertLess(packet["steps_seen"], packet["trace_steps"])
        self.assertEqual(packet["prefix_steps"], packet["steps_seen"])

    def test_floor_termination_ignores_future_near_bucket_samples(self):
        samples = bounced(.5)+[sample([.5, 0., -.21], floor=1.)]+capture()[2:]
        value, row, packet = evaluate(samples)
        self.assertFalse(row["success"])
        self.assertEqual(packet["steps_seen"], 3)
        self.assertEqual(packet["first_external_step"], 3)
        self.assertEqual(packet["prefix_steps"], 2)
        self.assertLessEqual(value, -.1)

    def test_external_threshold_matches_native_strict_threshold(self):
        at_threshold = bounced()+[sample([.1, 0., TOP-.0375], table=.01)]
        above_threshold = bounced()+[sample([.1, 0., TOP-.0375], table=.0100001)]
        _, _, at_packet = evaluate(at_threshold)
        _, _, above_packet = evaluate(above_threshold)
        self.assertIsNone(at_packet["first_external_step"])
        self.assertEqual(above_packet["first_external_step"], 3)

    def test_preexternal_retention_helps_miss_but_never_becomes_capture(self):
        short_samples = bounced()+[sample([0., 0., TOP])]+[sample([0., 0., .08])]*24
        long_samples = bounced()+[sample([0., 0., TOP])]+[sample([0., 0., .08])]*120
        short, short_row, short_packet = evaluate(short_samples)
        long, long_row, long_packet = evaluate(long_samples)
        self.assertGreater(long, short)
        self.assertAlmostEqual(short_packet["max_retention_preexternal_s"], .1)
        self.assertAlmostEqual(long_packet["max_retention_preexternal_s"], .5)
        self.assertFalse(short_row["route_success"])
        self.assertFalse(long_row["route_success"])
        dirty, dirty_row, dirty_packet = evaluate(long_samples+[sample([0., 0., .08], table=.1)])
        self.assertFalse(dirty_row["route_success"])
        self.assertEqual(dirty_packet["max_retention_preexternal_s"], long_packet["max_retention_preexternal_s"])
        self.assertAlmostEqual(dirty, long-.1)

    def test_partial_exit_resets_continuous_retention_as_before(self):
        samples = bounced()+[sample([0., 0., TOP])]+[sample([0., 0., .08])]*150
        samples += [sample([0., 0., TOP])]+[sample([0., 0., .08])]*150
        _, row, packet = evaluate(samples)
        self.assertFalse(row["route_success"])
        self.assertAlmostEqual(row["max_retention_s"], 150*DT)
        self.assertEqual(packet["max_retention_preexternal_s"], row["max_retention_s"])

    def test_valid_no_board_and_invalid_are_separate(self):
        value, _, packet = evaluate([mouth(.1)])
        self.assertFalse(packet["board_bounce_preexternal"])
        self.assertEqual(value, -.8)
        self.assertEqual(subject.reward({"invalid": "ik_rejected", "route_success": False}), -1.)
        with self.assertRaises(ValueError):
            subject.reward({"invalid": "ik_rejected", "route_success": True})

    def test_missing_legacy_or_conflicting_feature_packets_fail_closed(self):
        _, row, packet = evaluate(bounced()+[mouth(.1)])
        bad_packets = [None, {}, dict(packet, feature_version="old"),
            dict(packet, first_external_step=2), dict(packet, prefix_steps=0),
            dict(packet, max_retention_preexternal_s=2.)]
        changed = copy.deepcopy(packet)
        changed["observed"]["closest_opening_distance_m"] += .1
        bad_packets.append(changed)
        for bad in bad_packets:
            with self.subTest(packet=bad), self.assertRaises(ValueError):
                subject.reward(row, bad)

    def test_trace_input_validation_and_scene_inputs_are_not_mutated(self):
        samples = bounced()+[mouth(.1)]
        args = list(traces(samples))
        scene_before, args_before = copy.deepcopy(SCENE), [a.copy() for a in args]
        evidence(samples)
        self.assertEqual(SCENE, scene_before)
        for actual, before in zip(args, args_before):
            np.testing.assert_array_equal(actual, before)
        bad = []
        nan_positions = args[0].copy(); nan_positions[0, 0] = np.nan
        negative_forces = args[2].copy(); negative_forces[0, 2] = -.1
        bad.extend([(nan_positions, args[1], args[2], DT, NORMAL),
                    (args[0], args[1][:-1], args[2], DT, NORMAL),
                    (args[0], args[1], negative_forces, DT, NORMAL),
                    (args[0], args[1], args[2], DT*2, NORMAL),
                    (args[0], args[1], args[2], DT, [0, 0, 2])])
        for values in bad:
            with self.subTest(values=values), self.assertRaises(ValueError):
                subject.build_features(SCENE, [0, 0], *values)

    def test_statistics_keep_historical_capture_labels_and_rows_unchanged(self):
        scenarios = [capture(), capture(rim=True),
            bounced()+[sample([.4, 0., .3], table=1.)]+capture()[2:],
            bounced()+[mouth(.4)], [mouth(.2)]]
        rows, packets = [native_row(s) for s in scenarios], [evidence(s) for s in scenarios]
        rejected = dict(invalid="IK", route_success=False, route_quality=0, single_bounce_success=False)
        rows.append(rejected); packets.append(None)
        before = copy.deepcopy(rows)
        historical = farm_contract.statistics(rows)
        current = subject.statistics(rows, packets)
        for key in ("attempts", "captures", "clean", "single_bounce", "invalid"):
            self.assertEqual(current[key], historical[key])
        self.assertEqual(current["captures"], 2)
        self.assertEqual(current["clean"], 1)
        self.assertEqual(rows, before)
        self.assertNotEqual(current["mean_reward"], historical["mean_reward"])
        with self.assertRaises(ValueError):
            subject.statistics(rows, packets[:-1])

    def test_historical_reward_source_unchanged(self):
        path = pathlib.Path(farm_contract.__file__)
        # Frozen baseline reward file, recorded before the new module existed.
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
            "e1a0d46f48926105a58d885e7b1c8761f03b8745937d6232aa130f00043038b5")

    def test_empty_statistics_are_explicit(self):
        result = subject.statistics([], [])
        self.assertEqual(result["attempts"], 0)
        self.assertEqual(result["captures"], 0)
        self.assertIsNone(result["mean_reward"])


if __name__ == "__main__":
    unittest.main()
