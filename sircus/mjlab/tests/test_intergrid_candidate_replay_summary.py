"""Keep fresh candidate comparisons separate from repeats and archived results."""
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "Route study imports require PyTorch")
class CandidateReplaySummaryTests(unittest.TestCase):
    def setUp(self):
        from sircus.mjlab.intergrid_candidate_replay_summary import summarize

        self.summarize = summarize
        self.plan = dict(cases=[], setup_failures=[dict(layout=dict(id="blocked"), reason="no_proposals")])
        self.rows = {}

    def add_case(self, shots):
        identity = f"case-{len(self.plan['cases'])}"
        case = dict(layout=dict(id=identity), prior_selector_id=f"{identity}-route_selector",
                    prior_direct_first_id=f"{identity}-direct_first", requests=[], candidates=[])
        for index, (route, result, previously_captured) in enumerate(shots):
            name = f"{identity}-replay-{index:02d}"
            method = "baseline_repeat" if index == 0 else "alternative"
            request = dict(id=name, method=method, intended_route=route)
            case["requests"].append(request)
            case["candidates"].insert(0, dict(id=name, already_direct_first=previously_captured is not None,
                                               prior_direct_first_capture=previously_captured))
            self.rows[name] = dict(request, native_release=True, invalid_physics=result == "invalid_physics",
                                   route_success=result in {"direct", "table", "unclassified_capture"},
                                   shot_route=result if result in {"direct", "table"} else None,
                                   experimental_route=dict(robot_assisted_capture=result == "arm_assisted_capture"))
        self.plan["cases"].append(case)

    def test_repeat_wins_prior_direct_first_replays_and_novel_wins_separate(self):
        self.add_case([("direct", "flight_miss", None), ("direct", "direct", True),
                       ("direct", "table", None), ("table", "table", None)])
        self.add_case([("table", "table", False), ("table", "invalid_physics", None)])
        self.add_case([("direct", "invalid_physics", None)])
        self.add_case([("direct", "arm_assisted_capture", None), ("table", "flight_miss", True)])
        self.add_case([("table", "table", None), ("table", "unclassified_capture", None)])

        report = self.summarize(self.plan, self.rows)
        self.assertEqual((report["positions"], report["no_alternative_positions"],
                          report["no_novel_alternative_positions"]), (5, 1, 2))
        self.assertEqual(report["baseline_repeat"], dict(
            releases=5, captures=2, matches_prior_direct_first=1,
            outcomes=dict(flight_miss=1, table=2, invalid_physics=1, arm_assisted_capture=1)))
        self.assertEqual(report["novel_alternatives"], dict(
            releases=4, captures=3, positions_tested=3, positions_captured=2,
            same_route_capture_positions=2, other_route_capture_positions=1,
            outcomes=dict(table=2, invalid_physics=1, unclassified_capture=1)))
        self.assertEqual(report["prior_direct_first_replays"], dict(
            releases=2, captures=1, previous_captures=2, capture_disagreements=1,
            outcomes=dict(direct=1, flight_miss=1)))
        self.assertEqual(report["paired_novel"], dict(both_capture=1, repeat_only=1, novel_only=1, neither_capture=0))
        self.assertEqual(report["setup_failures"], self.plan["setup_failures"])
        self.assertEqual([row["id"] for row in report["cases"][0]["alternatives"]],
                         ["case-0-replay-01", "case-0-replay-02", "case-0-replay-03"])

    def test_missing_replay_or_mismatched_metadata_is_rejected(self):
        self.add_case([("direct", "flight_miss", False)])
        with self.assertRaisesRegex(ValueError, "one native release"):
            self.summarize(self.plan, {})
        self.plan["cases"][0]["candidates"][0]["prior_direct_first_capture"] = None
        with self.assertRaisesRegex(ValueError, "direct-first comparison metadata"):
            self.summarize(self.plan, self.rows)
