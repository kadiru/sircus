"""Exercise paired accounting and fixed-pose evidence without a GPU rollout."""
from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec("torch"), "Existing native audit imports require PyTorch")
class ClearanceAuditTests(unittest.TestCase):
    def setUp(self):
        from sircus.mjlab import clearance_expansion_audit as audit
        from sircus.mjlab.benchmark_verify import Checks

        self.audit = audit
        self.checks = Checks
        self.layout = dict(id="case", context_id="case", bucket_xy_m=[.8, -.5],
                           split="test", stratum="interior")

    def request(self, identity="source", gap=.2):
        return dict(id=identity, context_id="case", bucket_xy_m=[.8, -.5], phase="test", role="test",
                    intended_route="direct", method="direct", values=[.4, -.5, .3, 28., gap],
                    board_heading_offset_deg=0., board_twist_deg=0.,
                    selected_q10=[.1] * 6, selected_q14=[.2] * 6)

    @staticmethod
    def observation(label="direct", contacts=0):
        return dict(invalid_physics=label == "invalid_physics",
                    route_success=label in {"direct", "table", "unclassified_capture"},
                    shot_route=label if label in {"direct", "table"} else None,
                    experimental_route=dict(robot_assisted_capture=label == "arm_assisted_capture",
                                            robot_contact_steps=contacts))

    @staticmethod
    def stage(seconds=()):
        return dict(proposals=[], admitted=[], records=[dict(timing=dict(wall_s=t)) for t in seconds])

    def paired_fixture(self):
        layouts, slots, records, sites = [], [], {}, []
        combinations = [(False, .2, .2, "direct", "flight_miss"),
                        (True, None, .21, None, "table"),
                        (True, .22, .23, "flight_miss", "direct"),
                        (True, None, None, None, None)]
        for i, (trigger, left, right, lout, rout) in enumerate(combinations):
            layout = dict(self.layout, id=f"case-{i}", context_id=f"case-{i}")
            layouts.append(layout)
            sites.append(dict(layout=layout, trigger=trigger,
                              latency_s=dict(current_planner=2. * (i + 1), clearance_expansion=4. * (i + 1)),
                              plan={name: self.stage([i + 1.]) for name in self.audit.STAGES}))
            for method, gap, label in zip(self.audit.METHODS, (left, right), (lout, rout), strict=True):
                identity = layout["id"] + "-" + method
                request = None if gap is None else dict(self.request(identity, gap), context_id=layout["id"])
                slots.append(dict(id=identity, method=method, context_id=layout["id"],
                                  bucket_xy_m=layout["bucket_xy_m"],
                                  candidate_pool="branches" if i == 2 and method == "clearance_expansion" else "base",
                                  intended_route=None if request is None else "direct", request=request))
                if request is not None:
                    records[identity] = self.observation(label, contacts=int(label == "flight_miss"))
        return layouts, slots, records, sites

    def test_unresolved_slots_remain_in_denominator_and_repeats_stay_visible(self):
        report = self.audit.summarize(*self.paired_fixture())
        old, new = (report["methods"][name] for name in self.audit.METHODS)
        self.assertEqual((old["positions"], old["admitted"], old["captures"], old["capture_rate"]), (4, 2, 1, .25))
        self.assertEqual((new["positions"], new["admitted"], new["captures"], new["capture_rate"]), (4, 3, 2, .5))
        self.assertEqual((old["unresolved"], new["unresolved"], old["flight_misses"], new["flight_misses"]), (2, 1, 1, 1))
        self.assertEqual((old["robot_contacts"], new["robot_contacts"]), (1, 1))
        self.assertEqual(new["selected_stages"], dict(base=2, fallback=0, expansion=0, branches=1, unresolved=1))
        self.assertEqual({key: report["paired"][key] for key in ("gained", "lost", "both", "neither", "changed_positions")},
                         dict(gained=2, lost=1, both=0, neither=1, changed_positions=2))
        self.assertEqual(report["same_command"], dict(positions=1, capture_disagreements=1, outcome_disagreements=1))
        self.assertEqual(report["triggered"], dict(positions=3, current_captures=0, expanded_captures=2, changed_commands=2))
        self.assertEqual(report["latency"]["current_planner"]["median_s"], 5.)
        self.assertEqual(report["latency"]["clearance_expansion"]["median_s"], 10.)

    def test_missing_extra_or_duplicate_native_observation_is_rejected(self):
        fixture = self.paired_fixture()
        del fixture[2][fixture[1][0]["id"]]
        with self.assertRaisesRegex(ValueError, "exactly one observation"):
            self.audit.summarize(*fixture)
        fixture = self.paired_fixture()
        fixture[2]["unplanned"] = self.observation()
        with self.assertRaisesRegex(ValueError, "exactly one observation"):
            self.audit.summarize(*fixture)
        fixture = self.paired_fixture()
        fixture[1].append(deepcopy(fixture[1][0]))
        with self.assertRaisesRegex(ValueError, "one attempt"):
            self.audit.summarize(*fixture)

    def test_nontriggered_command_changes_are_rejected(self):
        fixture = self.paired_fixture()
        fixture[1][1]["request"]["selected_q14"][0] += .5
        with self.assertRaisesRegex(ValueError, "Non-triggered"):
            self.audit.summarize(*fixture)

    def test_nonfinite_negative_or_boolean_timings_are_rejected(self):
        for bad in (float("nan"), float("inf"), -1., True):
            with self.subTest(value=bad):
                fixture = self.paired_fixture()
                fixture[3][0]["plan"]["branches"]["records"][0]["timing"]["wall_s"] = bad
                with self.assertRaisesRegex(ValueError, "finite and nonnegative"):
                    self.audit.summarize(*fixture)

    def test_total_wall_latency_cannot_be_less_than_its_baseline_preparation(self):
        fixture = self.paired_fixture()
        fixture[3][0]["latency_s"]["clearance_expansion"] = 1.
        with self.assertRaisesRegex(ValueError, "include baseline preparation"):
            self.audit.summarize(*fixture)

    def test_robot_assisted_retention_is_reported_separately_from_standard_capture(self):
        fixture = self.paired_fixture()
        fixture[2][fixture[1][0]["id"]] = self.observation("arm_assisted_capture", contacts=4)
        result = self.audit.summarize(*fixture)["methods"]["current_planner"]
        self.assertEqual(result["captures"], 0)
        self.assertEqual(result["outcomes"]["arm_assisted_capture"], 1)
        self.assertEqual(result["robot_contacts"], 2)

    def branch_fixture(self):
        source = self.request()
        proposal = {key: value for key, value in source.items() if not key.startswith("selected_")}
        proposal.update(id="branch", source_request_id=source["id"],
                        seed_q10=list(source["selected_q10"]), seed_q14=list(source["selected_q14"]))
        branches = [dict(proposal, id=f"branch-b{i}", proposal_id="branch", native_prepared=True,
                         command_values=proposal["values"], selected_q10=list(source["selected_q10"]),
                         selected_q14=[.3 + .1 * i] + [.2] * 5) for i in range(2)]
        record = dict(proposal_id="branch", proposal_sha256=self.audit.digest(proposal),
                      admitted=2, admitted_request_ids=[row["id"] for row in branches])
        return dict(proposals=[proposal], admitted=branches, records=[record]), {source["id"]: source}

    def audit_branches(self, artifact, originals, proof=None):
        with patch.object(self.audit, "_proof", side_effect=proof or (lambda *args: dict(targets=[1, 2, 3]))):
            return self.audit.audit_branches(artifact, self.layout, originals, Path("unused"), {}, {}, {}, self.checks())

    def test_multiple_distinct_board_branches_with_fixed_dropper_are_valid(self):
        artifact, originals = self.branch_fixture()
        self.assertEqual(self.audit_branches(artifact, originals), artifact["admitted"])

    def test_branch_may_not_move_dropper_joints(self):
        artifact, originals = self.branch_fixture()
        artifact["admitted"][1]["selected_q10"][0] += .01
        with self.assertRaisesRegex(ValueError, "dropper joints unchanged"):
            self.audit_branches(artifact, originals)

    def test_branch_may_not_move_board_or_release_target(self):
        artifact, originals = self.branch_fixture()
        def proof(request, *_):
            return dict(targets=[1, 2, 3.1 if request["id"] == "branch-b1" else 3])
        with self.assertRaisesRegex(ValueError, "unchanged full board and release targets"):
            self.audit_branches(artifact, originals, proof)

    def test_duplicate_joint_branches_modulo_rotation_are_rejected(self):
        artifact, originals = self.branch_fixture()
        artifact["admitted"][1]["selected_q14"] = [value + math.tau for value in artifact["admitted"][0]["selected_q14"]]
        with self.assertRaisesRegex(ValueError, "distinct board configuration"):
            self.audit_branches(artifact, originals)

    def test_branch_count_cannot_hide_an_extra_admission(self):
        artifact, originals = self.branch_fixture()
        artifact["admitted"].append(dict(artifact["admitted"][0], id="hidden"))
        with self.assertRaisesRegex(ValueError, "complete ordered alternative admissions"):
            self.audit_branches(artifact, originals)

    def test_source_proposal_tampering_is_rejected(self):
        artifact, originals = self.branch_fixture()
        artifact["proposals"][0]["values"] = [.4, -.5, .3, 28., .4]
        with self.assertRaisesRegex(ValueError, "frozen branch source proposal"):
            self.audit_branches(artifact, originals)

    def test_path_proof_is_rehashed_and_verified_before_acceptance(self):
        with TemporaryDirectory() as folder:
            folder = Path(folder)
            anchor = dict(bucket_xy_m=self.layout["bucket_xy_m"])
            (folder / "anchor.json").write_text(json.dumps(anchor))
            request = self.request()
            proof = dict(passed=True, q10=request["selected_q10"], q14=request["selected_q14"],
                         heading_key="heading", board_heading_offset_deg=0., board_twist_deg=0.)
            (folder / "path.json").write_text(json.dumps(proof))
            request.update(context_resolution_file="anchor.json", context_resolution_sha256=self.audit.sha(folder / "anchor.json"),
                           prepared_connection_file="path.json", prepared_connection_sha256=self.audit.sha(folder / "path.json"),
                           heading_key="heading")
            with (patch.object(self.audit, "verify_resolution") as resolution,
                  patch.object(self.audit, "verify_connection") as connection):
                result = self.audit._proof(request, self.layout, folder, dict(records=[]), {}, {}, self.checks())
                self.assertTrue(result["passed"])
                resolution.assert_called_once()
                connection.assert_called_once()
                (folder / "path.json").write_text(json.dumps(dict(proof, passed=False)))
                with self.assertRaisesRegex(ValueError, "verified exact arm paths"):
                    self.audit._proof(request, self.layout, folder, dict(records=[]), {}, {}, self.checks())

    def test_interactive_inventory_is_frozen_not_relisted_at_audit_time(self):
        from sircus.mjlab import interactive_physics_check, intergrid_system_study

        with TemporaryDirectory() as folder:
            folder = Path(folder)
            system = folder / "system"
            system.mkdir()
            (system / "layouts.json").write_text(json.dumps(dict(
                train=dict(layouts=[dict(bucket_xy_m=[.3, .4])]), test=dict(layouts=[dict(bucket_xy_m=[.5, .6])]))))
            live = folder / "live"
            relative = "evidence/interactive-old/result.json"
            path = live / relative
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(dict(bucket_xy_m=[.7, .8])))
            inventory = [dict(file=relative, sha256=self.audit.sha(path), bucket_xy_m=[.7, .8])]
            later = live / "evidence/interactive-new/result.json"
            later.parent.mkdir(parents=True)
            later.write_text(json.dumps(dict(bucket_xy_m=[.9, 1.])))
            spec = dict(source="frozen", system_comparison=dict(digest="old"), interactive_history=inventory)
            args = SimpleNamespace(system_study=system, system_audit=folder / "audit.json", interactive_history=live)
            def source(_):
                return {}, [], [], [dict(bucket_xy_m=[.1, .2])], dict(source="frozen"), []
            with (patch.object(intergrid_system_study, "inputs", side_effect=source),
                  patch.object(interactive_physics_check, "historical_layouts", return_value=(None, None, [], dict(digest="old")))):
                result = self.audit.bound_inputs(args, spec, self.checks())
                self.assertEqual([row["bucket_xy_m"] for row in result[3]], [[.1, .2], [.3, .4], [.5, .6], [.7, .8]])
                path.write_text(json.dumps(dict(bucket_xy_m=[.71, .8])))
                with self.assertRaisesRegex(ValueError, "frozen interactive evidence"):
                    self.audit.bound_inputs(args, spec, self.checks())

    def test_interactive_inventory_uses_producer_path_order_across_both_folders(self):
        from sircus.mjlab import interactive_physics_check, intergrid_system_study

        with TemporaryDirectory() as folder:
            folder = Path(folder)
            (folder / "layouts.json").write_text(json.dumps(dict(train=dict(layouts=[]), test=dict(layouts=[]))))
            for name, xy in (("evidence/interactive-z/result.json", [.7, .8]),
                             ("evidence-acceptance/interactive-a/result.json", [.9, .6])):
                path = folder / name
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(dict(bucket_xy_m=xy)))
            paths = sorted({path for name in ("evidence", "evidence-acceptance")
                            for path in (folder / name).glob("interactive-*/result.json")})
            inventory = [dict(file=path.relative_to(folder).as_posix(), sha256=self.audit.sha(path),
                              bucket_xy_m=self.audit.read(path)["bucket_xy_m"]) for path in paths]
            self.assertEqual([item["file"].split("/")[0] for item in inventory], ["evidence", "evidence-acceptance"])
            spec = dict(system_comparison={}, interactive_history=inventory)
            args = SimpleNamespace(system_study=folder, system_audit=folder / "audit.json", interactive_history=folder)
            with (patch.object(intergrid_system_study, "inputs", side_effect=lambda _: ({}, [], [], [], {}, [])),
                  patch.object(interactive_physics_check, "historical_layouts", return_value=(None, None, [], {}))):
                result = self.audit.bound_inputs(args, spec, self.checks())
                self.assertEqual([row["bucket_xy_m"] for row in result[3]], [[.7, .8], [.9, .6]])
                spec["interactive_history"] = list(reversed(inventory))
                with self.assertRaisesRegex(ValueError, "Sorted unique frozen interactive files"):
                    self.audit.bound_inputs(args, spec, self.checks())


if __name__ == "__main__":
    unittest.main()
