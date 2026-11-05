"""Synthetic CPU guard fixtures only; no fixture is native study evidence."""
import copy
import unittest
from unittest.mock import patch
from pathlib import Path
import json
import tempfile
import hashlib

import numpy as np

from sircus.mjlab import clean650_verify as verify
from sircus.mjlab.benchmark_verify import Checks, EvidenceError
from sircus.mjlab.tests.test_replay650_verify import group_fixture


def row(ident, **change):
    value = dict(id=ident, native_release=True, invalid=None, invalid_physics=False,
                 scalar_score_audit_passed=True, route_success=True, route_quality=2,
                 whole_ball_entry=True, external_contact_steps=0, max_retention_s=1.,
                 minimum_mouth_clearance_mm=8., closest_opening_distance_m=.01,
                 values=[.47, -.27, .38, 28., .3868], bucket_xy_m=[.1, -.6],
                 q10=[0.]*6, q14=[0.]*6)
    value.update(change)
    return value


def controls():
    return [row(f"{step}-{role}-{i}", substeps=step, role=role,
                route_success=role == "positive_control", route_quality=2 if role == "positive_control" else 0)
            for step in (64, 128) for role in ("positive_control", "negative_control") for i in range(3)]


def verification_rows():
    return [row(f"{step}-{i}", substeps=step, pair_id=f"p{i}", group_id="verify-00", role="verification", repeat_index=i)
            for step in (64, 128) for i in range(3)]


class CleanAuditTests(unittest.TestCase):
    def test_explicit_fine_profile_preserves_original_coarse_literals(self):
        coarse = verify.profile_for()
        fine = verify.profile_for("sircus-clean650-height-tilt-gap-fine-v2")
        self.assertEqual((coarse["coarse_requests"], coarse["max_requests"], coarse["audit_version"]),
                         (361, 496, "sircus-clean650-independent-audit-v1"))
        self.assertEqual((fine["coarse_requests"], fine["max_requests"], fine["audit_version"]),
                         (76, 211, "sircus-clean650-independent-audit-v2"))
        fine["heights"] = (1.,)
        self.assertEqual(verify.profile_for("sircus-clean650-height-tilt-gap-fine-v2")["heights"], (.375, .38, .385))
        with self.assertRaisesRegex(EvidenceError, "Unknown"):
            verify.profile_for("unversioned-fine-override")

    def test_fine_grid_is75_plus_reference_and_narrow_refinement_offsets(self):
        from sircus.mjlab import clean650_contract as protocol
        seed = row("source", source_record_id="historical-source", source_record_sha256="a"*64)
        fine = verify.profile_for("sircus-clean650-height-tilt-gap-fine-v2")
        values = verify.coarse_values(seed, fine)
        self.assertEqual(len(values), 76)
        self.assertEqual(len({tuple(v) for v in values}), 75)
        self.assertEqual(values[0], [.47, -.27, .375, 27., .38])
        self.assertEqual(values[74], [.47, -.27, .385, 29., .40])
        self.assertEqual(values[-1], seed["values"])
        self.assertEqual(values, [g["values"] for g in protocol.coarse(seed, "fine-v2")])
        local = verify.local_values(seed, fine)
        self.assertEqual(local[0], [.47, -.27, .375, 27.75, .3843])
        self.assertEqual(local[-1], [.47, -.27, .385, 28.25, .3893])
        np.testing.assert_allclose(local, [g["values"] for g in protocol.refinement([seed], "fine-v2")], rtol=0, atol=1e-12)

    def test_fine_settings_cannot_relabel_a_coarse_grid_or_change_the_budget(self):
        from sircus.mjlab import clean650_contract as protocol
        fine = verify.profile_for("sircus-clean650-height-tilt-gap-fine-v2")
        verify.audit_settings(protocol.settings("fine-v2"), Checks(), fine)
        verify.audit_settings(protocol.settings(), Checks())
        with self.assertRaisesRegex(EvidenceError, "protocol/grid"):
            verify.audit_settings(protocol.settings(), Checks(), fine)
        changed = protocol.settings("fine-v2");changed["max_requests"] += 1
        with self.assertRaisesRegex(EvidenceError, "max_requests"):
            verify.audit_settings(changed, Checks(), fine)

    def test_independent_accounting_matches_producer_without_using_producer_selection(self):
        from sircus.mjlab import clean650_contract as protocol
        data = controls()
        data[0].update(native_release=False, invalid="strict_endpoint_rejected")
        self.assertEqual(verify.statistics(data), protocol.statistics(data))
        self.assertEqual(verify.control_gate(controls()), protocol.control_gate(controls()))
        chosen = [row("b"), row("a", minimum_mouth_clearance_mm=12.), row("reject", native_release=False, invalid="endpoint")]
        self.assertEqual([r["id"] for r in verify.select(chosen, 3)], [r["id"] for r in protocol.ranked(chosen, 3)])

    def test_independent_qualification_matches_full_native_position_protocol(self):
        from sircus.mjlab import clean650_contract as protocol
        data = verification_rows()
        full = {r["id"]: np.zeros((round(.350*240*r["substeps"])+1, 3)) for r in data}
        positions = {r["id"]: full[r["id"]][-1] for r in data}
        self.assertEqual(verify.qualify(data, positions), protocol.qualification(data, full, controls_passed=True, source_clean=True))
        self.assertEqual(verify.qualify(data, positions, source_clean=False), protocol.qualification(data, full, controls_passed=True, source_clean=False))

    def test_grid_has360unique_grid_values_plus_exact_reference(self):
        seed = row("seed")
        grid = verify.coarse_values(seed)
        self.assertEqual(len(grid), 361)
        self.assertEqual(len({tuple(v) for v in grid}), 361)
        self.assertEqual(grid[0], [.47, -.27, .22, 16., .20])
        self.assertEqual(grid[359], [.47, -.27, .42, 36., .56])
        self.assertEqual(grid[-1], seed["values"])
        grid[-1][0] += 1
        self.assertEqual(seed["values"][0], .47)

    def test_quality_selection_never_promotes_invalid_or_unreleased_rows(self):
        rows = [row("near", route_success=False, route_quality=0, minimum_mouth_clearance_mm=100.),
                row("rim", route_quality=1, minimum_mouth_clearance_mm=100.), row("clean"),
                row("invalid", invalid_physics=True), row("rejected", native_release=False, invalid="endpoint")]
        self.assertEqual([r["id"] for r in verify.select(rows, 5)], ["clean", "rim", "near"])

    def test_missing_mouth_margin_is_last_and_equal_rows_use_id(self):
        rows = [row("z", minimum_mouth_clearance_mm=None), row("b"), row("a")]
        self.assertEqual([r["id"] for r in verify.select(rows, 3)], ["a", "b", "z"])

    def test_unique_verification_targets_do_not_round_or_replace_executed_actions(self):
        first = row("a", minimum_mouth_clearance_mm=10.)
        duplicate = row("b", minimum_mouth_clearance_mm=9.)
        duplicate["values"][2] += 1e-12
        distinct = row("c", minimum_mouth_clearance_mm=8.)
        distinct["values"][2] += .001
        actual = verify.select_unique([first, duplicate, distinct], 5)
        self.assertEqual([r["id"] for r in actual], ["a", "c"])
        self.assertIs(actual[0], first)
        self.assertIs(actual[1], distinct)
        self.assertEqual(duplicate["values"][2], .38+1e-12)

    def test_local_refinement_is27_declared_offsets_without_xy_changes(self):
        source = row("source")
        values = verify.local_values(source)
        self.assertEqual(len(values), 27)
        self.assertEqual(values[13], source["values"])
        self.assertEqual(values[0], [.47, -.27, .36, 26., .3668])
        self.assertEqual(values[-1], [.47, -.27, .40, 30., .4068])

    def test_refinement_and_verification_are_bound_to_exact_previous_records(self):
        from sircus.mjlab import clean650_contract as protocol
        source = row("earlier-source")
        groups = protocol.refinement([source])
        verify.audit_refinement(groups, [source], Checks())
        groups[0]["values"][0] += .001
        with self.assertRaisesRegex(EvidenceError, "literal local"):
            verify.audit_refinement(groups, [source], Checks())
        groups = protocol.verification([source])
        verify.audit_verification(groups, [source], Checks())
        groups[0]["q14"][1] = .00001
        with self.assertRaisesRegex(EvidenceError, "exact source field"):
            verify.audit_verification(groups, [source], Checks())

    def test_adaptive_stage_binds_only_the_actual_earlier_record_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            study = Path(directory)
            before = [row("coarse-a"), row("coarse-b")]
            lines = [json.dumps(r).encode()+b"\n" for r in before+[row("future")]]
            (study/"records.jsonl").write_bytes(b"".join(lines))
            for name, value in (("refinement-selection.json", {"source_ids": []}),
                                ("refinement-plan.json", []), ("refinement-groups.json", [])):
                (study/name).write_text(json.dumps(value))
            proofs = [{}]*363
            stage = dict(phase="refinement", prior_records_sha256=hashlib.sha256(b"".join(lines[:2])).hexdigest(),
                         files_sha256={name:verify.sha(study/name) for name in
                                       ("refinement-selection.json", "refinement-plan.json", "refinement-groups.json")},
                         geometry_sha256=verify.digest(proofs))
            verify.audit_adaptive_stage(study, [stage], 0, "refinement", [], [], [], before, proofs, Checks())
            lines[0] = json.dumps(row("changed-earlier")).encode()+b"\n"
            (study/"records.jsonl").write_bytes(b"".join(lines))
            with self.assertRaisesRegex(EvidenceError, "earlier observations"):
                verify.audit_adaptive_stage(study, [stage], 0, "refinement", [], [], [], before, proofs, Checks())

    def test_positive_must_be_clean_at_both_steps_and_negative_actual_valid_miss(self):
        data = controls()
        self.assertEqual(verify.control_gate(data)["status"], "passed")
        self.assertEqual(verify.control_gate(data[:-1])["status"], "failed")
        data[0]["route_quality"] = 1
        self.assertFalse(verify.control_gate(data)["positive_control"])
        data = controls();data[-1].update(native_release=False, invalid="endpoint")
        self.assertFalse(verify.control_gate(data)["negative_control"])

    def test_qualification_requires_clean_source_controls_and_fine_step(self):
        data = verification_rows();positions = {r["id"]: [0., 0., .38] for r in data}
        self.assertTrue(verify.qualify(data, positions)["qualified"])
        self.assertFalse(verify.qualify(data, positions, source_clean=False)["qualified"])
        self.assertFalse(verify.qualify(data, positions, controls_passed=False)["qualified"])
        data[-1]["route_quality"] = 1
        self.assertFalse(verify.qualify(data, positions)["qualified"])
        data[-1]["route_quality"] = 2
        positions[data[-1]["id"]] = [.00500001, 0., .38]
        self.assertFalse(verify.qualify(data, positions)["qualified"])

    def test_same_frozen_joints_required_for_matched_verification(self):
        data = verification_rows()
        self.assertEqual(len(verify.replay.check_pairing(data, Checks())), 3)
        data[-1]["q14"][0] = 1e-10
        with self.assertRaisesRegex(EvidenceError, "same frozen input"):
            verify.replay.check_pairing(data, Checks())

    def test_opened_plan_retains_rejects_and_unique_repeat_ids(self):
        _, _, groups, _ = group_fixture()
        groups = groups[:2];groups[-1]["eligible"] = False
        plan = verify.make_plan(groups, "controls_pre", steps=(64, 128), repeats=3)
        self.assertEqual(len(plan), 12)
        self.assertEqual(sum(not p["eligible"] for p in plan), 6)
        self.assertEqual(len({p["id"] for p in plan}), 12)
        self.assertEqual(len({p["pair_id"] for p in plan}), 6)

    def test_geometry_rechecks_every_recorded_attempt_and_clearance(self):
        _, _, groups, proofs = group_fixture();groups, proofs = groups[:1], proofs[:1]
        scene = json.loads((Path(__file__).resolve().parents[3]/"sircus/isaac/scene.json").read_text())
        scene["table"] = dict(center_m=[0., 0., 0.], size_m=[4., 4., .1])
        class Gate:
            calls = 0
            def __init__(self, *args): pass
            def check(self, *args):
                self.__class__.calls += 1
                return dict(reason=0, minimum_clearance_lower_bound_m=.006)
        with patch("sircus.mjlab.corrected_geometry.StrictEndpointGate", Gate):
            verify.audit_geometry(groups, proofs, scene, Path("unused"), Checks())
        self.assertEqual(Gate.calls, 1)
        proofs[0]["minimum_clearance_lower_bound_m"] = .0049
        groups[0]["geometry_sha256"] = verify.digest(proofs[0])
        with self.assertRaisesRegex(EvidenceError, "admitted clearance"):
            verify.audit_geometry(groups, proofs, scene, Path("unused"), Checks())


if __name__ == "__main__":
    unittest.main()
