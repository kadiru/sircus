"""Independent geometry audit tests with synthetic exactly solvable chains."""

import copy
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))

import continuous_geometry_audit as audit
from continuous_feasibility import ACTION_NAMES, HIGH, LOW, canonical_hash, decode_action
from expanded_pose_menu import endpoint_targets
import joint_feasibility


def sign(value):
    result = copy.deepcopy(value)
    result.pop("fingerprint", None)
    result["fingerprint"] = canonical_hash(result)
    return result


class GeometryAuditTests(unittest.TestCase):
    def setUp(self):
        self.scene = json.loads((SOURCE / "scene.json").read_text())
        self.cell = dict(id="toy-train", bucket_xy_m=[.27, -.10], split="train")
        self.action = decode_action(self.cell, [0., .4, .35, .24, math.radians(29)], "exact")
        targets = endpoint_targets(self.scene, self.cell, self.action)
        self.links = {arm: [(targets[arm].copy() if i == 0 else np.eye(4),
                             np.array([0., 0., 1.]), -2*math.pi, 2*math.pi)
                            for i in range(6)] for arm in ("Arm10", "Arm14")}
        self.starts = {arm: [0.]*6 for arm in ("Arm10", "Arm14")}
        self.bad_action = decode_action(self.cell, [0., .6, .35, .24, math.radians(29)], "exact")

    def solve(self, *args):
        return dict(q=[0.]*6, solved=True, residual=0., position_error_mm=0., orientation_error_deg=0.)

    def decision(self, action):
        return joint_feasibility.preflight(self.scene, self.cell, action, self.links, self.starts,
                                           solver=self.solve, solver_protocol="synthetic-exact-FK")

    def result(self, classification="unchanged"):
        original = self.action if classification == "unchanged" else self.bad_action
        first = self.decision(original)
        final = self.decision(self.action) if classification == "adjusted" else (
            first if classification == "unchanged" else None)
        chosen = self.action if classification != "rejected" else None
        delta = np.array([chosen[n]-original[n] for n in ACTION_NAMES]) if chosen is not None else np.zeros(5)
        source = SOURCE / "continuous_geometry_layer.py"
        protocol = dict(name="bounded-continuous-coupled-least-squares-v1", enabled=True,
                        max_nfev=24, max_residual_evaluations=512, max_normalized_adjustment=.25,
                        distance="L2 of five action deltas divided by original HIGH-LOW",
                        local_bound="Per-component absolute normalized displacement",
                        optimizer="scipy.optimize.least_squares; one deterministic start",
                        pose_residual_weight=10000., action_distance_weight=1.,
                        clearance_residual_weight=1000.,
                        admission="Only unchanged joint_feasibility preflight plus independent validation")
        repair = dict(attempted=classification != "unchanged", candidate=self.action if classification == "adjusted" else None,
                      candidate_preflight=final if classification == "adjusted" else None,
                      ik_seeds_rad=self.starts if classification == "adjusted" else None,
                      residual_evaluations=1 if classification != "unchanged" else 0)
        return sign(dict(schema="sircus-continuous-geometry-layer-v1", classification=classification,
                         accepted=classification != "rejected", original_action=original,
                         action=chosen, original_preflight=first, final_preflight=final,
                         repair=repair, original_reason=first["reason"], action_names=list(ACTION_NAMES),
                         delta_physical=None if chosen is None else delta.tolist(),
                         delta_normalized=None if chosen is None else (delta/(HIGH-LOW)).tolist(),
                         normalized_adjustment_l2=None if chosen is None else float(np.linalg.norm(delta/(HIGH-LOW))),
                         reason="no_verified_repair_within_bounded_work" if chosen is None else None,
                         binding=dict(original_preflight_fingerprint=first["fingerprint"], protocol=protocol,
                                      source_sha256={source.name:audit.hashlib.sha256(source.read_bytes()).hexdigest()}),
                         preflight_calls=2 if classification == "adjusted" else 1,
                         training_updates=0, native_shots=0, bucket_impossibility_proven=False,
                         release_authorized=False))

    def check(self, value, action=None):
        return audit.audit_result(value, self.scene, self.cell,
                                  action or value["original_action"], self.links, self.starts,
                                  seeds=self.starts)

    def population(self):
        rows, cases = [], []
        for index, classification in enumerate(audit.CLASSIFICATIONS):
            result = self.result(classification)
            identifier = "synthetic-" + str(index)
            clean = index == 0
            cases.append(dict(case_id=identifier, cell=self.cell, action=result["original_action"],
                              starts=self.starts, seeds=self.starts, clean=clean))
            rows.append(dict(case_id=identifier, cell_id=self.cell["id"], clean=clean, result=result))
        inputs = dict(synthetic=True, expected_counts=dict(requested=3, clean=1), scene=self.scene,
                      links=self.links, cases=cases, bindings=dict(source="synthetic-only"),
                      protected_before={"synthetic-input": "0"*64})
        report = sign(dict(schema="sircus-continuous-geometry-train-retention-v1", seed=20261122,
                           rows=rows, counts=dict(requested=3, unchanged=1, adjusted=1, rejected=1),
                           clean_counts=dict(requested=1, unchanged=1, adjusted=0, rejected=0),
                           reasons={"outer_link_length_bound": 2}, bindings=inputs["bindings"],
                           protected_after=inputs["protected_before"], optimizer_updates=0,
                           new_native_shots=0, release_authorized=False, synthetic=True,
                           candidate_bank_read=False, model_loaded=False, held_out_shot_files_read=0))
        return report, inputs

    def test_all_three_classifications_independently_verify(self):
        for classification in audit.CLASSIFICATIONS:
            with self.subTest(classification=classification):
                self.assertEqual(self.check(self.result(classification)), classification)

    def test_signed_result_and_preflight_fingerprints_are_mandatory(self):
        value = self.result()
        value["classification"] = "adjusted"
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            self.check(value)
        value = self.result()
        value["original_preflight"]["reason"] = "fabricated"
        value = sign(value)
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            self.check(value)

    def test_resigned_lies_about_unchanged_or_output_are_rejected(self):
        for modify in (lambda x: x.update(action=self.bad_action),
                       lambda x: x.update(accepted=False),
                       lambda x: x["repair"].update(attempted=True),
                       lambda x: x.update(release_authorized=True)):
            value = self.result(); modify(value)
            with self.subTest(value=value["classification"]), self.assertRaises(ValueError):
                self.check(sign(value))

    def test_changed_starts_or_seeds_invalidate_both_arm_binding(self):
        value = self.result()
        changed = copy.deepcopy(self.starts); changed["Arm14"][0] = .01
        with self.assertRaisesRegex(ValueError, "Stale"):
            audit.audit_result(value, self.scene, self.cell, self.action, self.links, changed,
                               seeds=self.starts)
        with self.assertRaisesRegex(ValueError, "Stale"):
            audit.audit_result(value, self.scene, self.cell, self.action, self.links, self.starts,
                               seeds=changed)

    def test_rejection_cause_is_rederived_even_after_resigning(self):
        value = self.result("rejected")
        decision = value["original_preflight"]
        decision["reason"] = "board_table_clearance"
        decision["qualification"]["reason"] = "board_table_clearance"
        value["original_preflight"] = sign(decision)
        value["binding"]["original_preflight_fingerprint"] = value["original_preflight"]["fingerprint"]
        value["original_reason"] = "board_table_clearance"
        with self.assertRaisesRegex(ValueError, "cause"):
            self.check(sign(value))

    def test_adjustment_needs_final_geometry_and_exact_displacement(self):
        value = self.result("adjusted")
        value["delta_physical"][0] = 99
        with self.assertRaisesRegex(ValueError, "displacement"):
            self.check(sign(value))
        value = self.result("adjusted")
        value["final_preflight"] = value["original_preflight"]
        with self.assertRaises(ValueError):
            self.check(sign(value))
        value = self.result("rejected")
        value["action"] = self.action
        with self.assertRaisesRegex(ValueError, "fabricated"):
            self.check(sign(value))

    def test_synthetic_population_never_receives_real_pass(self):
        report, inputs = self.population()
        result = audit.audit_analysis(report, inputs, verify_files=False)
        self.assertEqual(result["status"], "passed_synthetic")
        self.assertEqual(result["counts"], dict(requested=3, unchanged=1, adjusted=1, rejected=1))
        self.assertFalse(result["capture_replayed"])
        self.assertFalse(result["release_authorized"])
        inputs.pop("synthetic")
        with self.assertRaises(ValueError):
            audit.audit_analysis(report, inputs, verify_files=False)

    def test_coverage_order_clean_labels_and_summaries_cannot_be_relabelled(self):
        changes = [lambda x: x["rows"].pop(), lambda x: x["rows"].reverse(),
                   lambda x: x["rows"][0].update(clean=False),
                   lambda x: x["counts"].update(unchanged=2),
                   lambda x: x["clean_counts"].update(requested=3),
                   lambda x: x.update(reasons={}),
                   lambda x: x.update(optimizer_updates=1),
                   lambda x: x.update(new_native_shots=1),
                   lambda x: x.update(candidate_bank_read=True),
                   lambda x: x.update(model_loaded=True),
                   lambda x: x.update(held_out_shot_files_read=1),
                   lambda x: x.update(synthetic=False),
                   lambda x: x.update(bindings={})]
        for change in changes:
            report, inputs = self.population(); change(report)
            with self.subTest(change=change), self.assertRaises(ValueError):
                audit.audit_analysis(sign(report), inputs, verify_files=False)

    def test_current_protected_file_bytes_are_rehashed(self):
        report, inputs = self.population()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "original-checkpoint.bin"
            path.write_bytes(b"synthetic bytes only")
            hashes = {str(path): audit.hashlib.sha256(path.read_bytes()).hexdigest()}
            inputs["protected_before"] = hashes
            report["protected_after"] = dict(hashes)
            audit.audit_analysis(sign(report), inputs)
            path.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Protected input bytes changed"):
                audit.audit_analysis(sign(report), inputs)

    def test_failed_repair_still_requires_valid_continuous_geometry_and_null_delta(self):
        value = self.result("rejected")
        value["delta_physical"] = [0.]*5
        value["delta_normalized"] = [0.]*5
        with self.assertRaisesRegex(ValueError, "displacement"):
            self.check(sign(value))
        value = self.result("rejected")
        value["repair"]["candidate"] = dict(self.action, board_x_m=99.)
        with self.assertRaisesRegex(ValueError, "derived geometry"):
            self.check(sign(value))

    def test_repair_budget_protocol_and_source_cannot_be_relabelled(self):
        mutations = [lambda x: x["binding"]["protocol"].update(max_nfev=257),
                     lambda x: x["binding"]["protocol"].update(pose_residual_weight=0.),
                     lambda x: x["repair"].update(residual_evaluations=513),
                     lambda x: x["binding"].update(source_sha256={}),
                     lambda x: x.update(preflight_calls=999)]
        for mutate in mutations:
            value = self.result("rejected"); mutate(value)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                self.check(sign(value))


if __name__ == "__main__":
    unittest.main()
