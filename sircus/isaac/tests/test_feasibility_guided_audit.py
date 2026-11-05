"""Synthetic corruption fixtures for independent guided evidence checks."""

import copy
import json
import math
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.stats import qmc

ISAAC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ISAAC))
import feasibility_guided_audit as audit
from continuous_feasibility import ACTION_NAMES, LOW, HIGH, canonical_hash, decode_action
from expanded_farm import make_spec


def seal(value):
    value.pop("fingerprint", None)
    value["fingerprint"] = canonical_hash(value)
    return value


def fixture():
    scene = json.loads((ISAAC / "scene.json").read_text())
    spec = make_spec(scene, 10)
    half = np.array([math.radians(5), .025, .025, .060, math.radians(6)])
    records, rows = [], []
    home = {n: [0.]*6 for n in ("Arm10", "Arm14")}
    for i, cell in enumerate(spec["cells"]):
        action = decode_action(cell, [0., .4, .35, .24, math.radians(29)], f"anchor-{i}")
        ik = {n: dict(q=[0.]*6, solved=True) for n in home}
        row = dict(cell_id=cell["id"], cell_index=i, bucket_xy_m=cell["bucket_xy_m"],
                   map_candidates=0, anchor_checks=[], status="unresolved", anchor_action=None,
                   anchor_ik=None, low=LOW.tolist(), high=HIGH.tolist(), calibration=[])
        if i:
            values = np.array([action[k] for k in ACTION_NAMES])
            record = dict(cell_id=cell["id"], accepted=True, action=action, ik=ik, proposal=0)
            records.append(record)
            distance = np.linalg.norm(np.array([.67, -.10])-cell["bucket_xy_m"])
            prior = [0., np.clip(distance, .29, 1.19), .35, .24, math.radians(29)]
            score = float(np.square((values-prior)/(HIGH-LOW)).sum())
            lo, hi = np.maximum(LOW, values-half), np.minimum(HIGH, values+half)
            unit = qmc.Sobol(d=5, scramble=True, seed=20261113+i).random_base2(4)
            probes = []
            for j, request in enumerate(lo+unit*(hi-lo)):
                picked = decode_action(cell, request, f"support-{i:02d}-1.0-{j:02d}")
                probes.append(dict(probe=j, action=picked, qualification=dict(action=picked, accepted=True),
                                   path=dict(accepted=True), accepted=True))
            calibration = dict(scale=1., low=lo.tolist(), high=hi.tolist(), requests=16,
                               accepted=16, acceptance=1., records=probes)
            row.update(status="supported", map_candidates=1,
                       anchor_checks=[dict(proposal=0, action_id=action["id"], score=score, path=dict(accepted=True))],
                       anchor_proposal=0, anchor_action=action, anchor_ik=ik, anchor_score=score,
                       low=lo.tolist(), high=hi.tolist(), calibration=[calibration],
                       selected_width_scale=1., probe_acceptance=1., qualification_target_met=True)
        rows.append(row)
    feasibility = seal(dict(source_fingerprint=spec["fingerprint"], ik_protocol="full_robust_v2",
                            asset_sha256={"poses.json": "a"*64}, records=records))
    support = seal(dict(schema="sircus-feasibility-guided-support-v1", policy_geometry_only=True,
                        source_fingerprint=spec["fingerprint"], feasibility_fingerprint=feasibility["fingerprint"],
                        asset_sha256=feasibility["asset_sha256"], canonical_home_joint_rad=home,
                        action_names=list(ACTION_NAMES), global_low=LOW.tolist(), global_high=HIGH.tolist(),
                        base_half_width=half.tolist(), width_scales=[1., .5, .25, .125],
                        probe_count=16, probe_seed=20261113, minimum_probe_acceptance=.75,
                        anchor_path_samples=16, cells=rows, supported_cells=34,
                        probe_requests=34*16, probe_accepted=34*16))
    return support, feasibility, spec


class GuidedSupportAuditTests(unittest.TestCase):
    def setUp(self):
        self.support, self.map, self.spec = fixture()

    def verify(self):
        return audit.validate_support_evidence(self.support, self.map, self.spec)

    def test_all_geometry_probes_and_unresolved_cell_are_retained(self):
        result = self.verify()
        self.assertEqual(result["supported_cells"], 34)
        self.assertEqual(result["unresolved_cells"], 1)
        self.assertEqual(result["probe_requests"], 544)

    def test_resigned_anchor_ranking_change_is_rejected(self):
        self.support["cells"][1]["anchor_checks"][0]["score"] += .1
        seal(self.support)
        with self.assertRaisesRegex(ValueError, "anchor rank"):
            self.verify()

    def test_resigned_probe_action_or_denominator_change_is_rejected(self):
        self.support["cells"][1]["calibration"][0]["records"][0]["action"]["height_m"] += .01
        seal(self.support)
        with self.assertRaisesRegex(ValueError, "probe request"):
            self.verify()

    def test_resigned_empty_support_success_is_rejected(self):
        self.support["cells"][0]["status"] = "supported"
        seal(self.support)
        with self.assertRaisesRegex(ValueError, "Empty geometry support"):
            self.verify()

    def test_resigned_local_width_change_is_rejected(self):
        self.support["cells"][1]["high"][3] += .01
        seal(self.support)
        with self.assertRaisesRegex(ValueError, "selected local support"):
            self.verify()

    def test_probe_failure_cannot_be_claimed_accepted(self):
        self.support["cells"][1]["calibration"][0]["records"][0]["qualification"]["accepted"] = False
        seal(self.support)
        with self.assertRaisesRegex(ValueError, "probe request"):
            self.verify()

    def test_changed_source_or_poses_hash_is_rejected(self):
        self.support["asset_sha256"] = {"poses.json": "b"*64}
        seal(self.support)
        with self.assertRaisesRegex(ValueError, "binding changed"):
            self.verify()

    def test_decode_and_physical_density_offsets_preserve_ratios(self):
        n = len(self.spec["cells"])
        normalized = np.zeros((n, 5))
        actual = audit.independent_physical_actions(normalized, self.support, self.spec["cells"])
        for i, row in enumerate(self.support["cells"]):
            np.testing.assert_allclose(actual[i], (np.array(row["low"])+row["high"])/2, atol=2e-15, rtol=0)
        jac = audit.physical_log_density_offset(self.support, self.spec["cells"])
        old = np.linspace(-.3, .7, n)
        new = np.linspace(.5, -.4, n)
        np.testing.assert_allclose(np.exp((new+jac)-(old+jac)), np.exp(new-old), rtol=1e-13)
        self.assertNotEqual(jac[0], jac[1])

    def test_rehashed_guided_study_cannot_change_budget_or_drop_support_binding(self):
        from feasibility_guided_pilot import make_study
        study = make_study(dict(spec=self.spec, source=dict(camera_sha256="a"*64)), 4, self.support)
        study.update(support_sha256="b"*64, feasibility_sha256="c"*64, coverage_batches=4)
        seal(study)
        audit.validate_guided_study(study, self.spec, self.support, "b"*64, "c"*64)
        for key, value in (("seed", 1), ("training_batches", 8), ("support_sha256", "d"*64)):
            changed = copy.deepcopy(study)
            changed[key] = value
            seal(changed)
            with self.subTest(field=key), self.assertRaises(ValueError):
                audit.validate_guided_study(changed, self.spec, self.support, "b"*64, "c"*64)


if __name__ == "__main__":
    unittest.main()
