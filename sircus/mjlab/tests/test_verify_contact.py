"""Evidence corruption tests: altered inputs, missing failures and false counts."""

import copy
import json
from pathlib import Path
import unittest

import numpy as np

from sircus.mjlab.verify_contact import Audit, audit_records, digest, family_stats, replay_scores, SOURCE_ID, STUDY_VERSION
from sircus.mjlab.contract import ScoreStream, FORCE_NAMES


def fixture():
    source = dict(id=SOURCE_ID, values=[.47, -.28, .4, 36., .3], q10=[0.] * 6,
        q14=[1.] * 6, bucket_xy_m=[.16, -.64])
    historical = [dict(source, id=f"prior-{i}", phase="final_confirmation", confirmation_source_id=SOURCE_ID,
        route_success=i < 13, values=[.47, -.28, .4, 36., .3 + i * .00001]) for i in range(20)]
    plan = [dict(source, id=f"nominal-{i:02d}", family="nominal") for i in range(2)]
    plan += [dict(r, family="original_final_development", historical_success=r["route_success"]) for r in historical]
    rng = np.random.default_rng(20261124)
    plan += [dict(source, id=f"fresh-{i:02d}", family="fresh_perturbation",
        values=(np.array(source["values"]) + rng.normal(0, [.001, .001, .002, .35, .002])).tolist()) for i in range(3)]
    for p in plan:
        p.update(geometry_eligible=True, geometry_reason=0)
    variants = [dict(id=f"{m}-{g}-{s}", board_mode=m, gravity_mode=g, substeps=s)
        for m, g in (("articulated", "legacy"), ("articulated", "passive_only"), ("kinematic", "passive_only"))
        for s in (8, 16, 32)]
    spec = dict(version=STUDY_VERSION, source=source, source_id=SOURCE_ID, seed=20261124,
        noise_std=[.001, .001, .002, .35, .002], plan=plan, variants=variants, reset_repeats=2)
    spec["fingerprint"] = digest(spec)
    rows = []
    for v in variants:
        for reset in (0, 1):
            for p in plan:
                rows.append(dict(p, id=f'{v["id"]}-reset-{reset}-{p["id"]}', source_plan_id=p["id"],
                    variant=v["id"], reset=reset,
                    contact_protocol={k: v[k] for k in ("board_mode", "gravity_mode", "substeps")},
                    native_release=True, scalar_score_audit_passed=True, invalid=None,
                    route_success=True, complete_success=True, success=True, board_bounce=True,
                    whole_ball_entry=True, max_retention_s=1., external_contact_steps=0,
                    route_quality=2, quality_score=2, scoring_version="sircus-isaac-fixed-shot-v5",
                    route_version="sircus-board-route-v1", max_integrated_joint_excursion_rad=0.,
                    peak_integrated_joint_velocity_rad_s=0.))
    groups = []
    for v in variants:
        selected = [r for r in rows if r["variant"] == v["id"]]
        groups.append(dict(v, attempts=len(selected), captures=len(selected),
            family_results=[dict(family=f, **family_stats([r for r in selected if r["family"] == f]))
                for f in ("nominal", "original_final_development", "fresh_perturbation")]))
    report = dict(version=STUDY_VERSION, status="completed", source=source, source_id=SOURCE_ID,
        contract_fingerprint=spec["fingerprint"], groups=groups, records=len(rows))
    return spec, report, rows, [source, *historical]


class ContactVerificationTests(unittest.TestCase):
    def result(self, f):
        return {r["name"]: r["passed"] for r in audit_records(*f).checks}

    def test_legitimate_factorial_passes(self):
        results = self.result(fixture())
        self.assertTrue(all(results.values()), results)

    def test_omitted_outcome_cannot_hide_in_revised_total(self):
        f = fixture(); f[2].pop(); f[1]["records"] -= 1
        self.assertFalse(self.result(f)["no_missing_duplicate_or_extra_outcomes"])

    def test_duplicate_instead_of_failed_outcome_rejected(self):
        f = fixture(); f[2][-1] = copy.deepcopy(f[2][0])
        self.assertFalse(self.result(f)["no_missing_duplicate_or_extra_outcomes"])

    def test_branch_substitution_is_rejected(self):
        f = fixture(); f[2][0]["q14"] = [1.01] * 6
        self.assertFalse(self.result(f)["native_requests_and_cached_joint_identity"])

    def test_changed_prior_twenty_not_relabelled(self):
        f = fixture(); p = next(p for p in f[0]["plan"] if p["family"] == "original_final_development")
        p["historical_success"] = False
        self.assertFalse(self.result(f)["all_original_twenty_preserved"])

    def test_changed_fresh_sample_fails_even_with_updated_fingerprint(self):
        f = fixture(); p = next(p for p in f[0]["plan"] if p["family"] == "fresh_perturbation")
        p["values"][2] += .001
        payload = dict(f[0]); payload.pop("fingerprint")
        f[0]["fingerprint"] = f[1]["contract_fingerprint"] = digest(payload)
        self.assertFalse(self.result(f)["independently_regenerated_fresh_requests"])

    def test_lying_group_count_fails(self):
        f = fixture(); f[1]["groups"][0]["family_results"][0]["captures"] -= 1
        self.assertFalse(self.result(f)["articulated-legacy-8_nominal_statistics"])

    def test_invalid_release_and_partial_capture_rejected(self):
        f = fixture(); f[2][0]["invalid"] = "endpoint_collision"
        self.assertFalse(self.result(f)["native_release_and_scalar_audit_accounting"])
        f = fixture(); f[2][0]["max_retention_s"] = .9
        self.assertFalse(self.result(f)["unchanged_complete_capture_definition"])

    def test_raw_trajectory_change_is_detected_by_independent_rescore(self):
        scene = json.loads((Path(__file__).resolve().parents[2] / "isaac/scene.json").read_text())
        normal = [0., 0., 1.]; bucket = [.16, -.64]
        position = np.tile([.5, -.2, -.3], (960, 1, 1))
        velocity = np.zeros((960, 1, 3)); forces = np.zeros((960, 1, 6))
        score = ScoreStream(scene, bucket, normal)
        score.step(position[0, 0], velocity[0, 0], dict(zip(FORCE_NAMES, forces[0, 0])))
        row = dict(score.result(), id="synthetic-fall", bucket_xy_m=bucket, board_normal=normal)
        arrays = dict(position_m=position, velocity_m_s=velocity, forces_N=forces)
        audit = Audit(); replay_scores(audit, scene, [row], arrays, "mock", 0)
        self.assertTrue(all(c["passed"] for c in audit.checks))
        arrays["position_m"][:, :, 2] = .5  # Erase recorded fall without changing claimed result.
        audit = Audit(); replay_scores(audit, scene, [row], arrays, "mock", 0)
        self.assertFalse(next(c["passed"] for c in audit.checks if c["name"].endswith("independent_scalar_rescore")))


if __name__ == "__main__":
    unittest.main()
