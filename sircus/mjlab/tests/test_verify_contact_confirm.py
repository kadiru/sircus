"""Independent confirmation-accounting corruption tests, without native physics."""

import copy
import unittest

from sircus.mjlab.contact_confirm import confirmation_plan
from sircus.mjlab.verify_contact_confirm import audit_records, digest, SOURCE_ID


def fixture():
    original = dict(id=SOURCE_ID, values=[.47, -.28, .4, 36., .3], q10=[0.] * 6,
        q14=[1.] * 6, bucket_xy_m=[.16, -.64])
    plan = confirmation_plan(original)
    for p in plan:
        p.update(q10=original["q10"], q14=original["q14"], geometry_eligible=True, geometry_reason=0)
    variants = [dict(id=f"articulated-{g}-{s}", board_mode="articulated", gravity_mode=g, substeps=s)
        for g, s in (("legacy", 8), ("passive_only", 32), ("passive_only", 64))]
    spec = dict(version="sircus-contact-fresh-confirmation-v1", source=original, source_id=SOURCE_ID,
        seed=20261125, noise_std=[.001, .001, .002, .35, .002], plan=plan, variants=variants, reset_repeats=2)
    spec["fingerprint"] = digest(spec)
    def row(v, reset, p):
        suffix = f'singleton-{p["id"]}' if reset == "singleton" else f'reset-{reset}-{p["id"]}'
        return dict(p, id=f'{v["id"]}-{suffix}', source_plan_id=p["id"], variant=v["id"], reset=reset,
            bucket_xy_m=original["bucket_xy_m"],
            contact_protocol={k: v[k] for k in ("board_mode", "gravity_mode", "substeps")},
            native_release=True, scalar_score_audit_passed=True, invalid=None, route_success=True,
            route_quality=2, board_bounce=True, max_integrated_joint_excursion_rad=0.)
    rows = [row(v, reset, p) for v in variants for reset in (0, 1) for p in plan]
    singletons = [row(v, "singleton", p) for v in variants[1:] for p in (plan[0], plan[16])]
    for r in singletons:
        r["family"] = "singleton_diagnostic"
    rows += singletons
    groups = [dict(v, attempts=232, captures=232,
        family_results=[dict(family=f, attempts=2 * n, independent_requests=n, captures=2 * n,
            clean=2 * n, invalid=0, board_bounces=2 * n, maximum_integrated_arm_hinge_excursion_rad=0.)
            for f, n in (("nominal", 16), ("fresh_perturbation", 100))]) for v in variants]
    report = dict(version=spec["version"], status="completed", source=original, contract_fingerprint=spec["fingerprint"],
        groups=groups, records=700, singletons=[dict(variant=r["variant"], source_plan_id=r["source_plan_id"], capture=True) for r in singletons])
    return spec, report, rows, original


class ConfirmationVerificationTests(unittest.TestCase):
    def checks(self, f):
        return {c["name"]: c["passed"] for c in audit_records(*f).checks}

    def test_complete_matched_data_passes(self):
        checks = self.checks(fixture())
        self.assertTrue(all(checks.values()), checks)

    def test_missing_failed_throw_not_hidden_by_revised_report(self):
        f = fixture(); f[2].pop(200); f[1]["records"] -= 1
        self.assertFalse(self.checks(f)["all_696_paired_outcomes_present_once"])

    def test_singleton_not_counted_as_fresh(self):
        f = fixture(); f[2][-1]["reset"] = 0
        self.assertFalse(self.checks(f)["four_separate_singleton_diagnostics"])

    def test_more_replays_do_not_claim_more_independent_requests(self):
        f = fixture(); f[1]["groups"][1]["family_results"][1]["independent_requests"] = 200
        self.assertFalse(self.checks(f)["articulated-passive_only-32_fresh_perturbation_statistics"])

    def test_native_joint_substitution_rejected(self):
        f = fixture(); f[2][0]["q10"] = [.01] * 6
        self.assertFalse(self.checks(f)["same_requests_and_joint_branches_every_protocol"])

    def test_false_singleton_outcome_rejected(self):
        f = fixture(); f[1]["singletons"][0]["capture"] = False
        self.assertFalse(self.checks(f)["singleton_report_outcomes"])

    def test_regenerated_fresh_noise_independent_of_new_fingerprint(self):
        f = fixture(); p = next(p for p in f[0]["plan"] if p["family"] == "fresh_perturbation")
        p["values"][3] += .001
        payload = copy.deepcopy(f[0]); payload.pop("fingerprint")
        f[0]["fingerprint"] = f[1]["contract_fingerprint"] = digest(payload)
        self.assertFalse(self.checks(f)["independently_regenerated_100_requests"])


if __name__ == "__main__":
    unittest.main()
