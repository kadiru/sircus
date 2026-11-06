"""Synthetic CPU fixtures for the fixed follow-up; no native performance claim."""
import copy
import unittest

import numpy as np

from sircus.mjlab import tilt650_verify as audit
from sircus.mjlab.benchmark_verify import Checks, EvidenceError
from sircus.mjlab.tests.test_xy650_verify import row


def sources():
    return [row(ident, values=[.47+delta[0], -.27+delta[1], .38, 30., gap],
                bucket_xy_m=[1.12+delta[0], -.27+delta[1]], translation_m=delta,
                route_success=True, route_quality=1, rim_band_contact=True, substeps=64)
            for ident, (delta, gap) in zip(audit.SOURCE_IDS, audit.SOURCE_CONFIGS)]


def fresh_rows(quality=1):
    return [row(f"verification-{step}-tilt30-00-r{i}", role="verification", phase="verification", group_id="tilt30-00",
                substeps=step, repeat_index=i, pair_id=f"verification-tilt30-00-r{i}", route_success=True,
                route_quality=quality, scalar_score_audit_passed=True)
            for step in (64, 128) for i in range(3)]


class TiltAuditTests(unittest.TestCase):
    def test_literal_source_ids_and_geometry_are_not_resampled(self):
        selected = sources()
        self.assertEqual(audit.select_sources(selected, dict(verification=[]), Checks()), selected)
        self.assertEqual(audit.SOURCE_IDS, tuple(f"coarse-64-xy-grid-{i:04d}-r0" for i in (43, 70, 96, 151, 177)))
        changed = copy.deepcopy(selected);changed[0]["values"][4] += .001
        with self.assertRaisesRegex(EvidenceError, "recipe"):
            audit.select_sources(changed, dict(verification=[]), Checks())
        with self.assertRaisesRegex(EvidenceError, "not one of"):
            audit.select_sources(selected, dict(verification=[selected[0]]), Checks())
        extra = row("coarse-64-extra", values=selected[0]["values"], route_success=True, route_quality=1, substeps=64)
        with self.assertRaisesRegex(EvidenceError, "All and only"):
            audit.select_sources(selected+[extra], dict(verification=[]), Checks())

    def test_fresh_capture_and_clean_tiers_do_not_relabel_parent_rim(self):
        rows = fresh_rows();positions = {r["id"]: [0., 0., .3] for r in rows}
        capture = audit.qualification(rows, positions, controls_passed=True, parent_source_capture=True)
        self.assertEqual(capture["status"], "capture_only")
        self.assertTrue(capture["capture_qualified"])
        self.assertFalse(capture["clean_qualified"])
        self.assertFalse(capture["qualified"])
        self.assertTrue(capture["parent_source_capture"])
        for r in rows:r["route_quality"] = 2
        clean = audit.qualification(rows, positions, controls_passed=True, parent_source_capture=True)
        self.assertEqual(clean["status"], "qualified")
        self.assertTrue(clean["capture_qualified"] and clean["clean_qualified"] and clean["qualified"])
        # Parent quality was1 throughout; only NEW repeats determine the tier.
        self.assertTrue(all(r["route_quality"] == 1 for r in sources()))

    def test_control_parent_physics_oracle_drift_and_finer_capture_all_gate_results(self):
        rows = fresh_rows(2);positions = {r["id"]: [0., 0., .3] for r in rows}
        for kwargs in (dict(controls_passed=False, parent_source_capture=True), dict(controls_passed=True, parent_source_capture=False)):
            result = audit.qualification(rows, positions, **kwargs)
            self.assertEqual(result["status"], "failed")
            self.assertFalse(result["qualified"] or result["capture_qualified"])
        for changes in (dict(route_success=False, route_quality=0), dict(invalid_physics=True), dict(scalar_score_audit_passed=False)):
            changed = copy.deepcopy(rows);changed[-1].update(changes)
            self.assertFalse(audit.qualification(changed, positions, controls_passed=True, parent_source_capture=True)["capture_qualified"])
        moved = copy.deepcopy(positions);moved[rows[-1]["id"]] = [.00500001, 0., .3]
        self.assertFalse(audit.qualification(rows, moved, controls_passed=True, parent_source_capture=True)["capture_qualified"])
        self.assertEqual(audit.qualification(rows[:-1], positions, controls_passed=True, parent_source_capture=True)["status"], "pending")

    def test_independent_qualification_matches_producer_full_dense_position_schema(self):
        from sircus.mjlab import tilt650_contract as producer
        for quality in (1, 2):
            rows = fresh_rows(quality)
            arrays = {r["id"]: np.zeros((round(.35*240*r["substeps"])+1, 3)) for r in rows}
            points = {r["id"]: arrays[r["id"]][-1] for r in rows}
            self.assertEqual(audit.qualification(rows, points, controls_passed=True, parent_source_capture=True),
                             producer.qualification(rows, arrays, controls_passed=True, parent_source_capture=True))

    def test_source_branches_and_query_provenance_are_frozen(self):
        from sircus.mjlab import tilt650_contract as producer
        from sircus.mjlab import xy650_contract as xy
        selected = sources();controls = xy.bind([row("positive-control", role="positive_control"),
                                                row("negative-control", role="negative_control")], "a"*64)
        groups = producer.groups(selected, controls, "a"*64)
        audit.audit_groups(groups, selected, controls, "a"*64, Checks())
        changed = copy.deepcopy(groups);changed[2]["q14"][0] += 1e-9
        with self.assertRaisesRegex(EvidenceError, "requested joints"):
            audit.audit_groups(changed, selected, controls, "a"*64, Checks())
        changed = copy.deepcopy(groups);changed[2]["parent_route_quality"] = 2
        with self.assertRaisesRegex(EvidenceError, "rim source"):
            audit.audit_groups(changed, selected, controls, "a"*64, Checks())
        changed = copy.deepcopy(groups);changed[2]["source_record_sha256"] = "c"*64
        with self.assertRaisesRegex(EvidenceError, "parent record bytes"):
            audit.audit_groups(changed, selected, controls, "a"*64, Checks())

    def test_literal54_requests_keep_every_reject_and_exact_pair_inputs(self):
        from sircus.mjlab import tilt650_contract as producer
        from sircus.mjlab import xy650_contract as xy
        selected = sources();controls = xy.bind([row("positive-control", role="positive_control"),
                                                row("negative-control", role="negative_control")], "a"*64)
        groups = producer.groups(selected, controls, "a"*64)
        for i, group in enumerate(groups):group.update(eligible=i != 3, geometry_row=i)
        plans = [audit.xy_audit.make_plan(g, phase, steps=audit.STEPS, repeats=3) for g, phase in
                 ((groups[:2], "controls_pre"), (groups[2:], "verification"), (groups[:2], "controls_post"))]
        ledger = sum(plans, [])
        self.assertEqual([len(p) for p in plans], [12, 30, 12])
        self.assertEqual(len(ledger), 54)
        self.assertEqual(len({r["id"] for r in ledger}), 54)
        self.assertEqual(sum(not r["eligible"] for r in ledger), 6)
        rows = fresh_rows();audit.replay.check_pairing(rows, Checks())
        rows[-1]["q10"][0] += 1e-9
        with self.assertRaisesRegex(EvidenceError, "same frozen input"):
            audit.replay.check_pairing(rows, Checks())

    def test_settings_and_versions_are_explicit(self):
        from sircus.mjlab import tilt650_contract as producer
        audit.audit_settings(producer.settings(), Checks())
        self.assertEqual(audit.VERSION, producer.AUDIT_VERSION)
        self.assertEqual(audit.STUDY_VERSION, producer.VERSION)
        self.assertEqual(audit.PARENT_AUDIT_SHA, producer.PARENT_AUDIT_SHA)
        changed = producer.settings();changed["max_requests"] = 55
        with self.assertRaisesRegex(EvidenceError, "max_requests"):
            audit.audit_settings(changed, Checks())


if __name__ == "__main__":
    unittest.main()
