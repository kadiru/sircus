"""Synthetic trust-boundary and diagnostic accounting tests; CPU only."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from sircus.mjlab import gap_flight_diagnosis as d
from sircus.mjlab.tests.test_consistent_flight_diagnosis import write, rebind, pins


def fixture(root):
    study = root/'study'; study.mkdir(); audit = root/'audit.json'
    protocol = dict(substeps=64, methods=list(d.METHODS), test_layouts=50, maximum_fresh_requests=112,
        control_requests=12, new_train_requests=0, noise=False, no_test_flight_search=True,
        model_weights_unchanged=True, candidates=16, maximum_geometry_checks=4,
        changed_candidate_indices=[8, 9], gap_residual_endpoints=[-1, 1], unchanged_candidate_count=14,
        no_hard_aperture_veto=True, post_bounce_robot_clearance_added=False)
    spec = dict(version='sircus-gap-coverage-study-v1', source_sha256={'source.py': 'a'*64},
                native_source_sha256={'native.py': 'b'*64}, protocol=protocol)
    spec['fingerprint'] = d.digest(spec); write(study/'contract.json', spec)
    rows = []
    def add(role, phase, method, context, xy):
        success = role == 'positive_control'
        rows.append(dict(id='row-'+str(len(rows)), role=role, phase=phase, method=method,
            context_id=context, bucket_xy_m=xy, values=[.5, -.2, .3, 30., .2],
            native_release=True, route_success=success, route_quality=1 if success else 0,
            invalid=None, invalid_physics=False, whole_ball_entry=success,
            scalar_score_audit_passed=True, action_substituted=False))
    for phase in ('controls_pre', 'controls_post'):
        if phase == 'controls_post':
            for method in d.METHODS:
                for i in range(50): add('test', 'test', method, 'context-'+str(i), [.2+i*.01, -.3])
        for role in ('positive_control', 'negative_control'):
            for i in range(3): add(role, phase, 'control', role, [.8, -.2])
    packet = dict(frozen_before_any_native_outcome=True, native_planning_queries=0, frozen_unix_s=5.,
        requests=[{k: r[k] for k in ('id', 'role', 'phase', 'method', 'context_id', 'bucket_xy_m', 'values')} for r in rows])
    write(study/'prediction-plan.json', packet)
    (study/'records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    comparisons = [dict(method=m, counts=d.statistics([r for r in rows if r['role'] == 'test' and r['method'] == m])) for m in d.METHODS]
    report = dict(version=spec['version'], status='completed', phase='completed', contract_fingerprint=spec['fingerprint'],
        source_sha256=spec['source_sha256'], protocol=protocol, counts=d.statistics(rows), comparisons=comparisons,
        acquired_unix_s=6., control_gates={k:d.control_gate([r for r in rows if r['phase']==phase]) for k,phase in [('pre','controls_pre'),('post','controls_post')]})
    proof = dict(version='sircus-gap-coverage-audit-v2', status='passed', failures=[],
        checks=[dict(label='synthetic fixture only', passed=True)], checks_count=1,
        contract_fingerprint=spec['fingerprint'], requested_trials=112, training_requests=0, native_releases=112,
        producer_source_sha256=spec['source_sha256'], native_source_sha256=spec['native_source_sha256'], comparisons=comparisons)
    write(study/'report.json', report); write(audit, proof); rebind(study, audit)
    return study, audit


class BindingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.study, self.audit = fixture(self.root)
    def tearDown(self): self.temp.cleanup()
    def verify(self):
        with patch.object(d, 'SOURCE_HASHES', pins(self.study, self.audit)):
            return d.verify_inputs(self.study, self.audit)
    def test_all112_ordered_requests_and50paired_layouts_preserved(self):
        result = self.verify(); self.assertEqual(len(result), 7); self.assertEqual(len(result[-1]), 112)
    def test_actual_pins_reject_synthetic_data(self):
        with self.assertRaises(ValueError): d.verify_inputs(self.study, self.audit)
    def test_mutated_evidence_fails_even_with_rebound_report_pin(self):
        p = self.study/'records.jsonl'; p.write_text(p.read_text()+'\n')
        with self.assertRaises(ValueError): self.verify()
    def test_full_audit_requires_explicit_true_not_truthiness(self):
        a = d.read(self.audit); a['checks'][0]['passed'] = 1; write(self.audit, a)
        with self.assertRaises(ValueError): self.verify()
    def test_changed_plan_or_native_action_not_hidden_by_rehash(self):
        p = self.study/'records.jsonl'; rows = [json.loads(x) for x in p.read_text().splitlines()]
        rows[10]['values'][4] += .01; p.write_text(''.join(json.dumps(x)+'\n' for x in rows)); rebind(self.study, self.audit)
        with self.assertRaises(ValueError): self.verify()
    def test_missing_requested_slot_cannot_change_denominator(self):
        p = self.study/'records.jsonl'; p.write_text('\n'.join(p.read_text().splitlines()[:-1])+'\n'); rebind(self.study, self.audit)
        with self.assertRaises(ValueError): self.verify()
    def test_source_or_existing_output_cannot_be_overwritten(self):
        for target in (self.study, self.study/'derived'):
            with self.assertRaises(ValueError): d.build(self.study, self.audit, target)


class DiagnosticTests(unittest.TestCase):
    def test_unknown_setup_metrics_remain_null_not_zero(self):
        p = d.setup_proof({}, {}, {'reason':'finite_ik_unresolved'})
        self.assertIsNone(p['valid_ik_branches']); self.assertIsNone(p['home_minimum_clearance_m'])
        self.assertEqual(p['endpoint_examples'], [])
    def test_setup_preserves_clearance_and_arm_branch_evidence(self):
        connection = dict(reason='strict_endpoint_clearance_or_pose_rejected', endpoint_attempts=dict(
            valid_ik_branches=[10, 5], pair_attempts=[dict(branch10=0, branch14=0, checked=dict(
                board_position_error_m=1e-9, strict_geometry=dict(minimum_clearance_m=.001419,
                closest_pair=['Arm14/link4', 'Arm14/link6'])))]))
        p = d.setup_proof({}, {}, connection)
        self.assertEqual(p['valid_ik_branches'], [10,5]); self.assertAlmostEqual(p['endpoint_examples'][0]['minimum_clearance_m'], .001419)
        self.assertIsNone(p['endpoint_examples'][0]['dropper_position_error_m'])
    def test_early_contact_without_rim_crossing_stays_unknown(self):
        item = dict(values=[.5,0.,.3,30.,.1], bucket_xy_m=[.8,0.], observed=dict(first_flight_available=True,
            cutoff_contact_channels=['robot'], cutoff_time_s=.108, rebound_time_s=.1, rim_crossing=None,
            rim_forward_error_mm=None, rim_lateral_error_mm=None))
        d.enrich_flight(item); o = item['observed']
        self.assertAlmostEqual(o['rebound_to_contact_ms'],8.); self.assertIs(o['early_robot_contact'], True)
        self.assertIsNone(o['rim_forward_error_mm']); self.assertIsNone(o['rim_lateral_error_mm'])
    def test_recorded_rim_projection_has_signed_forward_and_lateral_error(self):
        item = dict(values=[.5,0.,.3,30.,.1], bucket_xy_m=[.8,0.], observed=dict(first_flight_available=True,
            cutoff_contact_channels=['table'], cutoff_time_s=.6, rebound_time_s=.2,
            rim_crossing={'xy_m':[.65,.02]}))
        d.enrich_flight(item); self.assertAlmostEqual(item['observed']['rim_forward_error_mm'], -150.)
        self.assertAlmostEqual(item['observed']['rim_lateral_error_mm'],20.)
    def test_bucket_retention_never_promoted_to_complete_route_capture(self):
        o = dict(native_release=True, capture=False, bucket_capture=True)
        self.assertIn('route scoring failed', d.prior.classification(o)); self.assertIs(o['capture'],False)
    def test_pairs_retain_every_original_and_wider_outcome(self):
        items = [dict(id=m+str(i), context_id=str(i), method=m, values=[.5,0.,.3,30.,.2],
                      bucket_xy_m=[i*.01,0.], observed=dict(capture=(i==0 if m=='consistent' else i==1)))
                 for m in d.METHODS for i in range(50)]
        unchanged = copy.deepcopy(items); p, table = d.paired_results(items)
        self.assertEqual(len(table),50); self.assertEqual(p['gap_coverage']['wins'],['0'])
        self.assertEqual(p['gap_coverage']['losses'],['1']); self.assertEqual(items,unchanged)
        with self.assertRaises(ValueError): d.paired_results(items[:-1])
    def test_import_has_no_engine_or_cuda_library(self):
        code = 'import sys; import sircus.mjlab.gap_flight_diagnosis; assert not any(x in sys.modules for x in ("torch","mujoco","warp","mjlab"))'
        subprocess.run([sys.executable,'-c',code],check=True)


if __name__ == '__main__': unittest.main()
