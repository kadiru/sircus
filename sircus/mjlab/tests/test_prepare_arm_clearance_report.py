"""Synthetic publication fixtures; no model, simulator or hardware is executed."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from sircus.mjlab import prepare_arm_clearance_report as publisher
from sircus.mjlab.arm_clearance_study import protocol
from sircus.mjlab.arm_clearance_selector import RULE


def write(path, value): path.write_text(json.dumps(value, sort_keys=True, allow_nan=False)+'\n')
def snapshot(path): return {str(p.relative_to(path)): p.read_bytes() for p in path.rglob('*') if p.is_file()}
def count_rows(rows):
    return dict(requested=len(rows), native_releases=sum(r['native_release'] for r in rows),
        captures=sum(r['capture'] for r in rows), clean=sum(r['clean'] for r in rows),
        rejected=sum(not r['native_release'] for r in rows), invalid=0,
        native_misses=sum(r['native_release'] and not r['capture'] for r in rows))


def fixture(root, status='completed', audit=True):
    study = root/'study'; study.mkdir()
    spec = dict(version=publisher.VERSION, protocol=protocol(), maximum_requests=112, new_training_requests=0,
        source_sha256={'fixture-only.py': '1'*64}, native_source_sha256={'native-fixture-only.py': '2'*64}, clearance_rule=RULE)
    spec['fingerprint'] = publisher.digest(spec); write(study/'contract.json', spec)
    write(study/'prediction-plan.json', dict(version=publisher.VERSION, contract_fingerprint=spec['fingerprint'], fixture=True))
    positions = []
    for method in publisher.METHODS:
        for i in range(50):
            native = i < (42 if method == 'consistent' else 40)
            capture = i < (24 if method == 'consistent' else 26)
            positions.append(dict(context_id='clearance-test-'+str(i), method=method, record_id=method+'-'+str(i),
                bucket_xy_m=[.2+i*.01, -.3], values=[.6, -.1, .3, 28, .15], native_release=native, capture=capture,
                clean=capture and i < 12, whole_ball_entry=capture if native else None, retention_s=1 if capture else 0 if native else None,
                invalid_physics=False if native else None, clearance_rejected=method == 'arm_clearance' and 40 <= i < 42))
    comparisons = [dict(method=m, counts=count_rows([r for r in positions if r['method'] == m])) for m in publisher.METHODS]
    control = dict(requested=6, native_releases=6, captures=3, clean=3, rejected=0, native_misses=3, invalid=0)
    totals = {k: sum(r['counts'][k] for r in comparisons)+2*control[k] for k in publisher.COUNT_FIELDS}
    report = dict(version=publisher.VERSION, status=status, phase='completed' if status == 'completed' else status,
        scope='Synthetic publication fixture only', protocol=protocol(), contract_fingerprint=spec['fingerprint'],
        planned_layouts=50, prediction_plan_sha256=publisher.sha(study/'prediction-plan.json'),
        counts=totals, comparisons=comparisons, test_positions=positions,
        control_gates={key: dict(status='passed', counts=control) for key in ('pre', 'post')})
    (study/'raw-fixture.dat').write_bytes(b'Synthetic bytes, never native evidence.\n')
    manifest = {p.name: publisher.sha(p) for p in study.iterdir() if p.is_file()}
    write(study/'evidence-manifest.json', manifest); report['evidence_sha256'] = manifest; write(study/'report.json', report)
    enriched = copy.deepcopy(positions)
    for row in enriched:
        row['invalid'] = False
        for s in publisher.SEGMENTS: row['robot_contact_'+s] = (not row['capture'] and s != 'after_terminal') if row['native_release'] else None
        row.update(predicted_clearance_min_m=.004 if row['method'] == 'arm_clearance' else None,
                   predicted_clearance_closest_geom='arm10/finger' if row['method'] == 'arm_clearance' else None,
                   first_post_bounce_contact_channels=['bucket'] if row['capture'] else ['robot'] if row['native_release'] else None,
                   first_contact_delay_ms=100 if row['native_release'] else None)
    contact_groups = []
    for method in publisher.METHODS:
        native = [r for r in enriched if r['method'] == method and r['native_release']]
        group = dict(method=method, requested=50, native_releases=len(native), observed_native=len(native), missing_native=0, force_threshold_N=.01)
        for s in publisher.SEGMENTS: group['robot_contact_trials_'+s] = sum(r['robot_contact_'+s] for r in native)
        contact_groups.append(group)
    proof = dict(version=publisher.AUDIT_VERSION, status='passed', failures=[],
        checks=[dict(label='Synthetic fixture full proof', passed=True)], checks_count=1,
        auditor_sha256=publisher.sha(Path(publisher.__file__).with_name('arm_clearance_verify.py')),
        report_sha256=publisher.sha(study/'report.json'), contract_fingerprint=spec['fingerprint'],
        evidence_manifest_sha256=publisher.sha(study/'evidence-manifest.json'), prediction_plan_sha256=publisher.sha(study/'prediction-plan.json'),
        requested_trials=112, native_releases=totals['native_releases'], training_requests=0,
        producer_source_sha256=spec['source_sha256'], native_source_sha256=spec['native_source_sha256'],
        comparisons=copy.deepcopy(comparisons), control_gates=copy.deepcopy(report['control_gates']), test_positions=enriched, contact_comparisons=contact_groups)
    audit_path = root/'audit.json'
    if audit: write(audit_path, proof)
    return study, audit_path, proof


class ArmClearancePublicationTests(unittest.TestCase):
    def test_full_bound_proof_publishes_counts_contacts_and_exact_examples_without_raw_data(self):
        with tempfile.TemporaryDirectory() as temp:
            study, audit, _ = fixture(Path(temp)); original = snapshot(study)
            value = publisher.compact(study, audit)
            self.assertEqual(value['status'], 'verified')
            self.assertEqual([r['counts']['captures'] for r in value['comparisons']], [24, 26])
            self.assertEqual(len(value['test_positions']), 100)
            self.assertEqual([r['robot_contact_trials_post_rebound'] for r in value['contact_comparisons']], [18, 14])
            self.assertEqual(value['audit']['source_sha256'], publisher.sha(audit))
            self.assertEqual(value['test_positions'][-1]['robot_contact_before_terminal'], None)
            self.assertEqual(value['test_positions'][-1]['retention_s'], None)
            self.assertEqual(snapshot(study), original)
            self.assertNotIn('native_cache_file', json.dumps(value))

    def test_progress_or_missing_audit_never_displays_comparison_outcomes(self):
        for status, expected in (('preparing', 'preparing'), ('prepared', 'pending'), ('queued', 'queued'), ('running', 'running'), ('completed', 'awaiting_audit')):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as temp:
                study, audit, _ = fixture(Path(temp), status=status, audit=False)
                result = publisher.compact(study, audit)
                self.assertEqual(result['status'], expected)
                self.assertEqual(result['comparisons'], [])
                self.assertEqual(result['test_positions'], [])
                self.assertIsNone(result['audit']['passed'])

    def test_each_changed_native_artifact_or_plan_is_rejected(self):
        for name in ('contract.json', 'report.json', 'prediction-plan.json', 'raw-fixture.dat', 'evidence-manifest.json'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp:
                study, audit, _ = fixture(Path(temp)); path = study/name; path.write_bytes(path.read_bytes()+b' ')
                with self.assertRaises(ValueError): publisher.compact(study, audit)

    def test_incomplete_or_wrong_proof_fails_closed(self):
        changes = {'version':'old', 'checks_count':2, 'checks':[], 'failures':['x'], 'status':'pending', 'requested_trials':111,
            'training_requests':1, 'auditor_sha256':'0'*64, 'report_sha256':'0'*64, 'contract_fingerprint':'0'*64,
            'evidence_manifest_sha256':'0'*64, 'prediction_plan_sha256':'0'*64, 'producer_source_sha256':{}, 'native_source_sha256':{}}
        for key, value in changes.items():
            with self.subTest(key=key), tempfile.TemporaryDirectory() as temp:
                study, audit, proof = fixture(Path(temp)); proof[key] = value; write(audit, proof)
                with self.assertRaises(ValueError): publisher.compact(study, audit)
        with tempfile.TemporaryDirectory() as temp:
            study, audit, proof = fixture(Path(temp)); proof['checks'][0]['passed'] = False; write(audit, proof)
            with self.assertRaises(ValueError): publisher.compact(study, audit)

    def test_paired_positions_and_contacts_cannot_disagree_with_native_outcomes(self):
        mutations = [lambda p:p['test_positions'].pop(), lambda p:p['test_positions'][1].update(record_id=p['test_positions'][0]['record_id']),
            lambda p:p['test_positions'][0].update(capture=False), lambda p:p['test_positions'][0].update(values=[.5, 0, .2, 12, .3]),
            lambda p:p['test_positions'][42].update(robot_contact_before_terminal=False),
            lambda p:p['contact_comparisons'][0].update(robot_contact_trials_post_rebound=0),
            lambda p:p['contact_comparisons'][0].update(force_threshold_N=.1),
            lambda p:p['contact_comparisons'][0].update(observed_native=41), lambda p:p['contact_comparisons'].pop()]
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temp:
                study, audit, proof = fixture(Path(temp)); mutation(proof); write(audit, proof)
                with self.assertRaises((ValueError, KeyError)): publisher.compact(study, audit)

    def test_failed_audit_or_pipeline_remains_failed_and_preserves_other_display_sections(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); study, audit, proof = fixture(root)
            display = root/'display.json'; write(display, dict(historical={'count':24}, interactive_policy='unchanged'))
            original = snapshot(study); proof['status'] = 'failed'; write(audit, proof)
            result = publisher.publish(display, study, audit)
            self.assertEqual(result['status'], 'audit_failed'); self.assertEqual(result['comparisons'], [])
            current = publisher.read(display)
            self.assertEqual(current['historical'], {'count':24}); self.assertEqual(current['interactive_policy'], 'unchanged')
            result = publisher.publish(display, study, None, pipeline=dict(status='failed', error='Own bounded queue expired'))
            self.assertEqual(result['status'], 'failed'); self.assertEqual(result['error'], 'Own bounded queue expired')
            self.assertEqual(snapshot(study), original)

    def test_publication_refuses_any_path_inside_native_study(self):
        with tempfile.TemporaryDirectory() as temp:
            study, audit, _ = fixture(Path(temp))
            with self.assertRaisesRegex(ValueError, 'outside'): publisher.publish(study/'display.json', study, audit)


if __name__ == '__main__': unittest.main()
