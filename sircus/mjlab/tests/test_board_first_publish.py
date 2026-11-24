import json
from pathlib import Path
import tempfile
import unittest

from sircus.mjlab import board_first_publish as module


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.study = Path(self.temp.name) / 'evidence'
        self.study.mkdir()
        self.counts = dict(scorable_native=10, old_captures=4, new_captures=5,
            gained_captures=2, lost_captures=1, unchanged_captures=3,
            unchanged_misses=3, invalid_native=1)
        (self.study / 'manifest.json').write_text('{}')
        summary = dict(version='sircus-board-first-saved-rescore-v1', status='verified',
            totals=self.counts, buckets=[dict(bucket_xy_m=[1., .025], counts=self.counts)],
            metadata={}, exclusions={})
        (self.study / 'summary.json').write_text(json.dumps(summary))
        audit = dict(version='sircus-board-first-saved-rescore-audit-v1', status='passed',
            summary_sha256=module.sha(self.study / 'summary.json'),
            manifest_sha256=module.sha(self.study / 'manifest.json'))
        (self.study / 'audit.json').write_text(json.dumps(audit))

    def test_additive_publication_preserves_old_results(self):
        path = Path(self.temp.name) / 'display.json'
        prior = dict(clear_shot_range={'historical': True}, planner='hybrid')
        path.write_text(json.dumps(prior))
        module.publish(self.study, path)
        actual = json.loads(path.read_text())
        for key, value in prior.items():
            self.assertEqual(actual[key], value)
        self.assertEqual(actual['board_first_study']['rescore']['counts']['gained'], 2)
        self.assertEqual(actual['board_first_study']['comparison']['status'], 'pending')

    def test_changed_audited_summary_fails_without_publishing(self):
        path = Path(self.temp.name) / 'display.json'
        path.write_text('{"preserved":true}')
        (self.study / 'summary.json').write_text((self.study / 'summary.json').read_text() + ' ')
        with self.assertRaisesRegex(ValueError, 'bind'):
            module.publish(self.study, path)
        self.assertEqual(path.read_text(), '{"preserved":true}')

    def test_no_output_inside_original_evidence(self):
        with self.assertRaisesRegex(ValueError, 'outside'):
            module.publish(self.study, self.study / 'display.json')

    def test_gain_loss_counts_cannot_inflate_denominator(self):
        with self.assertRaisesRegex(ValueError, 'accounting'):
            module.normalized_counts(dict(self.counts, scorable_native=11))

    def test_changed_shot_context_must_reconcile_with_audited_totals(self):
        summary = module.read(self.study/'summary.json')
        summary['changes'] = [dict(old_capture=False, new_capture=True, assistance_surfaces=['robot'])]
        (self.study/'summary.json').write_text(json.dumps(summary))
        audit = module.read(self.study/'audit.json')
        audit['summary_sha256'] = module.sha(self.study/'summary.json')
        (self.study/'audit.json').write_text(json.dumps(audit))
        with self.assertRaisesRegex(ValueError, 'Changed-shot accounting'):
            module.rescore_section(self.study)


class ComparisonPublicationTests(PublicationTests):
    def setUp(self):
        super().setUp()
        self.comparison = Path(self.temp.name) / 'comparison'
        self.comparison.mkdir()
        self.audit_path = Path(self.temp.name) / 'comparison-audit.json'
        self.methods = ('original', 'retrained')
        layouts = [dict(id=f'p-{i}', bucket_xy_m=[.1 + i * .02, -.4]) for i in range(50)]
        self.write('layouts.json', dict(layouts=layouts))
        for method in self.methods:
            self.write(method + '-checkpoint.json', dict(method=method, weights=[1, 2, 3]))
        contract = dict(version='sircus-board-first-selector-comparison-v1', methods=list(self.methods),
            test_positions=50, physics_substeps=64, attempts_per_method_per_position=1,
            scoring_version='sircus-board-first-native-capture-v4', planning_budget_s=30.,
            frozen_inputs={name: module.sha(self.comparison/name) for name in
                ('layouts.json','original-checkpoint.json','retrained-checkpoint.json')})
        contract['fingerprint'] = module.digest(contract)
        self.write('contract.json', contract)
        slots, plans, rows = [], [], []
        for i, layout in enumerate(layouts):
            for method in self.methods:
                identity = layout['id'] + '-' + method
                refused = i >= (48 if method == 'original' else 49)
                request = None if refused else dict(id=identity, bucket_xy_m=layout['bucket_xy_m'])
                slot = dict(id=identity, context_id=layout['id'], method=method,
                    bucket_xy_m=layout['bucket_xy_m'], request=request)
                slots.append(slot)
                plans.append(dict(choice=slot, method=method, decision_latency_s=2. if method == 'original' else 3.))
                if not refused:
                    hit = i < (35 if method == 'original' else 40)
                    invalid = i == (47 if method == 'original' else 48)
                    rows.append(dict(id=identity, bucket_xy_m=layout['bucket_xy_m'],
                        adapter_directory='native-'+method, native_release=True,
                        board_first_scalar_audit_passed=True,
                        board_first_score=dict(scoring_version=contract['scoring_version'],
                            route_success=hit, invalid_physics=invalid)))
        rows += [dict(id=f'control-{i}', adapter_directory='native-controls') for i in range(18)]
        self.write_rows(rows)
        self.write('plans.json', plans)
        bindings = dict(contract['frozen_inputs'], **{'plans.json': module.sha(self.comparison/'plans.json')})
        self.write('choices.json', dict(slots=slots, bindings=bindings))
        planned = {method:[slot['request'] for slot in slots if slot['method']==method and slot['request'] is not None]
                   for method in self.methods}
        self.write('native-started.json', dict(bindings=dict(bindings, **{'choices.json':module.sha(self.comparison/'choices.json')}),
                    requests_sha256=module.digest(planned)))
        self.report = dict(version=contract['version'], status='completed', phase='completed', native_completed=True,
            controls={phase:dict(status='passed', board_first_passed=True) for phase in ('controls_pre','controls_post')},
            progress=dict(prepared=100, planned_releases=97, completed_releases=97),
            comparison=dict(methods={
                'original':dict(positions=50,captures=35,releases=48,setup_rejections=2,invalid_physics=1,
                    flight_misses=12,capture_rate=.7,mean_planning_s=2.,median_planning_s=2.,max_planning_s=2.),
                'retrained':dict(positions=50,captures=40,releases=49,setup_rejections=1,invalid_physics=1,
                    flight_misses=8,capture_rate=.8,mean_planning_s=3.,median_planning_s=3.,max_planning_s=3.)},
                paired=dict(both=35,original_only=0,retrained_only=5,neither=10)))
        self.seal()

    def write(self, name, value):
        (self.comparison/name).write_text(json.dumps(value))

    def write_rows(self, rows):
        (self.comparison/'rows.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))

    def seal(self):
        files = ('contract.json','layouts.json','plans.json','choices.json','native-started.json',
                 'original-checkpoint.json','retrained-checkpoint.json','rows.jsonl')
        self.write('evidence-manifest.json', {name:module.sha(self.comparison/name) for name in files})
        self.report['evidence_manifest_sha256'] = module.sha(self.comparison/'evidence-manifest.json')
        self.write('report.json', self.report)
        audit = dict(version='sircus-board-first-selector-independent-audit-v1', status='passed',
            report_sha256=module.sha(self.comparison/'report.json'),
            evidence_manifest_sha256=module.sha(self.comparison/'evidence-manifest.json'),
            contract_sha256=module.sha(self.comparison/'contract.json'),
            choices_sha256=module.sha(self.comparison/'choices.json'),
            native_started_sha256=module.sha(self.comparison/'native-started.json'),
            checkpoint_sha256={method:module.sha(self.comparison/(method+'-checkpoint.json')) for method in self.methods},
            comparison=self.report['comparison'],frozen_before_test=True,checks_count=123)
        self.audit_path.write_text(json.dumps(audit))

    def publish_comparison(self, path=None):
        return module.publish(self.study, path or Path(self.temp.name)/'display.json',
                              comparison=self.comparison, comparison_audit=self.audit_path)

    def test_final_comparison_preserves_old_panels_and_rescore(self):
        path = Path(self.temp.name)/'display.json'
        prior = dict(planner='hybrid', old={'exact':'value'})
        path.write_text(json.dumps(prior))
        before = module.compose(self.study)['rescore']
        section = self.publish_comparison(path)
        saved = module.read(path)
        self.assertEqual(saved['planner'], 'hybrid')
        self.assertEqual(saved['old'], prior['old'])
        self.assertEqual(section['rescore'], before)
        comparison = section['comparison']
        self.assertEqual([m['captures'] for m in comparison['methods']], [35,40])
        self.assertEqual([m['mean_planning_s'] for m in comparison['methods']], [2.,3.])
        self.assertIn('(+5)', comparison['conclusion'])
        self.assertIn('live planner is unchanged', comparison['conclusion'])
        self.assertTrue(comparison['frozen_before_test'])

    def test_partial_run_is_not_published(self):
        self.report['progress']['completed_releases'] = 96
        self.seal()
        with self.assertRaisesRegex(ValueError, 'Partial'):
            self.publish_comparison()

    def test_new_weights_cannot_be_substituted_even_with_resealed_audit(self):
        self.write('retrained-checkpoint.json', {'weights':[99]})
        self.seal()
        with self.assertRaisesRegex(ValueError, 'Frozen model/input'):
            self.publish_comparison()

    def test_native_start_must_bind_frozen_choices_and_weights(self):
        start = module.read(self.comparison/'native-started.json')
        start['bindings'].pop('original-checkpoint.json')
        self.write('native-started.json', start)
        self.seal()
        with self.assertRaisesRegex(ValueError, 'Native start'):
            self.publish_comparison()

    def test_stale_independent_audit_refuses_publication(self):
        self.report['phase'] = 'changed'
        self.write('report.json', self.report)
        with self.assertRaises(ValueError):
            self.publish_comparison()

    def test_duplicate_drops_cannot_inflate_capture_count(self):
        rows = [json.loads(line) for line in (self.comparison/'rows.jsonl').read_text().splitlines()]
        self.write_rows(rows + [rows[0]])
        self.seal()
        with self.assertRaisesRegex(ValueError, 'one native drop'):
            self.publish_comparison()

    def test_audited_summary_must_match_recorded_captures(self):
        self.report['comparison']['methods']['original']['captures'] = 36
        self.seal()
        with self.assertRaisesRegex(ValueError, 'totals differ'):
            self.publish_comparison()

    def test_cannot_publish_comparison_into_immutable_sources_or_audit(self):
        for path in (self.comparison/'new-display.json', self.audit_path):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, 'outside'):
                self.publish_comparison(path)

    def test_comparison_requires_path_and_audit_pair(self):
        with self.assertRaisesRegex(ValueError, 'together'):
            module.compose(self.study, comparison=self.comparison)
        with self.assertRaisesRegex(ValueError, 'together'):
            module.compose(self.study, comparison_audit=self.audit_path)

    def test_failed_new_scoring_control_blocks_final_comparison(self):
        self.report['controls']['controls_post']['board_first_passed'] = False
        self.seal()
        with self.assertRaisesRegex(ValueError, 'control gates'):
            self.publish_comparison()


if __name__ == '__main__':
    unittest.main()
