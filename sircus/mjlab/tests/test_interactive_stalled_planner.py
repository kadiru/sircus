"""Synthetic bridge/provenance checks; no native capture evidence or GPU use."""
import copy
from contextlib import ExitStack, nullcontext
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from sircus.mjlab import interactive_stalled_planner as live
from sircus.mjlab.benchmark_verify import sha
from sircus.mjlab.fast_planning_study import COPIED_INPUTS
from sircus.mjlab.stalled_reference_audit import VERSION as AUDIT_VERSION
from sircus.mjlab.stalled_reference_study import promotion_decision


class LiveStalledTests(unittest.TestCase):
    def fixture(self, root):
        classic = SimpleNamespace(_native={"ref": {"scene": {}}, "home": {}},
            provenance={"scene_sha256": "scene", "flight_rule_sha256": "rule"})
        fast, study = root/'fast', root/'study'
        fast.mkdir(); study.mkdir()
        for name in COPIED_INPUTS:
            (fast/name).write_text('{}'); (study/name).write_text('{}')
        (study/'setup-home.json').write_text('{}')
        comparison = {"methods": {"baseline": {"captures": 15}, "hybrid": {"captures": 16}},
            "latency": {"baseline": {"mean_s": 10}, "hybrid": {"mean_s": 10.5}}}
        diagnostic = {"methods": {"hybrid": {"outcome": "table"}}}
        report = {"comparison": comparison, "diagnostic": diagnostic,
            "decision": promotion_decision(comparison, diagnostic),
            "budgets": {"positions": 20, "planning_s": 30., "reference_path_s": 5.}}
        contract = {"methods": ["baseline", "hybrid"], "test_positions": 20,
            "planning_budget_s": 30., "reference_path_s": 5., "promotion_time_ratio": 1.10,
            "early_score_m": .005, "attempts_per_method_per_position": 1,
            "source_sha256": {}, "frozen_inputs": {}, "scene": {},
            "identity": {"scene_sha256": "scene"}, "flight_ranking": {"rule_sha256": "rule"}, "fingerprint": "contract"}
        audit = {"version": AUDIT_VERSION, "comparison": comparison}
        artifact = {"report.json": "report", "audit": "audit", "choices.json": "choices"}
        args = SimpleNamespace(fast_study=fast, stalled_study=study, stalled_audit=root/'audit')
        return args, classic, report, audit, artifact, contract

    def test_gate_requires_passed_bound_audit_and_predeclared_thresholds(self):
        with TemporaryDirectory() as temp:
            args, classic, report, audit, artifacts, contract = self.fixture(Path(temp))
            with patch.object(live, 'verified_inputs', return_value=((1, 2), {})), patch(
                    'sircus.mjlab.intergrid_route_refit_inputs._audited_source',
                    return_value=(report, {}, audit, artifacts, contract)):
                common, binding = live.verified_stalled_inputs(args, classic)
                self.assertEqual(common, (1, 2))
                self.assertTrue(binding['promotion_decision']['eligible'])
                for key, wrong in [('planning_budget_s', 60), ('reference_path_s', 10),
                                   ('test_positions', 10), ('promotion_time_ratio', 2), ('early_score_m', 0)]:
                    previous = contract[key]; contract[key] = wrong
                    with self.assertRaises(ValueError): live.verified_stalled_inputs(args, classic)
                    contract[key] = previous
                report['diagnostic']['methods']['hybrid']['outcome'] = 'flight_miss'
                report['decision'] = promotion_decision(report['comparison'], report['diagnostic'])
                with self.assertRaises(ValueError): live.verified_stalled_inputs(args, classic)

    def test_frozen_input_change_prevents_enablement(self):
        with TemporaryDirectory() as temp:
            args, classic, report, audit, artifacts, contract = self.fixture(Path(temp))
            with patch.object(live, 'verified_inputs', return_value=((1,), {})), patch(
                    'sircus.mjlab.intergrid_route_refit_inputs._audited_source',
                    return_value=(report, {}, audit, artifacts, contract)):
                (args.stalled_study/COPIED_INPUTS[0]).write_text('{"changed": true}')
                with self.assertRaisesRegex(ValueError, 'input differs'):
                    live.verified_stalled_inputs(args, classic)

    def test_scoped_live_wrapper_binds_before_calling_same_bounded_selector(self):
        with patch('sircus.mjlab.stalled_reference_plan.bind_live_request') as bind, patch(
                'sircus.mjlab.stalled_reference_plan.stalled_reference_plan', return_value='done') as plan:
            layout = {"id": "new-request", "bucket_xy_m": [.6, -.4]}
            self.assertEqual(live._isolated_plan(layout, 'warm_early', Path('/tmp/example'), 'input', deadline=30), 'done')
            bind.assert_called_once_with(Path('/tmp/example/requests/new-request'))
            plan.assert_called_once_with(layout, 'warm_early', 'input', deadline=30)
            with self.assertRaises(ValueError):
                live._isolated_plan({**layout, "id": '../escape'}, 'warm_early', Path('/tmp/example'))

    def test_copy_proofs_preserves_nested_reference_fallback_and_hashes(self):
        with TemporaryDirectory() as temp:
            source, target = Path(temp)/'source', Path(temp)/'target'
            source.mkdir()
            fallback = source/'resolutions/fallback.json'; fallback.parent.mkdir(); fallback.write_text('{}')
            context = source/'direct-contexts/context.json'; context.parent.mkdir(); context.write_text('{}')
            connection = source/'direct-connections/shot.json'; connection.parent.mkdir()
            connection.write_text(json.dumps({"fallback_resolution": {"file": 'resolutions/fallback.json', "sha256": sha(fallback)}}))
            request = {"context_resolution_file": 'direct-contexts/context.json', "context_resolution_sha256": sha(context),
                       "prepared_connection_file": 'direct-connections/shot.json', "prepared_connection_sha256": sha(connection)}
            live.copy_prepared_proofs(source, target, request)
            self.assertEqual((target/'resolutions/fallback.json').read_bytes(), fallback.read_bytes())
            fallback.write_text('{"changed": true}')
            with self.assertRaises(ValueError):
                live.copy_prepared_proofs(source, Path(temp)/'new-target', request)

    def test_native_release_dispatches_saved_proof_without_replanning(self):
        for mode, context in [('reference', 'resolutions/context.json'), ('direct_first', 'direct-contexts/context.json')]:
            with self.subTest(mode=mode), TemporaryDirectory() as temp, ExitStack() as stack:
                directory = Path(temp)
                request = {"id": "one", "bucket_xy_m": [.6, -.4], "values": [.5, -.2, .3, 30, .2],
                           "context_resolution_file": context, "context_resolution_sha256": "resolution"}
                path = directory/'request.json'; path.write_text(json.dumps(request))
                row = {**request, "native_release": True, "max_retention_s": 1., "native_cache_sha256": "cache",
                       "scoring_version": "version", "scalar_score_audit_passed": True}
                fake = SimpleNamespace(adapter=SimpleNamespace(_environment=lambda timing: SimpleNamespace(geometry='same')),
                                       evaluate=lambda requests, phase: ([row], {}))
                reference = stack.enter_context(patch('sircus.mjlab.cell1050_heading_native.HeadingEngine', return_value=fake))
                direct = stack.enter_context(patch('sircus.mjlab.direct_path_first.DirectPathEngine', return_value=fake))
                hydrate = stack.enter_context(patch('sircus.mjlab.direct_path_first.hydrate_direct_resolution', return_value={"sha256": "resolution"}))
                load = stack.enter_context(patch('sircus.mjlab.parallel_capture_grid_workers.load_resolution', return_value={"sha256": "resolution"}))
                for target, result in [('gpu_slot.foreign_processes', []), ('intergrid_system_study.require_storage', None),
                    ('cell1050_rebound_study.annotate', None), ('intergrid_route_labels.annotate_routes', None),
                    ('intergrid_matched_route_study.outcome', 'direct')]:
                    stack.enter_context(patch('sircus.mjlab.'+target, return_value=result))
                stack.enter_context(patch('sircus.mjlab.gpu_slot.gpu_slot', return_value=nullcontext()))
                stack.enter_context(patch.object(live, '_trace', return_value=[]))
                result=live._hybrid_native_drop({}, {"geometry": "same"}, directory, sha(path), 'lease', 1, lambda e: None)
                self.assertTrue(result['capture'])
                self.assertEqual(direct.call_count, int(mode == 'direct_first'))
                self.assertEqual(hydrate.call_count, int(mode == 'direct_first'))
                self.assertEqual(reference.call_count, int(mode == 'reference'))
                self.assertEqual(load.call_count, int(mode == 'reference'))
                path.write_text('{}')
                with self.assertRaisesRegex(ValueError, 'request changed'):
                    live._hybrid_native_drop({}, {}, directory, 'old-sha', 'lease', 1, lambda e: None)


if __name__ == '__main__': unittest.main()
