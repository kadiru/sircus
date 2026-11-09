"""Reuse the exact-action/option tamper suite with an explicit v3 factory."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from sircus.mjlab import table_training_native as native
from sircus.mjlab import table_assisted_score as score
from sircus.mjlab.tests.test_policy64_native import Adapter, EngineTests, source


class V3Adapter(Adapter):
    def evaluate(self, requests, phase):
        rows, timing = super().evaluate(requests, phase)
        for row in rows:
            if not row['native_release']:
                continue
            row.update(
                route_version=score.ROUTE_VERSION, reward_version=score.REWARD_VERSION,
                score_adapter_version=native.ADAPTER_VERSION, scalar_score_audit_passed=True,
                historical_v2_score=dict(scoring_version=score.CONTAINMENT_VERSION, route_success=False),
                board_bounce=True, closest_opening_distance_m=.2, forbidden_contact_steps=0,
                max_retention_s=0., single_board_bounce_success=False,
            )
            if self.mode != 'wrong_scorer':
                row['scoring_version'] = score.VERSION
                row['reward'] = score.reward(row)
            if self.mode == 'wrong_route': row['route_version'] = 'v2'
            if self.mode == 'wrong_reward_version': row['reward_version'] = 'v2'
            if self.mode == 'wrong_adapter': row['score_adapter_version'] = 'v2'
            if self.mode == 'wrong_reward': row['reward'] += 1
            if self.mode == 'nan_reward': row['reward'] = float('nan')
            if self.mode == 'missing_history': row.pop('historical_v2_score')
            if self.mode == 'missing_oracle': row['scalar_score_audit_passed'] = False
        return rows, timing


class TableEngineTests(EngineTests):
    def fixture(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        directory = Path(temp.name) / 'native'
        with patch.object(native, 'CalibratedAdapter', V3Adapter):
            return native.Engine(dict(scene={}), Path('unused'), directory,
                prior=dict(records=[source()]), calibration=dict(score_config={'verified':'calibration'}),
                home=dict(home={'verified':'home'}), batch=32)

    def test_v3_factory_is_lazy_explicit_and_receives_pinned64_original_bounds(self):
        with patch.object(native, 'make_environment') as make:
            engine = self.fixture()
            make.assert_not_called()
            engine.adapter.kwargs['environment_factory']()
            make.assert_called_once_with({}, Path('unused'), {'verified':'calibration'},
                n=32, bounds=native.WIDE_BOUNDS, substeps=64)

    def test_mixed_v2_v3_semantics_fail_closed(self):
        for mode in ('wrong_route','wrong_reward_version','wrong_adapter','wrong_reward',
                     'nan_reward','missing_history','missing_oracle'):
            e = self.fixture(); r = self.request(e); e.adapter.mode = mode
            with self.subTest(mode=mode), self.assertRaises(RuntimeError):
                e.evaluate([r], 'test')

    def test_native_v3_fields_overwrite_caller_labels_and_keep_historical_score(self):
        e = self.fixture(); r = self.request(e)
        r.update(route_success=True, reward=99, historical_v2_score={'forged':True})
        before = copy.deepcopy(r)
        rows, _ = e.evaluate([r], 'test')
        self.assertEqual(r, before)
        self.assertFalse(rows[0]['route_success'])
        self.assertEqual(rows[0]['scoring_version'], score.VERSION)
        self.assertEqual(rows[0]['training_engine_version'], native.VERSION)
        self.assertEqual(rows[0]['historical_v2_score']['scoring_version'], score.CONTAINMENT_VERSION)
        self.assertEqual(rows[0]['reward'], score.reward(rows[0]))


if __name__ == '__main__':
    unittest.main()
