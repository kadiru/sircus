"""Recorded-flight diagnostics preserve evidence, missing values and scope."""
import copy
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from sircus.mjlab import ensemble_flight_diagnosis as diagnosis


def observed(**changes):
    value = dict(native_release=True, capture=False, whole_entry=False,
                 retention_s=0., invalid_physics=False, board_bounce=True)
    value.update(changes)
    return value


def nominal_record():
    return dict(id='ensemble-test-29', context_id='physics-ensemble-test-29',
        method='ensemble', values=[.5, -.2, .3, 30., .25],
        bucket_xy_m=[.9, -.2], native_release=True, route_success=False,
        whole_ball_entry=False, max_retention_s=0., invalid_physics=False,
        board_bounce=True, pose_gate=dict(board_position_error_mm=.2),
        max_board_drift_mm=.05, bucket_collider_error_m=1e-8,
        model_prediction=dict(capture_probability=.99, uncertainty=.001,
            aperture_distance_m=.01, rebound_velocity_m_s=[1., 0., .8],
            ranking_score=.989, member_capture_probability=[.989, .99, .991]))


def training_record(ident, features, *, capture=False, mask=True):
    return dict(id=ident, x=list(features), y=[0., 0., 0., float(capture)],
                mask=[1., 1., 1., float(mask)], role='train')


class EnsembleFlightDiagnosisTests(unittest.TestCase):
    def test_conditional_ballistic_formula_and_immutable_inputs(self):
        origin=[.1, .2, .4];velocity=[1., -.2, 2.];times=[0., .1, .2]
        gravity=[0., 0., -9.81];before=copy.deepcopy((origin,velocity,times,gravity))
        result=diagnosis.ballistic(origin,velocity,times,gravity)
        np.testing.assert_allclose(result,[[.1,.2,.4],[.2,.18,.55095],[.3,.16,.6038]],rtol=0,atol=1e-12)
        self.assertEqual((origin,velocity,times,gravity),before)
        # This is contact-free conditioning, not a capture decision.
        self.assertIsInstance(result,list)

    def test_ballistic_rejects_nonfinite_negative_duplicate_or_missing_times(self):
        args=([0.,0.,0.],[0.,0.,1.],[0.,.1],[0.,0.,-9.81])
        edits=[(0,[0.,0.]),(1,[0.,float('nan'),1.]),(2,[-.1,0.]),
               (2,[0.]),(2,[0.,0.]),(2,[.1,0.]),(2,[0.,float('inf')]),
               (2,[[0.,.1]]),(3,[0.,0.,float('-inf')])]
        for index,value in edits:
            changed=list(args);changed[index]=value
            with self.subTest(index=index,value=value),self.assertRaises(ValueError):
                diagnosis.ballistic(*changed)

    def test_score_bins_include_rejections_and_boundary_scores_once(self):
        scores=[0.,.499,.5,.899,.9,.989,.99,1.]
        rows=[dict(prediction=dict(capture_score=p),observed=observed()) for p in scores]
        rows[1]['observed'].update(native_release=False)
        rows[4]['observed'].update(capture=True,whole_entry=True,retention_s=1.)
        before=copy.deepcopy(rows);result=diagnosis.bins(rows)
        self.assertEqual(rows,before)
        self.assertEqual([r['requested'] for r in result],[2,2,2,2])
        self.assertEqual(sum(r['requested'] for r in result),len(rows))
        self.assertEqual(result[0]['rejected'],1)
        self.assertEqual(result[0]['native_misses'],1)
        self.assertEqual(result[2]['captures'],1)
        self.assertEqual(result[-1]['high'],1.)
        for row in result:
            self.assertEqual(row['requested'],row['captures']+row['native_misses']+row['rejected'])

    def test_empty_score_bins_remain_missing_instead_of_zero_confidence(self):
        result=diagnosis.bins([])
        self.assertEqual(len(result),4)
        self.assertTrue(all(r['requested']==0 and r['mean_score'] is None for r in result))

    def test_example_selection_is_deterministic_and_discloses_curated_cases(self):
        rows=[]
        for ident,score,suffix in [('a',.99,'29'),('b',.99,'49'),('c',.98,'11'),
                                   ('d',.96,'36'),('e',.95,'35'),('f',.94,'40')]:
            rows.append(dict(id=ident,prediction=dict(capture_score=score),
                             context_id='physics-ensemble-test-'+suffix))
        before=copy.deepcopy(rows)
        self.assertEqual(diagnosis.selected_ids(rows),['a','b','c','d','e','f'])
        self.assertEqual(diagnosis.selected_ids(list(reversed(rows))),['a','b','c','d','e','f'])
        self.assertEqual(rows,before)
        # A curated diagnostic case already among top three must not duplicate.
        rows[3]['prediction']['capture_score']=1.
        result=diagnosis.selected_ids(rows)
        self.assertEqual(result.count('d'),1)

    def test_classification_keeps_setup_entry_capture_and_unknown_crossing_distinct(self):
        self.assertIn('no native ball flight',diagnosis.classification(observed(native_release=False)))
        self.assertIn('one-second capture observed',diagnosis.classification(observed(capture=True,whole_entry=True,retention_s=1.)))
        self.assertEqual(diagnosis.classification(observed(whole_entry=True,retention_s=.7)),
                         'Entry observed but capture failed')
        contact=diagnosis.classification(observed(cutoff_contact_channels=['bucket']))
        self.assertIn('bucket contact before an observed',contact)
        self.assertNotIn('missed by',contact)
        self.assertIn('no contact-limited descending rim crossing observed',
                      diagnosis.classification(observed()))

    def test_signed_center_crossing_margin_does_not_promote_capture(self):
        for margin,rendered in [(-.012,'-12.0 mm'),(.002,'2.0 mm')]:
            result=diagnosis.classification(observed(rim_crossing=dict(time_s=.2),mouth_clearance_m=margin))
            self.assertIn('No whole entry',result)
            self.assertIn('center-at-rim',result)
            self.assertIn(rendered,result)
            self.assertNotIn('capture observed',result)

    def test_training_support_masks_nearest_capture_and_uses_fixed_normalization(self):
        row=nominal_record();x=(np.r_[row['values'],row['bucket_xy_m']]-[.67,-.2,.35,30.,.3,.67,-.2])/[.57,.45,.3,30.,.3,.57,.45]
        train=[training_record('nearest-invalid',x,capture=True,mask=False),
               training_record('z-capture',x+.1,capture=True),
               training_record('a-capture',x+.1,capture=True)]
        before=copy.deepcopy((row,train));result=diagnosis.support(row,train)
        self.assertEqual(result['nearest_train_id'],'nearest-invalid')
        self.assertFalse(result['nearest_train_capture'])
        self.assertEqual(result['nearest_capture_train_id'],'a-capture')
        self.assertAlmostEqual(result['nearest_capture_distance'],np.sqrt(7)*.1)
        self.assertEqual(result['outside_training_feature_range'],[])
        self.assertIn('not a reliability bound',result['scope'])
        self.assertEqual((row,train),before)

    def test_training_feature_range_is_descriptive_not_an_outcome(self):
        row=nominal_record();x=(np.r_[row['values'],row['bucket_xy_m']]-[.67,-.2,.35,30.,.3,.67,-.2])/[.57,.45,.3,30.,.3,.57,.45]
        train=[training_record('only-train',x-.1,capture=True)]
        result=diagnosis.support(row,train)
        self.assertEqual(result['outside_training_feature_range'],list(range(7)))
        self.assertNotIn('capture',result)

    def test_compact_row_preserves_observation_provenance_and_no_raw_robot_state(self):
        row=nominal_record();row.update(q10=[1.]*6,q14=[2.]*6,trajectory_m=[[0.,0.,0.],[1.,1.,1.]])
        x=(np.r_[row['values'],row['bucket_xy_m']]-[.67,-.2,.35,30.,.3,.67,-.2])/[.57,.45,.3,30.,.3,.57,.45]
        train=[training_record('teacher',x,capture=True)];before=copy.deepcopy(row)
        result=diagnosis.make_row(row,train)
        self.assertEqual(result['record_sha256'],diagnosis.digest(row))
        self.assertEqual(result['record_id'],row['id'])
        self.assertEqual(result['prediction']['capture_score'],.99)
        self.assertFalse(result['observed']['capture'])
        self.assertFalse(result['observed']['whole_entry'])
        self.assertEqual(result['observed']['pose_gate'],row['pose_gate'])
        self.assertAlmostEqual(result['board_bucket_distance_m'],.4)
        self.assertTrue(all(k not in result for k in ('q10','q14','trajectory_m')))
        result['values'][0]=999;result['observed']['pose_gate']['board_position_error_mm']=999
        self.assertEqual(row,before)

    def test_build_refuses_existing_or_nested_output_before_loading_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);study=root/'study';study.mkdir();audit=root/'proof.json'
            for output in (study/'derived',root):
                with patch.object(diagnosis,'verified_inputs') as verify:
                    with self.subTest(output=output),self.assertRaisesRegex(ValueError,'outside immutable evidence'):
                        diagnosis.build(study,audit,output)
                    verify.assert_not_called()
            self.assertEqual(list(study.iterdir()),[])

    def test_module_imports_without_torch_simulator_or_cuda(self):
        script=('import sys; from sircus.mjlab import ensemble_flight_diagnosis; '
                'assert not any(x in sys.modules for x in ("torch","warp","mujoco","mjlab"))')
        completed=subprocess.run([sys.executable,'-c',script],capture_output=True,text=True)
        self.assertEqual(completed.returncode,0,completed.stderr)


if __name__=='__main__':unittest.main()
