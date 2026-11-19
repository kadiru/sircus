"""Failure summaries preserve denominators and measured contact provenance."""

import json
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

SOURCE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SOURCE))
import continuous_failure_analysis as fa


class FailureAnalysisTests(unittest.TestCase):
    def row(self,radius=.4,invalid=None,capture=False):
        return dict(invalid=invalid,route_success=capture,route_quality=1 if capture else 0,
                    success=capture,board_bounce=invalid is None,whole_ball_entry=capture,
                    radius_m=radius,bucket_reference_distance_m=.5)

    def test_all_invalid_requests_still_count_in_capture_denominator(self):
        rows=[self.row(capture=True),self.row(invalid='endpoint_ik'),self.row(invalid='endpoint_ik')]
        result=fa.statistics(rows)
        self.assertEqual(result['attempts'],3)
        self.assertEqual(result['released'],1)
        self.assertAlmostEqual(result['capture_rate'],1/3)
        self.assertEqual(result['capture_rate_given_release'],1)
        self.assertEqual(result['invalid_reasons'],{'endpoint_ik':2})

    def test_bin_edges_keep_each_request_exactly_once(self):
        rows=[self.row(radius=r) for r in fa.RADIUS_EDGES]
        result=fa.binned(rows,'radius_m',fa.RADIUS_EDGES)
        self.assertEqual(sum(x['attempts'] for x in result),len(rows))
        self.assertEqual(result[0]['attempts'],1)
        self.assertEqual(result[-1]['attempts'],2)

    def test_first_and_peak_external_surfaces_are_different_measured_facts(self):
        forces=np.zeros((5,6));forces[1,2]=2.;forces[3,4]=8.
        positions=np.zeros((5,3));positions[1]=[.2,.1,.04]
        result=fa.surface_events(forces,positions,[0,0],[.5,0],5)
        self.assertEqual(result['first_surfaces'],['table'])
        self.assertEqual(result['peak_surfaces'],['robot'])
        self.assertEqual(result['first_sample'],1)
        self.assertIn('ball center',result['position_provenance'])
        self.assertAlmostEqual(result['goal_axis_offset_mm'],-300)
        self.assertAlmostEqual(result['goal_lateral_offset_mm'],100)
        self.assertNotIn('short',result)
        self.assertNotIn('long',result)

    def test_simultaneous_surface_ties_are_kept(self):
        forces=np.zeros((3,6));forces[1,2:4]=3.
        result=fa.surface_events(forces,np.zeros((3,3)),[0,0],[.5,0],3)
        self.assertEqual(result['first_surfaces'],['table','floor'])
        self.assertEqual(result['peak_surfaces'],['table','floor'])

    def test_post_score_floor_contact_does_not_become_a_failure_cause(self):
        forces=np.zeros((5,6));forces[4,3]=100.
        result=fa.surface_events(forces,np.zeros((5,3)),[0,0],[.5,0],3)
        self.assertEqual(result['first_surfaces'],[])
        self.assertEqual(result['peak_surfaces'],[])
        self.assertEqual(result['peak_external_force_N'],0)

    def test_prior_bucket_contact_is_not_a_short_or_long_diagnosis(self):
        forces=np.zeros((5,6));forces[1,1]=3.;forces[3,2]=10.
        result=fa.surface_events(forces,np.zeros((5,3)),[0,0],[.5,0],5)
        self.assertTrue(result['bucket_contact_before_first_external'])
        forces[1,1]=0;forces[3,1]=3.
        result=fa.surface_events(forces,np.zeros((5,3)),[0,0],[.5,0],5)
        self.assertFalse(result['bucket_contact_before_first_external'])

    def test_robot_board_contact_order_is_measured_without_gripper_attribution(self):
        for robot,board,label in ((0,2,'before_first_board_contact'),(2,2,'same_sample_as_first_board_contact'),(3,1,'after_first_board_contact')):
            forces=np.zeros((5,6));forces[robot,4]=4.;forces[board,0]=2.
            result=fa.surface_events(forces,np.zeros((5,3)),[0,0],[.5,0],5)
            self.assertEqual(result['first_robot_relative_to_board'],label)
            self.assertEqual(result['first_board_contact_sample'],board)
            self.assertEqual(result['first_robot_contact_sample'],robot)
            self.assertNotIn('gripper',result)
        forces=np.zeros((5,6));forces[1,4]=2.
        result=fa.surface_events(forces,np.zeros((5,3)),[0,0],[.5,0],5)
        self.assertEqual(result['first_robot_relative_to_board'],'no_board_contact_observed')

    def test_retained_ball_with_early_robot_contact_remains_a_route_failure(self):
        early=np.zeros((5,6));early[0,4]=3.;early[2,0]=1.
        missed=self.row(capture=False);missed.update(success=True,whole_ball_entry=True)
        good=self.row(capture=True)
        pose=dict(board_center_error_mm=.2,board_normal_error_deg=.1,ball_first_sample_error_from_initialized_release_mm=.17)
        for row,forces in ((missed,early),(good,np.zeros((5,6)))):
            row['observed_pose']=pose
            row['external_contact']=fa.surface_events(forces,np.zeros((5,3)),[0,0],[.5,0],5)
        result=fa.summarize_contacts([missed,good])
        self.assertEqual(result['raw_retained_route_failures'],1)
        self.assertEqual(result['raw_retained_route_failure_first_external_surface'],{'robot':1})
        self.assertEqual(result['raw_retained_route_failure_robot_timing'],{'before_first_board_contact':1})
        stats=fa.statistics([missed,good])
        self.assertEqual(stats['raw_retained_captures'],2)
        self.assertEqual(stats['captures'],1)
        self.assertEqual(stats['capture_rate'],.5)

    def test_requested_actual_pose_keeps_native_body_and_initialized_ball_distinct(self):
        scene=json.loads((SOURCE/'scene.json').read_text())
        cell=dict(bucket_xy_m=[.9,0]);row=dict(board_x_m=.4,board_y_m=0.,height_m=.3,gap_m=.2,tilt_deg=30)
        # Desired normal points along +X and upward. Construct a real rotation.
        rotation=Rotation.from_euler('y',30,degrees=True)
        q=rotation.as_quat()[[3,0,1,2]]
        origin=np.array([2.5,0,0]);board=np.array([.401,0,.3])+origin
        release=np.array([.4,0,.3+.2+.0375])+origin
        result=fa.observed_pose(scene,cell,row,np.array([release]),np.array([board]),np.array([q]),origin)
        self.assertAlmostEqual(result['board_center_error_mm'],1)
        self.assertLess(result['board_normal_error_deg'],1e-5)
        self.assertAlmostEqual(result['ball_first_sample_error_from_initialized_release_mm'],0)
        self.assertIn('no grasp',result['scope'])

    def test_invalid_contact_values_rejected(self):
        for value in (-1.,float('nan')):
            forces=np.zeros((3,6));forces[0,0]=value
            with self.assertRaises(ValueError):fa.surface_events(forces,np.zeros((3,3)),[0,0],[1,0],3)


if __name__=='__main__':unittest.main()
