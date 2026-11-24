"""Bucket approach geometry and forecast-boundary regressions (no GPU)."""
from copy import deepcopy
from types import SimpleNamespace
import unittest

import numpy as np

from sircus.mjlab.bucket_approach import BucketApproach, convex_signed_distance, _hull


def scene():
    return dict(ball=dict(radius_m=.0375), bucket=dict(segments=32,
        inner_bottom_diameter_m=.150, outer_bottom_diameter_m=.158,
        inner_top_diameter_m=.177, outer_top_diameter_m=.200,
        assumed_bottom_thickness_m=.008, assumed_interior_depth_m=.175))


def ranker():
    # member_flight only needs these geometric fields for the nominal rim band.
    rim=.183
    angles=np.arange(32)*2*np.pi/32+np.pi/32
    planes=np.column_stack((-np.cos(angles),-np.sin(angles),np.full(32,.0885*np.cos(np.pi/32))))
    return SimpleNamespace(gravity=np.array([0.,0.,-9.81]), coefficients=(.8,.8,.7,.8),
        table_center=np.array([0.,0.,-.025]),table_half=np.array([2.,2.,.025]),
        geom=dict(ball_radius_m=.0375,rim_m=rim,gravity_m_s2=[0.,0.,-9.81], inward_rim_halfplanes=planes.tolist()))


class BucketApproachTests(unittest.TestCase):
    def setUp(self):
        self.checker=BucketApproach(scene())

    def test_open_top_and_empty_interior_are_not_solid_hull(self):
        distances=self.checker.distances([[0,0,.25],[0,0,.11]], [0,0])
        self.assertGreater(distances.min(),0)
        self.assertLess(self.checker.distances([[.09,0,.15]], [0,0]).min(),0)

    def test_base_collision_and_translated_bucket(self):
        self.assertLess(self.checker.distances([[0,0,.03]], [0,0]).min(),0)
        points=np.array([[.09,0,.15],[0,0,.11]])
        np.testing.assert_allclose(self.checker.distances(points,[0,0]),
            self.checker.distances(points+[1.05,.025,0],[1.05,.025]),atol=1e-14)

    def test_exact_convex_distance_keeps_clear_corner(self):
        cube=_hull('cube',[[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])
        d=convex_signed_distance(np.array([[1.1,1.1,0],[0,0,0],[1,0,0]]),cube)
        np.testing.assert_allclose(d,[np.sqrt(.02),-1,0],atol=1e-12)
        # Inflating each face by .12 would falsely overlap this corner.
        self.assertGreater(d[0]-.12,0)

    def test_rising_wall_hit_is_seen_before_later_descending_rim(self):
        flight=self.checker.assess_flight([-.3,0,.0375],[1.8,0,1.5],.25,[0,0])
        self.assertEqual(flight['status'],'sampled_overlap')
        self.assertIsNotNone(flight['first_predicted_contact'])
        self.assertTrue(flight['first_predicted_contact']['rising'])
        self.assertTrue(flight['first_predicted_contact']['geom'].startswith('bucket/wall'))

    def test_center_drop_stops_at_full_entry_without_penalizing_floor(self):
        flight=self.checker.assess_flight([0,0,.4],[0,0,0],.22,[0,0])
        self.assertEqual(flight['status'],'clear')
        later=self.checker.assess_flight([0,0,.4],[0,0,0],.28,[0,0])
        self.assertEqual(later['status'],'sampled_overlap')
        self.assertEqual(later['first_predicted_contact']['geom'],'bucket/base')

    def test_between_sample_bound_marks_unproven_not_contact(self):
        checker=BucketApproach(scene(),step_s=.01)
        # Both sampled sphere centers are just clear of rim height; speed bound
        # conservatively covers the interval without pretending contact occurred.
        row=checker.assess_flight([.11,0,.231],[0,0,-.1],.01,[0,0])
        self.assertIsNone(row['first_predicted_contact'])
        self.assertLess(row['minimum_clearance_lower_bound_m'],row['minimum_sampled_clearance_m'])

    def test_full_route_marks_warning_but_never_forbids_candidate_or_capture(self):
        r=ranker()
        prediction=dict(board_launch_m=[-.3,0,.0375],board_velocity_m_s=[1,0,2.2],
            direct=dict(rim_clearance_m=.02,arm_clearance_m=-.08),table=None,
            contact_ranking_version='permitted-contacts',score_m=.02)
        request=dict(values=[-.3,0,0,28,.4],bucket_xy_m=[0,0])
        original=deepcopy(prediction)
        result=self.checker.assess(request,r,prediction)
        self.assertEqual(result['routes']['direct']['status'],'sampled_overlap')
        self.assertLess(result['score_m'],0)
        self.assertIsNotNone(result['score_m'])
        self.assertFalse(result['rejects_candidates'])
        self.assertFalse(result['modifies_native_scoring'])
        self.assertEqual(prediction,original)
        self.assertEqual(result['routes']['direct']['original_score_m'],.02)

    def test_missing_complete_rim_horizon_remains_unknown(self):
        r=ranker()
        prediction=dict(board_launch_m=[-.3,0,.04],board_velocity_m_s=[1,0,.1],
            direct=dict(rim_clearance_m=None,arm_clearance_m=None),table=None)
        request=dict(values=[-.3,0,0,28,.4],bucket_xy_m=[0,0])
        result=self.checker.assess(request,r,prediction)
        self.assertIsNone(result['score_m'])
        self.assertEqual(result['routes']['direct']['status'],'unsupported_horizon')

    def test_incoming_horizon_cannot_be_silently_omitted(self):
        r=ranker()
        prediction=dict(board_launch_m=[-.3,0,.04],board_velocity_m_s=[1,0,2.2],
            direct=dict(rim_clearance_m=.02,arm_clearance_m=.03),table=None)
        request=dict(values=[-.3,0,0,28,20.],bucket_xy_m=[0,0])
        result=self.checker.assess(request,r,prediction)
        self.assertIsNone(result['score_m'])
        self.assertEqual(result['missing_reason'],'unsupported_complete_incoming_horizon')

    def test_endpoint_time_offset_and_unchanged_rim_graze_score(self):
        r=ranker()
        prediction=dict(board_launch_m=[-.3,0,.0375],board_velocity_m_s=[1.8,0,1.5],
            direct=dict(rim_clearance_m=-.03,arm_clearance_m=-.08),table=None,
            contact_ranking_version='permitted-contacts',score_m=-.03)
        request=dict(values=[-.3,0,0,28,.4],bucket_xy_m=[0,0])
        before=deepcopy(prediction)
        result=self.checker.assess(request,r,prediction)
        # No model outcome or native label can enter this input/output API;
        # a warned rim-assisted shot remains an eligible candidate.
        self.assertFalse(result['rejects_candidates'])
        self.assertEqual(prediction,before)
        seg=self.checker.assess_flight([-.3,0,.0375],[1.8,0,1.5],.25,[0,0],start_time_s=.7)
        contact=seg['first_predicted_contact']
        self.assertAlmostEqual(contact['time_s'],contact['segment_time_s']+.7)
        self.assertTrue(contact['rising'])

    def test_compiled_geometry_matches_native_distance_and_has_translation_independent_hash(self):
        try:
            import mujoco as mj
        except ImportError:
            self.skipTest('MuJoCo unavailable')
        from sircus.isaac.geometry import bucket_wedge
        cfg=scene()
        meshes=''.join('<mesh name="w%d" vertex="%s"/>' % (i,' '.join(map(str,np.array(bucket_wedge(cfg['bucket'],i)).ravel()))) for i in range(32))
        geoms=''.join('<geom name="bucket/wall%02d" type="mesh" mesh="w%d"/>' % (i,i) for i in range(32))
        xml=f'<mujoco><asset>{meshes}</asset><worldbody><body name="bucket" mocap="true">{geoms}<geom name="bucket/base" type="cylinder" size=".079 .004" pos="0 0 .004"/></body><body name="ball"><freejoint/><geom name="ball/geom" type="sphere" size=".0375" mass=".1"/></body></worldbody></mujoco>'
        model=mj.MjModel.from_xml_string(xml); data=mj.MjData(model)
        mj.mj_forward(model,data)
        gate=SimpleNamespace(m=model,d=data,mj=mj,ball_geom=model.geom('ball/geom').id)
        r=SimpleNamespace(clearance=gate)
        first=BucketApproach(cfg);first._bind_compiled(r,[0,0])
        # Native CPU distance agrees with exact convex-hull/cylinder distances.
        point=np.array([.145,.015,.14]);data.qpos[:3]=point;mj.mj_forward(model,data)
        native=min(mj.mj_geomDistance(model,data,gate.ball_geom,model.geom(name).id,10.,None)
                   for name in [w['name'] for w in first.walls]+['bucket/base'])
        self.assertAlmostEqual(float(first.distances([point],[0,0]).min()),native,places=6)
        data.mocap_pos[0]=[1.1,-.44,0];mj.mj_forward(model,data)
        second=BucketApproach(cfg);second._bind_compiled(r,[1.1,-.44])
        self.assertEqual(first.geometry_sha256,second.geometry_sha256)
        self.assertEqual(second.geometry_source,'compiled_mujoco_convex_mesh_vertices_and_cylinder')

    def test_invalid_horizons_and_geometry_fail(self):
        for h in (0,-1,1.5001,float('nan')):
            with self.assertRaises(ValueError):self.checker.assess_flight([0,0,1],[1,0,0],h,[0,0])
        for spacing in (0,.011,float('inf')):
            with self.assertRaises(ValueError):BucketApproach(scene(),step_s=spacing)


if __name__=='__main__':unittest.main()
