"""Synthetic first-flight controls; no simulator or GPU is initialized."""

import copy
import json
import math
from pathlib import Path
import unittest

import numpy as np

from sircus.mjlab import calibrated_score
from sircus.mjlab.tests.test_calibrated_score import config, resign
from sircus.mjlab.xy650_metrics import CROSSING_PLANE, VERSION, first_flight


ROOT = Path(__file__).resolve().parents[3]


class FirstFlightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene = json.loads((ROOT/'sircus/isaac/scene.json').read_text())
        cls.cfg = config(cls.scene)
        cls.xy = np.asarray(cls.scene['bucket']['center_xy_m'])
        cls.rim = cls.scene['bucket']['assumed_bottom_thickness_m'] + cls.scene['bucket']['assumed_interior_depth_m']
        cls.radius = cls.scene['ball']['radius_m']

    def trace(self, z=None, *, x=0.):
        z = np.asarray([.30,.27,.25,.24,.21,.17,.13] if z is None else z, dtype=float)
        p = np.tile([*self.xy,0.], (len(z),1)); p[:,0] += x; p[:,2] = z
        v = np.zeros_like(p); v[2,2] = .1
        f = np.zeros((len(z)-1,6)); f[0,0] = 1.
        return dict(position_m=p, velocity_m_s=v, forces_N=f, dt_s=.01)

    def metric(self, trace, **kw):
        return first_flight(self.scene,self.xy,[0,0,1],self.cfg,trace,**kw)

    def test_isolated_rebound_interpolates_center_crossing_and_facet_clearance(self):
        arrays=self.trace();before=copy.deepcopy(arrays);out=self.metric(arrays)
        self.assertEqual(out['version'],VERSION);self.assertEqual(out['crossing_plane'],CROSSING_PLANE)
        self.assertTrue(out['available']);self.assertIsNone(out['reason'])
        self.assertEqual((out['start_index'],out['end_index'],out['sample_count']),(2,6,5))
        self.assertEqual(out['cutoff_reason'],'horizon');self.assertEqual(out['cutoff_contact_channels'],[])
        crossing=out['first_crossing'];self.assertEqual(crossing['interval_index'],4)
        fraction=(.21-self.rim)/(.21-.17)
        self.assertAlmostEqual(crossing['fraction'],fraction)
        self.assertAlmostEqual(crossing['time_s'],.04+.01*fraction)
        self.assertEqual(crossing['xy_m'],self.xy.tolist())
        expected=self.scene['bucket']['inner_top_diameter_m']/2*math.cos(math.pi/self.scene['bucket']['segments'])-self.radius
        self.assertAlmostEqual(out['mouth_clearance_m'],expected,places=12)
        self.assertAlmostEqual(out['closest_opening_distance_m'],abs(.17-self.rim))
        for key in arrays:np.testing.assert_array_equal(arrays[key],before[key])
        self.assertNotIn('position_m',out);self.assertNotIn('success',out)
        json.dumps(out,allow_nan=False)

    def test_first_floor_contact_excludes_post_contact_states_and_roll(self):
        arrays=self.trace();arrays['forces_N'][4,3]=1.
        arrays['position_m'][5:]=[*self.xy,self.rim]
        out=self.metric(arrays)
        self.assertEqual((out['end_index'],out['sample_count']),(4,3))
        self.assertEqual(out['cutoff_contact_channels'],['floor'])
        self.assertEqual(out['cutoff_reason'],'contact');self.assertIsNone(out['first_crossing'])
        self.assertIsNone(out['mouth_clearance_m'])
        self.assertAlmostEqual(out['closest_opening_distance_m'],.21-self.rim)

    def test_rim_contact_prevents_later_bucket_roll_from_improving_metrics(self):
        arrays=self.trace();arrays['forces_N'][3,1]=2.
        arrays['position_m'][4:]=[*self.xy,self.rim]
        out=self.metric(arrays)
        self.assertEqual(out['end_index'],3);self.assertEqual(out['cutoff_contact_channels'],['bucket'])
        self.assertIsNone(out['first_crossing'])
        self.assertAlmostEqual(out['closest_opening_distance_m'],.24-self.rim)

    def test_every_contact_channel_and_exact_force_threshold(self):
        for i,name in enumerate(calibrated_score.FORCE_NAMES):
            arrays=self.trace();arrays['forces_N'][4,i]=.0100001
            with self.subTest(name=name):
                out=self.metric(arrays);self.assertEqual(out['end_index'],4)
                self.assertEqual(out['cutoff_contact_channels'],[name])
            arrays['forces_N'][4,i]=.01
            self.assertEqual(self.metric(arrays)['end_index'],6)
        arrays=self.trace();arrays['forces_N'][4,[0,2,5]]=1.
        self.assertEqual(self.metric(arrays)['cutoff_contact_channels'],['board','table','unattributed'])

    def test_contact_at_start_keeps_only_that_pre_contact_state(self):
        arrays=self.trace();arrays['forces_N'][2,4]=1.
        out=self.metric(arrays)
        self.assertEqual((out['start_index'],out['end_index'],out['sample_count']),(2,2,1))
        self.assertIsNone(out['first_crossing'])
        self.assertAlmostEqual(out['closest_opening_distance_m'],.25-self.rim)

    def test_center_plane_is_distinct_from_whole_ball_upper_cap_entry(self):
        out=self.metric(self.trace())
        crossing=out['first_crossing']
        cap_time=.05+.01*(.17+self.radius-self.rim)/(.17-.13)
        self.assertLess(crossing['time_s'],cap_time)
        self.assertNotIn('whole_ball_entry',out);self.assertNotIn('capture',out)
        # A negative aperture margin is retained; it never implies entry.
        side=self.metric(self.trace(x=.06))
        plane=np.asarray(self.cfg['compiled_planes']);h=np.linalg.norm(plane[:,:2],axis=1)
        expected=np.min((plane[:,0]*.06+plane[:,2]*self.rim+plane[:,3])/h)-self.radius
        self.assertLess(expected,0);self.assertAlmostEqual(side['mouth_clearance_m'],expected)

    def test_first_descending_crossing_and_exact_plane_equality_convention(self):
        arrays=self.trace([.3,.27,self.rim-.01,self.rim+.02,self.rim+.03,self.rim,self.rim-.02])
        out=self.metric(arrays)
        self.assertEqual(out['first_crossing']['interval_index'],4)
        self.assertEqual(out['first_crossing']['fraction'],1.)
        # Starting on the plane without an in-segment above state is no crossing.
        arrays=self.trace([.3,.27,self.rim,self.rim-.01,self.rim-.02])
        self.assertIsNone(self.metric(arrays)['first_crossing'])

    def test_rebound_requires_recent_board_contact_outward_velocity_and_free_interval(self):
        cases=[]
        arrays=self.trace();arrays['forces_N'][:,0]=0.;cases.append(arrays)
        arrays=self.trace();arrays['velocity_m_s'][2,2]=.05;cases.append(arrays)
        arrays=self.trace([.30,.27,.25,.24,.21,.17,.13,.10]);arrays['velocity_m_s'][2,2]=0.;arrays['velocity_m_s'][-1,2]=.1;cases.append(arrays)
        arrays=self.trace();arrays['forces_N'][1,0]=1.;cases.append(arrays)
        for arrays in cases:
            out=self.metric(arrays);self.assertFalse(out['available'])
            self.assertEqual(out['reason'],'no_board_rebound');self.assertEqual(out['sample_count'],0)

    def test_missing_and_invalid_rows_are_explicitly_unavailable(self):
        for arrays in (None,{},dict(position_m=[]),{k:v for k,v in self.trace().items() if k!='dt_s'}):
            out=self.metric(arrays);self.assertEqual(out['reason'],'missing_trace')
        out=self.metric(None,valid=False)
        self.assertEqual(out['reason'],'invalid_physics');self.assertFalse(out['available'])
        for key in ('start_index','end_index','start_time_s','end_time_s','cutoff_reason','first_crossing','mouth_clearance_m','closest_opening_distance_m'):
            self.assertIsNone(out[key])
        self.assertEqual(out['cutoff_contact_channels'],[])
        with self.assertRaises(ValueError):self.metric(self.trace(),valid=1)

    def test_present_malformed_trace_calibration_or_normal_fails_closed(self):
        for field in ('position_m','velocity_m_s','forces_N'):
            arrays=self.trace();arrays[field].flat[-1]=np.nan
            with self.assertRaises(ValueError):self.metric(arrays)
        arrays=self.trace();arrays['forces_N']=arrays['forces_N'][:-1]
        with self.assertRaises(ValueError):self.metric(arrays)
        arrays=self.trace();arrays['time_s']=np.arange(7)*.02
        with self.assertRaises(ValueError):self.metric(arrays)
        with self.assertRaises(ValueError):first_flight(self.scene,self.xy,[0,0,2],self.cfg,self.trace())
        bad=copy.deepcopy(self.cfg);bad['compiled_planes'][0][3]+=.01
        with self.assertRaises(ValueError):first_flight(self.scene,self.xy,[0,0,1],bad,self.trace())

    def test_contact_distances_are_validated_but_do_not_override_force_cutoff(self):
        arrays=self.trace();arrays['contact_distance_m']=np.full((6,2),np.inf)
        arrays['contact_distance_m'][3,0]=-.001
        self.assertEqual(self.metric(arrays)['end_index'],6)
        arrays['contact_distance_m'][3,0]=np.nan
        with self.assertRaises(ValueError):self.metric(arrays)

    def test_verified_compiled_rim_is_used_and_nonfinite_scene_radius_rejected(self):
        cfg=resign(dict(self.cfg,compiled_rim_m=self.rim+4e-7))
        out=first_flight(self.scene,self.xy,[0,0,1],cfg,self.trace())
        self.assertAlmostEqual(out['first_crossing']['fraction'],(.21-cfg['compiled_rim_m'])/.04)
        self.assertAlmostEqual(out['closest_opening_distance_m'],abs(.17-cfg['compiled_rim_m']))
        bad=copy.deepcopy(self.scene);bad['ball']['radius_m']=float('nan')
        with self.assertRaises(ValueError):first_flight(bad,self.xy,[0,0,1],self.cfg,self.trace())


if __name__=='__main__':unittest.main()
