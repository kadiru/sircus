import unittest
import numpy as np
from sircus.mjlab.clearance_horizon_diagnosis import descending_time,horizon_relation,event_clock

class HorizonTests(unittest.TestCase):
    def test_descending_root_and_absent_crossing(self):
        self.assertAlmostEqual(descending_time(.5,0.,0.,-10.),np.sqrt(.1))
        self.assertIsNone(descending_time(.1,0.,.5,-10.))
        with self.assertRaises(ValueError):descending_time(.1,0.,.5,0.)
    def test_all_member_coverage_and_late_contact_remain_distinct(self):
        r=horizon_relation(.335,[.321,.322,.323,.324,.327]);self.assertTrue(r['event_after_all_horizons']);self.assertEqual(r['members_covering_event'],0)
        r=horizon_relation(.323,[.321,.322,.323,.324,.327]);self.assertEqual(r['members_covering_event'],3);self.assertFalse(r['event_after_all_horizons'])
        self.assertIsNone(horizon_relation(None,[.321]*5))
        with self.assertRaises(ValueError):horizon_relation(.3,[.32]*4)
    def test_contact_interval_start_is_relative_to_actual_rebound_not_absolute_clock(self):
        f=np.zeros((10,6));f[7,4]=1.;trace=dict(time_s=np.arange(11)*.01,forces_N=f)
        metric=dict(available=True,start_index=3,end_index=5,cutoff_reason='contact',cutoff_contact_channels=['bucket'])
        out=event_clock(trace,metric);self.assertAlmostEqual(out['first_contact_delay_s'],.02);self.assertAlmostEqual(out['first_robot_delay_s'],.04)
        self.assertEqual(out['first_contact_channels'],['bucket']);self.assertAlmostEqual(out['observed_rebound_time_s'],.03)
        self.assertEqual(event_clock(trace,{'available':False}),{'available':False})

if __name__=='__main__':unittest.main()
