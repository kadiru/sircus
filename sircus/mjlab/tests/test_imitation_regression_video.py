import unittest
import numpy as np
from sircus.mjlab.imitation_regression_video import choose_records,frame_plan,stage


class RecordedVideoTests(unittest.TestCase):
    def rows(self):
        return [dict(id=v,context_id='case',variant=v,phase='diagnostic',repeat=0) for v in ('current_exact','corrected_exact','blend50')]

    def test_selection_is_always_first_repeat_not_success_selected(self):
        rows=self.rows()+[dict(self.rows()[0],id='later-winner',repeat=1,route_success=True)]
        self.assertEqual([r['id'] for r in choose_records(rows,'case')],['current_exact','corrected_exact','blend50'])
        for wrong in (rows[1:],rows+[self.rows()[0]],[]):
            with self.assertRaises(ValueError):choose_records(wrong,'case')

    def test_no_flight_card_has_no_native_tick_or_fake_geometry(self):
        row=dict(self.rows()[1],values=[.5,-.2,.4,28,.2],bucket_xy_m=[.7,-.3],invalid='finite_ik_unresolved')
        frames,clips=frame_plan([dict(row=row,label='Rejected',record_sha256='a'*64,no_flight=True)],fps=10)
        self.assertEqual(len(frames),30);self.assertTrue(all(f['source_tick'] is None and f['source_time_s'] is None for f in frames));self.assertTrue(clips[0]['no_flight']);self.assertNotIn('held_joints_rad',clips[0])

    def test_native_frames_use_exact_ticks_and_mixed_movie_is_contiguous(self):
        row=dict(self.rows()[0],values=[.5,-.2,.4,28,.2],bucket_xy_m=[.7,-.3],route_success=True,whole_ball_entry=True,native_cache_row=0,native_initial_q10=[0]*6,native_initial_q14=[0]*6)
        bundle=dict(row=row,label='Capture',record_sha256='a'*64,cache_sha256='b'*64,time_s=np.arange(15361)/15360,events={'board':.1,'capture':1.},scored_until_s=1.)
        refusal=dict(row=dict(row,id='decline',variant='corrected_exact',invalid='finite_ik_unresolved'),label='Decline',record_sha256='c'*64,no_flight=True)
        frames,clips=frame_plan([bundle,refusal,bundle],fps=10)
        self.assertEqual(clips[0]['end_s'],clips[1]['start_s']);self.assertEqual(clips[1]['end_s'],clips[2]['start_s']);self.assertEqual(clips[-1]['end_s'],len(frames)/10)
        for i,f in enumerate(frames):
            self.assertEqual(f['frame'],i)
            if f['source_tick'] is not None:self.assertAlmostEqual(f['source_time_s'],f['source_tick']/15360,places=12)

    def test_retention_after_forbidden_contact_is_never_captioned_capture(self):
        b=dict(row=dict(route_success=False,max_retention_s=1.01,whole_ball_entry=True),events={'capture':2.0})
        self.assertIn('route rejected',stage(dict(segment='normal',source_time_s=2.1),b));self.assertIn('route rejected',stage(dict(segment='outcome_hold',source_time_s=2.2),b))

if __name__=='__main__':unittest.main()
