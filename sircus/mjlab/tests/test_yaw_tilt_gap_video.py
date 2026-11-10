import unittest
import numpy as np
from sircus.mjlab.yaw_tilt_gap_video import choose_records, frame_plan, stage

FAR='corrected-imitation-test-r0-c6-n1'

def rows():
    base=dict(context_id=FAR,repeat=0,values=[.65,-.2,.45,30,.08],bucket_xy_m=[1.1,-.53])
    out=[dict(base,id=v,variant=v,phase='references') for v in ('far_success_reference','far_rotated')]
    out.extend(dict(base,id=f'c{i}-r0',variant='selected',phase='selected_repeats',selected_from_id=f'c{i}',grid_index=i,route_success=i==5) for i in (5,2,9,1))
    return out

class VideoTests(unittest.TestCase):
    def test_grid_order_and_repeat_zero_not_outcome_select_video(self):
        data=rows();data.append(dict(data[-1],id='later-win',repeat=1,route_success=True))
        self.assertEqual([r['id'] for r in choose_records(data,FAR)],['far_success_reference','far_rotated','c1-r0','c2-r0','c5-r0'])
        self.assertEqual(choose_records(data,FAR),choose_records(list(reversed(data)),FAR))

    def test_no_captures_still_has_baseline_reference_video(self):
        self.assertEqual(len(choose_records(rows()[:2],FAR)),2)

    def test_missing_or_duplicate_first_repeat_is_rejected(self):
        for bad in (rows()[1:],rows()+[rows()[0]],rows()+[rows()[2]]):
            with self.assertRaises(ValueError):choose_records(bad,FAR)
        with self.assertRaises(ValueError):choose_records(rows(),'other')

    def test_refusal_has_no_fabricated_native_flight(self):
        row=dict(rows()[0],invalid='finite_ik_unresolved')
        frames,clips=frame_plan([dict(row=row,label='Declined',record_sha256='a'*64,no_flight=True)],fps=10)
        self.assertEqual(len(frames),30);self.assertTrue(clips[0]['no_flight']);self.assertTrue(all(f['source_tick'] is None for f in frames))

    def test_saved_timeline_and_rejected_retention(self):
        row=dict(rows()[0],route_success=False,whole_ball_entry=True,max_retention_s=1.01,native_cache_row=0,native_initial_q10=[0]*6,native_initial_q14=[0]*6)
        bundle=dict(row=row,label='Rejected retained route',record_sha256='a'*64,cache_sha256='b'*64,time_s=np.arange(15361)/15360,events={'board':.1,'capture':1.},scored_until_s=1.)
        frames,clips=frame_plan([bundle],fps=10)
        self.assertEqual(clips[-1]['end_s'],len(frames)/10)
        for f in frames:self.assertAlmostEqual(f['source_time_s'],f['source_tick']/15360,places=12)
        self.assertIn('route rejected',stage(dict(segment='outcome_hold',source_time_s=1.),bundle))

if __name__=='__main__':unittest.main()
