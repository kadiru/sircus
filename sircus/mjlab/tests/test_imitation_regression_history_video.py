import unittest
import numpy as np
from sircus.mjlab import imitation_regression_history_video as V
from sircus.mjlab.prepare_imitation_regression import CASES


class HistoricalReplay(unittest.TestCase):
    def rows(self):
        return [dict(id=m+'-literal',context_id=CASES[0],phase='test',method=m,route_success=m=='current_bc',native_release=m=='current_bc') for m in ('current_bc','corrected_bc')]

    def test_literal_original_pair_not_new_intervention(self):
        rows=self.rows();selected=V.choose_records(rows,CASES[0]);self.assertEqual([r['id'] for r in selected],[r['id'] for r in rows]);self.assertTrue(all('Historical' in r['label'] for r in selected))
        for altered in (rows[:1],rows+[rows[0]],[dict(rows[0],phase='diagnostic'),rows[1]],[dict(rows[0],route_success=False),rows[1]]):
            with self.assertRaises(ValueError):V.choose_records(altered,CASES[0])
        with self.assertRaises(ValueError):V.choose_records(rows,'unknown')

    def test_no_unrecorded_variant_or_joint_pose_added_to_decline(self):
        r=dict(self.rows()[1],values=[.5,-.2,.4,28,.2],bucket_xy_m=[.7,-.3],invalid='finite_ik_unresolved')
        frames,clips=V.frame_plan([dict(row=r,variant='corrected_exact',no_flight=True,label='Historical decline',record_sha256='a'*64)],fps=10)
        self.assertNotIn('variant',r);self.assertEqual(clips[0]['variant'],'corrected_exact');self.assertEqual(len(frames),30);self.assertTrue(all(f['source_tick'] is None for f in frames));self.assertNotIn('held_joints_rad',clips[0])

    def test_historical_native_samples_keep_original_row_hash_and_no_variant_mutation(self):
        row=dict(self.rows()[0],values=[.5,-.2,.4,28,.2],bucket_xy_m=[.7,-.3],whole_ball_entry=True,native_cache_row=0,native_initial_q10=[0]*6,native_initial_q14=[0]*6)
        bundle=dict(row=row,variant='current_exact',label='Historical capture',record_sha256='a'*64,cache_sha256='b'*64,time_s=np.arange(15361)/15360,events={'capture':1.},scored_until_s=1.)
        frames,clips=V.frame_plan([bundle],fps=10);self.assertNotIn('variant',row);self.assertEqual(clips[0]['record_sha256'],'a'*64);self.assertEqual(clips[0]['variant'],'current_exact')
        self.assertTrue(all(abs(f['source_time_s']-f['source_tick']/15360)<1e-12 for f in frames))

if __name__=='__main__':unittest.main()
