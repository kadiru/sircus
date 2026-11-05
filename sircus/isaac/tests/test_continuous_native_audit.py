"""Full-rate native outcome reconstruction rejects corrupted state/score claims."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

SOURCE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SOURCE))
import continuous_native_audit as ca
from board_route import BoardRoute
from entry_quality import EntryQuality
from farm_contract import reward
from rl_contract import ShotScore


class NativeAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.scene=json.loads((SOURCE/'scene.json').read_text())
        self.origins=np.array([[0,0,0],[3,0,0]],dtype=float)
        self.cells=[dict(id=f'test-{i}',bucket_xy_m=[.67,-.42]) for i in range(2)]
        positions=np.repeat(self.origins[None],2,axis=0)
        positions[0,:,2]=.3;positions[1,:,2]=-.3
        boards=np.repeat(self.origins[None],2,axis=0);boards[:,:,:]+=[.67,-.1,.4]
        self.data=dict(time_step=np.array([11,12]),position_m=positions,velocity_m_s=np.zeros((2,2,3)),
            forces_N=np.zeros((2,2,6)),joint_position_rad=np.zeros((2,2,2,6)),joint_velocity_rad_s=np.zeros((2,2,2,6)),
            board_position_m=boards,board_quaternion_wxyz=np.tile([1.,0,0,0],(2,2,1)))
        self.path=self.root/'physics.npz';np.savez_compressed(self.path,**self.data)
        actions=[dict(id=f'action-{i}',height_m=.4,gap_m=.2,tilt_deg=25.,board_x_m=.67,board_y_m=-.1) for i in range(2)]
        rows=[]
        for i,cell in enumerate(self.cells):
            bucket=dict(self.scene['bucket'],center_xy_m=cell['bucket_xy_m'])
            score=ShotScore(bucket,.0375);quality=EntryQuality(bucket,.0375);route=BoardRoute(1/240)
            for j in range(2):
                score.step(positions[j,i]-self.origins[i],0,1/240,0)
                quality.step(positions[j,i]-self.origins[i],0,score.entered)
                route.step(dict.fromkeys(ca.FORCES,0.))
            result=score.result();q=quality.result(result['success'])
            rows.append(dict(actions[i],**result,**q,**route.result(result['success'],q['quality_score']),
                cell_id=cell['id'],invalid=None,qualification=dict(accepted=True),path_rejection=None,
                settled=dict(position_error_mm=0.,normal_error_deg=0.,joint_speed_rad_s=0.,position_drift_mm=0.,normal_drift_deg=0.,dropper_position_error_mm=0.),
                settled_board_center_m=[.67,-.1,.4],settled_board_normal=[0,0,1],release_ball_center_m=[0,0,.3]))
        self.trial=dict(rows=rows,actions=actions,enabled=[True,True],rewards=[reward(r) for r in rows],release_step=10,score_end_step=12,
            physics=dict(file=self.path.name,sha256=ca.digest(self.path),samples=2))
        self.clip=dict(rows=copy.deepcopy(rows),actions=copy.deepcopy(actions),release_step=10,
            frames=[dict(phase='setup',physics_step=8),dict(phase='release',physics_step=12,
                all_ball_world_m=positions[1].tolist(),all_joints_rad={name:np.zeros((2,6)).tolist() for name in ('Arm10','Arm14')})])

    def verify(self,trial=None,clip=None):
        return ca.verify_trial(self.root,trial or self.trial,self.scene,self.cells,self.origins,self.clip if clip is None else clip)

    def rewrite_data(self):
        np.savez_compressed(self.path,**self.data)
        self.trial['physics']['sha256']=ca.digest(self.path)

    def test_all_native_misses_rescore_and_frame_states_bind(self):
        result=self.verify()
        self.assertEqual(result['rescored_releases'],2)
        self.assertEqual(result['native_steps'],2)
        self.assertEqual(result['captures'],0)
        self.assertEqual(result['frame_readbacks'],1)

    def test_whole_ball_one_second_capture_and_post_success_freeze_reconstruct(self):
        count=253
        self.data={k:np.repeat(v[:1],count,axis=0) for k,v in self.data.items()}
        self.data['time_step']=np.arange(11,11+count)
        for i,cell in enumerate(self.cells):
            self.data['position_m'][:,i,:2]=self.origins[i,:2]+cell['bucket_xy_m']
        self.data['position_m'][:2,:,2]=.3
        self.data['position_m'][2,:,2]=.18  # mouth crossing before whole-ball entry
        self.data['position_m'][3:,:,2]=.06
        self.data['forces_N'][0,:,0]=1.
        self.data['velocity_m_s'][1,:,2]=1.
        self.data['forces_N'][243:,:,2]=20.  # ignored after success at index242
        self.trial['score_end_step']=10+count;self.trial['physics']['samples']=count
        for i,cell in enumerate(self.cells):
            bucket=dict(self.scene['bucket'],center_xy_m=cell['bucket_xy_m'])
            score=ShotScore(bucket,.0375);quality=EntryQuality(bucket,.0375);route=BoardRoute(1/240)
            for j in range(count):
                score.step(self.data['position_m'][j,i]-self.origins[i],self.data['forces_N'][j,i,0],1/240,self.data['velocity_m_s'][j,i,2])
                quality.step(self.data['position_m'][j,i]-self.origins[i],0.,score.entered)
                route.step(dict(zip(ca.FORCES,self.data['forces_N'][j,i])))
                if score.success:break
            outcome=score.result();q=quality.result(outcome['success'])
            self.trial['rows'][i].update(outcome,**q,**route.result(outcome['success'],q['quality_score']),release_ball_center_m=self.data['position_m'][0,i].tolist())
            self.trial['rows'][i]['release_ball_center_m']=(self.data['position_m'][0,i]-self.origins[i]).tolist()
            self.trial['rewards'][i]=reward(self.trial['rows'][i])
        self.rewrite_data()
        self.clip['rows']=copy.deepcopy(self.trial['rows'])
        self.clip['frames'][1].update(physics_step=10+count,all_ball_world_m=self.data['position_m'][-1].tolist())
        result=self.verify()
        self.assertEqual(result['captures'],2)
        self.assertEqual(result['clean_captures'],2)
        self.assertTrue(all(r['max_retention_s']>=1-1e-9 and r['external_contact_steps']==0 for r in self.trial['rows']))

    def test_rehashed_clock_corruption_rejected(self):
        self.data['time_step'][1]=13;self.rewrite_data()
        with self.assertRaisesRegex(ValueError,'clock coverage'):self.verify()

    def test_rehashed_force_changes_reconstructed_outcome(self):
        self.data['forces_N'][0,0,0]=3.;self.rewrite_data()
        with self.assertRaisesRegex(ValueError,'rescored outcome'):self.verify()

    def test_false_capture_label_rejected_even_with_trial_clip_agreement(self):
        self.trial['rows'][0]['success']=True
        self.trial['rewards'][0]=reward(self.trial['rows'][0])
        self.clip['rows']=copy.deepcopy(self.trial['rows'])
        with self.assertRaisesRegex(ValueError,'rescored outcome'):self.verify()

    def test_rehashed_nonfinite_native_state_rejected(self):
        self.data['joint_position_rad'][1,0,0,0]=float('nan');self.rewrite_data()
        with self.assertRaisesRegex(ValueError,'packet joint'):self.verify()

    def test_rehashed_unnormalized_quaternion_rejected(self):
        self.data['board_quaternion_wxyz'][0,1]=[2.,0,0,0];self.rewrite_data()
        with self.assertRaisesRegex(ValueError,'quaternion'):self.verify()

    def test_camera_ball_or_joint_readback_must_match_native_sample(self):
        for key in ('ball','joint'):
            clip=copy.deepcopy(self.clip)
            if key=='ball':clip['frames'][1]['all_ball_world_m'][0][0]+=.01
            else:clip['frames'][1]['all_joints_rad']['Arm14'][0][0]+=.01
            with self.assertRaisesRegex(ValueError,'Camera .* readback'):self.verify(clip=clip)

    def test_measured_settling_guard_cannot_be_bypassed(self):
        self.trial['rows'][0]['settled']['position_error_mm']=2.1
        with self.assertRaisesRegex(ValueError,'release admission'):self.verify()

    def test_physics_omission_only_allowed_for_all_rejected_attempts(self):
        trial=copy.deepcopy(self.trial);trial['physics']=None;trial['score_end_step']=trial['release_step']
        with self.assertRaisesRegex(ValueError,'Missing physics'):self.verify(trial=trial,clip={"frames":[]})
        for i,cell in enumerate(self.cells):
            trial['rows'][i].update(ca.expected_empty(self.scene,cell,1/240),invalid='endpoint_unresolved')
            trial['rewards'][i]=reward(trial['rows'][i])
        result=self.verify(trial=trial,clip={"frames":[]})
        self.assertEqual(result['rescored_releases'],0)
        self.assertEqual(result['invalid_cells'],2)
        self.assertEqual(result['native_steps'],0)

    def test_original_capture_scoring_freezes_after_success_or_floor_exit(self):
        # Later contact must not change an already frozen per-cell result.
        self.data={k:np.concatenate([v,v[-1:]],axis=0) for k,v in self.data.items()}
        self.data['time_step']=np.array([11,12,13]);self.data['forces_N'][2,:,2]=20
        self.trial['score_end_step']=13;self.trial['physics']['samples']=3
        self.rewrite_data()
        result=self.verify()
        self.assertEqual(result['captures'],0)


if __name__=='__main__':unittest.main()
