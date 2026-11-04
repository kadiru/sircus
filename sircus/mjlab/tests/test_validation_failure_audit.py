"""CPU-only validation scoping, geometry diagnostics and immutable output tests."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from sircus.mjlab import validation_failure_audit as audit
from sircus.mjlab.calibrated_score_oracle import score_trace
from sircus.mjlab.tests.test_calibrated_score import config

ROOT=Path(__file__).resolve().parents[3]


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,allow_nan=False)+'\n')


class DiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene=audit.read(ROOT/'sircus/isaac/scene.json');cls.cfg=config(cls.scene)

    def row(self):
        values=[.6,-.2,.3,30.,.2];bucket=[.4,-.4]
        release=[.6,-.2,.3+.2+self.scene['ball']['radius_m']]
        return dict(id='source-validation-0',values=values,bucket_xy_m=bucket,native_release=True,
            board_center_m=values[:3],board_normal=audit.expected_normal(bucket,values).tolist(),
            board_quaternion_wxyz=[1.,0.,0.,0.],native_initial_ball_center_m=release,
            native_initial_bucket_center_m=[*bucket,0.],q10=[0.]*6,q14=[0.]*6,
            native_initial_q10=[0.]*6,native_initial_q14=[0.]*6,
            pose_gate=dict(dropper_error_mm=.1),max_board_drift_mm=0.,max_integrated_joint_excursion_rad=0.,
            peak_integrated_joint_velocity_rad_s=0.,method='ppo',seed=1,samples=256,context_id='validation-0')

    def arrays(self,end=1.6):
        row=self.row();time=np.arange(round(end/.001)+1)*.001
        radius=self.scene['ball']['radius_m'];b=self.scene['bucket'];top=b['assumed_bottom_thickness_m']+b['assumed_interior_depth_m']
        knots=[row['native_initial_ball_center_m'],np.r_[row['values'][:2],row['values'][2]+radius],
               np.r_[row['bucket_xy_m'],top+.07],np.r_[row['bucket_xy_m'],b['assumed_bottom_thickness_m']+radius]]
        p=np.column_stack([np.interp(time,[0,.08,.30,.48],[v[k] for v in knots]) for k in range(3)])
        v=np.gradient(p,time,axis=0);normal=np.array(row['board_normal']);v[79]=-normal;v[82]=normal
        f=np.zeros((len(time)-1,6));f[80,0]=1.
        return dict(position_m=p,velocity_m_s=v,forces_N=f,time_s=time,contact_distance_m=np.full((len(f),2),np.inf))

    def scored(self,arrays):
        row=self.row();row.update(score_trace(self.scene,row['bucket_xy_m'],row['board_normal'],self.cfg,arrays));return row

    def test_initial_pose_is_initial_fk_not_settled_or_surface_gap(self):
        row=self.row();row['board_normal']=(np.asarray(row['board_normal'])*(1+1e-8)).tolist()
        result=audit.pose_record(row,self.scene,row['native_initial_ball_center_m'])
        self.assertAlmostEqual(result['errors']['board_normal_deg'],0.,delta=1e-6)
        self.assertEqual(result['dynamic_settled_pose'],'not_recorded')
        self.assertEqual(result['requested']['gap_above_center_plane_m'],.2)
        self.assertIn('CENTER plane',result['scope'])
        rejected=dict(values=row['values'],bucket_xy_m=row['bucket_xy_m'],native_release=False)
        self.assertFalse(audit.pose_record(rejected,self.scene)['available'])
        self.assertIsNone(audit.pose_record(rejected,self.scene)['errors'])

    def test_observed_whole_entry_plane_short_long_lateral_and_unknown(self):
        row=self.row();row['board_center_m']=[.1,-.4,.3]
        top=self.scene['bucket']['assumed_bottom_thickness_m']+self.scene['bucket']['assumed_interior_depth_m']
        height=top-self.scene['ball']['radius_m']
        for u,v,expected in ((-.2,0.,'short'),(.2,0.,'long'),(0.,.2,'lateral'),(0.,0.,'over_mouth')):
            p=np.array([[row['bucket_xy_m'][0]+u,row['bucket_xy_m'][1]+v,height+.1],
                        [row['bucket_xy_m'][0]+u,row['bucket_xy_m'][1]+v,height-.1]])
            data=dict(position_m=p,time_s=np.array([0.,.1]),forces_N=np.zeros((1,6)))
            actual=audit.direction_at_crossing(self.scene,self.cfg,row,data,1,0)
            self.assertEqual(actual['region'],expected);self.assertAlmostEqual(actual['crossing']['ball_center_m'][2],height)
            data['forces_N'][0,1]=1.
            changed=audit.direction_at_crossing(self.scene,self.cfg,row,data,1,0)
            self.assertEqual(changed['region'],'unknown');self.assertEqual(changed['geometric_region'],expected)
        data['position_m'][:,2]=[height-.1,height+.1]
        self.assertEqual(audit.direction_at_crossing(self.scene,self.cfg,row,data,1,0)['region'],'no_crossing')
        self.assertEqual(audit.direction_at_crossing(self.scene,self.cfg,row,data,1,None)['region'],'unknown')

    def test_captured_prefix_is_not_reclassified_by_later_escape(self):
        arrays=self.arrays();row=self.scored(arrays)
        self.assertTrue(row['route_success'])
        end=row['physics_steps']
        if end+2<len(arrays['position_m']):
            arrays['position_m'][end+2:]=[1.,1.,-1.]
            arrays['forces_N'][end+1:,3]=100.
        observed=audit.diagnose_native(row,self.scene,self.cfg,arrays)
        self.assertIn(observed['category'],('captured_clean','captured_assisted'))
        self.assertTrue(observed['observed_events']['terminal_prefix_only'])

    def test_interruption_and_horizon_censoring_are_separate_flags(self):
        arrays=self.arrays(end=1.1);row=self.scored(arrays)
        self.assertTrue(row['whole_ball_entry']);self.assertFalse(row['route_success'])
        observed=audit.diagnose_native(row,self.scene,self.cfg,arrays)
        self.assertEqual(observed['category'],'entry_not_retained')
        self.assertEqual(observed['retention_status'],'horizon_censored')
        arrays['position_m'][-20:-10,2]=self.scene['bucket']['assumed_interior_depth_m']+.01
        row=self.scored(arrays);observed=audit.diagnose_native(row,self.scene,self.cfg,arrays)
        self.assertEqual(observed['retention_status'],'interrupted')
        self.assertTrue(observed['observed_events']['containment_interruption_observed'])
        self.assertTrue(observed['observed_events']['retention_horizon_censored'])

    def test_replay_is_recorded_ball_path_and_static_initial_geometry(self):
        arrays=self.arrays();row=self.scored(arrays);record=audit.diagnose_native(row,self.scene,self.cfg,arrays)
        record['replay_id']='val-abc'
        replay=audit.compact_replay(row,record,arrays,dict(contract_fingerprint='a'*64))
        self.assertEqual(replay['id'],'val-abc');self.assertEqual(replay['role'],'validation')
        self.assertEqual(replay['result']['request_id'],row['id'])
        self.assertEqual(replay['result']['method'],'validation_recorded_replay')
        self.assertEqual(replay['result']['trajectory'][0],dict(time=0.,ball=arrays['position_m'][0].tolist()))
        self.assertEqual(replay['result']['success'],row['route_success'])
        self.assertIn('static initial',replay['scope'])
        for point in replay['result']['trajectory']:
            index=np.argmin(np.abs(arrays['time_s']-point['time']))
            self.assertEqual(point['ball'],arrays['position_m'][index].tolist())


class SourceFixtureTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.study=self.root/'study';self.study.mkdir()
        self.scene=audit.read(ROOT/'sircus/isaac/scene.json');self.cfg=config(self.scene)
        layouts=[dict(id=f'val-{i:02}',bucket_xy_m=[.2+i*.01,-.4]) for i in range(12)]
        self.contract=audit.signed(dict(version=audit.STUDY_VERSION,protocol=dict(study_kind='synthetic_fixture',milestones=[0,256,512,1024],seeds=[1,2,3]),
            splits=dict(validation=layouts),identity=dict(scene_sha256=audit.digest(self.scene),source_sha256={},calibration=dict(score_config=self.cfg),
                trace_schema=dict(native_dt_s=.001,horizon_s=1.6))))
        self.workers=[]
        for method in ('sweep','ppo'):
            for seed in (1,2,3):
                selected=0 if method=='sweep' and seed==1 else 256 if seed==3 else 1024
                worker=dict(method=method,seed=seed,status='completed',selected_samples=selected);self.workers.append(worker)
                prefix=self.study/f'runs/{method}-seed{seed}';prefix.mkdir(parents=True)
                rows=[]
                for n in (0,256,512,1024):
                    for layout in layouts:
                        for repeat in range(3):
                            rows.append(dict(id=f'{method}-{seed}-validation-{n:04d}-{layout["id"]}-r{repeat}',role='validation',method=method,seed=seed,samples=n,
                                context_id=layout['id'],bucket_xy_m=layout['bucket_xy_m'],repeat=repeat,values=[.6,-.2,.3,30.,.2],native_release=False,
                                invalid='synthetic_setup_reject',route_success=False))
                # These metadata rows must never trigger trace access.
                rows.insert(0,dict(id='poison-train',role='train',native_cache_file='forbidden-train.npz'))
                rows.append(dict(id='poison-test',role='test',native_cache_file='forbidden-test.npz'))
                (prefix/'rows.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in rows))
                write(prefix/'selected.json',dict(selected_samples=selected,frozen_before_test=True))
                write(prefix/'manifest.json',{str(p.relative_to(prefix)):audit.sha(p) for p in prefix.rglob('*') if p.is_file()})
        write(self.study/'contract.json',self.contract)
        self.report=audit.signed(dict(version=audit.STUDY_VERSION,status='completed',phase='completed',contract_fingerprint=self.contract['fingerprint'],protocol=self.contract['protocol'],runs=self.workers))
        write(self.study/'report.json',self.report);write(self.study/'splits.json',self.contract['splits'])
        write(self.study/'calibration.json',dict(score_config=self.cfg));write(self.study/'verification.json',{});write(self.study/'setup-home.json',{})
        self.seal()

    def tearDown(self):self.temp.cleanup()

    def seal(self):
        write(self.study/'manifest.json',{str(p.relative_to(self.study)):audit.sha(p) for p in self.study.rglob('*') if p.is_file() and p!=self.study/'manifest.json'})
        self.auditpath=self.root/'audit.json'
        self.full=audit.signed(dict(version=audit.SOURCE_AUDIT_VERSION,status='passed',checks=99,failures=[],study_kind='synthetic_fixture',synthetic_fixture=True,
            contract_fingerprint=self.contract['fingerprint'],report_sha256=audit.sha(self.study/'report.json'),report_fingerprint=self.report['fingerprint'],
            study_manifest_sha256=audit.sha(self.study/'manifest.json'),source_sha256={},auditor_dependencies={},
            auditor_sha256=audit.sha(ROOT/'sircus/mjlab/calibrated_benchmark_verify.py'),
            role_counts=dict(validation=dict(attempts=864,captures=0,native_releases=0))))
        write(self.auditpath,self.full)

    def execute(self,output=None):
        return audit.build(self.study,self.auditpath,ROOT,output or self.root/'output',expected_contract_fingerprint=self.contract['fingerprint'],
                           expected_audit_sha256=audit.sha(self.auditpath),synthetic=True)

    def test_complete864_requests216_selected_and_no_train_test_trace_access(self):
        before={str(p.relative_to(self.study)):audit.sha(p) for p in self.study.rglob('*') if p.is_file()}
        with patch.object(audit.np,'load',side_effect=AssertionError('No trace should open for rejected requests')):
            result=self.execute()
        self.assertEqual(result['validation_requests'],864);self.assertEqual(result['primary_selected_requests'],216)
        report=audit.read(self.root/'output/report.json');audit.verify_signed(report,'diagnostic')
        self.assertTrue(report['synthetic_fixture']);self.assertEqual(len(report['cells']),120)
        self.assertEqual(report['cohorts']['selected']['attempts'],216)
        self.assertEqual([v['attempts'] for v in report['cohorts']['milestones']],[216]*4)
        for cell in report['cells']:
            self.assertEqual(cell['attempts'],9);self.assertEqual(set(cell['categories']),set(audit.CATEGORIES))
            self.assertEqual(cell['categories']['setup_rejected'],9);self.assertEqual(sum(cell['categories'].values()),9)
            self.assertIsNone(cell['pose_audit']['errors']['board_position_mm'])
        self.assertTrue(all(not r['recorded_replay_available'] and 'path' not in r for r in report['representatives']))
        after={str(p.relative_to(self.study)):audit.sha(p) for p in self.study.rglob('*') if p.is_file()}
        self.assertEqual(before,after)

    def test_pinned_audit_and_output_boundary_fail_closed(self):
        with self.assertRaisesRegex(ValueError,'Pinned'):
            audit.build(self.study,self.auditpath,ROOT,self.root/'bad',expected_contract_fingerprint=self.contract['fingerprint'],expected_audit_sha256='0'*64,synthetic=True)
        with self.assertRaisesRegex(ValueError,'outside'):self.execute(self.study/'bad')
        self.assertFalse((self.study/'bad').exists())

    def test_validation_role_cannot_forge_train_identity(self):
        path=self.study/'runs/sweep-seed1/rows.jsonl';rows=[json.loads(x) for x in path.read_text().splitlines()]
        next(r for r in rows if r.get('role')=='validation')['id']='sweep-1-train-00000'
        path.write_text(''.join(json.dumps(v)+'\n' for v in rows))
        with self.assertRaisesRegex(ValueError,'TRAIN/TEST'):
            audit.read_validation_rows(path,'sweep',1,self.contract)

    def test_mixed_packet_rejected_before_dense_object_array_access(self):
        prefix=self.study/'runs/sweep-seed1';rows=[json.loads(x) for x in (prefix/'rows.jsonl').read_text().splitlines()]
        row=next(r for r in rows if r.get('role')=='validation');row.update(native_release=True,native_cache_file='native-0000.npz',native_cache_row=0)
        (prefix/'native').mkdir();packet=prefix/'native/native-0000.npz'
        np.savez_compressed(packet,record_ids=np.array(['poison-train-0000']),position_m=np.array([object()],dtype=object))
        row['native_cache_sha256']=audit.sha(packet)
        (prefix/'rows.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in rows))
        write(prefix/'manifest.json',{str(p.relative_to(prefix)):audit.sha(p) for p in prefix.rglob('*') if p.is_file() and p!=prefix/'manifest.json'})
        self.seal()
        with self.assertRaisesRegex(ValueError,'mixed trace packet'):
            self.execute()


if __name__=='__main__':unittest.main()
