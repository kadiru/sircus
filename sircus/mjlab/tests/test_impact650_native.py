"""Off-center reset/provider tests with fake tensors and a dry-run environment."""

import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from sircus.mjlab import impact650_native as native,impact650_contract as protocol
from sircus.mjlab.impact650_geometry import impact_targets
from sircus.mjlab.tests.test_clean650_geometry import Array
from sircus.mjlab.tests.test_impact650_geometry import config,request


TORCH=SimpleNamespace(as_tensor=lambda v,**kw:Array(v),float64=np.float64)


class ImpactNativeTests(unittest.TestCase):
    def test_provider_targets_drive_real_reset_and_pad_without_moving_board_target(self):
        r=request();cfg=config();bucket=np.array([r['bucket_xy_m']]);values=np.array([r['values']]);offset=np.array([r['impact_offset_m']])
        provider=native.ImpactFixedEndpointIK(cfg,bucket,values,offset,[r['q10']],[r['q14']],[True])
        with patch.dict('sys.modules',torch=TORCH):
            result=provider.solve(np.repeat(bucket,3,axis=0),np.repeat(values[:,:3],3,axis=0),np.repeat(values[:,3],3),np.repeat(values[:,4],3))
        expected=impact_targets(cfg,bucket,values,offset)
        np.testing.assert_array_equal(result['targets']['release_center_m'],np.repeat(expected['release_center_m'],3,axis=0))
        np.testing.assert_array_equal(result['q14'].numpy(),np.repeat([r['q14']],3,axis=0))
        wrong=values.copy();wrong[0,0]+=.01
        with patch.dict('sys.modules',torch=TORCH),self.assertRaises(RuntimeError):provider.solve(bucket,wrong[:,:3],wrong[:,3],wrong[:,4])

    def fixture(self,*,bad=None):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup);out=Path(temporary.name);(out/'native/64').mkdir(parents=True)
        cfg=config();r=request();targets=impact_targets(cfg,[r['bucket_xy_m']],[r['values']],[r['impact_offset_m']])
        r.update(eligible=True,substeps=64,repeat_index=0,group_id=r['id'],phase='coarse',query_fingerprint='a'*64,
                 source_record_id='source',source_record_sha256='b'*64,impact_point_world_m=targets['impact_point_world_m'][0].tolist(),
                 release_center_m=targets['release_center_m'][0].tolist(),release_center_height_m=float(targets['release_center_height_m'][0]),
                 impact_to_bucket_horizontal_m=float(targets['impact_to_bucket_horizontal_m'][0]))
        original=object();env=SimpleNamespace(scene=cfg,device='cpu',ik=original)
        def run(bucket,values,locks,*,record):
            if bad=='raise':raise RuntimeError('dry-run failure')
            solved=env.ik.solve(bucket,values[:,:3],values[:,3],values[:,4]);release=solved['targets']['release_center_m']
            positions=np.repeat(release[None],3,axis=0)
            if bad=='spawn':positions[0,:,0]+=.01
            env.last_arrays=dict(position_m=positions,time_s=np.array([0.,.1,.2]))
            q10,q14=(solved[k].numpy() for k in ('q10','q14'))
            row=dict(values=values[0].tolist(),bucket_xy_m=bucket[0].tolist(),q10=q10[0].tolist(),q14=q14[0].tolist(),
                native_release=True,route_success=False,route_quality=0,invalid=None,invalid_physics=False,board_normal=[0.,0.,1.],
                native_initial_q10=q10[0].tolist(),native_initial_q14=q14[0].tolist(),release_ball_center_m=release[0].tolist(),
                native_initial_ball_center_m=positions[0,0].tolist(),pose_gate=dict(dropper_error_mm=3. if bad=='fk' else .001),
                sentinel_native_field={'kept':True},trajectory_m=[[1,2,3]])
            if bad=='metadata':row['release_ball_center_m'][0]+=.01
            return ([] if bad=='count' else [row]),dict(synthetic=True)
        env.run=run;executor=native.NativeExecutor(cfg,None,dict(score_config={}),out,32,lambda _:None)
        return executor,env,original,r,out

    def execute(self,executor,env,requests):
        with patch.dict('sys.modules',torch=TORCH),patch.object(executor,'environment',return_value=env), \
             patch('sircus.mjlab.xy650_metrics.first_flight',return_value=dict(version='synthetic',available=False)):
            executor.execute(requests)

    def test_native_records_actual_shifted_spawn_frozen_q_and_rejected_denominators(self):
        executor,env,original,r,out=self.fixture();rejected=dict(r,id='reject',eligible=False)
        self.execute(executor,env,[r,rejected]);self.assertIs(env.ik,original)
        a,b=executor.rows;self.assertEqual(a['sentinel_native_field'],{'kept':True});self.assertNotIn('trajectory_m',a)
        self.assertEqual(a['release_center_m'],r['release_center_m']);self.assertEqual(a['release_ball_center_m'],r['release_center_m'])
        self.assertEqual(a['native_initial_ball_center_m'],r['release_center_m']);self.assertEqual(a['native_release_target_error_m'],0.)
        self.assertEqual(a['q14'],r['q14']);self.assertEqual(a['execution_fingerprint'],protocol.execution_fingerprint(a))
        self.assertFalse(b['native_release']);self.assertIsNone(b['execution_fingerprint']);self.assertIsNone(b['first_flight'])
        with np.load(out/a['native_directory']/a['native_cache_file'],allow_pickle=False) as cache:
            np.testing.assert_array_equal(cache['position_m'][0,0],r['release_center_m'])
            self.assertEqual(cache['record_ids'].tolist(),[r['id']])

    def test_wrong_spawn_center_metadata_or_actual_fk_never_passes(self):
        for bad in ('spawn','metadata','fk','count'):
            executor,env,original,r,_=self.fixture(bad=bad)
            with self.subTest(bad=bad),self.assertRaises(RuntimeError):self.execute(executor,env,[r])
            self.assertIs(env.ik,original);self.assertEqual(executor.rows,[])

    def test_failed_run_restores_provider_without_old_module_mutation(self):
        executor,env,original,r,_=self.fixture(bad='raise')
        with self.assertRaisesRegex(RuntimeError,'dry-run failure'):self.execute(executor,env,[r])
        self.assertIs(env.ik,original)


if __name__=='__main__':unittest.main()
