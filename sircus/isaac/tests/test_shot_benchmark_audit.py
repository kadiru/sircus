"""Synthetic CPU receipts/traces, never Isaac run or performance evidence."""
import copy
import gzip
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SOURCE))
import shot_benchmark_audit as audit
import shot_benchmark_contract as contract
import shot_benchmark_reward as rewards
import test_continuous_native_audit as fixture_module
import continuous_policy


def save(path, value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,allow_nan=False))


def compressed(path, value):
    path = Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    raw=json.dumps(value,allow_nan=False,separators=(",",":")).encode()
    path.write_bytes(gzip.compress(raw,mtime=0))
    receipt=dict(file=path.name,sha256=audit.digest(path),bytes=path.stat().st_size,
                 uncompressed_sha256=hashlib.sha256(raw).hexdigest())
    save(str(path)+".receipt.json",receipt)
    return receipt


class NativeWrapperTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixture_module.NativeAuditTests()
        self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        f=self.fixture
        self.trial=copy.deepcopy(f.trial)
        self.trial.update(cells=f.cells,origins_m=f.origins.tolist())
        self.trial['reward_features']=[]
        for i,row in enumerate(self.trial['rows']):
            row['scoring_board_normal']=[0,0,1]
            self.trial['reward_features'].append(rewards.build_features(f.scene,f.cells[i]['bucket_xy_m'],
                f.data['position_m'][:,i]-f.origins[i],f.data['velocity_m_s'][:,i],
                f.data['forces_N'][:,i],1/240,[0,0,1]))
        self.trial['rewards']=[rewards.reward(r,p) for r,p in zip(self.trial['rows'],self.trial['reward_features'])]

    def verify(self):
        return audit.verify_native_trial(self.fixture.root,self.trial,self.fixture.scene)

    def test_original_physical_audit_and_new_reward_recompute_without_mutation(self):
        before=copy.deepcopy(self.trial)
        result=self.verify()
        self.assertEqual(result['rescored_releases'],2)
        self.assertEqual(result['native_steps'],2)
        self.assertEqual(result['new_rewards_recomputed'],2)
        self.assertEqual(self.trial,before)
        self.assertEqual(self.trial['rewards'],[-.8,-.8])

    def test_new_reward_corruption_is_not_hidden_by_legacy_adapter(self):
        self.trial['rewards'][0]=10.
        with self.assertRaisesRegex(ValueError,'New benchmark reward'):self.verify()

    def test_rehashed_prefix_feature_corruption_rejected(self):
        self.trial['reward_features'][0]['max_retention_preexternal_s']=1.
        with self.assertRaisesRegex(ValueError,'preexternal feature'):self.verify()

    def test_original_numeric_speed_guard_still_rejects(self):
        # Row gate remains original. Full 8/6-DOF velocity reconstruction is the
        # separate unchanged verify_settling_history production audit.
        self.trial['rows'][0]['settled']['joint_speed_rad_s']=.02
        with self.assertRaisesRegex(ValueError,'release admission'):self.verify()

    def test_original_native_false_capture_still_rejected(self):
        self.trial['rows'][0]['success']=True
        with self.assertRaisesRegex(ValueError,'rescored outcome'):self.verify()

    def test_trace_byte_corruption_rejected(self):
        self.fixture.path.write_bytes(b'corrupted trace bytes')
        with self.assertRaisesRegex(ValueError,'physics checksum'):self.verify()


class ReceiptAndProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.name='trials/train-00000.json.gz'
        self.receipt=compressed(self.root/self.name,dict(label='synthetic fixture',value=7))

    def test_compressed_and_uncompressed_bytes_both_bound(self):
        value,receipt=audit.read_trial(self.root,self.name,self.receipt['sha256'])
        self.assertEqual(value['value'],7)
        self.assertEqual(receipt['uncompressed_sha256'],self.receipt['uncompressed_sha256'])
        saved=self.root/(self.name+'.receipt.json')
        changed=json.loads(saved.read_text());changed['uncompressed_sha256']='0'*64;save(saved,changed)
        with self.assertRaisesRegex(ValueError,'Uncompressed trial checksum'):
            audit.read_trial(self.root,self.name,self.receipt['sha256'])

    def test_compressed_byte_tamper_and_traversal_rejected(self):
        (self.root/self.name).write_bytes(b'wrong')
        with self.assertRaisesRegex(ValueError,'Compressed trial checksum'):
            audit.read_trial(self.root,self.name,self.receipt['sha256'])
        with self.assertRaisesRegex(ValueError,'Unsafe'):
            audit.read_trial(self.root,'../outside.json.gz','0'*64)

    def test_production_sequence_counts_and_no_test_before_final(self):
        splits=contract.make_splits([[2,2]])
        sequence=audit.expected_sequence(contract.make_protocol(),splits,contract.SEEDS[0],False)
        self.assertEqual(len(sequence),301)
        train=[s for s in sequence if s['mode']=='train']
        self.assertEqual(len(train),289)
        self.assertEqual(sum(sum(s['enabled']) for s in train),10000)
        evaluation=[s for s in sequence if s['mode']!='train']
        self.assertEqual(sum(len(s['cells']) for s in evaluation),420)
        self.assertEqual([s['attempts_before'] for s in evaluation if s['mode']=='eval-test'],[10000,10000])
        smoke=audit.expected_sequence(contract.make_protocol(),splits,contract.SEEDS[0],True)
        self.assertEqual([s['mode'] for s in smoke],['train','smoke-validation'])
        self.assertEqual(sum(sum(s['enabled']) for s in smoke),70)

    def test_source_hash_receipt_cannot_omit_or_relabel_executed_file(self):
        actual={p.name:audit.digest(p) for p in sorted(SOURCE.glob('*.py'))}
        binding=dict(source_sha256=actual)
        self.assertEqual(audit.verify_source(binding,{},{}),actual)
        changed=copy.deepcopy(binding);changed['source_sha256'].pop('settled_transit.py')
        with self.assertRaisesRegex(ValueError,'source bytes'):audit.verify_source(changed,{}, {})
        with self.assertRaisesRegex(ValueError,'bindings differ'):
            audit.verify_source(binding,dict(source_sha256={}),{})


class SyntheticSmokeWorkflowTests(unittest.TestCase):
    """Complete synthetic all-rejected receipts exercise the orchestration audit.

    FK/physics are not fabricated as successful shots: every row is explicitly
    rejected, no ball is released and no performance is claimed.
    """
    def setUp(self):
        from continuous_feasibility import decode_action
        from continuous_policy import build_observations, physical_actions
        from expanded_pose_menu import endpoint_targets
        from settled_transit import protocol as setup_protocol
        from shot_benchmark_methods import GuidedSweep, geometry_prior
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.run=self.root/'synthetic-smoke';self.run.mkdir()
        (self.run/'checkpoints').mkdir()
        self.scene=json.loads((SOURCE/'scene.json').read_text())
        self.splits=contract.make_splits([[2.,2.]])
        save(self.root/'splits.json',self.splits)
        pool=dict(scope='Synthetic fixture published-coordinate pool only',coordinates=[[2.,2.]],
            sources=[dict(blob='1'*40,path='baselines/synthetic/fixture.json',ref='synthetic-only',added_coordinates=1)],
            refs_examined=1,unique_blobs_examined=1)
        save(self.root/'historical-coordinates.json',pool)
        camera=dict(eye_m=[10.,-6.,2.5],target_m=[7.8,-3.5,.4])
        save(self.run/'selected-camera.json',camera)
        spec=contract.signed(dict(scene=self.scene,cells=self.splits['train']))
        save(self.run/'contract.json',spec)
        source=dict(commit='a'*40,archive_sha256='b'*64,geometry_fingerprint=spec['fingerprint'],
            camera_sha256=audit.digest(self.run/'selected-camera.json'),scoring_version='synthetic-fixture-only')
        experiment=dict(schema='sircus-shot-benchmark-experiment-v1',experiment_id='synthetic-not-native',
            protocol=contract.make_protocol(),reward=rewards.protocol(),source=source,
            split_manifest_file='splits.json',split_manifest_sha256=audit.digest(self.root/'splits.json'),
            historical_exclusion_file='historical-coordinates.json',
            historical_exclusion_sha256=audit.digest(self.root/'historical-coordinates.json'))
        self.plan=self.root/'experiment.json';save(self.plan,experiment)
        source_hashes={p.name:audit.digest(p) for p in sorted(SOURCE.glob('*.py'))}
        save(self.run/'experiment-binding.json',dict(experiment_sha256=audit.digest(self.plan),
            split_sha256=experiment['split_manifest_sha256'],source=source,setup=setup_protocol(),source_sha256=source_hashes))
        seed=contract.SEEDS[0];sweep=GuidedSweep(self.splits['train'],seed)
        receipts={};saved_trials=[];clips=[];checkpoints=[]
        def checkpoint(label):
            name='checkpoints/'+label+'.json';sweep.save(self.run/name)
            reference=dict(file=name,sha256=audit.digest(self.run/name),state_hash=sweep.state()['fingerprint'],training_attempts=sweep.observed)
            checkpoints.append(reference);return reference
        def trial(cells,normalized,cp,mode,proposal=None,picked=None,request_order=None):
            index=len(saved_trials)
            physical=physical_actions(normalized,contract.LOW,contract.HIGH)
            actions=[decode_action(cell,action,f'{mode}-{index}-{i}') for i,(cell,action) in enumerate(zip(cells,physical))]
            histories=[];rows=[]
            for cell,action in zip(cells,actions):
                targets=endpoint_targets(self.scene,cell,action)
                center=targets['board_world'][:3,3].tolist();normal=targets['board_normal_world'].tolist();drop=targets['Arm10'][:3,3].tolist()
                normal_drift=float(np.rad2deg(np.arccos(np.clip(np.dot(normal,normal),-1,1))))
                history=[]
                for j in range(7):
                    history.append(dict(iteration=0,measurement_start_step=300+j*60,physics_step=360+j*60,
                        board_center_m=center,board_normal=normal,dropper_flange_m=drop,
                        previous_board_center_m=center,previous_board_normal=normal,previous_dropper_flange_m=drop,
                        position_error_mm=0.,normal_error_deg=normal_drift,dropper_position_error_mm=0.,
                        position_drift_mm=0.,normal_drift_deg=normal_drift,dropper_drift_mm=0.,joint_speed_rad_s=0.,
                        arms={arm:dict(position_rad=[0.]*6,velocity_rad_s=[0.]*6,target_rad=[0.]*6) for arm in ('Arm10','Arm14')},
                        articulation_joint_names=dict(Arm10=['joint'+str(i) for i in range(1,7)]+['finger_left_joint','finger_right_joint'],Arm14=['joint'+str(i) for i in range(1,7)]),
                        all_articulation_joint_velocity=dict(Arm10=[0.]*8,Arm14=[0.]*6)))
                histories.append(history)
                row=dict(action,**audit.native.expected_empty(self.scene,cell,1/240),cell_id=cell['id'],split=cell['split'],
                    invalid='synthetic_rejection',qualification=dict(accepted=False,reason='synthetic_rejection'),
                    path_proxy=None,path_rejection=None,settled=history[-1],settled_board_center_m=center,
                    settled_board_normal=normal,scoring_board_normal=normal,release_ball_center_m=targets['release_center_m'].tolist())
                rows.append(row)
            arrays={arm:np.zeros((35,6)).tolist() for arm in ('Arm10','Arm14')}
            state={arm:{key:copy.deepcopy(arrays[arm]) for key in ('position_rad','velocity_rad_s','target_rad')} for arm in arrays}
            gains={arm:{key:np.ones((35,6)).tolist() for key in ('effort','stiffness','damping')} for arm in arrays}
            transit=[dict(phase='start',step=0,state=state,original_gains=gains),dict(phase='tail',step=60,state=state),dict(phase='restored_original_hold',step=300,state=state,restored_gains=gains)]
            home=dict(protocol='canonical-episode-home-reset-before-visible-drive-v1',episode_reset=True,
                path_eligible=[True]*35,path_checks=[dict(accepted=True) for _ in range(35)],physics_step=0,
                target_joint_rad=arrays,measured_joint_rad=arrays,measured_joint_velocity_rad_s=arrays)
            payload=dict(schema='sircus-shot-benchmark-native-trial-v1',mode=mode,batch=index,checkpoint=cp,
                source=source,split_manifest_sha256=experiment['split_manifest_sha256'],cells=cells,enabled=[True]*35,
                rows=rows,actions=actions,observations=build_observations(cells).tolist(),actions_normalized=normalized.tolist(),
                learner_packet=picked,proposal_metadata=proposal,rewards=[-1.]*35,reward_features=[None]*35,
                training_request_order=request_order,
                home=home,release_step=720,score_end_step=720,physics=None,origins_m=np.zeros((35,3)).tolist(),
                setup_controller=setup_protocol(),settling_history=histories,transit_history=transit,correction_history=[],trajectory_m=[[] for _ in range(35)])
            name=f'trials/{mode}-{index:05d}.json.gz';receipt=compressed(self.run/name,payload);receipts[name]=receipt['sha256']
            saved_trials.append(payload)
            if mode=='train':
                image=self.run/'frames/00000.jpg';image.parent.mkdir()
                # Byte-binding fixture only; no decoded/rendered image claim.
                image.write_bytes(b'\xff\xd8synthetic-frame-checksum-fixture-only\xff\xd9')
                frame=dict(frame=0,physics_step=60,phase='setup',eye_m=camera['eye_m'],target_m=camera['target_m'],
                    jpeg_sha256=audit.digest(image),jpeg_quality=82,resolution=[2,2],all_ball_world_m=np.zeros((35,3)).tolist(),all_joints_rad=arrays)
                clips.append(dict(trial_file=name,trial_sha256=receipt['sha256'],camera_sha256=source['camera_sha256'],mode=mode,batch=index,
                    fps=30,start_frame=0,end_frame=1,frames=[frame],rows=rows,actions=actions,release_step=720))
        order=contract.batch_plan(contract.make_protocol(),seed)[0]['site_indices']
        cp=checkpoint('before-0000');proposed=sweep.propose([self.splits['train'][i] for i in order])
        normalized=geometry_prior(self.splits['train'])['actions_normalized'];normalized[order]=proposed['actions_normalized']
        trial(self.splits['train'],normalized,cp,'train',proposed['proposal_metadata'],request_order=order)
        sweep.observe([dict(metadata,split='train',reward=-1.) for metadata in proposed['proposal_metadata']])
        cp=checkpoint('smoke-final');predicted=sweep.predict(self.splits['validation'])
        trial(self.splits['validation'],predicted['actions_normalized'],cp,'smoke-validation',
            picked=dict(deterministic=True,prediction_metadata=predicted['prediction_metadata']))
        self.trials=saved_trials;self.receipts=receipts
        save(self.run/'clips.json',dict(clips=clips))
        counts=audit.summary(saved_trials[0]['rows'],[-1.]*35)
        progress=dict(schema='sircus-shot-benchmark-progress-v1',experiment_id=experiment['experiment_id'],
            method='sweep',seed=seed,status='completed',source=source,split_manifest_sha256=experiment['split_manifest_sha256'],
            train=counts,learning_curve=[dict(counts,wall_s=1.)],checkpoints=[],updates=0,
            source_trial_sha256=receipts,training_wall_s=1.,elapsed_s=2.,smoke=True,
            resources={phase:dict(wall_s=.2,cpu_s=.1,peak_rss_mib=2.,peak_gpu_mib=0.) for phase in ('headless','render')},
            batches=[dict(index=i,mode=t['mode'],filming=i==0,setup_wall_s=.1,score_wall_s=.05,wall_s=.2) for i,t in enumerate(saved_trials)])
        save(self.run/'progress.json',progress)
        save(self.run/'result.json',dict(status='completed',smoke=True,source=source,study_fingerprint=experiment['protocol']['fingerprint'],
            split_fingerprint=self.splits['fingerprint'],checkpoints=[],updates=0,train=counts,frames=1))

    def test_complete_synthetic_smoke_receipt_is_explicitly_smoke_only(self):
        result=audit.audit_run(self.run,self.plan)
        self.assertEqual(result['status'],'passed')
        self.assertTrue(result['smoke_only'])
        self.assertEqual(result['paid_train_attempts'],35)
        self.assertEqual(result['evaluation_attempts'],35)
        self.assertEqual(result['rescored_releases'],0)
        self.assertTrue(all(t['settlement']['full_articulation_velocity_gate_verified'] for t in result['trials']))

    def test_corrupted_finger_readback_rejected_even_after_trial_rehash(self):
        name='trials/train-00000.json.gz';trial=copy.deepcopy(self.trials[0])
        trial['settling_history'][0][-1]['all_articulation_joint_velocity']['Arm10'][6]=.02
        trial['rows'][0]['settled']=trial['settling_history'][0][-1]
        receipt=compressed(self.run/name,trial)
        progress=json.loads((self.run/'progress.json').read_text());progress['source_trial_sha256'][name]=receipt['sha256'];save(self.run/'progress.json',progress)
        clips=json.loads((self.run/'clips.json').read_text());clips['clips'][0].update(trial_sha256=receipt['sha256'],rows=trial['rows']);save(self.run/'clips.json',clips)
        with self.assertRaisesRegex(ValueError,'joint-speed gate'):
            audit.audit_run(self.run,self.plan)


@unittest.skipIf(continuous_policy.torch is None,'Configured CPU PyTorch required for likelihood audit')
class CPULikelihoodTests(unittest.TestCase):
    def setUp(self):
        from shot_benchmark_methods import make_cpu_ppo
        self.cells=contract.make_splits([[2,2]])['train']
        self.model=make_cpu_ppo(contract.SEEDS[0])

    def packet(self,deterministic=False):
        torch=continuous_policy.torch
        observations=torch.as_tensor(continuous_policy.build_observations(self.cells),dtype=torch.float32)
        picked=continuous_policy.sample(self.model,observations,
            generator=torch.Generator().manual_seed(77),deterministic=deterministic)
        return dict(cells=self.cells,actions_normalized=picked['actions_normalized'].tolist(),
            learner_packet=dict(deterministic=deterministic,latent=picked['latent'].tolist(),logp=picked['logp'].tolist()))

    def test_cpu_policy_likelihood_and_independent_gaussian_arithmetic_agree(self):
        error=audit.verify_policy_sample(self.model,self.packet(),deterministic=False)
        self.assertLessEqual(error,audit.LOGP_TOLERANCE)
        self.assertLessEqual(audit.verify_policy_sample(self.model,self.packet(True),deterministic=True),audit.LOGP_TOLERANCE)

    def test_saved_logp_or_normalized_action_tampering_rejected(self):
        trial=self.packet();trial['learner_packet']['logp'][0]+=.1
        with self.assertRaisesRegex(ValueError,'likelihood'):audit.verify_policy_sample(self.model,trial,deterministic=False)
        trial=self.packet();trial['actions_normalized'][0][0]+=.1
        with self.assertRaisesRegex(ValueError,'sampled latent'):audit.verify_policy_sample(self.model,trial,deterministic=False)

    def test_random_sample_cannot_be_relabeled_as_frozen_mean_prediction(self):
        trial=self.packet();trial['learner_packet']['deterministic']=True
        with self.assertRaisesRegex(ValueError,'deterministic policy mean'):
            audit.verify_policy_sample(self.model,trial,deterministic=True)


if __name__=='__main__':unittest.main()
