"""Synthetic immutable receipts; no GPU, network, optimizer or robot access."""
import copy
import gzip
import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

import shot_benchmark_reach_audit as audit
from continuous_feasibility import decode_action, geometry_check
from expanded_pose_menu import endpoint_targets, outer_bounds
from prepare_assets import chain, fk


class ReachAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name);self.run=self.root/'experiment/runs/sweep';self.run.mkdir(parents=True)
        self.assets=self.root/'assets';self.assets.mkdir();(self.run/'trials').mkdir()
        self.snapshot=self.root/'frozen-progress.json'
        for name in ('dropper.urdf','board_arm.urdf'):
            robot=ET.Element('robot',name='synthetic-test-only')
            for i in range(1,7):
                j=ET.SubElement(robot,'joint',name=f'joint{i}')
                ET.SubElement(j,'origin',xyz='0 0 0.1' if i==1 else '0.15 0 0',rpy='0 0 0')
                ET.SubElement(j,'axis',xyz='0 0 1');ET.SubElement(j,'limit',lower='-6.28',upper='6.28')
            ET.ElementTree(robot).write(self.assets/name)
        self.links={arm:chain(ET.parse(self.assets/file).getroot()) for arm,file in (('Arm10','dropper.urdf'),('Arm14','board_arm.urdf'))}
        self.scene=json.loads((Path(audit.__file__).with_name('scene.json')).read_text())
        self.scene['arms']['base_spacing_m']=1.0
        self.source=dict(commit='a'*40,archive_sha256='b'*64,geometry_fingerprint='c'*64,camera_sha256='d'*64,scoring_version='synthetic-only')
        dependencies={Path(m.__file__).name:audit.sha(m.__file__) for m in audit.DEPENDENCY_MODULES}
        self.binding=dict(source=self.source,source_sha256=dependencies)
        self.write(self.run/'experiment-binding.json',self.binding);self.write(self.run/'contract.json',dict(scene=self.scene))
        cells=[dict(id=f'train-{i:03d}',split='train',bucket_xy_m=[.9+i*.001,0.]) for i in range(35)]
        actions=[decode_action(c,[0.,1.2 if i==0 else .4,.35 if i==0 else .14,.24 if i==0 else .08,np.deg2rad(12.)],f'synthetic-{i}') for i,c in enumerate(cells)]
        rows=[]
        for i,(cell,action) in enumerate(zip(cells,actions)):
            targets=endpoint_targets(self.scene,cell,action)
            bounds=outer_bounds(targets,self.links)
            reason='outer_link_length_bound' if i==0 else 'endpoint_ik_unresolved_Arm10' if i==1 else 'inactive_split'
            qualification=dict(action=action,geometry=geometry_check(self.scene,cell,action),outer_bounds=bounds,reason=reason,accepted=False,ik={})
            if i==1:
                target=targets['Arm10'];current=fk(np.zeros(6),self.links['Arm10'])
                rotation=Rotation.from_matrix(target[:3,:3]@current[:3,:3].T).as_rotvec()
                error=np.r_[current[:3,3]-target[:3,3],.25*rotation]
                qualification['ik']['Arm10']=dict(q=[0.]*6,solved=False,residual=float(np.linalg.norm(error)),position_error_mm=float(np.linalg.norm(error[:3])*1000),orientation_error_deg=float(np.rad2deg(np.linalg.norm(rotation))),attempts=[dict(nfev=40,residual=float(np.linalg.norm(error)))])
                self.assertTrue(bounds['eligible'])
            if i==0:self.assertFalse(bounds['eligible'])
            rows.append(dict(cell_id=cell['id'],split='train',invalid=reason,route_success=False,route_quality=0,qualification=qualification,settled={},settled_board_center_m=[.5,0,.3]))
        self.trial=dict(schema='sircus-shot-benchmark-native-trial-v1',mode='train',batch=0,source=self.source,split_manifest_sha256='e'*64,cells=cells,enabled=[True,True]+[False]*33,rows=rows,actions=actions,training_request_order=[0,1],home=dict(measured_joint_rad={a:[[0.]*6 for _ in range(35)] for a in ('Arm10','Arm14')}),proposal_metadata=[dict(cell_id=cells[i]['id'],bucket_xy_m=cells[i]['bucket_xy_m'],global_request_index=i,phase='synthetic_prior') for i in range(2)])
        self.name='trials/train-00000.json.gz'
        self.progress=dict(schema='sircus-shot-benchmark-progress-v1',smoke=False,method='sweep',seed=20261121,source=self.source,split_manifest_sha256='e'*64,train=dict(attempts=2,rejected=2,captures=0,clean=0,failures={'outer_link_length_bound':1,'endpoint_ik_unresolved_Arm10':1}),source_trial_sha256={},elapsed_s=1.)
        self.save_trial()
        # A held-out file can be unreadable without this TRAIN-only diagnostic
        # ever opening it or using it to choose representative failures.
        held=self.run/'trials/eval-test-00001.json.gz';held.write_bytes(b'not-even-gzip')
        self.progress['source_trial_sha256']['trials/eval-test-00001.json.gz']=audit.sha(held)
        self.write(self.snapshot,self.progress)

    def write(self,path,value):path.write_text(json.dumps(value))
    def save_trial(self):
        with gzip.open(self.run/self.name,'wt') as stream:json.dump(self.trial,stream)
        self.progress['source_trial_sha256'][self.name]=audit.sha(self.run/self.name)
        self.write(self.snapshot,self.progress)
    def result(self):return audit.audit(self.snapshot,self.run,self.assets)

    def test_reconciles_all_active_attempts_and_never_reads_held_out_or_inactive(self):
        result=self.result()
        self.assertEqual(result['schema'],audit.SCHEMA)
        self.assertEqual(result['totals']['attempts'],2);self.assertEqual(result['totals']['rejected'],2)
        self.assertEqual(result['totals']['released'],0);self.assertEqual(result['snapshot']['trial_count'],1)
        self.assertEqual(sum(c['count'] for c in result['categories']),2)
        self.assertEqual(sum(c['attempts'] for c in result['cells']),2)
        self.assertEqual({c['id'] for c in result['categories']},{'outer_reach','dropper_ik'})
        for bins in result['parameter_bins'].values():self.assertEqual(sum(b['attempts'] for b in bins),2)
        self.assertEqual(len(result['cases']),2)
        self.assertEqual(result['gate_stats']['Arm14_not_evaluated_after_Arm10_failure'],1)
        self.assertEqual(len(result['probe_candidates']),1)

    def test_requested_and_candidate_coordinates_use_common_arm10_frame(self):
        result=self.result();case=next(c for c in result['cases'] if c['reason']=='endpoint_ik_unresolved_Arm10')
        np.testing.assert_allclose(case['arms']['Arm10']['pivot_world_m'],[0,0,.1],atol=1e-12)
        np.testing.assert_allclose(case['arms']['Arm14']['pivot_world_m'],[1,0,.1],atol=1e-12)
        target=endpoint_targets(self.scene,self.trial['cells'][1],self.trial['actions'][1])
        base=audit.bases_for(self.scene)['Arm14']
        np.testing.assert_allclose(case['arms']['Arm14']['target_world_m'],(base@target['Arm14'])[:3,3],atol=1e-12)
        np.testing.assert_allclose(case['arms']['Arm10']['best_joint_points_world_m'][-1],[.75,0,.1],atol=1e-12)
        self.assertFalse(case['native_release']);self.assertIn('numerical',case['executed_pose_scope'])
        self.assertIn('Not evaluated',case['arms']['Arm14']['candidate_scope'])
        self.assertTrue(case['arms']['Arm10']['within_outer_bound'])
        self.assertFalse(case['arms']['Arm10']['ik_solved'])

    def test_changed_trial_bytes_or_snapshot_totals_are_rejected(self):
        original=(self.run/self.name).read_bytes();(self.run/self.name).write_bytes(original+b'corrupt')
        with self.assertRaisesRegex(ValueError,'trial bytes changed'):self.result()
        (self.run/self.name).write_bytes(original)
        self.progress['train']['attempts']=3;self.write(self.snapshot,self.progress)
        with self.assertRaisesRegex(ValueError,'counts differ'):self.result()

    def test_false_outer_rejection_and_wrong_saved_ik_residual_are_rejected(self):
        self.trial['rows'][1]['qualification']['reason']='outer_link_length_bound';self.save_trial()
        with self.assertRaisesRegex(ValueError,'no violated necessary bound'):self.result()
        self.trial['rows'][1]['qualification']['reason']='endpoint_ik_unresolved_Arm10'
        self.trial['rows'][1]['qualification']['ik']['Arm10']['residual']+=.01;self.save_trial()
        with self.assertRaisesRegex(ValueError,'residual differs'):self.result()

    def test_duplicate_request_index_and_wrong_source_dependency_are_rejected(self):
        self.trial['proposal_metadata'][1]['global_request_index']=0;self.save_trial()
        with self.assertRaisesRegex(ValueError,'request indices'):self.result()
        self.trial['proposal_metadata'][1]['global_request_index']=1;self.save_trial()
        self.binding['source_sha256']['prepare_assets.py']='f'*64;self.write(self.run/'experiment-binding.json',self.binding)
        with self.assertRaisesRegex(ValueError,'dependency differs'):self.result()

    def test_representative_selection_is_deterministic_and_not_reward_ranked(self):
        a=self.result();b=self.result()
        self.assertEqual(a['cases'],b['cases'])
        self.assertEqual(a['probe_candidates'],b['probe_candidates'])
        self.assertNotIn('reward',a['cases'][0])
        self.assertEqual(sum(c['count'] for c in a['categories']),a['totals']['rejected'])
        self.assertEqual(a['unresolved_ik_summary']['Arm10']['nfev_total']['median'],40)


if __name__=='__main__':unittest.main()
