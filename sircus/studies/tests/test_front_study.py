import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from sircus.studies import front_study as study
from sircus.studies.profiles import front_scene,historical_scene,profile_metadata
from sircus.studies.run_front_study import cluster_summary,challenge_manifest,layout_interval,curves
from physics import rotation


class FrontStudyTests(unittest.TestCase):
    def model_and_manifest(self):
        used=set();manifest={key:study.sample_layouts(42,key,n,used=used) for key,n in [('train',8),('validation',2),('test',2)]}
        x=[r['observation'] for r in manifest['train']]
        y=[[0,-30,0,0,220] for _ in x]
        return study.fit_model(x,y,1.,1),manifest

    def test_front_axes_and_preserved_assets(self):
        front=front_scene();old=historical_scene();normal=np.asarray(rotation(front['board']['rpy_deg']))@np.array([0,0,1])
        self.assertLess(normal[1],-.49)
        self.assertGreater(old['bin']['center_xy_mm'][1],0)
        self.assertEqual(front['board']['size_mm'],[200,300,5])
        self.assertEqual(front['bin']['height_interpretation'],'unresolved')
        self.assertEqual(front['robot14'],old['robot14'])
        self.assertFalse(profile_metadata()['hardware_enabled'])
        self.assertEqual(front['pickup_mm'],[150,-600,200])
        self.assertIn('proxy',front['measurement_status']['loading_pose'])

    def test_sampler_reproducible_disjoint_nested(self):
        used=set();a=study.sample_layouts(4,'train',8,used=used);b=study.sample_layouts(4,'test',8,used=used)
        self.assertEqual(a,study.sample_layouts(4,'train',8))
        self.assertEqual(a[:4],study.sample_layouts(4,'train',4))
        self.assertFalse({r['layout_sha256'] for r in a}&{r['layout_sha256'] for r in b})
        for row in a+b:self.assertTrue(study.validate_layout(row['scene'])['in_distribution'])

    def test_historical_and_modified_assets_rejected(self):
        with self.assertRaises(ValueError):study.validate_layout(historical_scene())
        scene=front_scene();scene['ball']['mass_kg']=.4
        with self.assertRaises(ValueError):study.validate_layout(scene)
        scene=front_scene();scene['bin']['center_xy_mm'][1]=-600
        with self.assertRaises(ValueError):study.validate_layout(scene)
        self.assertFalse(study.validate_layout(scene,allow_ood=True)['in_distribution'])

    def test_teacher_cannot_see_held_out(self):
        row=study.sample_layouts(4,'test',1)[0]
        with self.assertRaises(ValueError):study.search_episode(row,2)

    def test_prediction_is_frozen_matrix_not_search(self):
        model,manifest=self.model_and_manifest();checkpoint=study.checkpoint(model,[r['scene_id'] for r in manifest['train']],manifest,{})
        before=study.digest(checkpoint)
        with patch.object(study,'search_episode',side_effect=AssertionError('Search forbidden')),patch.object(study,'evaluate_action',side_effect=AssertionError('Evaluation forbidden')):
            predicted=study.predict(checkpoint,manifest['test'][0]['scene'])
        self.assertAlmostEqual(predicted['action']['gap_mm'],220)
        self.assertEqual(study.digest(checkpoint),before)

    def test_checksum_source_and_split_leakage_guards(self):
        model,manifest=self.model_and_manifest();cp=study.checkpoint(model,[manifest['train'][0]['scene_id']],manifest,{})
        tampered=copy.deepcopy(cp);tampered['model']['weights'][0][0]=50
        with self.assertRaises(ValueError):study.validate_checkpoint(tampered)
        with patch.object(study,'study_contract',return_value={'changed':True}):
            with self.assertRaises(ValueError):study.validate_checkpoint(cp)
        manifest['test'][0]=copy.deepcopy(manifest['train'][0])
        with self.assertRaises(ValueError):study.checkpoint(model,[manifest['train'][0]['scene_id']],manifest,{})

    def test_train_normalization_does_not_use_test(self):
        model,manifest=self.model_and_manifest()
        expected=np.mean([r['observation'] for r in manifest['train']],axis=0)
        np.testing.assert_allclose(model['mean'],expected)
        with self.assertRaises(ValueError):study.checkpoint(model,[manifest['test'][0]['scene_id']],manifest,{})

    def test_invalid_ik_is_counted_failure(self):
        with patch.object(study,'feasibility',return_value={'arm_reachable':False,'board_arm_reachable':True}),patch.object(study,'rollout',side_effect=AssertionError('No rollout when unreachable')):
            row=study.evaluate_action(front_scene(),study.FIXED_ACTION)
        self.assertFalse(row['result']['success']);self.assertEqual(study.summarize([row])['endpoint_infeasible'],1)
        action=dict(study.FIXED_ACTION,gap_mm=10000)
        self.assertEqual(study.evaluate_action(front_scene(),action)['result']['outcome'],'invalid_action')

    def test_challenge_repeats_clustered_and_paired(self):
        groups=challenge_manifest(4,4);noises=[g for g in groups if g['kind']=='release_noise']
        self.assertEqual([r['rollout_seed'] for r in noises[0]['episodes']],[r['rollout_seed'] for r in noises[-1]['episodes']])
        self.assertEqual(len({r['layout_sha256'] for r in noises[0]['episodes']}),4)
        rows=[{'layout_sha256':str(i//3),'result':{'success':True,'outcome':'captured'},'feasibility':{'arm_reachable':True,'board_arm_reachable':True}} for i in range(12)]
        summary=cluster_summary(rows,1)
        self.assertEqual(summary['episodes'],12);self.assertEqual(summary['layouts'],4)
        self.assertLess(summary['ci95'][0],1)
        self.assertEqual(layout_interval([1]*4,1),summary['ci95'])
        self.assertGreater(layout_interval([0]*4,1)[1],0)

    def test_reject_unbounded_curve_before_output_created(self):
        with self.assertRaises(ValueError):curves('/tmp/never-created-sircus-study',sizes=(128,8))
        with self.assertRaises(ValueError):curves('/tmp/never-created-sircus-study',seeds=(1,2,3,4))


if __name__=='__main__':unittest.main()
