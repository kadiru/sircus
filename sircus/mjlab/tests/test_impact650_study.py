"""Synthetic orchestration: inclusive budget, frozen selection and control closure."""
import json
import math
from pathlib import Path
import tempfile
from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from sircus.mjlab import impact650_contract as p,impact650_study as study
from sircus.mjlab.corrected_contract import sha
from sircus.mjlab.tests.test_impact650_contract import sources
from sircus.mjlab.tests.test_xy650_study import SCENE,sources as old_sources,GeometryFixture,ExecutorFixture


class ImpactFixture(GeometryFixture):
    def check_many(self,requests):
        rows=super().check_many(requests)
        for g in rows:
            x,y,z,t,gap=g['values'];u,v=g['impact_offset_m'];bx,by=g['bucket_xy_m']
            length=math.hypot(bx-x,by-y);dx,dy=(bx-x)/length,(by-y)/length
            c,s=math.cos(math.radians(t)),math.sin(math.radians(t))
            point=[x+u*c*dx-v*dy,y+u*c*dy+v*dx,z-u*s]
            release=[*point[:2],point[2]+gap+.0375]
            g.update(impact_point_world_m=point,release_center_m=release,release_center_height_m=release[2],
                     impact_to_bucket_horizontal_m=math.dist(point[:2],[bx,by]))
        return rows


class ImpactStudyTests(unittest.TestCase):
    def fixture(self,fail=False):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);root=Path(temp.name)
        parent=root/'parent';parent.mkdir();assets=root/'assets';assets.mkdir();(root/'asset.txt').write_text('synthetic asset')
        for n in ('calibration.json','verification.json'):(parent/n).write_text('{}')
        calibration=dict(calibration_sha256='a'*64,verification_sha256='b'*64,score_config_sha256='c'*64,score_config={})
        spec=dict(scene=SCENE,calibration=calibration,packages={'synthetic':'fixture'},asset_sha256={'asset.txt':sha(root/'asset.txt')},
                  scene_fingerprint='d'*64,fingerprint='e'*64)
        controls=p.xy.controls(old_sources());geometry=ImpactFixture();made=[]
        args=SimpleNamespace(parent_study=parent,parent_audit=root/'audit.json',assets=assets,output=root/'evidence',batch=32,wall_time_limit=3600)
        def executor(*a):
            result=ExecutorFixture(*a,fail_initial=fail);made.append(result);return result
        with patch.object(p,'load_inputs',return_value=(spec,sources(),controls,{})), \
             patch.object(study.importlib.metadata,'version',return_value='fixture'), \
             patch('sircus.mjlab.learning_solve.required_asset_paths',return_value=['asset.txt']), \
             patch.object(study,'source_manifest',return_value={'synthetic.py':'f'*64}), \
             patch.object(study,'storage_check',return_value={'synthetic':True}), \
             patch('sircus.mjlab.impact650_geometry.ImpactGeometry',return_value=geometry), \
             patch('sircus.mjlab.impact650_native.NativeExecutor',side_effect=executor),patch('builtins.print'):
            report=study.run(args)
        return args.output,report,made[0]

    def test189_inclusive_requests_diverse_selection_and_byte_manifest(self):
        output,report,executor=self.fixture()
        self.assertEqual([len(c) for c in executor.calls],[12,135,30,12])
        self.assertEqual(report['counts']['attempts'],189)
        self.assertEqual(report['gate']['qualified_settings'],5)
        self.assertEqual(report['gate']['capture_qualified_settings'],5)
        selected=json.loads((output/'verification-selection.json').read_text())
        self.assertEqual([r['family_id'] for r in selected[:3]],['family-00','family-01','family-02'])
        for actual,source in zip(report['verification'],selected):
            self.assertEqual(actual['impact_offset_m'],source['impact_offset_m'])
            self.assertEqual(actual['release_center_height_m'],source['release_center_height_m'])
            self.assertTrue(actual['qualification']['source_capture'])
        stage=json.loads((output/'verification-stage.json').read_text())
        prefix=''.join(json.dumps(r,allow_nan=False)+'\n' for r in executor.rows[:147]).encode()
        import hashlib
        self.assertEqual(stage['prior_records_sha256'],hashlib.sha256(prefix).hexdigest())
        for name,value in json.loads((output/'evidence-manifest.json').read_text()).items():self.assertEqual(sha(output/name),value)
        self.assertEqual(len(report['trace_examples']),17)

    def test_control_failure_blocks_all135_search_and30_verification_requests(self):
        output,report,executor=self.fixture(fail=True)
        self.assertEqual(len(executor.rows),12)
        self.assertEqual(report['gate']['status'],'blocked_controls')
        self.assertEqual(report['verification'],[])
        self.assertEqual(json.loads((output/'verification-plan.json').read_text()),[])
        self.assertFalse((output/'verification-stage.json').exists())

    def test_worker_rejects_over_one_hour_before_gpu_or_output(self):
        args=['worker','--parent-study','p','--parent-audit','a','--assets','assets','--output','out','--gpu-lock','lock','--wall-time-limit','3601']
        with patch('sys.argv',args),patch('sircus.mjlab.gpu_slot.gpu_slot') as slot:
            with self.assertRaisesRegex(ValueError,'30..3600'):study.main()
            slot.assert_not_called()

    def test_acquired_slot_timer_reserves_cleanup_and_is_always_disarmed(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'new'
            args=['worker','--parent-study','p','--parent-audit','a','--assets','assets','--output',str(output),
                  '--gpu-lock','lock','--wall-time-limit','120']
            with patch('sys.argv',args),patch('sircus.mjlab.gpu_slot.gpu_slot',return_value=nullcontext()), \
                 patch.object(study,'run',return_value={}),patch.object(study.signal,'signal') as handler, \
                 patch.object(study.signal,'setitimer') as timer:
                study.main()
                self.assertEqual([c.args[1] for c in timer.call_args_list],[110,0])
                callback=handler.call_args.args[1]
                with self.assertRaises(TimeoutError):callback(None,None)
            self.assertEqual(json.loads(output.with_name('new-budget.json').read_text())['limit_s'],120)


if __name__=='__main__':unittest.main()
