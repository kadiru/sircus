"""Synthetic controller checks: exact finite plan and control-failure closure."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from sircus.mjlab import tilt650_contract as p, tilt650_study as study, xy650_contract as xy
from sircus.mjlab.corrected_contract import sha
from sircus.mjlab.learning_contract import digest
from sircus.mjlab.tests.test_xy650_study import SCENE, sources, GeometryFixture, ExecutorFixture


class TiltStudyTests(unittest.TestCase):
    def fixture(self, fail=False):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);root=Path(temp.name)
        parent=root/'parent';parent.mkdir();assets=root/'assets';assets.mkdir()
        (root/'asset.txt').write_text('synthetic identity')
        for n in ('calibration.json','verification.json'):(parent/n).write_text('{}')
        calibration=dict(calibration_sha256='a'*64,verification_sha256='b'*64,score_config_sha256='c'*64,score_config={})
        identities={'asset.txt':sha(root/'asset.txt')}
        spec=dict(scene=SCENE,calibration=calibration,packages={'synthetic':'fixture'},asset_sha256=identities,
                  scene_fingerprint=xy.scene_fingerprint(SCENE,calibration,identities),fingerprint='d'*64)
        original=sources();controls=xy.controls(original)
        selected=[]
        for i,ident in enumerate(p.SOURCE_IDS):
            r=copy.deepcopy(original[1]);r.update(id=ident,values=[.47,-.21,.38,30.,.41],translation_m=[0.,.06],
                                               query_fingerprint='e'*64,route_quality=1,route_success=True,
                                               native_release=True,invalid=None,invalid_physics=False);selected.append(r)
        args=SimpleNamespace(parent_study=parent,parent_audit=root/'audit.json',assets=assets,output=root/'evidence',batch=32,wall_time_limit=1800)
        geometry=GeometryFixture();made=[]
        def executor(*a):
            result=ExecutorFixture(*a,fail_initial=fail);made.append(result);return result
        with patch.object(p,'load_parent',return_value=(spec,selected,controls,{})), \
             patch.object(study.importlib.metadata,'version',return_value='fixture'), \
             patch('sircus.mjlab.learning_solve.required_asset_paths',return_value=['asset.txt']), \
             patch.object(study,'source_manifest',return_value={'synthetic.py':'f'*64}), \
             patch.object(study,'storage_check',return_value={'synthetic':True}), \
             patch('sircus.mjlab.clean650_geometry.CleanGeometry',return_value=geometry), \
             patch('sircus.mjlab.xy650_native.NativeExecutor',side_effect=executor),patch('builtins.print'):
            report=study.run(args)
        return args.output,report,made[0]

    def test_exact54_requests_frozen_joints_and_manifest(self):
        output,report,executor=self.fixture()
        self.assertEqual(len(executor.rows),54)
        self.assertEqual([len(c) for c in executor.calls],[12,30,12])
        self.assertEqual(len(report['verification']),5)
        self.assertEqual(report['gate']['qualified_settings'],5)
        self.assertEqual(report['gate']['capture_qualified_settings'],5)
        self.assertTrue(all(v['parent_route_quality']==1 for v in report['verification']))
        self.assertEqual(len(report['trace_examples']),14)
        manifest=json.loads((output/'evidence-manifest.json').read_text())
        self.assertEqual(manifest,report['evidence_sha256'])
        for name,value in manifest.items():self.assertEqual(sha(output/name),value)
        self.assertEqual(json.loads((output/'source-selection.json').read_text())[0]['route_quality'],1)

    def test_initial_control_failure_keeps30_candidates_closed(self):
        _,report,executor=self.fixture(fail=True)
        self.assertEqual(len(executor.rows),12)
        self.assertEqual(report['gate']['status'],'blocked_controls')
        self.assertEqual(report['gate']['qualified_settings'],0)
        self.assertEqual(report['gate']['capture_qualified_settings'],0)
        self.assertEqual(report['control_gates']['controls_post']['status'],'not_run')
        self.assertTrue(all(v['qualification']['status']=='pending' for v in report['verification']))


if __name__=='__main__':unittest.main()
