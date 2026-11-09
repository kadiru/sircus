"""Frozen run ledger and launch boundaries; no simulator or GPU is imported."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch

import numpy as np
SOURCE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SOURCE))
import candidate_native_runner as runner
from continuous_feasibility import canonical_hash,decode_action


class FrozenRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.dest=self.root/"run";self.dest.mkdir()
        assets=self.root/"assets";assets.mkdir()
        xml='<robot>'+''.join(f'<joint name="joint{i}" type="revolute"><origin xyz="0 0 0" rpy="0 0 0"/><axis xyz="0 0 1"/><limit lower="-6.28" upper="6.28" effort="1"/></joint>' for i in range(1,7))+'</robot>'
        for file in ('dropper.urdf','board_arm.urdf'):(assets/file).write_text(xml)
        for file in ('selected-camera.json','probes.json','rim-rays.json'):(self.dest/file).write_text('{}')
        cp=self.root/'parent.pt';cp.write_bytes(b'synthetic frozen model')
        self.cells=[dict(id=f'train-{i:03d}',bucket_xy_m=[.1+i*.01,-.1],split='train') for i in range(35)]
        actions=[decode_action(c,[0,.4,.35,.24,.5],c['id']) for c in self.cells]
        self.plan=dict(fingerprint='f'*64,cells=self.cells,native_binding=dict(original_source=dict(geometry_fingerprint='g'*64,camera_sha256=runner.digest(self.dest/'selected-camera.json')),checkpoint_path=str(cp),checkpoint_sha256=runner.digest(cp),source_scene_sha256=canonical_hash({}),source_commit='a'*40,source_archive_sha256='b'*64,split_manifest_sha256='c'*64),cohorts=[dict(method=m,actions=actions,native_enabled=[True]*35,decline_reasons=[None]*35,ik_seeds=[None]*35) for m in ('parent','candidate_bank')])
        self.path=self.root/'plan.json';self.path.write_text(json.dumps(self.plan))
        self.plan['input_bindings']={'original':{'split_manifest_sha256':'c'*64}}
        self.path.write_text(json.dumps(self.plan))
        self.env=types.SimpleNamespace(n=35,cfg={},spec={},dest=self.dest,args=types.SimpleNamespace(assets=str(assets)),sim=types.SimpleNamespace(device='cpu'),origins=np.zeros((35,3)),dt=1/240,steps=0,setup_wall_s=0.)
        self.calls=[]
        def home():return dict(synthetic=True)
        def shot_batch(actions,**kwargs):
            self.calls.append((self.env.trial_mode,copy.deepcopy(actions),kwargs))
            self.env.qualifications=[dict(accepted=False,reason='synthetic')]*35
            self.env.path_reasons=[None]*35;self.env.path_checks=[None]*35
            self.env.scoring_normals=np.tile([0,0,1],(35,1));self.env.release_step=self.env.steps
            self.env.transit_history=[];self.env.correction_history=[]
            rows=[dict(invalid='synthetic',route_success=False,route_quality=0,external_contact_steps=0) for _ in range(35)]
            return rows,dict(wall_s=0.,settling=[],trajectory_m=[])
        self.env.home=home;self.env.shot_batch=shot_batch
        self.torch=types.SimpleNamespace(Tensor=type('FakeTensor',(),{}),tensor=lambda *a,**k:None,manual_seed=lambda *a:None,set_float32_matmul_precision=lambda *a:None,
            backends=types.SimpleNamespace(cuda=types.SimpleNamespace(matmul=types.SimpleNamespace(allow_tf32=True)),cudnn=types.SimpleNamespace(allow_tf32=True)),
            cuda=types.SimpleNamespace(reset_peak_memory_stats=lambda:None,synchronize=lambda:None,mem_get_info=lambda:(100,200),max_memory_allocated=lambda:30,max_memory_reserved=lambda:40))

    def execute(self):
        def write(path,value):Path(path).write_text(json.dumps(value,indent=2))
        with patch.dict(sys.modules,{'torch':self.torch}),patch.object(runner,'validate_comparison'),patch.object(runner,'reward',return_value=-1.),patch.object(runner.os,'chown'),patch.object(runner,'atomic',side_effect=write):
            runner.run_comparison(self.env,dict(spec=dict(fingerprint='g'*64,cells=self.cells)),self.path,process_started=time.monotonic()-1)

    def test_exact_two_cohorts_seventy_requests_and_frozen_model(self):
        before=(self.root/'parent.pt').read_bytes();self.execute()
        value=json.loads((self.dest/'result.json').read_text())
        self.assertEqual([x[0] for x in self.calls],['parent','candidate_bank'])
        self.assertTrue(all(x[2]['record_video'] is False for x in self.calls))
        self.assertEqual(value['native_requests'],70);self.assertEqual(value['optimizer_updates'],0)
        self.assertEqual([x['summary']['rejected'] for x in value['trials']],[35,35])
        self.assertEqual((self.root/'parent.pt').read_bytes(),before)
        self.assertEqual(value['checkpoint_sha256_before'],value['checkpoint_sha256_after'])

    def test_checkpoint_or_camera_mismatch_stops_before_any_shot(self):
        (self.root/'parent.pt').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'model bytes'):self.execute()
        self.assertEqual(self.calls,[])

    def test_entry_returns_before_any_historical_training_branch(self):
        tree=ast.parse((SOURCE/'film_cylinder_farm.py').read_text())
        calls=[node for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='run_comparison']
        self.assertEqual(len(calls),1)
        parents=[n for n in ast.walk(tree) if isinstance(n,ast.If) and calls[0] in list(ast.walk(n)) and any(isinstance(x,ast.Return) for x in n.body)]
        self.assertTrue(parents)
        source=(SOURCE/'candidate_native_runner.py').read_text()
        self.assertNotIn('torch.optim',source);self.assertNotIn('load_checkpoint(',source)


if __name__=='__main__':unittest.main()
