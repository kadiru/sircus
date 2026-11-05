"""Independent artifact audit rejects internally rehashed, altered requests."""

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

SOURCE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SOURCE))
import continuous_feasibility as cf
from continuous_feasibility_audit import audit
from expanded_farm import make_spec as original_spec


def write(path,value):
    path.write_text(json.dumps(value,allow_nan=False))


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.run=self.root/'run';self.run.mkdir()
        self.producer=self.root/'producer';self.producer.mkdir()
        self.assets=self.root/'assets';self.assets.mkdir()
        scene=json.loads((SOURCE/'scene.json').read_text())
        source=original_spec(scene,10)
        # All requests deliberately fail the cheap table-volume gate. This is
        # valid diagnostic data: failed attempts must still survive the audit.
        source['scene']['table']['center_m'][2]=1.0
        self.spec=cf.make_spec(source,1)
        joints=''.join(f'<joint name="joint{i}" type="revolute"><origin xyz="0 0 .1" rpy="0 0 0"/><axis xyz="0 0 1"/><limit lower="-3" upper="3"/></joint>' for i in range(1,7))
        for name in ('dropper.urdf','board_arm.urdf'):(self.assets/name).write_text('<robot>'+joints+'</robot>')
        write(self.assets/'poses.json',dict(dropper_joint_rad=[0]*6,board_joint_rad=[0]*6))
        (self.producer/'continuous_feasibility.py').write_text('# frozen fixture source\n')
        self.result=cf.build_map(self.spec,self.assets)
        write(self.run/'contract.json',self.spec)
        write(self.run/'feasibility.json',self.result)
        write(self.run/'run-provenance.json',dict(cuda_visible_devices='',samples_per_cell=1,
            source_sha256={'continuous_feasibility.py':hashlib.sha256((self.producer/'continuous_feasibility.py').read_bytes()).hexdigest()},assets=self.result['asset_sha256']))

    def rewrite_result(self,result):
        result['fingerprint']=cf.canonical_hash({k:v for k,v in result.items() if k!='fingerprint'})
        write(self.run/'feasibility.json',result)

    def test_complete_all_rejected_map_passes_with_denominators_preserved(self):
        result=audit(self.run,self.producer,self.assets)
        self.assertEqual(result['status'],'passed')
        self.assertEqual(result['requests'],35)
        self.assertEqual(result['qualified'],0)
        self.assertEqual(result['cells_with_qualified'],0)
        self.assertEqual(sum(result['reasons'].values()),35)

    def test_rehashed_action_mutation_is_not_a_new_policy_projection(self):
        result=copy.deepcopy(self.result)
        result['records'][0]['action']['gap_m']+=.001
        self.rewrite_result(result)
        with self.assertRaisesRegex(ValueError,'sampled action'):
            audit(self.run,self.producer,self.assets)

    def test_rehashed_admission_label_mutation_is_rejected(self):
        result=copy.deepcopy(self.result)
        result['records'][0]['accepted']=True
        self.rewrite_result(result)
        with self.assertRaisesRegex(ValueError,'admission/rejection'):
            audit(self.run,self.producer,self.assets)

    def test_rehashed_geometry_evidence_mutation_is_rejected(self):
        result=copy.deepcopy(self.result)
        result['records'][0]['geometry']['metrics']['table_top_z_m']+=.1
        self.rewrite_result(result)
        with self.assertRaisesRegex(ValueError,'geometric gate'):
            audit(self.run,self.producer,self.assets)

    def test_rehashed_missing_request_still_rejected(self):
        result=copy.deepcopy(self.result)
        result['records'].pop()
        self.rewrite_result(result)
        with self.assertRaisesRegex(ValueError,'Missing feasibility'):
            audit(self.run,self.producer,self.assets)

    def test_frozen_producer_and_assets_must_match(self):
        (self.producer/'continuous_feasibility.py').write_text('# changed source\n')
        with self.assertRaisesRegex(ValueError,'producer source changed'):
            audit(self.run,self.producer,self.assets)


if __name__=='__main__':unittest.main()
