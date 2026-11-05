"""Portable decoder trust and all-context equivalence checks; synthetic fixtures."""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import continuous_policy as policy
import feasibility_guided_decoder as decoder
from continuous_feasibility import ACTION_NAMES, canonical_hash
from feasibility_guided_support import physical_actions
from test_feasibility_guided_audit import fixture, seal
import test_feasibility_guided_audit as audit_fixtures


def portable_fixture():
    support, _, spec = fixture()
    cells=[dict(id=r['cell_id'],bucket_xy_m=r['bucket_xy_m'],status=r['status'],
                support_available=r['status']=='supported',low=r['low'],high=r['high']) for r in support['cells']]
    seeds={r['cell_id']:({n:r['anchor_ik'][n]['q'] for n in ('Arm10','Arm14')} if r['status']=='supported' else None) for r in support['cells']}
    value=dict(schema=decoder.SCHEMA,policy_version=decoder.POLICY_VERSION,
               geometry_fingerprint=spec['fingerprint'],study_fingerprint='a'*64,domain_fingerprint='b'*64,
               source_support_fingerprint=support['fingerprint'],source_support_sha256='c'*64,source_result_sha256='d'*64,
               action_names=list(ACTION_NAMES),cells=cells,ik_seed_joint_rad=seeds,
               checkpoints={stage:dict(file=f'policy-{stage}.pt',sha256='e'*64,weight_hash='f'*64,
                                      actor_weight_hash='1'*64,update_count=0,attempted_shots=0) for stage in ('initial','final')})
    value['decoder_core_fingerprint']=canonical_hash(decoder._decoder_core(value))
    value['seed_core_fingerprint']=canonical_hash(seeds)
    seal(value)
    return support,spec,value


class PortableDecoderGuards(unittest.TestCase):
    def setUp(self):
        self.support,self.spec,self.decoder=portable_fixture()
        self.fp=self.decoder['fingerprint']

    def test_expected_manifest_fingerprint_is_mandatory_and_distinct(self):
        self.assertNotEqual(self.fp,self.decoder['source_support_fingerprint'])
        with self.assertRaises(TypeError):decoder.validate_decoder(self.decoder)
        with self.assertRaises(ValueError):decoder.validate_decoder(self.decoder,expected_fingerprint=None)
        with self.assertRaises(ValueError):decoder.validate_decoder(self.decoder,expected_fingerprint='0'*64)
        decoder.validate_decoder(self.decoder,expected_fingerprint=self.fp)

    def test_resigned_compact_file_cannot_replace_trusted_manifest_identity(self):
        bad=copy.deepcopy(self.decoder);bad['cells'][1]['high'][3]-=.001
        bad['decoder_core_fingerprint']=canonical_hash(decoder._decoder_core(bad));seal(bad)
        with self.assertRaisesRegex(ValueError,'Trusted portable decoder'):
            decoder.validate_decoder(bad,expected_fingerprint=self.fp)

    def test_decoder_and_seed_core_hashes_are_independent(self):
        for target in ('decode','seed'):
            bad=copy.deepcopy(self.decoder)
            if target=='decode':bad['cells'][1]['low'][2]+=.001
            else:bad['ik_seed_joint_rad'][bad['cells'][1]['id']]['Arm10'][0]+=.01
            seal(bad)
            with self.subTest(target=target),self.assertRaisesRegex(ValueError,'core changed'):
                decoder.validate_decoder(bad,expected_fingerprint=bad['fingerprint'])

    def test_unknown_or_shifted_context_never_uses_nearest_known_bounds(self):
        good=self.spec['cells'][1]
        for bad in (dict(good,id='unknown'),dict(good,bucket_xy_m=[good['bucket_xy_m'][0]+1e-5,good['bucket_xy_m'][1]])):
            with self.subTest(bad=bad),self.assertRaisesRegex(ValueError,'Unknown bucket context'):
                decoder.decode_actions(np.zeros((1,5)),self.decoder,[bad],expected_fingerprint=self.fp)

    def test_unresolved_is_disabled_unless_explicit_accounting_override(self):
        cells=self.spec['cells'];values=np.zeros((35,5))
        self.assertFalse(self.decoder['cells'][0]['support_available'])
        with self.assertRaisesRegex(ValueError,'Unresolved geometry support'):
            decoder.decode_actions(values,self.decoder,cells,expected_fingerprint=self.fp)
        with self.assertRaisesRegex(ValueError,'no IK warm start'):
            decoder.ik_seeds(self.decoder,[cells[0]],expected_fingerprint=self.fp)
        actual=decoder.decode_actions(values,self.decoder,cells,expected_fingerprint=self.fp,allow_unresolved=True)
        np.testing.assert_array_equal(actual,physical_actions(values,self.support,cells))
        self.assertIsNone(decoder.ik_seeds(self.decoder,[cells[0]],expected_fingerprint=self.fp,allow_unresolved=True)[0])

    def test_available_context_decode_is_continuous_and_seed_is_copy(self):
        cells=self.spec['cells'][1:]
        raw=np.zeros((34,5))
        middle=decoder.decode_actions(raw,self.decoder,cells,expected_fingerprint=self.fp)
        tiny=decoder.decode_actions(raw+1e-6,self.decoder,cells,expected_fingerprint=self.fp)
        self.assertTrue((tiny>middle).all())
        seeds=decoder.ik_seeds(self.decoder,cells,expected_fingerprint=self.fp)
        seeds[0]['Arm10'][0]=99
        self.assertEqual(self.decoder['ik_seed_joint_rad'][cells[0]['id']]['Arm10'][0],0)

    def test_falsely_enabled_unresolved_or_extra_fields_are_rejected(self):
        for mode in ('enabled','extra'):
            bad=copy.deepcopy(self.decoder)
            if mode=='enabled':bad['cells'][0]['support_available']=True
            else:bad['cells'][1]['raw_observation']=[1,2,3]
            bad['decoder_core_fingerprint']=canonical_hash(decoder._decoder_core(bad));seal(bad)
            with self.subTest(mode=mode),self.assertRaises(ValueError):
                decoder.validate_decoder(bad,expected_fingerprint=bad['fingerprint'])


@unittest.skipIf(policy.torch is None,'Configured CPU Torch unavailable')
class PortableExportChecks(unittest.TestCase):
    def run_fixture(self,folder):
        audit_fixtures.GuidedChainAuditTests().make_run(folder)

    def test_export_all_policy_contexts_including_inactive_and_original_model_guards(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)/'raw';self.run_fixture(root)
            output=Path(name)/'portable'
            artifact,receipt=decoder.export(root,output)
            self.assertEqual(receipt['policy_trial_batches'],8)
            self.assertEqual(receipt['verified_decoded_contexts'],280)
            self.assertEqual(receipt['verified_active_contexts'],240)
            self.assertEqual(receipt['verified_inactive_contexts'],40)
            self.assertEqual(receipt['excluded_coverage_batches'],0)
            self.assertTrue(receipt['raw_and_compact_affine_outputs_bit_identical'])
            self.assertEqual(receipt['decoder_sha256'],decoder.digest(output/'decoder.json'))
            self.assertEqual(set(p.name for p in output.iterdir()),{'decoder.json','decoder-export-audit.json','policy-initial.pt','policy-final.pt'})
            loaded,saved,portable=decoder.load_policy(output/'policy-final.pt',output/'decoder.json',expected_decoder_fingerprint=artifact['fingerprint'])
            self.assertEqual(saved['support_fingerprint'],artifact['source_support_fingerprint'])
            self.assertNotEqual(saved['support_fingerprint'],artifact['fingerprint'])
            self.assertEqual(saved['update_count'],1)
            with self.assertRaises(ValueError):decoder.export(root,output)
            with self.assertRaises(ValueError):decoder.load_policy(output/'policy-initial.pt',output/'decoder.json',expected_decoder_fingerprint=artifact['fingerprint'],stage='final')

    def test_inactive_context_action_corruption_is_not_skipped(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);self.run_fixture(root)
            report=decoder.read(root/'result.json');summary=report['batches'][0]
            path=root/summary['file'];trial=decoder.read(path)
            index=next(i for i,flag in enumerate(trial['enabled']) if not flag)
            trial['actions'][index]['gap_m']+=.001
            path.write_text(json.dumps(trial));summary['sha256']=decoder.digest(path)
            (root/'result.json').write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError,'Stored trial action'):
                decoder.build_decoder(root)

    def test_latent_normalized_inconsistency_rejected_even_if_source_rehashed(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);self.run_fixture(root)
            report=decoder.read(root/'result.json');summary=report['batches'][0]
            path=root/summary['file'];trial=decoder.read(path)
            trial['actions_normalized'][0][1]+=.01
            path.write_text(json.dumps(trial));summary['sha256']=decoder.digest(path)
            (root/'result.json').write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError,'latent-to-normalized'):
                decoder.build_decoder(root)

    def test_partial_result_and_changed_checkpoint_rejected_before_export_creation(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name)/'raw';self.run_fixture(root)
            report=decoder.read(root/'result.json');report['status']='running'
            (root/'result.json').write_text(json.dumps(report))
            output=Path(name)/'bad'
            with self.assertRaisesRegex(ValueError,'completed matching'):
                decoder.export(root,output)
            self.assertFalse(output.exists())
            report['status']='completed';(root/'result.json').write_text(json.dumps(report))
            with (root/'policy-final.pt').open('ab') as f:f.write(b'changed')
            with self.assertRaisesRegex(ValueError,'checkpoint bytes'):
                decoder.export(root,output)
            self.assertFalse(output.exists())


if __name__=='__main__':unittest.main()
