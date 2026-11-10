"""Strict dual-source manifests with fabricated files only; no real model/app."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import candidate_v4_inputs as subject
from candidate_v4_evaluation import prepare_evaluation
from candidate_continuation_ledger import canonical_hash
from test_candidate_v4_evaluation import fixture as original_fixture


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')


def synthetic_inputs(root):
    host=Path('/synthetic-v4-workspace')
    source=root/'new-source/sircus/isaac';source.mkdir(parents=True);(source/'only.py').write_text('# synthetic new source\n')
    old=root/'old-source/sircus/isaac';old.mkdir(parents=True);(old/'only.py').write_text('# synthetic old source\n')
    src,prod=subject._source(source),subject._source(old)
    initial=original_fixture();original=copy.deepcopy(initial['original_plan']);e=copy.deepcopy(initial['prior_evidence'])
    original['bindings']['producer_source_sha256']=prod
    original['fingerprint']=canonical_hash({k:v for k,v in original.items() if k!='fingerprint'})
    e['events'][0]['payload']['plan_fingerprint']=original['fingerprint']
    previous='0'*64
    for event in e['events']:
        event['previous_sha256']=previous
        event['fingerprint']=canonical_hash({k:v for k,v in event.items() if k!='fingerprint'});previous=event['fingerprint']
    e['journal_head_sha256']=previous
    protected=root/'protected';assessment=protected/'assessment'
    save(assessment/'assessment-plan.json',original)
    for i,event in enumerate(e['events']):save(assessment/f'journal/event-{i:06d}.json',event)
    for name in e['inventory']:
        p=assessment/name
        if not p.exists():p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'synthetic-only artifact')
    e['inventory']={str(p.relative_to(assessment)):dict(sha256=subject.sha(p),bytes=p.stat().st_size)
        for p in sorted(assessment.rglob('*')) if p.is_file()}
    e['inventory_sha256']=canonical_hash(e['inventory']);e['original_plan_sha256']=subject.sha(assessment/'assessment-plan.json')
    plan=prepare_evaluation(original,e,evaluation_source_sha256=src,synthetic=True)
    bindings=dict(source_sha256=prod)
    paths={
        'prepared_plan':protected/'prepared.json','parent_checkpoint':protected/'parent.pt',
        'final_checkpoint':protected/'final.pt','original_manifest':protected/'original-manifest.json',
        'original_learning_result':protected/'learned.json','original_native_binding':protected/'binding.json',
        'old_assessment_plan':assessment/'assessment-plan.json','old_assessment_inventory':protected/'inventory.json',
        'old_component_summary':protected/'components.json'}
    save(paths['prepared_plan'],plan)
    paths['parent_checkpoint'].write_bytes(b'synthetic parent');paths['final_checkpoint'].write_bytes(b'synthetic V4')
    old_files={}
    for role in ('parent_checkpoint','fresh_split_manifest','train_bank'):
        path=paths['parent_checkpoint'] if role=='parent_checkpoint' else protected/(role+'.json')
        if not path.exists():save(path,{})
        old_files[role]=dict(path=str(host/path.relative_to(root)),sha256=subject.sha(path))
    save(paths['original_manifest'],dict(schema='sircus-candidate-continuation-production-inputs-v1',
        workspace_host=str(host),source_host=str(host/'old-source'),source_sha256=prod,files=old_files,
        protected_roots=[str(host/'protected'),str(host/'old-source')],source_run=str(host/'protected'),
        assets_dir=str(host/'protected'),frozen_experiment_root=str(host/'protected'),
        execution_authorized=False,release_authorized=False))
    save(paths['original_learning_result'],dict(status='learning_complete_audit_pending',synthetic=True,bindings=bindings,checkpoint_sha256=subject.V4_CHECKPOINT_SHA256))
    save(paths['original_native_binding'],dict(fingerprint=original['bindings']['native_binding_sha256']))
    save(paths['old_assessment_inventory'],e['inventory'])
    save(paths['old_component_summary'],dict(status='component_checks_passed_full_run_incomplete',
        full_combined_audit_status='not_run_incomplete',components={k:dict(status='passed') for k in ('learning','native_train','startup')},
        closed_input_sha256=dict(final_checkpoint=subject.V4_CHECKPOINT_SHA256)))
    files={k:dict(path=str(host/p.relative_to(root)),sha256=subject.sha(p)) for k,p in paths.items()}
    manifest=dict(schema=subject.SCHEMA,synthetic=True,workspace_host=str(host),source_host=str(host/'new-source'),
        source_sha256=src,original_source_host=str(host/'old-source'),original_source_sha256=prod,
        files=files,expected_runtime=copy.deepcopy(subject.PINNED_RUNTIME),prepared_plan_fingerprint=plan['fingerprint'],
        protected_roots=[str(host/x) for x in ('protected','new-source','old-source')],
        execution_authorized=False,release_authorized=False)
    path=root/'manifest.json';save(path,manifest)
    geometry=dict(paths=dict(parent_checkpoint=paths['parent_checkpoint']),bindings=bindings)
    return dict(root=root,host=host,source=source,old=old,paths=paths,plan=plan,manifest=manifest,
        path=path,geometry=geometry)


class InputTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.f=synthetic_inputs(Path(self.temp.name));self.source=patch.object(subject,'SOURCE',self.f['source'])
        self.source.start();self.addCleanup(self.source.stop)
    def load(self):
        return subject.load_evaluation_manifest(self.f['path'],subject.sha(self.f['path']),container_root=self.f['root'])
    def rewrite(self):save(self.f['path'],self.f['manifest'])
    def test_original_and_execution_sources_are_distinct_and_preserved(self):
        m,paths,mapped,plan,protected=self.load()
        self.assertNotEqual(m['source_sha256'],m['original_source_sha256'])
        self.assertEqual(plan['requests'],210);self.assertEqual(plan['prior_assessment']['unresolved'],35)
        self.assertEqual(plan['bindings']['producer_source_sha256'],subject._source(self.f['old']))
    def test_new_source_tamper_cannot_be_hidden_by_original_source(self):
        (self.f['source']/'only.py').write_text('# changed new\n')
        with self.assertRaisesRegex(ValueError,'evaluation source'):self.load()
    def test_original_source_tamper_cannot_be_hidden_by_new_source(self):
        (self.f['old']/'only.py').write_text('# changed old\n')
        with self.assertRaisesRegex(ValueError,'original source'):self.load()
    def test_input_byte_change_and_missing_role_fail(self):
        self.f['paths']['final_checkpoint'].write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'input bytes'):self.load()
        self.f['manifest']['files'].pop('final_checkpoint');self.rewrite()
        with self.assertRaisesRegex(ValueError,'role coverage'):self.load()
    def test_manifest_cannot_authorize_native_execution(self):
        self.f['manifest']['execution_authorized']=True;self.rewrite()
        with self.assertRaisesRegex(ValueError,'permission'):self.load()
    def test_unprotected_model_is_rejected(self):
        self.f['manifest']['protected_roots']=[str(self.f['host']/'new-source'),str(self.f['host']/'old-source')];self.rewrite()
        with self.assertRaisesRegex(ValueError,'protected'):self.load()
    def test_extra_old_test_dispatch_and_deleted_partial_evidence_fail(self):
        p=self.f['paths']['old_assessment_plan'].parent/'native/assessment-000006.dispatch.json';p.write_text('{}')
        with self.assertRaisesRegex(ValueError,'Partial assessment'):self.load()
        p.unlink();(self.f['paths']['old_assessment_plan'].parent/'native/assessment-000003.npz').unlink()
        with self.assertRaisesRegex(ValueError,'Partial assessment'):self.load()
    def test_symlink_and_parent_path_escape_fail_closed(self):
        p=self.f['root']/'outside';p.write_text('{}')
        original=self.f['paths']['final_checkpoint'];original.unlink();original.symlink_to(p)
        with self.assertRaisesRegex(ValueError,'Linked'):self.load()
    def test_wrong_pinned_runtime_rejected_without_loading_torch(self):
        self.f['manifest']['expected_runtime']['numpy']='different';self.rewrite()
        with self.assertRaisesRegex(ValueError,'runtime'):self.load()
    def test_synthetic_history_cannot_be_relabelled_as_real(self):
        self.f['manifest']['synthetic']=False;self.rewrite()
        with self.assertRaises(ValueError):self.load()
    def test_geometry_bindings_and_component_proofs_checked_before_return(self):
        with patch.object(subject,'_geometry_inputs',return_value=copy.deepcopy(self.f['geometry'])) as geometry:
            i=subject.production_inputs(self.f['path'],subject.sha(self.f['path']),container_root=self.f['root'])
        self.assertEqual(i['bindings'],self.f['geometry']['bindings'])
        self.assertEqual(i['execution']['source_sha256'],self.f['manifest']['source_sha256'])
        self.assertNotEqual(i['bindings']['source_sha256'],i['execution']['source_sha256'])
        req=subject.model_request(i,self.f['root']/'new-output/model-check.json')
        self.assertEqual(req['files']['final_checkpoint']['sha256'],subject.sha(self.f['paths']['final_checkpoint']))
        self.assertEqual(req['original_source_sha256'],i['bindings']['source_sha256'])
        self.assertEqual(req['fingerprint'],canonical_hash({k:v for k,v in req.items() if k!='fingerprint'}))
        geometry.assert_called_once()
    def test_learning_bindings_cannot_be_rewritten_to_new_source(self):
        g=copy.deepcopy(self.f['geometry']);g['bindings']['source_sha256']=self.f['manifest']['source_sha256']
        with patch.object(subject,'_geometry_inputs',return_value=g),self.assertRaisesRegex(ValueError,'trained result'):
            subject.production_inputs(self.f['path'],subject.sha(self.f['path']),container_root=self.f['root'])
    def test_original_bank_asset_protection_cannot_be_omitted(self):
        g=copy.deepcopy(self.f['geometry']);outside=self.f['root']/'unprotected-bank/input.json'
        outside.parent.mkdir();outside.write_text('{}');g['paths']['bank']=outside
        with patch.object(subject,'_geometry_inputs',return_value=g),self.assertRaisesRegex(ValueError,'protection omitted original'):
            subject.production_inputs(self.f['path'],subject.sha(self.f['path']),container_root=self.f['root'])
    def test_frozen_guard_rechecks_original_manifest_banks_outside_new_direct_roles(self):
        with patch.object(subject,'_geometry_inputs',return_value=copy.deepcopy(self.f['geometry'])):
            i=subject.production_inputs(self.f['path'],subject.sha(self.f['path']),container_root=self.f['root'])
        guard=subject.frozen_verifier(i);self.assertEqual(guard()['model_hashes'],self.f['plan']['model_hashes'])
        bank=self.f['root']/'protected/train_bank.json';bank.write_text('{"changed":true}')
        with self.assertRaisesRegex(ValueError,'input bytes changed'):guard()
    def test_external_manifest_hash_and_duplicate_keys_fail(self):
        with self.assertRaisesRegex(ValueError,'manifest bytes'):
            subject.load_evaluation_manifest(self.f['path'],'0'*64,container_root=self.f['root'])
        self.f['path'].write_text('{"schema":1,"schema":2}')
        with self.assertRaisesRegex(ValueError,'Duplicate'):self.load()


if __name__=='__main__':unittest.main()
