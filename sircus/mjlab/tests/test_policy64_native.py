"""Mock-only geometry-before-policy integration; no native/GPU allocation."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from sircus.mjlab import policy64_native as native
from sircus.mjlab.learning_contract import digest


def source():
    return dict(id='shared',values=[.5,-.2,.3,30.,.4],bucket_xy_m=[.3,-.6],q10=[.1]*6,q14=[.2]*6,
                source_record_sha256='a'*64,route_success=True,reward=99.)


class Adapter:
    def __init__(self,ref,assets,directory,**kwargs):
        self.directory=Path(directory);self.kwargs=copy.deepcopy(kwargs);self.resolve_calls=[];self.calls=[];self.storage_calls=0
        self.env=None
        self.source=copy.deepcopy(kwargs['prior_records'][0]);self.mode=None
        proof=self.directory/'contexts/proof.json';proof.parent.mkdir();proof.write_text('{"geometry_only":true}\n')
        effective=copy.deepcopy(self.source);effective.update(values=[.5,-.2,.325,30.,.4],derivation_offset=[0.,0.,.025,0.,0.])
        self.resolution=dict(source_hint=copy.deepcopy(self.source),effective_source_hint=effective,file='contexts/proof.json',sha256=native.sha(proof),
            home_sha256='b'*64,layout_eligible=True,hard_reject=False,anchor_admitted=True,selection_uses_flight_labels=False)
    def resolve_context(self,c):
        self.resolve_calls.append(copy.deepcopy(c));return copy.deepcopy(self.resolution),dict(wall_s=.1,geometry_s=.08,path_s=.02)
    def storage_check(self):self.storage_calls+=1
    def evaluate(self,requests,phase):
        self.calls.append((copy.deepcopy(requests),phase));rows=[]
        if self.mode!='rejected' and self.env is None:
            self.env=SimpleNamespace(effective_options=dict(version='sircus-calibrated-native-warp-options-v1',
                substeps=64,native=dict(timestep=1/15360),warp=dict(timestep=float(1/15360))))
        for r in requests:
            row=dict(id=r['id'],bucket_xy_m=r['bucket_xy_m'],values=r['values'],phase=phase,action_substituted=False,
                     native_release=self.mode!='rejected',invalid='setup_rejected' if self.mode=='rejected' else None,
                     route_success=False,route_quality=0,contact_protocol=dict(substeps=64),scoring_version=native.SCORER_VERSION,
                     sentinel_native={'actual':True})
            if self.mode=='wrong_action':row['values']=[10.]*5
            if self.mode=='substituted':row['action_substituted']=True
            if self.mode=='wrong_resolution':row['contact_protocol']['substeps']=128
            if self.mode=='wrong_scorer':row['scoring_version']='old'
            rows.append(row)
        if self.mode=='truncated':rows=[]
        return rows,dict(wall_s=.2,native_releases=sum(r['native_release'] for r in rows))


class EngineTests(unittest.TestCase):
    def fixture(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);directory=Path(temp.name)/'native'
        with patch.object(native,'CalibratedAdapter',Adapter):
            engine=native.Engine(dict(scene={}),Path('unused'),directory,prior=dict(records=[source()]),
                calibration=dict(score_config={'verified':'calibration'}),home=dict(home={'verified':'home'}),batch=32)
        return engine

    def context(self):return dict(id='query',split='test',bucket_xy_m=[.6,-.3])

    def test_resolve_preserves_effective_geometry_and_logs_before_actor_without_flight_labels(self):
        e=self.fixture();context=self.context();before=copy.deepcopy(context)
        anchor,metadata,timing=e.resolve(context)
        self.assertEqual(context,before);self.assertEqual(anchor['values'],[.5,-.2,.325,30.,.4]);self.assertEqual(timing['wall_s'],.1)
        self.assertEqual(metadata['source_hint']['values'],source()['values'])
        self.assertNotIn('route_success',metadata['source_hint']);self.assertNotIn('reward',e.adapter.kwargs['prior_records'][0])
        logged=json.loads((e.directory/'context-resolutions.jsonl').read_text())
        self.assertFalse(logged['selection_or_flight_query']);self.assertTrue(logged['geometry_search_at_inference']);self.assertFalse(logged['ball_flight_search_at_inference'])
        self.assertEqual(logged['role'],'test');self.assertEqual(e.adapter.calls,[])
        self.assertEqual(e.adapter.kwargs['calibration'],{'verified':'calibration'});self.assertEqual(e.adapter.kwargs['home'],{'verified':'home'})

    def test_same_anchor_across_checkpoints_is_immutable_and_drift_is_rejected(self):
        e=self.fixture();a,m,_=e.resolve(self.context());a['values'][0]=9.;m['source_hint']['q10'][0]=9.
        again,metadata,_=e.resolve(dict(self.context(),id='another-checkpoint'))
        self.assertEqual(again['values'][0],.5);self.assertEqual(metadata['source_hint']['q10'][0],.1)
        e.adapter.resolution['effective_source_hint']['derivation_offset'][2]=.05
        e.adapter.resolution['effective_source_hint']['values'][2]=.35
        with self.assertRaisesRegex(ValueError,'across methods'):e.resolve(self.context())

    def test_unknown_source_changed_derivation_nonboolean_eligibility_and_flight_selection_fail(self):
        edits=[lambda r:r['source_hint'].update(id='unknown'),lambda r:r['source_hint'].update(source_record_sha256='c'*64),
               lambda r:r['effective_source_hint']['values'].__setitem__(0,.51),lambda r:r.update(layout_eligible=1),
               lambda r:r.update(hard_reject=True),lambda r:r.update(selection_uses_flight_labels=True)]
        for change in edits:
            e=self.fixture();change(e.adapter.resolution)
            with self.assertRaises(ValueError):e.resolve(self.context())
            self.assertEqual(e.adapter.calls,[])

    def test_context_proof_byte_binding_and_relative_path_are_required(self):
        for mode in ('changed','absolute','escaping','missing'):
            e=self.fixture()
            if mode=='changed':(e.directory/'contexts/proof.json').write_text('changed')
            if mode=='absolute':e.adapter.resolution['file']='/private/proof.json'
            if mode=='escaping':e.adapter.resolution['file']='../proof.json'
            if mode=='missing':e.adapter.resolution['file']='missing.json'
            with self.subTest(mode=mode),self.assertRaises(ValueError):e.resolve(self.context())

    def request(self,e):
        anchor,metadata,_=e.resolve(dict(id='control',role='control',bucket_xy_m=[.6,-.3]))
        return dict(id='shot',role='control',bucket_xy_m=[.6,-.3],values=anchor['values'],source_teacher_id=anchor['id'],
            **metadata,observation=[0.]*9,route_success=True,normalized_residual=[0.]*5)

    def test_frozen_control_metadata_delegates_without_resampling_and_preserves_native_outcomes(self):
        e=self.fixture();request=self.request(e);before=copy.deepcopy(request)
        rows,timing=e.evaluate([request],'controls_pre')
        self.assertEqual(request,before);self.assertEqual(e.adapter.calls[0],([request],'controls_pre'))
        self.assertFalse(rows[0]['route_success']);self.assertEqual(rows[0]['sentinel_native'],{'actual':True})
        self.assertEqual(rows[0]['source_hint'],request['source_hint']);self.assertEqual(rows[0]['adapter_directory'],'native')
        self.assertEqual(rows[0]['observation'],[0.]*9);self.assertEqual(timing['wall_s'],.2)
        self.assertEqual(len(e.adapter.resolve_calls),1)

    def test_rejected_queries_remain_in_denominator_and_no_action_repair_occurs(self):
        e=self.fixture();r=self.request(e);e.adapter.mode='rejected';r['values']=[10.]*5
        rows,timing=e.evaluate([r],'test');self.assertEqual(len(rows),1);self.assertFalse(rows[0]['native_release'])
        self.assertEqual(rows[0]['values'],[10.]*5);self.assertEqual(timing['native_releases'],0)

    def test_wrong_action_count_resolution_scorer_and_substitution_fail_closed(self):
        for mode in ('wrong_action','truncated','substituted','wrong_resolution','wrong_scorer'):
            e=self.fixture();r=self.request(e);e.adapter.mode=mode
            with self.subTest(mode=mode),self.assertRaises(RuntimeError):e.evaluate([r],'test')

    def test_storage_delegates_and_context_roles_are_explicit(self):
        e=self.fixture();e.storage_check();self.assertEqual(e.adapter.storage_calls,1)
        self.assertGreater(e.storage()['free_bytes'],0)
        with self.assertRaisesRegex(ValueError,'Contradictory'):e.resolve(dict(self.context(),role='train'))
        with self.assertRaises(ValueError):e.resolve(dict(id='bad',bucket_xy_m=[float('nan'),0]))

    def test_actual_native_release_freezes_options_and_binds_exact_file_bytes(self):
        e=self.fixture();r=self.request(e)
        rows,_=e.evaluate([r],'test')
        path=e.directory/'effective-options.json';self.assertTrue(path.is_file())
        self.assertEqual(json.loads(path.read_text()),e.adapter.env.effective_options)
        self.assertEqual(rows[0]['effective_options_file'],path.name)
        self.assertEqual(rows[0]['effective_options_sha256'],native.sha(path))
        original=path.read_bytes();e.evaluate([dict(r,id='again')],'test')
        self.assertEqual(path.read_bytes(),original)
        path.write_bytes(original+b' ')
        before=len(e.adapter.calls)
        with self.assertRaisesRegex(RuntimeError,'bytes changed'):e.evaluate([dict(r,id='tamper')],'test')
        self.assertEqual(len(e.adapter.calls),before)

    def test_options_changes_missing_file_wrong_version_or_substeps_fail_before_rollout(self):
        for mutation in ('value','version','substeps','missing','nonfinite'):
            e=self.fixture();r=self.request(e);e.evaluate([r],'test');before=len(e.adapter.calls)
            if mutation=='value':e.adapter.env.effective_options['warp']['timestep']=.001
            if mutation=='version':e.adapter.env.effective_options['version']='unverified'
            if mutation=='substeps':e.adapter.env.effective_options['substeps']=128
            if mutation=='missing':(e.directory/'effective-options.json').unlink()
            if mutation=='nonfinite':e.adapter.env.effective_options['native']['timestep']=float('nan')
            with self.subTest(mutation=mutation),self.assertRaises(RuntimeError):e.evaluate([dict(r,id='again')],'test')
            self.assertEqual(len(e.adapter.calls),before)

    def test_geometry_rejections_need_no_allocated_env_but_native_claim_without_options_fails(self):
        e=self.fixture();r=self.request(e);e.adapter.mode='rejected'
        rows,_=e.evaluate([r],'test')
        self.assertFalse(rows[0]['native_release']);self.assertFalse((e.directory/'effective-options.json').exists())
        e.adapter.mode=None
        with patch.object(e.adapter,'evaluate',return_value=([dict(id=r['id'],bucket_xy_m=r['bucket_xy_m'],
                values=r['values'],native_release=True,action_substituted=False,
                contact_protocol=dict(substeps=64),scoring_version=native.SCORER_VERSION)],{})):
            with self.assertRaisesRegex(RuntimeError,'lacks allocated'):e.evaluate([r],'test')


if __name__=='__main__':unittest.main()
