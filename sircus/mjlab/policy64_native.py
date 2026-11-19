"""Geometry-before-policy bridge to the unchanged calibrated 64-step adapter.

The caller owns GPU scheduling, trial budgets and the top-level training ledger.
This bridge neither starts a service nor selects actions from ball-flight data.
"""

import copy
import json
from pathlib import Path
import re
import shutil

import numpy as np

from .calibrated_adapter import CalibratedAdapter
from .calibrated_admission import source_geometry
from .corrected_contract import sha
from .learning_contract import digest


VERSION = "sircus-policy64-calibrated-engine-v1"
SCORER_VERSION = "sircus-calibrated-native-capture-v2"


def _vector(value,size,name):
    out=np.asarray(value,dtype=float)
    if out.shape!=(size,) or not np.isfinite(out).all():
        raise ValueError('Finite '+name+' required')
    return out.tolist()


def _hash(value,name):
    if not isinstance(value,str) or re.fullmatch(r'[a-f0-9]{64}',value) is None:
        raise ValueError('Exact '+name+' SHA256 required')
    return value


class Engine:
    def __init__(self,ref,assets,directory,*,prior,calibration,home,batch=32):
        if type(batch) is not int or not 1<=batch<=32:raise ValueError('Bounded native batch1..32 required')
        if not isinstance(prior,dict) or not isinstance(prior.get('records'),list) or not prior['records']:
            raise ValueError('Nonempty frozen shared prior required')
        # Only geometric source fields reach the resolver or actor metadata.
        self.sources=[source_geometry(r) for r in copy.deepcopy(prior['records'])]
        if len({r['id'] for r in self.sources})!=len(self.sources):raise ValueError('Shared source IDs must be unique')
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        config=calibration.get('score_config',calibration)
        initial=home.get('home',home)
        self.adapter=CalibratedAdapter(ref,assets,self.directory,prior_records=self.sources,
            calibration=config,home=initial,batch=batch)
        self._anchors={}
        self._effective_options_sha256=None

    def _freeze_effective_options(self,*,required=False):
        """Persist allocated engine options; mocks must supply the same evidence."""
        env=getattr(self.adapter,'env',None)
        if env is None:
            if required or self._effective_options_sha256 is not None:
                raise RuntimeError('Native release lacks allocated effective-options evidence')
            return None
        options=getattr(env,'effective_options',None)
        if (not isinstance(options,dict)
            or options.get('version')!='sircus-calibrated-native-warp-options-v1'
            or type(options.get('substeps')) is not int or options['substeps']!=64
            or any(not isinstance(options.get(k),dict) or not options[k] for k in ('native','warp'))):
            raise RuntimeError('Allocated effective options differ from verified pinned64 engine')
        try: raw=(json.dumps(options,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
        except (ValueError,TypeError) as error:
            raise RuntimeError('Allocated effective options are not finite JSON evidence') from error
        path=self.directory/'effective-options.json'
        if path.exists():
            if path.read_bytes()!=raw:raise RuntimeError('Allocated effective-options evidence bytes changed')
        elif self._effective_options_sha256 is not None:
            raise RuntimeError('Frozen effective-options evidence disappeared')
        else:
            with path.open('xb') as stream:stream.write(raw)
        fingerprint=sha(path)
        if self._effective_options_sha256 is not None and fingerprint!=self._effective_options_sha256:
            raise RuntimeError('Allocated effective-options fingerprint changed')
        self._effective_options_sha256=fingerprint
        return dict(effective_options_file=path.name,effective_options_sha256=fingerprint)

    def storage_check(self):
        return self.adapter.storage_check()

    def storage(self):
        self.storage_check()
        return dict(free_bytes=shutil.disk_usage(self.directory).free,adapter_directory='native')

    def resolve(self,context):
        if not isinstance(context,dict) or not isinstance(context.get('id'),str) or not context['id']:
            raise ValueError('Stable nonempty context ID required')
        query=copy.deepcopy(context);query['bucket_xy_m']=_vector(query.get('bucket_xy_m'),2,'bucket XY')
        if 'role' in query and 'split' in query and query['role']!=query['split']:
            raise ValueError('Contradictory context roles')
        role=query.get('split',query.get('role','unspecified'))
        if role not in ('train','validation','test','control','unspecified'):
            raise ValueError('Unexpected context role')
        resolution,timing=self.adapter.resolve_context(query)
        if not isinstance(resolution,dict) or not isinstance(timing,dict):
            raise ValueError('Context resolver must return evidence and timing')
        hint=resolution.get('source_hint',{})
        matches=[r for r in self.sources if r['id']==hint.get('id')]
        if len(matches)!=1:raise ValueError('Geometry resolver selected an unknown shared source')
        anchor=copy.deepcopy(matches[0])
        for key in ('values','bucket_xy_m','source_record_sha256'):
            if hint.get(key)!=anchor.get(key):raise ValueError('Geometry resolver changed shared source provenance')
        filename=resolution.get('file')
        if not isinstance(filename,str) or Path(filename).is_absolute() or '..' in Path(filename).parts:
            raise ValueError('Relative context proof file required')
        proof=(self.directory/filename).resolve()
        if not proof.is_relative_to(self.directory.resolve()) or not proof.is_file():
            raise ValueError('Context proof is missing or escapes adapter directory')
        if sha(proof)!=_hash(resolution.get('sha256'),'context proof'):
            raise ValueError('Context proof bytes changed before prediction')
        _hash(resolution.get('home_sha256'),'context home')
        flags=('layout_eligible','hard_reject','anchor_admitted')
        if any(type(resolution.get(k)) is not bool for k in flags):raise ValueError('Context eligibility must be explicit')
        if resolution['hard_reject'] and resolution['layout_eligible']:
            raise ValueError('Hard rejected context cannot be layout eligible')
        if resolution.get('selection_uses_flight_labels') is not False:
            raise ValueError('Pre-policy setup must exclude flight labels')
        effective=copy.deepcopy(resolution.get('effective_source_hint',hint))
        if effective.get('id')!=anchor['id'] or effective.get('bucket_xy_m')!=anchor['bucket_xy_m']:
            raise ValueError('Effective anchor lost original source identity')
        offset=np.asarray(effective.get('derivation_offset',[0.]*5),dtype=float)
        values=np.asarray(effective.get('values'),dtype=float)
        if offset.shape!=(5,) or values.shape!=(5,) or not np.isfinite(offset).all() or not np.isfinite(values).all():
            raise ValueError('Finite exact effective-anchor derivation required')
        if not np.allclose(values,np.asarray(anchor['values'])+offset,atol=1e-12,rtol=0):
            raise ValueError('Effective anchor differs from frozen geometry derivation')
        metadata=dict(source_hint=anchor,effective_source_hint=effective,
            context_resolution_file=filename,context_resolution_sha256=resolution['sha256'],context_home_sha256=resolution['home_sha256'],
            context_layout_eligible=resolution['layout_eligible'],context_anchor_admitted=resolution['anchor_admitted'],context_hard_reject=resolution['hard_reject'])
        key=digest(query['bucket_xy_m']);identity=digest(dict(anchor=effective,metadata=metadata))
        if key in self._anchors and self._anchors[key]!=identity:
            raise ValueError('Pre-policy anchor changed across methods or checkpoints')
        self._anchors[key]=identity
        row=dict(context_id=query['id'],role=role,bucket_xy_m=query['bucket_xy_m'],source_teacher_id=anchor['id'],
            **metadata,timing=copy.deepcopy(timing),selection_or_flight_query=False,
            geometry_search_at_inference=True,ball_flight_search_at_inference=False,engine_version=VERSION)
        with (self.directory/'context-resolutions.jsonl').open('a') as stream:
            stream.write(json.dumps(row,allow_nan=False)+'\n')
        return copy.deepcopy(effective),copy.deepcopy(metadata),copy.deepcopy(timing)

    def evaluate(self,requests,phase):
        if not isinstance(phase,str) or not phase:raise ValueError('Nonempty execution phase required')
        normalized=[]
        for request in requests:
            r=copy.deepcopy(request)
            r['bucket_xy_m']=_vector(r.get('bucket_xy_m'),2,'bucket XY')
            r['values']=_vector(r.get('values'),5,'action values')
            normalized.append(r)
        # Fail before another rollout if an already allocated engine/evidence
        # changed; after the first rollout require real allocated options.
        self._freeze_effective_options()
        rows,timing=self.adapter.evaluate(normalized,phase)
        if len(rows)!=len(normalized):raise RuntimeError('Adapter changed requested trial count')
        effective=self._freeze_effective_options(required=any(r.get('native_release') for r in rows))
        result=[]
        for measured,request in zip(rows,normalized):
            if measured.get('id')!=request.get('id') or measured.get('bucket_xy_m')!=request['bucket_xy_m'] or measured.get('values')!=request['values']:
                raise RuntimeError('Adapter replaced or reordered an exact requested action')
            if measured.get('action_substituted') is not False:raise RuntimeError('Action substitution is forbidden')
            if measured.get('native_release'):
                if measured.get('contact_protocol',{}).get('substeps')!=64 or measured.get('scoring_version')!=SCORER_VERSION:
                    raise RuntimeError('Native release differs from pinned64/calibrated scoring')
            # Native observations always outrank any historical/outcome-like
            # caller metadata. Exact action identity was checked above.
            row=dict(request);row.update(copy.deepcopy(measured));row['adapter_directory']='native'
            if measured.get('native_release'):row.update(effective)
            result.append(row)
        return result,timing
