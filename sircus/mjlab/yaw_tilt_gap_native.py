"""Finite two-parameter native64 wrapper; immutable old recipes and fresh yaw proofs."""
from __future__ import annotations
import copy
from pathlib import Path
import numpy as np
from .benchmark_verify import digest,contained,read,sha
from .table_correction_native import MatchedEngine as HistoricalMatchedEngine,copy_bound
from .calibrated_admission import verify_connection
from .recovery_regression_diagnosis import setup_evidence
from .yaw_tilt_gap_geometry import TiltGapResolver,rotation_evidence
from . import yaw_tilt_gap_contract as C


class SweepEngine:
    def __init__(self,ref,assets,directory,*,parent,sources,prior,calibration,home,plan,yaw_study=None,yaw_rows=None,namespace='native'):
        self.parent=HistoricalMatchedEngine(ref,assets,directory,parent=parent,sources=sources,
            prior=prior,calibration=calibration,home=home)
        self.engine=self.parent.engine;self.sources=self.parent.sources;self.directory=Path(directory)
        self.namespace=namespace;self.plan={r['id']:copy.deepcopy(r) for r in plan};self.yaw_by_context={r['context_id']:r for r in (yaw_rows or [])}
        if namespace=='native-yaw':
            base=self.engine.adapter.resolver;exact={}
            for old in yaw_rows:
                anchor=base.resolve(old['bucket_xy_m'])
                if anchor['sha256']!=old['context_resolution_sha256'] or not anchor['cache_hit']:raise ValueError('Original yaw resolution required')
                proof=copy_bound(Path(yaw_study)/'native-yaw',self.directory,old['connection_file'],old['connection_sha256'])
                verify_connection(proof,anchor,base.home,old['values'])
                if not proof['passed'] or proof['q10']!=old['q10'] or proof['q14']!=old['q14']:raise ValueError('Exact yaw reference branch required')
                exact[digest(dict(bucket_xy_m=old['bucket_xy_m'],values=old['values']))]=proof
            self.engine.adapter.resolver=TiltGapResolver(base,sources,yaw_rows,plan,exact)

    def evaluate(self,requests,phase):
        for request in requests:
            original=self.plan.get(request.get('selected_from_id',request['id']))
            if original is None:raise ValueError('Unknown frozen request')
            expected=copy.deepcopy(original)
            if request['variant']=='selected':
                expected.update(id=request['id'],phase='selected_repeats',variant='selected',method='selected',repeat=request['repeat'],
                    selected_from_id=request['selected_from_id'],selected_from_sha256=request['selected_from_sha256'])
            if request!=expected or request['phase']!=phase or request['adapter_directory']!=self.namespace:
                raise ValueError('Native request differs from frozen plan/namespace')
            source=self.sources[request['source_record_id']]
            if digest(source)!=request['source_record_sha256'] or request['bucket_xy_m']!=source['bucket_xy_m']:
                raise ValueError('Frozen source or bucket changed')
            if request['values'][:3]!=source['values'][:3] or request['substeps']!=64:raise ValueError('Only tilt and gap may vary')
        rows,timing=self.engine.evaluate(requests,phase)
        if [r['id'] for r in rows]!=[r['id'] for r in requests]:raise RuntimeError('Native request order changed')
        env=self.engine.adapter.env
        if env is not None and (env.effective_options!=self.parent.parent_options or env.compiled_geometry!=self.parent.parent_geometry):
            raise RuntimeError('Native physics/geometry changed')
        for row,request in zip(rows,requests):
            row['adapter_directory']=self.namespace;source=self.sources[request['source_record_id']]
            if request['variant'] in ('control_exact','far_success_reference'):
                bound=source
            elif request['values']==source['values']:
                bound=self.yaw_by_context[request['context_id']]
            else:bound=None
            if bound:
                for key in ('connection_file','connection_sha256','native_release','q10','q14'):
                    if row.get(key)!=bound.get(key):raise RuntimeError('Original reference branch changed: '+key)
            elif not row['connection_file'].startswith('yaw-tilt-gap-connections/'):
                raise RuntimeError('New tilt/gap must carry a distinct full-orientation proof')
            if row.get('native_release'):
                for key in ('q10','q14'):
                    if row.get('native_initial_'+key)!=np.asarray(row[key],np.float32).astype(float).tolist():raise RuntimeError('Observed reset differs from admitted joints')
            row['yaw_evidence']=rotation_evidence(self.engine.adapter.ref['scene'],self.engine.adapter.resolver.solver.gate,row)
            proofs=[]
            for name,hkey in (('connection_file','connection_sha256'),('context_resolution_file','context_resolution_sha256')):
                path=contained(self.directory,row[name])
                if sha(path)!=row[hkey]:raise RuntimeError('Setup evidence changed')
                proofs.append(read(path))
            row['setup_proof']=setup_evidence(row,*proofs)
        return rows,timing
