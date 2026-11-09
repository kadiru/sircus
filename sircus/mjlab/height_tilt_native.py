"""Grid admission and unchanged v3 native64 with immutable exact-branch replay."""
from __future__ import annotations
import copy
from pathlib import Path
import time
import numpy as np
from .table_correction_native import MatchedEngine
from .calibrated_admission import VERSION as GEOMETRY_VERSION, CONNECTION_POLICY_VERSION, verify_connection
from .benchmark_verify import contained,read,sha,digest


class HeightTiltEngine:
    def __init__(self,ref,assets,directory,*,parent,sources,prior,calibration,home,plan):
        # Only the audited constructor/cache hydrator is reused. The old
        # tilt-only request validator remains unchanged in its own module.
        self.parent=MatchedEngine(ref,assets,directory,parent=parent,sources=sources,prior=prior,calibration=calibration,home=home)
        self.engine=self.parent.engine;self.directory=Path(directory)
        self.sources=self.parent.sources;self.plan={r['id']:copy.deepcopy(r) for r in plan};self.matrix={}
        if len(self.plan)!=120:raise ValueError('Exact120 unique grid requests required')
        if self.engine.adapter.env is not None:raise RuntimeError('CPU admission must not initialize native GPU')

    def _original(self,request):
        source=self.sources[request['source_record_id']]
        if digest(source)!=request['source_record_sha256']:raise ValueError('Original source binding changed')
        if request['bucket_xy_m']!=source['bucket_xy_m']:raise ValueError('Bucket position changed')
        values=request['values']
        if len(values)!=5 or [values[i] for i in (0,1,4)]!=[source['values'][i] for i in (0,1,4)]:
            raise ValueError('Only board height and tilt may change')
        return source

    def admit(self,request):
        self._original(request)
        if request!=self.plan.get(request['id']):raise ValueError('Admission differs from frozen grid')
        if self.engine.adapter.env is not None:raise RuntimeError('Admission phase must stay CPU-only')
        start=time.monotonic();resolver=self.engine.adapter.resolver
        anchor=resolver.resolve(request['bucket_xy_m'])
        if anchor['sha256']!=request['context_resolution_sha256'] or not anchor['cache_hit']:
            raise ValueError('Exact original resolution must be reused')
        proof=resolver.connect(anchor,request['values']);verify_connection(proof,anchor,resolver.home,request['values'])
        endpoint=proof.get('endpoint_attempts') or {}
        return dict(candidate_id=request['id'],context_id=request['context_id'],grid_index=request['grid_index'],
            case_grid_index=request['case_grid_index'],values=request['values'],bucket_xy_m=request['bucket_xy_m'],
            status='completed',passed=bool(proof['passed']),reason=proof['reason'],
            valid_ik_branches=endpoint.get('valid_ik_branches'),connection_file=proof['file'],connection_sha256=proof['sha256'],
            resolution_file=anchor['file'],resolution_sha256=anchor['sha256'],
            q10=proof['q10'] if proof['passed'] else None,q14=proof['q14'] if proof['passed'] else None,
            elapsed_s=time.monotonic()-start)

    def hydrate_matrix(self,matrix):
        """Load exact disk proofs after CPU process exits; never solve again."""
        resolver=self.engine.adapter.resolver
        if len(matrix)!=120:raise ValueError('Complete matrix required')
        self.matrix={}
        for record in matrix:
            request=self.plan.get(record['candidate_id'])
            if request is None or record['status']!='completed':raise ValueError('Only completed frozen grid admitted to native phase')
            if record['values']!=request['values'] or record['bucket_xy_m']!=request['bucket_xy_m']:
                raise ValueError('Matrix changed requested action')
            anchor=resolver.resolve(request['bucket_xy_m'])
            if not anchor['cache_hit'] or anchor['sha256']!=record['resolution_sha256']:
                raise ValueError('Matrix resolution does not match original')
            path=contained(self.directory,record['connection_file'])
            if sha(path)!=record['connection_sha256']:raise ValueError('Matrix connection proof bytes changed')
            proof=dict(read(path),file=record['connection_file'],sha256=record['connection_sha256'],cache_hit=True)
            verify_connection(proof,anchor,resolver.home,request['values'])
            if bool(proof['passed'])!=record['passed']:raise ValueError('Matrix changed admission outcome')
            key=digest(dict(version=GEOMETRY_VERSION,connection_policy=CONNECTION_POLICY_VERSION,
                resolution_sha256=anchor['sha256'],values=request['values'],bucket_xy_m=request['bucket_xy_m']))
            if path.stem!=key:raise ValueError('Connection cache key changed')
            resolver._connected[key]=copy.deepcopy(proof);self.matrix[request['id']]=copy.deepcopy(record)

    def evaluate(self,requests,phase):
        expected=[]
        for request in requests:
            source=self._original(request);variant=request['variant']
            if variant in ('baseline','control_exact'):
                if request['values']!=source['values']:raise ValueError('Original/control action must remain exact')
                expected.append(source)
            elif variant in ('grid','selected'):
                identity=request['id'] if variant=='grid' else request['selected_from_id']
                frozen=self.plan.get(identity);record=self.matrix.get(identity)
                if frozen is None or record is None or not record['passed']:
                    raise ValueError('Native candidate must have a completed admitted matrix proof')
                if request['values']!=frozen['values'] or request['grid_index']!=frozen['grid_index']:
                    raise ValueError('Native candidate differs from frozen matrix')
                expected.append(record)
            else:raise ValueError('Unknown grid-study variant')
        rows,timing=self.engine.evaluate(requests,phase)
        if len(rows)!=len(requests):raise RuntimeError('Native request accounting changed')
        env=self.engine.adapter.env
        if env is not None and (env.effective_options!=self.parent.parent_options or env.compiled_geometry!=self.parent.parent_geometry):
            raise RuntimeError('Native physics/geometry changed from audited parent')
        for row,bound in zip(rows,expected):
            if row.get('native_release') is not True:raise RuntimeError('Frozen admitted action changed into a refusal')
            for key in ('connection_file','connection_sha256','q10','q14'):
                if row.get(key)!=bound.get(key):raise RuntimeError('Executed action changed its frozen admission branch: '+key)
            for key in ('q10','q14'):
                expected_q=np.asarray(row[key],dtype=np.float32).astype(float).tolist()
                if row['native_initial_'+key]!=expected_q:raise RuntimeError('Observed native reset joint branch differs')
        return rows,timing
