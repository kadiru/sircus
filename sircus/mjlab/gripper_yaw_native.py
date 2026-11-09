"""Two colliding saved recipes: fixed original versus bucket-facing wrist yaw."""
from __future__ import annotations
import copy
import numpy as np
from .benchmark_verify import digest,contained,read,sha
from .table_correction_native import MatchedEngine as HistoricalMatchedEngine,META,PARAMETERS
from .table_training_contract import captured
from .recovery_regression_diagnosis import setup_evidence
from .regression_drop_gap_native import realization_comparison
from .gripper_yaw_geometry import YawResolver,yaw_target,yaw_evidence

VERSION='sircus-gripper-yaw-v1'
CASES=tuple('corrected-imitation-test-'+s for s in ('r0-c6-n1','r1-c2-n0'))
VARIANTS=('current_exact','corrected_exact','bucket_facing')
ROLES=('direct_control','table_control','negative_control')
MAX_REQUESTS=36
DIAGNOSIS_SHA256='e16f551802c6c378e7911a0bca2470e4302babb075f8ef570df189839a7bd428'
MODE='bucket_facing_gripper_yaw'


def validate_rule(rule):
    if (set(rule)!={'mode','contexts','diagnosis_sha256','description','rotation','preserve_board_joints'}
        or rule['mode']!=MODE or rule['contexts']!=list(CASES) or rule['diagnosis_sha256']!=DIAGNOSIS_SHA256
        or rule['rotation']!='Rz(atan2(bucket_y-board_y,bucket_x-board_x)) @ diag(1,-1,-1)'
        or rule['preserve_board_joints'] is not True or not isinstance(rule['description'],str) or not rule['description'].strip()):
        raise ValueError('Literal two-case bucket-facing yaw intervention required')
    return copy.deepcopy(rule)


def source_pairs(rows):
    pairs=[]
    for context in CASES:
        pair={}
        for method in ('current_bc','corrected_bc'):
            matches=[r for r in rows if r.get('phase')=='test' and r.get('method')==method and r.get('context_id')==context]
            if len(matches)!=1:raise ValueError('One exact historical method/context row required')
            pair[method]=matches[0]
        a,b=pair['current_bc'],pair['corrected_bc']
        if not captured(a) or captured(b) or any(r.get('native_release') is not True or r.get('invalid_physics') for r in (a,b)):
            raise ValueError('Admitted original capture and colliding corrected miss required')
        if a['bucket_xy_m']!=b['bucket_xy_m'] or not b.get('forbidden_contact_steps',0):raise ValueError('Exact colliding pair required')
        if any(a[k]!=b[k] for k in META):raise ValueError('Historical pair source identity differs')
        pairs.append(pair)
    return pairs


def make_plan(rows,controls,rule):
    validate_rule(rule);pairs=source_pairs(rows);plan=[]
    if len(controls)!=3 or [r['role'] for r in controls]!=list(ROLES) or not all(r.get('native_release') is True for r in controls):raise ValueError('Three admitted original controls required')
    def append(source,variant,phase,repeat):
        request={k:copy.deepcopy(source[k]) for k in META}
        v=list(map(float,source['values']));yaw=float(np.arctan2(source['bucket_xy_m'][1]-v[1],source['bucket_xy_m'][0]-v[0])) if variant=='bucket_facing' else 0.
        namespace='native-controls' if phase.startswith('controls_') else ('native-yaw' if variant=='bucket_facing' else 'native')
        request.update(id=f'gripper-yaw-{phase}-{source["context_id"]}-{variant}-r{repeat}',phase=phase,
            role=source['role'] if phase.startswith('controls_') else 'diagnostic',method='control' if phase.startswith('controls_') else variant,
            variant=variant,repeat=repeat,context_id=source['context_id'],bucket_xy_m=copy.deepcopy(source['bucket_xy_m']),values=v,command_values=copy.deepcopy(v),
            source_record_id=source['id'],source_record_sha256=digest(source),donor_record_id=None,donor_record_sha256=None,
            realization_reference_record_id=source['id'],realization_reference_record_sha256=digest(source),
            parameter_indices=[],interpolation_alpha=None,requested_gripper_yaw_rad=yaw,preserved_q14=copy.deepcopy(source['q14']),
            adapter_directory=namespace,substeps=64,perturbation_mixed_units=[0.]*5,selection_or_learning=False,
            original_role=source['role'],posthoc_development=True,
            admission_mode='fixed_board_yaw_strict_proof' if variant=='bucket_facing' else 'exact_parent_branch')
        plan.append(request)
    for s in controls:
        for r in range(3):append(s,'control_exact','controls_pre',r)
    for pair in pairs:
        for variant in VARIANTS:
            for r in range(3):append(pair['current_bc' if variant=='current_exact' else 'corrected_bc'],variant,'diagnostic',r)
    for s in controls:
        for r in range(3):append(s,'control_exact','controls_post',r)
    if len(plan)!=36 or len({r['id'] for r in plan})!=36:raise ValueError('Exactly36unique slots required')
    return plan


class MatchedEngine(HistoricalMatchedEngine):
    def __init__(self,*args,rule,yaw=False,**kwargs):
        self.rule=validate_rule(rule);self.yaw=bool(yaw)
        super().__init__(*args,**kwargs)
        if yaw:
            selected=[s for s in self.sources.values() if s.get('method')=='corrected_bc' and s['context_id'] in CASES]
            self.engine.adapter.resolver=YawResolver(self.engine.adapter.resolver,selected)

    def evaluate(self,requests,phase):
        for request in requests:
            source=self.sources[request['source_record_id']];changed=request['variant']=='bucket_facing'
            if (request['phase']!=phase or digest(source)!=request['source_record_sha256'] or request['values']!=source['values']
                or request['command_values']!=source['values'] or request['bucket_xy_m']!=source['bucket_xy_m']
                or request['preserved_q14']!=source['q14'] or changed!=self.yaw
                or request['substeps']!=64 or request['parameter_indices']!=[] or request['perturbation_mixed_units']!=[0.]*5
                or request['selection_or_learning'] is not False or request['interpolation_alpha'] is not None
                or request['donor_record_id'] is not None or request['donor_record_sha256'] is not None
                or request['realization_reference_record_id']!=source['id'] or request['realization_reference_record_sha256']!=digest(source)):
                raise ValueError('Only explicit wrist yaw may change')
            expected_yaw=float(np.arctan2(source['bucket_xy_m'][1]-source['values'][1],source['bucket_xy_m'][0]-source['values'][0])) if changed else 0.
            if request['requested_gripper_yaw_rad']!=expected_yaw:raise ValueError('Requested yaw differs from frozen bucket direction')
            if changed and (source['method']!='corrected_bc' or source['context_id'] not in CASES):raise ValueError('Only declared corrected collisions may rotate')
        rows,timing=self.engine.evaluate(requests,phase)
        if [r['id'] for r in rows]!=[r['id'] for r in requests]:raise RuntimeError('Request order changed')
        env=self.engine.adapter.env
        if env is not None and (env.effective_options!=self.parent_options or env.compiled_geometry!=self.parent_geometry):raise RuntimeError('Physics/geometry changed')
        for row,request in zip(rows,requests):
            source=self.sources[row['source_record_id']]
            if not self.yaw:
                for key in ('connection_file','connection_sha256','native_release','q10','q14'):
                    if row.get(key)!=source.get(key):raise RuntimeError('Original proof/branch changed: '+key)
            else:
                if row['connection_sha256']==source['connection_sha256'] or row['q14']!=source['q14']:raise RuntimeError('Yaw proof reused original or changed board')
            if row.get('native_release'):
                for key in ('q10','q14'):
                    if row['native_initial_'+key]!=np.asarray(row[key],np.float32).astype(float).tolist():raise RuntimeError('Native reset differs')
            row['yaw_evidence']=yaw_evidence(self.engine.adapter.ref['scene'],self.engine.adapter.resolver.solver.gate,row,source)
            row['realization_comparison']=realization_comparison(row,source)
            proofs=[]
            for name,hkey in (('connection_file','connection_sha256'),('context_resolution_file','context_resolution_sha256')):
                path=contained(self.directory,row[name])
                if sha(path)!=row[hkey]:raise RuntimeError('Setup evidence changed')
                proofs.append(read(path))
            row['setup_proof']=setup_evidence(row,*proofs)
        return rows,timing
