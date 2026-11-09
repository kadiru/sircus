"""Exact drop-gap regression replays and one frozen coordinate intervention.

The caller owns the finite GPU lease. The two historical arms retain their
original admission decisions and joints. Changed actions use a new strict
connection proof under the same 19-source geometry bank.
"""
from __future__ import annotations
import copy
import math
import numpy as np
from .benchmark_verify import digest
from .table_correction_native import MatchedEngine as HistoricalMatchedEngine, META, PARAMETERS
from .table_training_contract import captured

VERSION = 'sircus-regression-drop-gap-correction-v1'
CASES = tuple('recovery-test-' + x for x in ('r0-c5-n1','r0-c6-n1'))
VARIANTS = ('parent_exact','recovery_exact','corrected')
ROLES = ('direct_control','table_control','negative_control')
MAX_REQUESTS = 36
# Two released recovery misses gained about31/38mm drop gap versus captures.
# Restoring the matched parent gap tests this hypothesis; outcomes are unknown.
FROZEN_COORDINATE = 4
DIAGNOSIS_SHA256 = '511e9f82004af0d603628e25adf8fab71284faa8f11de9307b4c8a611d049ab7'


def validate_rule(rule):
    if FROZEN_COORDINATE is None:
        raise ValueError('Correction has not been frozen after diagnosis')
    index = rule.get('parameter_index')
    if type(index) is not int or index != FROZEN_COORDINATE or rule.get('parameter') != PARAMETERS[index]:
        raise ValueError('Only the one evidence-selected coordinate may change')
    if rule.get('mode') != 'restore_matched_parent_coordinate' or rule.get('contexts') != list(CASES):
        raise ValueError('Literal two-case matched-parent intervention required')
    if not isinstance(rule.get('description'),str) or not rule['description'].strip():
        raise ValueError('Evidence-based description required')
    value=rule.get('diagnosis_sha256')
    if value != DIAGNOSIS_SHA256:
        raise ValueError('Exact saved CPU diagnosis hash required')
    return copy.deepcopy(rule)


def source_pairs(rows):
    pairs=[]
    for context in CASES:
        pair={}
        for method in ('parent_bc','recovery_bc'):
            matches=[r for r in rows if r.get('phase')=='test' and r.get('method')==method and r.get('context_id')==context]
            if len(matches)!=1: raise ValueError('One exact historical method/context row required')
            pair[method]=matches[0]
        parent,recovery=(pair[m] for m in ('parent_bc','recovery_bc'))
        if (not captured(parent) or captured(recovery) or recovery.get('invalid_physics')
                or parent.get('native_release') is not True or recovery.get('native_release') is not True
                or parent['bucket_xy_m']!=recovery['bucket_xy_m']):
            raise ValueError('Literal admitted parent capture and valid released recovery failure required')
        for key in META:
            if parent[key]!=recovery[key]:raise ValueError('Historical pair lost common geometry: '+key)
        pairs.append(pair)
    return pairs


def make_plan(rows, controls, rule):
    rule=validate_rule(rule);pairs=source_pairs(rows);plan=[]
    if len(controls)!=3 or [r['role'] for r in controls]!=list(ROLES):
        raise ValueError('Exactly the three original17 matched controls required')
    if not all(r.get('native_release') is True for r in controls):
        raise ValueError('Original controls must be admitted')
    def append(source,variant,phase,repeat,donor=None):
        values=list(map(float,source['values']))
        if len(values)!=5 or not all(map(math.isfinite,values)):raise ValueError('Finite exact action required')
        if variant=='corrected':
            values[rule['parameter_index']]=float(donor['values'][rule['parameter_index']])
            if values==source['values']:raise ValueError('Frozen intervention must change its named coordinate')
        request={k:copy.deepcopy(source[k]) for k in META}
        request.update(id=f'drop-gap-{phase}-{source["context_id"]}-{variant}-r{repeat}',phase=phase,
            role=source['role'] if phase.startswith('controls_') else 'diagnostic',
            method='control' if phase.startswith('controls_') else variant,variant=variant,repeat=repeat,
            context_id=source['context_id'],bucket_xy_m=copy.deepcopy(source['bucket_xy_m']),
            values=values,command_values=copy.deepcopy(values),source_record_id=source['id'],source_record_sha256=digest(source),
            donor_record_id=donor['id'] if donor else None,donor_record_sha256=digest(donor) if donor else None,
            adapter_directory='native-controls' if phase.startswith('controls_') else 'native',
            substeps=64,perturbation_mixed_units=[0.]*5,selection_or_learning=False,
            original_role=source['role'],posthoc_development=True,
            admission_mode=('new_action_strict_proof' if variant=='corrected' else
                'exact_parent_branch' if source['native_release'] else 'cached_parent_refusal'))
        plan.append(request)
    for source in controls:
        for repeat in range(3):append(source,'control_exact','controls_pre',repeat)
    for pair in pairs:
        for variant in VARIANTS:
            source=pair['parent_bc' if variant=='parent_exact' else 'recovery_bc']
            for repeat in range(3):append(source,variant,'diagnostic',repeat,pair['parent_bc'] if variant=='corrected' else None)
    for source in controls:
        for repeat in range(3):append(source,'control_exact','controls_post',repeat)
    if len(plan)!=36 or len({r['id'] for r in plan})!=36:raise ValueError('Exactly36 unique slots required')
    return plan


def realization_comparison(row, source):
    """Compare actual admitted branches and settled board pose to saved source.

    A gap-only requested coordinate does not guarantee the same IK branch.
    Retain any such change explicitly rather than forcing old joint vectors.
    """
    if row.get('native_release') is not True:return None
    result=dict(source_record_id=source['id'])
    for key in ('q10','q14'):
        actual=np.asarray(row[key],float);before=np.asarray(source[key],float)
        result[key+'_equal_to_source']=bool(np.array_equal(actual,before))
        result[key+'_max_abs_delta_rad']=float(np.max(np.abs(actual-before)))
    center=np.asarray(row['board_center_m'],float);old_center=np.asarray(source['board_center_m'],float)
    normal=np.asarray(row['board_normal'],float);old_normal=np.asarray(source['board_normal'],float)
    if (center.shape!=(3,) or old_center.shape!=(3,) or normal.shape!=(3,) or old_normal.shape!=(3,)
            or not all(np.all(np.isfinite(a)) for a in (center,old_center,normal,old_normal))
            or np.linalg.norm(normal)<=0 or np.linalg.norm(old_normal)<=0):
        raise RuntimeError('Actual and authenticated source board poses required')
    result['board_center_delta_m']=(center-old_center).tolist()
    cosine=np.dot(normal,old_normal)/(np.linalg.norm(normal)*np.linalg.norm(old_normal))
    result['board_normal_angle_deg']=float(np.degrees(np.arccos(np.clip(cosine,-1.,1.))))
    return result


class MatchedEngine(HistoricalMatchedEngine):
    def __init__(self,*args,rule,**kwargs):
        self.rule=validate_rule(rule)
        super().__init__(*args,**kwargs)

    def evaluate(self,requests,phase):
        for request in requests:
            source=self.sources[request['source_record_id']]
            if digest(source)!=request['source_record_sha256']:raise ValueError('Immutable parent row changed')
            expected=copy.deepcopy(source['values'])
            if request['variant']=='corrected':
                donor=self.sources[request['donor_record_id']]
                if (digest(donor)!=request['donor_record_sha256'] or donor['context_id']!=source['context_id']
                        or donor['method']!='parent_bc' or not captured(donor)):
                    raise ValueError('Matched captured parent donor changed')
                expected[self.rule['parameter_index']]=donor['values'][self.rule['parameter_index']]
            elif request['variant'] not in ('parent_exact','recovery_exact','control_exact'):
                raise ValueError('Unknown frozen intervention variant')
            if request['values']!=expected or request['command_values']!=expected or request['bucket_xy_m']!=source['bucket_xy_m']:
                raise ValueError('Only one frozen coordinate may differ')
        rows,timing=self.engine.evaluate(requests,phase)
        if [r['id'] for r in rows]!=[r['id'] for r in requests]:raise RuntimeError('Request order/accounting changed')
        env=self.engine.adapter.env
        if env is not None and (env.effective_options!=self.parent_options or env.compiled_geometry!=self.parent_geometry):
            raise RuntimeError('Original native options or geometry changed')
        for row,request in zip(rows,requests):
            source=self.sources[request['source_record_id']]
            if row['variant']!='corrected':
                for key in ('connection_file','connection_sha256','native_release','q10','q14','invalid'):
                    if key=='invalid' and source.get('native_release'):continue
                    if row.get(key)!=source.get(key):raise RuntimeError('Original admission or branch changed: '+key)
            elif not row.get('connection_sha256') or row['connection_sha256']==source.get('connection_sha256'):
                raise RuntimeError('Changed action reused original connection proof')
            if row.get('native_release'):
                for key in ('q10','q14'):
                    if row.get('native_initial_'+key)!=np.asarray(row[key],np.float32).astype(float).tolist():
                        raise RuntimeError('Actual native reset differs from admitted branch')
            row['realization_comparison']=realization_comparison(row,source)
        return rows,timing
