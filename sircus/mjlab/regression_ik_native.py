"""Frozen matched coordinate paths for two previously rejected IK requests.

No solver tolerances, geometry or scoring are changed. Exact historical recipes
retain their admitted branches/refusals; novel actions obtain strict proofs.
"""
from __future__ import annotations
import copy
import math
import numpy as np
from .benchmark_verify import digest, contained, read, sha
from .table_correction_native import MatchedEngine as HistoricalMatchedEngine, META, PARAMETERS
from .table_training_contract import captured
from .regression_drop_gap_native import realization_comparison as source_realization
from .recovery_regression_diagnosis import setup_evidence

VERSION = 'sircus-regression-ik-feasibility-v1'
CASES = tuple('recovery-test-' + x for x in ('r1-c1-n1','r2-c4-n0'))
VARIANTS = ('parent_exact','recovery_exact','parent_xy','parent_height_gap','parent_tilt','blend25','blend50','blend75')
ROLES = ('direct_control','table_control','negative_control')
MAX_REQUESTS = 66
DIAGNOSIS_SHA256 = '511e9f82004af0d603628e25adf8fab71284faa8f11de9307b4c8a611d049ab7'
COORDINATE_GROUPS = {'parent_xy':[0,1], 'parent_height_gap':[2,4], 'parent_tilt':[3]}
BLEND_FRACTIONS = {'blend25':.25, 'blend50':.5, 'blend75':.75}
MODE = 'matched_parent_coordinate_blocks_and_interpolation'
NEW_VARIANTS = VARIANTS[2:]


def validate_rule(rule):
    if (rule.get('mode') != MODE or rule.get('contexts') != list(CASES)
            or rule.get('coordinate_groups') != COORDINATE_GROUPS
            or rule.get('blend_fractions') != BLEND_FRACTIONS
            or rule.get('diagnosis_sha256') != DIAGNOSIS_SHA256):
        raise ValueError('Literal two-case coordinate blocks and interpolation required')
    if (set(rule) != {'mode','contexts','coordinate_groups','blend_fractions','diagnosis_sha256','description'}
            or any(type(i) is not int for group in rule['coordinate_groups'].values() for i in group)
            or any(type(a) not in (float,int) for a in rule['blend_fractions'].values())):
        raise ValueError('Literal typed rule schema required')
    if not isinstance(rule.get('description'),str) or not rule['description'].strip():
        raise ValueError('Evidence-based description required')
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
        if (not captured(parent) or captured(recovery) or recovery.get('route_success') is True or recovery.get('invalid_physics')
                or parent.get('native_release') is not True or recovery.get('native_release') is not False
                or recovery.get('invalid') != 'finite_ik_unresolved'
                or parent['bucket_xy_m']!=recovery['bucket_xy_m']):
            raise ValueError('Literal admitted parent capture and finite-IK recovery refusal required')
        for key in META:
            if parent[key]!=recovery[key]:raise ValueError('Historical pair lost common geometry: '+key)
        pairs.append(pair)
    return pairs


def variant_values(source, donor, variant):
    values=list(map(float,source['values']))
    if len(values)!=5 or not all(map(math.isfinite,values)):raise ValueError('Finite exact source action required')
    if variant in NEW_VARIANTS:
        target=list(map(float,donor['values']))
        if len(target)!=5 or not all(map(math.isfinite,target)):raise ValueError('Finite exact donor action required')
        if variant in COORDINATE_GROUPS:
            for index in COORDINATE_GROUPS[variant]:values[index]=target[index]
        else:
            alpha=BLEND_FRACTIONS[variant]
            values=[a+alpha*(b-a) for a,b in zip(values,target)]
        if values==source['values']:raise ValueError('Novel action must differ from original recovery')
    elif variant not in ('parent_exact','recovery_exact','control_exact'):
        raise ValueError('Unknown frozen variant')
    return values


def make_plan(rows, controls, rule):
    validate_rule(rule);pairs=source_pairs(rows);plan=[]
    if len(controls)!=3 or [r['role'] for r in controls]!=list(ROLES):
        raise ValueError('Exactly the three original17 matched controls required')
    if not all(r.get('native_release') is True for r in controls):raise ValueError('Original controls must be admitted')
    def append(source,variant,phase,repeat,donor=None,reference=None):
        reference=source if reference is None else reference
        values=variant_values(source,donor,variant)
        request={k:copy.deepcopy(source[k]) for k in META}
        request.update(id=f'ik-{phase}-{source["context_id"]}-{variant}-r{repeat}',phase=phase,
            role=source['role'] if phase.startswith('controls_') else 'diagnostic',
            method='control' if phase.startswith('controls_') else variant,variant=variant,repeat=repeat,
            context_id=source['context_id'],bucket_xy_m=copy.deepcopy(source['bucket_xy_m']),
            values=values,command_values=copy.deepcopy(values),source_record_id=source['id'],source_record_sha256=digest(source),
            donor_record_id=donor['id'] if donor else None,donor_record_sha256=digest(donor) if donor else None,
            realization_reference_record_id=reference['id'],realization_reference_record_sha256=digest(reference),
            parameter_indices=copy.deepcopy(COORDINATE_GROUPS.get(variant,list(range(5)) if variant in BLEND_FRACTIONS else [])),
            interpolation_alpha=BLEND_FRACTIONS.get(variant),
            adapter_directory='native-controls' if phase.startswith('controls_') else 'native',
            substeps=64,perturbation_mixed_units=[0.]*5,selection_or_learning=False,
            original_role=source['role'],posthoc_development=True,
            admission_mode=('new_action_strict_proof' if variant in NEW_VARIANTS else
                'exact_parent_branch' if source['native_release'] else 'cached_parent_refusal'))
        plan.append(request)
    for source in controls:
        for repeat in range(3):append(source,'control_exact','controls_pre',repeat)
    for pair in pairs:
        for variant in VARIANTS:
            source=pair['parent_bc' if variant=='parent_exact' else 'recovery_bc']
            for repeat in range(3):
                append(source,variant,'diagnostic',repeat,pair['parent_bc'] if variant in NEW_VARIANTS else None,pair['parent_bc'])
    for source in controls:
        for repeat in range(3):append(source,'control_exact','controls_post',repeat)
    if len(plan)!=MAX_REQUESTS or len({r['id'] for r in plan})!=MAX_REQUESTS:raise ValueError('Exactly66 unique slots required')
    return plan


def realization_comparison(row, reference):
    # A rejected recovery action has no measured pose; never compare with its
    # placeholder zero joints. All diagnostics use the admitted parent basis.
    value=source_realization(row,reference)
    if value is not None:
        value['basis']='exact_control_source' if row['phase'].startswith('controls_') else 'matched_successful_parent'
    return value


class MatchedEngine(HistoricalMatchedEngine):
    def __init__(self,*args,rule,**kwargs):
        self.rule=validate_rule(rule)
        super().__init__(*args,**kwargs)

    def evaluate(self,requests,phase):
        for request in requests:
            if request['phase']!=phase:raise ValueError('Exact execution phase required')
            source=self.sources[request['source_record_id']]
            if digest(source)!=request['source_record_sha256']:raise ValueError('Immutable parent row changed')
            reference=self.sources[request['realization_reference_record_id']]
            if digest(reference)!=request['realization_reference_record_sha256'] or reference['native_release'] is not True:
                raise ValueError('Admitted realization reference changed')
            variant=request['variant'];donor=None
            if variant in NEW_VARIANTS:
                donor=self.sources[request['donor_record_id']]
                if (digest(donor)!=request['donor_record_sha256'] or donor['context_id']!=source['context_id']
                        or donor['method']!='parent_bc' or not captured(donor)):
                    raise ValueError('Matched captured parent donor changed')
            if phase.startswith('controls_'):
                if reference['id']!=source['id'] or variant!='control_exact':raise ValueError('Exact control reference required')
            elif reference['method']!='parent_bc' or reference['context_id']!=source['context_id'] or not captured(reference):
                raise ValueError('Matched successful parent realization basis required')
            expected=variant_values(source,donor,variant)
            if (request['values']!=expected or request['command_values']!=expected
                    or request['bucket_xy_m']!=source['bucket_xy_m']):
                raise ValueError('Only frozen coordinate paths may differ')
            if (request['parameter_indices']!=COORDINATE_GROUPS.get(variant,list(range(5)) if variant in BLEND_FRACTIONS else [])
                    or request['interpolation_alpha']!=BLEND_FRACTIONS.get(variant)):
                raise ValueError('Requested coordinate-path metadata changed')
        rows,timing=self.engine.evaluate(requests,phase)
        if [r['id'] for r in rows]!=[r['id'] for r in requests]:raise RuntimeError('Request order/accounting changed')
        env=self.engine.adapter.env
        if env is not None and (env.effective_options!=self.parent_options or env.compiled_geometry!=self.parent_geometry):
            raise RuntimeError('Original native options or geometry changed')
        for row,request in zip(rows,requests):
            source=self.sources[request['source_record_id']]
            if row['variant'] not in NEW_VARIANTS:
                for key in ('connection_file','connection_sha256','native_release','q10','q14','invalid'):
                    if key=='invalid' and source.get('native_release'):continue
                    if row.get(key)!=source.get(key):raise RuntimeError('Original admission or branch changed: '+key)
            elif not row.get('connection_sha256') or row['connection_sha256']==source.get('connection_sha256'):
                raise RuntimeError('Changed action reused refused original connection proof')
            if row.get('native_release'):
                for key in ('q10','q14'):
                    if row.get('native_initial_'+key)!=np.asarray(row[key],np.float32).astype(float).tolist():
                        raise RuntimeError('Actual native reset differs from admitted branch')
            row['realization_comparison']=realization_comparison(row,self.sources[request['realization_reference_record_id']])
            proofs=[]
            for name,hkey in (('connection_file','connection_sha256'),('context_resolution_file','context_resolution_sha256')):
                path=contained(self.directory,row[name])
                if sha(path)!=row[hkey]:raise RuntimeError('Realized setup evidence changed')
                proofs.append(read(path))
            row['setup_proof']=setup_evidence(row,*proofs)
        return rows,timing
