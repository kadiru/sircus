"""Exact parent-branch replay and one freshly admitted parameter correction.

No GPU is acquired here. The finite worker owns its lease and deadline. Existing
native physics/scoring/admission modules remain unchanged.
"""
from __future__ import annotations
import copy
import json
import math
from pathlib import Path
import shutil

import numpy as np
from .table_training_native import Engine
from .calibrated_admission import (VERSION as GEOMETRY_VERSION, CONNECTION_POLICY_VERSION,
    anchor_search_spec, verify_resolution, verify_connection)
from .benchmark_verify import contained, read, sha, digest

VERSION = 'sircus-table-training-correction-v1'
PARAMETERS = ('board_x_m', 'board_y_m', 'board_height_m', 'tilt_deg', 'drop_gap_m')
META = ('source_hint','effective_source_hint','source_teacher_id','context_resolution_file',
    'context_resolution_sha256','context_home_sha256','context_layout_eligible',
    'context_anchor_admitted','context_hard_reject','observation')
CONTROL_ROLES = ('direct_control','table_control','negative_control')
MAX_REQUESTS = 42
CONTEXTS = tuple(f'table-training-test-{i:02d}' for i in (0,1,18,26))


def validate_rule(rule):
    index = rule.get('parameter_index')
    if type(index) is not int or index != 3:
        raise ValueError('Exactly one named action coordinate must change')
    target = rule.get('target_value')
    if type(target) not in (int, float) or target != 32.:
        raise ValueError('Frozen correction sets board tilt to 32 degrees')
    contexts = rule.get('contexts')
    if contexts != list(CONTEXTS):
        raise ValueError('Exactly the four prespecified development cases required')
    if not isinstance(rule.get('description'),str) or not rule['description'].strip():
        raise ValueError('Evidence-based correction description required')
    if rule.get('parameter') != PARAMETERS[index]:
        raise ValueError('Parameter name and mixed-unit index disagree')
    return copy.deepcopy(rule)


def make_plan(rows, rule):
    rule = validate_rule(rule); plan = []
    def append(source, variant, phase, repeat):
        changed = variant == 'corrected'; values = list(map(float,source['values']))
        if len(values) != 5 or not all(map(math.isfinite,values)):
            raise ValueError('Exact five-coordinate parent action required')
        if changed: values[rule['parameter_index']] = float(rule['target_value'])
        request = {key:copy.deepcopy(source[key]) for key in META}
        request.update(id=f'correction-{phase}-{source["context_id"]}-{variant}-r{repeat}',
            phase=phase,role=source['role'] if phase.startswith('controls_') else 'diagnostic',
            method='control' if phase.startswith('controls_') else variant,
            variant=variant,repeat=repeat,context_id=source['context_id'],
            bucket_xy_m=copy.deepcopy(source['bucket_xy_m']),values=values,command_values=copy.deepcopy(values),
            source_record_id=source['id'],source_record_sha256=digest(source),
            substeps=64,perturbation_mixed_units=[0.]*5,selection_or_learning=False,
            posthoc_development=True)
        plan.append(request)
    controls = []
    for role in CONTROL_ROLES:
        found = [r for r in rows if r.get('phase')=='controls_pre' and r.get('role')==role and r.get('repeat')==0]
        if len(found)!=1 or found[0].get('native_release') is not True:
            raise ValueError('Audited admitted direct/table/miss control required')
        controls.append(found[0])
    for source in controls:
        for repeat in range(3): append(source,'control_exact','controls_pre',repeat)
    for context in rule['contexts']:
        found = [r for r in rows if r.get('phase')=='test' and r.get('method')=='ppo' and r.get('context_id')==context]
        if len(found)!=1 or found[0].get('route_success') or found[0].get('invalid_physics'):
            raise ValueError('Exactly one previously observed PPO failure required')
        source=found[0]
        if (not source.get('native_release') or source['values'][3]>=22
                or math.dist(source['values'][:2],source['bucket_xy_m'])<.45):
            raise ValueError('Only admitted shallow far-shot failures are eligible')
        for variant in ('baseline','corrected'):
            for repeat in range(3): append(found[0],variant,'diagnostic',repeat)
    for source in controls:
        for repeat in range(3): append(source,'control_exact','controls_post',repeat)
    if len(plan)!=MAX_REQUESTS or len({r['id'] for r in plan})!=len(plan):
        raise ValueError('Bounded unique request plan required')
    return plan


def copy_bound(source_root, target_root, relative, expected):
    original = contained(Path(source_root), relative)
    if sha(original) != expected: raise ValueError('Parent evidence changed: '+relative)
    target = (Path(target_root)/relative).resolve()
    if not target.is_relative_to(Path(target_root).resolve()):
        raise ValueError('Copied evidence escaped output directory')
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        if sha(target)!=expected: raise ValueError('Existing evidence differs: '+relative)
    else: shutil.copyfile(original,target)
    if sha(target)!=expected: raise ValueError('Copied evidence differs')
    return dict(read(target),file=relative,sha256=expected,cache_hit=True)


class MatchedEngine:
    """Hydrate authenticated geometry caches; retain original branch for controls/baseline.

    GeometryResolver writes immutable JSON but has no disk-cache reader. Explicitly
    hydrate its two internal dictionaries after verifying their complete key schema
    and proof bytes. Corrected values receive a new public connect() admission.
    """
    def __init__(self, ref, assets, directory, *, parent, sources, prior, calibration, home, batch=32):
        self.parent, self.directory = Path(parent), Path(directory)
        self.sources = {row['id']:copy.deepcopy(row) for row in sources}
        if len(self.sources)!=len(sources): raise ValueError('Parent source row identities must be unique')
        self.engine = Engine(ref,assets,directory,prior=prior,calibration=calibration,home=home,batch=batch)
        resolver = self.engine.adapter.resolver
        for source in sources:
            resolution = copy_bound(self.parent/'native',self.directory,
                source['context_resolution_file'],source['context_resolution_sha256'])
            verify_resolution(resolution,resolver.sources,resolver.home)
            wanted = digest(dict(version=GEOMETRY_VERSION,bucket_xy_m=source['bucket_xy_m'],
                home_sha256=resolver.home['sha256'],sources=resolver.sources,bounds=resolver.bounds,
                anchor_search=anchor_search_spec()))
            if resolution['key']!=wanted or resolution['bucket_xy_m']!=source['bucket_xy_m']:
                raise ValueError('Resolution cache key differs from original scene/source/home')
            resolver._resolved[wanted]=copy.deepcopy(resolution)
            connection = copy_bound(self.parent/'native',self.directory,
                source['connection_file'],source['connection_sha256'])
            verify_connection(connection,resolution,resolver.home,source['values'])
            key = digest(dict(version=GEOMETRY_VERSION,connection_policy=CONNECTION_POLICY_VERSION,
                resolution_sha256=resolution['sha256'],values=source['values'],bucket_xy_m=source['bucket_xy_m']))
            if Path(source['connection_file']).stem!=key:
                raise ValueError('Connection cache key differs from exact parent action')
            resolver._connected[key]=copy.deepcopy(connection)
        self.parent_options=read(self.parent/'native/effective-options.json')
        self.parent_geometry=read(self.parent/'native/native-compiled-geometry.json')

    def evaluate(self, requests, phase):
        for request in requests:
            source=self.sources[request['source_record_id']]
            if digest(source)!=request['source_record_sha256']:
                raise ValueError('Parent request binding changed')
            expected=copy.deepcopy(source['values'])
            if request['variant']=='corrected':expected[3]=32.
            elif request['variant'] not in ('baseline','control_exact'):
                raise ValueError('Unknown correction variant')
            if request['values']!=expected or request['bucket_xy_m']!=source['bucket_xy_m']:
                raise ValueError('Only frozen tilt correction may change parent inputs')
        rows,timing=self.engine.evaluate(requests,phase)
        if len(rows)!=len(requests):raise RuntimeError('Native request accounting changed')
        env=self.engine.adapter.env
        if env is not None:
            if env.effective_options!=self.parent_options or env.compiled_geometry!=self.parent_geometry:
                raise RuntimeError('Native options or compiled geometry differ from audited parent')
        for row,request in zip(rows,requests):
            source=self.sources[request['source_record_id']]
            if row['variant'] in ('baseline','control_exact'):
                for key in ('connection_file','connection_sha256','native_release','q10','q14'):
                    if row.get(key)!=source.get(key):
                        raise RuntimeError('Exact replay changed parent admission/joint branch: '+key)
            elif not row.get('connection_sha256') or row['connection_sha256']==source.get('connection_sha256'):
                raise RuntimeError('Changed action reused original connection proof')
            if row.get('native_release'):
                for key in ('q10','q14'):
                    expected=np.asarray(row[key],dtype=np.float32).astype(float).tolist()
                    if row.get('native_initial_'+key)!=expected:
                        raise RuntimeError('Native time-zero joints differ from admitted branch')
        return rows,timing
