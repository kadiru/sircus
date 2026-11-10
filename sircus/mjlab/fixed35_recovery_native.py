"""Fresh native flight with authenticated unchanged parent yaw/joint branches."""
from __future__ import annotations
import copy
from pathlib import Path
import numpy as np
from .benchmark_verify import digest, read
from .calibrated_admission import (VERSION, CONNECTION_POLICY_VERSION,
    anchor_search_spec, verify_resolution, verify_connection)
from .fixed35_imitation_native import FixedYawEngine, BEHAVIOR_VERSION
from .table_correction_native import copy_bound


class RecoveryEngine(FixedYawEngine):
    """Only new numeric actions invoke IK; exact parent/repeat branches cannot drift."""
    def __init__(self, ref, assets, directory, *, parent, sources, **kwargs):
        super().__init__(ref,assets,directory,**kwargs)
        self.parent_rows={r['id']:copy.deepcopy(r) for r in sources}
        if len(self.parent_rows)!=len(sources):raise ValueError('Unique historical source rows required')
        resolver=self.adapter.resolver
        for source in sources:
            resolution=copy_bound(Path(parent)/'native-yaw',directory,source['context_resolution_file'],source['context_resolution_sha256'])
            verify_resolution(resolution,resolver.sources,resolver.home)
            key=digest(dict(version=VERSION,bucket_xy_m=source['bucket_xy_m'],home_sha256=resolver.home['sha256'],
                sources=resolver.sources,bounds=resolver.bounds,anchor_search=anchor_search_spec()))
            if resolution['key']!=key or resolution['bucket_xy_m']!=source['bucket_xy_m']:
                raise ValueError('Parent geometry resolution key changed')
            resolver.base._resolved[key]=copy.deepcopy(resolution)
            connection=copy_bound(Path(parent)/'native-yaw',directory,source['connection_file'],source['connection_sha256'])
            verify_connection(connection,resolution,resolver.home,source['values'])
            behavior=connection.get('release_behavior',{})
            key=digest(dict(version=BEHAVIOR_VERSION,connection_policy=CONNECTION_POLICY_VERSION,
                resolution_sha256=resolution['sha256'],values=source['values'],bucket_xy_m=source['bucket_xy_m'],release_behavior=behavior))
            if (behavior.get('version')!=BEHAVIOR_VERSION or connection.get('behavior_key')!=key
                    or Path(source['connection_file']).parent.as_posix()!='fixed-yaw-connections'
                    or Path(source['connection_file']).stem!=key):
                raise ValueError('Parent fixed-yaw behavior key changed')
            resolver._connected[key]=copy.deepcopy(connection)
        self.parent_options=read(Path(parent)/'native-yaw/effective-options.json')
        self.parent_geometry=read(Path(parent)/'native-yaw/native-compiled-geometry.json')

    def evaluate(self, requests, phase):
        for r in requests:
            source=self.parent_rows[r['parent_row_id']]
            if digest(source)!=r['parent_row_sha256'] or r['bucket_xy_m']!=source['bucket_xy_m']:
                raise ValueError('Frozen original parent/layout binding changed')
            if r.get('exact_parent_action') and r['values']!=source['values']:
                raise ValueError('Exact parent replay changed action')
            if r.get('phase')=='preserved' and not r.get('exact_parent_action'):
                raise ValueError('Preserved success requires exact parent action')
        rows,timing=super().evaluate(requests,phase)
        if self.adapter.env is not None and (self.adapter.env.effective_options!=self.parent_options
                                            or self.adapter.env.compiled_geometry!=self.parent_geometry):
            raise RuntimeError('Native options/geometry differ from audited parent')
        for r in rows:
            source=self.parent_rows[r['parent_row_id']]
            if r.get('exact_parent_action'):
                for key in ('connection_file','connection_sha256','q10','q14','native_release'):
                    if r.get(key)!=source.get(key):raise RuntimeError('Exact parent branch changed: '+key)
            if r.get('phase')=='repeat':
                for actual,expected in (('connection_sha256','selected_connection_sha256'),('q10','selected_q10'),('q14','selected_q14')):
                    if r.get(actual)!=r.get(expected):raise RuntimeError('Fresh repeat changed discovered branch: '+actual)
            if r.get('native_release'):
                for key in ('q10','q14'):
                    if r.get('native_initial_'+key)!=np.asarray(r[key],np.float32).astype(float).tolist():
                        raise RuntimeError('Native release differs from admitted joints')
        return rows,timing
