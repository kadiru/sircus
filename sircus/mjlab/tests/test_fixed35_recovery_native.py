"""Preserve exact admitted parent branches; fresh repeats may not drift."""
import copy
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from sircus.mjlab import fixed35_recovery_native as N
from sircus.mjlab import fixed35_recovery_contract as C
from sircus.mjlab.tests.test_fixed35_recovery_contract import fixture,score_fields


class NativeTests(unittest.TestCase):
    def test_authenticated_cache_hydration_preserves_parent_and_geometry_namespaces(self):
        source=fixture()['preserved'][0];geometry=[{'id':'geometry-anchor'}];home={'sha256':'home'}
        resolver=SimpleNamespace(sources=geometry,home=home,bounds={'mock':True},base=SimpleNamespace(_resolved={}),_connected={})
        resolution=dict(sha256=source['context_resolution_sha256'],bucket_xy_m=source['bucket_xy_m'])
        resolution['key']=N.digest(dict(version=N.VERSION,bucket_xy_m=source['bucket_xy_m'],home_sha256='home',sources=geometry,bounds=resolver.bounds,anchor_search=N.anchor_search_spec()))
        behavior=dict(version=N.BEHAVIOR_VERSION,yaw_rad=0.)
        key=N.digest(dict(version=N.BEHAVIOR_VERSION,connection_policy=N.CONNECTION_POLICY_VERSION,
            resolution_sha256=resolution['sha256'],values=source['values'],bucket_xy_m=source['bucket_xy_m'],release_behavior=behavior))
        source['connection_file']='fixed-yaw-connections/'+key+'.json'
        connection=dict(behavior_key=key,release_behavior=behavior,sha256=source['connection_sha256'])
        def initialize(engine,*args,**kwargs):
            engine.sources=geometry;engine.adapter=SimpleNamespace(resolver=resolver)
        def copied(original,directory,relative,expected):
            return dict(copy.deepcopy(resolution if relative==source['context_resolution_file'] else connection),file=relative)
        with patch.object(N.FixedYawEngine,'__init__',initialize),patch.object(N,'copy_bound',side_effect=copied),\
             patch.object(N,'verify_resolution'),patch.object(N,'verify_connection'),patch.object(N,'read',return_value={}):
            engine=N.RecoveryEngine({},'assets','target',parent='parent',sources=[source])
        self.assertEqual(engine.sources,geometry);self.assertEqual(list(engine.parent_rows),[source['id']])
        self.assertIn(resolution['key'],resolver.base._resolved);self.assertIn(key,resolver._connected)
        self.assertEqual(resolver._connected[key]['sha256'],source['connection_sha256'])

    def fixture(self):
        source=fixture()['preserved'][0];engine=N.RecoveryEngine.__new__(N.RecoveryEngine)
        engine.parent_rows={source['id']:copy.deepcopy(source)};engine.sources=['geometry-api-preserved']
        engine.adapter=SimpleNamespace(env=None);engine.parent_options={};engine.parent_geometry={}
        request=C.preserved_plan([source])[0]
        measured=dict(request,**score_fields(True),q10=source['q10'],q14=source['q14'],
            connection_file=source['connection_file'],connection_sha256=source['connection_sha256'],
            native_initial_q10=np.asarray(source['q10'],np.float32).astype(float).tolist(),
            native_initial_q14=np.asarray(source['q14'],np.float32).astype(float).tolist())
        return engine,source,request,measured

    def test_exact_replay_keeps_geometry_api_and_joint_branch(self):
        engine,source,request,row=self.fixture()
        with patch.object(N.FixedYawEngine,'evaluate',return_value=([row],{})):
            result,_=engine.evaluate([request],'preserved')
        self.assertEqual(result[0]['q10'],source['q10']);self.assertEqual(engine.sources,['geometry-api-preserved'])

    def test_parent_source_or_preserved_action_cannot_change(self):
        for field,value in (('parent_row_sha256','wrong'),('values',[.6,-.2,.5,28,.2]),('exact_parent_action',False)):
            engine,source,request,row=self.fixture();request[field]=value
            with patch.object(N.FixedYawEngine,'evaluate') as native:
                with self.assertRaises(ValueError):engine.evaluate([request],'preserved')
                native.assert_not_called()

    def test_exact_native_branch_or_geometry_drift_refuses_success(self):
        for field,value in (('q10',[3.]*6),('connection_sha256','bad'),('native_release',False),('native_initial_q14',[0.]*6)):
            engine,source,request,row=self.fixture();row[field]=value
            with patch.object(N.FixedYawEngine,'evaluate',return_value=([row],{})):
                with self.assertRaises(RuntimeError):engine.evaluate([request],'preserved')
        engine,source,request,row=self.fixture();engine.adapter.env=SimpleNamespace(effective_options={'changed':1},compiled_geometry={})
        with patch.object(N.FixedYawEngine,'evaluate',return_value=([row],{})):
            with self.assertRaises(RuntimeError):engine.evaluate([request],'preserved')

    def test_fresh_repeat_must_keep_selected_discovery_joints(self):
        engine,source,request,row=self.fixture()
        request.update(phase='repeat',exact_parent_action=False,selected_connection_sha256=row['connection_sha256'],selected_q10=row['q10'],selected_q14=row['q14'])
        row.update(request);row['selected_q10']=[4.]*6
        with patch.object(N.FixedYawEngine,'evaluate',return_value=([row],{})):
            with self.assertRaisesRegex(RuntimeError,'discovered branch'):engine.evaluate([request],'repeat')

if __name__=='__main__':unittest.main()
