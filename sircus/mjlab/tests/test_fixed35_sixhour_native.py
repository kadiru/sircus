"""More bouncer branches without changed physics or historical proof identity."""
import copy,json
from collections import deque
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch
from sircus.mjlab import fixed35_sixhour_native as N
from sircus.mjlab import fixed35_broad_native as old
from sircus.mjlab.tests import test_fixed35_broad_native as old_fixtures


class BranchTests(unittest.TestCase):
    def fixture(self):
        original,hint,arms,endpoint,anchor=old_fixtures.BroadTests().fixture()
        resolver=N.SixHourResolver(original.legacy,{}, {}, 'assets')
        return resolver,hint,anchor

    def test_only_rule_version_and_admitted_branch_limit_change(self):
        expected=dict(old.BRANCH_RULE,version=N.SIXHOUR_VERSION,maximum_admitted_branches=8)
        self.assertEqual(N.SIXHOUR_RULE,expected)
        self.assertEqual(old.BRANCH_RULE['maximum_admitted_branches'],2)

    def test_eight_distinct_bouncer_branches_and_pool_cache(self):
        resolver,hint,anchor=self.fixture()
        candidates=[]
        for i in range(12):
            candidates.append(dict(values=hint['values'],bucket_xy_m=hint['bucket_xy_m'],q10=[0.]*6,q14=[i*.1]*6,endpoint={'reason':None}))
        # A dropper-only alternative may not consume a bouncer slot.
        candidates.insert(1,dict(candidates[0],q10=[.5]*6))
        with patch.object(N,'verify_resolution'),patch.object(N,'verify_connection'),patch.object(resolver,'solve',return_value=dict(candidates=candidates,passed=True,reason=None)) as solve:
            pool=resolver.pool(anchor,hint['values']);again=resolver.pool(anchor,hint['values'])
            proof=resolver.prepared_connection(anchor,pool,7)
        self.assertEqual(len(pool['admitted']),8);self.assertEqual(pool,again);self.assertEqual(solve.call_count,1)
        self.assertEqual(resolver.planner.plan.call_count,8)
        self.assertEqual([r['candidate_index'] for r in pool['admitted']],[0,2,3,4,5,6,7,8])
        self.assertEqual(proof['broad_branch_slot'],7);self.assertEqual(proof['broad_rule_version'],N.SIXHOUR_VERSION)
        old_key=N.digest(dict(version=old.BROAD_VERSION,pool_key=pool['pool_key'],broad_pool_sha256=pool['sha256'],broad_branch_slot=7))
        self.assertNotEqual(proof['broad_key'],old_key)

    def test_same_strict_path_failures_do_not_fill_eight_slots(self):
        resolver,hint,anchor=self.fixture();resolver.planner.plan.return_value=dict(passed=False,reason='blocked')
        candidates=[dict(values=hint['values'],bucket_xy_m=hint['bucket_xy_m'],q10=[0.]*6,q14=[i*.1]*6,endpoint={'reason':None}) for i in range(16)]
        with patch.object(N,'verify_resolution'),patch.object(resolver,'solve',return_value=dict(candidates=candidates,passed=True,reason=None)):
            pool=resolver.pool(anchor,hint['values'])
        self.assertEqual(pool['admitted'],[]);self.assertEqual(resolver.planner.plan.call_count,16)
        self.assertEqual(pool['reason'],'finite_requested_setup_path_unresolved')

    def test_inherited_prepare_never_starts_physics_and_binds_all_eight(self):
        engine,proposal,pool=old_fixtures.PreparationTests().fixture();engine.__class__=N.SixHourEngine
        pool['admitted']=list(range(8))
        packets,records=engine.prepare([proposal])
        self.assertEqual(len(packets),8);self.assertEqual(records[0]['native_releases'],0)
        self.assertEqual(records[0]['admitted_request_sha256'],[N.digest(r) for r in packets])
        engine.adapter._environment.assert_not_called()


class HistoricalTests(unittest.TestCase):
    def fixture(self):
        home={'sha256':'home'};sources=[];bounds={};values=[.6,0,.3,28,.2];bucket=[.86,.025]
        resolution=dict(bucket_xy_m=bucket,sha256='resolution-sha',file='resolution.json')
        resolution['key']=N.digest(dict(version=N.VERSION,bucket_xy_m=bucket,home_sha256=home['sha256'],sources=sources,bounds=bounds,anchor_search=N.anchor_search_spec()))
        behavior={'version':N.BEHAVIOR_VERSION,'yaw_rad':.1}
        behavior_key=N.digest(dict(version=N.BEHAVIOR_VERSION,connection_policy=N.CONNECTION_POLICY_VERSION,resolution_sha256=resolution['sha256'],values=values,bucket_xy_m=bucket,release_behavior=behavior))
        pool_key=N.digest(dict(version=old.BROAD_VERSION,branch_rule=old.BRANCH_RULE,behavior_key=behavior_key,resolution_sha256=resolution['sha256']))
        selected=dict(slot=1,q10=[.1]*6,q14=[.3]*6,endpoint={'reason':None},path_attempt_index=0)
        pool=dict(version=old.BROAD_VERSION,branch_rule=copy.deepcopy(old.BRANCH_RULE),behavior_key=behavior_key,pool_key=pool_key,
            file='broad-pools/'+pool_key+'.json',sha256='pool-sha',admitted=[dict(selected,slot=0),selected],path_attempts=[{'passed':True}],endpoint_attempts={})
        pool.update(values=values,bucket_xy_m=bucket,resolution_file=resolution['file'],resolution_sha256=resolution['sha256'],
            home_sha256=home['sha256'],release_behavior=behavior,selection_uses_flight_labels=False)
        key=N.digest(dict(version=old.BROAD_VERSION,pool_key=pool_key,broad_pool_sha256=pool['sha256'],broad_branch_slot=1))
        connection=dict(passed=True,release_behavior=behavior,behavior_key=behavior_key,broad_rule_version=old.BROAD_VERSION,broad_key=key,
            broad_branch_slot=1,broad_pool_file=pool['file'],broad_pool_sha256=pool['sha256'],broad_pool_key=pool_key,
            q10=selected['q10'],q14=selected['q14'],endpoint=selected['endpoint'],proof=pool['path_attempts'][0],
            endpoint_attempts=pool['endpoint_attempts'],path_attempts=pool['path_attempts'])
        row=dict(id='parent-broad',native_release=True,route_success=True,invalid=None,invalid_physics=False,
            values=values,bucket_xy_m=bucket,context_resolution_file=resolution['file'],context_resolution_sha256=resolution['sha256'],
            connection_file='fixed-yaw-connections/'+key+'.json',connection_sha256='connection-sha',q10=selected['q10'],q14=selected['q14'],
            **{k:connection[k] for k in old.BROAD_FIELDS})
        rows={};resolver=SimpleNamespace(parent_rows=rows,parent_connections={},base=SimpleNamespace(_resolved={}),sources=sources,home=home,bounds=bounds)
        engine=SimpleNamespace(parent_rows=rows,adapter=SimpleNamespace(resolver=resolver),directory=Path('/target'),source_receipts=[])
        files={resolution['file']:resolution,row['connection_file']:connection,pool['file']:pool}
        return engine,row,files,resolution,connection,pool

    def hydrate(self,engine,row,files):
        with patch.object(N,'copy_bound',side_effect=lambda origin,target,file,checksum:copy.deepcopy(files[file])),patch.object(N,'verify_resolution'),patch.object(N,'verify_connection'):
            N.hydrate_broad_sources(engine,[row],{row['id']:Path('/parent/native-yaw')})

    def test_original_broad_proof_keeps_original_version_slot_and_joints(self):
        engine,row,files,resolution,connection,pool=self.fixture();self.hydrate(engine,row,files)
        self.assertEqual(engine.parent_rows[row['id']],row)
        self.assertEqual(engine.adapter.resolver.parent_connections[row['id']],connection)
        self.assertEqual(engine.adapter.resolver.base._resolved[resolution['key']],resolution)
        self.assertEqual(engine.source_receipts[0]['broad_rule_version'],old.BROAD_VERSION)
        self.assertEqual(connection['broad_branch_slot'],1)

    def test_tampered_source_joint_or_pool_branch_cannot_hydrate(self):
        for field in ('q14','slot','version','capture','pool_geometry'):
            engine,row,files,resolution,connection,pool=self.fixture()
            if field=='q14':row['q14']=[.4]*6
            elif field=='slot':connection['broad_branch_slot']=7
            elif field=='version':connection['broad_rule_version']=N.SIXHOUR_VERSION
            elif field=='capture':row['route_success']=False
            else:pool['bucket_xy_m']=[.1,.2]
            with self.subTest(field=field),self.assertRaises(ValueError):self.hydrate(engine,row,files)
            self.assertEqual(engine.parent_rows,{})

    def test_preserved_metadata_comes_from_original_connection(self):
        engine,row,files,_,connection,_=self.fixture()
        with TemporaryDirectory() as tmp:
            engine=N.SixHourEngine.__new__(N.SixHourEngine);engine.directory=Path(tmp)
            path=engine.directory/row['connection_file'];path.parent.mkdir(parents=True);path.write_text(json.dumps(connection))
            output=dict(id='preserved',exact_parent_action=True,connection_file=row['connection_file'])
            with patch.object(old.BroadEngine,'evaluate',return_value=([output],{})):
                rows,_=engine.evaluate([output],'preserved')
            self.assertEqual({k:rows[0][k] for k in old.BROAD_FIELDS},{k:connection[k] for k in old.BROAD_FIELDS})
            conflict=dict(output,broad_rule_version=N.SIXHOUR_VERSION)
            with patch.object(old.BroadEngine,'evaluate',return_value=([conflict],{})),self.assertRaisesRegex(RuntimeError,'metadata conflict'):
                engine.evaluate([conflict],'preserved')


if __name__=='__main__':unittest.main()
