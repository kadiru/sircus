import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from sircus.mjlab import fixed35_sixhour_verify as V
from sircus.mjlab.benchmark_verify import EvidenceError, digest, sha


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))
    return sha(path)


class RuleTests(unittest.TestCase):
    def test_endpoint_reuse_normalizes_only_explicit_rule_without_mutating_source(self):
        endpoint = dict(branch_rule=copy.deepcopy(V.SIXHOUR_RULE), pair_attempts=[{'checked': 'original'}])
        before = copy.deepcopy(endpoint)
        with patch.object(V.broad, 'audit_endpoint', return_value='checked') as reuse:
            result = V.audit_endpoint(endpoint, {}, {}, 'assets', (), None, V.AuditChecks(), 'example')
        self.assertEqual(result, 'checked')
        self.assertEqual(endpoint, before)
        delegated = reuse.call_args.args[0]
        self.assertEqual(delegated['branch_rule'], V.broad.BRANCH_RULE)
        self.assertEqual(delegated['pair_attempts'], before['pair_attempts'])
        self.assertIsNot(delegated['pair_attempts'], endpoint['pair_attempts'])

    def test_strict_pose_tolerance_or_seed_change_cannot_hide_in_rule_adapter(self):
        for key, value in [('position_tolerance_m', .005), ('seed_base', 99), ('maximum_admitted_branches', 9)]:
            endpoint = dict(branch_rule=dict(V.SIXHOUR_RULE, **{key: value}))
            with patch.object(V.broad, 'audit_endpoint', side_effect=AssertionError('must reject first')):
                with self.assertRaises(EvidenceError):
                    V.audit_endpoint(endpoint, {}, {}, 'assets', (), None, V.AuditChecks(), 'changed')

    def test_new_connection_cache_version_and_eighth_slot_are_explicit(self):
        pool = dict(pool_key='pool', bucket_xy_m=[1., 0.], values=[.6, 0., .2, 28., .1],
            endpoint_attempts={}, path_attempts=[{'passed': True}], timing={}, release_behavior={}, behavior_key='behavior',
            admitted=[dict(q10=[.1]*6, q14=[i*.1]*6, endpoint={}, path_attempt_index=0) for i in range(8)])
        anchor = dict(file='anchor.json', sha256='a'*64, anchor_values=[0]*5, anchor_derivation={})
        actual = V.expected_connection(pool, anchor, dict(sha256='h'*64), 'pool.json', '1'*64, 7)
        self.assertEqual(actual['q14'], pool['admitted'][7]['q14'])
        self.assertEqual(actual['broad_rule_version'], V.SIXHOUR_VERSION)
        self.assertEqual(actual['broad_key'], digest(dict(version=V.SIXHOUR_VERSION,
            pool_key='pool', broad_pool_sha256='1'*64, broad_branch_slot=7)))
        old = V.broad.expected_connection(pool, anchor, dict(sha256='h'*64), 'pool.json', '1'*64, 7)
        self.assertNotEqual(actual['broad_key'], old['broad_key'])


class PoolTests(unittest.TestCase):
    def run_pool(self, directory, count=8, duplicate=False):
        home = dict(sha256='h'*64, q10=[0]*6, q14=[0]*6)
        anchor = dict(home_passed=True, bucket_xy_m=[1., 0.])
        ah = write(directory/'anchor.json', anchor)
        candidates = [dict(q10=[0]*6, q14=[i*.1, 0, 0, 0, 0, 0], endpoint={'passed': True}) for i in range(count)]
        if duplicate:
            candidates[-1]['q14'] = candidates[0]['q14'][:]
        paths = [dict(passed=True, id=i) for i in range(count)]
        admitted = [dict(c, slot=i, candidate_index=i, path_attempt_index=i) for i, c in enumerate(candidates)]
        key = digest(dict(version=V.SIXHOUR_VERSION, branch_rule=V.SIXHOUR_RULE,
            behavior_key='behavior', resolution_sha256=ah))
        pool = dict(version=V.SIXHOUR_VERSION, branch_rule=V.SIXHOUR_RULE, selection_uses_flight_labels=False,
            values=[.6, 0., .2, 28., .1], bucket_xy_m=[1., 0.], resolution_file='anchor.json', resolution_sha256=ah,
            home_sha256=home['sha256'], pool_key=key, endpoint_attempts=dict(candidates=candidates, passed=True, reason=None),
            path_attempts=paths, admitted=admitted, reason=None)
        name='broad-pools/'+key+'.json'
        ph=write(directory/name, pool)
        record=dict(broad_pool_file=name, broad_pool_sha256=ph, broad_pool_key=key)
        proposal=dict(id='example', values=pool['values'], bucket_xy_m=pool['bucket_xy_m'])
        with patch('sircus.mjlab.calibrated_admission.verify_resolution', return_value=True), \
             patch('sircus.mjlab.calibrated_admission.verify_path', return_value=True) as path_check, \
             patch.object(V.fixed, 'audit_behavior', return_value=({}, 'behavior')), \
             patch.object(V, 'audit_endpoint', return_value={}):
            result=V.audit_pool(directory,record,proposal,{},'assets',(),None,home,dict(records=[]),V.AuditChecks())
        return result,path_check.call_count

    def test_all_eight_distinct_slots_have_paths_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            result,calls=self.run_pool(Path(directory))
        self.assertEqual(len(result[0]['admitted']),8)
        self.assertEqual(calls,8)

    def test_ninth_slot_and_q14_clone_are_rejected(self):
        for options in (dict(count=9), dict(duplicate=True)):
            with self.subTest(options=options), tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(EvidenceError):self.run_pool(Path(directory),**options)



class PlanTests(unittest.TestCase):
    def fixture(self):
        from sircus.mjlab.tests.test_fixed35_sixhour_contract import ContractTests
        data,confirmed,kwargs,plan=ContractTests().plan(500)
        return data,confirmed,kwargs,plan

    def test_primary_reconstructed_without_using_producer_and_all_four_anchors(self):
        from sircus.mjlab import fixed35_sixhour_contract as C
        data,confirmed,kw,expected=self.fixture();before=copy.deepcopy(data)
        with patch.object(C,'make_plan',side_effect=AssertionError('producer forbidden')):
            actual=V.independent_plan(data['numeric_sources'],confirmed,{},None,limit=500,**kw)
        self.assertEqual(actual,expected);self.assertEqual(data,before)
        anchors=[r['id'] for r in data['numeric_sources'] if r['context_id']==V.TARGET]
        self.assertEqual(len(anchors),4)
        self.assertTrue(all(actual['seed_counts'][identity]>0 for identity in anchors))
        for row in actual['rows']:
            self.assertEqual(row['context_id'],V.TARGET)
            self.assertIn(row['parent_row_id'],anchors)

    def test_window_reconstructed_from_exact_confirmed_sources(self):
        from sircus.mjlab import fixed35_sixhour_contract as C
        data,confirmed,kw,_=self.fixture()
        target=copy.deepcopy(next(r for r in data['numeric_sources'] if r['context_id']==V.TARGET))
        seeds=confirmed+[target]
        expected=C.make_window_plan(data['numeric_sources'],seeds,{},None,limit=200,**kw)
        with patch.object(C,'make_window_plan',side_effect=AssertionError('producer forbidden')):
            actual=V.independent_plan(data['numeric_sources'],seeds,{},None,stage='window',limit=200,**kw)
        self.assertEqual(actual,expected)
        self.assertEqual({r['context_id'] for r in actual['rows']},set(V.CASE_IDS))

    def test_only_geometric_diagnostics_receive_roundoff_tolerance(self):
        data,confirmed,kwargs,plan=self.fixture()
        expected=V.independent_plan(data['numeric_sources'],confirmed,{},None,limit=500,**kwargs)
        V.audit_plan(plan,expected,V.AuditChecks(),'exact')
        changed=copy.deepcopy(plan);changed['rows'][0]['values'][0]+=1e-13
        with self.assertRaises(EvidenceError):V.audit_plan(changed,expected,V.AuditChecks(),'changed command')
        changed=copy.deepcopy(plan);changed['rows'][0]['conditional_gap_interval_m'][1]+=1e-13
        V.audit_plan(changed,expected,V.AuditChecks(),'roundoff diagnostic')

    def test_gap_cannot_be_clipped_to_make_a_rejected_action_eligible(self):
        data,confirmed,_,_=self.fixture()
        actual=V.independent_plan(data['numeric_sources'],confirmed,{},None,limit=20,
            interval_fn=lambda *a,**k:(.05,.050000000001),reach_fn=lambda *a,**k:dict(passed=True))
        self.assertEqual(actual['rows'],[])
        self.assertGreater(actual['geometric_declines']['gap_outside_reach_interval'],0)

    def test_protocol_and_same_context_preservation_ids_are_distinct(self):
        from sircus.mjlab import fixed35_sixhour_contract as C
        data,confirmed,_,_=self.fixture();data['preserved']+=confirmed
        self.assertEqual(V.independent_preserved(data),C.preserved_plan(data['preserved']))
        ids=[r['id'] for r in V.independent_preserved(data)]
        self.assertEqual(len(ids),len(set(ids)))
        V.protocol_checks(C.protocol(),V.AuditChecks())
        for field,value in [('maximum_branches_per_numeric',9),('worker_limit_s',21600),('substeps',128),('policy_updated',True)]:
            with self.subTest(field=field),self.assertRaises(EvidenceError):
                V.protocol_checks(dict(C.protocol(),**{field:value}),V.AuditChecks())


class PreservationTests(unittest.TestCase):
    def fixture(self,directory):
        import numpy as np
        connection={key:None for key in V.BROAD_FIELDS}
        connection.update(broad_rule_version=V.broad.BROAD_VERSION,broad_branch_slot=1,broad_key='key',
            broad_pool_file='broad-pools/pool.json',broad_pool_sha256='a'*64,broad_pool_key='pool')
        name='fixed-yaw-connections/key.json';checksum=write(directory/'native-yaw'/name,connection)
        source=dict(id='authenticated-source',values=[.6,0,.3,28,.1],bucket_xy_m=[1.05,.025],q10=[.1]*6,q14=[.2]*6,
            connection_file=name,connection_sha256=checksum,native_release=True,**connection)
        row=dict(source,id='new-preservation',parent_row_id=source['id'],phase='preserved',native_cache_sha256='newcache')
        for field in ('q10','q14'):row['native_initial_'+field]=np.asarray(source[field],np.float32).astype(float).tolist()
        return row,source

    def test_geometry_adapter_keeps_actual_preserved_evidence_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary);row,source=self.fixture(directory);before=copy.deepcopy(row)
            with patch.object(V.broad,'audit_flights',return_value={}) as check,patch.object(V.six,'audit_geometry'):
                result=V.audit_preserved_geometry(directory,[row],dict(preserved=[source]),{},'assets',V.AuditChecks())
            self.assertEqual(row,before)
            self.assertEqual(result['historical_broad_preserved'],1)
            view=check.call_args.args[1][0]
            self.assertEqual(view['phase'],'repeat')
            self.assertEqual(view['q14'],source['q14'])
            self.assertEqual(view['connection_sha256'],source['connection_sha256'])
            self.assertEqual(view['native_cache_sha256'],row['native_cache_sha256'])

    def test_changed_source_joint_or_proof_cannot_use_geometry_adapter(self):
        for change in (dict(q14=[.3]*6),dict(connection_sha256='b'*64)):
            with tempfile.TemporaryDirectory() as temporary:
                directory=Path(temporary);row,source=self.fixture(directory);row.update(change)
                with patch.object(V.broad,'audit_flights',side_effect=AssertionError('must reject first')):
                    with self.assertRaises(EvidenceError):
                        V.audit_preserved_geometry(directory,[row],dict(preserved=[source]),{},'assets',V.AuditChecks())

if __name__=='__main__':unittest.main()
