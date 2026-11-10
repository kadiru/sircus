"""Independent saved-candidate, unchanged-dropper and no-new-IK evidence tests."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from sircus.mjlab import fixed35_bouncer_verify as V
from sircus.mjlab import fixed35_bouncer_contract as C
from sircus.mjlab.benchmark_verify import Checks,EvidenceError
from sircus.mjlab.tests.test_fixed35_bouncer_contract import fixture
from sircus.mjlab.tests.test_fixed35_six_contract import score_fields


def proof_fixture(root):
    data=fixture();s=data['numeric_sources'][0];pool=data['candidate_pools'][s['id']]
    pp=root/s['branch_pool_file'];pp.parent.mkdir(parents=True,exist_ok=True);pp.write_text(json.dumps(pool))
    s['branch_pool_sha256']=V.sha(pp);s['context_resolution_file']='resolution.json';s['context_resolution_sha256']='a'*64
    s['connection_file']='fixed-yaw-connections/parent.json';p=root/s['connection_file'];p.parent.mkdir(exist_ok=True)
    original=dict(release_behavior={'version':'behavior'},behavior_key='behavior-key');p.write_text(json.dumps(original));s['connection_sha256']=V.sha(p)
    plan,_=V.independent_plan(data,Checks());r=plan[0];c=pool['endpoint_attempts']['candidates'][r['candidate_index']]
    endpoint=dict(source_candidate=copy.deepcopy(c),source_candidate_sha256=V.digest(c),fresh_endpoint={'reason':0},native_pose_errors=[],targets={},
        candidates=[dict(copy.deepcopy(c),endpoint={'reason':0})],passed=True,reason=None,ik_solve_performed=False,selection_uses_flight_labels=False)
    key=V.digest(dict(version=V.BOUNCER_VERSION,resolution_sha256=s['context_resolution_sha256'],parent_connection_sha256=s['connection_sha256'],
        candidate_index=r['candidate_index'],candidate_sha256=V.digest(c),candidate_pool_sha256=s['branch_pool_sha256'],values=s['values'],bucket_xy_m=s['bucket_xy_m']))
    conn={k:copy.deepcopy(r[k]) for k in ('candidate_index','candidate_sha256','candidate_pool_file','candidate_pool_sha256','parent_row_id','parent_row_sha256','selected_q10','selected_q14')}
    conn.update(original,parent_connection_file=s['connection_file'],parent_connection_sha256=s['connection_sha256'],resolution_file=s['context_resolution_file'],
        resolution_sha256=s['context_resolution_sha256'],bouncer_rule_version=V.BOUNCER_VERSION,ik_solve_performed=False,bouncer_key=key,
        endpoint_attempts=endpoint,endpoint_eligible=True,passed=True,q10=c['q10'],q14=c['q14'],endpoint={'reason':0},proof={'passed':True},path_attempts=[{'passed':True}])
    r.update(bouncer_key=key,bouncer_rule_version=V.BOUNCER_VERSION,ik_solve_performed=False,connection_file='fixed-yaw-connections/'+key+'.json',native_release=True,q10=c['q10'],q14=c['q14'])
    return data,r,conn


class IndependentTests(unittest.TestCase):
    def test_literal_protocol_rejects_extra_search_training_or_changed_dropper(self):
        V.protocol_checks(C.protocol(),Checks())
        for key,value in [('maximum_requests',100),('new_ik',True),('training',True),('substeps',128),('q10_preserved_exactly',False),('wrapped_q14_tolerance_rad',.001),('worker_limit_s',7200)]:
            p=C.protocol();p[key]=value
            with self.subTest(key=key),self.assertRaises(EvidenceError):V.protocol_checks(p,Checks())

    def test_independent_plan_matches_without_calling_producer_selector(self):
        data=fixture();expected=C.make_plan(data['numeric_sources'],data['candidate_pools'])
        with patch.object(C,'make_plan',side_effect=AssertionError('producer called')),patch.object(C,'select_alternatives',side_effect=AssertionError('producer called')):
            actual=V.independent_plan(data,Checks())
        self.assertEqual(actual,expected);self.assertEqual(len(actual[0]),24)
        self.assertEqual(V.independent_preserved(data),C.preserved_plan(data['preserved']))

    def test_periodic_duplicates_and_changed_dropper_excluded(self):
        d=fixture();s=d['numeric_sources'][0];p=d['candidate_pools'][s['id']]
        selected,eligible=V.select_candidates(s,p,Checks())
        self.assertEqual([x['index'] for x in selected],[2,1]);self.assertEqual([x['index'] for x in eligible],[1,2,4])
        p['endpoint_attempts']['candidates']=[p['endpoint_attempts']['candidates'][i] for i in (0,2,3,5)]
        selected,_=V.select_candidates(s,p,Checks());self.assertEqual([x['index'] for x in selected],[1])

    def test_wrong_numeric_pool_rejected(self):
        d=fixture();p=d['candidate_pools'][d['numeric_sources'][0]['id']];p['values']=p['values'][:];p['values'][3]+=.1
        with self.assertRaises(EvidenceError):V.independent_plan(d,Checks())

    def test_exact_repeats_match_contract_and_require_captured_source(self):
        d=fixture();p=V.independent_plan(d,Checks())[0][0]
        r=dict(p,**score_fields(True),q10=p['selected_q10'],q14=p['selected_q14'],connection_sha256='b'*64)
        for group in range(3):self.assertEqual(V.independent_repeat(r,group),C.repeat_plan(r,group))
        r['route_success']=False
        with self.assertRaises(EvidenceError):V.independent_repeat(r,0)

    def test_saved_candidate_identity_exact_original_not_fresh_relabel(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);d,r,c=proof_fixture(root);candidate,endpoint=V.audit_candidate_identity(root,r,c,d,Checks())
            self.assertEqual(candidate,endpoint['source_candidate']);self.assertNotEqual(candidate,endpoint['candidates'][0])

    def test_changed_provenance_joints_ik_or_hidden_candidates_fail(self):
        for mutation in ('parent_hash','candidate_hash','joint','ik','new_candidate','path','key','dropper','behavior'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);d,r,c=proof_fixture(root)
                if mutation=='parent_hash':r['parent_row_sha256']='bad'
                if mutation=='candidate_hash':c['candidate_sha256']='bad'
                if mutation=='joint':c['q14']=list(c['q14']);c['q14'][0]+=.1
                if mutation=='ik':c['ik_solve_performed']=True
                if mutation=='new_candidate':c['endpoint_attempts']['candidates'].append(copy.deepcopy(c['endpoint_attempts']['candidates'][0]))
                if mutation=='path':c['proof']={'passed':True,'new':True}
                if mutation=='key':r['bouncer_key']='bad'
                if mutation=='dropper':r['selected_q10']=list(r['selected_q10']);r['selected_q10'][0]+=.1
                if mutation=='behavior':c['release_behavior']={'version':'different'}
                with self.assertRaises(EvidenceError):V.audit_candidate_identity(root,r,c,d,Checks())

    def test_tampered_archived_pool_bytes_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);d,r,c=proof_fixture(root);(root/r['candidate_pool_file']).write_text('{}')
            with self.assertRaises(EvidenceError):V.audit_candidate_identity(root,r,c,d,Checks())

    def test_unadmitted_candidate_cannot_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);d,r,c=proof_fixture(root);c['passed']=False
            with self.assertRaises(EvidenceError):V.audit_candidate_identity(root,r,c,d,Checks())


if __name__=='__main__':unittest.main()
