"""Independent lineage, native-branch binding and protocol tamper tests."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from sircus.mjlab import fixed35_six_verify as V
from sircus.mjlab import fixed35_six_contract as C
from sircus.mjlab.benchmark_verify import Checks,EvidenceError
from sircus.mjlab.tests.test_fixed35_six_contract import fixture
from sircus.mjlab.tests.test_calibrated_admission import proof

HOME=dict(q10=[0.]*6,q14=[0.]*6)


def branch_fixture(directory,slot=0,count=2):
    behavior=dict(version='sircus-continuous-bucket-facing-release-v1')
    common=dict(behavior_key='behavior',release_behavior=behavior,bucket_xy_m=[.1,-.65],values=[.6,-.1,.4,28,.24],
        resolution_file='resolution.json',resolution_sha256='r'*64,home_sha256='h'*64)
    key=V.digest(dict(version=V.BRANCH_VERSION,branch_rule=V.BRANCH_RULE,behavior_key=common['behavior_key'],resolution_sha256=common['resolution_sha256']))
    candidates=[dict(q10=[i*.2]*6,q14=[0.]*6,endpoint={'reason':0}) for i in range(count)]
    paths=[proof(np.zeros(12),np.asarray(c['q10']+c['q14'])) for c in candidates]
    admitted=[dict(slot=i,candidate_index=i,path_attempt_index=i,**copy.deepcopy(c)) for i,c in enumerate(candidates)]
    pool=dict(version=V.BRANCH_VERSION,pool_key=key,branch_rule=copy.deepcopy(V.BRANCH_RULE),
        selection_uses_flight_labels=False,**common,endpoint_attempts={'candidates':candidates},path_attempts=paths,admitted=admitted)
    name='branch-pools/'+key+'.json';path=directory/name;path.parent.mkdir(exist_ok=True);path.write_text(json.dumps(pool))
    branch_key=V.digest(dict(version=V.BRANCH_VERSION,pool_key=key,branch_pool_sha256=V.sha(path),ik_branch_slot=slot))
    connection=dict(**common,branch_pool_file=name,branch_pool_sha256=V.sha(path),branch_pool_key=key,
        branch_key=branch_key,ik_branch_slot=slot,branch_rule_version=V.BRANCH_VERSION,
        endpoint_attempts=copy.deepcopy(pool['endpoint_attempts']),path_attempts=copy.deepcopy(paths),passed=slot<count)
    if slot<count:connection.update(**copy.deepcopy(candidates[slot]),proof=copy.deepcopy(paths[slot]))
    row=dict(id='example',native_release=slot<count,connection_file='fixed-yaw-connections/'+branch_key+'.json',
        **{k:connection[k] for k in ('ik_branch_slot','branch_key','branch_pool_file','branch_pool_sha256','branch_pool_key','branch_rule_version')})
    return row,connection,pool,path


def endpoint_fixture():
    q=np.zeros((66,6));q[1,0]=.2
    provenance=[dict(seed=s,start_index=i) for s in (20261110,20261111) for i in range(33)]
    pools=[dict(q=q.tolist(),valid=[True]*66,residual=[0.]*66,provenance=copy.deepcopy(provenance)) for _ in range(2)]
    checks=dict(reason=0)
    return dict(branch_rule=copy.deepcopy(V.BRANCH_RULE),targets={},seeds=[20261110,20261111],ik_pools=pools,valid_ik_branches=[66,66],
        pair_attempts=[dict(branch10=0,branch14=0,seed_start10=provenance[0],seed_start14=provenance[0],checked=checks)],
        candidates=[dict(branch10=0,branch14=0,q10=q[0].tolist(),q14=q[0].tolist(),endpoint=checks)])


class ProtocolTests(unittest.TestCase):
    def test_protocol_is_literal_and_cannot_add_training_or_ui(self):
        V.protocol_checks(C.protocol(),Checks())
        for key,value in [('substeps',128),('maximum_requests',852),('preserved_recipes',24),('ui_changes',True),('training',True),('branch_slots',[0]),('noise',True)]:
            p=C.protocol();p[key]=value
            with self.subTest(key=key),self.assertRaises(EvidenceError):V.protocol_checks(p,Checks())

    def test_source_cohort_keeps29_and_rejects_changed_confirmation(self):
        d=fixture()
        for row in d['original_rows']+d['recovery_rows']:
            for key in ('q10','q14'):row['native_initial_'+key]=np.asarray(row[key],np.float32).astype(float).tolist()
        old={r['id']:r for r in d['original_rows']};current={r['id']:r for r in d['recovery_rows']}
        for row in d['recovery_rows']:
            if row['phase']=='preserved':row['parent_row_sha256']=V.digest(old[row['parent_row_id']])
            if row['phase']=='repeat':row['selected_from_sha256']=V.digest(current[row['selected_from_id']])
        for case in d['recovery_report']['cases']:
            if case['confirmed_recipe']:case['confirmed_recipe']=copy.deepcopy(current[case['confirmed_recipe']['id']])
        while len(d['recovery_rows'])<441:d['recovery_rows'].append(dict(id='filler'+str(len(d['recovery_rows']))))
        original=dict(rows=d['original_rows']);parent=dict(rows=d['recovery_rows'],report=d['recovery_report'])
        data=V.cohort(original,parent,Checks())
        self.assertEqual([len(data[k]) for k in ('preserved','cases','blocked')],[29,4,2])
        for mode in ('miss','joint','missing'):
            changed=copy.deepcopy(parent);row=next(r for r in changed['rows'] if r.get('phase')=='repeat')
            if mode=='miss':row['route_success']=False
            elif mode=='joint':row['q10'][0]=9
            else:changed['rows'].remove(row)
            with self.subTest(mode=mode),self.assertRaises(EvidenceError):V.cohort(original,changed,Checks())


class BranchProofTests(unittest.TestCase):
    def test_two_slots_bind_same_complete_pool_but_distinct_proofs(self):
        for slot in (0,1):
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);r,c,_,_=branch_fixture(root,slot)
                V.audit_branch_pool(root,r,c,{},Checks(),'example',home=HOME)

    def test_missing_branch_cannot_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);r,c,_,_=branch_fixture(root,1,1)
            V.audit_branch_pool(root,r,c,{},Checks(),'example',home=HOME)
            r['native_release']=True
            with self.assertRaises(EvidenceError):V.audit_branch_pool(root,r,c,{},Checks(),'example',home=HOME)

    def test_changed_slot_branch_path_or_pool_hash_rejected(self):
        for mode in ('slot','joints','path','hash','pool_contents'):
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);r,c,p,path=branch_fixture(root)
                if mode=='slot':r['ik_branch_slot']=1
                elif mode=='joints':c['q10'][0]=9
                elif mode=='path':c['proof']['waypoints'][-1][0]=9
                elif mode=='hash':c['branch_pool_sha256']='changed'
                else:p['selection_uses_flight_labels']=True;path.write_text(json.dumps(p))
                with self.subTest(mode=mode),self.assertRaises(EvidenceError):V.audit_branch_pool(root,r,c,{},Checks(),'example',home=HOME)

    def test_periodic_or_small_variations_do_not_qualify_as_two_distinct_branches(self):
        for delta in (0.,.01,2*np.pi):
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);r,c,p,path=branch_fixture(root)
                p['admitted'][1]['q10']=[delta]*6
                # Update byte hashes/keys honestly: semantic distinctness must still reject.
                path.write_text(json.dumps(p));c['branch_pool_sha256']=r['branch_pool_sha256']=V.sha(path)
                key=V.digest(dict(version=V.BRANCH_VERSION,pool_key=p['pool_key'],branch_pool_sha256=V.sha(path),ik_branch_slot=0))
                c['branch_key']=r['branch_key']=key;r['connection_file']='fixed-yaw-connections/'+key+'.json'
                with self.subTest(delta=delta),self.assertRaises(EvidenceError):V.audit_branch_pool(root,r,c,{},Checks(),'example',home=HOME)

    def test_even_unselected_second_slot_needs_source_home_and_strict_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);r,c,p,path=branch_fixture(root,slot=0)
            p['path_attempts'][1]['edges'][0]['minimum_clearance_m']=0.
            path.write_text(json.dumps(p));c['branch_pool_sha256']=r['branch_pool_sha256']=V.sha(path)
            c['path_attempts']=copy.deepcopy(p['path_attempts'])
            key=V.digest(dict(version=V.BRANCH_VERSION,pool_key=p['pool_key'],branch_pool_sha256=V.sha(path),ik_branch_slot=0))
            c['branch_key']=r['branch_key']=key;r['connection_file']='fixed-yaw-connections/'+key+'.json'
            with self.assertRaises(ValueError):V.audit_branch_pool(root,r,c,{},Checks(),'example',home=HOME)

    def test_complete_ik_pool_and_candidate_seed_binding(self):
        e=endpoint_fixture();V.audit_branch_endpoint(e,Checks(),'example')
        for mode in ('seed','joint','pool','clone','rule'):
            p=copy.deepcopy(e)
            if mode=='seed':p['pair_attempts'][0]['seed_start10']['seed']=1
            elif mode=='joint':p['candidates'][0]['q10'][0]=1
            elif mode=='pool':p['ik_pools'][0]['q'].pop()
            elif mode=='clone':p['pair_attempts'].append(copy.deepcopy(p['pair_attempts'][0]))
            else:p['branch_rule']['normal_only']=True
            with self.subTest(mode=mode),self.assertRaises(EvidenceError):V.audit_branch_endpoint(p,Checks(),'example')


if __name__=='__main__':unittest.main()
