"""CPU-only positive and adversarial evidence checks for recovery publication."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from sircus.mjlab import coverage_recovery_verify as A
from sircus.mjlab import coverage_recovery_contract as C
from sircus.mjlab.tests.test_coverage_training_verify import outcome
from sircus.mjlab.benchmark_verify import Checks,EvidenceError


def source(ident='c',released=True):
    r=outcome(ident,capture=False)
    r.update(context_id=ident,phase='test',role='test',method='selected_ppo',native_release=released,
        normalized_residual=[.15,-.25,.35,-.45,.55],bucket_xy_m=[.6,-.3],
        source_hint=dict(id='anchor',values=[.3,-.2,.46,28.,.2518]),effective_source_hint=dict(id='anchor',values=[.3,-.2,.46,28.,.2518]),
        observation=[.1]*9,source_teacher_id='anchor',context_resolution_file='context.json',context_resolution_sha256='c'*64,
        context_home_sha256='h'*64,context_layout_eligible=True,context_anchor_admitted=True,context_hard_reject=None)
    return r


def fixture(root,count=39):
    cases=[source('c'+str(i),i%2==0) for i in range(count)];plan=C.make_plan(cases,cases)
    search=[]
    for req in plan:
        row=outcome(req['id'],capture=req['context_id']=='c0' and req['proposal_index']==0)
        row.update(req,search_plan_sha256='later',q10=[.1]*6,q14=[.2]*6,connection_file='connection.json',connection_sha256='k'*64)
        row['learning_reward']=A.oracle.reward(row);search.append(row)
    selected,repeats=C.repeat_plan(search,cases)
    observed=[]
    for req in repeats:
        r=outcome(req['id']);r.update(req,q10=[.1]*6,q14=[.2]*6,connection_file='connection.json',connection_sha256='k'*64,repeat_plan_sha256='later')
        r['learning_reward']=A.oracle.reward(r);observed.append(r)
    def write(name,value):
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value))
    write('search-plan.json',dict(rows=plan,frozen_before_outcomes=True))
    for r in search:r['search_plan_sha256']=A.sha(root/'search-plan.json')
    # Repeat source digest includes the final immutable search-plan hash.
    selected,repeats=C.repeat_plan(search,cases)
    for req,r in zip(repeats,observed):r.update(req)
    write('repeat-plan.json',dict(selected=selected,rows=repeats,frozen_before_outcomes=True))
    for r in observed:r['repeat_plan_sha256']=A.sha(root/'repeat-plan.json')
    confirmed=C.confirmed_teachers(search,observed,selected)
    original=[dict(outcome('original'),context_id='original')];union=sorted(original+confirmed,key=lambda r:r['context_id'])
    write('bootstrap/teachers.json',union);write('bootstrap/recovery-teachers.json',confirmed)
    write('bootstrap/source-bindings.json',dict(parent_teacher_ids=['original'],parent_teacher_sha256={'original':A.digest(original[0])},
        new_teacher_ids=[r['id'] for r in confirmed],teacher_file_sha256=A.sha(root/'bootstrap/teachers.json'),repeat_plan_sha256=A.sha(root/'repeat-plan.json')))
    report=dict(selected=selected,confirmed_teachers=len(confirmed),search_coverage=[dict(context_id=c['context_id'],bucket_xy_m=c['bucket_xy_m'],
        counts=A.statistics([r for r in search if r['context_id']==c['context_id']]),confirmed=c['context_id']=='c0') for c in cases])
    return search+observed,cases,original,report,dict(plans=dict(search_sha256=A.sha(root/'search-plan.json')))


class RecoveryAudit(unittest.TestCase):
    def test_protocol_budget_guards(self):
        A.protocol_checks(C.protocol(),Checks())
        for key,value in [('substeps',128),('noise',True),('maximum_requests',1001),('actor_steps',801),('critic_steps',1),('test_layouts',69)]:
            p=C.protocol();p[key]=value
            with self.assertRaises(EvidenceError):A.protocol_checks(p,Checks())

    def test_independent_all16_setup_and_flight_proposals(self):
        for released in (False,True):
            for alternate in (False,True):
                row=source(released=released);other=copy.deepcopy(row);other.update(outcome('other'));other.update(context_id=row['context_id'])
                choices=[other] if alternate else []
                for index in range(16):
                    ra,ma=A.independent_proposal(row,index,choices);rb,mb=C.proposal(row,index,choices)
                    np.testing.assert_array_equal(ra,rb);self.assertEqual(ma,mb)
        for index in (-1,16,1.1):
            with self.assertRaises(EvidenceError):A.independent_proposal(source(),index,[])

    def test_alternate_capture_cannot_change_anchor(self):
        row=source();other=copy.deepcopy(row);other.update(outcome('other'));other['context_id']=row['context_id']
        other['effective_source_hint']['values'][0]+=.001
        with self.assertRaises(EvidenceError):A.independent_proposal(row,1,[other])

    def test_full624_plan_order_and_no_implicit_clipping(self):
        cases=[source('c'+str(i),i%2==0) for i in range(39)]
        self.assertEqual(A.independent_search_plan(cases,cases),C.make_plan(cases,cases))
        self.assertEqual([r['context_id'] for r in A.independent_search_plan(cases,cases)[:39]],[r['context_id'] for r in cases])
        cases[0]['normalized_residual'][0]=1.
        with self.assertRaises(EvidenceError):A.independent_search_plan(cases,cases)

    def test_confirmation_requires_two_unique_successes_same_branch(self):
        s=dict(outcome('discovery'),q10=[1],q14=[2],connection_file='a',connection_sha256='b',values=[0]*5,bucket_xy_m=[.3,-.2])
        selected=[dict(discovery_id='discovery')]
        repeats=[dict(s,id='repeat'+str(i),source_discovery_id='discovery',repeat=i) for i in range(2)]
        self.assertEqual(A.confirmed_teachers([s],repeats,selected,Checks()),[s])
        changed=copy.deepcopy(repeats);changed[1]['route_success']=False
        self.assertEqual(A.confirmed_teachers([s],changed,selected,Checks()),[])
        for change in ({'repeat':0},{'q10':[3]},{'connection_sha256':'other'}):
            changed=copy.deepcopy(repeats);changed[1].update(change)
            with self.assertRaises(EvidenceError):A.confirmed_teachers([s],changed,selected,Checks())
        with self.assertRaises(EvidenceError):A.confirmed_teachers([s],repeats[:1],selected,Checks())

    def test_search_and_teacher_end_to_end_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rows,cases,old,report,spec=fixture(root)
            actual=A.audit_search_and_teachers(root,rows,cases,cases,old,report,spec,Checks())
            self.assertEqual((len(actual[0]),len(actual[1]),actual[2]),(2,1,2))
            changed=copy.deepcopy(rows);changed[0]['values'][4]+=.001
            with self.assertRaises(EvidenceError):A.audit_search_and_teachers(root,changed,cases,cases,old,report,spec,Checks())
            changed=copy.deepcopy(rows);changed[0]['role']='test'
            with self.assertRaises(EvidenceError):A.audit_search_and_teachers(root,changed,cases,cases,old,report,spec,Checks())
            changed=copy.deepcopy(rows);changed[-1]['route_success']=False
            with self.assertRaises(EvidenceError):A.audit_search_and_teachers(root,changed,cases,cases,old,report,spec,Checks())
            path=root/'bootstrap/teachers.json';teachers=json.loads(path.read_text());teachers[0]['role']='test';path.write_text(json.dumps(teachers))
            with self.assertRaises(EvidenceError):A.audit_search_and_teachers(root,rows,cases,cases,old,report,spec,Checks())

    def test_fresh70_seed_and_gap_preserved(self):
        ex=[[.6,-.3],[.7,-.2]];actual=A.regenerated_test(ex)
        self.assertEqual(actual,C.test_split(ex));self.assertEqual(len(actual),70)
        seen=np.array(ex)
        for r in actual:
            self.assertGreaterEqual(np.linalg.norm(seen-r['bucket_xy_m'],axis=1).min(),.005);seen=np.vstack((seen,r['bucket_xy_m']))
        for bad in ([],[[float('nan'),0]],[[1,2,3]]):
            with self.assertRaises(EvidenceError):A.regenerated_test(bad)

    def test_inventory_mutation_and_filtered_split_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);history=root/'history';repo=root/'repo';study=root/'study'
            history.mkdir();(repo/'baselines').mkdir(parents=True);study.mkdir()
            (history/'exclusions.json').write_text(json.dumps([[.6,-.3],[.7,-.2]]))
            inventory=C.collect_exclusions(history,repo);split=dict(development=[],test=C.test_split(inventory['coordinates']))
            (study/'historical-coordinate-bindings.json').write_text(json.dumps(inventory));(study/'splits.json').write_text(json.dumps(split))
            A.audit_inventory(study,repo,history,dict(splits=split),[],Checks())
            changed=copy.deepcopy(split);changed['test'].pop();(study/'splits.json').write_text(json.dumps(changed))
            with self.assertRaises(EvidenceError):A.audit_inventory(study,repo,history,dict(splits=split),[],Checks())
            (study/'splits.json').write_text(json.dumps(split));(history/'exclusions.json').write_text('[]')
            with self.assertRaises(EvidenceError):A.audit_inventory(study,repo,history,dict(splits=split),[],Checks())

    def test_actor_record_algebra_and_independent_normalized_tolerance(self):
        pred=dict(source(),latent_mean=[.01,.02,.03,.04,.05]);anchor=np.asarray(pred['effective_source_hint']['values'])
        pred['normalized_residual']=np.tanh(pred['latent_mean']).tolist();pred['command_values']=(anchor+np.tanh(pred['latent_mean'])*A.RESIDUAL).tolist()
        latent=np.array(pred['latent_mean']);latent[3]+=2.9e-7
        with patch.object(A.common,'network_forward',return_value=latent):
            A.audit_actor_transform(pred,{'policy':{}},Checks(),'test')
            changed=copy.deepcopy(pred);changed['command_values'][3]+=1e-7
            with self.assertRaises(EvidenceError):A.audit_actor_transform(changed,{'policy':{}},Checks(),'test')
            changed=copy.deepcopy(pred);changed['latent_mean'][3]+=.001;changed['normalized_residual']=np.tanh(changed['latent_mean']).tolist();changed['command_values']=(anchor+np.tanh(changed['latent_mean'])*A.RESIDUAL).tolist()
            with self.assertRaises(EvidenceError):A.audit_actor_transform(changed,{'policy':{}},Checks(),'test')

    @unittest.skipUnless(importlib.util.find_spec('torch'),'Pinned CPU Torch required')
    def test_actor_only_optimizer_replay_and_tamper(self):
        import torch
        from sircus.mjlab.policy64_model import bootstrap
        from sircus.mjlab.benchmark_policy import ResidualPolicy
        torch.set_num_threads(2);torch.manual_seed(15);model=ResidualPolicy();before=copy.deepcopy(model.state_dict())
        rows=[dict(outcome(str(i)),observation=[.01*i]*9,normalized_residual=[.03*i]*5) for i in range(3)]
        fit=bootstrap(model,rows,rows,actor_steps=800,critic_steps=0)
        A.actor_finetune_replay(before,model.state_dict(),rows,fit,Checks())
        changed=copy.deepcopy(model.state_dict());changed['critic.0.bias'][0]+=.0001
        with self.assertRaises(EvidenceError):A.actor_finetune_replay(before,changed,rows,fit,Checks())
        changed=copy.deepcopy(rows);changed[0]['role']='test'
        with self.assertRaises(EvidenceError):A.actor_finetune_replay(before,model.state_dict(),changed,fit,Checks())

if __name__=='__main__':unittest.main()
