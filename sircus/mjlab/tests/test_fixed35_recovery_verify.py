"""Independent recovery protocol and tamper checks; no GPU or native launch."""
import copy
import hashlib
import json
import unittest
from sircus.mjlab import fixed35_recovery_verify as A
from sircus.mjlab import fixed35_recovery_contract as C
from sircus.mjlab.benchmark_verify import Checks,EvidenceError
from sircus.mjlab.tests.test_fixed35_imitation_verify import inference_fixture
from sircus.mjlab.tests.test_coverage_training_verify import outcome
from pathlib import Path
import tempfile


def fixture():
    """Canonical parent cohort with explicit known failed grid cells."""
    failures={f'fixed35-dev-grid-{y:02d}-{x:02d}' for y,x in A.FAILURE_CELLS}
    blocked={f'fixed35-dev-grid-{y:02d}-{x:02d}' for y,x in A.BLOCKED_CELLS}
    with tempfile.TemporaryDirectory() as tmp:
        rows,*_=inference_fixture(Path(tmp))
    for r in rows:
        capture=r['context_id'] not in failures
        r.update(route_success=capture,invalid=None,native_release=True,whole_ball_entry=capture,max_retention_s=1. if capture else 0.)
        if r['context_id'] in blocked:r.update(invalid='fixed_base_clearance_rejected',native_release=False)
        elif r['context_id']=='fixed35-dev-grid-04-06':r.update(invalid='finite_ik_unresolved',native_release=False)
        r.update(values=[.6,-.2,.32,28.,.2],q10=[0.]*6,q14=[0.]*6,connection_file='fixed-yaw-connections/abc.json',connection_sha256='a'*64)
    # Different, measured historical alternative exists at one failed location.
    for r in rows:
        if r['context_id']=='fixed35-dev-grid-01-06' and r['method']=='fixed35_bc':
            r.update(route_success=True,whole_ball_entry=True,max_retention_s=1.)
    return rows,A.partition_parent(rows,Checks())


def schedule_fixture(root,*,confirm=True,truncated=False):
    _,data=fixture()
    for r in data['refit']:r['values'][4]+=.01
    plan=A.independent_plan(data);rows=[];boundaries=[];timings=[];clock=0.
    (root/'repeat-plans').mkdir()
    def write(name,value):
        p=root/name;p.write_text(json.dumps(value));return p
    write('search-plan.json',dict(rows=plan,frozen_before_outcomes=True,maximum_requests=720))
    write('preserved-plan.json',dict(rows=A.independent_preserved(data),frozen_before_outcomes=True))
    def raw():return b''.join((json.dumps(r)+'\n').encode() for r in rows)
    def boundary(phase,**extra):
        nonlocal clock
        clock+=.1;boundaries.append(dict(phase=phase,rows_count=len(rows),rows_sha256=hashlib.sha256(raw()).hexdigest() if rows else None,acquired_elapsed_s=clock,**extra))
    def batch(phase,requests,capture=False):
        nonlocal clock
        start=clock+.1;clock+=.2;group=[]
        for r in requests:
            r=copy.deepcopy(r)
            if phase in ('search','preserved'):
                parent=next(p for p in data['current'] if p['context_id']==r['context_id'])
                r.update({k:parent.get(k) for k in ('q10','q14','connection_file','connection_sha256','native_release')})
                if phase=='search' and r['proposal_index']>0:r['native_release']=True
            else:r.update(q10=[0.]*6,q14=[0.]*6,connection_file='fixed-yaw-connections/abc.json',connection_sha256='a'*64,native_release=True)
            r.update(route_success=capture,invalid=None if r['native_release'] else 'finite_ik_unresolved',invalid_physics=False,native_cache_sha256=phase+str(len(timings)))
            if r['native_release']:
                r.update(native_initial_q10=r['q10'].copy(),native_initial_q14=r['q14'].copy())
            rows.append(r);group.append(r)
        timings.append(dict(phase=phase,acquired_start_s=start,acquired_end_s=clock));return group
    boundary('inputs-frozen');batch('controls_pre',[dict(id='pre'+str(i),phase='controls_pre') for i in range(9)]);boundary('controls_pre-complete')
    batch('preserved',A.independent_preserved(data),True);boundary('preserved-complete')
    stopped={}
    for index in range(2):
        boundary('search-round-frozen',proposal_index=index,contexts=[r['context_id'] for r in data['cases']])
        group=batch('search',[dict(p,search_plan_sha256=A.sha(root/'search-plan.json')) for p in plan if p['proposal_index']==index],capture=index==1)
        if index==1:
            repeat=[r for d in group for r in A.independent_repeat(d,0)];name=f'repeat-plans/round-{index:02d}.json'
            write(name,dict(proposal_index=index,rows=repeat,selected=[dict(id=d['id'],sha256=A.digest(d)) for d in group],rows_count_before=len(rows),rows_sha256_before=hashlib.sha256(raw()).hexdigest(),frozen_before_repeat_outcomes=True))
            boundary('repeat-plan-frozen',proposal_index=index,path=name,sha256=A.sha(root/name))
            batch('repeat',[dict(r,repeat_plan_file=name,repeat_plan_sha256=A.sha(root/name)) for r in repeat],confirm)
            if confirm:stopped={r['context_id']:'confirmed' for r in data['cases']}
        boundary('search-round-complete',proposal_index=index,stopped=stopped.copy())
        if truncated:break
    if truncated:clock=5401.
    reason='search_time_limit' if truncated else 'all_searchable_cases_confirmed'
    boundary('search-complete',completion_reason=reason,budget_truncated=truncated)
    batch('controls_post',[dict(id='post'+str(i),phase='controls_post') for i in range(9)]);boundary('controls_post-complete')
    (root/'rows.jsonl').write_bytes(raw())
    (root/'phase-boundaries.jsonl').write_text(''.join(json.dumps(b)+'\n' for b in boundaries))
    (root/'timing.jsonl').write_text(''.join(json.dumps(t)+'\n' for t in timings))
    return rows,dict(budget_truncated=truncated,completion_reason=reason),data,plan


class ParentTests(unittest.TestCase):
    def test_exact35_positions_and11failures(self):
        _,data=fixture()
        self.assertEqual([len(data[k]) for k in ('current','refit','preserved','failures','blocked','cases')],[35,35,24,11,2,9])

    def test_parent_case_cannot_jitter_hide_or_relabel_failure(self):
        for mode in ('jitter','drop','mislabel','capture','blocked'):
            rows,_=fixture();r=next(r for r in rows if r['method']=='current_bc' and r['context_id']=='fixed35-dev-grid-00-00')
            if mode=='jitter':r['bucket_xy_m'][0]+=.000001
            elif mode=='drop':rows.remove(r)
            elif mode=='mislabel':r['context_id']='another-case'
            elif mode=='capture':r['route_success']=True
            else:r.update(invalid='fixed_base_clearance_rejected',native_release=False)
            with self.subTest(mode=mode),self.assertRaises(EvidenceError):A.partition_parent(rows,Checks())

    def test_protocol_rejects_more_training_changed_grid_or_reduced_controls(self):
        A.protocol_checks(C.protocol(),Checks())
        for key,value in [('substeps',128),('grid_positions',70),('noise',True),('bucket_perturbations',True),('training',True),
                          ('actor_steps',1),('maximum_requests',844),('control_requests',0),('repeat_per_selected',2),('search_layouts',11)]:
            p=C.protocol();p[key]=value
            with self.subTest(key=key),self.assertRaises(EvidenceError):A.protocol_checks(p,Checks())


class PlannerTests(unittest.TestCase):
    def test_independent720plan_matches_producer(self):
        _,data=fixture();planned=A.independent_plan(data)
        self.assertEqual(planned,C.make_plan(data['cases'],data['refit'],data['preserved']))
        self.assertEqual((len(planned),len(set(r['id'] for r in planned))),(720,720))
        for i,r in enumerate(planned):
            self.assertEqual(r['proposal_index'],i//9)
            self.assertEqual(r['context_id'],data['cases'][i%9]['context_id'])
        self.assertEqual(A.independent_preserved(data),C.preserved_plan(data['preserved']))

    def test_literal_axes_and_clipping_preserve_evidence(self):
        _,data=fixture();r=data['cases'][0];r['values']=[.31,-.49,.13,9.,.055]
        p=A.independent_proposals(r,data['refit'],data['preserved'])
        self.assertEqual(p,C.proposals(r,data['refit'],data['preserved']))
        self.assertEqual(p[6]['raw_values'][-1],.035)
        self.assertEqual(p[6]['values'][-1],.05)
        self.assertTrue(p[6]['clipped'])
        self.assertEqual(p[68]['values'][2:],[.20,20.,.16])
        self.assertEqual(p[79]['values'][2:],[.44,32.,.32])

    def test_neighbor_recipe_keeps_fixed_bucket_and_historical_labels(self):
        _,data=fixture();plan=A.independent_plan(data)
        row=next(r for r in plan if r['proposal_index']==2)
        original=next(r for r in data['cases'] if r['context_id']==row['context_id'])
        teacher=next(r for r in data['preserved'] if r['id']==row['derivation']['parent_source_ids'][0])
        self.assertEqual(row['values'],teacher['values'])
        self.assertEqual(row['bucket_xy_m'],original['bucket_xy_m'])
        self.assertNotIn('route_success',row)
        self.assertFalse(row['selection_or_learning'])

    def test_fresh_repeats_keep_discovery_branch_and_original_parent(self):
        _,data=fixture();r=A.independent_plan(data)[0]
        r.update(route_success=True,native_release=True,invalid=None,invalid_physics=False,
            q10=[.1]*6,q14=[.2]*6,connection_sha256='a'*64)
        for group in range(3):
            repeat=A.independent_repeat(r,group)
            self.assertEqual(repeat,C.repeat_plan(r,group))
            self.assertEqual([s['repeat'] for s in repeat],[0,1,2])
            self.assertTrue(all(s['selected_from_sha256']==A.digest(r) for s in repeat))
            self.assertTrue(all(s['parent_row_id']==r['parent_row_id'] for s in repeat))
        r['route_success']=False
        with self.assertRaises(EvidenceError):A.independent_repeat(r,0)

    def test_three_fresh_successes_required_for_confirmation(self):
        _,data=fixture();parent=data['cases'][0];discovery=A.independent_plan(data)[0]
        discovery.update(route_success=True,native_release=True,invalid=None,invalid_physics=False,q10=[0.]*6,q14=[0.]*6,connection_sha256='a')
        repeats=A.independent_repeat(discovery,0)
        for r in repeats:r.update(route_success=True,native_release=True,invalid=None,invalid_physics=False)
        group=A.independent_groups(repeats,parent['context_id'])[0]
        self.assertEqual(group['status'],'confirmed')
        repeats[-1]['invalid_physics']=True
        self.assertEqual(A.independent_groups(repeats,parent['context_id'])[0]['status'],'failed')
        repeats.pop()
        self.assertEqual(A.independent_groups(repeats,parent['context_id'])[0]['status'],'incomplete')


class ScheduleTests(unittest.TestCase):
    def test_full_rounds_freeze_confirm_then_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rows,report,data,plan=schedule_fixture(root)
            accounting=A.audit_schedule(root,rows,report,data,plan,Checks())
            self.assertEqual((accounting['search_requests'],accounting['repeat_requests']),(18,27))
            self.assertEqual(len(accounting['stopped']),9)

    def test_legitimate_budget_cutoff_keeps_required_controls(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rows,report,data,plan=schedule_fixture(root,truncated=True)
            accounting=A.audit_schedule(root,rows,report,data,plan,Checks())
            self.assertEqual(accounting['total_requests'],51)
            self.assertTrue(accounting['budget_truncated'])

    def test_miss_cannot_be_relabelled_confirmed_or_frozen_late(self):
        for mode in ('miss','late','omitted','jitter','fake_cutoff'):
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);rows,report,data,plan=schedule_fixture(root,truncated=mode=='fake_cutoff')
                if mode=='miss':next(r for r in rows if r['phase']=='repeat')['route_success']=False
                elif mode=='late':
                    b=A.common.lines(root/'phase-boundaries.jsonl');next(x for x in b if x['phase']=='repeat-plan-frozen')['acquired_elapsed_s']=7000
                    (root/'phase-boundaries.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in b))
                elif mode=='omitted':rows.pop(34)
                elif mode=='jitter':next(r for r in rows if r['phase']=='search')['bucket_xy_m'][0]+=.001
                else:
                    b=A.common.lines(root/'phase-boundaries.jsonl');next(x for x in b if x['phase']=='search-complete')['acquired_elapsed_s']=5.
                    (root/'phase-boundaries.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in b))
                with self.subTest(mode=mode),self.assertRaises(EvidenceError):A.audit_schedule(root,rows,report,data,plan,Checks())


if __name__=='__main__':unittest.main()
