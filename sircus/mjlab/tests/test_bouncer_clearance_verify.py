import copy
import unittest
from pathlib import Path
import numpy as np
from sircus.mjlab import bouncer_clearance_verify as a


def fixture():
    sources=[];plan=[]
    for ci,(context,identifier) in enumerate(zip(a.CASES,a.SOURCE_IDS)):
        source=dict(id=identifier,context_id=context,phase='discovery',method='grid',native_release=True,
                    route_success=False,invalid_physics=False,diagnostic_robot_contact_steps=3,
                    values=[.6,-.14,.46,28.,.25],bucket_xy_m=[1.1,-.2])
        source.update({key:None for key in a.META});sources.append(source)
        for xi,dx in enumerate((-.12,-.06,0.,.06,.12)):
            for yi,dy in enumerate((-.12,-.06,0.,.06,.12)):
                local=xi*5+yi;values=[.6+dx,-.14+dy,.46,28.,.25]
                row={key:copy.deepcopy(source[key]) for key in a.META}
                row.update(id=f'bouncer-clearance-{context}-g{local:02d}',context_id=context,grid_index=ci*25+local,case_grid_index=local,
                    phase='discovery',variant='grid',method='grid',role='diagnostic',repeat=0,selection_or_learning=False,posthoc_development=True,
                    values=values,command_values=copy.deepcopy(values),bucket_xy_m=copy.deepcopy(source['bucket_xy_m']),
                    source_record_id=source['id'],source_record_sha256=a.digest(source),substeps=64,perturbation_mixed_units=[0.]*5)
                plan.append(row)
    return sources,plan


def discovery(index,quality=0,reward=0.,**overrides):
    row=dict(id=str(index),context_id=a.CASES[0],phase='discovery',native_release=True,route_success=False,
        board_bounce=True,invalid_physics=False,invalid=None,diagnostic_robot_contact_steps=0,
        route_quality=quality,learning_reward=reward,grid_index=index)
    row.update(overrides);return row


class IndependentBouncerClearanceTests(unittest.TestCase):
    def test_literal75_grid_preserves_bucket_height_tilt_gap(self):
        sources,plan=fixture();a.audit_grid(plan,sources,a.replay.AuditChecks())
        self.assertEqual(len(plan),75)
        self.assertEqual(plan[12]['values'],sources[0]['values'])
        self.assertEqual([r['values'][2:] for r in plan],[[.46,28.,.25]]*75)
        self.assertEqual(a.budget()['maximum_requested'],75+9+18+9)

    def test_altered_parameters_or_source_or_missing_proposal_rejected(self):
        sources,plan=fixture()
        for kind in ('gap','bucket','height','tilt','xy','index','source','missing'):
            bad=copy.deepcopy(plan)
            if kind=='gap':bad[15]['values'][4]+=.001
            if kind=='bucket':bad[15]['bucket_xy_m'][0]+=.001
            if kind=='height':bad[15]['values'][2]=.52
            if kind=='tilt':bad[15]['values'][3]=29
            if kind=='xy':bad[15]['values'][0]+=.001
            if kind=='index':bad[15]['grid_index']=16
            if kind=='source':bad[15]['source_record_sha256']='0'*64
            if kind=='missing':bad.pop()
            with self.subTest(kind=kind),self.assertRaises(a.EvidenceError):a.audit_grid(bad,sources,a.replay.AuditChecks())

    def test_source_must_be_exact_recorded_contact_miss(self):
        sources,plan=fixture()
        for key,value in [('id','different'),('context_id','different'),('native_release',False),
                          ('invalid_physics',True),('route_success',True),('diagnostic_robot_contact_steps',0)]:
            bad=copy.deepcopy(sources);bad[0][key]=value
            with self.subTest(key=key),self.assertRaises(a.EvidenceError):a.audit_grid(plan,bad,a.replay.AuditChecks())

    def test_selection_requires_full_terminal_robot_clearance_bounce_and_validity(self):
        good=discovery(10,reward=-.2)
        rows=[good,discovery(1,reward=20,diagnostic_robot_contact_steps=1),
              discovery(2,reward=20,board_bounce=False),discovery(3,reward=20,invalid_physics=True),
              discovery(4,reward=20,invalid='bad'),discovery(5,reward=20,native_release=False),
              discovery(6,reward=20,diagnostic_robot_contact_steps=None)]
        result=a.selected_clear(rows)
        self.assertEqual(result[a.CASES[0]],good)
        self.assertTrue(all(result[c] is None for c in a.CASES[1:]))
        self.assertFalse(a.v3.captured(good))
        self.assertIsNone(a.selected_clear(rows[1:])[a.CASES[0]])

    def test_selection_capture_then_quality_reward_index(self):
        rows=[discovery(1,2,100),discovery(7,1,10,route_success=True),
              discovery(6,2,3,route_success=True),discovery(4,2,3,route_success=True)]
        for order in (rows,list(reversed(rows))):
            self.assertEqual(a.selected_clear(order)[a.CASES[0]]['id'],'4')

    def test_repeat_clear_miss_without_relabeling_capture(self):
        _,plan=fixture();row=dict(plan[0],native_release=True,route_success=False,board_bounce=True,
            invalid_physics=False,invalid=None,diagnostic_robot_contact_steps=0,route_quality=0,learning_reward=-.2,
            connection_file='source',connection_sha256='a'*64,q10=[0.]*6,q14=[1.]*6)
        selected=a.expected_selection([row]);repeats=a.repeat_requests(plan,selected)
        self.assertFalse(selected[0]['selected']['capture'])
        self.assertEqual(len(repeats),3)
        self.assertEqual([r['repeat'] for r in repeats],[0,1,2])
        self.assertTrue(all(r['selected_from_sha256']==a.digest(row) for r in repeats))
        self.assertTrue(all(r['values']==row['values'] for r in repeats))
        self.assertEqual(a.repeat_requests(plan,a.expected_selection([])),[])

    def test_unattempted_matrix_does_not_count_as_a_rejection(self):
        _,plan=fixture()
        cells=[dict(candidate_id=p['id'],context_id=p['context_id'],grid_index=p['grid_index'],case_grid_index=p['case_grid_index'],
                    values=p['values'],bucket_xy_m=p['bucket_xy_m'],resolution_file=p['context_resolution_file'],resolution_sha256=p['context_resolution_sha256'],
                    status='not_attempted',passed=None,valid_ik_branches=None,connection_file=None,connection_sha256=None,q10=None,q14=None) for p in plan]
        self.assertEqual(a.audit_matrix(Path('/unused'),plan,cells,{}, {},a.replay.AuditChecks()),
                         dict(total=75,attempted=0,admitted=0,rejected=0,not_attempted=75))
        cells[0]['passed']=False
        with self.assertRaises(a.EvidenceError):a.audit_matrix(Path('/unused'),plan,cells,{}, {},a.replay.AuditChecks())

    def test_native_baseline_and_grid_keep_exact_proof_and_joint_branch(self):
        sources,plan=fixture();source=sources[0]
        source.update(q10=[0.]*6,q14=[1.]*6,connection_file='original',connection_sha256='a'*64)
        expected=a.request(source,'baseline','baseline','baseline',0,source['values'])
        observed=dict(expected,native_release=True,q10=source['q10'],q14=source['q14'],connection_file='original',connection_sha256='a'*64)
        a.audit_native_requests([observed],[expected],sources,[],a.replay.AuditChecks())
        for key,value in [('q14',[1.01]*6),('connection_sha256','b'*64),('native_release',False)]:
            changed=copy.deepcopy(observed);changed[key]=value
            with self.subTest(key=key),self.assertRaises(a.EvidenceError):
                a.audit_native_requests([changed],[expected],sources,[],a.replay.AuditChecks())
        grid=dict(plan[0],native_release=True,q10=[2.]*6,q14=[3.]*6,connection_file='fresh',connection_sha256='c'*64)
        matrix=[dict(candidate_id=grid['id'],**{key:grid[key] for key in ('q10','q14','connection_file','connection_sha256')})]
        a.audit_native_requests([grid],[plan[0]],sources,matrix,a.replay.AuditChecks())
        grid['q14']=[4.]*6
        with self.assertRaises(a.EvidenceError):a.audit_native_requests([grid],[plan[0]],sources,matrix,a.replay.AuditChecks())

    def test_zero_offset_cannot_change_to_another_valid_joint_branch(self):
        sources,_=fixture();source=sources[0]
        source.update(q10=[0.]*6,q14=[1.]*6,connection_file='original',connection_sha256='a'*64)
        cell=dict(candidate_id='zero',context_id=source['context_id'],case_grid_index=12,status='completed',
            passed=True,values=source['values'],**{key:source[key] for key in ('q10','q14','connection_file','connection_sha256')})
        a.audit_zero_offsets([cell],sources,a.replay.AuditChecks())
        for key,value in [('q14',[2.]*6),('connection_sha256','b'*64),('passed',False)]:
            changed=copy.deepcopy(cell);changed[key]=value
            with self.subTest(key=key),self.assertRaises(a.EvidenceError):
                a.audit_zero_offsets([changed],sources,a.replay.AuditChecks())

    def test_contact_timing_counts_late_hits_but_not_after_scoring_terminal(self):
        t=np.arange(11)*.1;f=np.zeros((10,6));f[1,0]=1;f[3,4]=1;f[4,2]=1;f[6,4]=1;f[8,4]=1
        f[0,4]=.01  # Threshold is strictly greater than .01 N.
        row=discovery(0,physics_steps=8)
        d=a.contact_diagnostic(t,f,row)
        self.assertEqual(d['robot_contact_steps'],2)
        self.assertEqual(d['robot_contact_steps_before_first_bucket_or_table'],1)
        self.assertEqual(d['robot_contact_steps_from_first_bucket_or_table'],1)
        self.assertAlmostEqual(d['first_board_contact_s'],.1)
        self.assertAlmostEqual(d['first_robot_contact_s'],.3)
        self.assertAlmostEqual(d['first_table_contact_s'],.4)
        self.assertFalse(d['capture'])
        self.assertIsNone(d['first_bucket_contact_s'])
        f[4,2]=0
        d=a.contact_diagnostic(t,f,row)
        self.assertEqual(d['robot_contact_steps_before_first_bucket_or_table'],2)
        self.assertEqual(d['robot_contact_steps_from_first_bucket_or_table'],0)
        self.assertIsNone(d['first_bucket_or_table_contact_s'])

    def test_contact_diagnostic_rejects_incomplete_trace(self):
        row=discovery(0,physics_steps=8)
        for t,f in [(np.arange(8),np.zeros((8,6))), (np.arange(11),np.zeros((7,6))),
                    (np.arange(11),np.zeros((10,5))), (np.arange(11),np.full((10,6),np.nan))]:
            with self.subTest(shape=f.shape),self.assertRaises(a.EvidenceError):a.contact_diagnostic(t,f,row)


if __name__=='__main__':unittest.main()
