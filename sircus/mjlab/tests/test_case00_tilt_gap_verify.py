import copy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from sircus.mjlab import case00_tilt_gap_verify as a


def fixture():
    source=dict(id='bouncer-clearance-table-training-test-00-g14',context_id='table-training-test-00',
        phase='discovery',method='grid',native_release=True,route_success=False,invalid_physics=False,
        invalid=None,board_bounce=True,diagnostic_robot_contact_steps=0,
        values=[.5979922029820652,-.023557257454087432,.46,28.,.25181883224993984],
        bucket_xy_m=[1.1095395651982425,-.22146373203137287],
        q10=[0.]*6,q14=[1.]*6,connection_file='original',connection_sha256='a'*64)
    source.update({key:None for key in a.META})
    plan=[]
    for ti,tilt in enumerate((20.,22.,24.,26.,28.,30.,32.)):
        for gi,gap in enumerate((-.06,-.04,-.02,-.01,0.,.01,.02,.04,.06,.08)):
            index=ti*10+gi;values=source['values'][:3]+[tilt,source['values'][4]+gap]
            row={key:copy.deepcopy(source[key]) for key in a.META}
            row.update(id=f'case00-tilt-gap-table-training-test-00-g{index:02d}',context_id=source['context_id'],
                grid_index=index,case_grid_index=index,phase='discovery',variant='grid',method='grid',role='diagnostic',repeat=0,
                selection_or_learning=False,posthoc_development=True,values=values,command_values=copy.deepcopy(values),
                bucket_xy_m=copy.deepcopy(source['bucket_xy_m']),source_record_id=source['id'],source_record_sha256=a.digest(source),
                substeps=64,perturbation_mixed_units=[0.]*5)
            plan.append(row)
    return [source],plan


def observed(proposal,**updates):
    row=dict(proposal,native_release=True,route_success=True,invalid_physics=False,invalid=None,board_bounce=True,
        diagnostic_robot_contact_steps=0,route_quality=1,learning_reward=3.,connection_file='fresh',connection_sha256='b'*64,
        q10=[0.]*6,q14=[1.]*6,direct_capture=True,table_assisted_capture=False,forbidden_contact_steps=0,
        closest_opening_distance_m=.0,max_retention_s=1.,single_board_bounce_success=True,
        scoring_version=a.v3.oracle.VERSION,route_version=a.v3.oracle.ROUTE_VERSION,reward_version=a.v3.oracle.REWARD_VERSION)
    row.update(updates);row['learning_reward']=a.v3.oracle.reward(row)
    return row


def empty_matrix(plan):
    return [dict(candidate_id=p['id'],context_id=p['context_id'],grid_index=p['grid_index'],case_grid_index=p['case_grid_index'],
        values=p['values'],bucket_xy_m=p['bucket_xy_m'],resolution_file=p['context_resolution_file'],resolution_sha256=p['context_resolution_sha256'],
        status='not_attempted',passed=None,reason='pending',valid_ik_branches=None,connection_file=None,connection_sha256=None,q10=None,q14=None) for p in plan]


class IndependentCase00TiltGapTests(unittest.TestCase):
    def test_seventy_literal_proposals_and_original_at_g44(self):
        sources,plan=fixture();a.audit_grid(plan,sources,a.replay.AuditChecks())
        self.assertEqual(plan[44]['values'],sources[0]['values'])
        self.assertEqual(plan[0]['values'][4],.19181883224993984)
        self.assertEqual(plan[-1]['values'][4],.33181883224993985)
        self.assertEqual(a.budget()['maximum_requested'],70+3+18+70*3)

    def test_changed_xyz_bucket_tilt_gap_metadata_or_missing_row_rejected(self):
        sources,plan=fixture()
        for kind in ('x','y','height','tilt','gap','bucket','index','source','missing'):
            bad=copy.deepcopy(plan)
            if kind in ('x','y','height','tilt','gap'):bad[3]['values'][('x','y','height','tilt','gap').index(kind)]+=.001
            if kind=='bucket':bad[3]['bucket_xy_m'][0]+=.001
            if kind=='index':bad[3]['grid_index']=4
            if kind=='source':bad[3]['source_record_sha256']='0'*64
            if kind=='missing':bad.pop()
            with self.subTest(kind=kind),self.assertRaises(a.EvidenceError):a.audit_grid(bad,sources,a.replay.AuditChecks())

    def test_parent_must_be_exact_contact_free_miss(self):
        sources,plan=fixture()
        for key,value in [('id','changed'),('board_bounce',False),('route_success',True),('native_release',False),
                          ('invalid_physics',True),('invalid','bad'),('diagnostic_robot_contact_steps',1)]:
            bad=copy.deepcopy(sources);bad[0][key]=value
            with self.subTest(key=key),self.assertRaises(a.EvidenceError):a.audit_grid(plan,bad,a.replay.AuditChecks())

    def test_selects_every_capture_in_grid_order_not_only_best(self):
        _,plan=fixture()
        rows=[observed(plan[12],route_quality=2),observed(plan[5]),observed(plan[2]),
              observed(plan[7],route_success=False),observed(plan[8],invalid_physics=True),
              observed(plan[9],invalid='bad'),observed(plan[10],native_release=False),
              observed(plan[1],phase='selected_repeats')]
        chosen=a.selected_captures(rows)
        self.assertEqual([r['grid_index'] for r in chosen],[2,5,12])
        self.assertEqual([r['grid_index'] for r in a.selected_captures(list(reversed(rows)))],[2,5,12])
        self.assertEqual(a.expected_selection([]),[])

    def test_all_captures_receive_three_unique_matched_repeats_with_source_hash(self):
        _,plan=fixture();rows=[observed(plan[5]),observed(plan[12])]
        selected=a.expected_selection(rows);repeats=a.repeat_requests(plan,selected)
        self.assertEqual(len(repeats),6)
        self.assertEqual([r['id'] for r in repeats],
                         [f'case00-tilt-gap-selected-g{g:02d}-r{r}' for g in (5,12) for r in range(3)])
        for original in rows:
            group=[r for r in repeats if r['selected_from_id']==original['id']]
            self.assertEqual(len(group),3)
            self.assertTrue(all(r['selected_from_sha256']==a.digest(original) and r['values']==original['values'] for r in group))
        all_rows=[observed(p) for p in plan];all_repeats=a.repeat_requests(plan,a.expected_selection(all_rows))
        self.assertEqual(len(all_repeats),210)
        self.assertEqual(len({r['id'] for r in all_repeats}),210)

    def test_original_cell_preserves_original_exact_joint_branch(self):
        sources,plan=fixture();cell=empty_matrix(plan)[44];source=sources[0]
        cell.update(status='completed',passed=True,**{key:source[key] for key in ('q10','q14','connection_file','connection_sha256')})
        a.audit_zero_offsets([cell],sources,a.replay.AuditChecks())
        for key,value in [('q14',[2.]*6),('connection_sha256','c'*64),('passed',False)]:
            bad=copy.deepcopy(cell);bad[key]=value
            with self.subTest(key=key),self.assertRaises(a.EvidenceError):a.audit_zero_offsets([bad],sources,a.replay.AuditChecks())

    def test_unattempted_cells_remain_distinct_from_ik_declines(self):
        _,plan=fixture();matrix=empty_matrix(plan)
        self.assertEqual(a.audit_matrix(Path('/unused'),plan,matrix,{}, {},a.replay.AuditChecks()),
            dict(total=70,attempted=0,admitted=0,rejected=0,not_attempted=70))
        matrix[0]['passed']=False
        with self.assertRaises(a.EvidenceError):a.audit_matrix(Path('/unused'),plan,matrix,{}, {},a.replay.AuditChecks())

    def test_report_accounts_each_capture_and_repeat_group_separately(self):
        # Cross-check independently reconstructed counts against the producer;
        # neither the verifier nor its oracle imports producer planning/scoring.
        from sircus.mjlab import case00_tilt_gap_study as producer
        sources,plan=fixture();matrix=empty_matrix(plan)
        for cell in matrix:cell.update(status='completed',passed=False,reason='fixture_decline')
        discovery=[observed(plan[5]),observed(plan[12])]
        for index in (5,12):matrix[index].update(passed=True,reason='passed')
        selection=a.expected_selection(discovery)
        repeats=[observed(r,route_success=(r['repeat']!=1),direct_capture=(r['repeat']!=1))
                 for r in a.repeat_requests(plan,selection)]
        rows=discovery+repeats;spec=dict(parent_bindings={},admission_matrix_sha256='d'*64,fingerprint='e'*64)
        with tempfile.TemporaryDirectory() as name:
            path=Path(name);(path/'parent-rows.json').write_text(json.dumps(sources))
            with contextlib.redirect_stdout(io.StringIO()):
                report=producer.report_state(path,spec,matrix,rows,selection,status='fixture',phase='fixture',cpu_elapsed_s=1.)
            a.audit_report(report,spec,sources,matrix,rows,selection,a.replay.AuditChecks())
            self.assertEqual(report['requested_configurations'],76)
            self.assertEqual(report['admission_summary']['declined'],68)
            self.assertEqual([g['summary']['captures'] for g in report['cases'][0]['candidate_repeats']],[2,2])
            broken=copy.deepcopy(report);broken['cases'][0]['candidate_repeats'][0]['summary']['captures']=3
            with self.assertRaises(a.EvidenceError):a.audit_report(broken,spec,sources,matrix,rows,selection,a.replay.AuditChecks())

    def test_parent_provenance_constants_not_mutated_by_grid_counts(self):
        self.assertEqual(a.PARENT_REPORT,'a284d8e785a3d5aec8a45f92877377f63812c38767d6738fe1b6669a69b399a4')
        self.assertEqual(a.PARENT_AUDIT,'7de9a6222d65aca1f1ac884566459bb30b250348f6c0175abc78177965b5e175')


if __name__=='__main__':unittest.main()
