"""Independent intervention and evidence-tamper checks; no native/GPU work."""
import ast
import copy
from pathlib import Path
import unittest
from sircus.mjlab import regression_drop_gap_verify as V


def fixtures():
    sources=[];controls=[]
    for i,context in enumerate((*V.ROLES,*V.CASES)):
        for method in (('control',) if i<3 else ('parent_bc','recovery_bc')):
            row={key:None for key in V.META}
            row.update(id=f'original-{context}-{method}',context_id=context,role=context if i<3 else 'test',
                phase='controls_pre' if i<3 else 'test',method=method,repeat=0,
                values=[.3,-.1,.35 if method!='recovery_bc' else .37,20.,.25 if method!='recovery_bc' else .28],
                bucket_xy_m=[.95,-.1],native_release=True,route_success=(i<2 or method=='parent_bc'),
                invalid=None,invalid_physics=False,q10=[.1]*6,q14=[.2]*6,
                board_center_m=[.3,-.1,.35],board_normal=[0.,0.,1.],pose_gate={},
                connection_file=f'connection/{i}-{method}.json',connection_sha256=f'{i}-{method}')
            (controls if i<3 else sources).append(row)
    rule=dict(parameter='drop_gap_m',parameter_index=4,mode='restore_matched_parent_coordinate',contexts=list(V.CASES),
        description='Restore only original-recovery gap; the board height remains its recovery value.',
        diagnosis_sha256='511e9f82004af0d603628e25adf8fab71284faa8f11de9307b4c8a611d049ab7')
    return sources,controls,rule


def planned():
    # The producer is used only to create a candidate for the independent checker.
    from sircus.mjlab import regression_drop_gap_native as N
    sources,controls,rule=fixtures()
    return sources,controls,rule,N.make_plan(sources,controls,rule)


def observed():
    sources,controls,rule,plan=planned();lookup={r['id']:r for r in sources+controls};rows=[]
    for request in plan:
        row=copy.deepcopy(request);source=lookup[row['source_record_id']]
        for key in ('connection_file','connection_sha256','native_release','q10','q14','invalid','board_center_m','board_normal','pose_gate'):
            row[key]=copy.deepcopy(source[key])
        if row['variant']=='corrected':
            row['connection_sha256']='new-'+row['context_id'];row['q10'][0]+=.03
        row['realization_comparison']=V.realized_comparison(row,source,V.replay.AuditChecks())
        row.update(scoring_version=V.oracle.VERSION,route_version=V.oracle.ROUTE_VERSION,reward_version=V.oracle.REWARD_VERSION,closest_opening_distance_m=.1,board_bounce=True,forbidden_contact_steps=0,max_retention_s=0.,route_success=False,route_quality=0,single_board_bounce_success=False)
        rows.append(row)
    return sources,controls,plan,rows


class IndependentPlanTests(unittest.TestCase):
    def test_exact36_requests_and_one_original_recovery_coordinate(self):
        sources,controls,rule,plan=planned()
        V.audit_plan(plan,sources,controls,rule,V.replay.AuditChecks())
        self.assertEqual(len(plan),36)
        self.assertEqual([r['phase'] for r in plan],['controls_pre']*9+['diagnostic']*18+['controls_post']*9)
        lookup={r['id']:r for r in sources+controls}
        for row in plan:
            source=lookup[row['source_record_id']]
            differences=[i for i,(a,b) in enumerate(zip(row['values'],source['values'])) if a!=b]
            self.assertEqual(differences,[4] if row['variant']=='corrected' else [])
        self.assertTrue(all(r['values'][2]==.37 for r in plan if r['variant']=='corrected'))

    def test_detects_combined_height_gap_and_other_plan_changes(self):
        sources,controls,rule,plan=planned()
        for mode in ('height','gap','bucket','namespace','donor','missing','duplicate','wrong_case','substeps','learning'):
            bad=copy.deepcopy(plan);r=bad[15]
            if mode=='height':r['values'][2]=.35;r['command_values'][2]=.35
            elif mode=='gap':r['values'][4]+=.001
            elif mode=='bucket':r['bucket_xy_m'][0]+=.1
            elif mode=='namespace':r['adapter_directory']='native-controls'
            elif mode=='donor':r['donor_record_sha256']='incorrect'
            elif mode=='missing':bad.pop()
            elif mode=='duplicate':bad[1]=bad[0]
            elif mode=='wrong_case':r['context_id']='recovery-test-r3-c1-n0'
            elif mode=='substeps':r['substeps']=128
            else:r['selection_or_learning']=True
            with self.subTest(mode=mode),self.assertRaises(V.EvidenceError):
                V.audit_plan(bad,sources,controls,rule,V.replay.AuditChecks())

    def test_rule_and_source_changes_rejected(self):
        sources,controls,rule,plan=planned()
        for mode in ('index','parameter','cases','diagnosis','description','cached_refusal','capture_label'):
            badrule=copy.deepcopy(rule);badsources=copy.deepcopy(sources)
            if mode=='index':badrule['parameter_index']=2
            elif mode=='parameter':badrule['parameter']='board_height_m'
            elif mode=='cases':badrule['contexts'].reverse()
            elif mode=='diagnosis':badrule['diagnosis_sha256']='bad'
            elif mode=='description':badrule['description']=''
            elif mode=='cached_refusal':badsources[1]['native_release']=False
            else:badsources[1]['route_success']=True
            with self.subTest(mode=mode),self.assertRaises(V.EvidenceError):
                V.audit_plan(plan,badsources,controls,badrule,V.replay.AuditChecks())

    def test_checker_does_not_import_producer_plan_or_scorer(self):
        tree=ast.parse(Path(V.__file__).read_text())
        imports=[n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        self.assertFalse(any('regression_drop_gap_native' in n or 'regression_drop_gap_study' in n or 'table_assisted_score.' in n for n in imports))
        names=[n.id for n in ast.walk(tree) if isinstance(n,ast.Name)]
        self.assertNotIn('make_plan',names)


class ObservationTests(unittest.TestCase):
    def test_exact_original_branches_fresh_correction_and_realization(self):
        sources,controls,plan,rows=observed()
        V.audit_observations(rows,plan,sources,controls,V.replay.AuditChecks())
        values=rows[15]['realization_comparison']
        self.assertTrue(values['q14_equal_to_source'])
        self.assertFalse(values['q10_equal_to_source'])
        self.assertAlmostEqual(values['q10_max_abs_delta_rad'],.03)

    def test_missing_duplicate_or_forged_evidence_fails(self):
        sources,controls,plan,rows=observed()
        for mode in ('missing','duplicate','old_branch','proof_reuse','repeat_proof','realization','forged_board'):
            bad=copy.deepcopy(rows)
            if mode=='missing':bad.pop()
            elif mode=='duplicate':bad[1]=bad[0]
            elif mode=='old_branch':bad[9]['q14'][0]+=.1
            elif mode=='proof_reuse':bad[15]['connection_sha256']=sources[1]['connection_sha256']
            elif mode=='repeat_proof':bad[15]['connection_sha256']='other-fresh-proof'
            elif mode=='realization':bad[15]['realization_comparison']['q14_equal_to_source']=False
            else:bad[15]['board_center_m'][0]+=.005
            with self.subTest(mode=mode),self.assertRaises(V.EvidenceError):
                V.audit_observations(bad,plan,sources,controls,V.replay.AuditChecks())

    def test_new_board_branch_is_reported_not_concealed_or_forced(self):
        sources,controls,plan,rows=observed();source=sources[1]
        for row in rows[15:18]:
            row['q14'][1]+=.02;row['board_center_m'][0]+=.0002
            row['board_normal']=[.001,0.,1.]
            row['realization_comparison']=V.realized_comparison(row,source,V.replay.AuditChecks())
        V.audit_observations(rows,plan,sources,controls,V.replay.AuditChecks())
        metric=rows[15]['realization_comparison']
        self.assertFalse(metric['q14_equal_to_source'])
        self.assertAlmostEqual(metric['board_center_delta_m'][0],.0002)
        self.assertGreater(metric['board_normal_angle_deg'],0)

    def test_corrected_declines_stay_in_denominator_without_pose_measurement(self):
        sources,controls,plan,rows=observed()
        for row in rows[15:18]:
            row.update(native_release=False,q10=None,q14=None,invalid='finite_ik_unresolved',realization_comparison=None)
        V.audit_observations(rows,plan,sources,controls,V.replay.AuditChecks())
        cases=V.independent_cases(plan,rows)
        self.assertEqual(cases[0]['variants']['corrected']['attempts'],3)
        self.assertEqual(cases[0]['variants']['corrected']['setup_rejections'],3)
        self.assertTrue(all(r['realization_comparison'] is None for r in cases[0]['outcomes'] if r['variant']=='corrected'))


if __name__=='__main__':unittest.main()
