"""Independent two-case feasibility plan and provenance tamper tests; CPU only."""
import ast
import copy
import json
from pathlib import Path
import re
import tempfile
import unittest
from sircus.mjlab import regression_ik_verify as V


def fixtures():
    sources=[];controls=[]
    for i,context in enumerate((*V.ROLES,*V.CASES)):
        for method in (('control',) if i<3 else ('parent_bc','recovery_bc')):
            recovery=method=='recovery_bc'
            row={key:None for key in V.META}
            row.update(id=f'original-{context}-{method}',context_id=context,role=context if i<3 else 'test',
                phase='controls_pre' if i<3 else 'test',method=method,repeat=0,
                values=[.31,-.11,.37,22.,.28] if recovery else [.3,-.1,.35,20.,.25],
                bucket_xy_m=[.95,-.1],native_release=not recovery,route_success=(i<2 or method=='parent_bc'),
                invalid='finite_ik_unresolved' if recovery else None,invalid_physics=False,
                q10=[0.]*6 if recovery else [.1]*6,q14=[0.]*6 if recovery else [.2]*6,
                connection_file=f'connections/{i}-{method}.json',connection_sha256=f'{i}-{method}')
            if not recovery:row.update(board_center_m=[.3,-.1,.35],board_normal=[0.,0.,1.],pose_gate={})
            (controls if i<3 else sources).append(row)
    rule=dict(mode=V.MODE,coordinate_groups={k:list(v) for k,v in V.GROUPS.items()},blend_fractions=V.FRACTIONS.copy(),
        contexts=list(V.CASES),description='Predeclared blocks and full-action continuation toward captured parent.',
        diagnosis_sha256='511e9f82004af0d603628e25adf8fab71284faa8f11de9307b4c8a611d049ab7')
    return sources,controls,rule


def planned():
    # Only the test harness imports the producer; the auditor is independent.
    from sircus.mjlab import regression_ik_native as N
    sources,controls,rule=fixtures()
    return sources,controls,rule,N.make_plan(sources,controls,rule)


def observed():
    sources,controls,rule,plan=planned();lookup={r['id']:r for r in sources+controls};rows=[]
    for request in plan:
        row=copy.deepcopy(request);source=lookup[row['source_record_id']]
        for key in ('connection_file','connection_sha256','native_release','q10','q14','invalid','board_center_m','board_normal','pose_gate'):
            row[key]=copy.deepcopy(source.get(key))
        if row['variant'] in V.VARIANTS[2:]:
            row.update(connection_sha256='new-'+row['context_id']+'-'+row['variant'],native_release=True,
                q10=[.13]*6,q14=[.22]*6,board_center_m=row['values'][:3],board_normal=[.001,0.,1.],invalid=None)
        reference=lookup[row['realization_reference_record_id']]
        row['realization_comparison']=V.realized_comparison(row,reference,V.replay.AuditChecks())
        row.update(scoring_version=V.oracle.VERSION,route_version=V.oracle.ROUTE_VERSION,reward_version=V.oracle.REWARD_VERSION,
            closest_opening_distance_m=.1,board_bounce=True,forbidden_contact_steps=0,max_retention_s=0.,route_success=False,
            route_quality=0,single_board_bounce_success=False)
        rows.append(row)
    return sources,controls,plan,rows


class IndependentPlanTests(unittest.TestCase):
    def test_exact66_requests_and_independent_block_interpolation_algebra(self):
        sources,controls,rule,plan=planned()
        V.audit_plan(plan,sources,controls,rule,V.replay.AuditChecks())
        self.assertEqual(len(plan),66)
        self.assertEqual([r['phase'] for r in plan],['controls_pre']*9+['diagnostic']*48+['controls_post']*9)
        lookup={r['id']:r for r in sources+controls}
        for row in plan:
            source=lookup[row['source_record_id']]
            differences=[i for i,(a,b) in enumerate(zip(row['values'],source['values'])) if a!=b]
            self.assertEqual(differences,row['parameter_indices'])
            if row['variant']=='blend50':self.assertEqual(row['values'],[(a+b)/2 for a,b in zip(source['values'],lookup[row['donor_record_id']]['values'])])
            if row['phase']=='diagnostic':self.assertEqual(lookup[row['realization_reference_record_id']]['method'],'parent_bc')

    def test_omitted_modified_or_replaced_requests_fail(self):
        sources,controls,rule,plan=planned()
        for mode in ('coordinate','bucket','namespace','donor','missing','duplicate','wrong_case','substeps','learning','reference','alpha','indices'):
            bad=copy.deepcopy(plan);r=bad[15]
            if mode=='coordinate':r['values'][2]+=.001;r['command_values'][2]+=.001
            elif mode=='bucket':r['bucket_xy_m'][0]+=.1
            elif mode=='namespace':r['adapter_directory']='native-controls'
            elif mode=='donor':r['donor_record_sha256']='incorrect'
            elif mode=='missing':bad.pop()
            elif mode=='duplicate':bad[1]=bad[0]
            elif mode=='wrong_case':r['context_id']='recovery-test-r3-c1-n0'
            elif mode=='substeps':r['substeps']=128
            elif mode=='learning':r['selection_or_learning']=True
            elif mode=='reference':r['realization_reference_record_id']=sources[1]['id']
            elif mode=='alpha':r['interpolation_alpha']=.5
            else:r['parameter_indices']=[0]
            with self.subTest(mode=mode),self.assertRaises(V.EvidenceError):V.audit_plan(bad,sources,controls,rule,V.replay.AuditChecks())

    def test_changed_rules_and_historical_claims_fail(self):
        sources,controls,rule,plan=planned()
        for mode in ('blocks','fractions','cases','diagnosis','description','released','capture_label','reason'):
            badrule=copy.deepcopy(rule);badsources=copy.deepcopy(sources)
            if mode=='blocks':badrule['coordinate_groups']['parent_height_gap']=[2]
            elif mode=='fractions':badrule['blend_fractions']['blend25']=.3
            elif mode=='cases':badrule['contexts'].reverse()
            elif mode=='diagnosis':badrule['diagnosis_sha256']='bad'
            elif mode=='description':badrule['description']=''
            elif mode=='released':badsources[1]['native_release']=True
            elif mode=='capture_label':badsources[1]['route_success']=True
            else:badsources[1]['invalid']='collision'
            with self.subTest(mode=mode),self.assertRaises(V.EvidenceError):V.audit_plan(plan,badsources,controls,badrule,V.replay.AuditChecks())

    def test_checker_does_not_import_producer_plan_or_scorer(self):
        tree=ast.parse(Path(V.__file__).read_text())
        imports=[n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        self.assertFalse(any('regression_ik_native' in n or 'regression_ik_study' in n or 'table_assisted_score.' in n or 'recovery_regression_diagnosis' in n for n in imports))
        self.assertNotIn('make_plan',[n.id for n in ast.walk(tree) if isinstance(n,ast.Name)])

    def test_gpu_identity_and_ancestry_hashes_are_unchanged(self):
        from sircus.mjlab.gpu_slot import GPU_UUID
        from sircus.mjlab import regression_drop_gap_verify as old
        text=Path(V.__file__).read_text();oldtext=Path(old.__file__).read_text()
        uuids=set(re.findall(r'GPU-[0-9a-f-]+',text))
        self.assertEqual(uuids,{GPU_UUID})
        self.assertEqual({V.PARENT_REPORT,V.PARENT_AUDIT,V.PARENT_FP},{old.PARENT_REPORT,old.PARENT_AUDIT,old.PARENT_FP})
        self.assertEqual(set(re.findall(r'\b[0-9a-f]{64}\b',text)),set(re.findall(r'\b[0-9a-f]{64}\b',oldtext)))


class ObservationTests(unittest.TestCase):
    def test_refusals_retained_and_realizations_reference_released_parent(self):
        sources,controls,plan,rows=observed()
        V.audit_observations(rows,plan,sources,controls,V.replay.AuditChecks())
        refused=[r for r in rows if r['variant']=='recovery_exact']
        self.assertEqual(len(refused),6)
        self.assertTrue(all(r['realization_comparison'] is None for r in refused))
        for row in rows:
            if row['phase']=='diagnostic' and row['native_release']:
                self.assertEqual(row['realization_comparison']['basis'],'matched_successful_parent')
                self.assertEqual(row['realization_comparison']['source_record_id'],row['realization_reference_record_id'])

    def test_missing_duplicate_or_forged_evidence_fails(self):
        sources,controls,plan,rows=observed()
        for mode in ('missing','duplicate','old_branch','proof_reuse','repeat_proof','realization','forged_board','refusal_flight','missing_reference','reference_basis'):
            bad=copy.deepcopy(rows)
            if mode=='missing':bad.pop()
            elif mode=='duplicate':bad[1]=bad[0]
            elif mode=='old_branch':bad[9]['q14'][0]+=.1
            elif mode=='proof_reuse':bad[15]['connection_sha256']=sources[1]['connection_sha256']
            elif mode=='repeat_proof':bad[15]['connection_sha256']='other-fresh-proof'
            elif mode=='realization':bad[15]['realization_comparison']['q14_equal_to_source']=True
            elif mode=='forged_board':bad[15]['board_center_m'][0]+=.005
            elif mode=='refusal_flight':bad[12]['native_cache_file']='fake.npz'
            elif mode=='missing_reference':bad[15]['realization_reference_record_sha256']='forged'
            else:bad[15]['realization_comparison']['basis']='unflown_recovery'
            with self.subTest(mode=mode),self.assertRaises(V.EvidenceError):V.audit_observations(bad,plan,sources,controls,V.replay.AuditChecks())

    def test_new_branch_and_board_change_reported_without_causal_shortcut(self):
        sources,controls,plan,rows=observed();reference=sources[0]
        for row in rows[15:18]:
            row['q14'][1]+=.02;row['board_center_m'][0]+=.0002
            row['board_normal']=[.002,0.,1.]
            row['realization_comparison']=V.realized_comparison(row,reference,V.replay.AuditChecks())
        V.audit_observations(rows,plan,sources,controls,V.replay.AuditChecks())
        self.assertFalse(rows[15]['realization_comparison']['q14_equal_to_source'])
        self.assertGreater(rows[15]['realization_comparison']['board_normal_angle_deg'],0)

    def test_new_declines_stay_in_denominator_without_fabricated_pose(self):
        sources,controls,plan,rows=observed()
        for row in rows[15:18]:row.update(native_release=False,q10=None,q14=None,invalid='finite_ik_unresolved',realization_comparison=None)
        V.audit_observations(rows,plan,sources,controls,V.replay.AuditChecks())
        cases=V.independent_cases(plan,rows)
        self.assertEqual(cases[0]['variants']['parent_xy']['attempts'],3)
        self.assertEqual(cases[0]['variants']['parent_xy']['setup_rejections'],3)


class SetupSummaryTests(unittest.TestCase):
    def test_independent_rejection_summary_and_tamper(self):
        # Deliberately small failed proof: no clearance or flight is invented.
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);native=root/'native';native.mkdir()
            resolution=dict(home_geometry={},layout_eligible=True,anchor_admitted=True)
            connection=dict(passed=False,reason='finite_ik_unresolved',action_substituted=False,
                endpoint_attempts=dict(valid_ik_branches=[0,8],candidates=[],pair_attempts=[]),proof=None)
            (native/'connection.json').write_text(json.dumps(connection));(native/'resolution.json').write_text(json.dumps(resolution))
            row=dict(id='declined',adapter_directory='native',connection_file='connection.json',context_resolution_file='resolution.json',
                connection_sha256=V.sha(native/'connection.json'),context_resolution_sha256=V.sha(native/'resolution.json'),native_release=False,q10=[0.]*6,q14=[0.]*6)
            row['setup_proof']=dict(reason='finite_ik_unresolved',group='finite_ik_dropper',passed=False,
                connection_sha256=row['connection_sha256'],resolution_sha256=row['context_resolution_sha256'],action_substituted=False,
                valid_ik_branches=[0,8],endpoint_examples=[],endpoint_pair_attempts=0,admitted_branches=[],layout_eligible=True,anchor_admitted=True,
                proof_minimum_clearance_m=None,home_minimum_clearance_m=None,home_closest_pair=None,fixed_base_blockers=[],required_clearance_m=.005,
                scope='Recorded finite admission proof; rejection does not prove unreachability, and a conservative zero bound does not prove penetration.')
            V.audit_setup_summaries(root,[row],V.replay.AuditChecks())
            for mode in ('invented_clearance','forged_branch','changed_bytes','wrong_group'):
                bad=copy.deepcopy(row)
                if mode=='invented_clearance':bad['setup_proof']['proof_minimum_clearance_m']=0.
                elif mode=='forged_branch':bad['setup_proof']['valid_ik_branches']=[1,8]
                elif mode=='changed_bytes':bad['connection_sha256']='bad'
                else:bad['setup_proof']['group']='unreachable'
                with self.subTest(mode=mode),self.assertRaises(V.EvidenceError):V.audit_setup_summaries(root,[bad],V.replay.AuditChecks())


if __name__=='__main__':unittest.main()
