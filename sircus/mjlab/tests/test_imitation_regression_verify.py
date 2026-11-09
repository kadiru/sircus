"""Independent six-case matched midpoint plan and provenance tamper tests; CPU only."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import re
import tempfile
import unittest
from sircus.mjlab import imitation_regression_verify as V


def fixtures():
    sources=[];controls=[]
    for i,context in enumerate((*V.ROLES,*V.CASES)):
        for method in (('control',) if i<3 else ('current_bc','corrected_bc')):
            recovery=method=='corrected_bc'
            row={key:None for key in V.META}
            row.update(id=f'original-{context}-{method}',context_id=context,role=context if i<3 else 'test',
                phase='controls_pre' if i<3 else 'test',method=method,repeat=0,
                values=[.31,-.11,.37,22.,.28] if recovery else [.3,-.1,.35,20.,.25],
                bucket_xy_m=[.95,-.1],native_release=not (recovery and context in V.IK_CASES),route_success=(i<2 or method=='current_bc'),
                invalid='finite_ik_unresolved' if recovery and context in V.IK_CASES else None,invalid_physics=False,
                q10=[0.]*6 if recovery and context in V.IK_CASES else [.1]*6,q14=[0.]*6 if recovery and context in V.IK_CASES else [.2]*6,
                connection_file=f'connections/{i}-{method}.json',connection_sha256=f'{i}-{method}')
            if row['native_release']:row.update(board_center_m=[.3,-.1,.35],board_normal=[0.,0.,1.],pose_gate={})
            (controls if i<3 else sources).append(row)
    rule=dict(mode=V.MODE,parameter_indices=list(range(5)),interpolation_alpha=.5,contexts=list(V.CASES),
        description='One fixed joint midpoint toward the preserved successful current recipe.',diagnosis_sha256=V.DIAGNOSIS_SHA256)
    return sources,controls,rule


def planned():
    # Only the test harness imports the producer; the auditor is independent.
    from sircus.mjlab import imitation_regression_native as N
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
    def test_exact72_requests_and_independent_midpoint_algebra(self):
        sources,controls,rule,plan=planned()
        V.audit_plan(plan,sources,controls,rule,V.replay.AuditChecks())
        self.assertEqual(len(plan),72)
        self.assertEqual([r['phase'] for r in plan],['controls_pre']*9+['diagnostic']*54+['controls_post']*9)
        lookup={r['id']:r for r in sources+controls}
        for row in plan:
            source=lookup[row['source_record_id']]
            differences=[i for i,(a,b) in enumerate(zip(row['values'],source['values'])) if a!=b]
            self.assertEqual(differences,row['parameter_indices'])
            if row['variant']=='blend50':self.assertEqual(row['values'],[a+.5*(b-a) for a,b in zip(source['values'],lookup[row['donor_record_id']]['values'])])
            if row['phase']=='diagnostic':self.assertEqual(lookup[row['realization_reference_record_id']]['method'],'current_bc')

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
            elif mode=='alpha':r['interpolation_alpha']=.25
            else:r['parameter_indices']=[0]
            with self.subTest(mode=mode),self.assertRaises(V.EvidenceError):V.audit_plan(bad,sources,controls,rule,V.replay.AuditChecks())

    def test_changed_rules_and_historical_claims_fail(self):
        sources,controls,rule,plan=planned()
        for mode in ('indices','alpha','cases','diagnosis','description','released','capture_label','reason'):
            badrule=copy.deepcopy(rule);badsources=copy.deepcopy(sources)
            if mode=='indices':badrule['parameter_indices']=[2]
            elif mode=='alpha':badrule['interpolation_alpha']=.3
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
        self.assertFalse(any('imitation_regression_native' in n or 'imitation_regression_study' in n or 'table_assisted_score.' in n or 'recovery_regression_diagnosis' in n for n in imports))
        self.assertNotIn('make_plan',[n.id for n in ast.walk(tree) if isinstance(n,ast.Name)])

    def test_gpu_identity_and_literal_parent_provenance(self):
        from sircus.mjlab.gpu_slot import GPU_UUID
        text=Path(V.__file__).read_text()
        self.assertEqual(set(re.findall(r'GPU-[0-9a-f-]+',text)),{GPU_UUID})
        self.assertEqual(V.PARENT_REPORT,'1fde3f7de5b5c2912a133c858105388a704a80ab0e2cb39e8d79c0c7a189bf14')
        self.assertEqual(V.PARENT_AUDIT,'ea4992828178d9e62ba41c63579ecc3876cb5bffaacf6b590c828d45876a7f63')
        self.assertEqual(V.PARENT_ROWS,'f9d48dd328e203b76d7f6be41c01e436ea938701821e902bbbd7945df7248843')
        self.assertEqual(V.PARENT_MANIFEST,'e65451629a36e3c486c3c7dbedd5f2de67de97fac46f6d4631eb3c38c68f3957')


class ObservationTests(unittest.TestCase):
    def test_refusals_retained_and_realizations_reference_released_parent(self):
        sources,controls,plan,rows=observed()
        V.audit_observations(rows,plan,sources,controls,V.replay.AuditChecks())
        refused=[r for r in rows if r['variant']=='corrected_exact' and not r['native_release']]
        self.assertEqual(len(refused),9)
        self.assertTrue(all(r['realization_comparison'] is None for r in refused))
        for row in rows:
            if row['phase']=='diagnostic' and row['native_release']:
                self.assertEqual(row['realization_comparison']['basis'],'matched_successful_current')
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
        self.assertEqual(cases[0]['variants']['blend50']['attempts'],3)
        self.assertEqual(cases[0]['variants']['blend50']['setup_rejections'],3)


class CrossSchemaTests(unittest.TestCase):
    def test_producer_case_summary_matches_independent_all_outcome_counts(self):
        from sircus.mjlab import imitation_regression_study as S
        sources,controls,plan,rows=observed()
        self.assertEqual(S.case_results(plan,rows),V.independent_cases(plan,rows))
        self.assertEqual(S.summarize(rows),V.statistics(rows))
        for variant in V.VARIANTS:
            group=[r for r in rows if r['variant']==variant]
            self.assertEqual(len(group),18)
            self.assertEqual(S.summarize(group),V.statistics(group))
        self.assertEqual(sum(r['admission_mode']=='cached_parent_refusal' for r in rows),9)

    def test_literal_protocol_uses_only_one_joint_midpoint(self):
        from sircus.mjlab.imitation_regression_contract import protocol
        p=protocol()
        for key,value in dict(substeps=64,maximum_requests=72,repeats=3,diagnostic_requests=54,control_requests=18,
                maximum_acquired_s=3600,maximum_queue_s=900,maximum_pipeline_s=6300,audit_reserve_s=1800,
                training=False,noise=False,posthoc_development=True,variants=list(V.VARIANTS),contexts=list(V.CASES),
                mode=V.MODE,parameter_indices=list(range(5)),interpolation_alpha=.5).items():
            self.assertEqual(p[key],value)

    def test_fixed72_request_batches_boundaries_and_budget(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/'study';root.mkdir();sources,controls,plan,rows=observed()
            def write(path,value):
                path.parent.mkdir(exist_ok=True,parents=True);path.write_text(json.dumps(value))
            write(root/'plan.json',plan);times=[]
            chunks=[(0,9,.5,1.5),(9,41,3.,4.),(41,63,5.,6.),(63,72,9.,10.)]
            for index,(start,end,begin,finish) in enumerate(chunks):
                name=f'requests/batch-{index:05d}.json';phase=plan[start]['phase']
                write(root/name,dict(phase=phase,rows=plan[start:end],pre_outcome=True))
                for row in rows[start:end]:row.update(request_packet=name,request_packet_sha256=V.sha(root/name))
                times.append(dict(phase=phase,request_packet=name,request_packet_sha256=V.sha(root/name),acquired_start_s=begin,acquired_end_s=finish))
            raw=[(json.dumps(r)+'\n').encode() for r in rows];(root/'rows.jsonl').write_bytes(b''.join(raw))
            (root/'timing.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in times))
            boundaries=[]
            for phase,count,t in [('plan-frozen',0,0.),('controls_pre-complete',9,2.),('diagnostic-complete',63,8.),('controls_post-complete',72,10.)]:
                boundaries.append(dict(phase=phase,rows_count=count,acquired_elapsed_s=t,plan_sha256=V.sha(root/'plan.json'),
                    rows_sha256=hashlib.sha256(b''.join(raw[:count])).hexdigest() if count else None))
            (root/'phase-boundaries.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in boundaries))
            budget=dict(acquired_unix_s=100.,limit_s=3600.,effective_worker_limit_s=3600.,global_deadline_unix_s=6390.,audit_reserve_s=1800,
                gpu_uuid='GPU-ab1950b0-01dd-0fab-407f-2d17936b27f4')
            write(root.with_name('study-budget.json'),budget);write(root.with_suffix('.pipeline.json'),dict(started_unix_s=90.,global_deadline_unix_s=6390.,global_limit_s=6300.))
            report=dict(elapsed_s=11.,deadlines=dict(worker_limit_s=3600.,effective_worker_limit_s=3600.,global_deadline_unix_s=6390.))
            packets,_=V.audit_timing(root,rows,report,V.replay.AuditChecks());self.assertEqual(packets['batches'],4)
            times[1]['acquired_end_s']=3600.
            (root/'timing.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in times))
            with self.assertRaises(V.EvidenceError):V.audit_timing(root,rows,report,V.replay.AuditChecks())


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
