"""CPU-only grid, immutable proof replay, selection and interruption checks."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from sircus.mjlab import bouncer_clearance_contract as C
from sircus.mjlab import bouncer_clearance_native as N
from sircus.mjlab import bouncer_clearance_study as S
from sircus.mjlab import bouncer_clearance_pipeline as P
from sircus.mjlab.tests.test_table_correction_native import fixture as parent_fixture


def fixture():
    old=parent_fixture()[0];rows=old[:3]
    for index,(identifier,context) in enumerate(zip(C.SOURCE_IDS,C.CONTEXTS)):
        row=copy.deepcopy(next(r for r in old if r['context_id']==context))
        row.update(id=identifier,phase='discovery',route_success=False,diagnostic_robot_contact_steps=12,
            values=[.3,-.1,.48 if index==2 else .46,24. if index==2 else 28.,.25])
        rows.append(row)
    return rows,None


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.sources=C.sources_from_parent(fixture()[0]);self.plan=C.grid_plan(self.sources)

    def test_grid_has75_exact_distinct_requests_and_only_xy_changes(self):
        self.assertEqual(len(self.plan),75);self.assertEqual(len({r['id'] for r in self.plan}),75)
        for c,context in enumerate(C.CONTEXTS):
            rows=self.plan[c*25:(c+1)*25];original=next(s for s in self.sources if s['context_id']==context)
            self.assertEqual([r['values'][:2] for r in rows],[[original['values'][0]+dx,original['values'][1]+dy] for dx in C.OFFSETS for dy in C.OFFSETS])
            for row in rows:
                self.assertEqual([row['values'][i] for i in (2,3,4)],[original['values'][i] for i in (2,3,4)])
                self.assertEqual(row['bucket_xy_m'],original['bucket_xy_m'])
                self.assertEqual(row['source_record_sha256'],N.digest(original))
                self.assertEqual(row['substeps'],64)

    def test_fixed_baselines_and_controls_are_complete_and_original(self):
        phases=[C.fixed_requests(self.sources,p) for p in ('controls_pre','baseline','controls_post')]
        self.assertEqual(list(map(len,phases)),[9,9,9])
        self.assertEqual(len({r['id'] for phase in phases for r in phase}),27)
        lookup={r['id']:r for r in self.sources}
        self.assertTrue(all(r['values']==lookup[r['source_record_id']]['values'] for group in phases for r in group))

    def test_only_robot_contact_free_discovery_selected_with_frozen_tie_break(self):
        rows=[]
        for index,quality,reward in [(0,1,10),(1,2,5),(2,2,6),(3,2,6)]:
            row=copy.deepcopy(self.plan[index]);row.update(native_release=True,route_success=True,
                invalid=False,invalid_physics=False,board_bounce=True,diagnostic_robot_contact_steps=0,route_quality=quality,learning_reward=reward,
                connection_file=f'{index}.json',connection_sha256=str(index)*64,q10=[0]*6,q14=[1]*6)
            rows.append(row)
        invalid=copy.deepcopy(rows[0]);invalid.update(grid_index=-1,route_quality=3,invalid_physics=True);rows.append(invalid)
        previous=copy.deepcopy(rows[1]);previous.update(phase='baseline',route_quality=99);rows.append(previous)
        selected=C.select_clear(rows)
        self.assertEqual(selected[0]['selected']['candidate_id'],self.plan[2]['id'])
        self.assertTrue(all(r['selected'] is None for r in selected[1:]))
        repeats=C.selected_requests(self.plan,selected)
        self.assertEqual(len(repeats),3);self.assertTrue(all(r['values']==self.plan[2]['values'] for r in repeats))
        self.assertEqual(75+27+len(repeats),105)

    def test_contact_free_miss_is_repeatable_evidence_but_collision_is_not_selected(self):
        row=copy.deepcopy(self.plan[0]);row.update(native_release=True,route_success=False,
            invalid=False,invalid_physics=False,board_bounce=True,diagnostic_robot_contact_steps=0,
            route_quality=0,learning_reward=-1,connection_file='proof.json',connection_sha256='a'*64,q10=[0]*6,q14=[1]*6)
        colliding=copy.deepcopy(row);colliding.update(id=self.plan[1]['id'],grid_index=1,
            route_success=True,route_quality=2,learning_reward=100,diagnostic_robot_contact_steps=1)
        no_board=copy.deepcopy(row);no_board.update(id=self.plan[2]['id'],board_bounce=False,learning_reward=99)
        chosen=C.select_clear([row,colliding,no_board])
        self.assertEqual(chosen[0]['selected']['candidate_id'],row['id'])
        self.assertFalse(chosen[0]['selected']['capture'])
        self.assertEqual(len(C.selected_requests(self.plan,chosen)),3)

    def test_original_sources_cannot_be_replaced_with_successes_or_unrelated_rows(self):
        original=fixture()[0]
        for mutation in ('missing','duplicate','captured','contact_free','context','not_released'):
            rows=copy.deepcopy(original)
            if mutation=='missing':rows.pop()
            elif mutation=='duplicate':rows.append(rows[-1])
            elif mutation=='captured':rows[-1]['route_success']=True
            elif mutation=='contact_free':rows[-1]['diagnostic_robot_contact_steps']=0
            elif mutation=='context':rows[-1]['context_id']='unrelated'
            elif mutation=='not_released':rows[-1]['native_release']=False
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):C.sources_from_parent(rows)

    def test_no_contact_free_candidate_means_no_selected_requests(self):
        chosen=C.select_clear([])
        self.assertEqual(len(chosen),3);self.assertTrue(all(r['selected'] is None for r in chosen))
        self.assertEqual(C.selected_requests(self.plan,chosen),[])
        self.assertEqual(C.budget()['maximum_requested'],111)

    def test_empty_matrix_keeps_all_unattempted_cells_distinct_from_declines(self):
        matrix=C.empty_matrix(self.plan)
        self.assertEqual(len(matrix),75)
        self.assertTrue(all(r['status']=='not_attempted' and r['passed'] is None for r in matrix))


class NativeTests(unittest.TestCase):
    def fixture(self,variant='grid',mode='ok'):
        sources=C.sources_from_parent(fixture()[0]);plan=C.grid_plan(sources)
        engine=N.BouncerClearanceEngine.__new__(N.BouncerClearanceEngine)
        engine.sources={r['id']:r for r in sources};engine.plan={r['id']:r for r in plan}
        request=copy.deepcopy(plan[0] if variant=='grid' else C.fixed_requests(sources,'baseline')[0])
        matrix=dict(candidate_id=plan[0]['id'],passed=True,connection_file='proof.json',connection_sha256='matrix',q10=[.2]*6,q14=[.3]*6)
        engine.matrix={plan[0]['id']:matrix};engine.parent=SimpleNamespace(parent_options={'exact':1},parent_geometry={'geometry':2})
        env=SimpleNamespace(effective_options={'exact':1},compiled_geometry={'geometry':2})
        def evaluate(requests,phase):
            output=[]
            for row in requests:
                bound=engine.sources[row['source_record_id']] if row['variant']=='baseline' else matrix
                observed=copy.deepcopy(row);observed.update(native_release=True)
                for key in ('connection_file','connection_sha256','q10','q14'):observed[key]=copy.deepcopy(bound[key])
                for key in ('q10','q14'):observed['native_initial_'+key]=np.asarray(observed[key],dtype=np.float32).astype(float).tolist()
                if mode=='branch':observed['q14'][0]+=.1
                if mode=='native_initial':observed['native_initial_q10'][0]+=.1
                if mode=='proof':observed['connection_sha256']='different'
                if mode=='declined':observed['native_release']=False
                if mode=='physics':env.effective_options={'different':1}
                output.append(observed)
            return output,{}
        engine.engine=SimpleNamespace(adapter=SimpleNamespace(env=env),evaluate=evaluate)
        return engine,request

    def test_exact_original_and_matrix_branches_execute(self):
        for variant in ('baseline','grid'):
            engine,request=self.fixture(variant);rows,_=engine.evaluate([request],'test');self.assertEqual(len(rows),1)

    def test_unadmitted_or_modified_candidate_cannot_execute(self):
        for mode in ('bucket','xy','gap','height','not_admitted','source'):
            engine,request=self.fixture()
            if mode=='bucket':request['bucket_xy_m'][0]+=.01
            elif mode=='xy':request['values'][0]+=.01
            elif mode=='gap':request['values'][4]+=.01
            elif mode=='height':request['values'][2]=.99
            elif mode=='source':request['source_record_sha256']='tampered'
            else:engine.matrix[request['id']]['passed']=False
            with self.subTest(mode=mode),self.assertRaises(ValueError):engine.evaluate([request],'test')

    def test_changed_native_joints_proof_options_or_admission_fail_closed(self):
        for mode in ('branch','native_initial','proof','declined','physics'):
            engine,request=self.fixture(mode=mode)
            with self.subTest(mode=mode),self.assertRaises(RuntimeError):engine.evaluate([request],'test')

    def test_matrix_hydration_reads_proofs_and_does_not_solve(self):
        sources=C.sources_from_parent(fixture()[0]);plan=C.grid_plan(sources);matrix=C.empty_matrix(plan)
        with tempfile.TemporaryDirectory() as tmp:
            engine=N.BouncerClearanceEngine.__new__(N.BouncerClearanceEngine);engine.directory=Path(tmp);engine.plan={r['id']:r for r in plan}
            resolver=SimpleNamespace(home={},_connected={},resolve=lambda xy:dict(cache_hit=True,sha256='resolution'))
            engine.engine=SimpleNamespace(adapter=SimpleNamespace(resolver=resolver))
            for row,request in zip(matrix,plan):
                key=N.digest(dict(version=N.GEOMETRY_VERSION,connection_policy=N.CONNECTION_POLICY_VERSION,
                    resolution_sha256='resolution',values=request['values'],bucket_xy_m=request['bucket_xy_m']))
                path=engine.directory/(key+'.json');path.write_text(json.dumps(dict(passed=True,q10=[0]*6,q14=[1]*6)))
                row.update(status='completed',passed=True,resolution_sha256='resolution',connection_file=path.name,connection_sha256=N.sha(path))
            with patch.object(N,'verify_connection',return_value=True):engine.hydrate_matrix(matrix)
            self.assertEqual(len(engine.matrix),75);self.assertTrue(resolver._connected)
            (engine.directory/matrix[0]['connection_file']).write_text('{}')
            with patch.object(N,'verify_connection',return_value=True),self.assertRaises(ValueError):engine.hydrate_matrix(matrix)
            matrix[0]['status']='not_attempted'
            with self.assertRaises(ValueError):engine.hydrate_matrix(matrix)


class ReportingTests(unittest.TestCase):
    def test_partial_cpu_stop_preserves75_cells_and_freezes_matrix(self):
        sources=C.sources_from_parent(fixture()[0]);plan=C.grid_plan(sources);matrix=C.empty_matrix(plan)
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)
            (output/'parent-rows.json').write_text(json.dumps(sources));(output/'request-contract.json').write_text('{}')
            matrix[0].update(status='completed',passed=False,reason='finite_ik_unresolved')
            S.finish_admission(output,dict(parent_bindings={},fingerprint='previous'),plan,sources,matrix,3590.,complete=False,error='cap')
            saved=json.loads((output/'admission-matrix.json').read_text());report=json.loads((output/'report.json').read_text())
            self.assertEqual(len(saved),75);self.assertEqual(report['admission_summary']['declined'],1)
            self.assertEqual(report['admission_summary']['unattempted'],74)
            self.assertEqual(report['native_requested'],0);self.assertEqual(report['status'],'admission_stopped')
            self.assertTrue(all(r['discovery'] is None for c in report['cases'] for r in c['grid']))
            self.assertTrue((output/'contract.json').exists() and (output/'evidence-manifest.json').exists())
            self.assertEqual(report['admission_matrix_sha256'],N.sha(output/'admission-matrix.json'))

    def test_invalid_physics_never_displayed_as_capture(self):
        sources=C.sources_from_parent(fixture()[0]);plan=C.grid_plan(sources);matrix=C.empty_matrix(plan)
        row=copy.deepcopy(plan[0]);row.update(native_release=True,route_success=True,invalid_physics=True,
            route_quality=2,direct_capture=True,table_assisted_capture=False,diagnostic_robot_contact_steps=0,
            forbidden_contact_steps=0,learning_reward=-1,invalid=None)
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp);(output/'parent-rows.json').write_text(json.dumps(sources))
            spec=dict(parent_bindings={},admission_matrix_sha256='fixture-only')
            report=S.report_state(output,spec,matrix,[row],[],status='running',phase='discovery',cpu_elapsed_s=1.)
            self.assertFalse(report['cases'][0]['grid'][0]['discovery']['captured'])
            self.assertEqual(report['cases'][0]['grid'][0]['discovery']['record_sha256'],N.digest(row))


class PipelineTests(unittest.TestCase):
    def test_deadline_signals_only_owned_child(self):
        child=SimpleNamespace(pid=123,returncode=None)
        child.poll=lambda:child.returncode
        def terminate():child.returncode=-15
        child.terminate=terminate;child.wait=lambda timeout:child.returncode
        child.kill=lambda:self.fail('Terminate returned; no kill needed')
        with tempfile.TemporaryDirectory() as tmp,patch.object(P.subprocess,'Popen',return_value=child),patch.object(P.time,'monotonic',side_effect=[0,10000]):
            path=Path(tmp);state={}
            code=P.supervise(['owned'],source=path,env={'CUDA_VISIBLE_DEVICES':''},log=None,state=state,
                statepath=path/'state.json',output=path/'result',phase='cpu_admission',limit=60)
            self.assertEqual(code,-15);self.assertEqual(state['child_pid'],123)
            self.assertIn('deadline',state['error'])


if __name__=='__main__':unittest.main()
