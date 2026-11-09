"""CPU-only tilt/gap contract, exact branches, selection and interruption checks."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from sircus.mjlab import case00_tilt_gap_contract as C
from sircus.mjlab import case00_tilt_gap_native as N
from sircus.mjlab import case00_tilt_gap_study as S
from sircus.mjlab import case00_tilt_gap_pipeline as P
from sircus.mjlab.tests.test_table_correction_native import fixture as parent_fixture


def fixture():
    old=parent_fixture()[0];rows=copy.deepcopy(old[:3]);row=copy.deepcopy(old[3])
    row.update(id=C.SOURCE_IDS[0],context_id=C.CONTEXTS[0],phase='discovery',route_success=False,
        diagnostic_robot_contact_steps=0,board_bounce=True,invalid_physics=False,
        values=[.5979922029820652,-.023557257454087432,.46,28.,.25181883224993984],
        bucket_xy_m=[1.1095395651982425,-.22146373203137287])
    rows.append(row);return rows,None


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.sources=C.sources_from_parent(fixture()[0]);self.plan=C.grid_plan(self.sources)

    def observed(self,index,success=True):
        row=copy.deepcopy(self.plan[index]);row.update(native_release=True,route_success=success,
            invalid=False,invalid_physics=False,board_bounce=True,diagnostic_robot_contact_steps=0,
            route_quality=2 if success else 0,learning_reward=4 if success else -1,
            connection_file='proof.json',connection_sha256='a'*64,q10=[0]*6,q14=[1]*6)
        return row

    def test70_literal_tilt_gap_pairs_preserve_xyz_and_bucket(self):
        self.assertEqual(len(self.plan),70);self.assertEqual(len({r['id'] for r in self.plan}),70)
        source=self.sources[-1]
        self.assertEqual([r['values'][3:] for r in self.plan],
            [[t,source['values'][4]+g] for t in (20.,22.,24.,26.,28.,30.,32.)
             for g in (-.06,-.04,-.02,-.01,0.,.01,.02,.04,.06,.08)])
        self.assertEqual([r['grid_index'] for r in self.plan],list(range(70)))
        for row in self.plan:
            self.assertEqual(row['values'][:3],source['values'][:3]);self.assertEqual(row['bucket_xy_m'],source['bucket_xy_m'])
            self.assertEqual(row['source_record_sha256'],N.digest(source));self.assertEqual(row['substeps'],64)
        self.assertEqual(self.plan[44]['values'],source['values'])

    def test_all_successes_get_three_fresh_repeats_with_unique_ids_and_frozen_order(self):
        rows=[self.observed(i) for i in reversed(range(70))];selection=C.select_captures(rows)
        self.assertEqual([s['selected']['grid_index'] for s in selection],list(range(70)))
        requests=C.selected_requests(self.plan,selection)
        self.assertEqual(len(requests),210);self.assertEqual(len({r['id'] for r in requests}),210)
        self.assertEqual(len(self.plan)+len(requests)+3+18,C.budget()['maximum_requested'])
        for index,row in enumerate(requests):
            self.assertEqual(row['values'],self.plan[index//3]['values']);self.assertEqual(row['repeat'],index%3)
            self.assertEqual(row['selected_from_sha256'],N.digest(self.observed(index//3)))

    def test_no_capture_means_no_repeat_selection_and_no_pretend_failures(self):
        self.assertEqual(C.select_captures([self.observed(0,False)]),[])
        self.assertEqual(C.selected_requests(self.plan,[]),[])

    def test_invalid_non_native_or_non_discovery_success_cannot_be_selected(self):
        invalid=[]
        for index,change in enumerate(({'invalid_physics':True},{'native_release':False},{'invalid':'bad'},
                                     {'phase':'baseline'},{'context_id':'foreign'})):
            row=self.observed(index);row.update(change);invalid.append(row)
        self.assertEqual(C.select_captures(invalid),[])

    def test_control_baseline_accounting_uses_original_actions(self):
        groups=[C.fixed_requests(self.sources,p) for p in ('controls_pre','baseline','controls_post')]
        self.assertEqual(list(map(len,groups)),[9,3,9]);self.assertEqual(len({r['id'] for g in groups for r in g}),21)
        lookup={r['id']:r for r in self.sources}
        self.assertTrue(all(r['values']==lookup[r['source_record_id']]['values'] for g in groups for r in g))

    def test_parent_clear_miss_source_must_be_exact_unique_and_valid(self):
        for change in ({'id':'wrong'},{'route_success':True},{'board_bounce':False},
                       {'diagnostic_robot_contact_steps':1},{'invalid_physics':True},{'native_release':False}):
            rows=copy.deepcopy(fixture()[0]);rows[-1].update(change)
            with self.subTest(change=change),self.assertRaises(ValueError):C.sources_from_parent(rows)
        rows=fixture()[0]
        with self.assertRaises(ValueError):C.sources_from_parent(rows+[rows[-1]])

    def test_empty_matrix_keeps_unattempted_distinct_from_declined(self):
        matrix=C.empty_matrix(self.plan);self.assertEqual(len(matrix),70)
        self.assertTrue(all(r['status']=='not_attempted' and r['passed'] is None for r in matrix))

class NativeTests(unittest.TestCase):
    def fixture(self,variant='grid',mode='ok'):
        sources=C.sources_from_parent(fixture()[0]);plan=C.grid_plan(sources)
        engine=N.Case00TiltGapEngine.__new__(N.Case00TiltGapEngine)
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

    def test_unchanged_grid_cell_cannot_substitute_another_admitted_branch(self):
        engine,_=self.fixture();request=copy.deepcopy(list(engine.plan.values())[44])
        source=engine.sources[request['source_record_id']]
        anchor=dict(sha256=source['context_resolution_sha256'],file='resolution.json',cache_hit=True)
        proof=dict(passed=True,reason=None,file=source['connection_file'],sha256=source['connection_sha256'],
            q10=copy.deepcopy(source['q10']),q14=copy.deepcopy(source['q14']))
        engine.engine.adapter.env=None
        engine.engine.adapter.resolver=SimpleNamespace(home={},resolve=lambda xy:anchor,connect=lambda a,v:proof)
        with patch.object(N,'verify_connection',return_value=True):
            self.assertTrue(engine.admit(request)['passed'])
            proof['q14'][0]+=.01
            with self.assertRaises(ValueError):engine.admit(request)

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
            engine=N.Case00TiltGapEngine.__new__(N.Case00TiltGapEngine);engine.directory=Path(tmp);engine.plan={r['id']:r for r in plan}
            resolver=SimpleNamespace(home={},_connected={},resolve=lambda xy:dict(cache_hit=True,sha256='resolution'))
            engine.engine=SimpleNamespace(adapter=SimpleNamespace(resolver=resolver))
            for row,request in zip(matrix,plan):
                key=N.digest(dict(version=N.GEOMETRY_VERSION,connection_policy=N.CONNECTION_POLICY_VERSION,
                    resolution_sha256='resolution',values=request['values'],bucket_xy_m=request['bucket_xy_m']))
                path=engine.directory/(key+'.json');path.write_text(json.dumps(dict(passed=True,q10=[0]*6,q14=[1]*6)))
                row.update(status='completed',passed=True,resolution_sha256='resolution',connection_file=path.name,connection_sha256=N.sha(path))
            with patch.object(N,'verify_connection',return_value=True):engine.hydrate_matrix(matrix)
            self.assertEqual(len(engine.matrix),70);self.assertTrue(resolver._connected)
            (engine.directory/matrix[0]['connection_file']).write_text('{}')
            with patch.object(N,'verify_connection',return_value=True),self.assertRaises(ValueError):engine.hydrate_matrix(matrix)
            matrix[0]['status']='not_attempted'
            with self.assertRaises(ValueError):engine.hydrate_matrix(matrix)


class ReportingTests(unittest.TestCase):
    def test_partial_cpu_stop_preserves70_cells_and_freezes_matrix(self):
        sources=C.sources_from_parent(fixture()[0]);plan=C.grid_plan(sources);matrix=C.empty_matrix(plan)
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)
            (output/'parent-rows.json').write_text(json.dumps(sources));(output/'request-contract.json').write_text('{}')
            matrix[0].update(status='completed',passed=False,reason='finite_ik_unresolved')
            S.finish_admission(output,dict(parent_bindings={},fingerprint='previous'),plan,sources,matrix,3590.,complete=False,error='cap')
            saved=json.loads((output/'admission-matrix.json').read_text());report=json.loads((output/'report.json').read_text())
            self.assertEqual(len(saved),70);self.assertEqual(report['admission_summary']['declined'],1)
            self.assertEqual(report['admission_summary']['unattempted'],69)
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
