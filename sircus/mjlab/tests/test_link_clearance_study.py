"""New branch menus preserve successful evidence and independent repeats."""
import copy
import math
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from sircus.mjlab.benchmark_verify import sha
from sircus.mjlab.sweep_worker import atomic
from sircus.mjlab.link_clearance_plan import (CASE_IDS,SEARCH_RULE,proposal_menu,
    preparation_selection,candidate_selection,equivalent_commands,joint_change,branch_solver)
from sircus.mjlab.link_clearance_study import (schedule,preservation_schedule,
    preservation_summary,repeat_schedule,verify_preflight,rebuild_preparation_engine)


def request(identity='old'):
    return dict(id=identity,values=[.60,-.04,.40,28.,.25],bucket_xy_m=[1.2,.147],
        board_heading_offset_deg=3.,board_twist_deg=180.,selected_q10=[.1]*6,selected_q14=[.2]*6,
        intended_route='table',prepared_connection_file='proof.json',prepared_connection_sha256='a'*64)


def case(identity=CASE_IDS[0]):
    sources=[]
    for i in range(8):
        r=request('source-'+str(i));r['values'][1]+=.025*i
        sources.append(dict(id=r['id'],source_request=r,strength='prior_robot_contact',prediction={'score_m':-.02*i}))
    return dict(id=identity,original=request(),directory='native-'+identity[-2:],sources=sources)

LOW=[.3,-.4,.2,10.,.08]
HIGH=[1.1,.4,.6,50.,.55]


def prepared(p,dx10=0.,dx14=0.):
    r=copy.deepcopy(p);r['selected_q10']=r['seed_q10'][:];r['selected_q14']=r['seed_q14'][:]
    r['selected_q10'][0]+=dx10;r['selected_q14'][0]+=dx14
    return r


class LinkClearanceTests(unittest.TestCase):
    def test_menu_preserves_sources_and_fixed_targets(self):
        c=case();before=copy.deepcopy(c);menu=proposal_menu(c,LOW,HIGH)
        self.assertEqual(c,before);self.assertEqual(menu,proposal_menu(c,LOW,HIGH))
        source={s['id']:s['source_request'] for s in c['sources']}
        self.assertEqual(sum(p['fixed_cartesian'] for p in menu),12)
        for p in menu:
            old=source[p['source_request_id']]
            self.assertEqual(p['bucket_xy_m'],old['bucket_xy_m'])
            self.assertEqual(p['values'][2:],old['values'][2:])
            self.assertNotIn('prepared_connection_file',p)
            self.assertNotIn('selected_q14',p)
            if p['fixed_cartesian']:
                self.assertEqual(p['values'],old['values'])
                self.assertEqual(p['board_heading_offset_deg'],old['board_heading_offset_deg'])
                self.assertEqual(p['board_twist_deg'],old['board_twist_deg'])
            self.assertTrue(all(lo<=v<=hi for lo,v,hi in zip(LOW,p['values'],HIGH)))
            self.assertLessEqual(math.dist(p['values'][:2],old['values'][:2]),.15+1e-12)

    def test_full_checks_reserve_fixed_pose_and_balance_transforms(self):
        menu=proposal_menu(case(),LOW,HIGH);scores={p['id']:{'score_m':-99. if p['fixed_cartesian'] else .1} for p in menu}
        wrist={p['id']:{'passed':True} for p in menu}
        chosen=preparation_selection(menu,scores,wrist)
        self.assertEqual(len(chosen),96)
        self.assertEqual([p['proposal_family'] for p in chosen[:4]],['fixed_cartesian','radial_shift','lateral_shift','board_twist'])
        self.assertEqual(sum(p['fixed_cartesian'] for p in chosen[:24]),6)
        self.assertEqual({p['proposal_family'] for p in chosen},{'fixed_cartesian','radial_shift','lateral_shift','board_twist'})
        wrist[chosen[0]['id']]['passed']=False
        self.assertNotIn(chosen[0],preparation_selection(menu,scores,wrist))

    def test_nested_reference_search_preserves_its_keywords_and_budget(self):
        observed=[]
        def implementation(targets,initial,**kwargs):
            observed.append((targets,initial,kwargs));return kwargs
        solve=branch_solver(implementation,1234)
        outer=solve(['pose'],'home',starts=1,iterations=80,known='source',normal_only=False)
        self.assertEqual(outer,dict(starts=16,iterations=180,seed=1234,preserve_known=[True],known='source',normal_only=False))
        nested=solve(['reference'],'other',starts=8,iterations=120,seed=77,preserve_known=[False],known='reference-joints')
        self.assertEqual(nested,dict(starts=8,iterations=120,seed=77,preserve_known=[False],known='reference-joints'))
        default=solve(['reference'],'other',starts=4,iterations=100,preserve_known=None)
        self.assertEqual(default,dict(starts=4,iterations=100,preserve_known=None))
        self.assertEqual(len(observed),3)

    def test_branch_selection_prioritizes_board_elbow_and_removes_numerical_duplicates(self):
        c=case();menu=proposal_menu(c,LOW,HIGH);p=next(p for p in menu if p['fixed_cartesian'])
        arm=prepared(p,dx14=.4);arm['id']='board'
        drop=prepared(p,dx10=1.2);drop['id']='dropper'
        duplicate=copy.deepcopy(arm);duplicate['id']='numerical';duplicate['selected_q14'][2]+=.001
        unchanged=prepared(p);unchanged['id']='unchanged'
        rows=[drop,arm,duplicate,unchanged];pred={r['id']:{'score_m':.1} for r in rows}
        sources={s['id']:s['source_request'] for s in c['sources']}
        chosen=candidate_selection(rows,pred,sources)
        self.assertEqual(len(chosen),2)
        self.assertIn(chosen[0]['id'],('board','numerical'))
        self.assertEqual(chosen[1]['id'],'dropper')
        self.assertTrue(equivalent_commands(arm,duplicate))
        duplicate['selected_q14'][2]+=.10;self.assertFalse(equivalent_commands(arm,duplicate))
        alternate=copy.deepcopy(arm);alternate['selected_q14'][0]+=2*math.pi
        self.assertTrue(equivalent_commands(arm,alternate))
        self.assertEqual(joint_change(unchanged,sources[p['source_request_id']])['maximum_rad'],0.)

    def test_preservation_and_maximum_budget(self):
        cases=[]
        for identity in CASE_IDS:
            c=case(identity);c['candidates']=[dict(request(identity+'-'+str(i)),context_id=identity) for i in range(32)];cases.append(c)
        preserved=[dict(id='old-'+str(i),original=request('preserve-'+str(i)),directory='preserved-'+str(i)) for i in range(3)]
        plan=dict(cases=cases,preserved_cases=preserved);before=copy.deepcopy(plan)
        slots=schedule(plan);sentinels=preservation_schedule(plan)
        self.assertEqual(len(slots),70);self.assertEqual(len(sentinels),9)
        good=dict(native_release=True,invalid_physics=False,route_success=True,shot_route='direct')
        rows=[dict(good,variant='search',candidate_id=s['candidate_id']) for s in slots if s['variant']=='search']
        repeats=repeat_schedule(plan,rows)
        self.assertEqual(len(repeats),192)
        self.assertEqual(len(slots)+len(sentinels)+len(repeats)+18,289)
        self.assertEqual(plan,before)
        for s in sentinels:
            self.assertEqual(s['request']['phase'],'preservation')
            source=next(c['original'] for c in preserved if c['id']==s['case_id'])
            for key in ('values','selected_q10','selected_q14','prepared_connection_sha256'):
                self.assertEqual(s['request'][key],source[key])
        measured=[dict(good,variant='preserved') for _ in range(9)]
        self.assertTrue(preservation_summary(measured)['passed'])
        measured[-1]['invalid_physics']=True;self.assertFalse(preservation_summary(measured)['passed'])
        self.assertFalse(preservation_summary(measured[:-1])['passed'])

    def test_rebuild_hydrates_completed_fallback_proof_before_any_new_checks(self):
        gate=object();planner=Namespace(gate=gate)
        engine=Namespace(adapter=Namespace(resolver=Namespace(planner=planner)))
        source=request();events=[]
        with patch('sircus.mjlab.link_clearance_study.engine_for',return_value=engine), \
             patch('sircus.mjlab.parallel_capture_grid_workers.load_resolution',side_effect=lambda e,xy:events.append(('hydrate',e,xy))) as load, \
             patch('sircus.mjlab.arm_check_speed.install_vectorized_gate',side_effect=lambda g:events.append(('gate',g))), \
             patch('sircus.mjlab.arm_path_speed.install_path_short_circuit',side_effect=lambda p:events.append(('path',p))):
            self.assertIs(rebuild_preparation_engine(None,source,Path('/unused'),{}),engine)
            load.assert_called_once_with(engine,source['bucket_xy_m'])
            self.assertEqual([e[0] for e in events],['hydrate','gate','path'])

    def test_rebuild_rejects_unverifiable_saved_resolution(self):
        engine=Namespace(adapter=Namespace(resolver=Namespace(planner=Namespace(gate=object()))))
        with patch('sircus.mjlab.link_clearance_study.engine_for',return_value=engine), \
             patch('sircus.mjlab.parallel_capture_grid_workers.load_resolution',side_effect=ValueError('source bytes changed')), \
             patch('sircus.mjlab.arm_check_speed.install_vectorized_gate') as gate, \
             patch('sircus.mjlab.arm_path_speed.install_path_short_circuit') as path:
            with self.assertRaisesRegex(ValueError,'source bytes changed'):
                rebuild_preparation_engine(None,request(),Path('/unused'),{})
            gate.assert_not_called();path.assert_not_called()

    def test_preflight_receipt_must_bind_current_inputs(self):
        with TemporaryDirectory() as folder:
            root=Path(folder)
            for name in ('contract.json','preparation.json','slots.json'):atomic(root/name,{'name':name})
            receipt=dict(status='passed',version='sircus-link-clearance-preparation-audit-v1',
                native_execution_performed=False,checks_count=10,contract_sha256=sha(root/'contract.json'),
                preparation_sha256=sha(root/'preparation.json'),slots_sha256=sha(root/'slots.json'))
            atomic(root/'audit.json',receipt);a=Namespace(output=root,preflight_audit=root/'audit.json')
            self.assertEqual(verify_preflight(a),receipt)
            atomic(root/'slots.json',{'tampered':True})
            with self.assertRaises(ValueError):verify_preflight(a)


if __name__=='__main__':unittest.main()
