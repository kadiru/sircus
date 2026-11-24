"""Fixed Cartesian context, bounded gap/tilt grid and causal branch/repeat guards."""
import copy
import math
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from sircus.mjlab.benchmark_verify import sha
from sircus.mjlab.sweep_worker import atomic
from sircus.mjlab.clear_shot_range_plan import (SOURCE_ID,CASE_IDS,GAP_DELTAS_M,TILT_DELTAS_DEG,
    proposal_menu,admit_board_branch,board_joint_change,fixed_board_solver,prepare_grid_command)
from sircus.mjlab.clear_shot_range_study import (summarize,schedule,repeat_schedule,
    preservation_schedule,verify_preflight)


def case():
    original=dict(id=SOURCE_ID,values=[.6803429,-.0110443,.46186837,25.46875,.10943864],
        bucket_xy_m=[1.192994,.147423],board_heading_offset_deg=6.94774,board_twist_deg=135.,
        selected_q10=[.1]*6,selected_q14=[.2]*6,intended_route='table',
        prepared_connection_file='proof.json',prepared_connection_sha256='a'*64)
    return dict(id=CASE_IDS[0],original=original,directory='native-09')


def prepared(p):
    return dict(copy.deepcopy(p),id=p['id']+'-b0',proposal_id=p['id'],selected_q10=p['seed_q10'][:],
        selected_q14=p['seed_q14'][:],prepared_connection_file='new.json',prepared_connection_sha256='b'*64)


def record(p,r):
    return dict(proposal_id=p['id'],admitted=1,admitted_request_ids=[r['id']],reason=None,attempted=True)


def good(r,variant='search',repeat=0):
    return dict(id=r['id']+'-'+variant+'-'+str(repeat),candidate_id=r['id'],context_id=r['context_id'],
        variant=variant,repeat=repeat,native_release=True,invalid_physics=False,route_success=True,shot_route='direct',
        experimental_route=dict(robot_contact_steps=0,robot_assisted_capture=False))


class ClearShotRangeTests(unittest.TestCase):
    def test_exact_grid_preserves_xyz_heading_twist_bucket_and_seed(self):
        c=case();before=copy.deepcopy(c);rows=proposal_menu(c)
        self.assertEqual(c,before);self.assertEqual(len(rows),49)
        self.assertEqual(len({tuple(r['values']) for r in rows}),49)
        self.assertEqual([(r['tilt_delta_deg'],r['gap_delta_m']) for r in rows],
            [(t,g) for t in TILT_DELTAS_DEG for g in GAP_DELTAS_M if t or g])
        self.assertTrue(all(r['tilt_delta_deg']==0 for r in rows[:9]))
        for row in rows:
            for key in ('bucket_xy_m','board_heading_offset_deg','board_twist_deg'):
                self.assertEqual(row[key],c['original'][key])
            self.assertEqual(row['values'][:3],c['original']['values'][:3])
            self.assertEqual(row['seed_q14'],c['original']['selected_q14'])
            self.assertNotIn('prepared_connection_file',row)
            self.assertNotIn('selected_q14',row)

    def test_gap_only_rejects_changed_board_branch_and_preserves_rejected_proof(self):
        c=case();p=proposal_menu(c)[0];r=prepared(p);r['selected_q14'][0]+=.01
        logs=[record(p,r)];before=copy.deepcopy((r,logs))
        accepted,changed,rejected=admit_board_branch(c['original'],p,[r],logs)
        self.assertEqual(accepted,[]);self.assertEqual(changed[0]['admitted'],0)
        self.assertEqual(changed[0]['reason'],'fixed_board_joint_branch_changed')
        self.assertEqual(rejected[0]['request'],r);self.assertEqual((r,logs),before)
        r['selected_q14']=c['original']['selected_q14'][:];r['selected_q14'][0]+=2*math.pi
        self.assertLess(board_joint_change(r,c['original']),1e-12)
        self.assertEqual(admit_board_branch(c['original'],p,[r],[record(p,r)])[0],[r])
        p2=proposal_menu(c)[9];r2=prepared(p2);r2['selected_q14'][0]+=.1
        self.assertEqual(admit_board_branch(c['original'],p2,[r2],[record(p2,r2)])[0],[r2])

    def test_gap_only_preserves_endpoint_without_changing_nested_reference_ik(self):
        def original(targets,initial,**kwargs):return kwargs
        solve=fixed_board_solver(original)
        self.assertEqual(solve([0],'initial',starts=1,iterations=80,known='saved',normal_only=False),
            dict(starts=1,iterations=80,known='saved',normal_only=False,preserve_known=[True]))
        self.assertEqual(solve([0],'reference',starts=16,iterations=180,known='other',preserve_known=[False],seed=77),
            dict(starts=16,iterations=180,known='other',preserve_known=[False],seed=77))
        arm=Namespace(solve_candidates=original)
        class Engine:
            adapter=Namespace(resolver=Namespace(solver=Namespace(ik=Namespace(board=arm))))
            def prepare(self,proposals,one_per_setting):
                self_observed=arm.solve_candidates([0],None,starts=1,iterations=80)
                assert self_observed['preserve_known']==[True] and one_per_setting
                raise RuntimeError('mock path failure')
        with self.assertRaisesRegex(RuntimeError,'mock path failure'):prepare_grid_command(Engine(),proposal_menu(case())[0])
        self.assertIs(arm.solve_candidates,original)

    def test_full_grid_keeps_rejections_and_timeouts_visible(self):
        c=case();menu=proposal_menu(c);a=prepared(menu[0]);b=prepared(menu[1])
        records=[record(menu[0],a),record(menu[1],b)]
        for p in menu[2:]:records.append(dict(proposal_id=p['id'],admitted=0,admitted_request_ids=[],reason='case_budget_not_attempted',attempted=False))
        records[2].update(attempted=True,reason='finite_ik_unresolved')
        records[3].update(attempted=True,reason='proposal_preparation_time_limit')
        c.update(proposals=menu,records=records,admitted=[a,b],candidates=[a,b])
        rows=[good(a),dict(good(b),route_success=False,shot_route=None)]
        rows += [good(a,'repeat',i) for i in (1,2,3)]
        out=summarize({'cases':[c]},rows)[0]
        self.assertEqual(len(out['grid']),49);self.assertEqual(out['checked'],4)
        self.assertEqual(out['grid'][0]['setup_status'],'admitted');self.assertTrue(out['grid'][0]['confirmed'])
        self.assertEqual(out['grid'][1]['initial_outcome'],'flight_miss')
        self.assertEqual(out['grid'][2]['setup_status'],'rejected')
        self.assertEqual(out['grid'][3]['setup_status'],'not_checked')
        self.assertIsNone(out['grid'][4]['initial_outcome'])
        self.assertEqual(out['results']['search']['captures'],1)
        self.assertEqual(out['results']['repeat']['captures'],3)

    def test_maximum_native_budget_and_exact_originals(self):
        c=case();c['candidates']=[prepared(p) for p in proposal_menu(c)]
        preserved=[dict(id='old-'+str(i),directory='preserved-'+str(i),original=c['original']) for i in range(3)]
        plan=dict(cases=[c],preserved_cases=preserved);before=copy.deepcopy(plan)
        slots=schedule(plan);repeats=repeat_schedule(plan,[good(r) for r in c['candidates']])
        self.assertEqual(len(slots),52);self.assertEqual(len(repeats),147)
        self.assertEqual(len(slots)+len(repeats)+len(preservation_schedule(plan))+18,226)
        self.assertEqual(plan,before)
        for s in slots[:3]:
            for key in ('values','selected_q10','selected_q14','prepared_connection_sha256'):
                self.assertEqual(s['request'][key],c['original'][key])

    def test_preflight_receipt_binding(self):
        with TemporaryDirectory() as folder:
            root=Path(folder)
            for name in ('contract.json','preparation.json','slots.json'):atomic(root/name,{'name':name})
            receipt=dict(status='passed',version='sircus-clear-shot-range-preparation-audit-v1',
                native_execution_performed=False,checks_count=10,contract_sha256=sha(root/'contract.json'),
                preparation_sha256=sha(root/'preparation.json'),slots_sha256=sha(root/'slots.json'))
            atomic(root/'audit.json',receipt);a=Namespace(output=root,preflight_audit=root/'audit.json')
            self.assertEqual(verify_preflight(a),receipt)
            atomic(root/'slots.json',{'changed':True})
            with self.assertRaises(ValueError):verify_preflight(a)


if __name__=='__main__':unittest.main()
