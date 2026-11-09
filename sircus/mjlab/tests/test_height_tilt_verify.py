import copy
import unittest
from pathlib import Path
from sircus.mjlab import height_tilt_verify as a


def fixture():
    sources=[];plan=[]
    for ci,context in enumerate(a.CASES):
        source=dict(id='parent-'+context,context_id=context,phase='test',method='ppo',native_release=True,route_success=False,
                    values=[.6,-.14,.34,18.,.25],bucket_xy_m=[1.1,-.2])
        source.update({key:None for key in a.META});sources.append(source)
        for hi,h in enumerate((.32,.36,.4,.44,.46,.48)):
            for ti,t in enumerate((16.,20.,24.,28.,32.)):
                local=hi*5+ti;values=[.6,-.14,h,t,.25]
                row={key:copy.deepcopy(source[key]) for key in a.META}
                row.update(id=f'height-tilt-{context}-g{local:02d}',context_id=context,grid_index=ci*30+local,case_grid_index=local,
                    phase='discovery',variant='grid',method='grid',role='diagnostic',repeat=0,selection_or_learning=False,posthoc_development=True,
                    values=values,command_values=values,bucket_xy_m=source['bucket_xy_m'],
                    source_record_id=source['id'],source_record_sha256=a.digest(source),substeps=64,perturbation_mixed_units=[0.]*5)
                plan.append(row)
    return sources,plan


class IndependentHeightTiltTests(unittest.TestCase):
    def test_literal120grid_and_narrow460_height(self):
        sources,plan=fixture();checks=a.replay.AuditChecks();a.audit_grid(plan,sources,checks)
        self.assertEqual(len(plan),120)
        self.assertEqual(sorted({r['values'][2] for r in plan}),[.32,.36,.4,.44,.46,.48])

    def test_altered_fixed_gap_or_missing_candidate_rejected(self):
        sources,plan=fixture()
        for kind in ('gap','bucket','height','index','source','missing'):
            bad=copy.deepcopy(plan)
            if kind=='gap':bad[15]['values'][4]+=.001
            if kind=='bucket':bad[15]['bucket_xy_m'][0]+=.001
            if kind=='height':bad[15]['values'][2]=.52
            if kind=='index':bad[15]['grid_index']=16
            if kind=='source':bad[15]['source_record_sha256']='0'*64
            if kind=='missing':bad.pop()
            with self.subTest(kind=kind),self.assertRaises(a.EvidenceError):a.audit_grid(bad,sources,a.replay.AuditChecks())

    def test_selection_only_valid_captures_then_quality_reward_index(self):
        def row(index,quality,reward,**other):
            return dict(id=str(index),context_id=a.CASES[0],phase='discovery',native_release=True,route_success=True,
                        route_quality=quality,learning_reward=reward,grid_index=index,**other)
        rows=[row(7,1,10),row(6,2,3),row(4,2,3),row(2,2,3,invalid_physics=True),row(1,2,20,invalid='setup')]
        result=a.selected_captures(rows)
        self.assertEqual(result[a.CASES[0]]['id'],'4')
        self.assertTrue(all(result[c] is None for c in a.CASES[1:]))
        self.assertEqual(a.selected_captures(list(reversed(rows)))[a.CASES[0]]['id'],'4')

    def test_unattempted_matrix_is_not_admission_rejection(self):
        _,plan=fixture()
        cells=[dict(candidate_id=p['id'],context_id=p['context_id'],grid_index=p['grid_index'],case_grid_index=p['case_grid_index'],
                    values=p['values'],bucket_xy_m=p['bucket_xy_m'],resolution_file=p['context_resolution_file'],resolution_sha256=p['context_resolution_sha256'],
                    status='not_attempted',passed=None,valid_ik_branches=None,connection_file=None,connection_sha256=None,q10=None,q14=None) for p in plan]
        counts=a.audit_matrix(Path('/unused'),plan,cells,{}, {},a.replay.AuditChecks())
        self.assertEqual(counts,dict(total=120,attempted=0,admitted=0,rejected=0,not_attempted=120))
        cells[0]['passed']=False
        with self.assertRaises(a.EvidenceError):a.audit_matrix(Path('/unused'),plan,cells,{}, {},a.replay.AuditChecks())

    def test_native_replay_preserves_exact_admitted_branch(self):
        sources,plan=fixture();source=sources[0]
        source.update(q10=[0.]*6,q14=[1.]*6,connection_file='original',connection_sha256='a'*64)
        expected=a.request(source,'baseline','baseline','baseline',0,source['values'])
        observed=dict(expected,native_release=True,q10=source['q10'],q14=source['q14'],connection_file='original',connection_sha256='a'*64)
        a.audit_native_requests([observed],[expected],sources,[],a.replay.AuditChecks())
        for key,value in [('q14',[1.01]*6),('connection_sha256','b'*64),('native_release',False)]:
            changed=copy.deepcopy(observed);changed[key]=value
            with self.subTest(key=key),self.assertRaises(a.EvidenceError):
                a.audit_native_requests([changed],[expected],sources,[],a.replay.AuditChecks())


if __name__=='__main__':unittest.main()
