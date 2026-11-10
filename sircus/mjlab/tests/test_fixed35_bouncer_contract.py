"""Bouncer-only frozen candidate selection preserves dropper/actions and branch lineage."""
import copy
from pathlib import Path
import unittest
import numpy as np
from sircus.mjlab import fixed35_bouncer_contract as C
from sircus.mjlab.tests.test_fixed35_six_contract import fixture as old_fixture,score_fields


def fixture():
    old=old_fixture()
    recovered=copy.deepcopy(next(r for r in old['cases'] if r['context_id'].endswith('01-05')))
    recovered.update(id='six-confirmed-'+recovered['context_id'],**score_fields(True))
    preserved=sorted(old['preserved']+[recovered],key=lambda r:r['context_id'])
    cases=[r for r in old['cases'] if r['context_id'] in C.CASE_IDS]
    numeric=[];pools={}
    for parent in cases:
        for index in range(4):
            row=copy.deepcopy(parent)
            row.update(id=f'six-p{index}-{parent["context_id"]}',phase='search',numeric_proposal_index=index,
                       ik_branch_slot=0,native_release=True,branch_pool_file=f'branch-pools/{parent["context_id"]}-{index}.json',branch_pool_sha256='a'*64)
            row['values'][2]+=.001*index
            candidates=[]
            for kind in ('same','wrist','proximal','wrapped_duplicate','second_proximal','changed_dropper'):
                c={k:copy.deepcopy(row[k]) for k in ('values','bucket_xy_m','q10','q14')}
                if kind=='wrist':c['q14'][-1]+=.4
                elif kind=='proximal':c['q14'][0]+=.3
                elif kind=='wrapped_duplicate':c['q14'][0]+=.3+2*np.pi
                elif kind=='second_proximal':c['q14'][1]+=.4
                elif kind=='changed_dropper':c['q14'][2]+=.3;c['q10'][0]+=.2
                candidates.append(c)
            pools[row['id']]=dict(values=row['values'],bucket_xy_m=row['bucket_xy_m'],endpoint_attempts=dict(candidates=candidates))
            numeric.append(row)
    return dict(old,cases=cases,failures=cases+old['blocked'],preserved=preserved,numeric_sources=numeric,candidate_pools=pools,
                sources=preserved+numeric,source_roots={})


class ContractTests(unittest.TestCase):
    def test_proximal_then_wrist_only_with_exact_parent_dropper(self):
        data=fixture();source=data['numeric_sources'][0];pool=data['candidate_pools'][source['id']]
        selected=C.select_alternatives(source,pool)
        self.assertEqual([r['candidate_index'] for r in selected],[2,1])
        self.assertEqual([r['proximal_changed'] for r in selected],[True,False])
        self.assertTrue(all(r['candidate']['q10']==source['q10'] for r in selected))
        self.assertTrue(all(r['candidate_sha256']==C.digest(r['candidate']) for r in selected))

    def test_wrap_duplicates_and_dropper_changes_cannot_supply_second_bouncer(self):
        data=fixture();s=data['numeric_sources'][0];p=data['candidate_pools'][s['id']]
        p['endpoint_attempts']['candidates']=[p['endpoint_attempts']['candidates'][i] for i in (0,2,3,5)]
        chosen=C.select_alternatives(s,p)
        self.assertEqual(len(chosen),1);self.assertEqual(chosen[0]['candidate_index'],1)
        p['values']=list(p['values']);p['values'][3]+=.1
        with self.assertRaises(ValueError):C.select_alternatives(s,p)

    def test_next_proximal_if_no_wrist_and_empty_pool_is_explicit(self):
        d=fixture();s=d['numeric_sources'][0];p=d['candidate_pools'][s['id']]
        p['endpoint_attempts']['candidates']=[p['endpoint_attempts']['candidates'][i] for i in (0,2,4)]
        self.assertEqual([r['candidate_index'] for r in C.select_alternatives(s,p)],[1,2])
        p['endpoint_attempts']['candidates']=p['endpoint_attempts']['candidates'][:1]
        self.assertEqual(C.select_alternatives(s,p),[])

    def test_frozen24_max_plan_interleaves_cases_without_numeric_or_q10_changes(self):
        d=fixture();before=copy.deepcopy(d);plan,inventory=C.make_plan(d['numeric_sources'],d['candidate_pools'])
        self.assertEqual(d,before);self.assertEqual((len(plan),len(inventory)),(24,12))
        self.assertEqual(len({r['id'] for r in plan}),24)
        for i in range(8):self.assertEqual([r['context_id'] for r in plan[i*3:i*3+3]],list(C.CASE_IDS))
        sources={r['id']:r for r in d['numeric_sources']}
        for r in plan:
            source=sources[r['parent_row_id']]
            self.assertEqual(r['values'],source['values']);self.assertEqual(r['bucket_xy_m'],source['bucket_xy_m'])
            self.assertEqual(r['selected_q10'],source['q10']);self.assertEqual(r['parent_row_sha256'],C.digest(source))
            self.assertNotIn('ik_branch_slot',r)
        self.assertEqual(C.protocol()['maximum_requests'],24+27+30+18)
        self.assertFalse(C.protocol()['new_ik']);self.assertFalse(C.protocol()['training'])

    def test_admitted_sources_prefer_slot0_without_duplicates(self):
        d=fixture();rows=d['numeric_sources'];alternate=copy.deepcopy(rows[0]);alternate.update(id='slot1',ik_branch_slot=1)
        self.assertEqual(C.numeric_sources(rows+[alternate]),rows)
        with self.assertRaises(ValueError):C.numeric_sources(rows[:-1])

    def test30_preserved_recipes_exact_and_no_candidate_selection(self):
        d=fixture();plan=C.preserved_plan(d['preserved'])
        self.assertEqual(len(plan),30)
        for source,r in zip(d['preserved'],plan):
            self.assertEqual(source['values'],r['values']);self.assertEqual(r['parent_row_sha256'],C.digest(source))
            self.assertTrue(r['exact_parent_action']);self.assertNotIn('candidate_index',r)

    def test_three_fresh_repeats_keep_candidate_and_source_lineage(self):
        d=fixture();plan,_=C.make_plan(d['numeric_sources'],d['candidate_pools']);p=plan[0]
        found=dict(p,**score_fields(True),q10=p['selected_q10'],q14=p['selected_q14'],connection_sha256='b'*64)
        repeats=C.repeat_plan(found,0)
        for r in repeats:
            self.assertEqual(r['selected_from_sha256'],C.digest(found));self.assertEqual(r['parent_row_id'],p['parent_row_id'])
            self.assertEqual(r['candidate_sha256'],p['candidate_sha256']);self.assertEqual(r['selected_q14'],found['q14'])
        measured=[dict(r,**score_fields(True),q10=found['q10'],q14=found['q14'],connection_sha256=found['connection_sha256']) for r in repeats]
        self.assertEqual(C.confirmation_groups(measured,p['context_id'])[0]['status'],'confirmed')
        measured[-1]['q14']=[99.]*6
        self.assertEqual(C.confirmation_groups(measured,p['context_id'])[0]['status'],'failed')

    def test_zero_candidate_case_is_exhausted_without_fabricating_requests(self):
        d=fixture();plan,_=C.make_plan(d['numeric_sources'],d['candidate_pools']);plan=[r for r in plan if r['context_id']!=C.CASE_IDS[0]]
        cases=C.case_results([],d['cases'],d['blocked'],plan,finished=True,budget_truncated=True)
        first=next(r for r in cases if r['context_id']==C.CASE_IDS[0])
        self.assertEqual((first['status'],first['planned_search_requests']),('search_exhausted',0))
        self.assertEqual(sum(r['status']=='fixed_base_clearance_blocked' for r in cases),2)

    def test_rule_is_frozen_and_published_parent_hashes_are_exact(self):
        self.assertEqual(C.read(Path(C.__file__).with_name('fixed35_bouncer_rule.json'))['protocol'],C.protocol())
        self.assertEqual(len(C.PARENT_REPORT_SHA),64);self.assertEqual(len(C.PARENT_AUDIT_SHA),64)

if __name__=='__main__':unittest.main()
