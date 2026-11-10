"""Audit variable queues, honest truncation and outcome-independent ordering."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from sircus.mjlab import fixed35_bouncer_schedule_verify as V
from sircus.mjlab.replay650_verify import AuditChecks
from sircus.mjlab.benchmark_verify import EvidenceError,sha


def fixture(root, *, cutoff=None, delayed=False):
    cases=[dict(context_id=c) for c in ('a','b','c')]
    plan=[dict(id=f'{c}-{i}',phase='search',context_id=c,values=[i])
          for c,n in zip(('a','b','c'),(2,1,3)) for i in range(n)]
    preserved=[dict(id=f'keep-{i}',phase='preserved') for i in range(30)]
    (root/'search-plan.json').write_text(json.dumps(dict(rows=plan)))
    (root/'repeat-plans').mkdir()
    rows=[];boundaries=[];timings=[];raw=[];clock=0.
    def boundary(phase,**extra):
        nonlocal clock
        clock+=1
        boundaries.append(dict(phase=phase,rows_count=len(rows),rows_sha256=hashlib.sha256(b''.join(raw)).hexdigest() if raw else None,acquired_elapsed_s=clock,**extra))
    def batch(phase,wanted):
        nonlocal clock
        start=clock+1;clock+=2
        timings.append(dict(phase=phase,acquired_start_s=start,acquired_end_s=clock))
        for row in wanted:
            r=dict(row,native_release=False,route_success=False)
            rows.append(r);raw.append((json.dumps(r)+'\n').encode())
    boundary('inputs-frozen');batch('controls_pre',[dict(id=f'pre{i}',phase='controls_pre') for i in range(9)])
    boundary('controls_pre-complete');batch('preserved',preserved);boundary('preserved-complete')
    for i in range(3):
        if cutoff==i:break
        if delayed and i==0:clock=2400
        active=[c for c in ('a','b','c') if i<{'a':2,'b':1,'c':3}[c]]
        boundary('search-round-frozen',proposal_index=i,contexts=active)
        wanted=[dict(next(r for r in plan if r['context_id']==c and r['values']==[i]),search_plan_sha256=sha(root/'search-plan.json')) for c in active]
        batch('search',wanted);boundary('search-round-complete',proposal_index=i,stopped={})
    truncated=cutoff is not None
    if truncated:clock=max(clock,2400)
    report=dict(budget_truncated=truncated,completion_reason='search_time_limit' if truncated else 'finite_candidate_or_confirmation_caps')
    boundary('search-complete',**report);batch('controls_post',[dict(id=f'post{i}',phase='controls_post') for i in range(9)])
    boundary('controls_post-complete')
    (root/'rows.jsonl').write_bytes(b''.join(raw))
    for name,values in [('phase-boundaries.jsonl',boundaries),('timing.jsonl',timings)]:
        (root/name).write_text(''.join(json.dumps(v)+'\n' for v in values))
    return rows,report,dict(cases=cases),plan,preserved


def check(root,parts):
    return V.audit_schedule(root,*parts,lambda *_:[],AuditChecks())


class BouncerScheduleTests(unittest.TestCase):
    def test_variable_case_lengths_complete_without_fake_truncation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);result=check(root,fixture(root))
            self.assertEqual((result['search_rounds'],result['search_requests'],result['total_requests']),(3,6,54))
    def test_real_cutoff_retains_honest_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);result=check(root,fixture(root,cutoff=1))
            self.assertTrue(result['budget_truncated']);self.assertEqual(result['search_requests'],3)
    def test_no_silent_early_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);parts=list(fixture(root,cutoff=1));parts[1]['budget_truncated']=False
            with self.assertRaises(EvidenceError):check(root,parts)
    def test_no_search_round_after_cutoff(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with self.assertRaises(EvidenceError):check(root,fixture(root,delayed=True))
    def test_missing_last_case_candidate_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);parts=list(fixture(root));parts[0]=[r for r in parts[0] if r['id']!='c-2']
            with self.assertRaises(EvidenceError):check(root,parts)
    def test_confirmation_requires_same_bouncer_candidate_and_joints(self):
        row=dict(context_id='a',confirmation_index=0,phase='repeat',native_release=True,
                 route_success=True,invalid=None,selected_from_id='discovery',selected_from_sha256='source',
                 values=[1,2,3,4,5],bucket_xy_m=[1,2],candidate_sha256='candidate',
                 q10=[0]*6,q14=[1]*6,connection_sha256='connection')
        rows=[dict(row,id=f'repeat-{i}',repeat=i) for i in range(3)]
        self.assertEqual(V.independent_groups(rows,'a')[0]['status'],'confirmed')
        rows[2]['q14']=[2]*6
        self.assertEqual(V.independent_groups(rows,'a')[0]['status'],'failed')

    def test_immutable_prefix_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);parts=fixture(root)
            with (root/'rows.jsonl').open('ab') as f:f.write(b'{}\n')
            # An unaccounted physical row must not disappear behind parsed rows.
            with self.assertRaises(EvidenceError):
                V.audit_schedule(root,parts[0]+[{}],*parts[1:],lambda *_:[],AuditChecks())

if __name__=='__main__':unittest.main()
