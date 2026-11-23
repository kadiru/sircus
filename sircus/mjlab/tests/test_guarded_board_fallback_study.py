import copy
import importlib.util
import math
import unittest
from pathlib import Path
from sircus.mjlab import guarded_board_fallback_study as study
from sircus.mjlab import board_fallback_study as previous
from sircus.mjlab.tests.test_board_fallback_study import provenance
from sircus.mjlab.tests.test_stalled_reference_study import choice,outcome
from sircus.mjlab.planning_budget_plan import _wrap_choice


class GuardedStudyTests(unittest.TestCase):
    def test_twenty_targets_exclude_previous_fifty_and_use_declared_strata(self):
        p=provenance();old=previous.make_layouts(p)
        p['excluded_coordinates_m'] += [r['bucket_xy_m'] for r in old]
        before=copy.deepcopy(p);rows=study.make_layouts(p)
        self.assertEqual(p,before);self.assertEqual(rows,study.make_layouts(p))
        self.assertEqual(len(rows),20);self.assertEqual(len({r['id'] for r in rows}),20)
        known=list(p['excluded_coordinates_m'])
        for row in rows:
            self.assertGreaterEqual(min(math.dist(row['bucket_xy_m'],xy) for xy in known),.005)
            known.append(row['bucket_xy_m'])
            self.assertEqual((row['context_id'],row['split']),(row['id'],'test'))
        self.assertEqual(sum('fixed base' in r['region'] for r in rows),4)
        self.assertEqual(sum('grid edge' in r['region'] for r in rows),8)

    def test_diagnostic_repeats_preserve_exact_geometry_joint_and_proof_fields(self):
        layout=dict(id='known',bucket_xy_m=[.948252196166334,-.09781574379730379])
        original=choice(layout);saved=copy.deepcopy(original)
        requests=study.repeated_requests(original,'guarded')
        frozen=_wrap_choice(original,'guarded')['request']
        self.assertEqual(original,saved);self.assertEqual(len(requests),3)
        for index,request in enumerate(requests,1):
            self.assertEqual(request['id'],f"{frozen['id']}-repeat-{index}")
            self.assertEqual(request['repeat'],index)
            self.assertEqual(request['source_request_id'],frozen['id'])
            self.assertEqual({k:v for k,v in request.items() if k not in {'id','repeat','source_request_id'}},
                             {k:v for k,v in frozen.items() if k!='id'})
        requests[0]['selected_q10'][0]=99
        self.assertNotEqual(requests[1]['selected_q10'][0],99)
        self.assertEqual(original,saved)
        self.assertEqual(study.repeated_requests(choice(layout,False),'baseline'),[])

    @unittest.skipIf(importlib.util.find_spec('torch') is None,'Pinned Ubuntu summary dependencies require Torch')
    def test_fresh_denominator_excludes_diagnostic_and_keeps_setup_declines(self):
        layouts=[dict(id=f'fresh-{i}',bucket_xy_m=[.3+.1*i,-.3]) for i in range(2)]
        slots=[];rows={};wrappers=[]
        for layout,labels in zip(layouts,[('direct','table'),(None,'flight_miss')]):
            for method,label in zip(study.METHODS,labels):
                inner=choice(layout,label is not None);slot=_wrap_choice(inner,method);slots.append(slot)
                wrappers.append(dict(variant=method,site=dict(layout=layout,choice=inner,request_wall_s=3.,decision_latency_s=2.)))
                if label is not None:rows[slot['id']]=outcome(label)
        result=study.summarize(layouts,slots,rows,wrappers)
        self.assertEqual(result['methods']['baseline']['positions'],2)
        self.assertEqual(result['methods']['baseline']['no_candidate'],1)
        self.assertEqual(result['methods']['guarded']['captures'],1)
        self.assertEqual(result['paired'],dict(both=1,gained=0,lost=0,neither=1))

    def test_preflight_is_a_distinct_non_native_stage(self):
        names=['parent-study','parent-audit','source-study','source-audit','reference','assets','evidence-root','output','gpu-lock']
        args=study.argument_parser().parse_args([x for n in names for x in ('--'+n,'/fixture')]+['--stage','preflight'])
        self.assertEqual(args.stage,'preflight');self.assertEqual(study.GPU_CAP_S,1800)


if __name__=='__main__':unittest.main()
