import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from sircus.mjlab import parallel_board_study as study
from sircus.mjlab.tests.test_stalled_reference_study import choice,outcome


class ParallelStudyTests(unittest.TestCase):
    def source(self,path):
        layouts=[dict(id=identity,bucket_xy_m=[.8+i*.01,-.1],context_id=identity,split='test')
                 for i,identity in enumerate(study.SOURCE_MISSES)]
        (path/'layouts.json').write_text(json.dumps(dict(layouts=layouts)))
        (path/'diagnostic-layout.json').write_text(json.dumps(dict(layout=dict(id='known-old',bucket_xy_m=[.948,-.097]))))
        report=dict(comparison=dict(positions=[dict(id=r['id'],outcomes=dict(baseline='flight_miss',guarded='flight_miss')) for r in layouts]),
                    diagnostic=dict(methods={v:dict(released=3,captures=3) for v in ('baseline','guarded')}))
        (path/'report.json').write_text(json.dumps(report));return report

    def test_exact_known_targets_are_marked_development(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);self.source(p);rows=study.make_layouts(p)
            self.assertEqual(len(rows),6)
            self.assertEqual([r['source_id'] for r in rows[:5]],list(study.SOURCE_MISSES))
            self.assertEqual([r['role'] for r in rows],['previous_miss']*5+['known_regression'])
            self.assertEqual(rows[-1]['bucket_xy_m'],[.948,-.097])
            self.assertEqual(len({r['context_id'] for r in rows}),6)

    def test_source_outcome_cannot_be_silently_relabelled(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);r=self.source(p);r['comparison']['positions'][0]['outcomes']['guarded']='direct'
            (p/'report.json').write_text(json.dumps(r))
            with self.assertRaises(ValueError):study.make_layouts(p)

    def test_native_channels_keep_original_and_alternative_proof_namespaces(self):
        rows=[]
        for i,lane in enumerate(('original','alternative')):
            layout=dict(id=f'p{i}',bucket_xy_m=[.4,-.1]);c=choice(layout)
            for method in study.METHODS:
                rows.append(dict(variant=method,site=dict(layout=layout,choice=c,selected_adapter_directory=lane)))
        before=copy.deepcopy(rows);channels=study.native_channels(rows)
        self.assertEqual({k:len(v) for k,v in channels.items()},dict(baseline=6,**{'parallel/original':3,'parallel/alternative':3}))
        self.assertEqual(len({r['id'] for rs in channels.values() for r in rs}),12)
        self.assertEqual(rows,before)
        for rs in channels.values():
            for r in rs:self.assertEqual(r['selected_q10'],before[0]['site']['choice']['request']['selected_q10'])

    @unittest.skipIf(importlib.util.find_spec('torch') is None,'Pinned Ubuntu summary dependency requires Torch')
    def test_repeat_denominator_keeps_no_setup_and_requires_three_for_reliability(self):
        layout=dict(id='p',source_id='old',bucket_xy_m=[.4,-.1],role='previous_miss');wrappers=[];rows={}
        for method in study.METHODS:
            c=choice(layout,method=='parallel')
            wrappers.append(dict(variant=method,site=dict(layout=layout,choice=c,request_wall_s=3.,decision_latency_s=2.,
                trigger=None,selected_lane='original',lanes=dict(alternative=None))))
            for i,r in enumerate(study.repeated_requests(c,method)):
                rows[r['id']]=outcome('direct' if i<2 else 'flight_miss')
        result=study.summarize([layout],{},rows,wrappers)
        self.assertEqual(result['methods']['baseline']['requested'],3)
        self.assertEqual(result['methods']['baseline']['releases'],0)
        self.assertEqual(result['methods']['parallel']['captures'],2)
        self.assertEqual(result['methods']['parallel']['reliable_positions'],0)

    def test_all_non_native_stages_are_explicit(self):
        names=['parent-study','parent-audit','source-study','source-audit','reference','assets','output','gpu-lock']
        for stage in ('initialize','prepare','preflight','run','finalize','audit'):
            a=study.argument_parser().parse_args([x for n in names for x in ('--'+n,'/fixture')]+['--stage',stage])
            self.assertEqual(a.stage,stage)
        self.assertEqual(study.GPU_CAP_S,1800)
