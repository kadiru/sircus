"""Matched denominator, new positions, and frozen evidence controls."""
import copy
import json
import importlib.util
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from sircus.mjlab import board_fallback_study as study
from sircus.mjlab.benchmark_verify import sha
from sircus.mjlab.tests.test_stalled_reference_study import choice, outcome
from sircus.mjlab.corrected_contract import grid_layouts

def provenance():
    grid={r["id"]:r["bucket_xy_m"] for r in grid_layouts()}
    return dict(excluded_coordinates_m=list(grid.values()),grid=grid,bounds=[[.1,1.24],[-.65,.25]],
        geometry=dict(fixed_arm_bounds={"Arm10":dict(center_xy_m=[0.,0.],radius_m=.09),
            "Arm14":dict(center_xy_m=[1.15,0.],radius_m=.09)},bucket_radius_m=.075,
            tabletop_xy_bounds_m=[[-.2,-.9],[1.5,.5]]))
from sircus.mjlab.planning_budget_plan import _wrap_choice


class BoardFallbackStudyTests(unittest.TestCase):
    def test_fifty_fresh_paired_layouts_are_deterministic_and_separated(self):
        inputs=provenance(); saved=copy.deepcopy(inputs)
        rows=study.make_layouts(inputs)
        self.assertEqual(inputs,saved);self.assertEqual(rows,study.make_layouts(inputs))
        self.assertEqual(len(rows),50);self.assertEqual(len({r['id'] for r in rows}),50)
        known=list(inputs['excluded_coordinates_m'])
        for row in rows:
            self.assertEqual(row['split'],'test');self.assertEqual(row['context_id'],row['id'])
            self.assertGreaterEqual(min(math.dist(row['bucket_xy_m'],p) for p in known),.005)
            known.append(row['bucket_xy_m'])
        self.assertEqual(sum('fixed base' in r['region'] for r in rows),10)
        self.assertEqual(sum('grid edge' in r['region'] for r in rows),20)
        self.assertNotEqual(rows,study.make_layouts(inputs,study.SEED+1))

    @unittest.skipIf(importlib.util.find_spec("torch") is None,"Existing summary dependencies require pinned Ubuntu Torch")
    def test_no_setup_and_misses_remain_in_paired_denominator(self):
        layouts=[dict(id=f'test-{i}',bucket_xy_m=[.3+i*.1,-.3]) for i in range(4)]
        slots=[];rows={};wrappers=[]
        for layout,labels in zip(layouts,[('direct','table'),(None,'direct'),('direct','flight_miss'),('flight_miss',None)]):
            for method,label in zip(study.METHODS,labels):
                inner=choice(layout,label is not None)
                slot=_wrap_choice(inner,method);slots.append(slot)
                if label is not None:rows[slot['id']]=outcome(label)
                wrappers.append(dict(variant=method,site=dict(layout=layout,choice=inner,request_wall_s=3.,decision_latency_s=2.)))
        result=study.summarize(layouts,slots,rows,wrappers)
        self.assertEqual(result['paired'],dict(both=1,gained=1,lost=1,neither=1))
        for method in study.METHODS:
            stats=result['methods'][method]
            self.assertEqual((stats['positions'],stats['admitted'],stats['no_candidate'],stats['captures']),(4,3,1,2))

    def test_fallback_counts_count_takeover_separately_from_changed_choice(self):
        wrappers=[dict(variant='baseline',site={})]
        for triggered,selected in [(False,None),(True,None),(True,'request-1')]:
            wrappers.append(dict(variant='fallback',site=dict(board_fallback=dict(triggered=triggered,selected_fallback_candidate_id=selected))))
        self.assertEqual(study.fallback_counts(wrappers),dict(activated=2,changed_choice=1))

    def test_recovery_binding_rejects_modified_preparation(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name); study_dir=root/'source';study_dir.mkdir()
            def put(path,value):path.write_text(json.dumps(value))
            put(study_dir/'report.json',dict(status='completed'))
            put(study_dir/'preparation.json',dict(cases=[]));put(study_dir/'contract.json',dict(version='fixture'))
            put(study_dir/'evidence-manifest.json',{n:sha(study_dir/n) for n in ('contract.json','preparation.json')})
            receipt=dict(status='passed',report_sha256=sha(study_dir/'report.json'),plan_sha256=sha(study_dir/'preparation.json'),
                         evidence_manifest_sha256=sha(study_dir/'evidence-manifest.json'))
            audit=root/'audit.json';put(audit,receipt)
            self.assertEqual(study.recovery_binding(study_dir,audit)['audit'],sha(audit))
            put(study_dir/'preparation.json',dict(cases=['added']))
            with self.assertRaises(ValueError):study.recovery_binding(study_dir,audit)

    def test_contract_check_rejects_any_frozen_input_mutation(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);f=root/'model.json';f.write_text('{}')
            spec=dict(frozen_inputs={'model.json':sha(f)},source_sha256={})
            spec['fingerprint']=study.digest(spec)
            (root/'contract.json').write_text(json.dumps(spec))
            self.assertEqual(study.frozen_check(SimpleNamespace(output=root)),spec)
            f.write_text('{"weights":1}')
            with self.assertRaises(ValueError):study.frozen_check(SimpleNamespace(output=root))


if __name__=='__main__':unittest.main()
