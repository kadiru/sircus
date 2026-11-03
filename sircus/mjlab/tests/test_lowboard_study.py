"""Orchestration/ancestry mocks only; none of these outcomes are physics evidence."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np

from sircus.mjlab import lowboard_contract as protocol, lowboard_inputs as inputs, lowboard_study as study
from sircus.mjlab import distance_contract as parent_protocol
from sircus.mjlab.tests.test_distance_contract import parent_inputs
from sircus.mjlab.tests.test_distance_study import FakeGeometry, gate, unresolved
from sircus.mjlab.learning_contract import digest


def write(path, value):
    path.write_text(json.dumps(value))


def fixture(root):
    scene, seeds = parent_inputs()
    for seed in seeds:seed.update(q10=[0.]*6,q14=[1.]*6)
    parent=root/'parent';parent.mkdir()
    spec=dict(version='sircus-bucket-distance-development-v1',scene=scene,seeds=seeds,
        settings=parent_protocol.settings(scene,seeds),source_sha256={},asset_sha256={},packages={})
    spec['fingerprint']=digest(spec);write(parent/'contract.json',spec)
    rows=[]
    for s in seeds:
        for i in range(3):rows.append(dict(copy.deepcopy(s),id=f"source-{s['id']}-{i}",role='control',
            source_anchor_id=s['id'],native_release=True,route_success=True,invalid=None))
    (parent/'records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    manifest={n:study.sha(parent/n) for n in ('contract.json','records.jsonl')}
    write(parent/'evidence-manifest.json',manifest)
    write(parent/'report.json',dict(status='completed',phase='completed',contract_fingerprint=spec['fingerprint'],evidence_sha256=manifest))
    write(parent/'audit-distance.json',dict(status='passed',contract_fingerprint=spec['fingerprint'],
        checks_count=1,checks=[dict(passed=True)],failures=[]))
    return parent,spec


class MockGeometry(FakeGeometry):
    def audit(self, ids, gaps):
        return [gate(self.scene,self.seeds[self.contexts[name]['source_anchor_id']],self.contexts[name],gap,
            eligible=self.contexts[name]['board_height_m']+gap+self.scene['ball']['radius_m']<.81)
            for name,gap in zip(ids,gaps)]


class MockEnvironment:
    allocated=[]
    def __init__(self,scene,assets,n,bounds,substeps):
        self.substeps=substeps;self.effective_options=dict(substeps=substeps);self.allocated.append(substeps)


def flight(allow_hits):
    def run(env,bucket,values,q10,q14,eligible,record):
        n=int(sum(eligible));env.last_arrays={k:np.zeros((960,n,w),np.float32)
            for k,w in [('position_m',3),('velocity_m_s',3),('forces_N',6)]}
        rows=[]
        reference=np.asarray([.16275255472120964,-.6365356466995716])
        for ok,xy,v in zip(eligible,bucket,values):
            if not ok:rows.append(unresolved(xy));continue
            control=np.linalg.norm(xy-reference)<1e-10
            hit=bool(control or allow_hits and abs(v[3]-36)<.8 and min(abs(v[2]-.30),abs(v[2]-.32))<.005)
            rows.append(dict(invalid=None,native_release=True,route_success=hit,route_quality=int(hit),
                board_bounce=True,whole_ball_entry=hit,closest_opening_distance_m=.01 if hit else .3,
                max_retention_s=float(hit),external_contact_steps=0,single_bounce_success=hit))
        return rows
    return run


class LowboardInputTests(unittest.TestCase):
    def test_exact_publication_exception_cannot_hide_native_source_change(self):
        name,(old,new)=next(iter(inputs.PUBLICATION_OVERRIDE.items()))
        self.assertIn(name,inputs.source_compatibility({name:new,'physics':'same'},{name:old,'physics':'same'}))
        for actual in ({name:'other','physics':'same'},{name:new,'physics':'changed'},{name:new}):
            with self.assertRaises(ValueError):inputs.source_compatibility(actual,{name:old,'physics':'same'})

    def test_parent_checks_raw_hashes_full_audit_and_exact_controls(self):
        for kind in ('good','raw','empty-audit','truthy-audit','wrong-control'):
            with tempfile.TemporaryDirectory() as directory:
                parent,spec=fixture(Path(directory))
                if kind=='raw':(parent/'records.jsonl').write_text('[]\n')
                elif kind in ('empty-audit','truthy-audit'):
                    a=study.read(parent/'audit-distance.json');a['checks']=[] if kind=='empty-audit' else [{'passed':1}];write(parent/'audit-distance.json',a)
                elif kind=='wrong-control':
                    rows=[json.loads(s) for s in (parent/'records.jsonl').read_text().splitlines()];rows[0]['q10'][0]=.1
                    (parent/'records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
                    m=study.read(parent/'evidence-manifest.json');m['records.jsonl']=study.sha(parent/'records.jsonl');write(parent/'evidence-manifest.json',m)
                    r=study.read(parent/'report.json');r['evidence_sha256']=m;write(parent/'report.json',r)
                with patch.object(inputs,'PARENT_FINGERPRINT',spec['fingerprint']):
                    if kind=='good':
                        scene,seeds,provenance,meta,_=inputs.prepare_inputs(parent)
                        self.assertEqual(len(seeds),8);self.assertEqual(len(provenance[0]['parent_control_record_ids']),3)
                    else:
                        with self.subTest(kind=kind),self.assertRaises(ValueError):inputs.prepare_inputs(parent)


class LowboardStudyTests(unittest.TestCase):
    def run_mock(self,root,allow_hits):
        parent,spec=fixture(root);args=NS(parent_study=parent,assets=root/'assets',output=root/'out',batch=64,wall_time_limit=120)
        def variations(scene,assets,nominal,requests,bounds):
            source={k:nominal[k] for k in ('values','bucket_xy_m','q10','q14')}
            return [gate(scene,source,dict(id=r['context_id'],bucket_xy_m=source['bucket_xy_m'],
                board_height_m=r['values'][2],tilt_deg=r['values'][3]),r['gap_m'],eligible=r['repeat_index']!=3) for r in requests]
        modules={'sircus.mjlab.bank_study':NS(run_bank_rows=flight(allow_hits),unresolved_row=unresolved),
            'sircus.mjlab.learning_solve':NS(required_asset_paths=lambda _:[]),
            'sircus.mjlab.distance_geometry':NS(DistanceGeometry=MockGeometry,audit_distance_variations=variations),
            'sircus.mjlab.timestep_env':NS(TimestepEnvironment=MockEnvironment,physics_settings=lambda s:dict(substeps=s))}
        MockEnvironment.allocated=[]
        with patch.dict('sys.modules',modules),patch.object(inputs,'PARENT_FINGERPRINT',spec['fingerprint']),patch.object(study,'validate_packages',lambda p:p),contextlib.redirect_stdout(io.StringIO()):
            report=study._run(args)
        rows=[json.loads(x) for x in (args.output/'records.jsonl').read_text().splitlines()]
        return args,report,rows

    def test_full_mock_keeps_near_limit_rejections_and_paired_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            args,report,rows=self.run_mock(Path(directory),True)
            self.assertEqual(report['status'],'completed');self.assertEqual(len(report['targets']),4)
            self.assertEqual(sum(r['phase']=='controls' for r in rows),24)
            self.assertEqual(sum(r['phase']=='coarse' for r in rows),640)
            self.assertEqual(len(report['selected_recipes']),8);self.assertEqual(len(report['trace_examples']),20)
            self.assertLessEqual(len(rows),1968);self.assertEqual(sorted(MockEnvironment.allocated),[32,64,128])
            self.assertEqual(sum(r['phase']=='coarse' and not r['eligible'] for r in rows),128)
            for identity in {r['repeat_id'] for r in rows if r['role']=='engineering_repeat'}:
                rr=[r for r in rows if r.get('repeat_id')==identity]
                self.assertEqual(sorted(r['substeps'] for r in rr),[32,64,128])
                self.assertTrue(all(all(r[k]==rr[0][k] for k in ('values','bucket_xy_m','q10','q14')) for r in rr))
                if rr[0]['repeat_index']==3:self.assertTrue(all(not r['native_release'] and r['invalid'] for r in rr))
            for trace in report['trace_examples']:
                row=next(r for r in rows if r['id']==trace['record_id'])
                self.assertEqual(trace['context_id'],row['context_id']);self.assertEqual(trace['bucket_xy_m'],row['bucket_xy_m'])
            for name,expected in report['evidence_sha256'].items():self.assertEqual(study.sha(args.output/name),expected)

    def test_zero_capture_search_keeps_observed_misses_and_only_used_options(self):
        with tempfile.TemporaryDirectory() as directory:
            args,report,rows=self.run_mock(Path(directory),False)
            self.assertEqual(report['selected_recipes'],[]);self.assertEqual(MockEnvironment.allocated,[64])
            self.assertEqual(len(report['trace_examples']),4);self.assertTrue(all(not r['route_success'] for r in report['trace_examples']))
            self.assertEqual(sum(r['role']=='engineering_repeat' for r in rows),0)
            self.assertEqual(report['counts']['requests'],664)


if __name__=='__main__':unittest.main()
