import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

from sircus.mjlab import neighborhood_contract as protocol
from sircus.mjlab import neighborhood_study as study
from sircus.mjlab.learning_contract import digest


def write(path,data):path.write_text(json.dumps(data))


def fixture(root):
    scene=dict(bucket={'center_xy_m':[.16275255472120964,-.6365356466995716]},ball={'radius_m':.0375})
    parents={}
    for kind in ('comparison','confirmation'):
        directory=root/kind;directory.mkdir();rows=[]
        for anchor in protocol.ANCHORS:
            if anchor['study']!=kind:continue
            rows.append(dict(id=anchor['record_id'],values=[.4708757793012148,-.2778934014509746,anchor['height'],anchor['tilt'],anchor['gap']],
                bucket_xy_m=scene['bucket']['center_xy_m'],q10=[0.]*6,q14=[0.]*6,native_release=True,route_success=True,invalid=''))
        spec=dict(scene=scene,packages={},asset_sha256={},source_sha256={});spec['fingerprint']=digest(spec)
        write(directory/'contract.json',spec);(directory/'records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        manifest={n:study.sha(directory/n) for n in ('contract.json','records.jsonl')};write(directory/'evidence-manifest.json',manifest)
        write(directory/'report.json',dict(status='completed',contract_fingerprint=spec['fingerprint'],evidence_sha256=manifest))
        write(directory/('audit-window.json' if kind=='comparison' else 'audit-window-confirmation.json'),dict(status='passed',contract_fingerprint=spec['fingerprint'],failures=[]))
        parents[kind]=directory
    return parents


class FakeGeometry:
    def __init__(self,scene,assets,anchors,bounds=None):self.anchors=anchors;self.contexts={}
    def prepare_contexts(self,contexts):
        self.contexts={r['id']:r for r in contexts}
        return [dict(**r,board_admitted=i!=0,reason='mock_reject' if i==0 else None,source_anchor_id=self.anchors[0]['id']) for i,r in enumerate(contexts)]
    def audit(self,ids,gaps):
        return [gate([.4708757793012148,-.2778934014509746,self.contexts[i]['board_height_m'],self.contexts[i]['tilt_deg'],g],i,not i.endswith('h00-t00')) for i,g in zip(ids,gaps)]


def gate(values,name,eligible=True):
    return dict(context_id=name,eligible=eligible,reason=None if eligible else 'mock_reject',values=values,
        bucket_xy_m=[.16275255472120964,-.6365356466995716],q10=[0.]*6,q14=[0.]*6,
        board_branch_sha256='mock',minimum_clearance_lower_bound_m=.01 if eligible else None)


class FakeEnvironment:
    def __init__(self,scene,assets,n,bounds,substeps):self.substeps=substeps;self.effective_options={'substeps':substeps}


def unresolved(bucket):
    return dict(invalid='mock',native_release=False,route_success=False,route_quality=0,board_bounce=False,
        closest_opening_distance_m=999.,max_retention_s=0.,external_contact_steps=0,single_bounce_success=False)


def fake_flight(env,bucket,values,q10,q14,eligible,record):
    n=int(sum(eligible));env.last_arrays=dict(position_m=np.zeros((960,n,3),np.float32),
        velocity_m_s=np.zeros((960,n,3),np.float32),forces_N=np.zeros((960,n,6),np.float32))
    return [dict(invalid='',native_release=True,route_success=True,route_quality=1,board_bounce=True,
        closest_opening_distance_m=.01,max_retention_s=1.,external_contact_steps=0,single_bounce_success=True,
        contact_protocol=dict(substeps=env.substeps,board_mode='articulated',gravity_mode='passive_only')) if e else unresolved(xy) for e,xy in zip(eligible,bucket)]


class NeighborhoodStudyTests(unittest.TestCase):
    def test_parent_tampering_and_unfinished_evidence_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            parents=fixture(Path(directory));scene,anchors,provenance,_=study.prepare_inputs(parents['comparison'],parents['confirmation'])
            self.assertEqual(len(anchors),3);self.assertEqual(len(provenance),2)
            p=parents['comparison']/'records.jsonl';p.write_text(p.read_text()+'\n')
            with self.assertRaises(ValueError):study.prepare_inputs(parents['comparison'],parents['confirmation'])

    def test_full_mock_preserves_rejected_attempts_and_identical_timestep_geometries(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);parents=fixture(root)
            args=SimpleNamespace(comparison_study=parents['comparison'],confirmation_study=parents['confirmation'],
                assets=root/'assets',output=root/'out',batch=64,wall_time_limit=60)
            def variations(scene,assets,nominal,requests,bounds):
                return [gate(r['values'],r['context_id'],r['repeat_index']!=3) for r in requests]
            modules={
                'sircus.mjlab.bank_study':SimpleNamespace(run_bank_rows=fake_flight,unresolved_row=unresolved),
                'sircus.mjlab.learning_solve':SimpleNamespace(required_asset_paths=lambda assets:[]),
                'sircus.mjlab.neighborhood_geometry':SimpleNamespace(NeighborhoodGeometry=FakeGeometry,audit_local_variations=variations),
                'sircus.mjlab.timestep_env':SimpleNamespace(TimestepEnvironment=FakeEnvironment,physics_settings=lambda step:{'substeps':step})}
            with patch.dict('sys.modules',modules),patch.object(study,'validate_packages',lambda p:p),contextlib.redirect_stdout(io.StringIO()):
                report=study._run(args)
            self.assertEqual(report['status'],'completed');self.assertEqual(len(report['contexts']),75)
            rows=study.lines(args.output/'records.jsonl');self.assertEqual(report['counts']['requests'],len(rows))
            self.assertLessEqual(len(rows),protocol.MAX_REQUESTS)
            self.assertEqual(sum(r['phase']=='coarse' for r in rows),1275)
            self.assertEqual(sum(r['role']=='control' for r in rows),27)
            for repeat in {r['repeat_id'] for r in rows if r['role']=='engineering_repeat'}:
                paired=[r for r in rows if r.get('repeat_id')==repeat]
                self.assertEqual(sorted(r['substeps'] for r in paired),[32,64,128])
                self.assertTrue(all(r['values']==paired[0]['values'] and r['q10']==paired[0]['q10'] and r['q14']==paired[0]['q14'] for r in paired))
                if paired[0]['repeat_index']==3:self.assertTrue(all(r['invalid'] and not r['native_release'] for r in paired))
            for name,expected in report['evidence_sha256'].items():self.assertEqual(study.sha(args.output/name),expected)
