"""Exact-shot protocol and native-orchestration mocks, never capture evidence."""
from collections import Counter
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

from sircus.mjlab import lowboard_inputs as inputs, lowboard_study as study
from sircus.mjlab import shallow_contract as protocol
from sircus.mjlab.tests.test_distance_contract import outcome, parent_inputs
from sircus.mjlab.tests.test_distance_study import gate, unresolved
from sircus.mjlab.tests.test_lowboard_study import fixture, MockGeometry, MockEnvironment


class ShallowContractTests(unittest.TestCase):
    def setUp(self):
        self.scene,self.seeds=parent_inputs()
        self.rays,self.targets=protocol.targets(self.scene,self.seeds)
        self.contexts=protocol.contexts(self.seeds,self.targets)
        self.dev=[c for c in self.contexts if c['stage']=='shallow']

    def test_exact_rounded_user_input_and_unchanged_four_targets(self):
        self.assertEqual([t['distance_m'] for t in self.targets],[.65,.70,.75,.77])
        self.assertEqual(len(self.contexts),12);self.assertEqual(len(self.dev),4)
        plan=protocol.coarse_plan(self.contexts,self.scene,self.seeds)
        self.assertEqual(len(plan),4)
        for c,p in zip(self.dev,plan):
            self.assertEqual((c['board_height_m'],c['tilt_deg'],p['gap_m']),(.380,28.,.3868))
            self.assertAlmostEqual(p['requested_ball_center_height_m'],.8043,15)
            self.assertAlmostEqual(p['release_headroom_m'],.0150243380486807,15)
            self.assertNotEqual(p['gap_m'],.3868243380486807)
        self.assertEqual(protocol.settings(self.scene,self.seeds)['candidate']['gap_m'],.3868)
        self.assertEqual(len(protocol.control_plan(self.seeds,self.targets)),24)
        self.assertEqual(protocol.MAX_REQUESTS,24+4+4*33*3)
        with self.assertRaises(ValueError):protocol.coarse_plan(self.contexts)

    def test_only_actual_exact_captures_get_fresh_seeded_paired_repeats(self):
        rows=[outcome(c,protocol.GAP,hit=i%2==0,suffix=str(i)) for i,c in enumerate(self.dev)]
        self.assertEqual(protocol.refinement_plan(self.contexts,rows),[])
        recipes=protocol.select_recipes(rows,self.targets)
        self.assertEqual(len(recipes),2)
        requests=protocol.repeat_plan(recipes)
        self.assertEqual(requests,protocol.repeat_plan(recipes));self.assertEqual(len(requests),66)
        self.assertEqual(Counter(r['phase'] for r in requests),{'nominal':6,'fresh':60})
        self.assertEqual(protocol.select_recipes([dict(r,route_success=False) for r in rows],self.targets),[])
        self.assertEqual(protocol.repeat_plan([]),[])
        self.assertEqual(protocol.SEED,20261208)
        for recipe in recipes:
            group=[r for r in requests if r['recipe_id']==recipe['id']]
            self.assertTrue(all(r['values']==recipe['source']['values'] for r in group[:3]))
            self.assertEqual(len({tuple(r['values']) for r in group}),31)
        altered=copy.deepcopy(rows);altered[0]['values'][4]+=.0001
        with self.assertRaises(ValueError):protocol.select_recipes(altered,self.targets)

    def run_mock(self,root,hit):
        parent,spec=fixture(root)
        args=NS(parent_study=parent,assets=root/'assets',output=root/'out',batch=64,wall_time_limit=120)
        def variations(scene,assets,source,requests,bounds):
            return [gate(scene,source,dict(id=r['context_id'],bucket_xy_m=source['bucket_xy_m'],
                board_height_m=r['values'][2],tilt_deg=r['values'][3]),r['gap_m'],eligible=r['repeat_index']!=3) for r in requests]
        def flight(env,bucket,values,q10,q14,eligible,record):
            n=int(sum(eligible));env.last_arrays={k:np.zeros((960,n,width),np.float32)
                for k,width in [('position_m',3),('velocity_m_s',3),('forces_N',6)]}
            reference=np.asarray(self.seeds[0]['bucket_xy_m']);rows=[]
            for ok,xy in zip(eligible,bucket):
                if not ok:rows.append(unresolved(xy));continue
                capture=bool(hit or np.linalg.norm(xy-reference)<1e-10)
                rows.append(dict(invalid=None,native_release=True,route_success=capture,route_quality=int(capture),
                    board_bounce=True,whole_ball_entry=capture,closest_opening_distance_m=.01 if capture else .3,
                    max_retention_s=float(capture),external_contact_steps=0,single_bounce_success=capture))
            return rows
        modules={'sircus.mjlab.bank_study':NS(run_bank_rows=flight,unresolved_row=unresolved),
            'sircus.mjlab.learning_solve':NS(required_asset_paths=lambda _:[]),
            'sircus.mjlab.distance_geometry':NS(DistanceGeometry=MockGeometry,audit_distance_variations=variations),
            'sircus.mjlab.timestep_env':NS(TimestepEnvironment=MockEnvironment,physics_settings=lambda s:dict(substeps=s))}
        MockEnvironment.allocated=[]
        with patch.dict('sys.modules',modules),patch.object(inputs,'PARENT_FINGERPRINT',spec['fingerprint']),patch.object(study,'validate_packages',lambda p:p),contextlib.redirect_stdout(io.StringIO()):
            report=study._run(args,protocol=protocol)
        rows=[json.loads(line) for line in (args.output/'records.jsonl').read_text().splitlines()]
        return report,rows

    def test_mock_capture_path_preserves_all_paired_joints_and_rejects(self):
        with tempfile.TemporaryDirectory() as directory:
            report,rows=self.run_mock(Path(directory),True)
        self.assertEqual(report['version'],protocol.VERSION);self.assertEqual(len(rows),424)
        self.assertEqual(len(report['selected_recipes']),4)
        self.assertEqual(sorted(MockEnvironment.allocated),[32,64,128])
        self.assertEqual([r['id'] for r in rows],[f'shallow-shot-{i:06d}' for i in range(424)])
        self.assertEqual(Counter(r['phase'] for r in rows),{'controls':24,'coarse':4,'nominal':36,'fresh':360})
        for repeat in {r['repeat_id'] for r in rows if r['role']=='engineering_repeat'}:
            group=[r for r in rows if r.get('repeat_id')==repeat]
            self.assertEqual([r['substeps'] for r in group],[32,64,128])
            self.assertTrue(all(all(r[k]==group[0][k] for k in ('values','bucket_xy_m','q10','q14','geometry')) for r in group))
            if group[0]['repeat_index']==3:self.assertTrue(all(not r['native_release'] for r in group))

    def test_mock_misses_keep_four_saved_flights_and_do_not_claim_fine_repeats(self):
        with tempfile.TemporaryDirectory() as directory:
            report,rows=self.run_mock(Path(directory),False)
        self.assertEqual(len(rows),28);self.assertEqual(report['selected_recipes'],[])
        self.assertEqual(MockEnvironment.allocated,[64]);self.assertEqual(len(report['trace_examples']),4)
        self.assertTrue(all(not t['route_success'] for t in report['trace_examples']))
        self.assertEqual(report['counts']['captures'],24)


if __name__=='__main__':unittest.main()
