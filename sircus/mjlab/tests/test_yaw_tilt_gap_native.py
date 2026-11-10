"""CPU checks for frozen geometry-only sweep, unchanged references and exact repeats."""
import copy
import math
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch
import numpy as np
from sircus.mjlab import yaw_tilt_gap_contract as C,yaw_tilt_gap_geometry as G,yaw_tilt_gap_native as N,yaw_tilt_gap_pipeline as P
from sircus.mjlab.table_correction_native import META
from sircus.mjlab.benchmark_verify import digest

SCENE={'board':{'mount_yaw_rad':math.pi/2,'tool_offset_m':.12},'arms':{'base_spacing_m':1.2,'board_base_yaw_rad':math.pi},'ball':{'radius_m':.0375}}


def fixture():
    sources=[];controls=[];yaw=[]
    for i,context in enumerate((*C.ROLES,C.FAR,C.NEAR)):
        for method in (('control',) if i<3 else ('current_bc','corrected_bc')):
            r={k:None for k in META};r.update(id=f'{context}-{method}',context_id=context,phase='test',role=context if i<3 else 'test',method=method,
                values=[.6536612003245014,-.20341247447399055,.4498183398237823,30.33068546648235,.07564961424222355],
                bucket_xy_m=[1.1070265259583456,-.53626584251982],q10=[.1]*6,q14=[.2]*6,native_release=True,invalid_physics=False)
            (controls if i<3 else sources).append(r)
            if method=='corrected_bc':
                yaw.append(dict(r,id='yaw-'+r['id'],source_record_id=r['id'],requested_gripper_yaw_rad=-.6333,q10=[.3]*6))
    return dict(sources=sources,controls=controls,yaw_rows=yaw)


class PlanTests(unittest.TestCase):
    def test49_grid_and27_fixed_references_without_source_mutation(self):
        loaded=fixture();before=copy.deepcopy(loaded);plan=C.make_plan(loaded)
        self.assertEqual(loaded,before);self.assertEqual(len(plan),76);self.assertEqual(len({r['id'] for r in plan}),76)
        self.assertEqual([r['phase'] for r in plan],['controls_pre']*9+['references']*9+['discovery']*49+['controls_post']*9)
        grid=[r for r in plan if r['variant']=='grid'];source=loaded['sources'][1]
        self.assertEqual([r['grid_index'] for r in grid],list(range(49)))
        self.assertEqual({r['values'][3] for r in grid},set(C.TILTS))
        self.assertEqual({r['values'][4] for r in grid},{source['values'][4]+g for g in C.GAP_OFFSETS})
        for r in grid:
            self.assertEqual(r['values'][:3],source['values'][:3]);self.assertEqual(r['bucket_xy_m'],source['bucket_xy_m'])
            self.assertEqual(r['requested_gripper_yaw_rad'],-.6333);self.assertEqual(r['substeps'],64)
            self.assertEqual(r['frozen_yaw_record_sha256'],digest(loaded['yaw_rows'][0]));self.assertEqual(r['parameter_indices'],[3,4])
        self.assertEqual(sum(r['values']==source['values'] for r in grid),1)

    def test_literal_rule_and_budget(self):
        rule=C.literal_rule();self.assertEqual(C.validate_rule(rule),rule);self.assertEqual(C.INPUT_KEYS,P.INPUT_KEYS)
        for key,value in (('substeps',128),('noise',True),('training',True),('context',C.NEAR),('gap_offsets_m',[0,.02])):
            with self.subTest(key=key),self.assertRaises(ValueError):C.validate_rule(dict(rule,**{key:value}))
        self.assertEqual(C.MAXIMUM_REQUESTS,76+49*3);self.assertEqual(C.protocol()['maximum_acquired_s'],3600)

    def test_selection_excludes_failed_invalid_refused_and_repeats_all_captures(self):
        plan=C.make_plan(fixture());grid=[r for r in plan if r['variant']=='grid'];rows=[]
        for i,r in enumerate(grid[:6]):
            rows.append(dict(r,native_release=i!=1,invalid_physics=i==2,invalid='bad' if i==3 else None,
                route_success=i!=4,connection_file='proof.json',connection_sha256='proof',q10=[.1]*6,q14=[.2]*6))
        selected=C.select_captures(list(reversed(rows)));self.assertEqual([r['grid_index'] for r in selected],[0,5])
        repeats=C.selected_requests(plan,selected);self.assertEqual(len(repeats),6)
        self.assertEqual([r['repeat'] for r in repeats],[0,1,2,0,1,2])
        for row in repeats:
            source=plan[[r['id'] for r in plan].index(row['selected_from_id'])]
            for k in ('values','bucket_xy_m','requested_gripper_yaw_rad','source_record_id','grid_index'):
                self.assertEqual(row[k],source[k])
        self.assertEqual(C.selected_requests(plan,[]),[])

    def test_request_mutations_block_before_native(self):
        loaded=fixture();plan=C.make_plan(loaded);worker=object.__new__(N.SweepEngine)
        worker.plan={r['id']:r for r in plan};worker.sources={r['id']:r for r in loaded['sources']};worker.namespace='native-yaw'
        worker.engine=SimpleNamespace(evaluate=Mock());row=next(r for r in plan if r['variant']=='grid')
        for key,value in (('requested_gripper_yaw_rad',0.),('values',[.7,-.2,.45,30,.08]),('substeps',128),('adapter_directory','native')):
            with self.subTest(key=key),self.assertRaises(ValueError):worker.evaluate([dict(row,**{key:value})],'discovery')
        worker.engine.evaluate.assert_not_called()


class GeometryTests(unittest.TestCase):
    def test_targets_hold_yaw_and_center_while_tilt_gap_change(self):
        loaded=fixture();source=loaded['sources'][1];old=loaded['yaw_rows'][0]
        a=G.intervention(SCENE,source,old,source['values']);v=list(source['values']);v[3]+=4;v[4]+=.06
        b=G.intervention(SCENE,source,old,v)
        self.assertEqual(a['requested_yaw_rad'],b['requested_yaw_rad']);self.assertEqual(a['requested_rotation_matrix'],b['requested_rotation_matrix'])
        self.assertEqual(b['fixed_board_center_m'],source['values'][:3]);self.assertFalse(b['board_branch_fixed'])
        self.assertAlmostEqual(b['release_center_m'][2]-a['release_center_m'][2],.06)
        self.assertNotEqual(a['requested_board_flange_local'],b['requested_board_flange_local']);self.assertNotEqual(digest(a),digest(b))
        self.assertFalse(b['branch_selection_uses_flight_outcomes'])

    def test_rotation_evidence_rejects_bad_board_or_dropper_pose(self):
        loaded=fixture();s=loaded['sources'][1];_,targets=G.yaw_target(SCENE,s['bucket_xy_m'],s['values'])
        r=dict(s,adapter_directory='native-yaw',native_initial_q10=[.1]*6,native_initial_q14=[.2]*6)
        board=G.base14(SCENE)@targets['board_flange_local'];drop=targets['dropper_flange_world']
        with patch.object(G,'body_transform',side_effect=[drop,board]):
            proof=G.rotation_evidence(SCENE,None,r);self.assertLess(proof['rotation_error_rad'],1e-9)
        for role in ('board','dropper'):
            wrong=copy.deepcopy(board if role=='board' else drop);wrong[2,3]+=.001
            with patch.object(G,'body_transform',side_effect=[wrong if role=='dropper' else drop,wrong if role=='board' else board]),self.assertRaises(RuntimeError):
                G.rotation_evidence(SCENE,None,r)

    def test_unflown_has_no_fabricated_observed_pose(self):
        r=dict(fixture()['sources'][1],adapter_directory='native-yaw',native_release=False)
        self.assertNotIn('measured_initial_rotation_matrix',G.rotation_evidence(SCENE,None,r))

    def test_exact_yaw_cache_is_returned_without_new_solver(self):
        loaded=fixture();source=loaded['sources'][1];plan=C.make_plan(loaded);proof=dict(passed=True,q10=[.4]*6,q14=[.2]*6)
        key=digest(dict(bucket_xy_m=source['bucket_xy_m'],values=source['values']))
        base=SimpleNamespace(sources=[],home={},_check_resolution_bytes=Mock())
        resolver=G.TiltGapResolver(base,loaded['sources'],loaded['yaw_rows'],plan,{key:proof})
        anchor={'bucket_xy_m':source['bucket_xy_m']}
        with patch.object(G,'verify_resolution'):
            result=resolver.connect(anchor,source['values'])
        self.assertTrue(result['cache_hit']);self.assertEqual(result['q10'],proof['q10']);self.assertEqual(proof,dict(passed=True,q10=[.4]*6,q14=[.2]*6))

if __name__=='__main__':unittest.main()
