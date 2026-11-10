"""Independent far-shot plan, selection and pose-evidence tamper checks."""
import ast
import copy
import math
from pathlib import Path
import unittest
import numpy as np
from sircus.mjlab import yaw_tilt_gap_verify as V


class TargetTests(unittest.TestCase):
    def test_scalar_targets_match_geometry_and_keep_world_yaw(self):
        from sircus.mjlab.learning_ik import shot_targets
        scene=dict(ball=dict(radius_m=.0375),board=dict(mount_yaw_rad=math.pi/2,tool_offset_m=.12),arms=dict(base_spacing_m=1.33731))
        xyz=[.6536612,-.2034124,.4498183];bucket=[1.1070265,-.5362658];angles=[]
        for tilt in V.TILTS:
            for delta in V.GAP_OFFSETS:
                values=xyz+[tilt,V.SOURCE_GAP+delta];actual=V.expected_targets(scene,values,bucket)
                reference=shot_targets(scene,np.asarray([bucket]),np.asarray([xyz]),[tilt],[values[4]])
                for key in ('board_flange_local','release_center_m'):
                    np.testing.assert_allclose(actual[key],reference[key][0],rtol=0,atol=1e-14)
                np.testing.assert_allclose(actual['dropper_flange_world'][:3,3],reference['dropper_flange_world'][0,:3,3],atol=1e-14)
                angles.append(actual['yaw_rad']);np.testing.assert_array_equal(actual['board_world'][:3,3],xyz)
        self.assertEqual(len(set(angles)),1)
        self.assertAlmostEqual(math.degrees(angles[0]),-36.2855,places=3)

    def test_source_yaw_is_not_modulo_half_turn(self):
        scene=dict(ball=dict(radius_m=.0375),board=dict(mount_yaw_rad=0.,tool_offset_m=.12),arms=dict(base_spacing_m=1.3))
        actual=V.expected_targets(scene,[.6,0.,.4,30.,.1],[.3,-.3])
        self.assertAlmostEqual(actual['yaw_rad'],-3*math.pi/4)
        np.testing.assert_allclose(actual['rotation_matrix'][:,0],[-2**-.5,-2**-.5,0],atol=1e-14)

    def test_no_producer_or_online_scorer_imports(self):
        tree=ast.parse(Path(V.__file__).read_text())
        forbidden={'yaw_tilt_gap_native','yaw_tilt_gap_geometry','yaw_tilt_gap_study','yaw_tilt_gap_contract','table_assisted_score'}
        for node in ast.walk(tree):
            if isinstance(node,ast.ImportFrom):
                self.assertFalse(any(name in forbidden for name in [node.module or '']+[a.name for a in node.names]))


class SelectionTests(unittest.TestCase):
    def test_every_valid_capture_selected_in_grid_order(self):
        def row(i,**kw):
            return dict(id=str(i),grid_index=i,phase='discovery',context_id=V.FAR,native_release=True,
                        invalid_physics=False,route_success=True,**kw)
        rows=[row(10),row(2),row(3,invalid='invalid')]
        bad=row(4);bad['invalid_physics']=True;rows.append(bad)
        bad=row(5);bad['native_release']=False;rows.append(bad)
        bad=row(6);bad['context_id']=V.NEAR;rows.append(bad)
        bad=row(7);bad['phase']='selected_repeats';rows.append(bad)
        self.assertEqual([r['grid_index'] for r in V.selected_captures(rows)],[2,10])


def plan_fixture():
    from sircus.mjlab.tests.test_gripper_yaw_verify import fixtures
    from sircus.mjlab import yaw_tilt_gap_contract as C
    sources,controls,_=fixtures();previous=[]
    for source in sources:
        if source['context_id']==V.FAR and source['method']=='corrected_bc':
            source['values'][3]=V.TILTS[3];source['values'][4]=V.SOURCE_GAP
        if source['method']=='corrected_bc':
            old=copy.deepcopy(source);old.update(id='yaw-'+source['context_id'],source_record_id=source['id'],
                requested_gripper_yaw_rad=math.atan2(source['bucket_xy_m'][1]-source['values'][1],source['bucket_xy_m'][0]-source['values'][0]))
            previous.append(old)
    loaded=dict(sources=sources,controls=controls,yaw_rows=previous)
    return loaded,C.make_plan(loaded),C.literal_rule()


class FrozenPlanTests(unittest.TestCase):
    def test_all76_base_slots_and_constant_yaw(self):
        loaded,plan,rule=plan_fixture();V.audit_plan(plan,loaded['sources'],loaded['controls'],loaded['yaw_rows'],rule,V.replay.AuditChecks())
        grid=[r for r in plan if r['phase']=='discovery']
        self.assertEqual(len(grid),49);self.assertEqual(len({r['requested_gripper_yaw_rad'] for r in grid}),1)
        self.assertEqual(grid[21]['values'],loaded['sources'][1]['values'])

    def test_grid_and_reference_changes_rejected(self):
        loaded,plan,rule=plan_fixture()
        for index,key,value in ((18,'values',[.6,-.1,.46,24.,V.SOURCE_GAP]),(18,'substeps',128),
            (18,'requested_gripper_yaw_rad',0.),(18,'adapter_directory','native'),
            (18,'source_record_sha256','wrong'),(9,'variant','grid'),(67,'phase','references')):
            bad=copy.deepcopy(plan);bad[index][key]=value
            with self.subTest(key=key),self.assertRaises(V.EvidenceError):
                V.audit_plan(bad,loaded['sources'],loaded['controls'],loaded['yaw_rows'],rule,V.replay.AuditChecks())
        with self.assertRaises(V.EvidenceError):
            V.audit_plan(plan[:-1],loaded['sources'],loaded['controls'],loaded['yaw_rows'],rule,V.replay.AuditChecks())

    def test_selection_and_three_repeat_contract_match(self):
        from sircus.mjlab import yaw_tilt_gap_contract as C
        loaded,plan,rule=plan_fixture();rows=[]
        for item in (plan[20],plan[22]):
            row=copy.deepcopy(item);row.update(native_release=True,invalid_physics=False,route_success=True,
                connection_file='new/proof',connection_sha256='proof',q10=[0]*6,q14=[0]*6)
            rows.append(row)
        selection=V.expected_selection(rows);self.assertEqual(selection,C.select_captures(rows))
        repeats=V.expected_repeat_plan(plan,selection);self.assertEqual(repeats,C.selected_requests(plan,selection))
        self.assertEqual(len(repeats),6)
        self.assertEqual([r['repeat'] for r in repeats],[0,1,2,0,1,2])


def observed_fixture():
    loaded,plan,rule=plan_fixture();originals={s['id']:s for s in loaded['sources']+loaded['controls']};previous={s['id']:s for s in loaded['yaw_rows']}
    rows=[]
    for request in [r for r in plan if r['phase']!='controls_post']:
        row=copy.deepcopy(request);source=previous[request['frozen_yaw_record_id']] if request['frozen_yaw_record_id'] else originals[request['source_record_id']]
        for key in ('connection_file','connection_sha256','native_release','q10','q14'):
            row[key]=copy.deepcopy(source[key])
        if row['phase']=='discovery' and row['grid_index']!=21:
            row.update(connection_file='yaw-tilt-gap-connections/'+row['id']+'.json',connection_sha256=row['id'])
        row.update(invalid_physics=False,route_success=row.get('grid_index')==2)
        rows.append(row)
    selection=V.expected_selection(rows);repeats=V.expected_repeat_plan(plan,selection)
    byid={r['id']:r for r in rows}
    for request in repeats:
        row=copy.deepcopy(byid[request['selected_from_id']]);row.update(request);rows.append(row)
    for request in [r for r in plan if r['phase']=='controls_post']:
        row=copy.deepcopy(request);source=originals[row['source_record_id']]
        for key in ('connection_file','connection_sha256','native_release','q10','q14'):row[key]=copy.deepcopy(source[key])
        row.update(invalid_physics=False,route_success=False);rows.append(row)
    return loaded,plan,rows,selection,repeats


class ObservationTests(unittest.TestCase):
    def check(self,loaded,plan,rows,selection,repeats):
        V.audit_observations(rows,plan,loaded['sources'],loaded['controls'],loaded['yaw_rows'],selection,repeats,V.replay.AuditChecks())

    def test_preserved_reference_and_confirmations(self):
        loaded,plan,rows,selection,repeats=observed_fixture();self.check(loaded,plan,rows,selection,repeats)
        self.assertEqual(len(rows),79);self.assertEqual(len(repeats),3)

    def test_missing_or_changed_repeat_or_reference_fails(self):
        loaded,plan,rows,selection,repeats=observed_fixture()
        for mode in ('missing','reference','zero','newproof','repeat','bucket'):
            changed=copy.deepcopy(rows)
            if mode=='missing':changed.pop()
            elif mode=='reference':changed[9]['q10'][0]+=.1
            elif mode=='zero':changed[39]['connection_sha256']='new'
            elif mode=='newproof':changed[18]['connection_file']='connections/old'
            elif mode=='repeat':changed[67]['connection_sha256']='new'
            else:changed[18]['bucket_xy_m'][0]+=.1
            with self.subTest(mode=mode),self.assertRaises(V.EvidenceError):self.check(loaded,plan,changed,selection,repeats)

    def test_published_case_matrix_matches_independent_counts(self):
        from sircus.mjlab import yaw_tilt_gap_study as S
        loaded,plan,rows,selection,repeats=observed_fixture()
        for row in rows:
            row.update(scoring_version=V.oracle.VERSION,route_version=V.oracle.ROUTE_VERSION,reward_version=V.oracle.REWARD_VERSION,
                closest_opening_distance_m=.1,board_bounce=True,forbidden_contact_steps=0,max_retention_s=0.,
                route_quality=0,single_board_bounce_success=False,diagnostic_robot_contact_steps=0)
        self.assertEqual(V.independent_cases(plan,rows,selection),S.case_results(plan,rows,selection))
        self.assertEqual(V.statistics(rows),S.summarize(rows))

    def test_cherry_picked_or_reordered_confirmation_fails(self):
        loaded,plan,rows,selection,repeats=observed_fixture()
        with self.assertRaises(V.EvidenceError):self.check(loaded,plan,rows,[],[])
        with self.assertRaises(V.EvidenceError):self.check(loaded,plan,rows,selection,list(reversed(repeats)))


class ExtensionTests(unittest.TestCase):
    def fixture(self):
        from sircus.mjlab.yaw_tilt_gap_geometry import intervention
        from sircus.mjlab.calibrated_admission import CONNECTION_POLICY_VERSION
        loaded,plan,rule=plan_fixture();source=loaded['sources'][1];old=loaded['yaw_rows'][0]
        scene=dict(ball=dict(radius_m=.0375),board=dict(mount_yaw_rad=math.pi/2,tool_offset_m=.12),arms=dict(base_spacing_m=1.33731))
        values=source['values'][:3]+[26.,source['values'][4]+.06];ext=intervention(scene,source,old,values)
        connection=dict(values=values,bucket_xy_m=source['bucket_xy_m'],yaw_intervention=ext,connection_policy=CONNECTION_POLICY_VERSION,resolution_sha256='resolution')
        connection['yaw_key']=V.digest(dict(version=ext['version'],connection_policy=CONNECTION_POLICY_VERSION,resolution_sha256='resolution',values=values,bucket_xy_m=source['bucket_xy_m'],yaw_intervention=ext))
        return connection,source,old,scene

    def test_extension_independently_matches_full_target(self):
        connection,source,old,scene=self.fixture();targets,key=V.audit_new_connection(connection,source,old,scene,V.replay.AuditChecks(),'test')
        self.assertEqual(key,connection['yaw_key']);self.assertAlmostEqual(targets['release_center_m'][2],source['values'][2]+.0375+source['values'][4]+.06)

    def test_changed_source_yaw_center_target_or_key_rejected(self):
        connection,source,old,scene=self.fixture()
        for mode in ('key','board','yaw','source','target','flight','parameters'):
            changed=copy.deepcopy(connection);ext=changed['yaw_intervention']
            if mode=='key':changed['yaw_key']='fake'
            elif mode=='board':ext['fixed_board_center_m'][2]+=.01
            elif mode=='yaw':ext['requested_yaw_rad']+=math.pi
            elif mode=='source':ext['frozen_yaw_record_sha256']='fake'
            elif mode=='target':ext['requested_board_flange_local'][0][3]+=.01
            elif mode=='flight':ext['branch_selection_uses_flight_outcomes']=True
            else:ext['changed_parameter_indices']=[0,1,2,3,4]
            with self.subTest(mode=mode),self.assertRaises(V.EvidenceError):
                V.audit_new_connection(changed,source,old,scene,V.replay.AuditChecks(),'tamper')


if __name__=='__main__':unittest.main()
