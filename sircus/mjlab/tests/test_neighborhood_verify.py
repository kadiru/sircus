import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest

import numpy as np

from sircus.mjlab import neighborhood_contract as protocol
from sircus.mjlab.admission import VERIFIED_PARK_Q10
from sircus.mjlab.contract import FORCE_NAMES, ScoreStream
from sircus.mjlab.neighborhood_verify import (audit_contexts,audit_rows,audit_plans,audit_native,audit_options,inside)
from sircus.mjlab.timestep_env import validate_options
from sircus.mjlab.verify_expansion import Audit,digest
from sircus.mjlab.window_verify import row_reward


def result(a):return {c['name']:c['passed'] for c in a.checks}
def write(root,name,value):(root/name).write_text(json.dumps(value))


def geometry_fixture():
    settings=protocol.settings();anchors=[]
    for i,anchor in enumerate(protocol.ANCHORS):
        anchors.append(dict(id=anchor['id'],values=settings['board_xy_m']+[anchor['height'],anchor['tilt'],anchor['gap']],
            bucket_xy_m=settings['bucket_xy_m'],q10=[.1*i]*6,q14=[float(i+1)]*6))
    spec=dict(settings=settings,scene={},anchors=anchors);prepared=[]
    for c in protocol.contexts():
        nearest=min(anchors,key=lambda a:(float(np.sum(((np.asarray([c['board_height_m'],c['tilt_deg']])-np.asarray(a['values'][2:4]))/[.04,4.])**2)),a['id']))
        p=dict(c,prepared=True,board_admitted=True,q14=nearest['q14'].copy(),reason=None,board_world=np.eye(4).tolist(),
            source_anchor_id=nearest['id'],source_anchor_geometry_sha256=digest(nearest))
        p['board_branch_sha256']=digest(dict(version='sircus-fixed-board-gap-window-geometry-v1',scene_sha256=digest({}),
            bucket_xy_m=settings['bucket_xy_m'],board_world=p['board_world'],q14=p['q14'],park_q10=list(VERIFIED_PARK_Q10)))
        prepared.append(p)
    c=prepared[0];values=settings['board_xy_m']+[c['board_height_m'],c['tilt_deg'],.3]
    g=dict(context_id=c['id'],values=values,bucket_xy_m=settings['bucket_xy_m'],q10=[0.]*6,q14=c['q14'].copy(),eligible=True,
        board_branch_sha256=c['board_branch_sha256'],minimum_clearance_lower_bound_m=.006,
        wrapper_version='sircus-multi-anchor-height-tilt-geometry-v1',source_anchor_id=c['source_anchor_id'],
        source_anchor_geometry_sha256=c['source_anchor_geometry_sha256'])
    row=dict(g,id='neighborhood-shot-000000',geometry=copy.deepcopy(g),context_id=c['id'],role='development',phase='coarse',gap_m=.3,
        native_release=True,invalid=None,route_success=False,route_quality=0,board_bounce=True,
        whole_ball_entry=False,max_retention_s=0.,external_contact_steps=0,closest_opening_distance_m=.3,single_bounce_success=False)
    row['learning_reward']=row_reward(row)
    return spec,prepared,[row]


class NeighborhoodGeometryAuditTests(unittest.TestCase):
    def test_all75_source_bound_contexts_and_rejected_contexts_remain_visible(self):
        spec,prepared,_=geometry_fixture();prepared[-1].update(board_admitted=False,q14=None,board_branch_sha256=None,reason='unresolved')
        a=Audit();audit_contexts(a,spec,prepared);self.assertTrue(all(result(a).values()),result(a))

    def test_omitted_context_or_wrong_anchor_or_joint_hash_is_rejected(self):
        for kind in ('omitted','anchor','joint'):
            spec,prepared,_=geometry_fixture()
            if kind=='omitted':prepared.pop()
            elif kind=='anchor':prepared[0]['source_anchor_id']='other'
            else:prepared[0]['q14'][0]+=.001
            a=Audit();audit_contexts(a,spec,prepared)
            with self.subTest(kind=kind):self.assertFalse(all(result(a).values()))

    def test_valid_rows_are_exactly_bound_to_context_and_geometry(self):
        spec,prepared,rows=geometry_fixture();a=Audit();audit_rows(a,spec,prepared,rows,[],[],dict(requests=[],rows=[]))
        self.assertTrue(all(result(a).values()),result(a))

    def test_changed_sweep_board_branch_fails_even_if_inline_geometry_is_changed(self):
        spec,prepared,rows=geometry_fixture();rows[0]['q14'][0]+=.001;rows[0]['geometry']['q14']=rows[0]['q14'].copy()
        a=Audit();audit_rows(a,spec,prepared,rows,[],[],dict(requests=[],rows=[]))
        self.assertFalse(result(a)['exact_fixed_xy_request_geometry_and_local_branch_identity'])

    def test_false_release_or_low_clearance_or_changed_reward_fails(self):
        for kind in ('release','clearance','reward','role'):
            spec,prepared,rows=geometry_fixture();r=rows[0]
            if kind=='release':r['eligible']=r['geometry']['eligible']=False
            elif kind=='clearance':r['minimum_clearance_lower_bound_m']=r['geometry']['minimum_clearance_lower_bound_m']=.0049
            elif kind=='reward':r['learning_reward']+=.01
            else:r['role']='test'
            a=Audit();audit_rows(a,spec,prepared,rows,[],[],dict(requests=[],rows=[]))
            with self.subTest(kind=kind):self.assertFalse(all(result(a).values()))

    def test_native_evidence_paths_cannot_escape_declared_roots(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):inside(directory,'../outside')


def plan_fixture(root):
    spec,_,_=geometry_fixture();anchors=spec['anchors'];controls=[];base=[];rows=[]
    contexts=protocol.contexts()
    for anchor in anchors:
        c=next(c for c in contexts if [c['board_height_m'],c['tilt_deg']]==anchor['values'][2:4])
        for repeat in range(3):base.append(dict(role='control',phase='controls',anchor_id=anchor['id'],context_id=c['id'],source_family=c['source_family'],gap_m=anchor['values'][4],repeat_index=repeat))
    for step in (32,64,128):
        for p in base:
            anchor=next(a for a in anchors if a['id']==p['anchor_id'])
            controls.append(dict(p,**{k:copy.deepcopy(anchor[k]) for k in ('values','bucket_xy_m','q10','q14')},id=f'source-{len(controls)}',substeps=step,
                native_release=True,invalid=None,route_success=True,route_quality=1,board_bounce=True,single_bounce_success=True,max_retention_s=1.,external_contact_steps=0,closest_opening_distance_m=.01))
    rows.extend(controls);coarse=protocol.coarse_plan()
    for p in coarse:rows.append(dict(p,id=f'source-{len(rows)}',substeps=64,native_release=False,invalid='unresolved',route_success=False,route_quality=0))
    recipes=protocol.select_recipes(rows);repeat_plan=protocol.repeat_plan(recipes)
    for step in (32,64,128):
        for p in repeat_plan:rows.append(dict(p,id=f'source-{len(rows)}',substeps=step,native_release=False,invalid='unresolved',route_success=False,route_quality=0))
    for name,value in (('control-plan.json',base),('coarse-plan.json',coarse),('refinement-plan.json',[]),('frozen-repeat-plan.json',repeat_plan)):write(root,name,value)
    return spec,rows,recipes


class NeighborhoodBudgetAuditTests(unittest.TestCase):
    def test_no_hit_coarse_contexts_have_no_adaptive_search_and_still_keep_all1275_attempts(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);spec,rows,recipes=plan_fixture(root);a=Audit();audit_plans(a,root,spec,rows,recipes)
            self.assertTrue(all(result(a).values()),result(a))

    def test_silent_geometry_rejection_omission_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);spec,rows,recipes=plan_fixture(root);rows.pop(500);a=Audit();audit_plans(a,root,spec,rows,recipes)
            self.assertFalse(result(a)['complete_1275_coarse_requests']);self.assertFalse(result(a)['no_retry_resampling_or_silent_budget_reduction'])

    def test_repeat_noise_is_frozen_and_cannot_be_reselected_after_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);spec,rows,recipes=plan_fixture(root);plan=json.loads((root/'frozen-repeat-plan.json').read_text());plan[4]['values'][2]+=.001;write(root,'frozen-repeat-plan.json',plan)
            a=Audit();audit_plans(a,root,spec,rows,recipes)
            self.assertFalse(result(a)['all_seeded_joint_noise_requests_frozen_before_outcomes'])

    def test_duplicate_repeat_cannot_replace_an_omitted_request(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);spec,rows,recipes=plan_fixture(root);rows[-1]=copy.deepcopy(rows[-2]);a=Audit();audit_plans(a,root,spec,rows,recipes)
            self.assertFalse(result(a)['no_retry_resampling_or_silent_budget_reduction'])


class NeighborhoodNativeAuditTests(unittest.TestCase):
    def fixture(self,substeps=128):
        scene=json.loads((Path(__file__).resolve().parents[2]/'isaac/scene.json').read_text());positions=np.tile([.5,-.2,-.3],(960,1,1));velocities=np.zeros_like(positions);forces=np.zeros((960,1,6))
        scorer=ScoreStream(scene,[.16,-.64],[0,0,1]);scorer.step(positions[0,0],velocities[0,0],dict(zip(FORCE_NAMES,forces[0,0])))
        row=dict(scorer.result(),id='native',bucket_xy_m=[.16,-.64],board_normal=[0,0,1],native_release=True,invalid=None,substeps=substeps,
            contact_protocol=dict(substeps=substeps,board_mode='articulated',gravity_mode='passive_only'))
        return scene,[row],dict(record_ids=np.asarray(['native']),position_m=positions,velocity_m_s=velocities,forces_N=forces)

    def test_scalar_score_is_recomputed_for_each_supported_timestep(self):
        for step in (32,64,128):
            scene,rows,arrays=self.fixture(step);a=Audit();audit_native(a,scene,rows,arrays,step,'native')
            with self.subTest(step=step):self.assertTrue(all(result(a).values()),result(a))

    def test_substituted_ids_protocol_false_capture_and_altered_arrays_are_rejected(self):
        for kind in ('id','protocol','capture','trace','shape'):
            scene,rows,arrays=self.fixture()
            if kind=='id':arrays['record_ids']=np.asarray(['other'])
            elif kind=='protocol':rows[0]['contact_protocol']['substeps']=64
            elif kind=='capture':rows[0]['route_success']=True
            elif kind=='shape':arrays['position_m']=arrays['position_m'][:959]
            else:arrays['position_m'][:,:,2]=.5
            a=Audit();audit_native(a,scene,rows,arrays,128,'native')
            with self.subTest(kind=kind):self.assertFalse(all(result(a).values()))


class NeighborhoodEffectiveOptionsTests(unittest.TestCase):
    def fixture(self,root):
        result={}
        for step in (32,64,128):
            native=NS(timestep=1/(240*step),tolerance=1e-8,gravity=[0,0,-9.81],solver=2,integrator=3,cone=0,iterations=100)
            warp=NS(timestep=np.asarray([native.timestep],np.float32),tolerance=np.asarray([1e-6],np.float32),gravity=np.asarray([[0,0,-9.81]],np.float32),solver=2,integrator=3,cone=0,iterations=100)
            result[str(step)]=validate_options(native,warp,step)
        write(root,'effective-options.json',result);return dict(effective_native_options=copy.deepcopy(result))

    def test_every_native_and_warp_option_is_independently_reconstructed(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);report=self.fixture(root);a=Audit();audit_options(a,root,report);self.assertTrue(all(result(a).values()),result(a))

    def test_wrong_report_and_unrounded_warp_timestep_cannot_be_hidden_by_float32_cast(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);report=self.fixture(root);options=read=json.loads((root/'effective-options.json').read_text());options['128']['warp']['timestep_s'][0]=1/30720;report['effective_native_options']=options;write(root,'effective-options.json',options)
            a=Audit();audit_options(a,root,report);self.assertFalse(result(a)['actual_native_and_warp_options_128'])
            report=self.fixture(root);report['effective_native_options']['32']['tolerance']=1e-4;a=Audit();audit_options(a,root,report)
            self.assertFalse(result(a)['all_effective_options_retained'])

    def test_wrong_native_or_warp_tolerance_cannot_be_hidden_in_matching_report(self):
        for kind,value in (('native',1e-6),('warp',1e-8),('warp',2e-6)):
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory);report=self.fixture(root);options=json.loads((root/'effective-options.json').read_text())
                if kind=='native':options['64']['tolerance']=value
                else:options['64']['warp']['tolerance'][0]=float(np.float32(value))
                report['effective_native_options']=copy.deepcopy(options);write(root,'effective-options.json',options)
                a=Audit();audit_options(a,root,report)
                with self.subTest(kind=kind,value=value):self.assertFalse(result(a)['actual_native_and_warp_options_64'])


if __name__=='__main__':unittest.main()
