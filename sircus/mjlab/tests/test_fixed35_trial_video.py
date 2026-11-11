"""Movie coverage, exact raw row addressing, time bounds and no physics stepping."""
import ast
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from sircus.mjlab import fixed35_trial_video as V


def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(data))


def fixture(root):
    assets=root/'assets';assets.mkdir()
    for name in ('dropper.urdf','board_arm.urdf'):(assets/name).write_text('<robot><link name="root"/></robot>')
    write(assets/'poses.json',{})
    repository=Path(V.__file__).resolve().parents[2];scene={'fixture':'same geometry'}
    inventory=dict(cohorts={},records=[])
    times=np.arange(1537)*V.DT;position=np.zeros((len(times),1,3));position[:,:,2]=.4
    for name in V.COHORTS:
        study=root/name;study.mkdir();cache=study/'native-yaw/one.npz';cache.parent.mkdir()
        row=dict(id=name+'-a',context_id=V.CONTEXTS[0],bucket_xy_m=V.cell_xy(V.CONTEXTS[0]),native_release=True,
            contact_protocol={'substeps':64},scoring_version=V.SCORER,adapter_directory='native-yaw',native_cache_file='one.npz',
            native_cache_row=0,native_initial_ball_center_m=[0.,0.,.4],native_initial_q10=[0]*6,native_initial_q14=[0]*6,
            physics_steps=1200,values=[.5,0,.3,28,.2])
        np.savez_compressed(cache,record_ids=np.array([row['id']]),time_s=times,position_m=position,forces_N=np.zeros((len(times)-1,1,6)))
        row['native_cache_sha256']=V.sha(cache)
        declined=dict(id=name+'-b',context_id=V.CONTEXTS[1],bucket_xy_m=V.cell_xy(V.CONTEXTS[1]),native_release=False,invalid='finite_ik_unresolved')
        (study/'rows.jsonl').write_text('\n'.join(json.dumps(r) for r in (row,declined))+'\n')
        contract=dict(protocol={'substeps':64},scene=scene,identity=dict(trace_schema={'native_dt_s':V.DT},scene_sha256=V.digest(scene),
            source_sha256={key:V.sha(repository/key) for key in V.SOURCE_MODULES},
            asset_sha256={str(p.relative_to(root)):V.sha(p) for p in assets.iterdir()}))
        contract['fingerprint']=V.digest(contract);write(study/'contract.json',contract)
        manifest={p:V.sha(study/p) for p in ('rows.jsonl','contract.json','native-yaw/one.npz')};write(study/'evidence-manifest.json',manifest)
        report=dict(status='completed',evidence_sha256=manifest,contract_fingerprint=contract['fingerprint']);write(study/'report.json',report)
        audit=dict(status='passed',full_checks_verified=True,native_checks_verified=True,failures=[],checks_count=1,checks=[{'passed':True}],
            report_sha256=V.sha(study/'report.json'),evidence_manifest_sha256=V.sha(study/'evidence-manifest.json'),contract_fingerprint=contract['fingerprint'])
        auditpath=root/(name+'-audit.json');write(auditpath,audit)
        inventory['cohorts'][name]=dict(path=str(study),audit_path=str(auditpath),report_sha256=V.sha(study/'report.json'),
            contract_sha256=V.sha(study/'contract.json'),manifest_sha256=V.sha(study/'evidence-manifest.json'),audit_sha256=V.sha(auditpath))
        for i,r in enumerate((row,declined),1):
            inventory['records'].append(dict(cohort=name,row_id=r['id'],row_line=i,row_sha256=V.digest(r),context_id=r['context_id'],
                native=dict(relative_path='native-yaw/one.npz',column=0,sha256=V.sha(cache)) if r['native_release'] else None))
    path=root/'inventory.json';write(path,inventory)
    return path,assets


class IntegrityTests(unittest.TestCase):
    def test_all_cohort_rows_are_covered_and_exact_native_column_loaded(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,assets=fixture(Path(tmp));data=V.load_inventory(path,assets)
            self.assertEqual(len(data['records']),8)
            record=data['records'][0];bundle=V.load_trace(record,data['sources'][record['cohort']],set())
            np.testing.assert_array_equal(bundle['positions'][0],[0.,0.,.4]);self.assertEqual(bundle['scored_until_s'],1200*V.DT)
    def test_omission_duplication_row_metadata_and_decline_flight_are_rejected(self):
        for mode in ('omit','duplicate','line','rowhash','column','decline'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as tmp:
                path,assets=fixture(Path(tmp));j=V.read(path)
                if mode=='omit':j['records'].pop()
                if mode=='duplicate':j['records'].append(j['records'][0])
                if mode=='line':j['records'][0]['row_line']=2
                if mode=='rowhash':j['records'][0]['row_sha256']='changed'
                if mode=='column':j['records'][0]['native']['column']=1
                if mode=='decline':j['records'][1]['native']={}
                write(path,j)
                with self.assertRaises(ValueError):V.load_inventory(path,assets)
    def test_changed_cache_is_rejected_without_executing_renderer(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,assets=fixture(Path(tmp));data=V.load_inventory(path,assets);record=data['records'][0];source=data['sources'][record['cohort']]
            cache=source['study']/record['native']['relative_path'];cache.write_bytes(cache.read_bytes()+b'changed')
            with self.assertRaisesRegex(ValueError,'raw archive changed'):V.load_trace(record,source,set())


class TimelineTests(unittest.TestCase):
    def test_end_tick_included_and_never_samples_after_original_terminal(self):
        times=np.arange(29248)*V.DT;ticks=V.frame_ticks(times,24)
        self.assertEqual(ticks[0],0);self.assertEqual(ticks[-1],29247);self.assertEqual(max(ticks),29247)
        self.assertTrue(all(times[i]<=times[-1] for i in ticks))
        self.assertLessEqual(len(ticks)/24-times[-1],2/24)
    def test_contact_between_frames_not_lost_and_exact_threshold(self):
        times=np.arange(20)*V.DT;forces=np.zeros((19,6));forces[3,0]=.011;forces[5,4]=.02;forces[6,1]=.01
        intervals=V.contact_intervals(times,forces)
        self.assertEqual(intervals['board'][0]['first_event_s'],times[4]);self.assertEqual(intervals['board'][0]['start_s'],times[3])
        self.assertEqual(V.active_contacts(intervals,0.,times[10]),['board','robot']);self.assertEqual(intervals['bucket'],[])
        self.assertEqual(V.active_contacts(intervals,times[10],times[19]),[])
    def test_contact_starting_at_score_terminal_is_not_displayed(self):
        times=np.arange(6)*V.DT;forces=np.zeros((5,6));forces[2,0]=1.;forces[3,4]=1.
        intervals=V.contact_intervals(times,forces)
        self.assertEqual(V.active_contacts(intervals,times[2],times[3]),['board'])
        self.assertEqual(V.active_contacts(intervals,-1.,0.),[])

    def test_invalid_trial_not_relabelled_success(self):
        self.assertIn('INVALID',V.outcome({'route_success':True,'invalid_physics':True}))
        self.assertEqual(V.outcome({'route_success':False}),'NO CAPTURE')
    def test_code_has_no_physics_step_or_interpolation_call(self):
        tree=ast.parse(Path(V.__file__).read_text())
        names=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
        self.assertFalse({'mj_step','mj_step1','mj_step2','interp'} & set(names));self.assertIn('mj_forward',names)
    def test_cpu_mesa_egl_requires_explicit_vendor_before_gl_import(self):
        env=dict(CUDA_VISIBLE_DEVICES='',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',LIBGL_ALWAYS_SOFTWARE='1',
            MESA_LOADER_DRIVER_OVERRIDE='llvmpipe',__EGL_VENDOR_LIBRARY_FILENAMES=V.MESA_VENDOR)
        with patch.object(V,'read',return_value={'ICD':{'library_path':'libEGL_mesa.so.0'}}),patch.object(V,'sha',return_value='vendor-sha'):
            self.assertEqual(V.validate_render_environment(env)['vendor_library'],'libEGL_mesa.so.0')
            for key in env:
                missing=dict(env);missing.pop(key)
                with self.subTest(key=key),self.assertRaises(ValueError):V.validate_render_environment(missing)
        with patch.object(V,'read',return_value={'ICD':{'library_path':'libEGL_nvidia.so.0'}}):
            with self.assertRaises(ValueError):V.validate_render_environment(env)
        self.assertEqual(V.validate_render_environment({'CUDA_VISIBLE_DEVICES':'','MUJOCO_GL':'osmesa'})['requested_backend'],'osmesa')

    def test_gpu_backend_and_visible_cuda_are_rejected(self):
        self.assertTrue(V.software_backend('llvmpipe','Mesa','')['software_only'])
        for gl,cuda in (('NVIDIA RTX4090',''),('llvmpipe','0')):
            with self.assertRaises(ValueError):V.software_backend(gl,'Mesa',cuda)


if __name__=='__main__':unittest.main()
