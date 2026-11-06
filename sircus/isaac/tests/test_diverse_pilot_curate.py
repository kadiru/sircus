"""Synthetic curation-integrity checks; no training, Torch, physics or media QA."""

import copy
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import diverse_pilot_curate as module
from diverse_pilot_report import GLOBAL_QUALIFICATION_SCOPE, SETUP_CHECK_SCHEMA


def sign(value):
    value=copy.deepcopy(value)
    value["fingerprint"]=module.canonical_digest(value)
    return value


class CuratorTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.run=self.root/"native";self.run.mkdir()
        self.export=self.root/"export";self.export.mkdir()
        self.setup=self.root/"setup";self.setup.mkdir()
        self.out=self.root/"curated"
        def write(path,value):
            path.write_text(json.dumps(value,sort_keys=True,allow_nan=False)+"\n")
            return module.sha(path)
        self.write=write
        cells=[dict(id=f"c{i}",split="test" if i<5 else "validation" if i<10 else "train",bucket_xy_m=[i/10,0.]) for i in range(35)]
        contract=sign(dict(scene=dict(ball=dict(radius_m=.0375)),cells=cells));write(self.run/"contract.json",contract)
        support=dict(schema="sircus-diverse-shot-support-v2",policy_geometry_only=True,source_fingerprint=contract["fingerprint"],
            global_low=[-2,0,0,0,0],global_high=[2,1,1,1,1],local_region_count=35,global_region_count=35,region_count=70,cells=[])
        for i,c in enumerate(cells):
            anchor=dict(height_m=.3)
            support["cells"].append(dict(cell_id=c["id"],bucket_xy_m=c["bucket_xy_m"],status="supported",
                local_region_count=1,global_region_count=1,regions=[
                    dict(id=0,kind="local_geometry",proposal=i,interior_qualified=False,anchor_action=anchor,
                         low=[0,.3,.2,.1,.2],high=[.1,.4,.3,.2,.3],anchor_ik={"Arm10":{"q":[0]*6}}),
                    dict(id=1,kind="global_exploration",proposal=-1,interior_qualified=False,anchor_action=anchor,
                         low=support["global_low"],high=support["global_high"],qualification_scope=GLOBAL_QUALIFICATION_SCOPE)]))
        support=sign(support);write(self.run/"support.json",support)
        feasibility_sha=write(self.run/"feasibility.json",dict(geometry_only=True))
        protocol=dict(version="sircus-driven-settlement-v5",release_position_mm=2,release_joint_velocity_rad_s=.005)
        study=sign(dict(schema=module.STUDY_SCHEMA,cells=cells,geometry_fingerprint=contract["fingerprint"],
            source=dict(camera_sha256=module.CAMERA_SHA256),training_batches=64,coverage_batches=0,
            filmed_training_batches=[0],domain=sign(dict(names=["a","b","c","d","e"])),
            support_fingerprint=support["fingerprint"],support_sha256=module.sha(self.run/"support.json"),
            feasibility_sha256=feasibility_sha,settling_protocol=protocol))
        write(self.run/"study.json",study)
        result=dict(schema=module.STUDY_SCHEMA,status="completed",study_fingerprint=study["fingerprint"],
            geometry_fingerprint=study["geometry_fingerprint"],source_sha256={"settled_transit.py":"c"*64})
        result_sha=write(self.run/"result.json",result)
        launch=dict(source_commit=module.FINAL_PRODUCER,integrity=dict(archive_sha256=module.FINAL_ARCHIVE,
            source_members_sha256={"sircus/isaac/settled_transit.py":"c"*64}))
        for folder in (self.run,self.setup):
            write(Path(str(folder)+".launch.json"),launch)
            write(Path(str(folder)+".supervisor.json"),dict(source_commit=module.FINAL_PRODUCER,status="completed",returncode=0))
        model_shas={}
        for stage in ("initial","final"):
            p=self.run/f"policy-{stage}.pt";p.write_bytes(b"synthetic-checkpoint-"+stage.encode());model_shas[p.name]=module.sha(p)
        native=dict(status="passed",source_result_sha256=result_sha,all_cell_observations=123)
        checkpoint=dict(status="passed",source_result_sha256=result_sha,updates=16,optimizer_samples=1600,
            support_sha256=module.sha(self.run/"support.json"),checkpoints=model_shas)
        native_sha=write(self.run/"native-audit.json",native)
        cp_sha=write(self.run/"checkpoint-audit.json",checkpoint)
        feas_sha=write(self.run/"feasibility-audit.json",dict(status="passed",feasibility_sha256=feasibility_sha))
        paired=[dict(position_error_mm=.1,position_drift_mm=.01,joint_speed_rad_s=.001) for _ in range(100)]
        setup=dict(schema="sircus-fixed-request-setup-check-v1",status="completed",source_result_sha256=module.GUIDED_RESULT_SHA256,
            geometry_fingerprint=study["geometry_fingerprint"],settling_protocol=protocol,attempts=100,before_admitted=40,after_admitted=75,
            repaired=35,regressions=0,ball_releases=0,before_failures={"endpoints_not_settled":60},after_failures={"endpoints_not_settled":25},paired=paired)
        setup_sha=write(self.setup/"setup-check.json",setup)
        history_sha=write(self.setup/"setup-check-raw.json",dict(records=[{"private_readback":[1,2,3]}]))
        setup_audit={k:v for k,v in setup.items() if k!="paired"}
        setup_audit.update(schema="sircus-fixed-request-setup-audit-v1",status="passed",source_setup_sha256=setup_sha,source_raw_sha256=history_sha)
        self.audit_path=self.root/"setup-audit.json";setup_audit_sha=write(self.audit_path,setup_audit)
        attached=dict(schema=SETUP_CHECK_SCHEMA,status="completed",source_guided_result_sha256=module.GUIDED_RESULT_SHA256,
            geometry_fingerprint=study["geometry_fingerprint"],support_sha256=study["support_sha256"],support_fingerprint=support["fingerprint"],
            settling_protocol=protocol,source_setup_check_sha256=setup_sha,**{k:setup[k] for k in ("attempts","before_admitted","after_admitted","repaired","regressions","ball_releases","before_failures","after_failures")})
        attached_sha=write(self.run/"setup-check.json",attached)
        self.cost_path=self.root/"cost.json";cost=dict(support_sha256=study["support_sha256"],support_fingerprint=support["fingerprint"],maximum_resident_set_kib=123)
        cost_sha=write(self.cost_path,cost)
        camera=Path(module.__file__).resolve().parents[2]/"baselines/isaac-feasibility-guided-pilot-20261105/selected-camera.json"
        (self.run/"selected-camera.json").write_bytes(camera.read_bytes())
        write(self.run/"camera-binding.json",dict(private_readback=[1,2,3]))
        probes=[]
        for kind,target in (("bucket",1),("rim",1),("table",2),("floor",3),("robot",4)):
            peak=[0.]*6;peak[target]=1.
            probes.append(dict(kind=kind,all_passed=True,per_cell_passed=[True]*35,peak_N=[list(peak) for _ in range(35)]))
        probes.append(dict(kind="foreign_cell_collision_filter",all_passed=True,peak_N=[0.]*6,foreign_ball_final_z_m=-2.))
        write(self.run/"probes.json",probes)
        write(self.run/"rim-rays.json",dict(passed=[True]*35,rays=[dict(force_readback=[1,2,3])],clearance_m=.01))
        native_clip=dict(mode="train",batch=0,trial_file="trials/train-0000.json",camera_sha256=module.CAMERA_SHA256,
            frames=[dict(frame=0,phase="setup"),dict(frame=1,phase="release")])
        write(self.run/"clips.json",dict(clips=[native_clip]))
        def media(name,frames):
            p=self.export/name;p.write_bytes(b"synthetic-film-"+name.encode())
            return dict(file=name,sha256=module.sha(p),decode_verified=True,frames=frames,duration_s=frames/30,fps=30,resolution=[1920,1080])
        encoded=dict(media("trial-000.mp4",2),mode="train",batch=0,trial_file=native_clip["trial_file"],camera_sha256=module.CAMERA_SHA256,
            native_frame_audit=dict(frames=2,native_frame_sha256={"frame0":"x","frame1":"y"},episode_home=dict(private_native_joints=[0]*6)),
            prior_training=dict(completed_batches=0,attempts=0,batch_ids=[]),foreground=dict(outcome=dict(state="released_miss")))
        edited=dict(media("action-trial-000.mp4",1),original_clip="trial-000.mp4",original_frames=2,retained_native_global_frames=[1])
        movie=media("trials.mp4",2);action=dict(media("action-trials.mp4",1),clips=[edited])
        counts=lambda n:dict(attempts=n,captures=8,clean=0,rejected=2,released=n-2,capture_rate=8/n)
        methods={k:dict(source_result_sha256=v,training=dict(attempts=1600,captures=8,clean=0,rejected=2),frozen={s:dict(initial=counts(n),final=counts(n)) for s,n in (("train",200),("validation",40),("test",40))})
            for k,v in (("broad",module.BASELINE_RESULT_SHA256),("guided",module.GUIDED_RESULT_SHA256),("diverse",result_sha))}
        comparison=dict(schema=module.COMPARISON_SCHEMA,status="completed",methods=methods,scientific_scope="Synthetic curation test, not native performance.",
            geometry_preparation=dict(process_resource_receipt=dict(sha256=cost_sha)),region_usage=dict(status="logged",cells=[]),setup_check=dict(sha256=attached_sha))
        comparison_sha=write(self.run/"comparison-audit.json",copy.deepcopy(comparison))
        comparison["independent_audit"]=dict(sha256=comparison_sha,report=read(self.run/"comparison-audit.json"))
        buffer=io.StringIO();writer=csv.DictWriter(buffer,fieldnames=[*module.CSV_FIELDS,"measured_board_x_m","all_articulation_joint_velocity"]);writer.writeheader()
        for c in cells:
            row=dict.fromkeys(writer.fieldnames,"0");row.update(cell_id=c["id"],split=c["split"],bucket_x_m=str(c["bucket_xy_m"][0]),bucket_y_m="0",measured_board_x_m="private",all_articulation_joint_velocity="private")
            writer.writerow(row)
        (self.export/"settings.csv").write_bytes(buffer.getvalue().encode())
        report=dict(status="completed",study=study,study_fingerprint=study["fingerprint"],geometry_fingerprint=study["geometry_fingerprint"],
            source_sha256={"result.json":result_sha,"study.json":module.sha(self.run/"study.json")},
            training=dict(attempts=1600,captures=8,invalid=2),audits={k:dict(sha256=v) for k,v in (("native",native_sha),("checkpoint",cp_sha),("feasibility",feas_sha))},
            comparison=comparison,movie=movie,action_movie=action,clips=[encoded],elapsed_s=999,learning_curve=[],
            scoring="Synthetic unchanged scoring",uncertainty="Reported175mm interpretation remains provisional",
            settings_csv=dict(sha256=module.sha(self.export/"settings.csv")))
        report_sha=write(self.export/"report.json",report)
        self.kwargs=dict(setup_run=self.setup,setup_audit=self.audit_path,support_cost=self.cost_path,
            expected_report_sha256=report_sha,expected_native_audit_sha256=native_sha,expected_checkpoint_audit_sha256=cp_sha,
            expected_setup_audit_sha256=setup_audit_sha,auditor_commit="a"*40,exporter_commit="b"*40)

    def curate(self,**changes):
        return module.curate(self.run,self.export,self.out,**dict(self.kwargs,**changes))

    def test_complete_bundle_binds_unchanged_support_models_and_omits_native_history(self):
        manifest=self.curate()
        for name,record in manifest["files"].items():
            self.assertEqual(module.sha(self.out/name),record["sha256"])
            self.assertEqual((self.out/name).stat().st_size,record["bytes"])
        self.assertEqual((self.out/"support.json").read_bytes(),(self.run/"support.json").read_bytes())
        self.assertEqual((self.out/"policy-final.pt").read_bytes(),(self.run/"policy-final.pt").read_bytes())
        all_json="\n".join(p.read_text() for p in self.out.glob("*.json"))
        self.assertNotIn('"private_readback"',all_json)
        self.assertNotIn('"private_native_joints"',all_json)
        self.assertNotIn('"force_readback"',all_json)
        self.assertIn('"omitted_raw_field": true',all_json)
        self.assertNotIn("measured_board_x_m",(self.out/"requested-settings.csv").read_text())
        self.assertIn("Not task-shot or capture counts",(self.out/"contact-probes.json").read_text())
        contact=read(self.out/"contact-probes.json")
        self.assertEqual(contact["per_cell_check_count"],175)
        self.assertEqual(contact["single_isolation_probe_count"],1)
        isolation=next(r for r in contact["probes"] if r["kind"]=="foreign_cell_collision_filter")
        self.assertIsNone(isolation["per_cell_passed_count"])
        self.assertEqual(isolation["single_isolation_passed_count"],1)
        self.assertEqual(read(self.out/"setup-summary.json")["producer"]["source_commit"],module.FINAL_PRODUCER)

    def test_real_six_probe_shape_rejects_missing_duplicates_or_false_cell_flags(self):
        original=read(self.run/"probes.json")
        mutations=(
            lambda rows:rows.pop(),
            lambda rows:rows[0].update(kind="rim"),
            lambda rows:rows[0]["per_cell_passed"].pop(),
            lambda rows:rows[0]["per_cell_passed"].__setitem__(3,False),
            lambda rows:rows[0]["per_cell_passed"].__setitem__(3,1),
            lambda rows:rows[-1].update(per_cell_passed=[True]*35),
        )
        for mutate in mutations:
            rows=copy.deepcopy(original);mutate(rows);self.write(self.run/"probes.json",rows)
            with self.subTest(mutate=mutate),self.assertRaises(ValueError):module.contact_receipts(self.run)

    def test_contact_flags_cannot_hide_bad_target_attribution_or_foreign_collision(self):
        original=read(self.run/"probes.json")
        mutations=(
            lambda rows:rows[0]["peak_N"][0].__setitem__(1,.01),
            lambda rows:rows[2]["peak_N"][0].__setitem__(1,.02),
            lambda rows:rows[3]["peak_N"][0].__setitem__(0,.02),
            lambda rows:rows[-1]["peak_N"].__setitem__(1,.02),
            lambda rows:rows[-1].update(foreign_ball_final_z_m=-1.),
        )
        for mutate in mutations:
            rows=copy.deepcopy(original);mutate(rows);self.write(self.run/"probes.json",rows)
            with self.subTest(mutate=mutate),self.assertRaisesRegex(ValueError,"native force"):
                module.contact_receipts(self.run)

    def test_changed_model_bytes_or_final_report_fail_before_output_creation(self):
        (self.run/"policy-final.pt").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError,"Model bytes"):self.curate()
        self.assertFalse(self.out.exists())
        with self.assertRaisesRegex(ValueError,"Final report bytes"):self.curate(expected_report_sha256="0"*64)

    def test_old_preflight_producer_cannot_be_relabelled_as_final(self):
        p=Path(str(self.setup)+".launch.json");value=read(p);value["source_commit"]="5"*40;self.write(p,value)
        with self.assertRaisesRegex(ValueError,"producer identity"):self.curate()
        self.assertFalse(self.out.exists())

    def test_failed_audit_and_incomplete_process_fail_closed(self):
        p=Path(str(self.run)+".supervisor.json");value=read(p);value["status"]="failed";self.write(p,value)
        with self.assertRaisesRegex(ValueError,"complete successfully"):self.curate()
        with self.assertRaisesRegex(ValueError,"native audit mismatch"):
            value["status"]="completed";self.write(p,value)
            self.curate(expected_native_audit_sha256="0"*64)

    def test_release_frame_removed_in_edit_is_rejected(self):
        p=self.export/"report.json";report=read(p);report["action_movie"]["clips"][0]["retained_native_global_frames"]=[0]
        self.kwargs["expected_report_sha256"]=self.write(p,report)
        with self.assertRaisesRegex(ValueError,"release frame was removed"):self.curate()

    def test_raw_history_has_no_copy_endpoint_and_output_must_be_new(self):
        self.out.mkdir();(self.out/"unrelated").write_text("keep")
        with self.assertRaisesRegex(ValueError,"new directory"):self.curate()
        self.assertEqual((self.out/"unrelated").read_text(),"keep")
        with self.assertRaisesRegex(ValueError,"Unsafe evidence"):
            module.safe_file(self.run,"../outside")
        with self.assertRaisesRegex(ValueError,"Sensitive configuration"):
            module.inspect({"api_key":"do-not-publish"})

    def test_unknown_edited_frame_and_duplicate_json_are_rejected(self):
        p=self.export/"report.json";report=read(p)
        report["action_movie"]["clips"][0].update(retained_native_global_frames=[1,999],frames=2)
        self.kwargs["expected_report_sha256"]=self.write(p,report)
        with self.assertRaisesRegex(ValueError,"does not belong"):self.curate()
        duplicate=self.root/"duplicate.json";duplicate.write_text('{"status":"passed","status":"failed"}')
        with self.assertRaisesRegex(ValueError,"Duplicate JSON"):module.read(duplicate)

    def test_linked_models_are_rejected_before_copy(self):
        p=self.run/"policy-final.pt";payload=p.read_bytes();p.unlink()
        target=self.root/"outside.pt";target.write_bytes(payload);p.symlink_to(target)
        with self.assertRaisesRegex(ValueError,"linked evidence"):self.curate()
        self.assertFalse(self.out.exists())


def read(path):return json.loads(Path(path).read_text())


if __name__=="__main__":unittest.main()
