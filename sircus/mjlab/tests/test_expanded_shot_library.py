from copy import deepcopy
import unittest
import json
from pathlib import Path
import tempfile

from sircus.mjlab import expanded_shot_library as library


def observation(identifier, *, capture=True, invalid=False, robot=False, qshift=0., seedshift=0.):
    request=dict(id=identifier,values=[.5,-.2,.3,28.,.35],bucket_xy_m=[.7,-.3],
                 board_heading_offset_deg=0.,board_twist_deg=0.,intended_route="direct",
                 seed_q10=[seedshift,0.,0.,0.,0.,0.],seed_q14=[0.]*6)
    native=dict(q10=[qshift,1.,2.,3.,4.,5.],q14=[0.,1.,2.,3.,4.,5.],
                release_ball_center_m=[.5,-.2,.6875],contact_protocol=library.PROTOCOL)
    score=dict(route_success=capture,invalid_physics=invalid,
               route_class="robot_assisted" if capture and robot else "direct" if capture else "not_captured",
               table_assisted_capture=False,route_quality=2)
    evidence=dict(native_sha256=identifier,native_row=0,capture=capture,invalid=invalid,
                  route=score["route_class"],historical_phase="test")
    return dict(request=request,native=native,score=score,evidence=evidence,
                provenance=dict(scene_sha256=library.NOMINAL_SCENE,
                                calibration_fingerprint=library.NOMINAL_CALIBRATION))


class ExpandedLibraryTests(unittest.TestCase):
    def test_build_binds_real_packet_and_rejects_changed_source(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            study=root/"source";study.mkdir()
            neural=root/"neural";neural.mkdir()
            rescore=root/"rescore";rescore.mkdir()
            validation_xy=next([.1+i*.01,.05] for i in range(100)
                if library.split_for([.1+i*.01,.05])=="validation")
            def dump(path,value):
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text(json.dumps(value))
            items=[];raws=[];enriched=[];packets=[]
            for i,xy in enumerate([[.7,-.3],validation_xy]):
                obs=observation(str(i));req=obs["request"];req["bucket_xy_m"]=xy
                native=dict(obs["native"],id=str(i),bucket_xy_m=xy,values=req["values"],
                            native_cache_sha256=str(i),native_cache_row=0)
                raws.append(native);packets.append(req)
                item=dict(id=str(i),status="scored",substeps=64,original=native,
                          provenance=obs["provenance"],new_score=dict(obs["score"],
                              scoring_version=library.board_first_score.VERSION),
                          source_study="source",native_sha256=str(i),native_row=0,phase="test")
                items.append(item)
            dump(study/"requests/one.json",dict(rows=packets))
            packet_sha=library.file_sha(study/"requests/one.json")
            for raw in raws:
                enriched.append(dict(raw,request_packet="requests/one.json",request_packet_sha256=packet_sha))
            for name,rows in [("shots.jsonl",raws),("rows.jsonl",enriched)]:
                (study/name).write_text("".join(json.dumps(row)+"\n" for row in rows))
            dump(study/"evidence-manifest.json",{name:library.file_sha(study/name)
                for name in ["rows.jsonl","requests/one.json"]})
            for i,item in enumerate(items):
                item["source"]=dict(file="source/shots.jsonl",line=i+1,
                                    metadata_sha256=library.file_sha(study/"shots.jsonl"))
            (rescore/"records.jsonl").write_text("".join(json.dumps(item)+"\n" for item in items))
            dump(rescore/"contract.json",{})
            dump(rescore/"summary.json",{})
            dump(rescore/"manifest.json",dict(artifacts={name:library.file_sha(rescore/name)
                for name in ["records.jsonl","contract.json"]}))
            dump(rescore/"audit.json",dict(status="passed",full_checks_verified=True,
                 manifest_sha256=library.file_sha(rescore/"manifest.json"),
                 summary_sha256=library.file_sha(rescore/"summary.json")))
            for name in ["training-dataset.jsonl","validation-dataset.jsonl"]:(neural/name).write_text("")
            dump(neural/"evidence-manifest.json",{name:library.file_sha(neural/name)
                for name in ["training-dataset.jsonl","validation-dataset.jsonl"]})
            summary=library.build(rescore,neural,root/"out",root=root,forced_train_positions=[[.7,-.3]])
            self.assertEqual(summary["included_native_attempts"],2)
            self.assertEqual(summary["candidate_settings"],1)
            self.assertEqual(library.verify_library(root/"out"),summary)
            (study/"requests/one.json").write_text("{}");
            with self.assertRaisesRegex(ValueError,"source checksum"):
                library.verify_library(root/"out")

    def test_nominal_orientation_checks_full_quaternion_not_only_normal(self):
        import math
        request=dict(values=[0.,0.,.3,0.,.2],bucket_xy_m=[1.,0.])
        scene=dict(board=dict(mount_yaw_rad=0.))
        native=dict(board_quaternion_wxyz=[math.sqrt(.5),0.,0.,math.sqrt(.5)])
        self.assertLess(library.nominal_orientation_error(request,native,scene),1e-7)
        # Both normals are vertical, but the board's in-plane twist differs.
        native["board_quaternion_wxyz"]=[1.,0.,0.,0.]
        self.assertAlmostEqual(library.nominal_orientation_error(request,native,scene),math.pi/2)

    def test_repeats_are_one_setting_with_fractional_target(self):
        a=observation("a");b=observation("b",capture=False)
        rows,sources=library.group_native([a,b],forced_train_positions=[[.7,-.3]])
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]["capture_probability"],.5)
        self.assertEqual(rows[0]["valid_native_count"],2)
        self.assertEqual(len(sources),1)
        self.assertEqual(sources[0]["capture_probability"],.5)

    def test_invalid_repeat_does_not_become_negative_label(self):
        rows,_=library.group_native([observation("a"),observation("b",capture=False,invalid=True)])
        self.assertEqual(rows[0]["capture_probability"],1.)
        self.assertEqual(rows[0]["invalid_native_count"],1)

    def test_duplicate_native_evidence_is_not_repeat(self):
        a=observation("a")
        with self.assertRaisesRegex(ValueError,"twice"):
            library.group_native([a,deepcopy(a)])

    def test_different_robot_pose_remains_different_setting(self):
        rows,_=library.group_native([observation("a"),observation("b",qshift=.3)])
        self.assertEqual(len(rows),2)

    def test_known_robot_contact_capture_is_kept(self):
        rows,sources=library.group_native([observation("a",robot=True)],forced_train_positions=[[.7,-.3]])
        self.assertEqual(rows[0]["capture_probability"],1.)
        self.assertEqual(sources[0]["observed_route"],"robot_assisted")
        self.assertEqual(sources[0]["intended_route"],"direct")
        self.assertNotEqual(sources[0]["observed_route"],sources[0]["intended_route"])

    def test_query_solved_joints_never_become_model_seed(self):
        a=observation("a",qshift=2.,seedshift=.3)
        rows,sources=library.group_native([a],forced_train_positions=[[.7,-.3]])
        self.assertEqual(rows[0]["request"]["seed_q10"][0],.3)
        self.assertEqual(sources[0]["seed_q10"][0],.3)
        self.assertEqual(sources[0]["q10"][0],2.)

    def test_canonical_seed_is_deterministic_and_aliases_counted(self):
        a=observation("a",seedshift=.3);b=observation("b",seedshift=.7)
        rows,_=library.group_native([b,a],forced_train_positions=[[.7,-.3]])
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]["pre_ik_feature_variants"],2)
        self.assertEqual(rows[0]["request"]["seed_q10"][0],.3)

    def test_no_validation_source_enters_candidate_library(self):
        a=observation("a")
        for seed in range(100):
            if library.split_for(a["request"]["bucket_xy_m"],seed=seed)=="validation":break
        rows,sources=library.group_native([a],seed=seed)
        self.assertEqual(rows[0]["split"],"validation")
        self.assertEqual(sources,[])

    def test_periodic_joint_representations_are_not_merged(self):
        import math
        rows,_=library.group_native([observation("a"),observation("b",qshift=2*math.pi)])
        self.assertEqual(len(rows),2)

    def test_release_point_part_of_setting_identity(self):
        a=observation("a");b=observation("b")
        b["native"]["release_ball_center_m"][0]+=.01
        rows,_=library.group_native([a,b])
        self.assertEqual(len(rows),2)

    def test_source_request_does_not_infer_intent_from_observed_route(self):
        a=observation("a",robot=True)
        a["request"]["intended_route"]="table"
        rows,sources=library.group_native([a],forced_train_positions=[[.7,-.3]])
        self.assertEqual(rows[0]["request"]["intended_route"],"table")
        self.assertEqual(sources[0]["intended_route"],"table")


if __name__=="__main__":unittest.main()
