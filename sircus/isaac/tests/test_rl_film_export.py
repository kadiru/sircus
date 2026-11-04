"""CPU-only checks for truthful training-time overlays and native film lineage."""

import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ISAAC=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ISAAC))
import rl_film_export as film


def row(captured=False,invalid=None):
    return dict(invalid=invalid,route_success=captured,route_quality=2 if captured else 0,
        single_bounce_success=captured,external_contact_steps=0,board_bounce=True,
        closest_opening_distance_m=.02,max_retention_s=1. if captured else 0.)


class FilmTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.rows=[[row(captured=i<(batch%35),invalid="guard" if i==34 else None)
                    for i in range(35)]for batch in range(128)]
        self.curve=film.rates(self.rows)
        self.assessments=[dict(after_train_batches=n,policy_update=n//2,attempts=280,
            captures=n,capture_rate=n/280)for n in film.ASSESSMENT_STAGES]

    def chapters(self,count=80):
        return [dict(batch=batch,cycle=i,source_start_frame=i*(count+20),
            source_end_frame_inclusive=i*(count+20)+count-1,source_frames=count,
            route_qualified_capture=False)for i,batch in enumerate(film.FILMED)]

    def test_rates_count_all_attempts_and_keep_rejected_releases_in_denominator(self):
        self.assertEqual(self.curve[34]["captures"],34)
        self.assertEqual(self.curve[34]["attempts"],35)
        self.assertEqual(self.curve[34]["invalid"],1)
        self.assertAlmostEqual(self.curve[34]["batch_capture_rate"],34/35)
        self.assertEqual(self.curve[-1]["cumulative_attempts"],4480)
        self.assertEqual(self.curve[-1]["cumulative_captures"],sum(x["route_success"]for b in self.rows for x in b))

    def test_window_is_exactly_last_eight_completed_batches(self):
        expected=sum(x["route_success"]for b in self.rows[3:11]for x in b)
        self.assertEqual(self.curve[10]["rolling_captures"],expected)
        self.assertEqual(self.curve[10]["rolling_attempts"],280)
        self.assertEqual(self.curve[1]["rolling_attempts"],70)
        with self.assertRaises(ValueError):film.rates(self.rows[:-1])
        bad=copy.deepcopy(self.rows);bad[8].pop()
        with self.assertRaises(ValueError):film.rates(bad)

    def test_pre_release_overlay_cannot_see_current_or_future_outcomes(self):
        initial=film.overlay_state(self.curve,self.assessments,0)
        self.assertIsNone(initial["previous"]);self.assertIsNone(initial["current"])
        self.assertEqual(initial["visible_curve"],[])
        self.assertEqual(initial["assessment"]["after_train_batches"],0)
        state=film.overlay_state(self.curve,self.assessments,63)
        self.assertEqual(state["sampling_update"],31)
        self.assertEqual(state["visible_curve"][-1]["batch"],62)
        self.assertEqual(state["assessment"]["after_train_batches"],32)
        poisoned=copy.deepcopy(self.curve)
        for future in poisoned[63:]:future["batch_capture_rate"]=999
        self.assertEqual(film.overlay_state(poisoned,self.assessments,63),state)

    def test_result_reveal_adds_only_current_completed_batch(self):
        state=film.overlay_state(self.curve,self.assessments,63,True)
        self.assertEqual(state["visible_curve"][-1]["batch"],63)
        self.assertEqual(state["current"]["batch"],63)
        self.assertEqual(state["assessment"]["after_train_batches"],64)
        final=film.overlay_state(self.curve,self.assessments,127,True)
        self.assertEqual(final["assessment"]["after_train_batches"],128)

    def test_film_keeps_all_scheduled_failures_and_declared_iteration_jumps(self):
        mapping,chapters,speed=film.frame_map(self.chapters())
        self.assertEqual(len(mapping),1920)
        self.assertEqual(speed,1)
        self.assertEqual([c["batch"]for c in chapters],list(film.FILMED))
        self.assertEqual({c["phase"]for c in chapters},{"Early training","Middle training","Late training"})
        self.assertTrue(all(not c["route_qualified_capture"]for c in chapters))
        self.assertEqual(len({m["source_frame"]for m in mapping}),len(mapping))
        self.assertTrue(all(b["source_frame"]>a["source_frame"]for a,b in zip(mapping,mapping[1:])))
        self.assertEqual(sum(c["frames"]for c in chapters),len(mapping))

    def test_short_film_is_never_stretched_and_long_film_is_bounded(self):
        mapping,_,speed=film.frame_map(self.chapters(40))
        self.assertEqual(len(mapping),960);self.assertEqual(speed,1)
        mapping,chapters,speed=film.frame_map(self.chapters(120))
        self.assertEqual(len(mapping),2700);self.assertAlmostEqual(speed,2880/2700)
        self.assertEqual(len({m["source_frame"]for m in mapping}),2700)
        with self.assertRaisesRegex(ValueError,"excessive retiming"):
            film.frame_map(self.chapters(130))

    def test_schedule_or_duplicate_frame_manipulation_is_rejected(self):
        bad=self.chapters();bad[8]["batch"]=9
        with self.assertRaisesRegex(ValueError,"schedule"):film.frame_map(bad)
        bad=self.chapters();bad[1].update(source_start_frame=0,source_end_frame_inclusive=79)
        with self.assertRaisesRegex(ValueError,"duplicated"):film.frame_map(bad)

    def test_interval_bounds_are_descriptive_and_symmetric_at_half_capture(self):
        low,high=film.wilson(140,280)
        self.assertAlmostEqual(low+high,1)
        self.assertLess(low,.5);self.assertGreater(high,.5)
        self.assertEqual(film.wilson(0,280)[0],0)
        self.assertEqual(film.wilson(280,280)[1],1)
        for args in((281,280),(-1,280),(0,0)):
            with self.assertRaises(ValueError):film.wilson(*args)

    def test_paths_cannot_escape_the_raw_evidence_or_follow_symlinks(self):
        (self.root/"okay.pt").write_bytes(b"checkpoint")
        self.assertEqual(film.safe_file(self.root,"okay.pt",".pt"),self.root/"okay.pt")
        (self.root/"alias.pt").symlink_to(self.root/"okay.pt")
        for name in("../okay.pt","/tmp/okay.pt","alias.pt","missing.pt"):
            with self.assertRaises(ValueError):film.safe_file(self.root,name,".pt")

    def trial(self):
        baseline=ISAAC.parents[1]/"baselines/isaac-expanded-farm-20261102"
        spec=json.loads((baseline/"contract.json").read_text())
        menus,_=film.load_native_menu(baseline/"expanded-menu-qualified.json",spec)
        actions=[copy.deepcopy(m[0])for m in menus]
        records=[]
        for c,a in zip(spec["cells"],actions):
            records.append(dict(row(),**a,cell_id=c["id"],bucket_xy_m=c["bucket_xy_m"],
                distance_m=a["radius_m"],settled=dict(position_error_mm=0,normal_error_deg=0,
                    joint_speed_rad_s=0,position_drift_mm=0,normal_drift_deg=0,dropper_position_error_mm=0)))
        checkpoint=self.root/"policy-00.pt";checkpoint.write_bytes(b"synthetic-checkpoint-byte-fixture")
        t=dict(schema=film.TRIAL_SCHEMA,source_fingerprint=spec["fingerprint"],
            setup_protocol=film.SETUP_PROTOCOL,policy_version=film.POLICY_VERSION,
            selected_indices=[0]*35,old_logp=[-math_log(24)]*35,entropy=[math_log(24)]*35,
            observations=np.zeros((35,5)).tolist(),candidate_features=np.zeros((35,24,10)).tolist(),
            actions=actions,rollout=records,rewards=[film.reward(r)for r in records],
            release_enabled=[True]*35,checkpoint_file=checkpoint.name,checkpoint_sha256=film.sha(checkpoint))
        return spec,menus,t

    def test_executed_categories_and_rewards_bind_actual_actions(self):
        spec,menus,trial=self.trial()
        with patch.object(film,"verify_physics",return_value=({"checked":35},[]))as native:
            rows,audit,_=film.verify_trial(self.root,trial,spec,np.zeros((35,3)),menus)
            self.assertEqual(len(rows),35);self.assertEqual(audit["checked"],35);native.assert_called_once()
        bad=copy.deepcopy(trial);bad["selected_indices"][0]=1
        with self.assertRaisesRegex(ValueError,"sampled category"):
            film.verify_trial(self.root,bad,spec,np.zeros((35,3)),menus)
        bad=copy.deepcopy(trial);bad["rewards"][0]=999
        with self.assertRaisesRegex(ValueError,"reward"):
            film.verify_trial(self.root,bad,spec,np.zeros((35,3)),menus)

    def test_canonical_setup_and_sampling_checkpoint_are_not_optional(self):
        spec,menus,trial=self.trial()
        bad=copy.deepcopy(trial);bad["setup_protocol"]="just-a-pretty-replay"
        with self.assertRaisesRegex(ValueError,"canonical"):
            film.verify_trial(self.root,bad,spec,np.zeros((35,3)),menus)
        bad=copy.deepcopy(trial);bad["selected_indices"][0]=24
        with self.assertRaisesRegex(ValueError,"category"):
            film.verify_trial(self.root,bad,spec,np.zeros((35,3)),menus)
        (self.root/trial["checkpoint_file"]).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError,"checkpoint bytes"):
            film.verify_trial(self.root,trial,spec,np.zeros((35,3)),menus)

    def checkpoint_audit(self):
        for name in("result.json","study.json","contract.json","expanded-menu-qualified.json"):
            (self.root/name).write_text(json.dumps(dict(name=name)))
        checkpoints={}
        for i in range(133):
            p=self.root/f"policy-{i:03}.pt";p.write_bytes(str(i).encode());checkpoints[p.name]=film.sha(p)
        trials={f"trial-{i:03}.json":f"digest-{i}" for i in range(160)}
        report=dict(source_fingerprint="geometry",final_policy_hash="final-weights")
        study=dict(fingerprint="study")
        audit=dict(schema=film.AUDIT_SCHEMA,status="passed",study_fingerprint="study",
            source_fingerprint="geometry",source_result_sha256=film.sha(self.root/"result.json"),
            source_study_sha256=film.sha(self.root/"study.json"),
            source_contract_sha256=film.sha(self.root/"contract.json"),
            menu_sha256=film.sha(self.root/"expanded-menu-qualified.json"),
            source_trial_sha256=trials,checkpoint_sha256=checkpoints,
            training_batches_checked=128,assessment_batches_checked=32,updates_checked=64,
            final_policy_hash="final-weights")
        (self.root/"checkpoint-audit.json").write_text(json.dumps(audit))
        return report,study,trials,audit

    def test_independent_audit_binds_all_sampling_checkpoints_and_final_weights(self):
        report,study,trials,audit=self.checkpoint_audit()
        proof=film.verify_checkpoint_audit(self.root,report,study,trials)
        self.assertEqual(proof["audit"]["updates_checked"],64)
        (self.root/"policy-132.pt").write_bytes(b"changed-final-weight-file")
        with self.assertRaisesRegex(ValueError,"changed after"):
            film.verify_checkpoint_audit(self.root,report,study,trials)

    def test_stale_audits_or_missing_training_data_are_rejected(self):
        report,study,trials,audit=self.checkpoint_audit()
        truncated=dict(trials);truncated.pop(next(iter(truncated)))
        with self.assertRaisesRegex(ValueError,"stale"):
            film.verify_checkpoint_audit(self.root,report,study,truncated)
        (self.root/"result.json").write_text("{}")
        with self.assertRaisesRegex(ValueError,"stale"):
            film.verify_checkpoint_audit(self.root,report,study,trials)

    @unittest.skipUnless(importlib.util.find_spec("PIL"), "Pillow is required for overlay rendering")
    def test_overlay_is_a_declared_transparent_data_layer_with_no_source_pixels(self):
        summary=dict(learning_curve=self.curve,known_layout_assessments=self.assessments,
            ball_radius_m=.0375,chapters=[dict(batch=b,foreground_action=dict(height_m=.38,gap_m=.26,tilt_deg=26))for b in film.FILMED])
        before=film.make_overlay(summary,0,False)
        after=film.make_overlay(summary,0,True)
        self.assertEqual(before.size,(1920,1080));self.assertEqual(before.mode,"RGBA")
        # Foreground drop/board/bucket region is untouched by overlays.
        self.assertEqual(before.getpixel((850,650)),(0,0,0,0))
        self.assertEqual(before.getpixel((1500,40)),(0,0,0,0))
        self.assertGreater(after.getpixel((1500,40))[3],0)


def math_log(value):
    import math
    return math.log(value)


if __name__=="__main__":
    unittest.main()
