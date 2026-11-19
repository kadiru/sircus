"""Guard fair comparison, complete denominators and the new read-only viewer."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import feasibility_guided_report as module
import feasibility_guided_server as server


def count(n,c,invalid,u=0):
    return dict(attempts=n,captures=c,invalid=invalid,clean=0,update_count=u,
                unique_layouts=25 if n==200 else 5,layouts_with_capture=1)


def fixture(guided=False):
    cells=[dict(id=f"c{i}",split="test" if i<5 else "validation" if i<10 else "train",
                bucket_xy_m=[i/10,0]) for i in range(35)]
    checks=[]
    for split,n in (("train",200),("validation",40),("test",40)):
        for u in (0,16):checks.append(dict(count(n,8 if u==0 else 16,0,u),split=split))
    return dict(status="completed",cells=cells,geometry_fingerprint="same-geometry",
                study_fingerprint="guided" if guided else "broad",scoring="unchanged",
                study=dict(schema=module.GUIDED_STUDY_SCHEMA if guided else "old",
                           source=dict(contract_sha256="same",camera_sha256="same"),training_batches=64),
                source_sha256={"result.json":"new" if guided else module.BASELINE_RESULT_SHA256},
                training=count(1600,62,200 if guided else 1214),assessments=checks,
                learning_curve=[],elapsed_s=3600,feasibility=dict(attempts=17920,admitted=155),
                audits={key:dict(report=dict(status="passed")) for key in ("native","checkpoint","feasibility")})


class ComparisonTests(unittest.TestCase):
    def test_same_budget_rejections_never_disappear(self):
        c=module.comparison(fixture(True),fixture())
        a,b=(c["methods"][key]["training"] for key in ("broad","guided"))
        self.assertEqual(a["attempts"],b["attempts"])
        self.assertEqual(b["released"],1400)
        self.assertEqual(b["capture_rate"],62/1600)
        self.assertEqual(b["capture_rate_per_released"],62/1400)
        self.assertEqual(c["final_policy_difference"]["test"]["guided_minus_broad_percentage_points"],0)
        self.assertIn("not globally unseen",c["scientific_scope"])

    def test_no_final_claim_for_partial_run_or_initial_only(self):
        g=fixture(True);g["status"]="in_progress";g["training"]=count(100,1,50)
        c=module.comparison(g,fixture())
        self.assertIsNone(c["methods"]["guided"]["frozen"]["test"]["final"])
        self.assertIsNone(c["final_policy_difference"]["test"]["guided_minus_broad_percentage_points"])
        g=fixture(True);g["assessments"]=[r for r in g["assessments"] if r["update_count"]==0]
        self.assertIsNone(module.comparison(g,fixture())["methods"]["guided"]["frozen"]["train"]["final"])

    def test_geometry_layout_split_camera_and_scoring_must_match(self):
        for mutate in (
            lambda g:g.update(geometry_fingerprint="different"),
            lambda g:g["cells"][0].update(bucket_xy_m=[99,0]),
            lambda g:g["cells"][0].update(split="train"),
            lambda g:g["study"]["source"].update(camera_sha256="different"),
            lambda g:g["study"]["source"].update(contract_sha256="different"),
            lambda g:g.update(scoring="different"),
        ):
            g=fixture(True);mutate(g)
            with self.subTest(mutate=mutate),self.assertRaises(ValueError):module.comparison(g,fixture())

    def test_baseline_identity_and_equal_attempt_budget_fail_closed(self):
        b=fixture();b["source_sha256"]["result.json"]="other"
        with self.assertRaisesRegex(ValueError,"baseline"):module.comparison(fixture(True),b)
        g=fixture(True);g["training"]["attempts"]=1599
        with self.assertRaisesRegex(ValueError,"budget"):module.comparison(g,fixture())
        g=fixture(True);g["assessments"][-1]["attempts"]=39
        with self.assertRaisesRegex(ValueError,"budgets"):module.comparison(g,fixture())

    def test_impossible_admission_counts_rejected(self):
        with self.assertRaisesRegex(ValueError,"counts"):module.compact_counts(count(10,8,3))
        self.assertIsNone(module.compact_counts(count(0,0,0))["capture_rate"])

    def test_comparison_is_non_mutating_and_support_cost_is_separate(self):
        g,b=fixture(True),fixture();before=copy.deepcopy(g)
        c=module.comparison(g,b,support=dict(seconds=12,qualified_cells=33))
        self.assertEqual(g,before)
        self.assertEqual(c["geometry_preparation"]["support"]["seconds"],12)
        self.assertIn("additional preprocessing",c["geometry_preparation"]["cost_scope"])

    def test_independent_comparison_receipt_binds_both_results_and_counts(self):
        g,b=fixture(True),fixture();c=module.comparison(g,b)
        audit=dict(schema=module.SCHEMA,status="completed",
                   original_result_sha256=b["source_sha256"]["result.json"],guided_result_sha256="new")
        for label,method in (("original","broad"),("guided","guided")):
            m=c["methods"][method]
            def original(row):
                row=copy.deepcopy(row);row["invalid"]=row["rejected"];return row
            audit[label]=dict(training=original(m["training"]),frozen={
                stage:{split:original(m["frozen"][split][stage]) for split in module.SPLITS}
                for stage in ("initial","final")})
        module.verify_comparison_audit(audit,c)
        nested=dict(schema="checkpoint-audit",status="passed",source_result_sha256="new",comparison=audit)
        self.assertEqual(module.verify_comparison_audit(nested,c),audit)
        nested["source_result_sha256"]="wrong"
        with self.assertRaisesRegex(ValueError,"containing audit"):module.verify_comparison_audit(nested,c)
        audit["guided"]["frozen"]["final"]["test"]["captures"]+=1
        with self.assertRaisesRegex(ValueError,"count mismatch"):module.verify_comparison_audit(audit,c)
        audit["guided_result_sha256"]="other"
        with self.assertRaisesRegex(ValueError,"different native"):module.verify_comparison_audit(audit,c)

    def test_compact_support_omits_probe_and_ik_readbacks_but_binds_decisions(self):
        raw=dict(schema="sircus-feasibility-guided-support-v1",policy_geometry_only=True,
                 cells=[dict(cell_id="a",status="supported",anchor_action={"height_m":.3},
                  anchor_ik={"Arm10":{"q":[.1,.2]}},anchor_checks=[{"readback":"private trace"}],
                  low=[0]*5,high=[1]*5,qualification_target_met=False,
                  calibration=[dict(scale=.125,low=[0]*5,high=[1]*5,requests=1,accepted=0,
                    acceptance=0,records=[{"qualification":{"q":[.4,.5]},"accepted":False}])])])
        raw["fingerprint"]=module.canonical_digest(raw)
        compact=module.compact_support(raw)
        text=json.dumps(compact)
        self.assertNotIn('"records":',text)
        self.assertNotIn('"anchor_ik":',text)
        self.assertNotIn('"q":',text)
        self.assertNotIn('private trace',text)
        self.assertEqual(compact["source_support_fingerprint"],raw["fingerprint"])
        self.assertEqual(compact["cells"][0]["calibration"][0]["records_count"],1)
        self.assertEqual(compact["cells"][0]["calibration"][0]["records_sha256"],
                         module.canonical_digest(raw["cells"][0]["calibration"][0]["records"]))
        raw["cells"][0]["status"]="unresolved"
        with self.assertRaisesRegex(ValueError,"fingerprint"):module.compact_support(raw)

    def test_posthoc_exclusion_recomputes_membership_and_binds_frozen_sources(self):
        b,g=fixture(),fixture(True)
        g["study"]["support_sha256"]="support-bytes"
        support=dict(fingerprint="support-fp",cells=[dict(cell_id="c10",
            low=[-.1,.5,.2,.1,.4],high=[.1,.6,.4,.3,.6])])
        action=dict(zip(("bearing_offset_rad","radius_m","height_m","gap_m","tilt_rad"),[0,.4,.3,.2,.5]))
        b["cells"][10].update(evaluation=[dict(count(8,8,0,16))],
            frozen_predictions=[dict(update_count=16,requested=action)])
        trials={f"trials/eval-{i:04d}.json":str(i) for i in range(76,84)}
        b["source_sha256"].update(trials)
        for key in ("native","checkpoint"):b["audits"][key]["sha256"]=key+"-audit"
        row=dict(cell_id="c10",split="train",bucket_xy_m=[1.,0],
            old_frozen_captures=8,old_frozen_attempts=8,
            old_requested_action=action,guided_low=support["cells"][0]["low"],
            guided_high=support["cells"][0]["high"],inside_guided_box=False,
            excluded_axes=[dict(axis="radius_m")])
        r=dict(schema="sircus-guided-support-success-exclusion-v1",status="completed",
            old_source_result_sha256=module.BASELINE_RESULT_SHA256,old_study_fingerprint=b["study_fingerprint"],
            geometry_fingerprint=g["geometry_fingerprint"],support_sha256="support-bytes",support_fingerprint="support-fp",
            original_native_audit_sha256="native-audit",original_checkpoint_audit_sha256="checkpoint-audit",
            source_trial_sha256=trials,records=[row],old_successful_sites=1,old_successful_repeated_captures=8,
            old_successful_settings_retained=0,old_successful_settings_excluded=1,excluded_repeated_successes=8,
            excluded_site_counts_by_axis={"radius_m":1})
        module.verify_support_exclusion(r,g,b,support)
        for field in ("old_source_result_sha256","support_sha256","geometry_fingerprint"):
            changed=copy.deepcopy(r);changed[field]="wrong"
            with self.subTest(field=field),self.assertRaises(ValueError):module.verify_support_exclusion(changed,g,b,support)
        changed=copy.deepcopy(r);changed["records"][0]["inside_guided_box"]=True
        with self.assertRaisesRegex(ValueError,"classification"):module.verify_support_exclusion(changed,g,b,support)
        changed=copy.deepcopy(r);changed["old_successful_settings_excluded"]=0
        with self.assertRaisesRegex(ValueError,"aggregate"):module.verify_support_exclusion(changed,g,b,support)

    def test_new_html_keeps_native_ui_and_adds_comparison_without_controls(self):
        text=server.index_html().decode()
        for name in ("map","video","curve","comparisonChecks","comparisonScope"):
            self.assertIn(f'id="{name}"',text)
        self.assertIn("/continuous_pilot.js",text)
        self.assertIn("/feasibility_guided_panel.js",text)
        self.assertNotIn("/api/physical",text)
        self.assertIn("previous broad-domain pilot remains frozen",text)
        self.assertIn("not executed motion or successful-shot coverage",text)


if __name__=="__main__":unittest.main()
