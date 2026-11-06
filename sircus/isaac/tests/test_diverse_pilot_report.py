"""Matched denominators, historical identity and read-only diverse evidence."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import diverse_pilot_report as module
import diverse_pilot_server as server


def count(n,c,invalid,u=0):
    return dict(attempts=n,captures=c,invalid=invalid,clean=0,update_count=u,
                unique_layouts=25 if n==200 else 5,layouts_with_capture=1,failures={"endpoints_not_settled":invalid})


def fixture(kind="diverse"):
    cells=[dict(id=f"c{i}",split="test" if i<5 else "validation" if i<10 else "train",
                bucket_xy_m=[i/10,0]) for i in range(35)]
    checks=[dict(count(n,8 if u==0 else 16,0,u),split=split)
            for split,n in (("train",200),("validation",40),("test",40)) for u in (0,16)]
    return dict(status="completed",cells=cells,geometry_fingerprint="geometry",study_fingerprint=kind,
        scoring="same scoring",study=dict(schema=module.STUDY_SCHEMA if kind=="diverse" else kind,
            source=dict(contract_sha256="contract",camera_sha256="camera"),training_batches=64,
            settling_protocol=dict(version="new-controller",unchanged_gates=True)),
        source_sha256={"result.json":{"broad":module.BASELINE_RESULT_SHA256,"guided":module.GUIDED_RESULT_SHA256,"diverse":"new-result"}[kind]},
        training=count(1600,62,{"broad":1214,"guided":879,"diverse":300}[kind]),
        assessments=checks,learning_curve=[],elapsed_s=3600,feasibility=dict(attempts=17920,admitted=155),
        audits={key:dict(report=dict(status="passed")) for key in ("native","checkpoint","feasibility")})


def support():
    raw=dict(schema=module.SUPPORT_SCHEMA,policy_geometry_only=True,
        cells=[dict(cell_id="c10",status="supported",bucket_xy_m=[1.,0],regions=[
            dict(id=0,low=[0,.3,.2,.1,.2],high=[.1,.4,.3,.2,.3],
                 anchor_ik={"Arm10":{"q":[.1,.2]}},calibration=[{"private":"probe readback"}]),
            dict(id=1,low=[1,.6,.4,.3,.4],high=[1.1,.7,.5,.4,.5])],
            anchor_checks=[{"private":"joint readback"}])])
    raw["fingerprint"]=module.canonical_digest(raw)
    return raw


class ComparisonTests(unittest.TestCase):
    def compare(self,new=None,broad=None,guided=None):
        return module.comparison(new or fixture(),broad or fixture("broad"),guided or fixture("guided"))

    def test_three_methods_share_attempt_budget_but_released_denominator_is_separate(self):
        comparison=self.compare()
        row=comparison["methods"]["diverse"]["training"]
        self.assertEqual(row["attempts"],1600)
        self.assertEqual(row["released"],1300)
        self.assertEqual(row["capture_rate"],62/1600)
        self.assertEqual(row["capture_rate_per_released"],62/1300)
        self.assertEqual(comparison["final_policy_difference"]["test"]["diverse_minus_guided_percentage_points"],0)
        self.assertIn("not globally unseen",comparison["scientific_scope"])
        self.assertIn("controller change together",comparison["scientific_scope"])
        self.assertEqual(comparison["methods"]["diverse"]["training_failure_counts"]["endpoints_not_settled"],300)

    def test_partial_run_does_not_publish_final_improvement(self):
        value=fixture();value["status"]="in_progress";value["training"]=count(25,1,12)
        comparison=self.compare(value)
        self.assertIsNone(comparison["methods"]["diverse"]["frozen"]["validation"]["final"])
        self.assertIsNone(comparison["final_policy_difference"]["test"]["diverse_minus_broad_percentage_points"])

    def test_both_original_baselines_have_pinned_identities(self):
        for kind in ("broad","guided"):
            value=fixture(kind);value["source_sha256"]["result.json"]="other-result"
            with self.subTest(kind=kind),self.assertRaisesRegex(ValueError,"Unexpected"):
                self.compare(**{kind:value})
            value=fixture(kind);value["status"]="in_progress"
            with self.subTest(kind=kind),self.assertRaisesRegex(ValueError,"frozen"):
                self.compare(**{kind:value})

    def test_contract_camera_geometry_scoring_and_split_fail_closed(self):
        for mutate in (
            lambda r:r.update(geometry_fingerprint="different"),
            lambda r:r.update(scoring="different"),
            lambda r:r["study"]["source"].update(camera_sha256="different"),
            lambda r:r["study"]["source"].update(contract_sha256="different"),
            lambda r:r["cells"][0].update(split="train"),
            lambda r:r["cells"][0].update(bucket_xy_m=[99,99]),
            lambda r:r["study"].update(settling_protocol=None),
        ):
            value=fixture();mutate(value)
            with self.subTest(mutate=mutate),self.assertRaises(ValueError):self.compare(value)

    def test_changed_budget_or_mismatched_frozen_repeat_budget_fails(self):
        value=fixture();value["training"]["attempts"]=1599
        with self.assertRaisesRegex(ValueError,"budget"):self.compare(value)
        value=fixture();value["assessments"][-1]["attempts"]=39
        with self.assertRaisesRegex(ValueError,"budgets"):self.compare(value)

    def test_comparison_does_not_mutate_evidence(self):
        value=fixture();before=copy.deepcopy(value)
        self.compare(value)
        self.assertEqual(value,before)

    def test_optional_setup_check_is_bound_to_frozen_requests_and_no_drops(self):
        report=fixture();report["study"].update(support_sha256="support-bytes",support_fingerprint="support-fp")
        value=dict(schema=module.SETUP_CHECK_SCHEMA,status="completed",source_guided_result_sha256=module.GUIDED_RESULT_SHA256,
            geometry_fingerprint=report["geometry_fingerprint"],support_sha256="support-bytes",support_fingerprint="support-fp",
            settling_protocol=report["study"]["settling_protocol"],source_setup_check_sha256="a"*64,
            attempts=100,before_admitted=20,after_admitted=70,repaired=55,regressions=5,ball_releases=0,
            before_failures={"endpoints_not_settled":80},after_failures={"endpoints_not_settled":30},
            paired=[{"joint_readback":"private"}],records=[{"physics":"private"}])
        receipt=module.setup_check_receipt(value,report)
        self.assertEqual(receipt["after_admitted"],70)
        self.assertNotIn("paired",receipt);self.assertNotIn("records",receipt)
        for mutate in (
            lambda r:r.update(source_guided_result_sha256="wrong"),
            lambda r:r.update(geometry_fingerprint="wrong"),
            lambda r:r.update(support_fingerprint="wrong"),
            lambda r:r.update(ball_releases=1),
            lambda r:r.update(attempts=99),
            lambda r:r.update(repaired=56),
            lambda r:r.update(after_failures={"endpoints_not_settled":29}),
            lambda r:r.update(settling_protocol={"different":"controller"}),
        ):
            wrong=copy.deepcopy(value);mutate(wrong)
            with self.subTest(mutate=mutate),self.assertRaises(ValueError):module.setup_check_receipt(wrong,report)

    def test_independent_receipt_binds_all_three_result_ids_and_every_stage(self):
        comparison=self.compare()
        receipt=dict(schema=module.SCHEMA,status="completed",methods=copy.deepcopy(comparison["methods"]))
        self.assertEqual(module.verify_comparison_audit(receipt,comparison),receipt)
        nested=dict(status="passed",source_result_sha256="new-result",comparison=receipt)
        module.verify_comparison_audit(nested,comparison)
        wrong=copy.deepcopy(receipt);wrong["methods"]["guided"]["source_result_sha256"]="wrong"
        with self.assertRaisesRegex(ValueError,"result bytes"):module.verify_comparison_audit(wrong,comparison)
        wrong=copy.deepcopy(receipt);wrong["methods"]["broad"]["frozen"]["test"]["final"]["captures"]+=1
        with self.assertRaisesRegex(ValueError,"count mismatch"):module.verify_comparison_audit(wrong,comparison)
        nested["status"]="pending"
        with self.assertRaisesRegex(ValueError,"Containing audit"):module.verify_comparison_audit(nested,comparison)

    def test_compact_union_preserves_separate_regions_and_binds_omitted_readbacks(self):
        raw=support();value=module.compact_support(raw)
        self.assertEqual(value["retained_region_count"],2)
        self.assertEqual(value["cells"][0]["regions"][1]["low"],[1,.6,.4,.3,.4])
        self.assertIn("source_support_fingerprint",value)
        self.assertEqual(value["cells"][0]["regions"][0]["omitted_fields_sha256"],
            module.canonical_digest({"anchor_ik":raw["cells"][0]["regions"][0]["anchor_ik"],
                                     "calibration":raw["cells"][0]["regions"][0]["calibration"]}))
        text=json.dumps(value)
        for word in ('"anchor_ik":','"q":','probe readback','joint readback'):
            self.assertNotIn(word,text)
        raw["cells"][0]["regions"][0]["low"][0]=9
        with self.assertRaisesRegex(ValueError,"fingerprint"):module.compact_support(raw)

    def test_region_counts_dont_disguise_unresolved_or_duplicate_ids(self):
        for mutate in (
            lambda r:r["cells"][0].update(status="unresolved"),
            lambda r:r["cells"][0]["regions"][1].update(id=0),
            lambda r:r["cells"][0]["regions"][0].update(high=[0,.4,.3,.2,.3]),
        ):
            raw=support();mutate(raw);raw.pop("fingerprint");raw["fingerprint"]=module.canonical_digest(raw)
            with self.subTest(mutate=mutate),self.assertRaises(ValueError):module.compact_support(raw)

    def test_region_usage_excludes_inactive_heldout_and_coverage_requests(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/"trials").mkdir()
            cells=[dict(id="c10",split="train"),dict(id="test",split="test")]
            (root/"study.json").write_text(json.dumps(dict(cells=cells)))
            sup=support();sup["cells"].append(dict(cell_id="test",status="unresolved",regions=[]))
            trial=dict(mode="train",region_indices=[1,0],active_cell_ids=["c10"])
            (root/"trials/a.json").write_text(json.dumps(trial))
            (root/"trials/b.json").write_text(json.dumps(dict(trial,mode="coverage",active_cell_ids=["test"])))
            value=module.region_usage(root,sup)
            self.assertEqual(value["logged_training_attempts"],1)
            self.assertEqual(value["cells"][0]["attempted_region_counts"],{"1":1})
            self.assertEqual(value["cells"][1]["sampled_regions"],0)
            trial["region_indices"][0]=999
            (root/"trials/a.json").write_text(json.dumps(trial))
            with self.assertRaisesRegex(ValueError,"Unknown requested"):module.region_usage(root,sup)

    def test_viewer_keeps_native_video_map_and_csv_with_no_command_endpoint(self):
        html=server.index_html().decode()
        for name in ("map","video","curve","comparisonChecks","comparisonSettling","comparisonRegions"):
            self.assertIn(f'id="{name}"',html)
        self.assertIn("/diverse_pilot_panel.js",html)
        self.assertIn("/settings.csv",html)
        self.assertNotIn("/api/physical",html)
        self.assertIn("Gaps between regions are not permitted",html)
        served=server.base_script().decode()
        self.assertIn("geometry-only region choice + local affine residual-v1",served)
        self.assertIn("'Global outer domain'",served)


if __name__=="__main__":unittest.main()
