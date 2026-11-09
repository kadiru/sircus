"""Queued/audited publication boundaries, using synthetic selector fixtures."""
import copy
import gzip
import json
from pathlib import Path
import sys
import tempfile
import unittest

SOURCE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(SOURCE))
import candidate_comparison_report as report
import test_candidate_selector as fixtures


class ComparisonReportTests(unittest.TestCase):
    def setUp(self):
        fixture=fixtures.SelectorTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        self.plan=fixture.build()
        self.root=fixture.root
        self.sha="e"*64

    def write(self, path, value):
        path.write_text(json.dumps(value))

    def seal_audit(self, path, audit):
        audit.pop("fingerprint", None)
        audit["fingerprint"] = report.canonical_hash(audit)
        self.write(path, audit)

    def refresh_binding(self, run, path, *, audit_change=None):
        audit=report.read(path)
        for filename,field in (("result.json","source_result_sha256"),
                               ("contract.json","source_contract_sha256"),
                               ("experiment-binding.json","source_binding_sha256")):
            audit[field]=report.digest(run/filename)
        if audit_change:audit_change(audit)
        self.seal_audit(path,audit)

    def test_queue_never_publishes_zero_as_completed_performance(self):
        value=report.build_report(self.plan,plan_sha256=self.sha,status="queued",queue_reason="other GPU owner")
        self.assertEqual(value["status"],"queued")
        self.assertEqual(value["audit"]["status"],"pending")
        self.assertEqual(sum(x["requests"] for x in value["methods"].values()),0)
        self.assertTrue(all(x["captures"] is None for x in value["methods"].values()))
        self.assertEqual(value["queue"]["reason"],"other GPU owner")

    def fixture(self):
        run=self.root/"run";run.mkdir();trials=[];methods={}
        scoring={name:"2"*64 for name in report.SCORE_FILES}
        sources=dict(scoring,**{"candidate_native_audit.py":"1"*64})
        contract=dict(scene={"synthetic":True})
        contract["fingerprint"]=report.canonical_hash(contract)
        original=dict(commit="1"*40,geometry_fingerprint=contract["fingerprint"])
        self.plan["native_binding"]=dict(checkpoint_sha256=self.plan["input_bindings"]["original"]["checkpoint_sha256"],
            checkpoint_path="/synthetic/model.npz",source_commit="3"*40,source_archive_sha256="4"*64,
            geometry_fingerprint=contract["fingerprint"],source_scene_sha256=report.canonical_hash(contract["scene"]),
            original_source=original,original_scoring_source_sha256=scoring,
            bank_preprocessing={},selection_wall_s=0.)
        self.plan.pop("fingerprint")
        self.plan["fingerprint"]=report.canonical_hash(self.plan)
        for index,method in enumerate(("parent","candidate_bank")):
            rows=[dict(route_success=i < (10 if method=="parent" else 12),route_quality=2 if i<3 else 1 if i<(10 if method=="parent" else 12) else 0,invalid="ik_unresolved" if i==34 else None) for i in range(35)]
            cohort=self.plan["cohorts"][index]
            packet=dict(schema="sircus-shot-benchmark-native-trial-v1",mode=method,
                comparison_fingerprint=self.plan["fingerprint"],cells=self.plan["cells"],
                rows=rows,actions=cohort["actions"],enabled=cohort["native_enabled"],proposal_ik_seeds=cohort["ik_seeds"])
            path=run/(method+".json.gz");path.write_bytes(gzip.compress(json.dumps(packet).encode()))
            metrics=dict(wall_s=4.,cpu_s=5.,peak_rss_mib=30.)
            trials.append(dict(method=method,file=path.name,sha256=report.digest(path),requests=35,metrics=metrics))
            captures=10 if method=="parent" else 12
            methods[method]=dict(counts=dict(requests=35,captures=captures,clean=3,rejected=1,released=34,
                failures=dict(captured=captures,released_miss=34-captures,ik_unresolved=1)),metrics=metrics)
        binding=dict(comparison_sha256=self.sha,comparison_fingerprint=self.plan["fingerprint"],
            checkpoint_sha256=self.plan["input_bindings"]["original"]["checkpoint_sha256"],original_source=original,
            source_commit="3"*40,source_archive_sha256="4"*64,source_sha256=sources,control_sha256={})
        self.write(run/"contract.json",contract)
        self.write(run/"experiment-binding.json",binding)
        self.write(run/"result.json",dict(schema=report.RESULT_SCHEMA,status="completed",
            comparison_fingerprint=self.plan["fingerprint"],comparison_sha256=self.sha,
            optimizer_updates=0,native_requests=70,source_sha256=sources,original_source=original,
            checkpoint_sha256_before=binding["checkpoint_sha256"],checkpoint_sha256_after=binding["checkpoint_sha256"],
            control_files={},trials=trials,initialization_shared_wall_s=2.,comparison_wall_s=8.,resource_scope="Separate intervals"))
        audit=dict(schema=report.AUDIT_SCHEMA,status="passed",comparison_fingerprint=self.plan["fingerprint"],
            comparison_sha256=self.sha,native_requests=70,optimizer_updates=0,held_out_shot_files_read=0,
            checkpoint_sha256=binding["checkpoint_sha256"],source_sha256=sources,audit_source_sha256=sources["candidate_native_audit.py"],
            source_result_sha256=report.digest(run/"result.json"),source_contract_sha256=report.digest(run/"contract.json"),
            source_binding_sha256=report.digest(run/"experiment-binding.json"),trials=trials,methods=methods,rescored_releases=68,full_rate_steps=20)
        path=self.root/"audit.json";self.seal_audit(path,audit)
        return run,path

    def test_final_rates_and_paired_gains_come_from_bound_audit(self):
        run,path=self.fixture()
        value=report.build_report(self.plan,plan_sha256=self.sha,run=run,audit_path=path)
        self.assertEqual(value["status"],"passed")
        self.assertEqual(value["methods"]["bank"]["captures"],12)
        self.assertEqual(value["paired"]["gains"],2)
        self.assertEqual(value["paired"]["losses"],0)
        self.assertEqual(len(value["cases"]),35)
        self.assertEqual(value["methods"]["parent"]["rejection_reasons"],{"ik_unresolved":1})
        self.assertEqual(value["methods"]["parent"]["outcome_breakdown"]["captured"],10)
        self.assertEqual(value["cases"][-1]["bank"]["rejection"],"ik_unresolved")

    def test_unbound_audit_result_and_trial_bytes_fail_closed(self):
        run,path=self.fixture()
        audit=json.loads(path.read_text());audit["comparison_fingerprint"]="bad";self.seal_audit(path,audit)
        with self.assertRaisesRegex(ValueError,"binding"):
            report.build_report(self.plan,plan_sha256=self.sha,run=run,audit_path=path)
        audit["comparison_fingerprint"]=self.plan["fingerprint"];self.seal_audit(path,audit)
        (run/"result.json").write_text("{}")
        with self.assertRaisesRegex(ValueError,"bytes changed"):
            report.build_report(self.plan,plan_sha256=self.sha,run=run,audit_path=path)

    def test_unsigned_changed_counts_and_unknown_schema_fail(self):
        run,path=self.fixture();original=report.read(path)
        audit=copy.deepcopy(original);audit["methods"]["parent"]["counts"]["captures"]+=1
        self.write(path,audit)
        with self.assertRaisesRegex(ValueError,"fingerprint"):
            report.build_report(self.plan,plan_sha256=self.sha,run=run,audit_path=path)
        self.seal_audit(path,audit)
        with self.assertRaisesRegex(ValueError,"counts disagree"):
            report.build_report(self.plan,plan_sha256=self.sha,run=run,audit_path=path)
        audit=copy.deepcopy(original);audit["schema"]="other-audit-v1";self.seal_audit(path,audit)
        with self.assertRaisesRegex(ValueError,"schema"):
            report.build_report(self.plan,plan_sha256=self.sha,run=run,audit_path=path)

    def test_contract_and_binding_bytes_are_required_and_bound(self):
        run,path=self.fixture()
        for name in ("contract.json","experiment-binding.json"):
            with self.subTest(file=name):
                original=(run/name).read_bytes();(run/name).write_bytes(original+b" ")
                with self.assertRaisesRegex(ValueError,"bytes changed"):
                    report.build_report(self.plan,plan_sha256=self.sha,run=run,audit_path=path)
                (run/name).write_bytes(original)

    def test_rehashed_wrong_checkpoint_source_and_producer_identity_fail(self):
        run,path=self.fixture();original_audit=report.read(path)
        for filename,field in (("result.json","checkpoint_sha256_after"),
                               ("experiment-binding.json","source_archive_sha256"),
                               ("experiment-binding.json","source_commit")):
            with self.subTest(field=field):
                original=report.read(run/filename);value=copy.deepcopy(original);value[field]="f"*64
                self.write(run/filename,value);self.refresh_binding(run,path)
                with self.assertRaisesRegex(ValueError,"identity changed"):
                    report.build_report(self.plan,plan_sha256=self.sha,run=run,audit_path=path)
                self.write(run/filename,original);self.seal_audit(path,copy.deepcopy(original_audit))
        audit=copy.deepcopy(original_audit);audit["audit_source_sha256"]="f"*64;self.seal_audit(path,audit)
        with self.assertRaisesRegex(ValueError,"source identity"):
            report.build_report(self.plan,plan_sha256=self.sha,run=run,audit_path=path)
        self.seal_audit(path,copy.deepcopy(original_audit))
        result=report.read(run/"result.json");result["source_sha256"]["candidate_native_audit.py"]="f"*64
        self.write(run/"result.json",result);self.refresh_binding(run,path)
        with self.assertRaisesRegex(ValueError,"source identity"):
            report.build_report(self.plan,plan_sha256=self.sha,run=run,audit_path=path)

    def test_resigned_wrong_geometry_and_packet_case_mapping_fail(self):
        run,path=self.fixture();original=report.read(run/"contract.json")
        contract=copy.deepcopy(original);contract["scene"]["synthetic"]=False
        contract.pop("fingerprint");contract["fingerprint"]=report.canonical_hash(contract)
        self.write(run/"contract.json",contract);self.refresh_binding(run,path)
        with self.assertRaisesRegex(ValueError,"geometry contract identity"):
            report.build_report(self.plan,plan_sha256=self.sha,run=run,audit_path=path)
        self.write(run/"contract.json",original);self.refresh_binding(run,path)
        packet_path=run/"parent.json.gz";packet=report.read(packet_path);packet["cells"]=list(reversed(packet["cells"]))
        packet_path.write_bytes(gzip.compress(json.dumps(packet).encode()))
        result=report.read(run/"result.json");result["trials"][0]["sha256"]=report.digest(packet_path)
        self.write(run/"result.json",result)
        self.refresh_binding(run,path,audit_change=lambda a:a["trials"][0].update(sha256=report.digest(packet_path)))
        with self.assertRaisesRegex(ValueError,"action/case mapping"):
            report.build_report(self.plan,plan_sha256=self.sha,run=run,audit_path=path)

    def test_unknown_progress_method_and_out_of_budget_counts_fail(self):
        run=self.root/"progress-run";run.mkdir()
        for method,count in (("ppo",1),("parent",36),("candidate_bank",True)):
            with self.subTest(method=method,count=count):
                self.write(run/"progress.json",dict(comparison_fingerprint=self.plan["fingerprint"],methods={method:dict(attempts=count)}))
                with self.assertRaisesRegex(ValueError,"progress"):
                    report.build_report(self.plan,plan_sha256=self.sha,run=run)


if __name__=="__main__":unittest.main()
