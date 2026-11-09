"""Fabricated publication receipts; no model, native runtime or live files.

Fixtures deliberately use real-result flag values solely to exercise the final
publisher's joins. Their claimed passes/performance are not experiment evidence.
"""

import ast
import copy
from dataclasses import asdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import candidate_continuation_report as report
from candidate_continuation_ledger import Ledger, Outcome, SourceBindings

H = "a"*64
SESSION = "b"*32
SOURCE_SHA = "c"*64


def seal(value):
    value.pop("fingerprint",None); value["fingerprint"] = report.canonical_hash(value)
    return value


def raw(value):
    return (json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n").encode()


def digest(value):
    return hashlib.sha256(raw(value)).hexdigest()


def event(events, kind, payload):
    item = seal(dict(schema="sircus-candidate-continuation-journal-v1",seq=len(events),
        previous_sha256=events[-1]["fingerprint"] if events else "0"*64,kind=kind,payload=payload))
    events.append(item)


@lru_cache(maxsize=1)
def fixture():
    ids = {s:[f"{s}-{i:03d}" for i in range(n)] for s,n in report.SPLITS.items()}
    source = SourceBindings(train_bank_sha256="1"*64,train_evidence_sha256=H,train_support_sha256=H,
        fresh_split_manifest_sha256="2"*64,fresh_support_sha256=H,guard_protocol_sha256=H,
        scorer_sha256=H,producer_source_sha256=SOURCE_SHA)
    bindings = dict(asdict(source),source_sha256=SOURCE_SHA,input_sha256="3"*64,
        support_sha256=H,learning_protocol_sha256=H)
    ledger = Ledger.new(ids["train"],ids["validation"],ids["test"],bindings=source)
    outcomes = []
    while ledger.next_batch() is not None:
        batch = ledger.next_batch(); ledger = ledger.reserve_batch(batch)
        for request in batch.requests:
            i = request.request_index
            outcomes.append(dict(request_id=request.request_id,released=True,decline_reason=None,
                source_sha256=H,captured=i%3 == 0,clean=i%6 == 0,**{"return":1.6 if i%6 == 0 else (1.5 if i%3 == 0 else -.8)}))
        ledger = ledger.record_batch(tuple(Outcome(r.request_id,True,source_sha256=H) for r in batch.requests))
    trained = ledger
    checkpoint = b"Fabricated checkpoint bytes: never loaded as a model."
    final_sha = hashlib.sha256(checkpoint).hexdigest()
    ledger = ledger.begin_assessment(final_sha)
    while ledger.next_batch() is not None:
        batch = ledger.next_batch(); ledger = ledger.reserve_batch(batch)
        ledger = ledger.record_batch(tuple(Outcome(r.request_id,True,source_sha256=H) for r in batch.requests))
    events = []
    event(events,"start",{})
    event(events,"outcomes",dict(results=outcomes))
    event(events,"final",dict(ledger=trained.to_dict()))
    learning_counts = dict(imitation_steps=240,imitation_presentations=30680,ppo_updates=10,ppo_steps=40,ppo_samples=1400)
    model_hashes = dict(parent_weight_hash=H,continued_weight_hash=H)
    controls = {n:H for n in ("probes.json","rim-rays.json")}
    binding = seal(dict(schema="sircus-candidate-continuation-native-binding-v1",source_sha256=SOURCE_SHA,
        startup_control_sha256=controls,parent_checkpoint_sha256=report.PARENT_SHA256))
    learned = seal(dict(schema="sircus-candidate-continuation-run-v1",status="learning_complete_audit_pending",
        synthetic=False,ledger=trained.to_dict(),bindings=bindings,counters=learning_counts,
        checkpoint_sha256=final_sha,journal_head_sha256=events[-1]["fingerprint"]))
    assessed = seal(dict(schema="sircus-candidate-continuation-assessment-result-v1",status="assessment_complete_audit_pending",
        synthetic=False,ledger=ledger.to_dict(),bindings=dict(bindings,native_binding_sha256=binding["fingerprint"]),
        final_checkpoint_sha256=final_sha,model_hashes=model_hashes,journal_head_sha256=H))
    comparison = dict(methods={},primary_comparison="continued_minus_candidate_bank",model_promoted=False,
        geometry_assisted_evaluation=True,inference_searches=0,assessment_optimizer_updates=0,
        bootstrap=dict(replicates=10000,seed=20261128,positions=70))
    for method in report.METHODS:
        comparison["methods"][method] = {}
        for split,n in report.SPLITS.items():
            every = 3 if method == "continued" else 4
            positions = [dict(context_id=c,released=True,captured=i%every == 0,clean=i%(every*2) == 0,decline_reason=None)
                for i,c in enumerate(ids[split])]
            captures,clean = sum(p["captured"] for p in positions),sum(p["clean"] for p in positions)
            comparison["methods"][method][split] = dict(requested=n,released=n,rejected=0,captures=captures,
                clean_captures=clean,capture_rate=captures/n,clean_capture_rate=clean/n,capture_given_release=captures/n,
                rejection_reasons={},per_position=positions)
    comparison["test_pairs"] = {}
    for control in ("parent","candidate_bank"):
        pair = {}
        for field in ("captured","clean"):
            one = comparison["methods"]["continued"]["test"]["per_position"]
            two = comparison["methods"][control]["test"]["per_position"]
            gains = [a["context_id"] for a,b in zip(one,two) if a[field] and not b[field]]
            losses = [a["context_id"] for a,b in zip(one,two) if b[field] and not a[field]]
            pair[field] = dict(gains=len(gains),losses=len(losses),ties=70-len(gains)-len(losses),
                difference_count=len(gains)-len(losses),difference_rate=(len(gains)-len(losses))/70,
                gain_context_ids=gains,loss_context_ids=losses,paired_bootstrap_95_percent_interval=[-.5,.5])
        comparison["test_pairs"]["continued_minus_"+control] = pair
    retention = dict(historical_reference_capture_counts=dict(parent=13,candidate_bank=16),
        historical_ids_available=True,historical_witnesses={})
    for method,n in (("parent",13),("candidate_bank",16)):
        witness = dict(requested_reference=n)
        for prefix,who in (("continued","continued"),("fresh_control",method)):
            losses = [p["context_id"] for p in comparison["methods"][who]["train"]["per_position"][:n] if not p["captured"]]
            witness[prefix+"_loss_context_ids"] = losses
            witness["continued_retained" if prefix == "continued" else "fresh_control_recaptured"] = n-len(losses)
        retention["historical_witnesses"][method] = witness
    comparison["train_retention"] = retention
    startup_receipt = dict(schema="sircus-candidate-continuation-startup-evidence-v1",status="completed")
    request = dict(session=SESSION,source_sha256=SOURCE_SHA,learning_result_sha256=digest(learned),
        assessment_result_sha256=digest(assessed),native_binding_sha256=digest(binding),startup_receipt_sha256=digest(startup_receipt))
    learning_audit = seal(dict(schema="sircus-candidate-continuation-learning-audit-v1",status="passed",synthetic=False,
        bindings=bindings,counters=learning_counts,result_sha256=digest(learned),checkpoint_sha256=final_sha,
        checkpoint_bytes_verified=True,checkpoint_ledger_journal_bound=True))
    native = seal(dict(schema="sircus-candidate-continuation-native-audit-v1",status="passed",synthetic=False,
        ledger=trained.summary(),requests=1400,releases=1400,rejections=0,captures=sum(o["captured"] for o in outcomes),
        clean_captures=sum(o["clean"] for o in outcomes),journal_head_sha256=events[-1]["fingerprint"],
        startup_controls=controls,native_binding_fingerprint=binding["fingerprint"],
        raw_assets_poses_camera_verified=True,parent_checkpoint_bytes_verified=True))
    assessment_audit = seal(dict(schema="sircus-candidate-continuation-assessment-audit-v1",status="passed",synthetic=False,
        ledger=ledger.summary(),requests=420,releases=420,source_result_sha256=digest(assessed),
        final_checkpoint_sha256=final_sha,model_hashes=model_hashes,comparison=comparison,
        native_binding_fingerprint=binding["fingerprint"],journal_head_sha256=H))
    startup = dict(status="passed",cohorts=9,initialized_trajectories=315,physics_steps=1584,
        receipt_sha256=digest(startup_receipt),startup_control_sha256=controls)
    combined = seal(dict(schema="sircus-candidate-continuation-combined-audit-v1",status="passed",all_components_passed=True,
        synthetic=False,session=SESSION,source_sha256=SOURCE_SHA,audit_request_sha256=digest(request),
        final_checkpoint_sha256=final_sha,learning=learning_audit,native_train=native,assessment=assessment_audit,startup=startup,
        audit_cost=dict(wall_s=1.,process_cpu_s=1.,lifetime_peak_rss_mib=100.,device="cpu",cuda_visible_devices=""),
        model_promoted=False,execution_authorized=False,release_authorized=False))
    counters = dict(train_requests=1400,train_releases=1400,assessment_requests=420,assessment_releases=420)
    resources = dict(phases={p:dict(wall_s=2.,aggregate_cpu_s=3.,lifetime_peak_rss_mib=100.,
        largest_child_lifetime_peak_rss_mib=50.,device=None) for p in report.PHASES},scope="Fabricated cost claims only.")
    application = seal(dict(schema="sircus-candidate-continuation-application-v1",status="completed_audited",synthetic=False,
        session=SESSION,counters=counters,audit=combined,assessment_result_sha256=digest(assessed),
        final_checkpoint_sha256=final_sha,resources=resources,native_application_starts=1,
        model_promoted=False,execution_authorized=False,release_authorized=False))
    launch = dict(schema="sircus-candidate-continuation-bounded-launch-v2",status="native_complete_independent_audit_required",
        session=SESSION,mock_only=False,native_application_starts=1,model_promoted=False,counters=counters,
        diagnostics=dict(complete=True,task_requests=0,cohorts=9,initialized_trajectories=315,physics_steps=1584))
    manifest = dict(schema="sircus-candidate-continuation-production-inputs-v1",source_sha256=SOURCE_SHA,
        execution_authorized=False,release_authorized=False,files={k:dict(path="/synthetic/"+k,sha256=v) for k,v in
            dict(parent_checkpoint=report.PARENT_SHA256,train_inputs="3"*64,train_bank="1"*64,fresh_split_manifest="2"*64).items()})
    files = {"application-result.json":application,"combined-audit.json":combined,"launch-result.json":launch,
        "audit-request.json":request,"learning/result.json":learned,"assessment/assessment-result.json":assessed,
        "native-binding.json":binding,"native-setup/startup/receipt.json":startup_receipt,"manifest.json":manifest}
    files.update({f"learning/journal/event-{i:06d}.json":e for i,e in enumerate(events)})
    return files, checkpoint


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.files,self.checkpoint = copy.deepcopy(fixture())
        self.write()

    def write(self):
        for name,value in self.files.items():
            path = self.root/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(raw(value))
        (self.root/"learning/final.pt").write_bytes(self.checkpoint)

    def args(self):
        return dict(manifest_path=self.root/"manifest.json",expected_manifest_sha256=digest(self.files["manifest.json"]),
            expected_application_sha256=digest(self.files["application-result.json"]),expected_audit_sha256=digest(self.files["combined-audit.json"]),
            expected_launch_sha256=digest(self.files["launch-result.json"]),expected_session=SESSION,expected_source_sha256=SOURCE_SHA)

    def build(self):
        return report.build_child(self.root,**self.args())

    def test_closed_audit_counts_curve_resources_and_checkpoint_are_compact(self):
        child = self.build()
        self.assertEqual(child["schema"],report.SCHEMA)
        self.assertEqual(child["counters"]["assessment_requests"],420)
        self.assertEqual(child["train_monitoring"]["totals"]["terminal"],1400)
        self.assertFalse(child["train_monitoring"]["audited"])
        self.assertEqual(child["comparison"]["methods"]["continued"]["test"]["requested"],70)
        self.assertFalse(child["new_trial_videos"])
        self.assertFalse(child["model_promoted"])
        self.assertIsNone(child["resources"]["phases"]["audit"]["device"])
        self.assertIn("does not independently remeasure",child["resource_verification"])
        serialized = json.dumps(child)
        for forbidden in ("/synthetic/","state_dict","workspace_host","board_x_m","candidate_id"):
            self.assertNotIn(forbidden,serialized)
        report.fingerprint(child,"test child")

    def test_each_external_hash_and_session_source_are_mandatory(self):
        for key in ("expected_application_sha256","expected_audit_sha256","expected_launch_sha256","expected_manifest_sha256","expected_source_sha256"):
            values = self.args(); values[key] = "0"*64
            with self.subTest(key=key),self.assertRaises(ValueError): report.build_child(self.root,**values)
        for session in ("", "0"*32, "not-a-session"):
            with self.subTest(session=session),self.assertRaises(ValueError):
                report.build_child(self.root,**dict(self.args(),expected_session=session))

    def test_missing_tampered_linked_model_and_receipt_fail(self):
        path = self.root/"learning/final.pt"; path.write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError,"checkpoint bytes"): self.build()
        path.unlink(); target = self.root/"other.pt"; target.write_bytes(self.checkpoint); path.symlink_to(target)
        with self.assertRaisesRegex(ValueError,"Linked"): self.build()
        path.unlink(); path.write_bytes(self.checkpoint)
        (self.root/"combined-audit.json").write_bytes(b"{}")
        with self.assertRaisesRegex(ValueError,"Externally bound"): self.build()
        (self.root/"combined-audit.json").unlink()
        with self.assertRaisesRegex(ValueError,"Missing"): self.build()

    def test_pending_synthetic_incomplete_components_or_wrong_launch_never_publish(self):
        for change in ("pending","synthetic","component","launch","promotion"):
            with self.subTest(change=change):
                self.files,self.checkpoint = copy.deepcopy(fixture())
                app = self.files["application-result.json"]; audit = self.files["combined-audit.json"]
                if change == "pending": app["status"] = "audit_pending"
                if change == "synthetic": audit["synthetic"] = True
                if change == "component": audit["assessment"]["status"] = "pending"; seal(audit["assessment"])
                if change == "launch": self.files["launch-result.json"]["status"] = "native_failed_no_retry"
                if change == "promotion": app["model_promoted"] = True
                seal(audit); app["audit"] = audit; seal(app); self.write()
                with self.assertRaises(ValueError): self.build()

    def test_audit_input_and_model_joins_survive_resealed_outer_receipts(self):
        for change in ("request","assessment_sha","checkpoint","native_binding","startup"):
            with self.subTest(change=change):
                self.files,self.checkpoint = copy.deepcopy(fixture())
                app = self.files["application-result.json"]; audit = self.files["combined-audit.json"]
                if change == "request": audit["audit_request_sha256"] = "0"*64
                if change == "assessment_sha": app["assessment_result_sha256"] = "0"*64
                if change == "checkpoint": audit["learning"]["checkpoint_sha256"] = "0"*64; seal(audit["learning"])
                if change == "native_binding": audit["assessment"]["native_binding_fingerprint"] = "0"*64; seal(audit["assessment"])
                if change == "startup": audit["startup"]["receipt_sha256"] = "0"*64
                seal(audit); app["audit"] = audit; seal(app); self.write()
                with self.assertRaises(ValueError): self.build()

    def test_partial_or_overspent_denominators_and_inconsistent_costs_fail(self):
        for key,value in (("train_requests",1399),("train_releases",1401),("assessment_requests",419),("assessment_releases",419)):
            self.files,self.checkpoint = copy.deepcopy(fixture())
            self.files["application-result.json"]["counters"][key] = value
            seal(self.files["application-result.json"]); self.write()
            with self.subTest(key=key),self.assertRaises(ValueError): self.build()
        self.files,self.checkpoint = copy.deepcopy(fixture())
        app = self.files["application-result.json"]; app["resources"]["phases"]["train"]["wall_s"] = -1
        seal(app); self.write()
        with self.assertRaisesRegex(ValueError,"measurement"): self.build()

    def test_comparison_positions_pairing_and_historical_retention_are_recounted(self):
        for change in ("count","position","pair","retention"):
            self.files,self.checkpoint = copy.deepcopy(fixture())
            app = self.files["application-result.json"]; audit = self.files["combined-audit.json"]
            comparison = audit["assessment"]["comparison"]
            if change == "count": comparison["methods"]["continued"]["test"]["captures"] += 1
            if change == "position": comparison["methods"]["continued"]["test"]["per_position"][0]["context_id"] = "other"
            if change == "pair": comparison["test_pairs"]["continued_minus_parent"]["captured"]["gains"] += 1
            if change == "retention": comparison["train_retention"]["historical_reference_capture_counts"]["parent"] = 12
            seal(audit["assessment"]); seal(audit); app["audit"] = audit; seal(app); self.write()
            with self.subTest(change=change),self.assertRaises(ValueError): self.build()

    def test_changed_journal_or_extra_file_does_not_produce_a_curve(self):
        path = self.root/"learning/journal/event-000001.json"
        value = json.loads(path.read_bytes()); value["payload"]["results"][0]["captured"] = False
        path.write_bytes(raw(value))
        with self.assertRaisesRegex(ValueError,"fingerprint"): self.build()
        self.write(); (path.parent/".pending-crash").write_text("synthetic partial")
        with self.assertRaisesRegex(ValueError,"partial"): self.build()

    def test_missing_nested_proofs_and_malformed_types_fail_closed(self):
        for change in ("proof","source","startup","counter_type"):
            self.files,self.checkpoint = copy.deepcopy(fixture())
            app,audit = self.files["application-result.json"],self.files["combined-audit.json"]
            if change == "proof":
                audit["learning"].pop("checkpoint_ledger_journal_bound"); seal(audit["learning"])
            if change == "source": self.files["manifest.json"]["files"].pop("train_inputs")
            if change == "startup": self.files["launch-result.json"]["diagnostics"]["cohorts"] = 15
            if change == "counter_type": app["counters"]["train_requests"] = True
            seal(audit); app["audit"] = audit; seal(app); self.write()
            with self.subTest(change=change),self.assertRaises(ValueError): self.build()

    def test_additive_update_preserves_root_and_prior_attempts_without_stale_running_copy(self):
        child = self.build()
        old = dict(schema="sircus-candidate-native-comparison-report-v1",methods={"historical":"unchanged"},
            audit={"status":"passed"},continuation=dict(status="running",session=SESSION,prior_attempts=[{"status":"failed","session":"1"*32}]))
        before = copy.deepcopy(old); updated = report.add_continuation(old,child)
        self.assertEqual(old,before)
        self.assertEqual({k:v for k,v in updated.items() if k != "continuation"},{k:v for k,v in old.items() if k != "continuation"})
        self.assertEqual(updated["continuation"]["prior_attempts"],old["continuation"]["prior_attempts"])
        self.assertEqual(report.add_continuation(updated,child),updated)
        old["continuation"].update(status="failed",session="2"*32)
        updated = report.add_continuation(old,child)
        self.assertEqual(len(updated["continuation"]["prior_attempts"]),2)

    def test_module_is_standard_library_only_and_has_no_execution_or_write_entry(self):
        tree = ast.parse(Path(report.__file__).read_text())
        imports = {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)}
        imports.update(a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names)
        self.assertTrue(imports <= {"collections","copy","hashlib","json","math","pathlib","random","re"})
        calls = [n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
        for forbidden in ("write_text","write_bytes","load","save","step","backward","Popen","run","acquire","unlink"):
            self.assertNotIn(forbidden,calls)

    def test_late_publication_cannot_displace_a_different_active_session(self):
        child = self.build()
        for status in ("prepared","queued","running","audit_pending","native_complete","unknown"):
            current = dict(status=status,session="d"*32,prior_attempts=[{"status":"failed","session":"1"*32}])
            historical = dict(schema="sircus-candidate-native-comparison-report-v1",continuation=current,
                methods={"historical":"unchanged"})
            before = copy.deepcopy(historical)
            with self.subTest(status=status),self.assertRaisesRegex(ValueError,"different-session active"):
                report.add_continuation(historical,child)
            self.assertEqual(historical,before)
            historical["continuation"]["session"] = SESSION
            updated = report.add_continuation(historical,child)
            self.assertEqual(updated["continuation"]["session"],SESSION)
            self.assertEqual(updated["continuation"]["prior_attempts"],current["prior_attempts"])


if __name__ == "__main__": unittest.main()
