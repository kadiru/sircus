"""Entry/combined-audit wiring with fake components; never creates an app."""

import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock,patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import candidate_continuation_entry as subject
import candidate_continuation_production as production
import candidate_continuation_watchdog as watchdog


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


class EntryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.output=Path(self.temp.name)
        self.session="a"*32
        self.argv=["--manifest",str(self.output/"manifest.json"),"--expected-manifest-sha256","b"*64,
            "--output",str(self.output),"--session",self.session]
        self.construct=patch.object(production,"ProductionComponents",return_value=object())
        self.inputs=patch.object(production,"production_inputs",return_value=dict(mock_inputs=True))
        self.publisher=patch.object(watchdog,"ProgressWriter",return_value=object())
        self.pipeline=patch.object(subject,"run_pipeline",return_value=dict(mock_complete=True))
        self.audit=patch.object(subject,"audit_application",return_value=dict(mock_audit=True))
        for p in (self.construct,self.inputs,self.publisher,self.pipeline,self.audit):
            p.start();self.addCleanup(p.stop)
        self.env=patch.dict(os.environ,{},clear=True);self.env.start();self.addCleanup(self.env.stop)

    def assert_inert(self):
        production.ProductionComponents.assert_not_called();production.production_inputs.assert_not_called()
        watchdog.ProgressWriter.assert_not_called();subject.run_pipeline.assert_not_called();subject.audit_application.assert_not_called()

    def test_default_or_both_modes_refuse_before_any_production_object(self):
        for flags in ([],["--execute-approved","--audit-only"]):
            with self.subTest(flags=flags),self.assertRaisesRegex(ValueError,"explicit future execution"):
                subject.main(self.argv+flags)
        self.assert_inert()

    def test_parser_requires_manifest_sha_output_and_session(self):
        for key in ("--manifest","--expected-manifest-sha256","--output","--session"):
            args=list(self.argv);index=args.index(key);del args[index:index+2]
            with self.subTest(key=key),contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):subject.main(args)
        self.assert_inert()

    def test_malformed_session_is_rejected_before_components(self):
        args=list(self.argv);args[args.index("--session")+1]="not-a-watchdog-session"
        os.environ["SIRCUS_CONTINUATION_SESSION"]="not-a-watchdog-session"
        with self.assertRaisesRegex(ValueError,"session"):
            subject.main(args+["--execute-approved","--headless","--enable_cameras"])
        self.assert_inert()

    def test_missing_native_flags_wrong_device_and_missing_cookie_refuse_before_construction(self):
        cases=(["--execute-approved"],
            ["--execute-approved","--headless"],
            ["--execute-approved","--headless","--enable_cameras","--device","cuda:1"],
            ["--execute-approved","--headless","--enable_cameras"])
        for flags in cases:
            with self.subTest(flags=flags),self.assertRaises(ValueError):subject.main(self.argv+flags)
        self.assert_inert()

    def test_approved_execution_mode_only_wires_mock_components_with_watchdog_cookie(self):
        os.environ["SIRCUS_CONTINUATION_SESSION"]=self.session
        result=subject.main(self.argv+["--execute-approved","--headless","--enable_cameras"])
        self.assertEqual(result,dict(mock_complete=True))
        production.ProductionComponents.assert_called_once_with(self.output/"manifest.json","b"*64,self.output,self.session,device="cuda:0")
        watchdog.ProgressWriter.assert_called_once_with(self.output,self.session)
        subject.run_pipeline.assert_called_once()
        production.production_inputs.assert_not_called();subject.audit_application.assert_not_called()

    def test_missing_or_wrong_output_is_not_created(self):
        absent=self.output/"absent"
        args=list(self.argv);args[args.index("--output")+1]=str(absent)
        with self.assertRaisesRegex(ValueError,"Supervisor-owned"):subject.main(args+["--audit-only"])
        self.assertFalse(absent.exists());self.assert_inert()

    def test_audit_only_requires_external_request_sha_before_loading_inputs(self):
        os.environ["CUDA_VISIBLE_DEVICES"]=""
        with self.assertRaisesRegex(ValueError,"External audit-request SHA"):
            subject.main(self.argv+["--audit-only"])
        self.assert_inert()

    def test_audit_only_requires_cuda_hidden_before_loading_inputs(self):
        with self.assertRaisesRegex(ValueError,"hide CUDA"):
            subject.main(self.argv+["--audit-only","--expected-audit-request-sha256","c"*64])
        self.assert_inert()

    def test_audit_only_routes_to_cpu_reader_without_app_or_pipeline(self):
        os.environ["CUDA_VISIBLE_DEVICES"]=""
        result=subject.main(self.argv+["--audit-only","--expected-audit-request-sha256","c"*64])
        self.assertEqual(result,dict(mock_audit=True))
        production.production_inputs.assert_called_once_with(self.output/"manifest.json","b"*64)
        subject.audit_application.assert_called_once_with(self.output,dict(mock_inputs=True),self.session,
            expected_request_sha256="c"*64)
        production.ProductionComponents.assert_not_called();watchdog.ProgressWriter.assert_not_called();subject.run_pipeline.assert_not_called()


class CombinedAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.output=Path(self.temp.name);self.session="a"*32;self.final_sha="f"*64;self.synthetic=True
        self.learned=dict(ledger=dict(mock_ledger=True),journal_head_sha256="c"*64,
            synthetic=self.synthetic,checkpoint_sha256=self.final_sha)
        self.assessed=dict(journal_head_sha256="d"*64,synthetic=self.synthetic,final_checkpoint_sha256=self.final_sha)
        self.controls={"probes.json":"1"*64,"rim-rays.json":"2"*64}
        self.native=dict(startup_control_sha256=self.controls)
        save(self.output/"learning/result.json",self.learned)
        save(self.output/"assessment/assessment-result.json",self.assessed)
        save(self.output/"native-binding.json",self.native)
        save(self.output/"native-setup/startup/receipt.json",dict(mock_startup=True))
        self.request=dict(session=self.session,source_sha256="b"*64,
            learning_result_sha256=subject.sha(self.output/"learning/result.json"),
            assessment_result_sha256=subject.sha(self.output/"assessment/assessment-result.json"),
            native_binding_sha256=subject.sha(self.output/"native-binding.json"),
            startup_receipt_sha256=subject.sha(self.output/"native-setup/startup/receipt.json"))
        self.bind_request()
        self.inputs=dict(paths={k:self.output/(k+".json") for k in ("dropper_urdf","board_urdf","poses","camera","parent_checkpoint")},
            bindings=dict(source_sha256="b"*64),train_supports=["mock_support"],
            train=dict(demonstrations=["mock_demo"],cells=[dict(id="mock_train")]),train_records={},records={},supports={},
            cells_by_split=dict(train=[],validation=[],test=[]),historical_capture_ids={},
            values=dict(scene=dict(mock_scene=True),source_contract=dict(cells=[dict(id="original_startup_cell")])))
        self.ledger=types.SimpleNamespace(train_context_ids=tuple(f"train-{i:03d}" for i in range(35)))
        self.ledger_patch=patch.object(subject.Ledger,"from_dict",return_value=self.ledger)
        self.ledger_patch.start();self.addCleanup(self.ledger_patch.stop)
        self.context=dict(bindings=dict(mock_binding="a"*64),parent_distributions={},continued_distributions={},
            model_hashes=dict(parent_weight_hash="e"*64,continued_weight_hash="f"*64))
        self.checks=dict(learning=Mock(return_value=dict(status="passed",synthetic=True,checkpoint_sha256=self.final_sha)),
            native=Mock(return_value=dict(status="passed",synthetic=True)),
            assessment=Mock(return_value=dict(status="passed",synthetic=True,final_checkpoint_sha256=self.final_sha)),
            startup=Mock(return_value=dict(status="passed",startup_control_sha256=self.controls)),
            context=Mock(return_value=self.context))
        camera=types.ModuleType("camera_options");camera.preview_origins=Mock(return_value="mock_origins")
        self.camera_patch=patch.dict(sys.modules,{"camera_options":camera});self.camera_patch.start();self.addCleanup(self.camera_patch.stop)
        self.env=patch.dict(os.environ,{"CUDA_VISIBLE_DEVICES":""},clear=True);self.env.start();self.addCleanup(self.env.stop)

    def bind_request(self):
        save(self.output/"audit-request.json",self.request)
        self.request_sha=subject.sha(self.output/"audit-request.json")

    def audit(self,**changes):
        kwargs=dict(expected_request_sha256=self.request_sha,checks=self.checks);kwargs.update(changes)
        return subject.audit_application(self.output,self.inputs,self.session,**kwargs)

    def test_four_independent_components_join_with_exact_bytes_and_frozen_context(self):
        result=self.audit()
        self.assertEqual(result["status"],"passed");self.assertTrue(result["all_components_passed"])
        self.assertTrue(result["synthetic"]);self.assertFalse(result["model_promoted"])
        self.assertEqual(result["final_checkpoint_sha256"],self.final_sha)
        self.assertEqual(result["audit_cost"]["device"],"cpu")
        self.assertEqual(result["audit_cost"]["cuda_visible_devices"],"")
        for check in self.checks.values():check.assert_called_once()
        self.checks["context"].assert_called_once_with(self.inputs,self.ledger,self.learned,self.output/"learning/final.pt",self.native)
        call=self.checks["assessment"].call_args.kwargs
        self.assertEqual(call["expected_result_sha256"],self.request["assessment_result_sha256"])
        self.assertIs(call["train_ledger"],self.ledger)
        self.assertEqual(call["final_checkpoint_path"],self.output/"learning/final.pt")
        self.assertEqual(call["startup_directory"],self.output/"native-setup")
        self.assertTrue((self.output/"combined-audit.json").is_file())
        with self.assertRaises(FileExistsError):self.audit()  # No overwrite or second publication.

    def test_cuda_visible_or_bad_external_request_sha_is_rejected_before_components(self):
        os.environ["CUDA_VISIBLE_DEVICES"]="0"
        with self.assertRaisesRegex(ValueError,"hide CUDA"):self.audit()
        os.environ["CUDA_VISIBLE_DEVICES"]=""
        with self.assertRaisesRegex(ValueError,"request bytes"):self.audit(expected_request_sha256="0"*64)
        for check in self.checks.values():check.assert_not_called()

    def test_rehashed_wrong_session_or_source_never_opens_components(self):
        for field in ("session","source_sha256"):
            original=self.request[field];self.request[field]="wrong";self.bind_request()
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,"source/session"):self.audit()
            self.request[field]=original
        for check in self.checks.values():check.assert_not_called()

    def test_result_native_or_startup_bytes_tamper_is_detected_before_verifiers(self):
        for relative in ("learning/result.json","assessment/assessment-result.json","native-binding.json","native-setup/startup/receipt.json"):
            path=self.output/relative;raw=path.read_bytes();path.write_bytes(raw+b" ")
            with self.subTest(path=relative),self.assertRaisesRegex(ValueError,"bytes changed"):self.audit()
            path.write_bytes(raw)
        for check in self.checks.values():check.assert_not_called()

    def test_any_component_failure_cannot_publish_or_promote(self):
        for name in ("learning","native","assessment","startup"):
            original=copy.deepcopy(self.checks[name].return_value)
            self.checks[name].return_value=dict(original,status="failed")
            with self.subTest(name=name),self.assertRaisesRegex(ValueError,"component failed"):self.audit()
            self.checks[name].return_value=original
            self.assertFalse((self.output/"combined-audit.json").exists())

    def test_exception_in_reader_preserves_failure_and_has_no_retry(self):
        self.checks["learning"].side_effect=RuntimeError("mock numerical failure")
        with self.assertRaisesRegex(RuntimeError,"mock numerical failure"):self.audit()
        self.checks["learning"].assert_called_once();self.checks["native"].assert_not_called()
        self.assertFalse((self.output/"combined-audit.json").exists())

    def test_startup_control_bindings_must_match_every_native_trial(self):
        self.checks["startup"].return_value=dict(status="passed",startup_control_sha256={"different":"3"*64})
        with self.assertRaisesRegex(ValueError,"startup controls differ"):self.audit()
        self.assertFalse((self.output/"combined-audit.json").exists())

    def test_synthetic_provenance_cannot_be_hidden_by_combined_header(self):
        self.assessed["synthetic"]=False;save(self.output/"assessment/assessment-result.json",self.assessed)
        self.request["assessment_result_sha256"]=subject.sha(self.output/"assessment/assessment-result.json");self.bind_request()
        with self.assertRaisesRegex(ValueError,"provenance|synthetic"):self.audit()
        self.assertFalse((self.output/"combined-audit.json").exists())

    def test_each_subaudit_provenance_must_match_its_bound_history(self):
        for name in ("learning","native","assessment"):
            original=copy.deepcopy(self.checks[name].return_value)
            self.checks[name].return_value["synthetic"]=False
            with self.subTest(name=name),self.assertRaisesRegex(ValueError,"provenance"):
                self.audit()
            self.checks[name].return_value=original
            self.assertFalse((self.output/"combined-audit.json").exists())

    def test_consistent_real_flags_remain_real_without_auto_promotion(self):
        self.learned["synthetic"]=self.assessed["synthetic"]=False
        save(self.output/"learning/result.json",self.learned);save(self.output/"assessment/assessment-result.json",self.assessed)
        self.request["learning_result_sha256"]=subject.sha(self.output/"learning/result.json")
        self.request["assessment_result_sha256"]=subject.sha(self.output/"assessment/assessment-result.json");self.bind_request()
        for name in ("learning","native","assessment"):self.checks[name].return_value["synthetic"]=False
        result=self.audit()
        self.assertFalse(result["synthetic"]);self.assertFalse(result["model_promoted"])
        self.assertFalse(result["execution_authorized"]);self.assertFalse(result["release_authorized"])


if __name__=="__main__":unittest.main()
