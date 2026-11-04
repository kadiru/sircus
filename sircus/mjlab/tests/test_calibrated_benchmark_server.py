"""Read-only projection and byte-bound disclosure controls; no sockets/GPU."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from sircus.mjlab import calibrated_benchmark_contract as contract
from sircus.mjlab import calibrated_benchmark_server as server
from sircus.mjlab.calibrated_admission import HOME_VERSION, _signed, validate_home
from sircus.mjlab.calibrated_score import CALIBRATION_VERSION, calibration_fingerprint
from sircus.mjlab.interface import RequestError
from sircus.mjlab.learning_contract import digest


class CalibratedServerTests(unittest.TestCase):
    def write(self, name, value):
        path = self.directory / name
        path.write_text(json.dumps(value, allow_nan=False))
        return path

    @staticmethod
    def sign(value):
        value.pop("fingerprint", None); value["fingerprint"] = digest(value)
        return value

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.directory = Path(self.temp.name)
        config = dict(version=CALIBRATION_VERSION, wall_tolerance_m=.001, floor_tolerance_m=.002,
            geometry_epsilon_m=1e-6, max_speed_m_s=20., retention_s=1., native_dt_s=1/15360,
            compiled_planes=[[1.,0.,0.,.1],[0.,1.,0.,.1],[-1.,0.,0.,.1]])
        config["fingerprint"] = calibration_fingerprint(config)
        self.write("calibration.json", dict(status="passed", checks=5, synthetic_fixture=False,
            score_config=config, score_config_sha256=digest(config), version="cal-controls"))
        self.write("verification.json", dict(status="passed", checks=8, synthetic_fixture=False,
            calibration_sha256=server.sha(self.directory/"calibration.json"), score_config_sha256=digest(config)))
        self.write("setup-home.json", _signed(dict(version=HOME_VERSION, passed=True, scene_sha256=digest({}), q10=[0.]*6,q14=[0.]*6)))
        self.calibration = contract.checked_calibration(self.directory/"calibration.json",self.directory/"verification.json")
        self.setup = contract.checked_setup_home(self.directory/"setup-home.json")
        self.identity = dict(scene_sha256="a"*64,source_sha256={"native":"b"*64}, calibration=self.calibration,setup=self.setup)
        self.protocol = contract.protocol()
        self.protocol["prior"] = dict(recipes=17,full_historical_cost_known=False)
        self.spec = self.sign(dict(version=contract.VERSION,identity=self.identity,protocol=self.protocol))
        self.write("contract.json", self.spec)
        self.raw = self.sign(dict(version=contract.VERSION,status="running",phase="training",protocol=self.protocol,
            contract_fingerprint=self.spec["fingerprint"],curves=[],runs=[],final_test=[]))
        self.write("report.json",self.raw)
        self.write("manifest.json",{})
        self.patches = [patch.object(server.GridState,"__init__",return_value=None),
            patch.object(server.GridState,"report",return_value={"preserved_history":True}),
            patch.object(server,"benchmark_helpers",return_value=(contract.checked_calibration,contract.checked_setup_home,
                lambda ref,cal,setup:dict(self.identity,calibration=cal,setup=setup),contract.validate_report,validate_home))]
        for p in self.patches: p.start()
        self.state = self.make_state()

    def tearDown(self):
        for p in reversed(self.patches): p.stop()
        self.temp.cleanup()

    def make_state(self):
        return server.CalibratedBenchmarkState({"scene":{}},self.directory,self.directory/"display.json",None,None,self.directory,
            benchmark_audit=self.directory/"independent-audit.json")

    def complete(self):
        counts = dict(attempts=36,captures=2,native_releases=34,invalid=2,clean=1,mean_reward=.2)
        self.raw["curves"] = [dict(method=m["id"],seed=seed,samples=n,validation=counts,resources={})
            for m in self.protocol["methods"] for seed in self.protocol["seeds"] for n in self.protocol["milestones"]]
        self.raw["final_test"] = [dict(method=m["id"],seed=seed,selected_samples=0,
            counts=dict(counts,attempts=72,native_releases=70),frozen_artifact_sha256="c"*64)
            for m in self.protocol["methods"] for seed in self.protocol["seeds"]]
        self.raw["status"] = "completed"
        self.write("report.json",self.sign(self.raw))

    def audit(self, **extra):
        value = dict(version=contract.AUDIT_VERSION,status="passed",checks=11,failures=[],
            study_kind="native_current_ball_v2",synthetic_fixture=False,
            contract_fingerprint=self.spec["fingerprint"],report_sha256=server.sha(self.directory/"report.json"),
            report_fingerprint=self.raw["fingerprint"],study_manifest_sha256=server.sha(self.directory/"manifest.json"),
            source_sha256=self.identity["source_sha256"],
            auditor_sha256=server.sha(Path(server.__file__).with_name("calibrated_benchmark_verify.py")),
            auditor_dependencies={
                "sircus/mjlab/benchmark_verify.py":server.sha(Path(server.__file__).with_name("benchmark_verify.py")),
                "sircus/mjlab/verify_capture_calibration.py":server.sha(Path(server.__file__).with_name("verify_capture_calibration.py"))})
        self.write("independent-audit.json",dict(value,**extra))

    def test_preserves_grid_read_only_and_live_train_counts(self):
        value = self.state.report()
        self.assertTrue(value["preserved_history"]);self.assertTrue(value["benchmark_read_only"])
        self.assertEqual(value["sweep_rl_benchmark"]["test_disclosure"],"hidden_until_completed_and_audited")
        self.assertNotIn("test",value["sweep_rl_benchmark"].get("audit",{}).get("role_counts",{}))

    def test_completed_test_hidden_until_exact_passed_audit(self):
        self.complete()
        self.assertEqual(self.state.benchmark_report()["final_test"],[])
        self.audit()
        self.assertEqual(len(self.state.benchmark_report()["final_test"]),6)
        self.assertEqual(self.state.benchmark_report()["test_disclosure"],"verified")

    def test_both_exact_auditor_dependency_hashes_required(self):
        self.complete();self.audit()
        valid=json.loads((self.directory/"independent-audit.json").read_text())["auditor_dependencies"]
        self.assertEqual(len(self.state.benchmark_report()["final_test"]),6)
        for key in valid:
            missing=dict(valid);missing.pop(key)
            wrong=dict(valid);wrong[key]="f"*64
            for changed in (missing,wrong):
                with self.subTest(key=key,changed=changed):
                    self.audit(auditor_dependencies=changed)
                    with self.assertRaises(RequestError):self.state.benchmark_report()
        self.audit(auditor_dependencies=dict(valid,unrecognized="a"*64))
        with self.assertRaises(RequestError):self.state.benchmark_report()

    def test_reader_revision_provenance_keeps_producer_contract(self):
        value=self.state.benchmark_report()["reader_provenance"]
        self.assertEqual(value["version"],server.READER_VERSION)
        self.assertEqual(value["reader_source_sha256"],server.sha(Path(server.__file__)))
        self.assertEqual(value["producer_contract_fingerprint"],self.spec["fingerprint"])
        self.assertIn("producer C",value["scope"])

    def test_audit_cannot_unlock_running_test_or_leak_role_counts(self):
        self.audit(role_counts={"test":dict(attempts=72,captures=70)})
        value=self.state.benchmark_report()
        self.assertEqual(value["final_test"],[]);self.assertNotIn("role_counts",value["audit"])

    def test_wrong_control_bytes_fail_closed_after_startup(self):
        for name in ("contract.json","calibration.json","verification.json","setup-home.json"):
            path=self.directory/name; raw=path.read_bytes();path.write_bytes(raw+b"\n")
            with self.assertRaises(RequestError):self.state.benchmark_report()
            path.write_bytes(raw)

    def test_synthetic_controls_not_served_as_native(self):
        value=json.loads((self.directory/"calibration.json").read_text());value["synthetic_fixture"]=True
        self.write("calibration.json",value)
        with self.assertRaises(RequestError):self.make_state()

    def test_changed_report_or_manifest_or_audit_source_fails_closed(self):
        self.complete();self.audit()
        for kwargs in ({"report_sha256":"f"*64},{"report_fingerprint":"f"*64},
                       {"study_manifest_sha256":"f"*64},{"source_sha256":{}},
                       {"auditor_sha256":"f"*64},{"synthetic_fixture":True},
                       {"checks":0},{"checks":True},{"failures":["mismatch"]}):
            self.audit(**kwargs)
            with self.assertRaises(RequestError):self.state.benchmark_report()

    def test_public_projection_strips_paths_raw_rows_and_test_details(self):
        raw=copy.deepcopy(self.raw)
        raw.update(command=["private"],raw_rows=[{"private":True}])
        raw["runs"]=[dict(method="ppo",seed=7,train_counts=dict(attempts=32,captures=0),
            admission=dict(requested=32,native_releases=30,rejected=2,reasons={"finite_ik_unresolved":2},private="/private"),
            resources=dict(path_s=2,context_resolution_s=3,private="/private"),checkpoint_path="/private")]
        raw["final_test"]=[dict(counts=dict(attempts=72,captures=71),raw_rows=["/private"])]
        value=server.public_calibrated_benchmark(raw,self.identity)
        self.assertEqual(value["runs"][0]["resources"]["path_s"],2)
        self.assertEqual(value["runs"][0]["admission"]["reasons"],{"finite_ik_unresolved":2})
        self.assertNotIn("/private",json.dumps(value));self.assertNotIn("raw_rows",json.dumps(value))
        self.assertEqual(value["final_test"],[])

    def test_post_is_blocked_without_invoking_state_or_solver(self):
        handler=server.calibrated_handler(Mock())
        obj=object.__new__(handler);obj._json=Mock(return_value="blocked")
        self.assertEqual(obj.do_POST(),"blocked");self.assertEqual(obj._json.call_args.args[0],405)

    def test_repository_historical_v1_archive_remains_original_and_separate(self):
        archive=Path(__file__).resolve().parents[3]/"baselines/mjlab-sweep-rl-benchmark-20261103"
        self.state.historical_benchmark=archive
        self.state.history_manifest_sha256=server.sha(archive/"SHA256SUMS")
        value=self.state.history_report()
        self.assertEqual(value["version"],"sircus-sweep-rl-benchmark-v1")
        self.assertTrue(value["historical"]);self.assertEqual(len(value["final_test"]),6)
        self.assertIn("strict",value["metric"])


if __name__ == "__main__": unittest.main()
