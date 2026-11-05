"""Synthetic dependency-injected controller fixtures, never native evidence."""

from contextlib import contextmanager
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from sircus.mjlab import clean650_contract as protocol, clean650_study as study
from sircus.mjlab.clean650_native import NativeExecutor
from sircus.mjlab.corrected_contract import sha
from sircus.mjlab.learning_contract import digest


SCENE = dict(ball=dict(radius_m=.0375))


def sources():
    return [dict(id=ident, role=role, source_record_id="historical-"+ident, source_record_sha256="a"*64,
                 values=[.47, -.27, .38, 28., .3868], bucket_xy_m=[.1, -.6], q10=[.1]*6, q14=[.2]*6)
            for ident, role in (("positive-control", "positive_control"), ("candidate-650", "candidate"),
                                ("negative-control", "negative_control"))]


class GeometryFixture:
    def __init__(self, *args): self.proofs, self.calls = [], []
    def check_many(self, requests):
        self.calls.append(copy.deepcopy(requests))
        result = []
        for request in requests:
            group = copy.deepcopy(request)
            if group.get("freeze_joints") is True:
                assert "q10" in group and "q14" in group
            else:
                group.setdefault("q10", [group["values"][2]]*6)
                group.setdefault("q14", [group["values"][3]/100]*6)
            proof = dict(group_id=group["id"], eligible=True, values=group["values"], bucket_xy_m=group["bucket_xy_m"])
            group.update(eligible=True, geometry_row=len(self.proofs), geometry_sha256=digest(proof))
            self.proofs.append(proof)
            result.append(group)
        return result


class ExecutorFixture:
    def __init__(self, scene, assets, calibration, output, batch, publish, *, fail_initial=False):
        self.output, self.publish, self.fail_initial = Path(output), publish, fail_initial
        self.rows, self.positions, self.calls = [], {}, []

    def execute(self, requests):
        self.calls.append(copy.deepcopy(requests))
        for request in requests:
            success = request["role"] != "negative_control"
            if self.fail_initial and request["phase"] == "controls_pre" and request["role"] == "positive_control" and request["substeps"] == 64 and request["repeat_index"] == 0:
                success = False
            row = dict(request, native_release=True, invalid=None, invalid_physics=False,
                       route_success=success, route_quality=2 if success else 0,
                       whole_ball_entry=success, board_bounce=True, max_retention_s=1. if success else 0.,
                       scalar_score_audit_passed=True, rim_band_contact=False, external_contact_steps=0,
                       minimum_mouth_clearance_mm=2. if success else None, closest_opening_distance_m=.01 if success else .8,
                       fixture_native_metadata=dict(synthetic=True))
            self.rows.append(row)
            ticks = round(.350*240*row["substeps"])+1
            self.positions[row["id"]] = np.broadcast_to(np.array([0., 0., .4]), (ticks, 3))
            with (self.output/"records.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False)+"\n")
        if requests:
            self.publish(requests[0]["phase"])


class CleanControllerTests(unittest.TestCase):
    def run_fixture(self, fail_initial=False):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        old = root/"old"; old.mkdir()
        assets = root/"assets"; assets.mkdir()
        (root/"asset.txt").write_text("synthetic asset identity")
        packages = {"synthetic": "fixture"}
        (old/"contract.json").write_text(json.dumps(dict(packages=packages, asset_sha256={"asset.txt":sha(root/"asset.txt")})))
        calibration, verification = root/"calibration.json", root/"verification.json"
        calibration.write_text("{}"); verification.write_text("{}")
        args = SimpleNamespace(old_study=old, assets=assets, calibration_report=calibration,
                               verification_report=verification, output=root/"evidence", batch=32, wall_time_limit=7200)
        config = dict(calibration_sha256="b"*64, verification_sha256="c"*64, score_config_sha256="d"*64, score_config={})
        geometry = GeometryFixture()
        executors = []
        def executor(*values):
            instance = ExecutorFixture(*values, fail_initial=fail_initial)
            executors.append(instance)
            return instance
        with patch.object(protocol, "load_inputs", return_value=(SCENE, sources(), config, {})), \
             patch.object(study, "validate_replay_packages", return_value=packages), \
             patch.object(study.importlib.metadata, "version", return_value="fixture"), \
             patch("sircus.mjlab.learning_solve.required_asset_paths", return_value=["asset.txt"]), \
             patch.object(study, "source_manifest", return_value={"synthetic-fixture.py":"e"*64}), \
             patch.object(study, "storage_check", return_value=dict(free_bytes=10**10, study_bytes=0)), \
             patch("sircus.mjlab.clean650_geometry.CleanGeometry", return_value=geometry), \
             patch("sircus.mjlab.clean650_native.NativeExecutor", side_effect=executor), \
             patch("builtins.print"):
            result = study.run(args)
        return args.output, result, geometry, executors[0]

    def assert_manifest(self, output, report):
        manifest = json.loads((output/"evidence-manifest.json").read_text())
        self.assertEqual(report["evidence_sha256"], manifest)
        for name, expected in manifest.items():
            self.assertEqual(sha(output/name), expected, name)

    def test_failed_initial_controls_skip_all_search_and_post_controls(self):
        output, report, geometry, executor = self.run_fixture(fail_initial=True)
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["gate"]["status"], "blocked_controls")
        self.assertEqual(report["counts"]["attempts"], 12)
        self.assertEqual({r["phase"] for r in executor.rows}, {"controls_pre"})
        self.assertEqual(report["control_gates"]["controls_post"]["status"], "not_run")
        self.assertEqual(report["verification"], [])
        self.assertEqual(len(geometry.calls), 1)
        for phase in ("refinement", "verification"):
            self.assertEqual(json.loads((output/(phase+"-plan.json")).read_text()), [])
            self.assertEqual(json.loads((output/(phase+"-selection.json")).read_text())["source_ids"], [])
        self.assertFalse(report["gate"]["more_ppo_started"])
        self.assert_manifest(output, report)

    def test_open_controller_pays_declared_budget_and_checks_controls_afterward(self):
        output, report, geometry, executor = self.run_fixture()
        self.assertEqual(report["counts"]["attempts"], 496)
        self.assertEqual(report["counts"]["phase_counts"], dict(controls_pre=12, coarse=361, refinement=81, verification=30, controls_post=12))
        self.assertEqual(report["gate"]["qualified_settings"], 5)
        self.assertEqual(report["gate"]["status"], "passed")
        self.assertTrue(all(g["freeze_joints"] is True for g in geometry.calls[-1]))
        rows = [json.loads(line) for line in (output/"records.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 496)
        self.assertTrue(all(r["fixture_native_metadata"] == dict(synthetic=True) for r in rows))
        self.assertEqual([r["phase"] for r in rows[-12:]], ["controls_post"]*12)
        initial = json.loads((output/"initial-plan.json").read_text())
        self.assertEqual(len(initial["coarse"]), 361)
        self.assertEqual(initial["controls_pre"][0]["q10"], sources()[0]["q10"])
        stages = json.loads((output/"adaptive-stages.json").read_text())
        self.assertEqual([r["phase"] for r in stages], ["refinement", "verification"])
        for stage in stages:
            for name, expected in stage["files_sha256"].items():
                self.assertEqual(sha(output/name), expected)
        self.assertEqual(len(report["trace_examples"]), 17)
        self.assert_manifest(output, report)


class NativeExecutionTests(unittest.TestCase):
    def test_exact_rows_keep_all_native_evidence_and_rejections_keep_denominators(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output/"native/64").mkdir(parents=True)
            publications = []
            executor = NativeExecutor({}, None, {}, output, 32, publications.append)
            env = SimpleNamespace()
            @contextmanager
            def endpoints(current, bucket, values, q10, q14, allowed):
                current.inputs = dict(bucket=bucket, values=values, q10=q10, q14=q14)
                yield
            def run(bucket, values, constraints, record):
                self.assertTrue(record)
                env.last_arrays = dict(position_m=np.zeros((3,len(bucket),3)), time_s=np.array([0., .1, .2]))
                return [dict(values=values[i].tolist(), bucket_xy_m=bucket[i].tolist(),
                             q10=env.inputs["q10"][i].tolist(), q14=env.inputs["q14"][i].tolist(),
                             native_release=True, route_success=True, invalid=None,
                             native_initial_q10=env.inputs["q10"][i].tolist(), native_initial_q14=env.inputs["q14"][i].tolist(),
                             sentinel_native_field=dict(contact_ids=[1,2], clock="observed"), trajectory_m=[[0,0,0]])
                        for i in range(len(bucket))], dict(synthetic=True)
            env.run = run
            groups = [dict(sources()[0], eligible=True, geometry_row=0)]
            accepted = protocol.plan(groups, "controls_pre")[0]
            rejected = dict(accepted, id="rejected-request", eligible=False)
            with patch.object(executor, "environment", return_value=env), patch("sircus.mjlab.bank_geometry.fixed_endpoints", endpoints):
                executor.execute([accepted, rejected])
            self.assertEqual(len(executor.rows), 2)
            a, b = executor.rows
            self.assertEqual(a["sentinel_native_field"], dict(contact_ids=[1,2], clock="observed"))
            self.assertNotIn("trajectory_m", a)
            self.assertEqual(a["native_initial_q10"], accepted["q10"])
            self.assertFalse(b["native_release"])
            self.assertEqual(b["invalid"], "strict_endpoint_rejected")
            self.assertNotIn("native_cache_file", b)
            self.assertEqual(executor.batches[0]["native_record_ids"], [accepted["id"]])
            cache = output/a["native_directory"]/a["native_cache_file"]
            self.assertEqual(a["native_cache_sha256"], sha(cache))
            with np.load(cache, allow_pickle=False) as archive:
                self.assertEqual(archive["record_ids"].tolist(), [accepted["id"]])
            self.assertEqual(publications, ["controls_pre-64"])


if __name__ == "__main__":
    unittest.main()
