"""CPU mocks for additive evidence inheritance and unbiased native selection."""

import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from sircus.mjlab import grid_diverse_study as d
from sircus.mjlab.grid_study_contract import make_report, validate_report
from sircus.mjlab.learning_contract import digest


ROOT = Path(__file__).resolve().parents[3]


def result(identity, request, *, success=False):
    return dict(
        request_id=request["request_id"],
        bucket_xy_m=request["bucket_xy"],
        method="sweep_planner",
        status_native=(
            "verified_shot" if success else "no_verified_solution_within_budget"
        ),
        success=success,
        qualified=success,
        constraints=request["constraints"],
        configuration=dict(
            board_center=[0.65, -0.2, 0.4],
            release_center=[0.65, -0.2, 0.7375],
            tilt_deg=32.0,
            gap_m=0.3,
            q10=[0.0] * 6,
            q14=[1.0] * 6,
        ),
        trajectory=[dict(time=0.1, ball=[0.65, -0.2, 0.7])],
        motion=dict(passed=True),
        metrics=dict(
            cpu_candidates=64,
            geometry_rejected=32,
            search_cpu_candidates=64,
            search_geometry_rejected=32,
            search_requests=16,
            confirmation_requests=23,
            search_captures=2,
            nominal_attempts=3,
            nominal_captures=3,
            fresh_attempts=20,
            fresh_captures=19,
            invalid=None,
            scalar_score_audit_passed=True,
        ),
        provenance=dict(
            identity, policy_used=False, saved_evidence=False, request_sha256="a" * 64
        ),
        message="Synthetic fixture only",
    )


class DiverseStudyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.parent = self.root / "parent-study"
        self.parent.mkdir()
        self.parent_source = self.root / "parent-source"
        self.parent_source.mkdir()
        self.repository = self.root / "current-source"
        self.repository.mkdir()
        for p in (self.parent_source, self.repository):
            (p / "mock-core.py").write_text("# mock native identity\n")
        self.scene = json.loads((ROOT / "sircus/isaac/scene.json").read_text())
        self.identity = dict(
            contract_fingerprint="b" * 64,
            scene_sha256=digest(self.scene),
            source_sha256={"mock-core.py": d.sha(self.parent_source / "mock-core.py")},
            asset_sha256={"mesh": "c" * 64},
            physics=copy.deepcopy(d.PHYSICS),
        )
        self.ref = dict(
            scene=self.scene,
            contract=dict(fingerprint="b" * 64),
            source_hashes=self.identity["source_sha256"],
            asset_hashes=self.identity["asset_sha256"],
            packages={"mock": "1"},
        )

    def tearDown(self):
        self.temp.cleanup()

    def parent_fixture(self):
        attempts, states, validation = {}, {}, {}
        for i, layout in enumerate(d.grid_layouts()):
            request = dict(
                request_id=layout["id"] + "-parent-original",
                mode="search",
                bucket_xy=layout["bucket_xy_m"],
                constraints=dict(board_position=None, dropper_position=None),
                search_budget=16,
                cpu_candidates=64,
            )
            payload = result(self.identity, request, success=i == 0)
            path = self.parent / "cells" / layout["id"] / "baseline"
            path.mkdir(parents=True)
            d.atomic(path / "request.json", request)
            d.atomic(path / "result.json", payload)
            manifest = {p.name: d.sha(p) for p in path.iterdir()}
            d.atomic(path / "manifest.json", manifest)
            attempts[layout["id"]] = [
                dict(
                    stage="baseline",
                    request=request,
                    result=payload,
                    result_sha256=d.sha(path / "result.json"),
                    evidence_sha256=manifest,
                )
            ]
            states[layout["id"]] = "verified_fresh" if i == 0 else "unresolved"
            if i == 0:
                validation[layout["id"]] = dict(
                    status="completed",
                    configuration_sha256=digest(payload["configuration"]),
                    nominal=dict(captures=3, attempts=3),
                    fresh=dict(captures=29, attempts=30),
                    seed=40,
                )
        ledger = dict(
            attempts=attempts,
            states=states,
            validation=validation,
            started_at="fixture",
        )
        protocol = dict(grid=d.grid_layouts(), fixture=True)
        report = make_report(
            attempts,
            identity=self.identity,
            protocol=protocol,
            status="completed",
            states=states,
            validation_by_cell=validation,
            started_at="fixture",
            updated_at="fixture",
        )
        contract = dict(
            identity=self.identity, protocol=protocol, packages=self.ref["packages"]
        )
        contract["fingerprint"] = digest(contract)
        selected = {
            c["id"]: c["selected"] for c in report["cells"] if c.get("selected")
        }
        for name, payload in (
            ("ledger.json", ledger),
            ("contract.json", contract),
            ("report.json", report),
            ("frozen-selection.json", selected),
        ):
            d.atomic(self.parent / name, payload)
        d.atomic(
            self.parent / "manifest.json",
            {
                str(p.relative_to(self.parent)): d.sha(p)
                for p in self.parent.rglob("*")
                if p.is_file() and p.name != "manifest.json"
            },
        )
        audit = dict(
            status="passed",
            failures=[],
            checks=99,
            contract_fingerprint=contract["fingerprint"],
            study_manifest_sha256=d.sha(self.parent / "manifest.json"),
            report_sha256=d.sha(self.parent / "report.json"),
            frozen_selection_sha256=d.sha(self.parent / "frozen-selection.json"),
            source_sha256=self.identity["source_sha256"],
            asset_sha256=self.identity["asset_sha256"],
            physics=d.PHYSICS,
            packages=self.ref["packages"],
        )
        audit_path = self.root / "audit.json"
        d.atomic(audit_path, audit)
        return audit_path, report

    def test_parent_full_manifest_source_scene_and_passed_audit_are_required(self):
        audit_path, report = self.parent_fixture()
        snapshot = d.load_parent(
            self.parent,
            audit_path,
            self.parent_source,
            self.ref,
            repository=self.repository,
        )
        self.assertEqual(snapshot["report"], report)
        (self.parent_source / "mock-core.py").write_text("changed")
        with self.assertRaisesRegex(ValueError, "parent source"):
            d.load_parent(
                self.parent,
                audit_path,
                self.parent_source,
                self.ref,
                repository=self.repository,
            )

    def test_parent_tamper_missing_files_and_failed_audit_rejected_before_native(self):
        audit_path, _ = self.parent_fixture()
        original = audit_path.read_bytes()
        bad = json.loads(original)
        bad["status"] = "failed"
        d.atomic(audit_path, bad)
        with self.assertRaisesRegex(ValueError, "did not pass"):
            d.load_parent_snapshot(self.parent, audit_path=audit_path)
        audit_path.write_bytes(original)
        path = (
            self.parent
            / "cells"
            / d.grid_layouts()[1]["id"]
            / "baseline"
            / "result.json"
        )
        path.write_text("{}")
        with self.assertRaisesRegex(ValueError, "evidence changed"):
            d.load_parent(
                self.parent,
                audit_path,
                self.parent_source,
                self.ref,
                repository=self.repository,
            )

    def test_compact_parent_snapshot_reader_checks_exact_expected_metadata(self):
        audit_path, _ = self.parent_fixture()
        full = d.load_parent_snapshot(self.parent, audit_path=audit_path)
        copy_root = self.root / "compact"
        copy_root.mkdir()
        for name in d.SNAPSHOT_FILES:
            source = audit_path if name == "audit.json" else self.parent / name
            (copy_root / name).write_bytes(source.read_bytes())
        got = d.load_parent_snapshot(copy_root, full["metadata"])
        self.assertEqual(got["metadata"], full["metadata"])
        bad = copy.deepcopy(full["metadata"])
        bad["report_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "does not match"):
            d.load_parent_snapshot(copy_root, bad)

    def test_archived_wrapper_preserves_original_provenance_and_counts_no_new_work(
        self,
    ):
        audit_path, old_report = self.parent_fixture()
        snapshot = d.load_parent(
            self.parent,
            audit_path,
            self.parent_source,
            self.ref,
            repository=self.repository,
        )
        cell = old_report["cells"][0]
        new_identity = copy.deepcopy(self.identity)
        new_identity["source_sha256"]["new-reader.py"] = "d" * 64
        request = dict(
            request_id="new-wrapper",
            mode="search",
            bucket_xy=cell["bucket_xy_m"],
            constraints=dict(board_position=None, dropper_position=None),
            search_budget=16,
            cpu_candidates=64,
        )
        wrapped, original, files = d.archived_wrapper(
            snapshot, cell, request, new_identity
        )
        self.assertEqual(wrapped["request_id"], "new-wrapper")
        self.assertNotEqual(wrapped["request_id"], original["request"]["request_id"])
        self.assertEqual(wrapped["configuration"], original["result"]["configuration"])
        self.assertEqual(wrapped["trajectory"], original["result"]["trajectory"])
        inherited = wrapped["provenance"]["parent_evidence"]
        self.assertEqual(
            inherited["original_request_id"], original["request"]["request_id"]
        )
        self.assertEqual(
            inherited["original_provenance"], original["result"]["provenance"]
        )
        self.assertEqual(inherited["original_metrics"], original["result"]["metrics"])
        self.assertEqual(inherited["original_result_file"], files["result"])
        self.assertTrue(wrapped["provenance"]["saved_evidence"])
        for name in (
            "cpu_candidates",
            "geometry_rejected",
            "search_cpu_candidates",
            "search_geometry_rejected",
            "search_requests",
            "confirmation_requests",
            "search_captures",
        ):
            self.assertEqual(wrapped["metrics"][name], 0)
        report = make_report(
            {cell["id"]: [dict(stage="baseline", request=request, result=wrapped)]},
            identity=new_identity,
            protocol={},
        )
        self.assertEqual(report["cells"][0]["selected"]["source"], "saved_reference")
        self.assertIsNone(report["cells"][0]["fresh"]["rate"])
        self.assertIsNone(report["cells"][0]["frozen_validation"])
        self.assertEqual(report["cells"][0]["saved_fresh"]["captures"], 19)

    def test_targets_include_parent_frozen_failures_but_preserve_reliable_recipes(self):
        _, report = self.parent_fixture()
        first = report["cells"][0]
        self.assertNotIn(first["id"], d.targeted_cell_ids(report))
        self.assertEqual(len(d.targeted_cell_ids(report)), 34)
        first["frozen_validation"]["fresh"]["captures"] = 26
        self.assertIn(first["id"], d.targeted_cell_ids(report))
        first["frozen_validation"]["fresh"]["captures"] = 30
        first["frozen_validation"]["nominal"]["captures"] = 2
        self.assertIn(first["id"], d.targeted_cell_ids(report))
        first["frozen_validation"] = None
        self.assertIn(first["id"], d.targeted_cell_ids(report))

    def test_prepare_only_archives_all35_without_native_or_relabeling_parent_rates(
        self,
    ):
        audit_path, _ = self.parent_fixture()
        snapshot = d.load_parent(
            self.parent,
            audit_path,
            self.parent_source,
            self.ref,
            repository=self.repository,
        )
        new_identity = copy.deepcopy(self.identity)
        new_identity["source_sha256"]["new-reader.py"] = "d" * 64
        args = SimpleNamespace(
            reference=self.root / "reference",
            assets=self.root / "assets",
            parent_study=self.parent,
            parent_audit=audit_path,
            parent_source=self.parent_source,
            output=self.root / "output",
            gpu_lock=self.root / "lock",
            prepare_only=True,
        )
        with patch.object(d, "load_reference", return_value=self.ref), patch.object(
            d, "load_parent", return_value=snapshot
        ), patch.object(d, "study_identity", return_value=new_identity), patch.object(
            d, "historical_hints", return_value=[]
        ), patch.object(
            d,
            "DiversePlanner",
            side_effect=AssertionError("prepare only must not create native planner"),
        ), patch(
            "builtins.print"
        ):
            d.run(args)
        report = validate_report(d.read(args.output / "report.json"))
        self.assertEqual(len(report["cells"]), 35)
        self.assertEqual(report["summary"]["saved_verified"], 1)
        self.assertEqual(report["summary"]["search_native_requests"], 0)
        self.assertEqual(report["summary"]["frozen_validation_completed"], 0)
        self.assertEqual(d.read(args.output / "ledger.json")["validation"], {})
        self.assertTrue((args.output / "parent-hints.json").exists())
        self.assertTrue((args.output / "parent-row-manifest.json").exists())
        contract = d.read(args.output / "contract.json")
        self.assertEqual(
            contract["parent_hints_sha256"],
            digest(d.read(args.output / "parent-hints.json")),
        )
        self.assertEqual(contract["protocol"]["target_selection"], d.TARGET_SELECTION)
        for cell in report["cells"]:
            stage = args.output / "cells" / cell["id"] / "baseline"
            self.assertTrue((stage / "original-request.json").exists())
            self.assertTrue((stage / "original-result.json").exists())
            self.assertTrue(all(a["saved_evidence"] for a in cell["attempts"]))

    def candidates(self, n=240, broad=100):
        rng = np.random.default_rng(91)
        lo, hi = np.asarray(d.WIDE_BOUNDS["low"]), np.asarray(d.WIDE_BOUNDS["high"])
        values = rng.uniform(lo, hi, (n, 5))
        return [
            dict(
                id=f"mock-{i}",
                values=v.tolist(),
                geometry_eligible=True,
                proposal=dict(
                    method="broad_space_filling" if i >= n - broad else "local_mock"
                ),
            )
            for i, v in enumerate(values)
        ]

    def test_native_selection_reserves_broad_slots_even_after_local_prefix(self):
        rows = self.candidates()
        ids = d.select_diverse(rows, seed=22)
        self.assertEqual(len(ids), 128)
        self.assertEqual(len(set(ids)), 128)
        self.assertEqual(
            sum(rows[i]["proposal"]["method"] in d.BROAD_METHODS for i in ids), 64
        )
        self.assertEqual(ids, d.select_diverse(rows, seed=22))
        self.assertNotEqual(ids, d.select_diverse(rows, seed=23))
        self.assertTrue(any(i >= 140 for i in ids))

    def test_geometry_rejects_never_enter_selection_and_sparse_quota_stays_honest(self):
        rows = self.candidates(n=160, broad=10)
        for row in rows[:20]:
            row["geometry_eligible"] = False
        ids = d.select_diverse(rows, seed=3)
        self.assertEqual(len(ids), 128)
        self.assertEqual(
            sum(rows[i]["proposal"]["method"] in d.BROAD_METHODS for i in ids), 10
        )
        self.assertTrue(all(i >= 20 for i in ids))
        for row in rows:
            row["geometry_eligible"] = False
        self.assertEqual(d.select_diverse(rows), [])
        for bad in [dict(budget=0), dict(broad_quota=129), dict(budget=True)]:
            with self.assertRaises(ValueError):
                d.select_diverse(rows, **bad)

    def test_pool_is_seeded_distinct_bounded_and_covers_energy_and_angle_ranges(self):
        scene = self.scene
        bucket = np.asarray([0.1, -0.65])
        hint = dict(
            values=[0.55, -0.35, 0.5, 58.0, 0.16],
            bucket_xy_m=bucket.tolist(),
            q10=[0.0] * 6,
            q14=[0.0] * 6,
        )
        first, meta = d.diverse_pool(scene, bucket, [hint], seed=3)
        same, same_meta = d.diverse_pool(scene, bucket, [hint], seed=3)
        self.assertEqual(len(first), 1024)
        np.testing.assert_array_equal(first, same)
        self.assertEqual(meta, same_meta)
        self.assertEqual(len({tuple(a) for a in first}), 1024)
        self.assertGreaterEqual(sum(m["method"] in d.BROAD_METHODS for m in meta), 512)
        joint = first[[m["method"] == "joint_gap_tilt_sweep" for m in meta]]
        self.assertAlmostEqual(joint[:, 4].min(), 0.06)
        self.assertAlmostEqual(joint[:, 4].max(), 0.64)
        self.assertAlmostEqual(joint[:, 3].min(), 12.0)
        self.assertAlmostEqual(joint[:, 3].max(), 60.0)
        lo, hi = np.asarray(d.WIDE_BOUNDS["low"]), np.asarray(d.WIDE_BOUNDS["high"])
        self.assertTrue(((first >= lo) & (first <= hi)).all())
        self.assertTrue((np.linalg.norm(first[:, :2] - bucket, axis=1) > 0.15).all())

    def test_all_proposals_are_audited_before_native_and_family_ranges_persist(self):
        planner = object.__new__(d.DiversePlanner)
        planner.directory = self.root / "mock-run"
        planner.directory.mkdir()
        planner.rows = []
        planner.counts = dict(
            cpu_candidates=0,
            geometry_rejected=0,
            search_requests=0,
            confirmation_requests=0,
            search_cpu_candidates=0,
            search_geometry_rejected=0,
            search_captures=0,
        )
        planner.admission = SimpleNamespace(
            check_home=Mock(return_value=dict(passed=True))
        )
        rows = self.candidates(n=256, broad=128)
        values = np.asarray([r["values"] for r in rows])
        meta = [r["proposal"] for r in rows]
        events = []

        def audit(bucket, vals, metadata):
            offset = sum(n for kind, n in events if kind == "audit")
            events.append(("audit", len(vals)))
            return rows[offset : offset + len(vals)]

        planner.audit = Mock(side_effect=audit)
        planner.storage_check = Mock()

        def execute(env, selected, phase):
            self.assertEqual(sum(n for kind, n in events if kind == "audit"), 256)
            events.append(("native", len(selected)))
            planner.rows.extend(
                dict(
                    row,
                    phase=phase,
                    native_release=True,
                    route_success=False,
                    invalid=None,
                )
                for row in selected
            )

        planner.execute = Mock(side_effect=execute)
        planner.outcome = Mock(return_value={"fixture": True})
        request = dict(bucket_xy=[0.1, -0.65], search_budget=128)
        got = planner.search(None, request, values, meta, seed=2)
        self.assertEqual(got, {"fixture": True})
        self.assertEqual(events[-1], ("native", 128))
        self.assertEqual(planner.counts["search_cpu_candidates"], 256)
        selected = json.loads((planner.directory / "diversity.json").read_text())
        self.assertTrue(selected["all_proposals_audited"])
        self.assertTrue(selected["quota_satisfied"])
        self.assertEqual(selected["selected_broad"], 64)
        self.assertEqual(selected["checked"]["count"], 256)
        self.assertEqual(selected["selected"]["count"], 128)

    def test_blocked_home_does_not_launch_native_or_claim_search_coverage(self):
        planner = object.__new__(d.DiversePlanner)
        planner.directory = self.root / "blocked"
        planner.directory.mkdir()
        planner.admission = SimpleNamespace(
            check_home=Mock(return_value=dict(passed=False))
        )
        planner.audit = Mock(side_effect=AssertionError("no audit"))
        planner.execute = Mock(side_effect=AssertionError("no native"))
        planner.outcome = Mock(return_value={"status_native": "common_home_unresolved"})
        got = planner.search(
            None,
            dict(bucket_xy=[0.1, -0.65], search_budget=128),
            np.zeros((1024, 5)),
            [{}] * 1024,
        )
        self.assertEqual(got["status_native"], "common_home_unresolved")
        planner.audit.assert_not_called()
        planner.execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
