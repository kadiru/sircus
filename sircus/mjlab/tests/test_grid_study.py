"""CPU-only study orchestration checks; no mesh, GPU or robot is created."""

import contextlib
import copy
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from sircus.mjlab import grid_study as grid
from sircus.mjlab import grid_study_contract
from sircus.mjlab.expanded_actions import WIDE_BOUNDS
from sircus.mjlab.learning_contract import digest


ROOT = Path(__file__).resolve().parents[3]


def shot(**extra):
    row = dict(
        id="selected-shot",
        bucket_xy_m=[0.3, -0.5],
        values=[0.67, -0.1, 0.32, 28.0, 0.3],
        q10=[0.0] * 6,
        q14=[1.0] * 6,
        geometry_eligible=True,
        geometry_reason="endpoint_admitted",
        native_release=True,
        route_success=True,
        invalid=None,
        scalar_score_audit_passed=True,
        board_center_m=[0.67, -0.1, 0.32],
        board_normal=[0.0, -0.5, 0.866],
        board_quaternion_wxyz=[1.0, 0.0, 0.0, 0.0],
        release_ball_center_m=[0.67, -0.1, 0.6575],
        link_positions10=[[0.0, 0.0, 0.0]] * 6,
        link_positions14=[[1.33731, 0.0, 0.0]] * 6,
        route_quality=2,
        max_retention_s=1.0,
        board_contact_episodes=1,
    )
    row.update(extra)
    return row


class GridStudyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.scene = json.loads((ROOT / "sircus/isaac/scene.json").read_text())

    def tearDown(self):
        self.temp.cleanup()

    def planner(self):
        # Bypass NativePlanner.__init__: these tests never load native assets.
        planner = object.__new__(grid.StudyPlanner)
        planner.ref = dict(
            scene=self.scene,
            contract=dict(fingerprint="a" * 64),
            source_hashes={"mock": "b" * 64},
            asset_hashes={"mock": "c" * 64},
        )
        planner.start_stage(self.directory, "cell-0", "mock")
        planner.admission = SimpleNamespace(
            admit=Mock(return_value=dict(passed=True)),
            connect_variation=Mock(return_value=dict(passed=True)),
        )
        return planner

    def test_expanded_pool_interleaves_sources_and_parameter_axes_without_mutating_hints(
        self,
    ):
        bucket = [0.3, -0.5]
        hints = [
            dict(
                source_id=name,
                bucket_xy_m=bucket,
                values=values,
                q10=[0.0] * 6,
                q14=[0.0] * 6,
                source_record_sha256=name * 64,
            )
            for name, values in [
                ("a", [0.67, -0.1, 0.32, 28.0, 0.3]),
                ("b", [0.67, -0.1, 0.32, 28.0, 0.4]),
            ]
        ]
        original = copy.deepcopy(hints)
        broad = np.asarray([[0.8, -0.1, 0.35, 30.0, 0.3]])
        with patch.object(
            grid, "proposals", return_value=(broad, [dict(method="broad_mock")])
        ):
            values, metadata = grid.candidate_pool(
                self.scene, bucket, hints, limit=200, seed=12, expanded=True
            )
        self.assertEqual(hints, original)
        self.assertEqual(
            [m["method"] for m in metadata[:2]], ["exact_historical_geometry"] * 2
        )
        self.assertEqual(
            [m["source_hint"]["source_id"] for m in metadata[2:6]], ["a", "b", "a", "b"]
        )
        self.assertEqual(
            [m["delta"] for m in metadata[2:6]], [-0.001, -0.001, 0.001, 0.001]
        )
        axes = {
            m.get("axis") for m in metadata if m["method"] == "local_interleaved_sweep"
        }
        self.assertEqual(axes, {"4", "2", "3", "radius", "bearing"})
        self.assertTrue(any(m["method"] == "broad_mock" for m in metadata))
        self.assertTrue(
            all(
                m["scope"] == "Geometry proposal only; no inherited flight label"
                for m in metadata
            )
        )
        self.assertEqual(len(values), len(metadata))

    def test_expanded_pool_excludes_previous_values_rejects_bad_bounds_and_is_seeded(
        self,
    ):
        bucket = [0.3, -0.5]
        anchor = [0.67, -0.1, 0.32, 28.0, 0.3]
        hints = [dict(bucket_xy_m=bucket, values=anchor, q10=[0.0] * 6, q14=[0.0] * 6)]
        low, high = np.asarray(WIDE_BOUNDS["low"]), np.asarray(WIDE_BOUNDS["high"])
        broad = np.asarray(
            [anchor, anchor, high + 0.1, [np.nan] * 5, [0.31, -0.49, 0.3, 30.0, 0.3]]
        )
        args = dict(
            scene=self.scene,
            bucket=bucket,
            hints=hints,
            limit=160,
            expanded=True,
            exclude=[anchor],
        )
        with patch.object(
            grid, "proposals", return_value=(broad, [dict(method="mock")] * len(broad))
        ):
            first, meta = grid.candidate_pool(**args, seed=7)
            same, same_meta = grid.candidate_pool(**args, seed=7)
            other, _ = grid.candidate_pool(**args, seed=8)
        np.testing.assert_array_equal(first, same)
        self.assertEqual(meta, same_meta)
        self.assertFalse(np.array_equal(first, other))
        self.assertTrue(np.isfinite(first).all())
        self.assertTrue(((first >= low) & (first <= high)).all())
        self.assertTrue((np.linalg.norm(first[:, :2] - bucket, axis=1) > 0.15).all())
        self.assertFalse(any(np.array_equal(value, anchor) for value in first))
        self.assertEqual(len(first), len({tuple(value) for value in first}))
        self.assertLessEqual(len(first), 160)

    def test_qualification_nominal_failure_is_recorded_without_fresh_retry(self):
        planner, source = self.planner(), shot()
        nominal = [shot(id=f"nominal-{i}", route_success=i != 2) for i in range(3)]
        planner.execute = Mock(return_value=nominal)
        planner.audit = Mock(
            side_effect=AssertionError("Fresh audit must not follow nominal failure")
        )
        record, measured, fresh = planner.confirm(None, source, seed=1)
        self.assertFalse(record["qualified"])
        self.assertEqual(record["nominal_counts"], dict(captures=2, attempts=3))
        self.assertEqual(measured, nominal)
        self.assertEqual(fresh, [])
        self.assertEqual(planner.execute.call_count, 1)
        saved = json.loads((self.directory / "qualification-000.json").read_text())
        self.assertEqual(saved["nominal"], [r["id"] for r in nominal])
        self.assertEqual(saved["nominal_counts"], dict(captures=2, attempts=3))

    def test_fresh_geometry_and_path_rejects_remain_in_full_twenty_denominator_and_proofs(
        self,
    ):
        planner, source = self.planner(), shot()
        fresh_inputs = [
            shot(
                id=f"input-{i}",
                geometry_eligible=i != 0,
                geometry_reason="geometry_reject" if i == 0 else "endpoint_admitted",
            )
            for i in range(20)
        ]
        planner.audit = Mock(return_value=fresh_inputs)
        planner.admission.connect_variation = Mock(
            side_effect=lambda _source, row: dict(
                passed=not row["id"].endswith("fresh-5"),
                reason="local_path_reject" if row["id"].endswith("fresh-5") else None,
                sampled_clearance_mm=5,
            )
        )

        def execute(_env, candidates, phase):
            return [
                dict(
                    row,
                    phase=phase,
                    native_release=row["geometry_eligible"],
                    route_success=row["geometry_eligible"],
                    scalar_score_audit_passed=row["geometry_eligible"],
                    invalid=(
                        None if row["geometry_eligible"] else "geometry_or_path_reject"
                    ),
                )
                for row in candidates
            ]

        planner.execute = Mock(side_effect=execute)
        record, nominal, fresh = planner.confirm(None, source, seed=9)
        self.assertTrue(record["qualified"])
        self.assertEqual(record["nominal_counts"], dict(captures=3, attempts=3))
        self.assertEqual(record["fresh_counts"], dict(captures=18, attempts=20))
        self.assertEqual(len(fresh), 20)
        self.assertEqual(len(record["variations"]), 20)
        self.assertEqual(sum(not row["admitted"] for row in record["variations"]), 2)
        self.assertEqual(
            record["variations"][0]["local_path"],
            dict(passed=False, reason="geometry_reject"),
        )
        self.assertEqual(
            record["variations"][5]["local_path"]["reason"], "local_path_reject"
        )
        self.assertTrue(
            all(
                set(("id", "values", "q10", "q14", "local_path", "admitted")).issubset(
                    row
                )
                for row in record["variations"]
            )
        )
        saved = json.loads((self.directory / "qualification-000.json").read_text())
        self.assertEqual(saved["fresh_counts"], dict(captures=18, attempts=20))
        self.assertEqual(saved["fresh"], [r["id"] for r in fresh])

    def test_frozen_thirty_repeats_continue_after_nominal_failure_without_fallback(
        self,
    ):
        planner, source = self.planner(), shot()
        planner.audit = Mock(return_value=[shot(id=f"input-{i}") for i in range(30)])

        def execute(_env, candidates, phase):
            return [
                dict(
                    row, phase=phase, route_success=not (phase == "nominal" and i == 1)
                )
                for i, row in enumerate(candidates)
            ]

        planner.execute = Mock(side_effect=execute)
        record, nominal, fresh = planner.confirm(
            None, source, seed=44, fresh_n=30, prefix="frozen"
        )
        self.assertEqual(record["nominal_counts"], dict(captures=2, attempts=3))
        self.assertEqual(record["fresh_counts"], dict(captures=30, attempts=30))
        self.assertEqual(len(record["variations"]), 30)
        self.assertEqual(record["purpose"], "frozen")
        self.assertEqual(planner.execute.call_count, 2)
        self.assertEqual(planner.admission.admit.call_count, 1)

    def test_outcome_counts_belong_only_to_matching_selected_cohort(self):
        planner, source = self.planner(), shot()
        planner.counts.update(confirmation_requests=77, search_requests=16)
        cohort = dict(
            source_id=source["id"],
            source_values=source["values"],
            source_q10=source["q10"],
            source_q14=source["q14"],
            motion=dict(passed=True),
            nominal=["n0", "n1", "n2"],
            fresh=[f"f{i}" for i in range(20)],
            nominal_counts=dict(captures=3, attempts=3),
            fresh_counts=dict(captures=18, attempts=20),
            quality=dict(clean=8, assisted=10),
            qualified=True,
        )
        request = dict(
            request_id="selected-request",
            bucket_xy=source["bucket_xy_m"],
            constraints=dict(board_position=None, dropper_position=None),
        )
        result = planner.outcome(request, source, cohort)
        self.assertTrue(result["success"])
        self.assertEqual(result["metrics"]["confirmation_requests"], 77)
        self.assertEqual(result["metrics"]["fresh_captures"], 18)
        self.assertEqual(result["metrics"]["fresh_attempts"], 20)
        self.assertEqual(result["metrics"]["nominal_captures"], 3)
        self.assertEqual(
            result["metrics"]["confirmation_configuration_sha256"],
            digest(result["configuration"]),
        )
        for key in ("values", "q10", "q14"):
            changed = copy.deepcopy(source)
            changed[key][0] += 0.01
            with self.assertRaisesRegex(
                ValueError, "[Cc]onfirmation|cohort|[Ss]elected"
            ):
                planner.outcome(request, changed, cohort)

    def test_run_freezes_selected_recipe_before_validation_and_never_reselects_on_failure(
        self,
    ):
        layout = dict(id="dev-grid-00-00", bucket_xy_m=[0.3, -0.5])
        config = dict(
            board_center=[0.67, -0.1, 0.32],
            release_center=[0.67, -0.1, 0.6575],
            tilt_deg=28.0,
            gap_m=0.3,
            q10=[0.0] * 6,
            q14=[1.0] * 6,
        )
        saved = dict(
            request_id=layout["id"] + "-saved",
            success=True,
            configuration=config,
            metrics={},
            provenance=dict(saved_evidence=True),
        )
        identity = dict(source_sha256={"mock": "a" * 64})
        output = self.directory / "run"
        calls = []

        class Planner:
            def __init__(self, *_args):
                pass

            def start_stage(self, directory, *_args):
                self.directory = directory
                directory.mkdir(parents=True, exist_ok=True)

            def check_options(self, _env):
                pass

            def confirm(self, _env, source, **kwargs):
                self.assert_frozen = json.loads(
                    (output / "frozen-selection.json").read_text()
                )
                calls.append(
                    dict(
                        source=copy.deepcopy(source),
                        kwargs=kwargs,
                        frozen=self.assert_frozen,
                    )
                )
                return (
                    dict(
                        nominal_counts=dict(captures=1, attempts=3),
                        fresh_counts=dict(captures=2, attempts=30),
                        quality=dict(clean=1, assisted=1, cohort="fresh"),
                    ),
                    [],
                    [],
                )

        def report(attempts, *, status, validation_by_cell, **_kwargs):
            selected = dict(
                configuration_sha256=digest(config), request_id=saved["request_id"]
            )
            return dict(
                status=status,
                summary={},
                updated_at="mock",
                cells=[dict(id=layout["id"], selected=selected)] if attempts else [],
                validation=copy.deepcopy(validation_by_cell),
            )

        @contextlib.contextmanager
        def slot(*_args, **_kwargs):
            yield None

        args = SimpleNamespace(
            reference=self.directory,
            assets=self.directory,
            gpu_lock=self.directory / "lock",
            output=output,
            prepare_only=False,
        )
        with (
            patch.object(
                grid,
                "load_reference",
                return_value=dict(source_hashes={}, packages={}, scene={}, public={}),
            ),
            patch.object(grid, "study_identity", return_value=identity),
            patch.object(grid, "historical_hints", return_value=[]),
            patch.object(grid, "grid_layouts", return_value=[layout]),
            patch.object(grid, "PROTOCOL", dict(stages=[])),
            patch.object(
                grid,
                "exact_source",
                return_value=dict(source=dict(values=[0.67, -0.1, 0.32, 28.0, 0.3])),
            ),
            patch.object(grid, "cached_result", return_value=copy.deepcopy(saved)),
            patch.object(grid, "StudyPlanner", Planner),
            patch.object(grid_study_contract, "make_report", side_effect=report),
            patch.object(grid_study_contract, "validate_report"),
            patch("sircus.mjlab.gpu_slot.gpu_slot", slot),
            patch.dict(
                "sys.modules",
                {
                    "sircus.mjlab.contact_env": SimpleNamespace(
                        ContactEnvironment=lambda *_args, **_kwargs: object()
                    )
                },
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            grid.run(args)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["kwargs"]["fresh_n"], 30)
        self.assertEqual(calls[0]["kwargs"]["prefix"], "frozen")
        self.assertEqual(
            calls[0]["frozen"][layout["id"]]["configuration_sha256"], digest(config)
        )
        self.assertEqual(calls[0]["source"]["q10"], config["q10"])
        ledger = json.loads((output / "ledger.json").read_text())
        self.assertEqual(len(ledger["attempts"][layout["id"]]), 1)
        self.assertEqual(
            ledger["validation"][layout["id"]]["fresh"], dict(captures=2, attempts=30)
        )
        self.assertEqual(
            json.loads((output / "report.json").read_text())["status"], "completed"
        )


if __name__ == "__main__":
    unittest.main()
