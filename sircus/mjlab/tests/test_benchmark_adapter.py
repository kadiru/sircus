"""Shared sweep/RL endpoint/path/native gates using CPU-only fakes."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from sircus.mjlab.benchmark_adapter import BenchmarkAdapter, RESIDUAL_HALF_RANGE
from sircus.mjlab.corrected_contract import PHYSICS, sha
from sircus.mjlab.learning_contract import digest


OPTIONS = dict(
    timestep=1 / 15360,
    solver=2,
    cone=0,
    integrator=3,
    iterations=100,
    tolerance=1e-8,
    gravity=[0.0, 0.0, -9.81],
)
SCORE = dict(
    route_success=True,
    route_quality=2,
    complete_success=True,
    board_bounce=True,
    whole_ball_entry=True,
    external_contact_steps=0,
    board_contact_episodes=1,
    physics_steps=3,
)


def hint(**extra):
    value = dict(
        values=[0.60, -0.1, 0.30, 30.0, 0.30],
        bucket_xy_m=[0.6, -0.4],
        q10=[0.1] * 6,
        q14=[0.2] * 6,
        source_id="teacher-a",
        source_record_sha256="a" * 64,
        route_success=True,
    )
    value.update(extra)
    return value


def request(name="shot-0", **extra):
    source = extra.pop("source_hint", hint())
    value = dict(
        id=name,
        bucket_xy_m=[0.62, -0.42],
        values=list(source["values"]),
        source_hint=source,
    )
    value.update(extra)
    return value


class Geometry:
    def __init__(self):
        self.calls = []

    def audit_values(self, bucket, values, **kwargs):
        self.calls.append((bucket.copy(), values.copy(), copy.deepcopy(kwargs)))
        lower, upper = np.asarray(kwargs["bounds"]["low"]), np.asarray(
            kwargs["bounds"]["high"]
        )
        valid = ((values >= lower) & (values <= upper)).all(2)
        # One deliberate endpoint rejection inside otherwise valid bounds.
        valid &= values[:, :, 3] < 35.0
        reasons = np.where(valid, 0, 7)
        q = np.concatenate([values, np.zeros((*values.shape[:2], 1))], axis=2)
        return valid, dict(
            q10=q, q14=q + 0.1, reasons=reasons, summary=dict(requests=len(bucket))
        )


class Admission:
    def __init__(self):
        self.home = dict(sha256="h" * 64)
        self.home_calls, self.anchor_calls, self.local_calls = [], [], []
        self.anchor_pass = True

    def check_home(self, bucket):
        self.home_calls.append(copy.deepcopy(bucket))
        return dict(passed=bucket[0] >= 0.2, reason="unproven_clearance")

    def admit(self, row):
        self.anchor_calls.append(copy.deepcopy(row))
        return dict(
            passed=self.anchor_pass,
            request_sha256=digest(row),
            scope="mock_common_home",
        )

    def connect_variation(self, anchor, row):
        self.local_calls.append((copy.deepcopy(anchor), copy.deepcopy(row)))
        return dict(
            passed=row["values"][4] <= 0.35,
            reason="local_sampled_path_rejected" if row["values"][4] > 0.35 else None,
            request_sha256=digest(row),
            nominal_request_sha256=digest(anchor),
        )


class Environment:
    device = "cpu"
    substeps = PHYSICS["substeps"]
    board_mode = PHYSICS["board_mode"]
    gravity_mode = PHYSICS["gravity_mode"]

    def __init__(self):
        self.sim = SimpleNamespace(
            mj_model=SimpleNamespace(opt=SimpleNamespace(**copy.deepcopy(OPTIONS)))
        )
        self.calls = []

    def run(self, bucket, values, *_args, **_kwargs):
        self.calls.append((bucket.copy(), values.copy()))
        n = len(bucket)
        self.last_arrays = dict(
            position_m=np.zeros((3, n, 3)),
            velocity_m_s=np.zeros((3, n, 3)),
            forces_N=np.zeros((3, n, 6)),
        )
        rows = [
            dict(
                **SCORE,
                bucket_xy_m=xy.tolist(),
                native_release=True,
                invalid=None,
                board_normal=[0.0, -0.5, 0.866],
                trajectory_m=[[0.0, 0.0, 0.0]] * 3,
            )
            for xy in bucket
        ]
        return rows, {}


class Scalar:
    def __init__(self, *_args):
        self.steps = []

    def step(self, *args):
        self.steps.append(args)

    def result(self):
        return dict(SCORE)


def bank_rows(env, bucket, values, _q10, _q14, eligible, *, record):
    from sircus.mjlab.bank_study import unresolved_row

    rows = [unresolved_row(xy) for xy in bucket]
    selected = np.flatnonzero(eligible)
    if len(selected):
        measured, _ = env.run(bucket[selected], values[selected], record=record)
        for i, row in zip(selected, measured):
            rows[i] = row
    return rows


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.geometry, self.admission, self.env = Geometry(), Admission(), Environment()
        self.factory = Mock(return_value=self.env)
        self.adapter = BenchmarkAdapter(
            dict(
                scene={},
                contract=dict(common_home={}),
                public=dict(effective_native_options=OPTIONS),
            ),
            self.root / "assets",
            self.root / "evidence",
            geometry=self.geometry,
            admission=self.admission,
            environment_factory=self.factory,
        )
        self.native = patch(
            "sircus.mjlab.bank_study.run_bank_rows", side_effect=bank_rows
        )
        self.scalar = patch("sircus.mjlab.contract.ScoreStream", Scalar)
        self.native.start()
        self.scalar.start()
        self.addCleanup(self.native.stop)
        self.addCleanup(self.scalar.stop)

    def tearDown(self):
        self.temp.cleanup()

    def test_original_teacher_bucket_is_retained_and_query_heading_resolved_before_local_action(
        self,
    ):
        first = request(values=[0.61, -0.1, 0.31, 31.0, 0.31])
        original = copy.deepcopy(first)
        rows, timing = self.adapter.evaluate([first], "train")
        self.assertEqual(first, original)
        self.assertEqual(len(self.geometry.calls), 2)
        np.testing.assert_array_equal(
            self.geometry.calls[0][2]["known_bucket_xy"], [[0.6, -0.4]]
        )
        np.testing.assert_array_equal(
            self.geometry.calls[1][2]["known_bucket_xy"], [[0.62, -0.42]]
        )
        resolved = self.admission.anchor_calls[0]
        self.assertEqual(resolved["bucket_xy_m"], first["bucket_xy_m"])
        self.assertEqual(self.admission.local_calls[0][0]["q10"], resolved["q10"])
        self.assertEqual(rows[0]["values"], first["values"])
        self.assertTrue(rows[0]["motion_admitted"])
        self.assertEqual(timing["requested"], timing["native_releases"])

    def test_every_requested_reject_retained_without_native_fallback(self):
        inputs = [
            request("ok"),
            request("endpoint", values=[0.6, -0.1, 0.3, 36.0, 0.3]),
            request("path", values=[0.6, -0.1, 0.3, 30.0, 0.4]),
            request("home", bucket_xy_m=[0.1, -0.42]),
        ]
        rows, timing = self.adapter.evaluate(inputs, "evaluation")
        self.assertEqual([r["id"] for r in rows], [r["id"] for r in inputs])
        self.assertEqual(timing["requested"], 4)
        self.assertEqual(timing["native_releases"], 1)
        self.assertEqual(
            [bool(r["native_release"]) for r in rows], [True, False, False, False]
        )
        self.assertEqual(rows[2]["invalid"], "local_sampled_path_rejected")
        self.assertEqual(rows[3]["invalid"], "anchor_common_home_clearance_unresolved")
        self.assertIsNone(rows[3]["endpoint_eligible"])
        self.assertIsNone(rows[3]["geometry_row"])
        self.assertEqual(rows[2]["geometry_row"], 2)
        self.assertFalse(rows[3]["motion_evaluated"])
        self.assertEqual(len(self.env.calls), 1)
        self.assertEqual(len(self.env.calls[0][0]), 1)

    def test_anchor_path_failure_never_allocates_gpu_environment(self):
        self.admission.anchor_pass = False
        rows, timing = self.adapter.evaluate([request()], "train")
        self.assertFalse(rows[0]["native_release"])
        self.assertEqual(rows[0]["invalid"], "anchor_common_home_path_unresolved")
        self.factory.assert_not_called()
        self.assertEqual(timing["gpu_s"], 0.0)
        self.assertEqual(len(self.geometry.calls), 1)

    def test_exact_path_caches_are_per_adapter_and_do_not_credit_extra_trials(self):
        self.adapter.evaluate([request("a")], "train")
        rows, _ = self.adapter.evaluate([request("b")], "train")
        self.assertEqual(len(self.admission.anchor_calls), 1)
        self.assertEqual(len(self.admission.local_calls), 1)
        self.assertEqual(self.adapter.totals["requested"], 2)
        self.assertEqual(self.adapter.totals["native_releases"], 2)
        self.assertEqual(self.adapter.totals["anchor_cache_hits"], 1)
        self.assertEqual(self.adapter.totals["local_path_cache_hits"], 1)
        self.assertEqual(len(self.env.calls), 2)
        self.assertTrue(rows[0]["native_release"])
        fresh = BenchmarkAdapter(
            self.adapter.ref,
            self.root / "assets",
            self.root / "other",
            geometry=Geometry(),
            admission=Admission(),
            environment_factory=Mock(return_value=Environment()),
        )
        fresh.evaluate([request("c")], "train")
        self.assertEqual(fresh.totals["anchors_prepared"], 1)
        self.assertEqual(fresh.totals["anchor_cache_hits"], 0)

    def test_original_outcome_labels_are_not_continuation_inputs(self):
        self.adapter.evaluate([request()], "train")
        preparation = json.loads(
            (self.adapter.directory / "preparations/00000.json").read_text()
        )
        self.assertNotIn("route_success", preparation["requests"][0]["source_hint"])
        self.assertEqual(
            preparation["requests"][0]["source_hint"]["source_id"], "teacher-a"
        )

    def test_native_arrays_and_gate_hashes_keep_exact_requested_ids(self):
        rows, _ = self.adapter.evaluate([request("a"), request("b")], "train")
        self.assertTrue(all(r["scalar_score_audit_passed"] for r in rows))
        self.assertTrue(all("trajectory_m" not in r for r in rows))
        with np.load(self.adapter.directory / "native-0000.npz") as arrays:
            self.assertEqual(arrays["record_ids"].tolist(), ["a", "b"])
            self.assertEqual(
                set(arrays.files),
                {"record_ids", "position_m", "velocity_m_s", "forces_N"},
            )
        for i, row in enumerate(rows):
            self.assertEqual(row["native_cache_row"], i)
            self.assertEqual(
                sha(self.adapter.directory / row["preparation_file"]),
                row["preparation_sha256"],
            )
            p = row["proposal"]
            self.assertEqual(
                sha(self.adapter.directory / p["anchor_file"]), p["anchor_sha256"]
            )
            self.assertEqual(
                sha(self.adapter.directory / p["local_path_file"]),
                p["local_path_sha256"],
            )

    def test_native_protocol_is_rechecked_and_never_silently_changed(self):
        self.adapter.evaluate([request("a")], "train")
        self.env.substeps = 8
        with self.assertRaisesRegex(ValueError, "contact protocol"):
            self.adapter.evaluate([request("b")], "train")
        self.assertEqual(len(self.env.calls), 1)

    def test_bad_native_options_fail_before_rollout(self):
        self.env.sim.mj_model.opt.iterations = 10
        with self.assertRaisesRegex(ValueError, "options changed"):
            self.adapter.evaluate([request()], "train")
        self.assertEqual(self.env.calls, [])

    def test_unique_ids_and_nonfinite_values_fail_closed_without_clipping(self):
        self.adapter.evaluate([request("a")], "train")
        with self.assertRaisesRegex(ValueError, "unique"):
            self.adapter.evaluate([request("a")], "train")
        with self.assertRaisesRegex(ValueError, "Finite values"):
            self.adapter.evaluate(
                [
                    request(
                        "b",
                        values=[float("nan")] * 5,
                    )
                ],
                "train",
            )
        self.assertEqual(len(self.env.calls), 1)

    def test_global_action_rejection_keeps_requested_value_without_projection(self):
        source = hint(values=[1.03, -0.1, 0.3, 30.0, 0.3])
        r = request(source_hint=source, values=[1.05, -0.1, 0.3, 30.0, 0.3])
        rows, _ = self.adapter.evaluate([r], "train")
        self.assertFalse(rows[0]["native_release"])
        self.assertEqual(rows[0]["values"][0], 1.05)
        self.factory.assert_not_called()

    def test_float32_requested_values_are_preserved_without_projection(self):
        values = (
            np.asarray(hint()["values"], dtype=np.float32)
            + RESIDUAL_HALF_RANGE.astype(np.float32)
        ).tolist()
        rows, _ = self.adapter.evaluate([request(values=values)], "train")
        self.assertEqual(rows[0]["values"], values)
        self.assertFalse(rows[0]["native_release"])

    def test_realized_perturbation_can_leave_nominal_family_but_keeps_all_physical_gates(
        self,
    ):
        command = [0.64, -0.1, 0.30, 30.0, 0.30]
        realized = [0.641, -0.1, 0.30, 30.0, 0.30]
        item = request(
            values=realized,
            command_values=command,
            normalized_residual=[1.0, 0.0, 0.0, 0.0, 0.0],
            perturbation_mixed_units=[0.001, 0.0, 0.0, 0.0, 0.0],
        )
        rows, timing = self.adapter.evaluate([item], "validation")
        self.assertEqual(rows[0]["values"], realized)
        self.assertTrue(rows[0]["native_release"])
        self.assertEqual(timing["requested"], 1)
        self.assertEqual(len(self.admission.local_calls), 1)
        saved = json.loads(
            (self.adapter.directory / "preparations/00000.json").read_text()
        )
        self.assertEqual(saved["requests"][0]["command_values"], command)
        self.assertEqual(
            saved["requests"][0]["perturbation_mixed_units"],
            item["perturbation_mixed_units"],
        )

    def test_storage_limit_preserves_existing_evidence_and_starts_no_native_work(self):
        self.adapter.min_free_bytes = 10**30
        marker = self.adapter.directory / "keep.txt"
        marker.write_text("preserve")
        with self.assertRaisesRegex(RuntimeError, "storage"):
            self.adapter.evaluate([request()], "train")
        self.assertEqual(marker.read_text(), "preserve")
        self.factory.assert_not_called()
        self.assertEqual(self.geometry.calls, [])

    def test_empty_batch_never_creates_model_or_trial(self):
        rows, timing = self.adapter.evaluate([], "train")
        self.assertEqual(rows, [])
        self.assertEqual(timing["requested"], 0)
        self.factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
