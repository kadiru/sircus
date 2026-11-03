"""Read-only auditor checks with synthetic evidence; no GPU or native meshes."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from sircus.mjlab import audit_grid_study as audit
from sircus.mjlab.contract import FORCE_NAMES, ScoreStream
from sircus.mjlab.corrected_contract import NOISE, sha
from sircus.mjlab.learning_contract import digest


ROOT = Path(__file__).resolve().parents[3]
BUCKET = [0.3, -0.5]
HOME = "a" * 64
VALUES = [0.67, -0.1, 0.32, 28.0, 0.3]


def motion(source):
    request = dict(bucket_xy_m=BUCKET, **source)
    return dict(
        passed=True,
        common_home_sha256=HOME,
        request_sha256=digest(request),
        report=dict(
            request=request,
            endpoint=dict(reason=0, strict_geometry=dict(passed=True)),
            home_geometry=dict(passed=True),
            minimum_clearance_lower_bound_m=0.006,
            exact_selected_joint_max_error_rad=0.0,
            returned_home_joint_max_error_rad=0.0,
        ),
    )


def fixture(*, purpose="qualification", rejected=(), nominal_failure=False):
    n = 30 if purpose == "frozen" else 20
    source = dict(values=VALUES, q10=[0.0] * 6, q14=[1.0] * 6)
    proof = motion(source)
    record = dict(
        source_id="source",
        source_values=source["values"],
        source_q10=source["q10"],
        source_q14=source["q14"],
        purpose=purpose,
        seed=456,
        perturbation_std=NOISE.tolist(),
        motion=proof,
        nominal=[],
        fresh=[],
        variations=[],
        qualified=False,
    )
    rows, geometry = {}, []
    for index in range(3):
        name = f"source-{purpose}-nominal-{index}"
        row = dict(
            source,
            id=name,
            phase="nominal",
            bucket_xy_m=BUCKET,
            native_release=True,
            route_success=not (nominal_failure and index == 2),
            invalid=None,
            scalar_score_audit_passed=True,
            route_quality=2,
        )
        rows[name] = row
        record["nominal"].append(name)
    record["nominal_counts"] = audit.cohort_counts(list(rows.values()))
    if nominal_failure and purpose == "qualification":
        return record, rows, geometry
    jitter = np.asarray(VALUES) + np.random.default_rng(456).normal(0, NOISE, (n, 5))
    for index, value in enumerate(jitter):
        name = f"source-{purpose}-fresh-{index}"
        variation = dict(id=name, values=value.tolist(), q10=[0.0] * 6, q14=[1.0] * 6)
        admitted = index not in rejected
        request = dict(
            bucket_xy_m=BUCKET, **{k: variation[k] for k in ("values", "q10", "q14")}
        )
        local = dict(
            passed=admitted,
            request=request,
            request_sha256=digest(request),
            nominal_request_sha256=proof["request_sha256"],
            common_home_sha256=HOME,
            endpoint=dict(reason=0, strict_geometry=dict(passed=True)),
            local_edges=[dict(passed=True)] * 4,
            minimum_clearance_lower_bound_m=0.006,
            returned_nominal_joint_max_error_rad=0.0,
        )
        variation.update(admitted=admitted, local_path=local)
        geometry.append(copy.deepcopy(variation))
        rows[name] = dict(
            variation,
            phase="fresh",
            bucket_xy_m=BUCKET,
            native_release=admitted,
            route_success=admitted,
            invalid=None if admitted else "path_rejected",
            scalar_score_audit_passed=admitted,
            route_quality=2 if admitted else 0,
        )
        record["fresh"].append(name)
        record["variations"].append(variation)
    record["fresh_counts"] = audit.cohort_counts(
        [rows[name] for name in record["fresh"]]
    )
    record["quality"] = dict(clean=n - len(rejected), assisted=0, cohort="fresh")
    record["qualified"] = bool(
        not nominal_failure and n == 20 and n - len(rejected) >= 18
    )
    return record, rows, geometry


class GridAuditTests(unittest.TestCase):
    def test_manifest_covers_all_files_and_detects_mutation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "row.json").write_text('{"ok":true}')
            (root / "manifest.json").write_text(
                json.dumps({"row.json": sha(root / "row.json")})
            )
            self.assertEqual(
                set(audit.check_manifest(root, audit.Checks())), {"row.json"}
            )
            (root / "row.json").write_text('{"ok":false}')
            with self.assertRaisesRegex(audit.EvidenceError, "hash changed"):
                audit.check_manifest(root, audit.Checks())

    def test_manifest_rejects_unlisted_files_and_path_escape(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "row").write_text("one")
            (root / "hidden").write_text("two")
            (root / "manifest.json").write_text(json.dumps({"row": sha(root / "row")}))
            with self.assertRaisesRegex(audit.EvidenceError, "every evidence file"):
                audit.check_manifest(root, audit.Checks())
            with self.assertRaisesRegex(audit.EvidenceError, "escapes"):
                audit.contained(root, "../outside")

    def test_twenty_qualification_keeps_rejects_in_denominator(self):
        record, rows, geometry = fixture(rejected=(2, 5))
        result = audit.check_confirmation(
            record, rows, geometry, BUCKET, HOME, audit.Checks(), "mock"
        )
        self.assertEqual(
            audit.cohort_counts(result["fresh"]), dict(captures=18, attempts=20)
        )
        self.assertTrue(record["qualified"])

    def test_frozen_thirty_is_not_reclassified_as_qualification(self):
        record, rows, geometry = fixture(purpose="frozen")
        result = audit.check_confirmation(
            record, rows, geometry, BUCKET, HOME, audit.Checks(), "mock"
        )
        self.assertEqual(len(result["fresh"]), 30)
        self.assertFalse(record["qualified"])

    def test_nominal_failure_skips_fresh_qualification(self):
        record, rows, geometry = fixture(nominal_failure=True)
        result = audit.check_confirmation(
            record, rows, geometry, BUCKET, HOME, audit.Checks(), "mock"
        )
        self.assertEqual(
            audit.cohort_counts(result["nominal"]), dict(captures=2, attempts=3)
        )
        self.assertEqual(result["fresh"], [])

    def test_seeded_value_or_branch_tamper_is_rejected(self):
        for field in ("values", "q14"):
            record, rows, geometry = fixture()
            record["variations"][0][field] = [
                v + 0.01 for v in record["variations"][0][field]
            ]
            with self.assertRaises(audit.EvidenceError):
                audit.check_confirmation(
                    record, rows, geometry, BUCKET, HOME, audit.Checks(), "mock"
                )

    def test_path_cannot_admit_wrong_nominal_or_clearance(self):
        for field, value in (
            ("nominal_request_sha256", "z" * 64),
            ("minimum_clearance_lower_bound_m", 0.004),
        ):
            record, rows, geometry = fixture()
            record["variations"][0]["local_path"][field] = value
            with self.assertRaises(audit.EvidenceError):
                audit.check_confirmation(
                    record, rows, geometry, BUCKET, HOME, audit.Checks(), "mock"
                )

    def trace_fixture(self, root):
        scene = json.loads((ROOT / "sircus/isaac/scene.json").read_text())
        position = np.zeros((960, 1, 3))
        position[:, :, 0] = 0.6
        position[:, :, 1] = 0.1
        position[:, :, 2] = -0.3
        velocity = np.zeros_like(position)
        forces = np.zeros((960, 1, len(FORCE_NAMES)))
        scorer = ScoreStream(scene, BUCKET, [0.0, 0.0, 1.0])
        for p, v, f in zip(position[:, 0], velocity[:, 0], forces[:, 0]):
            scorer.step(p, v, dict(zip(FORCE_NAMES, f)))
        row = dict(
            scorer.result(),
            id="mock-shot",
            bucket_xy_m=BUCKET,
            board_normal=[0.0, 0.0, 1.0],
            native_release=True,
            invalid=None,
            scalar_score_audit_passed=True,
            native_cache_file="native-0000.npz",
            native_cache_row=0,
        )
        np.savez_compressed(
            root / "native-0000.npz",
            position_m=position,
            velocity_m_s=velocity,
            forces_N=forces,
            record_ids=np.asarray(["mock-shot"]),
        )
        return scene, row

    def test_native_trace_is_independently_rescored_and_id_bound(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            scene, row = self.trace_fixture(root)
            traces = audit.rescore_rows(root, [row], scene, audit.Checks(), "mock")
            self.assertEqual(len(traces["mock-shot"]), 240)
            row["id"] = "another-shot"
            with self.assertRaisesRegex(audit.EvidenceError, "trace ID"):
                audit.rescore_rows(root, [row], scene, audit.Checks(), "mock")

    def test_wrong_score_or_unassigned_arrays_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            scene, row = self.trace_fixture(root)
            row["route_success"] = True
            with self.assertRaisesRegex(audit.EvidenceError, "route_success"):
                audit.rescore_rows(root, [row], scene, audit.Checks(), "mock")
            with self.assertRaisesRegex(
                audit.EvidenceError, "unassigned native arrays"
            ):
                audit.rescore_rows(root, [], scene, audit.Checks(), "mock")


if __name__ == "__main__":
    unittest.main()
