"""Complete tiny CPU/mocked study with cached branches and invalid confirmations."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from sircus.mjlab import expand_far
from sircus.mjlab.expand_bank import sha256
from sircus.mjlab.learning_contract import digest
from sircus.mjlab.learning_search import VERSION as BANK_VERSION
from sircus.mjlab.bank_study import unresolved_row
from sircus.mjlab.verify_far import verify


class FullMockStudyTests(unittest.TestCase):
    def test_mock_native_flow_retains_rejections_and_passes_artifact_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prior = root / "prior"
            prior.mkdir()
            old = root / "old"
            old.mkdir()
            original = dict(
                version=BANK_VERSION, templates=[dict(source_trial_id="original-train")]
            )
            original["sha256"] = digest(original)
            (old / "proposal-bank.json").write_text(json.dumps(original))
            np.savez(old / "geometry-test.npz", mask=np.zeros((12, 1), dtype=bool))
            indices = (8, 9, 10, 12, 19, 21, 23, 26, 35, 39, 41, 48)
            targets = [
                dict(
                    id=f"old-test-{i:02d}",
                    original_test_index=i,
                    bucket_xy_m=[1.1 + i * 0.001, -0.5],
                    role="targeted_development",
                )
                for i in indices
            ]
            spec = dict(
                scene={"test_fixture": True},
                old_bank_study_fingerprint="old-study",
                old_bank_sha256=original["sha256"],
                old_bank_file_sha256=sha256(old / "proposal-bank.json"),
                old_test_geometry_sha256=sha256(old / "geometry-test.npz"),
                development_layouts=targets,
                old_bounds=expand_far.OLD_BOUNDS,
                wide_bounds=expand_far.WIDE_BOUNDS,
            )
            spec["fingerprint"] = digest(spec)
            report = dict(
                status="completed",
                contract_fingerprint=spec["fingerprint"],
                native_search_attempts=0,
                confirmation_attempts=0,
                records=0,
            )
            (prior / "contract.json").write_text(json.dumps(spec))
            (prior / "report.json").write_text(json.dumps(report))
            (prior / "shots.jsonl").write_text("")
            selected = [
                t
                for t in targets
                if t["original_test_index"] in expand_far.MISSING_INDICES
            ]
            a = SimpleNamespace(
                prior_search=prior,
                old_bank_study=old,
                output=root / "result",
                assets=root / "assets",
                parent_study=root / "parent",
                seed_sources=4,
                native_per_target=1,
                proposal_rounds=1,
                proposals=32,
                batch=8,
                top=2,
                gpu_lock=root / "unused.lock",
                gpu_wait_timeout=1.0,
            )

            def propose(xy, n, rng, scene, seeds, **kwargs):
                return np.tile([0.65, 0, 0.36, 25, 0.24], (n, 1)), [
                    dict(
                        proposal_family="full-height-global",
                        seed_source_id=None,
                        round_index=0,
                    )
                    for _ in range(n)
                ]

            def audit(scene, assets, xy, values, **kwargs):
                mask = np.ones(values.shape[:2], dtype=bool)
                if len(xy) <= 8:
                    mask[-1] = False
                return mask, dict(
                    q10=np.zeros((*mask.shape, 6)), q14=np.zeros((*mask.shape, 6))
                )

            def native(env, xy, values, q10, q14, eligible, **kwargs):
                rows = []
                for i, point in enumerate(xy):
                    row = unresolved_row(point.tolist())
                    if eligible[i]:
                        row.update(
                            invalid=None,
                            route_success=True,
                            route_quality=2,
                            success=True,
                            complete_success=True,
                            board_bounce=True,
                            whole_ball_entry=True,
                            max_retention_s=1.0,
                            single_bounce_success=True,
                            board_contact_episodes=1,
                            closest_opening_distance_m=0.01,
                            native_release=True,
                            q10=q10[i].tolist(),
                            q14=q14[i].tolist(),
                            pose_gate=dict(
                                board_position_error_mm=0.0,
                                board_normal_error_deg=0.0,
                                dropper_error_mm=0.0,
                            ),
                            endpoint_collision_pairs=[],
                        )
                    rows.append(row)
                return rows

            with contextlib.ExitStack() as stack:
                stack.enter_context(
                    mock.patch.object(
                        expand_far,
                        "load_prior",
                        return_value=(spec, report, [], original, selected, []),
                    )
                )
                stack.enter_context(
                    mock.patch.object(expand_far, "far_proposals", side_effect=propose)
                )
                stack.enter_context(
                    mock.patch.object(
                        expand_far,
                        "outer_geometry_mask",
                        side_effect=lambda scene, xy, values, **kw: (
                            np.ones(values.shape[:2], bool),
                            np.ones(values.shape[:2], bool),
                        ),
                    )
                )
                stack.enter_context(
                    mock.patch(
                        "sircus.mjlab.expand_geometry.audit_values", side_effect=audit
                    )
                )
                stack.enter_context(
                    mock.patch(
                        "sircus.mjlab.bank_study.run_bank_rows", side_effect=native
                    )
                )
                stack.enter_context(
                    mock.patch(
                        "sircus.mjlab.gpu_slot.gpu_slot",
                        return_value=contextlib.nullcontext(),
                    )
                )
                stack.enter_context(
                    mock.patch.dict(
                        "sys.modules",
                        {
                            "sircus.mjlab.expanded_env": SimpleNamespace(
                                ExpandedEnvironment=lambda *a, **kw: object()
                            ),
                            "torch": SimpleNamespace(set_num_threads=lambda n: None),
                        },
                    )
                )
                stack.enter_context(
                    mock.patch(
                        "sircus.mjlab.expand_far.importlib.metadata.version",
                        return_value="mock",
                    )
                )
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                expand_far.run(a)
            final = json.loads((a.output / "report.json").read_text())
            self.assertEqual(final["native_search_attempts"], 6)
            self.assertEqual(final["confirmation_attempts"], 60)
            self.assertEqual(final["additions"], 6)
            self.assertGreater(final["confirmation_statistics"]["invalid"], 0)
            result = verify(a.output, prior, old)
            failures = [c["name"] for c in result["checks"] if not c["passed"]]
            # Only the deliberately tiny mocked budget differs from production.
            self.assertEqual(failures, ["bounded_native_budget"])


if __name__ == "__main__":
    unittest.main()
