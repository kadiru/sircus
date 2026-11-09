"""Synthetic bound receipts; no original observations, policy or native work."""

import copy
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
import candidate_train_evidence as collector


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, separators=(",", ":"), allow_nan=False).encode()
    path.write_bytes(gzip.compress(raw) if str(path).endswith(".gz") else raw)
    return collector.digest(path)


def fixture(root):
    run = root / "runs/ppo-seed20261122"
    cells = [dict(id=f"train-{i:03d}", split="train", bucket_xy_m=[i / 100, -.1]) for i in range(35)]
    source = dict(commit="frozen", geometry_fingerprint="a" * 64)
    checkpoint = run / "checkpoints/milestone-10000.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"synthetic checkpoint; never loaded")
    cp_sha = collector.digest(checkpoint)
    contract_sha = write(run / "contract.json", dict(scene={"synthetic": True}))
    split_sha = write(root / "split-manifest.json", dict(train=cells, validation=[{"not": "consumed"}], test=[]))
    experiment_sha = write(root / "experiment.json", dict(source=source, split_manifest_file="split-manifest.json",
        split_manifest_sha256=split_sha, protocol=dict(native_training_calls=289, site_counts=dict(train=35))))
    binding_sha = write(run / "experiment-binding.json", dict(source=source, experiment_sha256=experiment_sha,
                                                              split_sha256=split_sha))
    manifest, counters, reasons = {}, dict(attempts=0, captures=0, clean=0, rejected=0), {}
    captured_indices = (0, 1, 2, 9999)
    for batch, count in enumerate([35] * 285 + [7, 6, 6, 6]):
        before = counters["attempts"]
        actions = [dict(id=f"shot-{batch}-{i}", bearing_offset_rad=.01 * i,
                        radius_m=.4, height_m=.35, gap_m=.24, tilt_rad=.5) for i in range(35)]
        rows = []
        for i, action in enumerate(actions):
            index = before + i
            active = i < count
            capture = active and index in captured_indices
            clean = capture and index in (0, 9999)
            invalid = None if active and (capture or index % 10) else ("outer_link_length_bound" if active else "inactive_split")
            row = dict(action, cell_id=cells[i]["id"], split="train", invalid=invalid,
                       route_success=capture, route_quality=2 if clean else int(capture), external_contact_steps=index % 2)
            row.update(qualification=dict(accepted=True, action=copy.deepcopy(action),
                ik={n: dict(solved=True, residual=0., q=[.001 * i] * 6) for n in collector.ARMS}),
                path_proxy=dict(accepted=True, reasons=[]))
            rows.append(row)
            if active:
                for key, value in dict(attempts=1, captures=int(capture), clean=int(clean), rejected=int(invalid is not None)).items():
                    counters[key] += value
                reason = invalid or ("captured" if capture else "external_contact" if row["external_contact_steps"] else "released_miss")
                reasons[reason] = reasons.get(reason, 0) + 1
        packet = dict(schema="sircus-shot-benchmark-native-trial-v1", mode="train", batch=batch,
            source=source, split_manifest_sha256=split_sha, cells=cells, actions=actions, rows=rows,
            actions_normalized=[[.01 * i] * 5 for i in range(35)], enabled=[i < count for i in range(35)],
            training_request_order=list(range(count)),
            checkpoint=dict(file="checkpoints/milestone-10000.pt", sha256=cp_sha, training_attempts=before),
            home=dict(episode_reset=True, protocol="canonical-episode-home-reset-before-visible-drive-v1",
                measured_joint_rad={n: [[.002 * i] * 6 for i in range(35)] for n in collector.ARMS},
                path_eligible=[True] * 35), trajectory_m=["NEVER EXPORTED"])
        relative = f"trials/train-{batch:05d}.json.gz"
        manifest[relative] = write(run / relative, packet)
    # The held-out file deliberately does not exist; a TRAIN-only collector
    # must not open it, even though the trusted original audit binds it.
    manifest["trials/eval-test-99999.json.gz"] = "f" * 64
    progress = dict(status="completed", method="ppo", seed=20261122, smoke=False,
        source=source, split_manifest_sha256=split_sha, train=dict(counters, failures=reasons),
        checkpoints=[dict(training_attempts=10000, checkpoint_file="checkpoints/milestone-10000.pt", checkpoint_sha256=cp_sha)],
        source_trial_sha256=manifest)
    progress_sha = write(run / "progress.json", progress)
    audit = dict(status="passed", smoke_only=False, method="ppo", seed=20261122,
        paid_train_attempts=10000, evaluation_attempts=420, source=source,
        source_progress_sha256=progress_sha, experiment_sha256=experiment_sha,
        split_manifest_sha256=split_sha, source_binding_sha256=binding_sha,
        source_contract_sha256=contract_sha, source_trial_sha256=manifest,
        checkpoint_sha256={"checkpoints/milestone-10000.pt": cp_sha})
    audit_sha = write(Path(str(run) + ".audit.json"), audit)
    return dict(audit_sha=audit_sha, progress_sha=progress_sha, experiment_sha=experiment_sha,
                split_sha=split_sha, checkpoint_sha=cp_sha)


class CollectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.expected = fixture(self.root)
        self.run = self.root / "runs/ppo-seed20261122"

    def collect(self):
        return collector.collect(self.root, **self.expected)

    def rebind_packet(self, mutate, relative="trials/train-00000.json.gz"):
        path = self.run / relative
        packet = json.loads(gzip.decompress(path.read_bytes()))
        mutate(packet)
        new_sha = write(path, packet)
        progress = json.loads((self.run / "progress.json").read_text())
        progress["source_trial_sha256"][relative] = new_sha
        self.expected["progress_sha"] = write(self.run / "progress.json", progress)
        audit_path = Path(str(self.run) + ".audit.json")
        audit = json.loads(audit_path.read_text())
        audit["source_trial_sha256"][relative] = new_sha
        audit["source_progress_sha256"] = self.expected["progress_sha"]
        self.expected["audit_sha"] = write(audit_path, audit)

    def test_full_paid_budget_retains_every_success_and_never_reads_evaluation(self):
        original = collector.read
        visited = []

        def check(path, expected):
            self.assertNotIn("eval-", str(path))
            visited.append(Path(path))
            return original(path, expected)

        with patch.object(collector, "read", side_effect=check):
            result = self.collect()
        self.assertEqual(result["training_totals"]["attempts"], 10000)
        self.assertEqual((result["successful_shots"], result["clean_successes"]), (4, 2))
        self.assertEqual(len(result["canonical_starts"]), 35)
        self.assertEqual(len(result["train_trial_sha256"]), 289)
        self.assertEqual(len({x["success_key"] for x in result["successes"]}), 4)
        self.assertEqual([x["clean"] for x in result["successes"]], [True, False, False, True])
        self.assertEqual(result["successes"][0]["action"]["id"], "shot-0-0")
        self.assertEqual(result["successes"][1]["recorded_starts_rad"]["Arm10"], [.002] * 6)
        self.assertEqual(result["canonical_starts"]["train-001"]["recorded_starts_rad"]["Arm14"], [.002] * 6)
        self.assertNotIn("NEVER EXPORTED", json.dumps(result))
        self.assertEqual(result["evaluated_trial_files_read"], 0)
        self.assertEqual(result["optimizer_updates"], 0)
        unsigned = dict(result); fingerprint = unsigned.pop("fingerprint")
        self.assertEqual(collector.canonical_hash(unsigned), fingerprint)
        self.assertTrue(any(path.name == "train-00288.json.gz" for path in visited))

    def test_packet_hash_corruption_is_rejected_before_labels_are_used(self):
        path = self.run / "trials/train-00000.json.gz"
        path.write_bytes(path.read_bytes() + b"corrupt")
        with self.assertRaisesRegex(ValueError, "SHA mismatch"):
            self.collect()

    def test_changed_model_or_pending_audit_fails_closed(self):
        (self.run / "checkpoints/milestone-10000.pt").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "checkpoint bytes"):
            self.collect()
        (self.run / "checkpoints/milestone-10000.pt").write_bytes(b"synthetic checkpoint; never loaded")
        audit_path = Path(str(self.run) + ".audit.json")
        audit = json.loads(audit_path.read_text()); audit["status"] = "pending"
        self.expected["audit_sha"] = write(audit_path, audit)
        with self.assertRaisesRegex(ValueError, "passed production"):
            self.collect()

    def test_changed_paid_cell_or_heldout_active_context_is_rejected(self):
        self.rebind_packet(lambda packet: packet["cells"][0].update(split="test"))
        with self.assertRaisesRegex(ValueError, "TRAIN bucket"):
            self.collect()

    def test_duplicate_paid_indices_cannot_inflate_coverage(self):
        self.rebind_packet(lambda packet: packet["training_request_order"].__setitem__(1, 0))
        with self.assertRaisesRegex(ValueError, "request order"):
            self.collect()

    def test_missing_coupled_success_fit_is_not_silently_dropped(self):
        self.rebind_packet(lambda packet: packet["rows"][0]["qualification"]["ik"].pop("Arm14"))
        with self.assertRaisesRegex(ValueError, "coupled arm fit"):
            self.collect()

    def test_success_geometry_changed_after_qualification_is_rejected(self):
        self.rebind_packet(lambda packet: packet["actions"][0].update(gap_m=.25))
        with self.assertRaisesRegex(ValueError, "shot geometry"):
            self.collect()

    def test_wrong_total_or_checkpoint_lineage_is_rejected(self):
        self.rebind_packet(lambda packet: packet["checkpoint"].update(training_attempts=1))
        with self.assertRaisesRegex(ValueError, "checkpoint lineage"):
            self.collect()

    def test_path_escape_and_nonfinite_exports_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "escaped"):
            collector._inside(self.root, "../outside.json")
        with self.assertRaisesRegex(ValueError, "finite"):
            collector._joints({n: [float("nan")] * 6 for n in collector.ARMS})
        target = self.root / "nan.json"; target.write_text('{"q": NaN}')
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            collector.read(target, collector.digest(target))

    def test_script_is_portable_over_stdin_and_rejects_output_inside_inputs(self):
        code = (SOURCE / "candidate_train_evidence.py").read_text()
        args = [sys.executable, "-", "--benchmark", str(self.root)]
        for key, value in self.expected.items():
            args += ["--" + key.replace("_", "-"), value]
        args += ["--output", "-"]
        result = subprocess.run(args, input=code, capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(result.stdout)["successful_shots"], 4)
        args[-1] = str(self.root / "forbidden.json")
        result = subprocess.run(args, input=code, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("separate from frozen", result.stderr)
        self.assertFalse((self.root / "forbidden.json").exists())


if __name__ == "__main__":
    unittest.main()
