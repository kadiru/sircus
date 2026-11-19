"""Synthetic numeric traces only: no simulator, model, training or real shots."""

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
import candidate_continuation_native_score as subject
from board_route import BoardRoute
from continuous_native_audit import FORCES, expected_empty
from entry_quality import EntryQuality
from rl_contract import ShotScore
from shot_benchmark_reward import build_features, reward


def synthetic_rejected_packet(scene, count=5):
    """Minimal 35-slot all-rejected scoring fixture; no NPZ or native work.

    Outer audit tests must add their own bound reservation/dispatch/guard data.
    """
    assert type(count) is int and 1 <= count <= 35
    cells = [dict(id=f"train-{i:03d}" if i < count else f"inactive-{i:03d}",
                  split="train" if i < count else "inactive", bucket_xy_m=[.67, -.42]) for i in range(35)]
    actions = [dict(id=f"action-{i}") for i in range(35)]
    rows = [dict(action, **expected_empty(scene, cell, 1/240), cell_id=cell["id"], split=cell["split"],
                 invalid="synthetic_decline") for cell, action in zip(cells, actions)]
    return dict(schema=subject.PACKET_SCHEMA, mode="train", batch=0, synthetic=True,
        cells=cells, actions=actions, rows=rows, enabled=[i < count for i in range(35)],
        request_ids=[f"train-{i:06d}" for i in range(count)],
        choices=[dict(request_id=f"train-{i:06d}", context_id=cells[i]["id"], action=actions[i])
                 if i < count else None for i in range(35)],
        reward_features=[None]*35, rewards=[-1. if i < count else None for i in range(35)],
        release_step=10, score_end_step=10, origins_m=np.zeros((35,3)).tolist(), physics=None)


class NativeScoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.scene = json.loads((SOURCE/"scene.json").read_text())
        length, slots = 253, 35
        self.origins = np.asarray([[i*3., 0., 0.] for i in range(slots)])
        self.cells = [dict(id=f"train-{i:03d}" if i < 5 else f"inactive-{i:03d}",
                           split="train" if i < 5 else "inactive", bucket_xy_m=[.67, -.42])
                      for i in range(slots)]
        self.actions = [dict(id=f"action-{i}", height_m=.4, gap_m=.2, tilt_deg=25.,
                             board_x_m=.67, board_y_m=-.1) for i in range(slots)]
        positions = np.repeat(self.origins[None], length, axis=0)
        positions[:, :, :2] += [.67, -.42]
        positions[:2, :, 2] = .3; positions[2, :, 2] = .18; positions[3:, :, 2] = .06
        positions[1:, 3, 2] = -.3
        forces = np.zeros((length, slots, 6)); forces[0, :3, 0] = 1.
        forces[2, 1, 1] = 1.  # Rim-assisted physical capture.
        forces[4, 2, 2] = 1.  # Physical capture with forbidden external contact.
        forces[243:, :3, 2] = 20.  # Ignored after the unchanged success freeze.
        velocity = np.zeros((length, slots, 3)); velocity[1, :3, 2] = 1.
        boards = np.repeat(self.origins[None], length, axis=0); boards += [.67, -.1, .4]
        self.data = dict(time_step=np.arange(11, 11+length), position_m=positions,
            velocity_m_s=velocity, forces_N=forces, joint_position_rad=np.zeros((length, slots, 2, 6)),
            joint_velocity_rad_s=np.zeros((length, slots, 2, 6)), board_position_m=boards,
            board_quaternion_wxyz=np.tile([1., 0., 0., 0.], (length, slots, 1)))
        rows, features = [], []
        for i, (cell, action) in enumerate(zip(self.cells, self.actions)):
            invalid = None if i < 4 else "synthetic_guard_decline" if i == 4 else "inactive"
            if invalid:
                combined = expected_empty(self.scene, cell, 1/240); feature = None
            else:
                bucket = dict(self.scene["bucket"], center_xy_m=cell["bucket_xy_m"])
                score = ShotScore(bucket, self.scene["ball"]["radius_m"])
                quality = EntryQuality(bucket, self.scene["ball"]["radius_m"])
                route = BoardRoute(1/240)
                for j in range(length):
                    p = positions[j, i]-self.origins[i]; f = forces[j, i]
                    score.step(p, float(f[0]), 1/240, float(velocity[j, i, 2]))
                    quality.step(p, float(f[1]), score.entered)
                    route.step(dict(zip(FORCES, map(float, f))))
                    if score.success or p[2] < -.2: break
                outcome = score.result(); q = quality.result(outcome["success"])
                combined = dict(**outcome, **q, **route.result(outcome["success"], q["quality_score"]))
                feature = build_features(self.scene, cell["bucket_xy_m"], positions[:, i]-self.origins[i],
                                         velocity[:, i], forces[:, i], 1/240, [0., 0., 1.])
            rows.append(dict(action, **combined, cell_id=cell["id"], split=cell["split"], invalid=invalid,
                qualification=dict(accepted=not bool(invalid)), path_rejection=None,
                settled=dict(position_error_mm=0., normal_error_deg=0., joint_speed_rad_s=0.,
                    position_drift_mm=0., normal_drift_deg=0., dropper_position_error_mm=0.),
                settled_board_center_m=[.67, -.1, .4], settled_board_normal=[0., 0., 1.],
                scoring_board_normal=[0., 0., 1.], release_ball_center_m=(positions[0,i]-self.origins[i]).tolist()))
            features.append(feature)
        self.path = self.root/"train-000000.npz"
        self.packet = dict(schema=subject.PACKET_SCHEMA, mode="train", synthetic=True,
            cells=self.cells, actions=self.actions, rows=rows, enabled=[i < 5 for i in range(slots)],
            request_ids=[f"train-{i:06d}" for i in range(5)],
            choices=[dict(request_id=f"train-{i:06d}", context_id=self.cells[i]["id"], action=self.actions[i])
                     if i < 5 else None for i in range(slots)],
            reward_features=features, rewards=[reward(rows[i], features[i]) if i < 5 else None for i in range(slots)],
            release_step=10, score_end_step=10+length, origins_m=self.origins.tolist(),
            physics=dict(file=self.path.name, sha256="", samples=length))
        self.rewrite()

    def rewrite(self):
        np.savez_compressed(self.path, **self.data)
        self.packet["physics"]["sha256"] = hashlib.sha256(self.path.read_bytes()).hexdigest()

    def verify(self):
        return subject.verify_score(self.root, self.packet, self.scene)

    def test_clean_assisted_external_miss_and_reject_keep_paid_denominator(self):
        before = copy.deepcopy(self.packet); result = self.verify()
        self.assertEqual(self.packet, before)
        self.assertTrue(result["synthetic"])
        self.assertEqual([result[k] for k in ("requests", "releases", "rejections", "captures", "clean_captures")],
                         [5, 4, 1, 2, 1])
        self.assertEqual([r["return"] for r in result["terminal_results"][:2]], [1.6, 1.5])
        self.assertEqual(result["terminal_results"][4]["return"], -1.)
        self.assertEqual(result["physical"]["new_rewards_recomputed"], 35)

    def test_active_return_and_features_are_independently_rebuilt(self):
        for mutate in (lambda p:p["rewards"].__setitem__(0, 1.5),
                       lambda p:p["reward_features"][0].update(prefix_steps=1)):
            original = copy.deepcopy(self.packet); mutate(self.packet)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError): self.verify()
            self.packet = original

    def test_false_capture_and_clean_labels_fail_even_after_reward_edit(self):
        for mutate in (lambda p:p["rows"][3].update(success=True, route_success=True, route_quality=2),
                       lambda p:p["rows"][1].update(route_quality=2, quality_score=2)):
            original = copy.deepcopy(self.packet); mutate(self.packet)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError): self.verify()
            self.packet = original

    def test_clock_gaps_truncation_or_checksum_changes_fail(self):
        self.data["time_step"][3] += 1; self.rewrite()
        with self.assertRaisesRegex(ValueError, "clock coverage"): self.verify()
        self.data["time_step"][3] -= 1; self.rewrite()
        self.packet["physics"]["samples"] -= 1
        with self.assertRaisesRegex(ValueError, "sample count"): self.verify()
        self.packet["physics"]["samples"] += 1; self.path.write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "checksum"): self.verify()

    def test_rehashed_force_contact_changes_original_score(self):
        self.data["forces_N"][4, 0, 2] = 1.; self.rewrite()
        with self.assertRaisesRegex(ValueError, "rescored outcome"): self.verify()

    def test_rehashed_nonfinite_or_negative_observations_fail(self):
        for key, index, value in (("joint_position_rad", (0,0,0,0), float("nan")),
                                   ("forces_N", (0,0,0), -1.)):
            old = self.data[key][index]; self.data[key][index] = value; self.rewrite()
            with self.subTest(key=key), self.assertRaises(ValueError): self.verify()
            self.data[key][index] = old

    def test_requested_prefix_and_duplicate_identity_fail(self):
        for mutate in (lambda p:p["enabled"].__setitem__(0, False),
                       lambda p:p["enabled"].__setitem__(5, True),
                       lambda p:p["enabled"].__setitem__(0, 1),
                       lambda p:p["request_ids"].__setitem__(1, p["request_ids"][0]),
                       lambda p:p["request_ids"].pop(),
                       lambda p:p["rows"].pop()):
            original = copy.deepcopy(self.packet); mutate(self.packet)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError): self.verify()
            self.packet = original

    def test_inactive_rewards_choices_features_or_release_cannot_be_hidden(self):
        for mutate in (lambda p:p["rewards"].__setitem__(5, -1.),
                       lambda p:p["choices"].__setitem__(5, {}),
                       lambda p:p["reward_features"].__setitem__(5, {}),
                       lambda p:p["rows"][5].update(invalid=None)):
            original = copy.deepcopy(self.packet); mutate(self.packet)
            with self.subTest(mutate=mutate), self.assertRaisesRegex(ValueError, "Inactive"): self.verify()
            self.packet = original

    def test_request_action_or_heldout_association_changes_fail(self):
        for mutate in (lambda p:p["choices"][0].update(request_id="other"),
                       lambda p:p["choices"][0].update(context_id="other"),
                       lambda p:p["cells"][0].update(split="test"),
                       lambda p:p["rows"][0].update(split="validation"),
                       lambda p:p["choices"][0].update(action=dict(id="other"))):
            original = copy.deepcopy(self.packet); mutate(self.packet)
            with self.subTest(mutate=mutate), self.assertRaisesRegex(ValueError, "association"): self.verify()
            self.packet = original

    def test_all_rejected_without_physics_still_counts_every_request(self):
        self.packet = synthetic_rejected_packet(self.scene)
        result = self.verify()
        self.assertEqual((result["requests"], result["releases"], result["rejections"]), (5, 0, 5))
        self.assertEqual(result["physical"]["native_steps"], 0)

    def test_missing_physics_for_actual_release_fails(self):
        self.packet["physics"] = None; self.packet["score_end_step"] = 10
        with self.assertRaisesRegex(ValueError, "Missing physics"): self.verify()

    def test_guard_metric_and_flag_type_cannot_be_bypassed(self):
        for mutate in (lambda p:p["rows"][0]["settled"].update(joint_speed_rad_s=.02),
                       lambda p:p["rows"][0].update(route_success=1),
                       lambda p:p["rows"][0].update(invalid=False),
                       lambda p:p["rewards"].__setitem__(0, True)):
            original = copy.deepcopy(self.packet); mutate(self.packet)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError): self.verify()
            self.packet = original

    def test_unsafe_physics_path_and_unlabeled_evidence_fail(self):
        self.packet["physics"]["file"] = "../train-000000.npz"
        with self.assertRaisesRegex(ValueError, "Unsafe"): self.verify()
        self.packet["physics"]["file"] = self.path.name; self.packet.pop("synthetic")
        with self.assertRaisesRegex(ValueError, "synthetic label"): self.verify()

    def test_import_has_no_simulator_policy_or_optimizer_side_effect(self):
        code = ("import sys;sys.path.insert(0,%r);import candidate_continuation_native_score;"
                "assert 'torch' not in sys.modules;"
                "assert not any(n.startswith(('isaacsim','isaaclab')) for n in sys.modules)") % str(SOURCE)
        subprocess.run([sys.executable, "-c", code], check=True)


if __name__ == "__main__": unittest.main()
