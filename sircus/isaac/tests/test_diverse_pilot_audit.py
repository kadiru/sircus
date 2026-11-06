"""Synthetic corruption checks for the separate hybrid/settling audit."""

import copy
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

ISAAC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ISAAC))
import continuous_policy as old
import diverse_pilot_audit as audit
import diverse_support as support_module
from continuous_feasibility import ACTION_NAMES, LOW, HIGH, canonical_hash, decode_action
from expanded_farm import make_spec
from test_diverse_policy import fixture as policy_fixture


def seal(value):
    value.pop("fingerprint", None)
    value["fingerprint"] = canonical_hash(value)
    return value


def fixture():
    spec = make_spec(json.loads((ISAAC/"scene.json").read_text()), 10)
    _, support = policy_fixture()
    home = {name: [0.]*6 for name in ("Arm10", "Arm14")}
    records = []
    for row, cell in zip(support["cells"], spec["cells"]):
        row["cell_id"], row["bucket_xy_m"] = cell["id"], cell["bucket_xy_m"]
        row["anchor_checks"] = []
        row["map_candidates"] = row["region_count"]
        prior = np.array([0., np.clip(np.linalg.norm(np.array([.67, -.10])-cell["bucket_xy_m"]), .29, 1.19), .35, .24, math.radians(29)])
        pairs = []
        for region in row["regions"]:
            values = np.array([region["anchor_action"][key] for key in ACTION_NAMES])
            region["anchor_action"] = decode_action(cell, values, region["anchor_action"]["id"])
            travel = np.mean([np.square(region["anchor_ik"][name]["q"]).sum() for name in home])
            score = float(np.square((values-prior)/(HIGH-LOW)).sum()+.01*travel)
            region.update(anchor_score=score, anchor_path=dict(accepted=True))
            record = dict(cell_id=cell["id"], accepted=True, action=copy.deepcopy(region["anchor_action"]),
                          ik=copy.deepcopy(region["anchor_ik"]), proposal=region["proposal"])
            records.append(record)
            row["anchor_checks"].append(dict(proposal=region["proposal"], action_id=record["action"]["id"], score=score, path=dict(accepted=True)))
            pairs.append(region)
        pairs.sort(key=lambda r: (r["anchor_score"], r["proposal"]))
        for i, region in enumerate(pairs):
            region["id"] = i
        row["regions"] = pairs
        row["coverage"] = support_module._coverage(pairs)
    feasibility = seal(dict(source_fingerprint=spec["fingerprint"], ik_protocol="full_robust_v2",
                            asset_sha256={"poses.json": "a"*64}, records=records))
    support.update(source_fingerprint=spec["fingerprint"], feasibility_fingerprint=feasibility["fingerprint"],
        canonical_home_joint_rad=home, asset_sha256=feasibility["asset_sha256"], anchor_path_samples=16)
    seal(support)
    return support, feasibility, spec


class SupportAuditTests(unittest.TestCase):
    def setUp(self):
        self.support, self.map, self.spec = fixture()

    def verify(self):
        return audit.validate_support_evidence(self.support, self.map, self.spec)

    def test_every_path_qualified_anchor_retained(self):
        value = self.verify()
        self.assertEqual(value["retained_regions"], 100)
        self.assertEqual(value["supported_cells"], 34)
        self.assertTrue(value["all_path_qualified_anchors_retained"])

    def test_pruned_anchor_rejected_even_with_new_support_fingerprint(self):
        self.support["cells"][2]["regions"].pop()
        self.support["cells"][2]["region_count"] -= 1
        seal(self.support)
        with self.assertRaisesRegex(ValueError, "pruned or duplicated"):
            self.verify()

    def test_changed_region_rank_width_or_seed_rejected(self):
        for key in ("rank", "width", "seed"):
            changed = copy.deepcopy(self.support)
            region = changed["cells"][2]["regions"][0]
            if key == "rank":
                region["anchor_score"] += .01
            elif key == "width":
                region["low"][2] += .001
            else:
                region["anchor_ik"]["Arm14"]["q"][2] += .01
            seal(changed)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "region/action/width"):
                audit.validate_support_evidence(changed, self.map, self.spec)

    def test_independent_selected_region_decode_and_unresolved_dummy(self):
        indices = np.zeros(35, dtype=int); indices[2] = 2
        actual = audit.independent_physical_actions(np.zeros((35, 5)), indices, self.support, self.spec["cells"])
        expected = support_module.physical_actions(np.zeros((35, 5)), self.support, self.spec["cells"], indices, allow_unresolved=True)
        np.testing.assert_array_equal(actual, expected)
        indices[0] = 1
        with self.assertRaises(ValueError):
            audit.independent_physical_actions(np.zeros((35, 5)), indices, self.support, self.spec["cells"])

    def test_study_controller_change_is_explicit_shared_scoring_guards_preserved(self):
        from diverse_pilot import make_study
        study = make_study(dict(spec=self.spec, source=dict(camera_sha256="a"*64)), 4, self.support)
        study.update(support_sha256="b"*64, feasibility_sha256="c"*64)
        seal(study)
        audit.validate_diverse_study(study, self.spec, self.support, "b"*64, "c"*64)
        for key in ("seed", "controller", "gate", "budget", "window"):
            changed = copy.deepcopy(study)
            if key == "seed": changed["seed"] += 1
            elif key == "controller": changed["settling_protocol"]["transit_tail_max_s"] += 1
            elif key == "gate": changed["settling_protocol"]["release_position_mm"] = 3
            elif key == "budget": changed["training_batches"] = 8
            else: changed["measurement_window_s"] = .1
            seal(changed)
            with self.subTest(key=key), self.assertRaises(ValueError):
                audit.validate_diverse_study(changed, self.spec, self.support, "b"*64, "c"*64)


def settling_fixture():
    from expanded_pose_menu import endpoint_targets
    from prepare_assets import matrix
    from settled_transit import protocol
    support, _, spec = fixture()
    scene = spec["scene"]
    base = matrix([scene["arms"]["base_spacing_m"], 0, 0], [0, 0, np.pi])
    mount = matrix([0, 0, scene["board"]["tool_offset_m"]], [0, 0, scene["board"]["mount_yaw_rad"]])
    states = {name: {key: [[0.]*6 for _ in range(35)] for key in ("position_rad", "velocity_rad_s", "target_rad")}
              for name in ("Arm10", "Arm14")}
    gains = {name: {key: [[100.]*6 for _ in range(35)] for key in ("effort", "stiffness", "damping")}
             for name in states}
    transit = [dict(phase="start", step=0, state=states, original_gains=gains),
               dict(phase="tail", step=60, state=states),
               dict(phase="restored_original_hold", step=300, state=states, restored_gains=gains)]
    actions, records, rows = [], [], []
    for cell in spec["cells"]:
        action = decode_action(cell, [0., .4, .35, .24, math.radians(29)], "test")
        target = endpoint_targets(scene, cell, action)
        board = base@target["Arm14"]@mount
        m = dict(iteration=0, position_error_mm=0., normal_error_deg=0.,
            dropper_position_error_mm=0., joint_speed_rad_s=0., position_drift_mm=0.,
            normal_drift_deg=0., dropper_drift_mm=0., board_center_m=board[:3, 3].tolist(),
            board_normal=board[:3, 2].tolist(), dropper_flange_m=target["Arm10"][:3, 3].tolist(),
            arms={name: {key: [0.]*6 for key in states[name]} for name in states})
        # acos of a numerically unit normal need not be exactly zero.
        m["normal_error_deg"] = float(np.rad2deg(np.arccos(np.clip(board[:3, 2]@board[:3, 2], -1, 1))))
        m["normal_drift_deg"] = m["normal_error_deg"]
        m.update(previous_board_center_m=m["board_center_m"], previous_board_normal=m["board_normal"],
                 previous_dropper_flange_m=m["dropper_flange_m"])
        history = [dict(copy.deepcopy(m), measurement_start_step=300+60*j, physics_step=360+60*j) for j in range(7)]
        actions.append(action); records.append(history); rows.append(dict(invalid=None, settled=history[-1]))
    trial = dict(setup_controller=protocol(), settling_history=records, transit_history=transit,
        correction_history=[], actions=actions, rows=rows, release_step=720)
    return trial, spec, dict(settling_protocol=protocol())


class SettlingAuditTests(unittest.TestCase):
    def test_original_gates_and_actual_windows_verified(self):
        trial, spec, study = settling_fixture()
        value = audit.verify_settling_history(trial, spec["scene"], spec["cells"], study)
        self.assertEqual(value["measured_windows"], 245)
        self.assertTrue(value["original_holding_gains_restored"])

    def test_changed_readback_restore_cadence_and_final_admission_fail(self):
        for key in ("restore", "cadence", "pose", "speed", "row"):
            trial, spec, study = settling_fixture()
            if key == "restore": trial["transit_history"][-1]["restored_gains"] = copy.deepcopy(trial["transit_history"][-1]["restored_gains"]); trial["transit_history"][-1]["restored_gains"]["Arm10"]["stiffness"][0][0] += 1
            elif key == "cadence": trial["settling_history"][0][1]["physics_step"] += 1
            elif key == "pose": trial["settling_history"][0][0]["position_error_mm"] = 1
            elif key == "speed": trial["settling_history"][0][0]["joint_speed_rad_s"] = .001
            else: trial["rows"][0]["settled"] = dict(trial["rows"][0]["settled"], position_drift_mm=.2)
            with self.subTest(key=key), self.assertRaises(ValueError):
                audit.verify_settling_history(trial, spec["scene"], spec["cells"], study)

    def test_correction_must_be_quiet_and_only_for_incorrect_arm(self):
        trial, spec, study = settling_fixture()
        m = trial["settling_history"][0][-1]
        trial["correction_history"] = [dict(cell_id=spec["cells"][0]["id"], iteration=0, measured=m, arms=["Arm14"], fits={})]
        with self.assertRaisesRegex(ValueError, "already within tolerance"):
            audit.verify_settling_history(trial, spec["scene"], spec["cells"], study)
        m["arms"]["Arm10"]["position_rad"][0] = .03
        with self.assertRaisesRegex(ValueError, "unfinished joint transit"):
            audit.verify_settling_history(trial, spec["scene"], spec["cells"], study)


class SetupCheckAuditTests(unittest.TestCase):
    def make_setup(self, root):
        from collections import Counter
        from continuous_feasibility import split_for_index
        root = root.resolve()
        trial, spec, _ = settling_fixture()
        cells = [dict(cell, split=split_for_index(i)) for i, cell in enumerate(spec["cells"])]
        old, new = root/"old", root/"new"
        old.mkdir(); new.mkdir(); (old/"trials").mkdir()
        def write(where, name, value): (where/name).write_text(json.dumps(value))
        write(old, "contract.json", spec)
        old_study = dict(schema="sircus-native-feasibility-guided-pilot-v1", fingerprint="a"*64, cells=cells)
        write(old, "study.json", old_study)
        enabled = [c["split"] == "train" for c in cells]
        records, batches, before, after, paired = [], [], [], [], []
        for index in range(4):
            rows = [dict(cell_id=cell["id"], split=cell["split"], invalid=(None if i%3 == 0 else "endpoints_not_settled") if enabled[i] else "inactive_split")
                    for i, cell in enumerate(cells)]
            previous = dict(schema=old_study["schema"], study_fingerprint=old_study["fingerprint"], mode="train", batch=index,
                rows=rows, actions=trial["actions"], enabled=enabled)
            name = f"trials/train-{index:04d}.json"; write(old, name, previous)
            sha = audit.digest(old/name)
            batches.append(dict(file=name, sha256=sha, mode="train", batch=index))
            invalid = [None if use else "inactive_split" for use in enabled]
            records.append(dict(source_trial=name, source_trial_sha256=sha, actions=trial["actions"], enabled=enabled,
                invalid=invalid, settling_history=trial["settling_history"], transit_history=trial["transit_history"], correction_history=[]))
            for i, use in enumerate(enabled):
                if use:
                    before.append(rows[i]["invalid"]); after.append(None)
                    m = trial["settling_history"][i][-1]
                    paired.append(dict(cell_id=cells[i]["id"], batch=index, before_invalid=before[-1], after_invalid=None,
                        position_error_mm=m["position_error_mm"], position_drift_mm=m["position_drift_mm"], joint_speed_rad_s=m["joint_speed_rad_s"]))
        write(old, "result.json", dict(batches=batches))
        summary = dict(schema="sircus-fixed-request-setup-check-v1", status="completed", source_result_sha256="cd90e6ec9c1e1bc3729ca5113b67d98c6ebf680f0964cec68065acb1f98138ca",
            geometry_fingerprint=spec["fingerprint"], settling_protocol=trial["setup_controller"], ball_releases=0,
            attempts=100, before_admitted=before.count(None), after_admitted=100,
            before_failures=dict(Counter(x for x in before if x)), after_failures={}, repaired=sum(x is not None for x in before),
            regressions=0, paired=paired, selection="First four saved training batches in original order, all 25 training contexts; no held-out/outcome filtering")
        write(new, "setup-check.json", summary); write(new, "setup-check-raw.json", dict(records=records))
        return old, new

    def verify(self, old, new):
        original_digest = audit.digest
        def bound_digest(path):
            if Path(path) == old/"result.json":
                return "cd90e6ec9c1e1bc3729ca5113b67d98c6ebf680f0964cec68065acb1f98138ca"
            return original_digest(path)
        with patch.object(audit, "digest", side_effect=bound_digest), \
             patch("feasibility_guided_audit.backend_for_run", return_value=SimpleNamespace(validate_study=lambda *args: None)):
            return audit.audit_setup_check(new, old)

    def test_setup_only_counts_all_fixed_requests_and_no_release_claim(self):
        with tempfile.TemporaryDirectory() as name:
            old, new = self.make_setup(Path(name))
            result = self.verify(old, new)
            self.assertEqual(result["checked_attempts"], 100)
            self.assertEqual(result["after_admitted"], 100)
            self.assertEqual(result["ball_releases"], 0)
            self.assertEqual(len(result["history_audits"]), 4)
            self.assertTrue(all("no physical ball release" in h["cutoff_scope"] for h in result["history_audits"]))

    def test_setup_source_request_order_summary_and_heldout_corruption_rejected(self):
        for key in ("request", "order", "summary", "heldout", "gain"):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as name:
                old, new = self.make_setup(Path(name))
                raw = json.loads((new/"setup-check-raw.json").read_text())
                summary = json.loads((new/"setup-check.json").read_text())
                if key == "request": raw["records"][0]["actions"][2]["height_m"] += .01
                elif key == "order": raw["records"][0], raw["records"][1] = raw["records"][1], raw["records"][0]
                elif key == "summary": summary["repaired"] += 1
                elif key == "heldout": raw["records"][0]["invalid"][0] = None
                else: raw["records"][0]["transit_history"][-1]["restored_gains"]["Arm10"]["stiffness"][0][0] += 1
                (new/"setup-check-raw.json").write_text(json.dumps(raw)); (new/"setup-check.json").write_text(json.dumps(summary))
                with self.assertRaises(ValueError): self.verify(old, new)


@unittest.skipIf(old.torch is None, "Configured CPU Torch unavailable")
class HybridChainAuditTests(unittest.TestCase):
    def make_run(self, root):
        from diverse_pilot import make_study
        import diverse_policy as policy
        from farm_contract import reward, statistics
        torch = old.torch; torch.set_num_threads(1); torch.manual_seed(20261112)
        support, feasibility, spec = fixture()
        root.mkdir(exist_ok=True); (root/"trials").mkdir()
        def write(name, value): (root/name).write_text(json.dumps(value, allow_nan=False))
        write("support.json", support); write("feasibility.json", feasibility); write("contract.json", spec)
        study = make_study(dict(spec=spec, source=dict(camera_sha256="a"*64)), 4, support)
        study.update(support_sha256=audit.digest(root/"support.json"), feasibility_sha256=audit.digest(root/"feasibility.json")); seal(study); write("study.json", study)
        model = policy.DiversePolicy(support)
        optimizer = torch.optim.Adam(model.parameters(), lr=.0003)
        obs = torch.from_numpy(policy.build_observations(spec["cells"]))
        train = [i for i, cell in enumerate(study["cells"]) if cell["split"] == "train"]
        updates, batches, assessments, buffer = [], [], [], []
        def checkpoint(name, count):
            payload = policy.checkpoint_payload(model, optimizer, study_fingerprint=study["fingerprint"],
                geometry_fingerprint=spec["fingerprint"], domain_fingerprint=study["domain"]["fingerprint"],
                support_fingerprint=support["fingerprint"], update_count=len(updates), attempted_shots=count*25, seed=study["seed"])
            torch.save(payload, root/name)
            return dict(file=name, sha256=audit.digest(root/name), policy_hash=policy.weight_hash(model), actor_hash=policy.actor_weight_hash(model))
        def trial(mode, batch, cp):
            picked = policy.sample(model, obs, deterministic=mode == "eval")
            physical = support_module.physical_actions(picked["actions_normalized"], support, spec["cells"], picked["region_indices"], allow_unresolved=True)
            actions = [decode_action(cell, values, f"{mode}-{batch}-{i}") for i, (cell, values) in enumerate(zip(spec["cells"], physical))]
            enabled = [i in train for i in range(35)] if mode == "train" else [True]*35
            rows = [dict(action, cell_id=cell["id"], split=cell["split"],
                invalid="no_geometry_support" if i == 0 and enabled[i] else "synthetic_reject",
                route_success=False, route_quality=0, single_bounce_success=False)
                for i, (cell, action) in enumerate(zip(study["cells"], actions))]
            returns = torch.tensor([reward(row) for row in rows], dtype=torch.float32)
            home = dict(protocol=audit.HOME_PROTOCOL, episode_reset=True, physics_step=100,
                path_eligible=[True]*35, path_checks=[dict(accepted=True) for _ in range(35)],
                **{key: {arm: [[0.]*6 for _ in range(35)] for arm in ("Arm10", "Arm14")}
                   for key in ("target_joint_rad", "measured_joint_rad", "measured_joint_velocity_rad_s")})
            name = f"trials/{mode}-{len(batches)+len(assessments):04d}.json"
            data = dict(schema=study["schema"], mode=mode, batch=batch, update_count=len(updates),
                study_fingerprint=study["fingerprint"], checkpoint=cp, policy_hash=policy.weight_hash(model),
                observations=obs.tolist(), latent=picked["latent"].tolist(), actions_normalized=picked["actions_normalized"].tolist(),
                region_indices=picked["region_indices"].tolist(), region_logp=picked["region_logp"].tolist(),
                residual_logp=picked["residual_logp"].tolist(), logp=picked["logp"].tolist(),
                actions=actions, rows=rows, rewards=returns.tolist(), enabled=enabled,
                active_cell_ids=[cell["id"] for cell, use in zip(study["cells"], enabled) if use],
                optimizer_samples=25 if mode == "train" else 0, deterministic=mode == "eval", policy_action_used=True,
                home=home, release_step=200, support_fingerprint=support["fingerprint"])
            write(name, data)
            return dict(mode=mode, batch=batch, update_count=len(updates), file=name, sha256=audit.digest(root/name),
                **statistics([row for row, use in zip(rows, enabled) if use])), (obs[train], picked["latent"][train], picked["logp"][train], returns[train])
        initial = checkpoint("policy-initial.pt", 0)
        for repeat in range(2):
            summary, _ = trial("eval", repeat, initial); assessments.append(summary)
        for batch in range(4):
            cp = checkpoint(f"policy-before-{batch:03d}.pt", batch)
            summary, data = trial("train", batch, cp); batches.append(summary); buffer.append(data)
        change = policy.update(model, optimizer, *[torch.cat(group) for group in zip(*buffer)], study["ppo"])
        change.update(update_count=1, training_batches=[0, 1, 2, 3], sample_count=100); updates.append(change)
        final = checkpoint("policy-final.pt", 4)
        for repeat in range(2):
            summary, _ = trial("eval", repeat, final); assessments.append(summary)
        result = dict(schema=study["schema"], status="completed", study_fingerprint=study["fingerprint"], geometry_fingerprint=spec["fingerprint"],
            workcells=35, articulations=70, initial=initial, final=final, initial_hash=initial["policy_hash"], final_hash=final["policy_hash"],
            batches=batches, assessments=assessments, updates=updates, coverage=[])
        write("result.json", result)
        return result

    def test_fresh_chain_joint_likelihood_all_rejects_and_heldout_exclusion(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); self.make_run(root)
            result = audit.verify_checkpoint_chain(root)
            self.assertEqual(result["optimizer_samples"], 100)
            self.assertEqual(result["train_statistics"]["invalid"], 100)
            self.assertTrue(result["fresh_seeded_initialization_verified"])
            self.assertEqual(len(result["checkpoints"]), 6)
            self.assertLess(result["max_logp_error"], 1e-5)

    def test_rehashed_constituent_or_category_or_reward_corruption_rejected(self):
        for key in ("region_logp", "residual_logp", "region_indices", "rewards"):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as name:
                root = Path(name); result = self.make_run(root)
                summary = result["batches"][0]; path = root/summary["file"]; data = json.loads(path.read_text())
                data[key][2] += 1
                path.write_text(json.dumps(data)); summary["sha256"] = audit.digest(path)
                (root/"result.json").write_text(json.dumps(result))
                with self.assertRaises(ValueError): audit.verify_checkpoint_chain(root)

    def test_first_update_is_separately_labeled_partial(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); result = self.make_run(root)
            (root/"progress.json").write_text(json.dumps(dict(schema=result["schema"],
                study_fingerprint=result["study_fingerprint"], batches=result["batches"], updates=result["updates"])))
            receipt = audit.verify_first_update(root)
            self.assertEqual(receipt["schema"], "sircus-diverse-settled-first-update-audit-v1")
            self.assertEqual(receipt["optimizer_samples"], 100)
            self.assertTrue(receipt["geometry_fingerprint"])


if __name__ == "__main__": unittest.main()
