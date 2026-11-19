"""Bounded PhysX diagnostic grid; search results are NOT learned inference."""

import copy
import datetime
import hashlib
import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch
from generalized_rl import Policy, SceneShot
from ik_diagnostics import solve_board
from learning_contract import contract, observation
from rl_contract import action_values

DISTANCES = [0.20, 0.25, 0.30, 0.35, 0.40]
TILTS = [12.0, 16.0, 20.0, 24.0, 28.0, 36.0, 45.0]
GAPS = [0.06, 0.10, 0.15, 0.20, 0.25]


def layouts():
    return [
        {
            "id": f"distance-diagnostic-{i}",
            "board_center_m": [0.67, -0.1, 0.4],
            "bucket_xy_m": [0.67, -0.1 - d],
            "distance_m": d,
            "difficulty": "near" if d < 0.267 else "medium" if d < 0.333 else "far",
        }
        for i, d in enumerate(DISTANCES)
    ]


class AuditShot(SceneShot):
    def __init__(self, *args):
        super().__init__(*args)
        self.use_extended = False
        self.ik_cache = {}
        self.ik_evidence = []

    def ik(self, target, name):
        if name != "Arm14":
            return super().ik(target, name)
        key = (self.use_extended, tuple(np.round(target.ravel(), 10)))
        if key not in self.ik_cache:
            self.ik_cache[key] = solve_board(
                target,
                self.links,
                self.poses["board_joint_rad"],
                extended=self.use_extended,
            )
        result = self.ik_cache[key]
        self.ik_evidence.append(result)
        if not result["solved"]:
            self.ik_failure_component = name
            raise ValueError("Diagnostic IK unresolved")
        return np.array(result["q"])


def run(sim, arms, ball, camera, contact, cfg, poses, args, atomic):
    torch.set_num_threads(2)
    out = Path(args.output)
    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
    dest = out / ("distance-sweep-" + stamp)
    dest.mkdir()
    env = AuditShot(
        sim, arms, ball, camera, contact, copy.deepcopy(cfg), poses, args, atomic
    )
    policies = {}
    checkpoints = {}
    for path in args.sweep_policy:
        p = Path(path)
        loaded = torch.load(p, weights_only=True, map_location=sim.device)
        if loaded["contract"]["fingerprint"] != contract(cfg)["fingerprint"]:
            raise ValueError("Incompatible selected checkpoint")
        label = "ppo_" + str(loaded["contract"]["run_train_layouts"])
        policy = Policy().to(sim.device)
        policy.load_state_dict(loaded["state_dict"])
        policy.eval()
        policies[label] = policy
        checkpoints[label] = {
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "selected_episode": loaded["episodes"],
        }
    if len(policies) != 2:
        raise ValueError("Require both distinct policies")
    rows = []
    start = time.monotonic()
    planned = 5 * (2 * 2 + 1 + len(TILTS) * len(GAPS) + 3)  # 215 shots
    spec = {
        "schema": "sircus-ik-distance-diagnostic-v1",
        "study": dest.name,
        "physics_contract": contract(cfg)["fingerprint"],
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(__file__).parent.glob("*.py")
        },
        "checkpoints": checkpoints,
        "layouts": layouts(),
        "grid": {"tilt_deg": TILTS, "heading_deg": [0.0], "gap_m": GAPS},
        "selection": "Per-distance grid search by capture then reward; 3 fresh repeats. Diagnostic search, not generalization or policy inference.",
        "scope": "Fixed board center; bucket moves only along negative Y. Endpoints only; no collision-safe paths or physical validation.",
    }
    atomic(dest / "contract.json", spec)

    def status(state):
        summaries = []
        for scene in layouts():
            group = [r for r in rows if r["scene_id"] == scene["id"]]
            grid = [r for r in group if r["kind"] == "grid"]
            selected = (
                max(grid, key=lambda r: (r["success"], r["reward"])) if grid else None
            )
            summaries.append(
                {
                    "distance_m": scene["distance_m"],
                    "policies": [r for r in group if r["kind"] in ("policy", "fixed")],
                    "grid_tested": len(grid),
                    "grid_captures": sum(r["success"] for r in grid),
                    "grid_invalid": sum(bool(r.get("invalid")) for r in grid),
                    "grid_best_action": selected["action"] if selected else None,
                    "repeats": [r for r in group if r["kind"] == "repeat"],
                }
            )
        audit = {}
        if args.ik_audit and Path(args.ik_audit).exists():
            raw = json.loads(Path(args.ik_audit).read_text())
            audit = {
                k: raw[k]
                for k in [
                    "state",
                    "completed",
                    "planned",
                    "stage_counts",
                    "historical_failures",
                    "scope",
                ]
            }
            audit["max_rescued_position_error_mm"] = max(
                [
                    r["solve"]["position_error_mm"]
                    for r in raw["cases"]
                    if r["historical_failure"] and r["solve"]["solved"]
                ]
                or [0]
            )
            audit["max_rescued_normal_error_deg"] = max(
                [
                    r["solve"]["normal_error_deg"]
                    for r in raw["cases"]
                    if r["historical_failure"] and r["solve"]["solved"]
                ]
                or [0]
            )
        value = {
            "schema": spec["schema"],
            "study": dest.name,
            "timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "state": state,
            "completed": len(rows),
            "planned": planned,
            "elapsed_s": time.monotonic() - start,
            "ik_audit": audit,
            "distances": summaries,
            "scope": spec["scope"],
            "selection": spec["selection"],
        }
        atomic(dest / "summary.json", value)
        atomic(out / "ik-distance.json", value)

    def shot(scene, action, kind, label, extended):
        env.use_extended = extended
        env.ik_evidence = []
        r = env.scene_shot(scene, action)
        r.update(
            kind=kind,
            label=label,
            solver="extended" if extended else "original",
            ik_stage=env.ik_evidence[-1]["stage"] if env.ik_evidence else None,
        )
        if not r.get("invalid"):
            # Direct settled physical board geometry, distinct from target FK.
            arm = arms["Arm14"]
            idx = arm.body_names.index("impact_board")
            r["final_board_center_m"] = arm.data.body_pos_w[0, idx].cpu().tolist()
        rows.append(r)
        with (dest / "shots.jsonl").open("a") as f:
            f.write(json.dumps(r) + "\n")
        status("running")
        print(
            "DISTANCE_SHOT",
            len(rows),
            planned,
            scene["distance_m"],
            label,
            r["success"],
            flush=True,
        )
        return r

    status("running")
    for scene in layouts():
        for label, policy in policies.items():
            with torch.no_grad():
                action = action_values(
                    policy.actor(
                        torch.tensor(observation(scene, cfg), device=sim.device)
                    )
                    .cpu()
                    .numpy()
                )
            for extended in [False, True]:
                shot(scene, action, "policy", label, extended)
        shot(
            scene,
            np.array(contract(cfg)["warm_start_action"]),
            "fixed",
            "warm_fixed",
            True,
        )
        grid = []
        for tilt, gap in itertools.product(TILTS, GAPS):
            grid.append(shot(scene, np.array([tilt, 0.0, gap]), "grid", "grid", True))
        best = max(grid, key=lambda r: (r["success"], r["reward"]))
        for _ in range(3):
            shot(
                scene, np.array(best["action"]), "repeat", "selected_grid_repeat", True
            )
    status("completed")
    atomic(
        out / "result.json",
        {"passed": True, "state": "distance_diagnostic_complete", "study": dest.name},
    )
