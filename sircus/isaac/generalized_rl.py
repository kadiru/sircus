"""Actual PhysX trials with a shared scene-conditioned one-step PPO policy."""

import copy
import datetime
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from fixed_rl import FixedShot
from learning_contract import contract, observation, outcome_summary, scenes
from prepare_assets import fk
from rl_contract import HIGH, LOW, action_values
from scipy.optimize import least_squares


class Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(11, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 3),
        )
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(11, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 1),
        )
        self.logstd = torch.nn.Parameter(torch.full((3,), -1.0))
        torch.nn.init.zeros_(self.actor[-1].weight)
        initial = np.arctanh(
            2 * (np.array([21.3590135, -0.9408402, 0.16769218]) - LOW) / (HIGH - LOW)
            - 1
        )
        with torch.no_grad():
            self.actor[-1].bias.copy_(torch.tensor(initial, dtype=torch.float32))

    def distribution(self, x):
        return torch.distributions.Normal(self.actor(x), self.logstd.exp())


class SceneShot(FixedShot):
    def __init__(self, *args):
        super().__init__(*args)
        self.preview = False
        self.bucket = self.sim._sircus_bucket

    def tick(self, render=False):
        super().tick(render=render and self.preview)

    def ik(self, target, name):
        if name != "Arm14":
            try:
                return super().ik(target, name)
            except ValueError:
                self.ik_failure_component = name
                raise

        # Board spin about its normal is unconstrained. No reward-based search.
        def residual(q):
            cur = fk(q, self.links)
            return np.r_[
                cur[:3, 3] - target[:3, 3], 0.25 * (cur[:3, 2] - target[:3, 2])
            ]

        initial = np.array(self.poses["board_joint_rad"])
        for seed in [
            initial,
            np.clip(
                initial + [0.1, -0.1, 0.1, 0, 0, 0], self.lo + 1e-4, self.hi - 1e-4
            ),
        ]:
            fit = least_squares(
                residual,
                seed,
                bounds=(self.lo, self.hi),
                max_nfev=80,
                ftol=1e-8,
                gtol=1e-8,
                xtol=1e-8,
            )
            if np.linalg.norm(fit.fun) < 1e-4:
                return fit.x
        self.ik_failure_component = "Arm14"
        raise ValueError("board endpoint unreachable")

    def place_bucket(self, xy):
        state = self.bucket.data.default_root_state.clone()
        state[0, :3] = torch.tensor([*xy, 0.0], device=self.sim.device)
        state[0, 7:] = 0
        self.bucket.write_root_state_to_sim(state)
        self.bucket.reset()

    def scene_shot(self, scene, action, preview=False):
        self.cfg["board"]["initial_center_m"] = scene["board_center_m"]
        self.cfg["bucket"]["center_xy_m"] = scene["bucket_xy_m"]
        self.place_bucket(scene["bucket_xy_m"])
        self.preview = preview
        self.ik_failure_component = None
        started = time.monotonic()
        r = self.shot(np.array(action))
        if r.get("invalid"):
            r["invalid_component"] = self.ik_failure_component
        r.update(
            scene_id=scene["id"],
            difficulty=scene["difficulty"],
            distance_m=scene["distance_m"],
            wall_s=time.monotonic() - started,
        )
        # Compact training data: trajectories only for selected diagnostic shots.
        if not preview:
            r.pop("trajectory_m", None)
        return r


def reset_audit(env, atomic, run):
    """Development-only checks; no selection using final test layouts."""
    probes = []
    for xy in ([0.47, -0.52], [0.81, -0.32], [0.60, -0.46]):
        env.place_bucket(xy)
        state = env.ball.data.default_root_state.clone()
        state[0, :3] = torch.tensor([*xy, 0.10], device=env.sim.device)
        state[0, 7:] = 0
        env.ball.write_root_state_to_sim(state)
        env.ball.reset()
        for _ in range(360):
            env.tick()
        pos = env.ball.data.root_pos_w[0].cpu().numpy()
        expected = (
            env.cfg["bucket"]["assumed_bottom_thickness_m"]
            + env.cfg["ball"]["radius_m"]
        )
        passed = bool(
            np.linalg.norm(pos[:2] - xy) < 0.005 and abs(pos[2] - expected) < 0.003
        )
        probes.append(
            {"bucket_xy_m": xy, "ball_final_m": pos.tolist(), "passed": passed}
        )
    atomic(run / "bucket-pose-probes.json", probes)
    if not all(p["passed"] for p in probes):
        raise RuntimeError("Moved bucket physical floor probe failed")
    dev = scenes("development", 80)[64:72]
    repeats = [
        [env.scene_shot(s, [21.3590135, -0.9408402, 0.16769218]) for s in dev]
        for _ in range(2)
    ]
    atomic(
        run / "repeatability.json",
        {
            "scope": "development-only repeated identical inputs; contact outcomes may be sensitive",
            "passes": repeats,
            "disagreements": sum(
                a["success"] != b["success"] for a, b in zip(*repeats)
            ),
        },
    )


def train_scene(sim, arms, ball, camera, contact, cfg, poses, args, atomic):
    torch.set_num_threads(2)
    torch.manual_seed(args.rl_seed)
    rng = np.random.default_rng(args.rl_seed)
    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.output)
    run = out / f"general-rl-{stamp}-n{args.train_layouts}-seed{args.rl_seed}"
    run.mkdir()
    spec = contract(cfg)
    spec["source_sha256"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(Path(__file__).parent.glob("*.py"))
    }
    spec["gpu"] = torch.cuda.get_device_name(0)
    spec["run_seed"] = args.rl_seed
    spec["run_train_layouts"] = args.train_layouts
    spec["run_episodes"] = args.general_episodes
    atomic(run / "contract.json", spec)
    pools = {
        k: scenes(k, n)
        for k, n in [("train", 1000), ("validation", 128), ("test", 256)]
    }
    if args.study_smoke:
        development = scenes("development", 128)
        pools = {
            "train": development[:64],
            "validation": development[64:72],
            "test": development[72:80],
        }
        spec["scope"] = (
            "development smoke only; final train/validation/test pools untouched"
        )
        atomic(run / "contract.json", spec)
    atomic(run / "splits.json", pools)
    env = SceneShot(
        sim, arms, ball, camera, contact, copy.deepcopy(cfg), poses, args, atomic
    )
    if args.study_smoke:
        reset_audit(env, atomic, run)
    policy = Policy().to(sim.device)
    opt = torch.optim.Adam(policy.parameters(), lr=spec["lr"])
    initial_state = copy.deepcopy(policy.state_dict())
    best = None
    curves = []
    all_rows = []
    start = time.monotonic()

    def save(name):
        torch.save(
            {
                "state_dict": policy.state_dict(),
                "contract": spec,
                "episodes": len(all_rows),
            },
            run / (name + ".tmp"),
        )
        (run / (name + ".tmp")).replace(run / name)

    def status(phase, **extra):
        v = {
            "timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "run": run.name,
            "phase": phase,
            "episodes": len(all_rows),
            "budget": args.general_episodes,
            "train_layouts": args.train_layouts,
            "seed": args.rl_seed,
            "scope": spec.get("scope", "privileged state study"),
            "elapsed_s": time.monotonic() - start,
            "learning_curve": curves,
            **extra,
        }
        atomic(run / "progress.json", v)
        atomic(out / "learning-study.json", v)

    def evaluate(pool, mode="policy"):
        result = []
        for scene in pool:
            if mode == "midpoint":
                action = action_values(np.zeros(3))
            elif mode == "warm_fixed":
                action = np.array(spec["warm_start_action"])
            else:
                with torch.no_grad():
                    action = action_values(
                        policy.actor(
                            torch.tensor(observation(scene, cfg), device=sim.device)
                        )
                        .cpu()
                        .numpy()
                    )
            result.append(env.scene_shot(scene, action))
        return result

    env.phase = "generalized_baseline"
    status("baseline")
    baseline = evaluate(pools["validation"][:32] * 2, "policy")
    atomic(run / "baseline-validation.json", baseline)
    curves.append({"episodes": 0, **outcome_summary(baseline)})
    save("initial.pt")
    save("best.pt")
    best = (
        sum(r["success"] for r in baseline),
        np.mean([r["reward"] for r in baseline]),
    )
    best_episode = 0
    pool = pools["train"][: args.train_layouts]
    order = np.arange(len(pool))
    rng.shuffle(order)
    index = 0
    while len(all_rows) < args.general_episodes:
        if shutil.disk_usage(run).free < 3 * 1024**3:
            raise RuntimeError("Storage below3GiB")
        env.phase = "scene_conditioned_ppo"
        states = []
        zs = []
        oldlogs = []
        rewards = []
        batch = min(64, args.general_episodes - len(all_rows))
        for _ in range(batch):
            if index == len(order):
                rng.shuffle(order)
                index = 0
            scene = pool[int(order[index])]
            index += 1
            x = torch.tensor(observation(scene, cfg), device=sim.device)
            with torch.no_grad():
                dist = policy.distribution(x)
                z = dist.sample()
                lp = dist.log_prob(z).sum()
            result = env.scene_shot(
                scene,
                action_values(z.cpu().numpy()),
                preview=(len(all_rows) % 512 == 0 and args.study_preview),
            )
            states.append(x)
            zs.append(z)
            oldlogs.append(lp)
            rewards.append(result["reward"] / 15)
            all_rows.append(result)
            if len(all_rows) % 8 == 0:
                status("training", recent=outcome_summary(all_rows[-128:]))
            with (run / "episodes.jsonl").open("a") as f:
                f.write(json.dumps({"episode": len(all_rows), **result}) + "\n")
        x = torch.stack(states)
        z = torch.stack(zs)
        old = torch.stack(oldlogs)
        ret = torch.tensor(rewards, device=sim.device)
        with torch.no_grad():
            adv = ret - policy.critic(x).squeeze(-1)
            adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-6)
        for _ in range(4):
            dist = policy.distribution(x)
            ratio = (dist.log_prob(z).sum(-1) - old).exp()
            actor = -torch.min(ratio * adv, ratio.clamp(0.8, 1.2) * adv).mean()
            critic = ((policy.critic(x).squeeze(-1) - ret) ** 2).mean()
            loss = actor + 0.5 * critic - 0.001 * dist.entropy().sum(-1).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            with torch.no_grad():
                policy.logstd.clamp_(-2.8, -0.2)
        save("latest.pt")
        status("training", recent=outcome_summary(all_rows[-128:]))
        print(
            "GENERAL_UPDATE",
            len(all_rows),
            outcome_summary(all_rows[-128:]),
            flush=True,
        )
        if len(all_rows) % 512 == 0 or len(all_rows) == args.general_episodes:
            env.phase = "validation"
            status("validation")
            val = evaluate(pools["validation"][:32] * 2)
            score = (
                sum(r["success"] for r in val),
                np.mean([r["reward"] for r in val]),
            )
            curves.append({"episodes": len(all_rows), **outcome_summary(val)})
            atomic(run / f"validation-{len(all_rows)}.json", val)
            if score > best:
                best = score
                best_episode = len(all_rows)
                save("best.pt")
            status("training", best_episode=best_episode)
    loaded = torch.load(run / "best.pt", weights_only=True, map_location=sim.device)
    assert loaded["contract"]["fingerprint"] == spec["fingerprint"]
    policy.load_state_dict(loaded["state_dict"])
    torch.save(
        {
            "state_dict": policy.state_dict(),
            "contract": spec,
            "episodes": best_episode,
            "training_budget_used": len(all_rows),
        },
        run / "selected.pt",
    )
    env.phase = "frozen_heldout"
    status("frozen_evaluation", best_episode=best_episode)
    summaries = {}
    for label, pool, mode in [
        ("validation", pools["validation"], "policy"),
        ("test", pools["test"], "policy"),
        ("test_warm_fixed", pools["test"], "warm_fixed"),
        ("test_midpoint", pools["test"], "midpoint"),
    ]:
        rows = evaluate(pool, mode)
        atomic(run / f"{label}.json", rows)
        summaries[label] = outcome_summary(rows)
        status("frozen_evaluation", completed=summaries, best_episode=best_episode)
    changed = float(
        sum((policy.state_dict()[k] - v).abs().sum() for k, v in initial_state.items())
    )
    summary = {
        "run": run.name,
        "contract": spec,
        "training_episodes": len(all_rows),
        "unique_training_layouts": len({r["scene_id"] for r in all_rows}),
        "selected_episode": best_episode,
        "parameter_change_l1": changed,
        "actor_parameter_change_l1": float(
            sum(
                (policy.state_dict()[k] - v).abs().sum()
                for k, v in initial_state.items()
                if k.startswith("actor.")
            )
        ),
        "learning_curve": curves,
        "evaluation": summaries,
        "elapsed_s": time.monotonic() - start,
        "scope": "Privileged geometry; endpoint-initialized cooperative shots. No grasp/path/physical validation.",
    }
    atomic(run / "summary.json", summary)
    status("completed", summary=summary)
    atomic(
        out / "result.json",
        {"passed": True, "state": "general_learning_complete", "run": run.name},
    )
    print("GENERAL_COMPLETE", json.dumps(summary), flush=True)
