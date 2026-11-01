"""Bounded PPO fine-tuning with measured release gates and frozen paired comparison."""

import copy
import datetime
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from generalized_rl import Policy
from learning_contract import contract as legacy_contract
from learning_contract import observation, outcome_summary
from pilot_contract import balanced_indices, contract, layouts
from rl_contract import action_values
from settled_shot import SettledShot


def run(sim, arms, ball, camera, contact, cfg, poses, args, atomic):
    torch.set_num_threads(2)
    torch.manual_seed(43)
    rng = np.random.default_rng(43)
    out = Path(args.output)
    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
    dest = out / (("pilot-smoke-" if args.pilot_smoke else "balanced-pilot-") + stamp)
    dest.mkdir()
    spec = contract(cfg, args.pilot_episodes)
    spec["source_sha256"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in Path(__file__).parent.glob("*.py")
    }
    references, provenance = {}, {}
    for path in args.reference_policy:
        path = Path(path)
        saved = torch.load(path, weights_only=True, map_location=sim.device)
        if saved["contract"]["fingerprint"] != legacy_contract(cfg)["fingerprint"]:
            raise ValueError(
                "Reference checkpoint is not the supported legacy contract"
            )
        label = "ppo_" + str(saved["contract"]["run_train_layouts"])
        net = Policy().to(sim.device)
        net.load_state_dict(saved["state_dict"])
        net.eval()
        references[label] = net
        provenance[label] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "selected_episode": saved["episodes"],
        }
    if set(references) != {"ppo_100", "ppo_1000"}:
        raise ValueError("Both historical selected policies are required")
    spec["references"] = provenance
    spec["gpu"] = torch.cuda.get_device_name(0)
    atomic(dest / "contract.json", spec)
    env = SettledShot(
        sim, arms, ball, camera, contact, copy.deepcopy(cfg), poses, args, atomic
    )
    curves, training, evaluations = [], [], {}
    started = time.monotonic()
    probes = []

    def publish(phase, **extra):
        report = {
            "run": dest.name,
            "phase": phase,
            "timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "episodes": len(training),
            "budget": args.pilot_episodes,
            "elapsed_s": time.monotonic() - started,
            "learning_curve": curves,
            "evaluation": evaluations,
            "pose_verification": pose_summary(probes),
            "scope": spec["scope"],
            "contract_fingerprint": spec["fingerprint"],
            **extra,
        }
        atomic(dest / "progress.json", report)
        atomic(out / "balanced-pilot.json", report)
        return report

    def shot(scene, action):
        if shutil.disk_usage(dest).free < 3 * 1024**3:
            raise RuntimeError("Free space below3GiB")
        return env.scene_shot(scene, action)

    publish("pose_verification")
    for scene in layouts("development", 4):
        r = shot(scene, np.array([24.0, 0.0, 0.15]))
        probes.append(r)
        atomic(dest / "pose-probes.json", probes)
        publish("pose_verification")
        print("POSE_PROBE", len(probes), r.get("invalid"), flush=True)
    # Unresolved endpoints must fail closed, but are not a settled-pose failure:
    # no release took place. Keep them in reports and require coverage in each bin.
    valid_counts = {
        name: sum(r["difficulty"] == name and not r.get("invalid") for r in probes)
        for name in ["near", "medium", "far"]
    }
    if (
        any(r.get("invalid") not in (None, "endpoint_ik") for r in probes)
        or min(valid_counts.values()) < 3
    ):
        publish("blocked_pose_verification")
        raise RuntimeError("Development pose gate failed; no training started")
    if args.pilot_smoke:
        publish("smoke_completed")
        return
    pools = {
        k: layouts(k, n) for k, n in [("train", 32), ("validation", 6), ("test", 12)]
    }
    atomic(dest / "splits.json", pools)
    policy = copy.deepcopy(references["ppo_100"])
    with torch.no_grad():
        policy.logstd.fill_(-0.7)
    initial = copy.deepcopy(policy.state_dict())
    opt = torch.optim.Adam(policy.parameters(), lr=0.0007)
    best_score, best_episode, best_state = None, 0, None

    def action_for(net, scene):
        with torch.no_grad():
            return action_values(
                net.actor(torch.tensor(observation(scene, cfg), device=sim.device))
                .cpu()
                .numpy()
            )

    def evaluate(net, pool, label):
        rows = []
        for scene in pool:
            a = (
                np.array(legacy_contract(cfg)["warm_start_action"])
                if net is None
                else action_for(net, scene)
            )
            rows.append(shot(scene, a))
        atomic(dest / (label + ".json"), rows)
        return rows

    def validate():
        nonlocal best_score, best_episode, best_state
        publish("validation")
        rows = evaluate(policy, pools["validation"], f"validation-{len(training)}")
        c = {"episodes": len(training), **outcome_summary(rows)}
        curves.append(c)
        score = (c["captures"], c["mean_reward"])
        if best_score is None or score > best_score:
            best_score, best_episode = score, len(training)
            best_state = copy.deepcopy(policy.state_dict())
        publish("training", selected_episode=best_episode)

    validate()
    while len(training) < args.pilot_episodes:
        states, latents, oldlogs, rewards = [], [], [], []
        for index in balanced_indices(pools["train"], 16, rng):
            scene = pools["train"][index]
            x = torch.tensor(observation(scene, cfg), device=sim.device)
            with torch.no_grad():
                distribution = policy.distribution(x)
                z = distribution.sample()
                oldlog = distribution.log_prob(z).sum()
            r = shot(scene, action_values(z.cpu().numpy()))
            training.append(r)
            with (dest / "episodes.jsonl").open("a") as f:
                f.write(json.dumps({"episode": len(training), **r}) + "\n")
            states.append(x)
            latents.append(z)
            oldlogs.append(oldlog)
            rewards.append(r["reward"] / 15)
            if len(training) % 8 == 0:
                publish(
                    "training",
                    recent=outcome_summary(training[-96:]),
                    selected_episode=best_episode,
                )
        x, z, old = torch.stack(states), torch.stack(latents), torch.stack(oldlogs)
        ret = torch.tensor(rewards, device=sim.device)
        with torch.no_grad():
            adv = ret - policy.critic(x).squeeze(-1)
            adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-6)
        for _ in range(4):
            distribution = policy.distribution(x)
            ratio = (distribution.log_prob(z).sum(-1) - old).exp()
            actor = -torch.min(ratio * adv, ratio.clamp(0.8, 1.2) * adv).mean()
            critic = ((policy.critic(x).squeeze(-1) - ret) ** 2).mean()
            loss = actor + 0.5 * critic - 0.001 * distribution.entropy().sum(-1).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            with torch.no_grad():
                policy.logstd.clamp_(-2.8, -0.2)
        torch.save(
            {
                "state_dict": policy.state_dict(),
                "contract": spec,
                "episodes": len(training),
            },
            dest / "latest.pt",
        )
        print(
            "PILOT_UPDATE", len(training), outcome_summary(training[-48:]), flush=True
        )
        if len(training) % 192 == 0 or len(training) == args.pilot_episodes:
            validate()
    trained_change = float(
        sum(
            (v - initial[k]).abs().sum()
            for k, v in policy.state_dict().items()
            if k.startswith("actor.")
        )
    )
    policy.load_state_dict(best_state)
    torch.save(
        {
            "state_dict": best_state,
            "contract": spec,
            "episodes": best_episode,
            "training_budget_used": len(training),
        },
        dest / "selected.pt",
    )
    publish("frozen_evaluation", selected_episode=best_episode)
    for label, net in [
        ("pilot_selected", policy),
        *references.items(),
        ("warm_fixed", None),
    ]:
        rows = evaluate(net, pools["test"], "test-" + label)
        stats = outcome_summary(rows)
        actions = np.array([r["action"] for r in rows])
        stats["action_min"] = actions.min(axis=0).tolist()
        stats["action_max"] = actions.max(axis=0).tolist()
        stats["settled_pose"] = pose_summary(rows)
        evaluations[label] = stats
        publish("frozen_evaluation", selected_episode=best_episode)
    report = publish(
        "completed",
        selected_episode=best_episode,
        actor_training_change_l1=trained_change,
        actor_selected_change_l1=float(
            sum(
                (v - initial[k]).abs().sum()
                for k, v in best_state.items()
                if k.startswith("actor.")
            )
        ),
        unique_training_layouts=len({r["scene_id"] for r in training}),
        train_by_distance=outcome_summary(training)["by_distance"],
        selected_sha256=hashlib.sha256((dest / "selected.pt").read_bytes()).hexdigest(),
    )
    atomic(dest / "summary.json", report)
    atomic(
        out / "result.json",
        {"passed": True, "state": "balanced_pilot_complete", "run": dest.name},
    )
    print("PILOT_COMPLETE", json.dumps(report), flush=True)


def pose_summary(rows):
    records = [r["settled_board"] for r in rows if r.get("settled_board")]
    if not records:
        return {"trials": len(rows), "measured": 0}
    return {
        "trials": len(rows),
        "measured": len(records),
        "invalid": sum(bool(r.get("invalid")) for r in rows),
        "initial_max_position_mm": max(r[0]["position_error_mm"] for r in records),
        "initial_max_normal_deg": max(r[0]["normal_error_deg"] for r in records),
        "final_max_position_mm": max(r[-1]["position_error_mm"] for r in records),
        "final_max_normal_deg": max(r[-1]["normal_error_deg"] for r in records),
        "final_max_joint_speed_rad_s": max(r[-1]["joint_speed_rad_s"] for r in records),
        "max_corrections": max(len(r) - 1 for r in records),
        "final_max_position_drift_mm": max(r[-1]["position_drift_mm"] for r in records),
        "final_max_normal_drift_deg": max(r[-1]["normal_drift_deg"] for r in records),
    }
