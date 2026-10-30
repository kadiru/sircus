"""Viam-invoked fixed-board release search with durable train/test separation.

This learns a release action for one scene. It is not an image-conditioned or
arbitrary-board-pose policy. Native Viam path validation is a separate replay.
"""
import asyncio
import copy
import inspect
import random
from datetime import datetime, timezone

from physics import simulate, validate_config
from trial_store import TrialStore, fingerprint

POLICY_SCHEMA = "sircus-two-arm-release-v1"
SELECTION_SCOPE = "one fixed board pose; synthetic compliant contact; endpoint IK checks; no native path checks"


async def _call(function, *args):
    value = function(*args)
    return await value if inspect.isawaitable(value) else value


async def _feasibility_call(function, *args):
    # IK may take seconds; keep Viam status/stop RPCs responsive while it runs.
    if inspect.iscoroutinefunction(function):
        return await function(*args)
    value = await asyncio.to_thread(function, *args)
    return await value if inspect.isawaitable(value) else value


def _stopped(stop_event):
    return stop_event is not None and stop_event.is_set()


def _check_stop(stop_event):
    if _stopped(stop_event):
        raise asyncio.CancelledError("Training stopped; completed evidence remains saved")


def _action(rng, i, budget, limits, best):
    u_limit, v_limit = limits
    initial = [(0, 0, 180), (0, -65, 120), (0, -65, 220), (0, 50, 140),
               (-65, 0, 180), (65, 0, 180), (0, -30, 300), (0, 30, 80)]
    if i < len(initial):
        u, v, g = initial[i]
        return dict(u_mm=max(-u_limit, min(u_limit, u)),
                    v_mm=max(-v_limit, min(v_limit, v)), gap_mm=g)
    if i < budget // 2 or best is None:
        return dict(u_mm=rng.uniform(-u_limit, u_limit),
                    v_mm=rng.uniform(-v_limit, v_limit), gap_mm=rng.uniform(60, 380))
    scale = max(.12, 1 - i / budget)
    a = {k: best["action"][k] + rng.gauss(0, width * scale)
         for k, width in (("u_mm", 120), ("v_mm", 120), ("gap_mm", 180))}
    return dict(u_mm=max(-u_limit, min(u_limit, a["u_mm"])),
                v_mm=max(-v_limit, min(v_limit, a["v_mm"])),
                gap_mm=max(40, min(450, a["gap_mm"])))


async def train(config, seed=42, budget=24, feasibility=None, on_progress=None,
                stop_event=None, store=None):
    config = copy.deepcopy(config)
    validate_config(config)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**30:
        raise ValueError("seed must be an integer from0 to2^30-1")
    if isinstance(budget, bool) or not isinstance(budget, int) or not 8 <= budget <= 80:
        raise ValueError("budget must be8..80")
    if feasibility is None:
        raise ValueError("Two-arm feasibility callback is required")
    store = store or TrialStore(config, metadata={"seed": seed, "budget": budget})
    if store.fingerprint != fingerprint(config):
        raise ValueError("Trial store scene does not match training scene")
    rng = random.Random(seed)
    limits = (config["board"]["size_mm"][0] * .4, config["board"]["size_mm"][1] * .4)
    history, test_rows = [], []
    best, policy = None, None

    async def notify(phase, trial, total, row=None):
        if on_progress is not None:
            await _call(on_progress, {"phase": phase, "trial": trial, "budget": total,
                         "best_reward": None if best is None else best["reward"],
                         "last_outcome": None if row is None else row["outcome"],
                         "current": row, "best": best, "session_id": store.session_id})

    async def evaluate(scene, action, trial_seed, phase, number):
        _check_stop(stop_event)
        check, stage = {}, "feasibility"
        try:
            check = await _feasibility_call(feasibility, scene, action)
            if not isinstance(check, dict):
                check = {}
                raise TypeError("feasibility must return a dictionary")
            _check_stop(stop_event)
            reachable = check.get("arm_reachable") is True and check.get("board_arm_reachable") is True
            if reachable:
                stage = "physics"
                result = await asyncio.to_thread(simulate, scene, action, seed=trial_seed, frames=False)
            else:
                result = {"reward": -1000.0, "retained": False,
                          "outcome": "unreachable_arm_target", "physically_verified": False,
                          "feasibility_reason": check.get("reason", check.get("error"))}
        except BaseException as error:
            # An interrupted attempt is evidence, not a negative learning sample.
            # to_thread cannot stop its worker; any late result is deliberately ignored.
            interrupted = {"reward": None, "retained": False,
                           "outcome": "evaluation_cancelled" if isinstance(error, asyncio.CancelledError)
                                      else "evaluation_failed",
                           "evaluation_stage": stage, "valid_for_training": False,
                           "error_type": type(error).__name__, "error": str(error),
                           "physically_verified": False}
            store.record_trial(phase, number, action, trial_seed, interrupted, check, scene_config=scene)
            raise
        result["valid_for_training"] = not bool(result.get("model_limit"))
        record = store.record_trial(phase, number, action, trial_seed, result, check, scene_config=scene)
        row = {"trial": number, "phase": phase, "action": action, "seed": trial_seed,
               "reward": float(result["reward"]), "retained": bool(result["retained"]),
               "outcome": result["outcome"], "arm_reachable": check.get("arm_reachable", False),
               "board_arm_reachable": check.get("board_arm_reachable", False),
               "impact_uv_mm": result.get("impact_uv_mm"), "landing_xy_mm": result.get("landing_xy_mm"),
               "impact_region": record["impact_region"], "record_id": record["record_id"],
               "valid_for_training": result["valid_for_training"]}
        return row, reachable and result["valid_for_training"]

    try:
        await notify("training", 0, budget)
        for i in range(budget):
            action = _action(rng, i, budget, limits, best)
            row, reachable = await evaluate(config, action, seed * 3, "train", i + 1)
            history.append(row)
            if reachable and (best is None or row["reward"] > best["reward"]):
                best = row
            await notify("training", i + 1, budget, row)
        if best is None:
            raise ValueError("No valid release with both arms reachable was found")
        _check_stop(stop_event)
        policy = {"schema": POLICY_SCHEMA, "config_fingerprint": fingerprint(config),
                  "best_action": copy.deepcopy(best["action"]), "train_seed": seed, "budget": budget,
                  "frozen_utc": datetime.now(timezone.utc).isoformat(), "physically_verified": False,
                  "selection_scope": SELECTION_SCOPE, "board_pose": copy.deepcopy(config["board"]),
                  "test_spec": {"trials": 10, "seed": seed * 3 + 1,
                                "release_xy_mm": 2, "release_gap_mm": 3}}
        store.save_policy(policy)  # Immutable and persisted before any held-out evaluation.
        testing = copy.deepcopy(config)
        testing["simulation"]["noise_xy_mm"] = 2.0
        testing["simulation"]["noise_gap_mm"] = 3.0
        for j in range(10):
            row, _ = await evaluate(testing, best["action"], seed * 3 + 1 + j * 104729, "test", j + 1)
            test_rows.append(row)
            await notify("testing", j + 1, 10, row)
        _check_stop(stop_event)
        nominal = await asyncio.to_thread(simulate, config, best["action"], seed=seed * 3, frames=True)
        output = {**policy, "policy": policy, "history": history, "best_result": nominal,
                  "test": {"retained_count": sum(row["retained"] for row in test_rows),
                           "trials": len(test_rows), "results": test_rows,
                           "scope": "synthetic release noise; fixed board pose; no physical or cross-pose validation"},
                  "session_id": store.session_id}
        store.save_summary({k: value for k, value in output.items() if k != "best_result"})
        output["output_paths"] = store.status()
        await notify("complete", budget, budget, best)
        return output
    except BaseException as error:
        summary = {"status": "cancelled" if isinstance(error, asyncio.CancelledError) else "failed",
                   "error_type": type(error).__name__, "message": str(error),
                   "completed_train_trials": len(history), "completed_test_trials": len(test_rows),
                   "policy_frozen": policy is not None, "history": history, "test_results": test_rows}
        try:
            store.save_summary(summary)
        except (FileExistsError, OSError):
            pass
        raise
