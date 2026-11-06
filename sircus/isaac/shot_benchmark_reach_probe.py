"""Bounded CPU-only re-solve of saved TRAIN endpoint IK requests.

This diagnostic never imports Isaac, Torch, a learner or a native launcher. It
preserves the original endpoint objectives, joint limits and <1e-4 residual
gate. A recovery establishes an endpoint solution only; a failed/time-limited
search never proves impossibility and never changes original trial labels.
"""

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from scipy.stats import qmc

from expanded_pose_native import verified_fit
from prepare_assets import chain, fk
from robust_ik import TOLERANCE, normal_jacobian

VERSION = "sircus-offline-endpoint-reprobe-v1"
MAX_CASES = 10
MAX_STARTS = 10
MAX_NFEV = 1000
MAX_WALL_S = 180.
MAX_RESIDUAL_EVALUATIONS = 100000
NEAR_RESIDUAL = 1e-3


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            value.update(block)
    return value.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read(path):
    def reject(value):
        raise ValueError("Nonfinite JSON: " + value)
    return json.loads(Path(path).read_bytes(), parse_constant=reject)


def validate_case(raw):
    """Normalize an explicit saved target; no context/pose is reconstructed here."""
    require(isinstance(raw, dict), "Expected saved probe case")
    result = copy.deepcopy(raw)
    require(isinstance(result.get("case_id"), str) and result["case_id"], "Missing case identity")
    require(result.get("split") == "train", "Offline probe accepts TRAIN cases only")
    require(result.get("arm") in ("Arm10", "Arm14"), "Unknown probe arm")
    target = result.get("target_arm_base", result.get("target_arm_base_4x4", result.get("target_arm_base_m")))
    target = np.asarray(target, dtype=float)
    require(target.shape == (4, 4) and np.isfinite(target).all()
            and np.allclose(target[3], [0, 0, 0, 1], atol=1e-10, rtol=0)
            and np.allclose(target[:3, :3].T @ target[:3, :3], np.eye(3), atol=1e-8, rtol=0)
            and abs(np.linalg.det(target[:3, :3])-1.) < 1e-8,
            "Expected rigid 4x4 target in the selected arm's base frame")
    fit = result.get("saved_fit", result.get("fit"))
    require(isinstance(fit, dict) and type(fit.get("solved")) is bool, "Missing saved IK fit")
    q = np.asarray(fit.get("q"), dtype=float)
    require(q.shape == (6,) and np.isfinite(q).all(), "Expected saved six-joint solution")
    require(type(fit.get("residual")) in (int, float) and math.isfinite(fit["residual"])
            and fit["residual"] >= 0, "Expected finite saved residual")
    require(fit["solved"] == (fit["residual"] < TOLERANCE), "Saved fit residual/acceptance disagree")
    home = result.get("home_q", result.get("home_joint_rad"))
    if home is not None:
        home = np.asarray(home, dtype=float)
        require(home.shape == (6,) and np.isfinite(home).all(), "Invalid saved home joints")
        home = home.tolist()
    require(isinstance(result.get("provenance"), dict) and result["provenance"], "Saved source provenance required")
    result.update(target_arm_base=target.tolist(), saved_fit=copy.deepcopy(fit), home_q=home)
    for alias in ("target_arm_base_4x4", "target_arm_base_m", "fit", "home_joint_rad"):
        result.pop(alias, None)
    return result


def select_cases(payload):
    """Preselect failures by saved arm/residual strata, never by probe outcome."""
    require(isinstance(payload, dict), "Expected audit or explicit case payload")
    # Reach audits also contain visualization `cases`; those are not the
    # separately preselected endpoint probe requests.
    explicit = "probe_candidates" not in payload and "cases" in payload
    rows = payload.get("cases") if explicit else payload.get("probe_candidates")
    require(isinstance(rows, list), "Expected cases or audit probe_candidates array")
    rows = [validate_case(row) for row in rows]
    require(len({row["case_id"] for row in rows}) == len(rows), "Duplicate probe case identity")
    if explicit:
        require(0 < len(rows) <= MAX_CASES, "Explicit probe requires 1..10 cases")
        return rows, dict(mode="explicit_preselected", maximum_cases=MAX_CASES,
                         selection_sha256=canonical_hash(rows))
    selected = []
    for arm in ("Arm10", "Arm14"):
        failures = [row for row in rows if row["arm"] == arm and not row["saved_fit"]["solved"]]
        failures.sort(key=lambda row: row["case_id"])
        chosen = []
        for near in (True, False):
            chosen.extend([row for row in failures if (row["saved_fit"]["residual"] < NEAR_RESIDUAL) == near][:2])
        ids = {row["case_id"] for row in chosen}
        chosen.extend([row for row in failures if row["case_id"] not in ids][:4-len(chosen)])
        selected.extend(chosen)
    for arm in ("Arm10", "Arm14"):
        controls = sorted([row for row in rows if row["arm"] == arm and row["saved_fit"]["solved"]],
                          key=lambda row: row["case_id"])
        selected.extend(controls[:1])
    require(selected, "Audit contains no eligible saved TRAIN cases")
    return selected, dict(mode="saved_arm_residual_stratification", maximum_cases=MAX_CASES,
        failure_goal_per_arm=4, near_residual_threshold=NEAR_RESIDUAL,
        failures_per_arm_stratum=2, solved_controls_per_arm=1,
        order="Stable case_id order within each pre-probe stratum; fill missing failure strata within the same arm",
        candidate_count=len(rows), selection_sha256=canonical_hash(selected))


class BudgetExhausted(Exception):
    pass


class Budget:
    def __init__(self, wall_s=MAX_WALL_S, evaluations=MAX_RESIDUAL_EVALUATIONS, clock=time.monotonic):
        require(type(wall_s) in (int, float) and math.isfinite(wall_s) and 0 < wall_s <= MAX_WALL_S,
                "Wall budget must be >0 and <=180 seconds")
        require(type(evaluations) is int and 0 < evaluations <= MAX_RESIDUAL_EVALUATIONS,
                "Residual evaluation budget must be 1..100000")
        self.wall_s, self.limit, self.clock = float(wall_s), evaluations, clock
        self.started, self.evaluations = clock(), 0

    def check(self):
        if self.clock()-self.started >= self.wall_s:
            raise BudgetExhausted("wall_time_budget")
        if self.evaluations >= self.limit:
            raise BudgetExhausted("residual_evaluation_budget")

    def residual(self):
        self.check()
        self.evaluations += 1


def starts_for(case, links):
    """Saved best, saved home (or declared midpoint), and eight new Sobol starts."""
    lower, upper = np.array([link[2] for link in links]), np.array([link[3] for link in links])
    require(lower.shape == upper.shape == (6,) and np.isfinite([lower, upper]).all()
            and (upper-lower > 2e-6).all(), "Expected finite six-joint limits")
    seed = int(canonical_hash(dict(version=VERSION, arm=case["arm"], target=case["target_arm_base"]))[:8], 16)
    unit = qmc.Sobol(d=6, scramble=True, seed=seed).random_base2(3)
    seeds = [("saved_best", np.asarray(case["saved_fit"]["q"]))]
    seeds.append(("saved_home" if case["home_q"] is not None else "midpoint_home_unavailable",
                  np.asarray(case["home_q"]) if case["home_q"] is not None else (lower+upper)/2))
    seeds.extend(("independent_sobol_"+str(i), lower+1e-6+point*(upper-lower-2e-6))
                 for i, point in enumerate(unit))
    return [(label, np.clip(np.asarray(q, dtype=float), lower+1e-6, upper-1e-6)) for label, q in seeds], seed


def probe_case(case, links, budget, max_nfev=MAX_NFEV):
    require(type(max_nfev) is int and 1 <= max_nfev <= MAX_NFEV, "Per-start nfev must be 1..1000")
    target = np.asarray(case["target_arm_base"])
    lower, upper = np.array([link[2] for link in links]), np.array([link[3] for link in links])
    normal_only = case["arm"] == "Arm14"
    seeds, seed = starts_for(case, links)
    attempts, best, stop = [], None, None
    started, initial_evaluations = budget.clock(), budget.evaluations

    def error(q):
        current = fk(q, links)
        angular = current[:3, 2]-target[:3, 2] if normal_only else Rotation.from_matrix(
            target[:3, :3] @ current[:3, :3].T).as_rotvec()
        return np.r_[current[:3, 3]-target[:3, 3], .25*angular]

    def residual(q):
        budget.residual()
        return error(q)

    for label, start in seeds:
        count_before = budget.evaluations
        try:
            budget.check()
            reduce_spin = normal_only and np.linalg.norm(normal_jacobian(start, links)[:, -1]) < 1e-10
            if reduce_spin:
                unpack = lambda v, spin=start[-1]: np.r_[v, spin]
                def jac(v):
                    budget.check()
                    return normal_jacobian(unpack(v), links)[:, :-1]
                fit = least_squares(lambda v: residual(unpack(v)), start[:-1], bounds=(lower[:-1], upper[:-1]),
                    jac=jac, max_nfev=max_nfev, ftol=1e-12, gtol=1e-12, xtol=1e-12)
                q = unpack(fit.x)
            else:
                def jac(q):
                    budget.check()
                    return normal_jacobian(q, links)
                fit = least_squares(residual, start, bounds=(lower, upper),
                    jac=jac if normal_only else "2-point", max_nfev=max_nfev,
                    ftol=1e-12, gtol=1e-12, xtol=1e-12)
                q = fit.x
            # Recompute FK and limits independently of SciPy success/fun fields.
            actual = error(q)
            norm = float(np.linalg.norm(actual))
            angle = float(np.arccos(np.clip(fk(q, links)[:3, 2] @ target[:3, 2], -1, 1))) if normal_only else float(np.linalg.norm(actual[3:])/.25)
            record = dict(q=np.asarray(q).tolist(), residual=norm, solved=norm < TOLERANCE,
                position_error_mm=float(np.linalg.norm(actual[:3])*1000), orientation_error_deg=math.degrees(angle))
            if record["solved"]:
                verified_fit(target, record, links, normal_only)
            attempt = dict(start=label, nfev=int(fit.nfev), residual_evaluations=budget.evaluations-count_before,
                reduced_null_spin=bool(reduce_spin), **record)
            attempts.append(attempt)
            if best is None or norm < best["residual"]:
                best = record
            if record["solved"]:
                break
        except BudgetExhausted as exc:
            stop = str(exc)
            attempts.append(dict(start=label, status="interrupted", reason=stop,
                residual_evaluations=budget.evaluations-count_before, nfev=None))
            break
    status = ("control_verified" if case["saved_fit"]["solved"] else "recovered_endpoint") if best and best["solved"] else "budget_exhausted" if stop else "still_unresolved"
    return dict(case_id=case["case_id"], arm=case["arm"], status=status, budget_stop=stop,
        case=copy.deepcopy(case), best=best, attempts=attempts, sobol_seed=seed,
        wall_s=budget.clock()-started, residual_evaluations=budget.evaluations-initial_evaluations,
        objective="position plus normal only" if normal_only else "position plus full orientation",
        endpoint_verified=bool(best and best["solved"]),
        scope="Offline endpoint diagnostic only; original trial label unchanged; no native release, path certification, reward or learner feedback")


def run_probe(payload, links, *, wall_s=MAX_WALL_S, evaluations=MAX_RESIDUAL_EVALUATIONS,
              max_nfev=MAX_NFEV, clock=time.monotonic):
    require(type(max_nfev) is int and 1 <= max_nfev <= MAX_NFEV, "Per-start nfev must be 1..1000")
    cases, selection = select_cases(payload)
    require(set(links) == {"Arm10", "Arm14"}, "Both arm chains required")
    budget = Budget(wall_s, evaluations, clock)
    results, stop = [], None
    for case in cases:
        try:
            budget.check()
        except BudgetExhausted as exc:
            stop = str(exc)
        if stop:
            results.append(dict(case_id=case["case_id"], arm=case["arm"], status="not_run",
                reason=stop, case=copy.deepcopy(case), endpoint_verified=False))
            continue
        result = probe_case(case, links[case["arm"]], budget, max_nfev)
        results.append(result)
        stop = result["budget_stop"]
    return dict(schema=VERSION, selection=selection, limits=dict(maximum_cases=MAX_CASES,
        maximum_starts_per_case=MAX_STARTS, max_nfev_per_start=max_nfev,
        maximum_residual_evaluations=evaluations, maximum_wall_s=float(wall_s),
        acceptance_residual=TOLERANCE, angular_residual_weight=.25),
        selection_sha256=canonical_hash(cases), cases=results,
        wall_s=clock()-budget.started, residual_evaluations=budget.evaluations,
        counts={name:sum(row["status"] == name for row in results) for name in (
            "recovered_endpoint", "control_verified", "still_unresolved", "budget_exhausted", "not_run")},
        native_trial_labels_changed=False, learner_feedback=False, simulator_started=False,
        scope="CPU-only TRAIN diagnostic. A recovered endpoint is not a successful or collision-certified shot; an unresolved or unrun case is not proof of impossibility.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Audit with probe_candidates, or explicit JSON cases")
    parser.add_argument("--assets", type=Path, required=True, help="Existing read-only generated dropper/board_arm URDFs")
    parser.add_argument("--output", type=Path, required=True, help="New diagnostic JSON; never native progress/result")
    parser.add_argument("--max-wall-s", type=float, default=MAX_WALL_S)
    parser.add_argument("--max-nfev", type=int, default=MAX_NFEV)
    parser.add_argument("--max-residual-evaluations", type=int, default=MAX_RESIDUAL_EVALUATIONS)
    args = parser.parse_args()
    if args.output.exists() or args.output.name in ("progress.json", "result.json", "experiment-binding.json"):
        parser.error("Choose a new, separately named offline diagnostic output")
    assets = {"Arm10":args.assets/"dropper.urdf", "Arm14":args.assets/"board_arm.urdf"}
    # Bind parsed inputs to the same bytes hashed, even if a live audit changes
    # its path while this separate offline computation is in progress.
    input_bytes = args.input.read_bytes()
    assets_bytes = {name:path.read_bytes() for name, path in assets.items()}
    def reject(value):
        raise ValueError("Nonfinite JSON: " + value)
    payload = json.loads(input_bytes, parse_constant=reject)
    links = {name:chain(ET.fromstring(raw)) for name, raw in assets_bytes.items()}
    result = run_probe(payload, links, wall_s=args.max_wall_s,
        evaluations=args.max_residual_evaluations, max_nfev=args.max_nfev)
    result["source"] = dict(input_file=str(args.input.resolve()), input_sha256=hashlib.sha256(input_bytes).hexdigest(),
        assets_sha256={name:hashlib.sha256(raw).hexdigest() for name, raw in assets_bytes.items()},
        diagnostic_source_sha256=digest(__file__))
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(output=str(args.output), counts=result["counts"],
        wall_s=result["wall_s"], residual_evaluations=result["residual_evaluations"])), flush=True)


if __name__ == "__main__":
    main()
