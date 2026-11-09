"""Independent CPU verification of candidate provenance and admission evidence.

No IK optimization, actor sampling, native shot, reward change or training.
Checks source aliases against exact supplied TRAIN actions and recomputes each
recorded attempt's geometry, FK, endpoint proxies and path from its common start.
"""

import copy
from collections import Counter
import hashlib
import math
from pathlib import Path

import numpy as np
from scipy.stats import qmc

from continuous_feasibility import (ACTION_NAMES, HIGH, LOW, approximate_clearance,
                                    approximate_path_check, canonical_hash,
                                    decode_action, geometry_check)
from expanded_pose_menu import endpoint_targets, outer_bounds
from joint_feasibility import validate_preflight
from shot_benchmark_reach_audit import fit_details

VERSION = "sircus-feasible-candidates-audit-v1"
KINDS = ("parent", "nearby", "global", "historical_train_success")


def require(value, message):
    if not value:
        raise ValueError(message)


def signed(value):
    require(isinstance(value, dict), "Expected signed artifact")
    unsigned = copy.deepcopy(value)
    expected = unsigned.pop("fingerprint", None)
    require(expected == canonical_hash(unsigned), "Artifact fingerprint mismatch")


def values(action):
    return [float(action[name]) for name in ACTION_NAMES]


def physical_key(cell_id, action):
    return (cell_id, canonical_hash(values(action)))


def audit_decision(decision, scene, cell, action, links, starts):
    """Recompute all reached gates, including reasons on declined candidates."""
    binding = decision["binding"]
    validate_preflight(decision, scene, cell, action, links, starts,
                       seeds=binding["ik_seeds_rad"],
                       solver_protocol=binding["solver_protocol"])
    qualification = decision["qualification"]
    geom = geometry_check(scene, cell, action)
    require(qualification["geometry"] == geom, "Geometry evidence changed")
    reason, goals = None, None
    if not geom["accepted"]:
        reason = geom["reasons"][0]
        require(not qualification["ik"], "IK ran after geometry decline")
    else:
        targets = endpoint_targets(scene, cell, action)
        bounds = outer_bounds(targets, links)
        require(qualification["outer_bounds"] == bounds, "Outer reach evidence changed")
        if not bounds["eligible"]:
            reason = "outer_link_length_bound"
            require(not qualification["ik"], "IK ran outside necessary bound")
        else:
            fits = qualification["ik"]
            reached = []
            for name in ("Arm10", "Arm14"):
                require(name in fits, "Missing reached arm fit")
                reached.append(name)
                fit_details(targets[name], fits[name], links[name], np.eye(4), name == "Arm14")
                if not fits[name]["solved"]:
                    reason = "endpoint_ik_unresolved_" + name
                    break
            require(list(fits) == reached, "IK short-circuit order changed")
            if reason is None:
                clear = approximate_clearance(scene, links, fits)
                require(qualification["approximate_clearance"] == clear, "Endpoint clearance changed")
                if not clear["accepted"]:
                    reason = clear["reasons"][0]
                else:
                    goals = {n: fits[n]["q"] for n in ("Arm10", "Arm14")}
    endpoint_accepted = reason is None
    require(qualification["accepted"] == endpoint_accepted
            and qualification["reason"] == reason, "Endpoint admission/reason differs")
    if endpoint_accepted:
        path = approximate_path_check(scene, links, starts, goals, samples=16, cell=cell)
        require(decision["path"] == path, "Path proxy evidence differs")
        if not path["accepted"]:
            reason = path["reasons"][0]
    else:
        require(decision["path"] is None, "Path evaluated after endpoint decline")
    require(decision["accepted"] == (reason is None) and decision["reason"] == reason,
            "Candidate admission/reason differs")
    require(decision["release_authorized"] is False and decision["bucket_impossibility_proven"] is False,
            "Candidate evidence claims release or impossibility")
    return bool(decision["accepted"])


def audit_candidates(plan, result, successes, scene, links, common_starts, *,
                     parent_actions, parent_latent_means, parent_latent_scales):
    """Verify every raw source alias, exact successful action and recorded gate."""
    signed(plan); signed(result)
    require(plan.get("schema") == "sircus-train-feasible-candidates-v1"
            and result.get("schema") == "sircus-train-feasible-candidates-v1-qualification",
            "Unknown candidate artifact schema")
    require(result["plan_fingerprint"] == plan["fingerprint"], "Qualification references another plan")
    require(result.get("source_sha256") == plan.get("source_sha256")
            and {"feasible_candidates.py","continuous_feasibility.py","joint_feasibility.py"}.issubset(plan.get("source_sha256", {})),
            "Missing candidate source binding")
    for artifact in (plan,result):
        for name, expected in artifact.get("source_sha256", {}).items():
            require(Path(name).name == name and hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() == expected,
                    "Candidate source bytes changed")
        require(artifact.get("optimizer_updates") == artifact.get("native_shots") == 0
                and artifact.get("release_authorized") is False, "Candidate artifact claims new shots/training/release")
    cells = {c["id"]: c for c in plan["cells"]}
    require(len(cells) == len(plan["cells"]) and all(c["split"] == "train" for c in cells.values()),
            "Expected unique TRAIN contexts")
    require(set(common_starts) == set(cells), "Common-start coverage differs")
    if "commonstarts" in result:
        require(result["commonstarts"] == common_starts and result["scene_sha256"] == canonical_hash(scene),
                "Qualification state/scene binding differs")
    historic = {s["case_id"]: s for s in successes}
    require(len(historic) == len(successes), "Duplicate historic case identity")
    require(set(parent_actions) == set(parent_latent_means) == set(parent_latent_scales) == set(cells),
            "Bound parent context coverage differs")
    # Independently regenerate each prespecified numerical proposal. Do not call
    # the generator under audit or accept an altered plan's self-description.
    expected_samples = {}
    for i, cell in enumerate(plan["cells"]):
        name = cell["id"]
        mean = np.asarray(parent_latent_means[name], dtype=float)
        scale = np.asarray(parent_latent_scales[name], dtype=float)
        require(mean.shape == scale.shape == (5,) and np.isfinite(mean).all()
                and np.isfinite(scale).all() and (scale > 0).all(), "Invalid bound parent distribution")
        distribution = plan["parent_distributions"][name]
        require(distribution["latent_mean"] == mean.tolist() and distribution["latent_scale"] == scale.tolist()
                and values(distribution["exact_action"]) == values(parent_actions[name]),
                "Plan parent differs from bound parent inputs")
        rng = np.random.default_rng(plan["protocol"]["random_seed"] + i)
        latent = mean + scale*rng.standard_normal((plan["protocol"]["nearby_per_cell"],5))
        for j, vector in enumerate(LOW+(np.tanh(latent)+1)*(HIGH-LOW)/2):
            expected_samples[name,"nearby",j] = canonical_hash(vector.tolist())
        count = plan["protocol"]["global_per_cell"]
        if count:
            unit = qmc.Sobol(d=5,scramble=True,seed=plan["protocol"]["random_seed"]+100000+i).random_base2(int(math.log2(count)))
            for j, vector in enumerate(LOW+unit*(HIGH-LOW)):
                expected_samples[name,"global",j] = canonical_hash(vector.tolist())
    planned, alias_count, history_seen = {}, Counter(), {}
    parent_seen, sample_seen = set(), set()
    for record in plan["records"]:
        identifier, cell_id, action = record["candidate_id"], record["cell_id"], record["action"]
        require(identifier not in planned and cell_id in cells, "Duplicate candidate or non-TRAIN context")
        key = physical_key(cell_id, action)
        require(identifier == canonical_hash(dict(cell_id=cell_id,values=values(action))),
                "Candidate identity differs from exact physical action")
        require(key not in (x[0] for x in planned.values()), "Exact physical duplicate was not merged")
        expected = decode_action(cells[cell_id], values(action), action["id"])
        for name, value in expected.items():
            require(name == "id" or abs(action[name]-value) <= 1e-8, "Candidate geometry differs from parameters")
        require(record["values"] == values(action), "Physical parameter vector changed")
        require(record["sources"], "Candidate has no provenance")
        witness_cases = [w["case_id"] for w in record.get("witnesses", [])]
        alias_cases = [s["case_id"] for s in record["sources"] if s["kind"] == "historical_train_success"]
        require(witness_cases == alias_cases, "Historical witness coverage/order changed")
        targets=endpoint_targets(scene,cells[cell_id],action)
        for witness in record.get("witnesses", []):
            case_id=witness["case_id"]
            require(case_id in historic and witness["ik"] == historic[case_id]["ik"], "Historical witness changed")
            for name in ("Arm10","Arm14"):
                fit_details(targets[name],witness["ik"][name],links[name],np.eye(4),name=="Arm14")
                require(witness["ik"][name]["solved"] is True, "Historical witness is unresolved")
        for source in record["sources"]:
            kind = source["kind"]
            require(kind in KINDS, "Unknown source family")
            alias_count[kind] += 1
            require(source["physical_sha256"] == key[1], "Source physical identity changed")
            if kind == "historical_train_success":
                case_id = source["case_id"]
                require(case_id in historic and case_id not in history_seen, "Missing or duplicate TRAIN case alias")
                old = historic[case_id]
                require(old["cell"]["split"] == "train" and old["cell"]["id"] == cell_id
                        and old["cell"]["bucket_xy_m"] == cells[cell_id]["bucket_xy_m"]
                        and physical_key(cell_id, old["action"]) == key,
                        "Successful TRAIN action/context was changed")
                history_seen[case_id] = identifier
            elif kind == "parent":
                require(cell_id not in parent_seen, "Duplicate parent alias")
                require(physical_key(cell_id,parent_actions[cell_id]) == key,
                        "Exact frozen parent action was changed")
                parent_seen.add(cell_id)
            else:
                index = source["index"]
                require(type(index) is int and 0 <= index < plan["protocol"][kind + "_per_cell"],
                        "Sample index outside prespecified budget")
                require((cell_id,kind,index) not in sample_seen, "Duplicate sampler index")
                require(expected_samples[cell_id,kind,index] == key[1], "Prespecified numerical proposal changed")
                sample_seen.add((cell_id,kind,index))
        planned[identifier] = (key, record)
    expected_counts = {"parent":len(cells), "nearby":len(cells)*plan["protocol"]["nearby_per_cell"],
                       "global":len(cells)*plan["protocol"]["global_per_cell"],
                       "historical_train_success":len(historic)}
    require({kind:alias_count[kind] for kind in KINDS} == expected_counts and parent_seen == set(cells)
            and set(history_seen) == set(historic), "Raw candidate source union lost or added entries")
    if "source_requests" in plan:
        require(plan["source_requests"] == expected_counts and plan["requests"] == sum(expected_counts.values())
                and plan["unique_candidates"] == len(planned) and plan["historical_cases"] == len(historic)
                and plan["all_historical_actions_included"] is True, "Plan source accounting differs")
    require(len(result["records"]) == len(planned), "Qualification coverage differs")
    observed, admitted_ids, attempts = set(), set(), 0
    reasons = Counter()
    by_id = {}
    for row in result["records"]:
        identifier = row["candidate_id"]
        require(identifier in planned and identifier not in observed, "Unknown or duplicate qualification")
        observed.add(identifier)
        record = planned[identifier][1]
        require(row["cell_id"] == record["cell_id"] and row["action"] == record["action"]
                and row["sources"] == record["sources"], "Qualification changed candidate/provenance")
        recorded = row["attempts"]
        require(recorded, "No admission attempt recorded")
        witnesses = record.get("witnesses", [])
        if witnesses:
            seed_options = [(dict(kind="historical_witness",case_id=w["case_id"]),
                             {name:w["ik"][name]["q"] for name in ("Arm10","Arm14")}) for w in witnesses]
        else:
            seed_options = [(dict(kind="common_start"),common_starts[row["cell_id"]])]
        require(len(recorded) <= len(seed_options), "Recorded attempts exceed declared seed budget")
        for i, attempt in enumerate(recorded):
            decision = attempt["decision"]
            expected_source, expected_seeds = seed_options[i]
            require(attempt["seed_source"] == expected_source
                    and decision["binding"]["ik_seeds_rad"] == expected_seeds,
                    "Attempt seed association differs from planned witness/common state")
            accepted = audit_decision(decision, scene, cells[row["cell_id"]], row["action"], links,
                                      common_starts[row["cell_id"]])
            attempts += 1
            require(not accepted or i == len(recorded)-1, "Attempts continued after admission")
        require(row["decision"] == recorded[-1]["decision"], "Selected decision differs from last attempt")
        require(row["decision"]["accepted"] or len(recorded) == len(seed_options),
                "Declined candidate did not exhaust declared finite seeds")
        if "witness_count" in row:
            expected_unneeded=[w["case_id"] for w in witnesses[len(recorded):]] if witnesses else []
            require(row["witness_count"] == len(witnesses) and row["attempted_seed_count"] == len(recorded)
                    and row["unneeded_witness_cases"] == expected_unneeded, "Seed/witness summary accounting differs")
        require(type(row["accepted"]) is bool and row["accepted"] == row["decision"]["accepted"] and row["reason"] == row["decision"]["reason"],
                "Qualification summary differs from evidence")
        if row["accepted"]:
            admitted_ids.add(identifier)
        else:
            reasons[row["reason"]] += 1
        by_id[identifier] = row
    if "requests" in result:
        require(result["requests"] == len(planned) and result["admitted"] == len(admitted_ids),
                "Qualification total accounting differs")
    for kind in KINDS:
        aliases=[row for row in by_id.values() for source in row["sources"] if source["kind"]==kind]
        computed=dict(requested_aliases=len(aliases),admitted_aliases=sum(row["accepted"] for row in aliases),
                      declined_aliases=sum(not row["accepted"] for row in aliases),
                      unique_candidates=len({row["candidate_id"] for row in aliases}),
                      exclusion_reasons=dict(sorted(Counter(row["reason"] for row in aliases if not row["accepted"]).items())))
        if "source_survival" in result:
            require(result["source_survival"][kind] == computed, "Source survival accounting differs")
    for cell in plan["cells"]:
        rows=[row for row in by_id.values() if row["cell_id"]==cell["id"]]
        admitted=sum(row["accepted"] for row in rows)
        if "per_cell" in result:
            actual=[row for row in result["per_cell"] if row["cell_id"]==cell["id"]]
            expected=dict(cell_id=cell["id"],requests=len(rows),admitted=admitted,declined=len(rows)-admitted,
                          status="supported" if admitted else "unresolved",
                          reasons=dict(sorted(Counter(row["reason"] for row in rows if not row["accepted"]).items())))
            require(actual == [expected], "Per-context survival accounting differs")
    kept_history = [k for k, identifier in history_seen.items() if identifier in admitted_ids]
    excluded_history = [dict(case_id=k, candidate_id=identifier, reason=by_id[identifier]["reason"])
                        for k, identifier in history_seen.items() if identifier not in admitted_ids]
    clean_ids={k for k,s in historic.items() if s.get("clean",s.get("route_quality")==2)}
    value = dict(schema=VERSION, status="passed", plan_fingerprint=plan["fingerprint"],
                 qualification_fingerprint=result["fingerprint"], contexts=len(cells),
                 raw_source_requests=sum(alias_count.values()), source_requests=dict(alias_count),
                 unique_candidates=len(planned), admitted=len(admitted_ids), declined=len(planned)-len(admitted_ids),
                 reasons=dict(reasons), independently_rechecked_attempts=attempts,
                 regenerated_parent_and_sample_aliases=len(cells)+len(expected_samples),
                 successful_train_shots=len(historic), successful_train_survive=len(kept_history),
                 successful_train_excluded=len(excluded_history), exclusions=excluded_history,
                 clean_train_shots=len(clean_ids), clean_train_survive=len(clean_ids.intersection(kept_history)),
                 new_native_shots=0, optimizer_updates=0, release_authorized=False,
                 audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 scope="TRAIN-informed proposal union; geometry-only admission independently rechecked. Historical captures are not re-executed or guaranteed by admission.")
    value["fingerprint"] = canonical_hash(value)
    return value
