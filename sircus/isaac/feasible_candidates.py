"""TRAIN-only finite shot support around an unchanged saved parent policy.

Historical successful TRAIN actions inform proposal inclusion, never geometry
acceptance. Exact physical actions are retained; no flight model, reward ranking,
projection, clipping, optimizer, native simulation or robot command runs here.
"""

import concurrent.futures
import copy
import hashlib
import math
from pathlib import Path
import time
from collections import Counter

import numpy as np
from scipy.stats import qmc

from continuous_feasibility import ACTION_NAMES, HIGH, LOW, canonical_hash, decode_action
from expanded_pose_menu import endpoint_targets
from joint_feasibility import preflight
from shot_benchmark_reach_audit import fit_details

VERSION = "sircus-train-feasible-candidates-v1"
KINDS = ("parent", "nearby", "global", "historical_train_success")
ARMS = ("Arm10", "Arm14")
SCOPE = (
    "Finite TRAIN-context proposal union: exact frozen-parent choices, parent Gaussian "
    "samples, broad Sobol samples and all recorded successful paid TRAIN actions. "
    "History informs proposal inclusion, not geometry gates. No held-out outcomes, "
    "flight prediction/ranking, action projection, training, native motion or release. "
    "Qualified means inherited endpoint/FK/capsule and sampled-path checks only."
)


def require(value, message):
    if not value:
        raise ValueError(message)


def plain(value):
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, dict):
        require(all(isinstance(k, str) for k in value), "Expected string JSON keys")
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    require(value is None or type(value) in (bool, str, int, float), "Expected JSON values")
    require(type(value) is not float or math.isfinite(value), "Nonfinite candidate data")
    return value


def signed(value):
    result = plain(value)
    result["fingerprint"] = canonical_hash(result)
    return result


def unsigned(value):
    require(isinstance(value, dict), "Expected signed candidate artifact")
    result = copy.deepcopy(value)
    fingerprint = result.pop("fingerprint", None)
    require(fingerprint == canonical_hash(plain(result)), "Candidate artifact fingerprint mismatch")
    return result


def source_hashes():
    import continuous_feasibility
    import joint_feasibility
    files = [Path(__file__), Path(continuous_feasibility.__file__), Path(joint_feasibility.__file__)]
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def train_cells(cells):
    require(isinstance(cells, (list, tuple)) and cells, "Expected nonempty TRAIN cells")
    result = []
    for cell in cells:
        require(isinstance(cell, dict) and cell.get("split") == "train", "Candidates require TRAIN cells only")
        require(isinstance(cell.get("id"), str) and cell["id"], "Missing TRAIN cell identity")
        xy = np.asarray(cell.get("bucket_xy_m"), dtype=float)
        require(xy.shape == (2,) and np.isfinite(xy).all(), "Expected finite TRAIN bucket XY")
        result.append(dict(id=cell["id"], bucket_xy_m=xy.tolist(), split="train"))
    require(len({c["id"] for c in result}) == len(result)
            and len({tuple(c["bucket_xy_m"]) for c in result}) == len(result), "Duplicate TRAIN context")
    return result


def values_of(action):
    values = np.asarray([action[k] for k in ACTION_NAMES], dtype=float)
    require(values.shape == (5,) and np.isfinite(values).all()
            and np.all(values >= LOW) and np.all(values <= HIGH), "Physical action outside original bounds")
    return values


def exact_action(cell, value, identifier):
    if not isinstance(value, dict):
        return decode_action(cell, value, identifier)
    require(isinstance(value.get("id"), str) and value["id"], "Missing exact action identity")
    expected = decode_action(cell, values_of(value), value["id"])
    for key, target in expected.items():
        if key == "id":
            continue
        actual = value.get(key)
        require(type(actual) in (float, int) and math.isfinite(actual)
                and abs(actual-target) <= 1e-8, "Derived action geometry disagrees: " + key)
    return plain({key: copy.deepcopy(value[key]) for key in expected})


def by_cell(value, cells, label):
    if isinstance(value, dict):
        require(set(value) == {c["id"] for c in cells}, label + " context membership changed")
        return {c["id"]: copy.deepcopy(value[c["id"]]) for c in cells}
    require(isinstance(value, (list, tuple, np.ndarray)) and len(value) == len(cells),
            label + " needs one entry per TRAIN context")
    return {c["id"]: copy.deepcopy(row) for c, row in zip(cells, value)}


def generate_candidates(cells, parent_actions, parent_latent_means, parent_latent_scales,
                        successes, *, random_seed=20261124, nearby_per_cell=16,
                        global_per_cell=64):
    """Create a deterministic proposal union; retain every physical source alias.

    Parent maps may be keyed by context ID or aligned with ``cells``. Each
    successful TRAIN row is {case_id, cell:{id,bucket_xy_m,split:'train'},
    action:<exact decoded action>, ik:{Arm10:fit,Arm14:fit}}. Optional outcome
    metadata is ignored. Historical witnesses are not acceptance certificates.
    """
    cells = train_cells(cells)
    require(type(random_seed) is int and 0 <= random_seed < 2**32-100000-len(cells), "Invalid proposal seed")
    require(type(nearby_per_cell) is int and 0 <= nearby_per_cell <= 1024, "Choose 0..1024 nearby proposals")
    require(type(global_per_cell) is int and 0 <= global_per_cell <= 1024
            and (global_per_cell == 0 or global_per_cell & (global_per_cell-1) == 0),
            "Global Sobol count must be zero or a power of two up to 1024")
    parent = by_cell(parent_actions, cells, "Parent actions")
    means = by_cell(parent_latent_means, cells, "Parent means")
    scales = by_cell(parent_latent_scales, cells, "Parent scales")
    require(isinstance(successes, (list, tuple)), "Expected successful TRAIN record list")
    lookup = {cell["id"]: cell for cell in cells}
    records, identities, source_counts, distributions = [], {}, Counter(), {}

    def add(cell, action, source, witness=None):
        vector = values_of(action).tolist()
        # Full precision JSON identity: there is no rounding, tolerance merge or
        # nearest-neighbor substitution. Action IDs do not change physical shots.
        identity = canonical_hash(dict(cell_id=cell["id"], values=vector))
        source = plain(dict(source, original_action_id=action["id"], physical_sha256=canonical_hash(vector)))
        source_counts[source["kind"]] += 1
        if identity not in identities:
            identities[identity] = len(records)
            records.append(dict(candidate_id=identity, cell_id=cell["id"], action=copy.deepcopy(action),
                                values=vector, sources=[], witnesses=[]))
        row = records[identities[identity]]
        require(row["values"] == vector, "Candidate physical identity collision")
        row["sources"].append(source)
        if witness is not None:
            row["witnesses"].append(plain(witness))

    for i, cell in enumerate(cells):
        name = cell["id"]
        mean, scale = np.asarray(means[name], dtype=float), np.asarray(scales[name], dtype=float)
        require(mean.shape == scale.shape == (5,) and np.isfinite(mean).all()
                and np.isfinite(scale).all() and (scale > 0).all(), "Invalid saved parent Gaussian")
        action = exact_action(cell, parent[name], "parent-" + name)
        distributions[name] = dict(latent_mean=mean.tolist(), latent_scale=scale.tolist(), exact_action=action)
        add(cell, action, dict(kind="parent", index=0))
        rng = np.random.default_rng(random_seed+i)
        latent = mean + scale*rng.standard_normal((nearby_per_cell, 5))
        for j, vector in enumerate(LOW+(np.tanh(latent)+1)*(HIGH-LOW)/2):
            add(cell, decode_action(cell, vector, f"nearby-{name}-{j:04d}"), dict(kind="nearby", index=j))
        if global_per_cell:
            points = qmc.Sobol(d=5, scramble=True, seed=random_seed+100000+i).random_base2(int(math.log2(global_per_cell)))
            for j, vector in enumerate(LOW+points*(HIGH-LOW)):
                add(cell, decode_action(cell, vector, f"global-{name}-{j:04d}"), dict(kind="global", index=j))
    case_ids = set()
    for success in successes:
        require(isinstance(success, dict) and isinstance(success.get("case_id"), str)
                and success["case_id"] and success["case_id"] not in case_ids,
                "Missing or duplicate historical TRAIN case identity")
        cell = train_cells([success["cell"]])[0]
        require(cell["id"] in lookup and lookup[cell["id"]] == cell, "Historical TRAIN context changed")
        ik = success.get("ik")
        require(isinstance(ik, dict) and set(ik) == set(ARMS), "Expected both historical IK witnesses")
        require(all(isinstance(ik[n], dict) and ik[n].get("solved") is True for n in ARMS),
                "Successful TRAIN shot lacks solved endpoint witnesses")
        action = exact_action(cell, success["action"], success["case_id"])
        add(cell, action, dict(kind="historical_train_success", case_id=success["case_id"]),
            witness=dict(case_id=success["case_id"], ik=copy.deepcopy(ik)))
        case_ids.add(success["case_id"])
    return signed(dict(schema=VERSION, cells=cells, parent_distributions=distributions,
        protocol=dict(random_seed=random_seed, nearby_per_cell=nearby_per_cell,
            global_per_cell=global_per_cell, action_names=list(ACTION_NAMES), low=LOW.tolist(), high=HIGH.tolist(),
            nearby_sampling="Independent NumPy Gaussian latent draws using saved parent means/scales; exact tanh/affine decode",
            global_sampling="Scrambled five-dimensional Sobol over the unchanged original action domain",
            deduplication="Exact cell and physical-five-vector identity only; all source aliases retained",
            witness_policy="Fresh original robust IK seeded by each preserved successful witness, until first admission; no action change"),
        source_sha256=source_hashes(), records=records,
        source_requests={kind: source_counts[kind] for kind in KINDS},
        requests=sum(source_counts.values()), unique_candidates=len(records),
        historical_cases=len(case_ids), all_historical_actions_included=True,
        scope=SCOPE, optimizer_updates=0, native_shots=0, release_authorized=False))


def validate_plan(plan):
    value = unsigned(plan)
    require(value.get("schema") == VERSION and value.get("source_sha256") == source_hashes(),
            "Candidate source version changed")
    cells = train_cells(value["cells"])
    by_id = {cell["id"]: cell for cell in cells}
    protocol = value["protocol"]
    require(protocol["action_names"] == list(ACTION_NAMES)
            and protocol["low"] == LOW.tolist() and protocol["high"] == HIGH.tolist(), "Candidate action domain changed")
    require(type(protocol["nearby_per_cell"]) is int and 0 <= protocol["nearby_per_cell"] <= 1024
            and type(protocol["global_per_cell"]) is int and 0 <= protocol["global_per_cell"] <= 1024,
            "Candidate proposal counts changed")
    require(set(value["parent_distributions"]) == set(by_id), "Parent distribution membership changed")
    ids, sources, cases, proposal_indices = set(), Counter(), set(), set()
    for row in value["records"]:
        require(row["cell_id"] in by_id and row["candidate_id"] not in ids, "Candidate context/identity changed")
        action = exact_action(by_id[row["cell_id"]], row["action"], row["candidate_id"])
        vector = values_of(action).tolist()
        require(row["values"] == vector and row["candidate_id"] == canonical_hash(dict(cell_id=row["cell_id"], values=vector)),
                "Candidate physical identity changed")
        require(isinstance(row["sources"], list) and row["sources"], "Candidate has no source aliases")
        for alias in row["sources"]:
            require(alias["kind"] in KINDS and alias["physical_sha256"] == canonical_hash(vector), "Candidate alias changed")
            sources[alias["kind"]] += 1
            if alias["kind"] == "historical_train_success":
                require(alias["case_id"] not in cases, "Historical alias duplicated")
                cases.add(alias["case_id"])
            else:
                key = (row["cell_id"], alias["kind"], alias["index"])
                require(key not in proposal_indices, "Parent proposal alias duplicated")
                proposal_indices.add(key)
                if alias["kind"] == "parent":
                    expected_parent = exact_action(by_id[row["cell_id"]],
                        value["parent_distributions"][row["cell_id"]]["exact_action"], "parent")
                    require(values_of(expected_parent).tolist() == vector, "Exact parent action changed")
        require([w["case_id"] for w in row["witnesses"]] == [s["case_id"] for s in row["sources"] if s["kind"] == "historical_train_success"],
                "Historical witness aliases missing or reordered")
        ids.add(row["candidate_id"])
    require(value["source_requests"] == {kind:sources[kind] for kind in KINDS}
            and value["requests"] == sum(sources.values()) and value["unique_candidates"] == len(ids)
            and value["historical_cases"] == len(cases), "Candidate accounting changed")
    expected_indices = {(cell["id"], kind, index) for cell in cells for kind, count in (
        ("parent", 1), ("nearby", protocol["nearby_per_cell"]), ("global", protocol["global_per_cell"]))
        for index in range(count)}
    require(proposal_indices == expected_indices and value.get("all_historical_actions_included") is True,
            "Candidate source union incomplete")
    require(value.get("optimizer_updates") == value.get("native_shots") == 0
            and value.get("release_authorized") is False, "Candidate artifact cannot train or release")
    return value


def _qualify_one(job):
    scene, cell, row, links, starts, solver, solver_protocol = job
    # Verify every preserved witness before using any, including duplicates not
    # needed after an earlier branch admits the action.
    targets = endpoint_targets(scene, cell, row["action"])
    for witness in row["witnesses"]:
        require(set(witness["ik"]) == set(ARMS), "Historical witness arm mismatch")
        for name in ARMS:
            fit_details(targets[name], witness["ik"][name], links[name], np.eye(4), name == "Arm14")
            require(witness["ik"][name]["solved"] is True, "Historical witness is unresolved")
    seed_options = [(dict(kind="historical_witness", case_id=w["case_id"]),
                     {name:w["ik"][name]["q"] for name in ARMS}) for w in row["witnesses"]]
    if not seed_options:
        seed_options = [(dict(kind="common_start"), starts)]
    attempts = []
    for source, seeds in seed_options:
        kwargs = dict(seeds=seeds)
        if solver is not None:
            kwargs.update(solver=solver, solver_protocol=solver_protocol)
        decision = preflight(scene, cell, row["action"], links, starts, **kwargs)
        attempts.append(dict(seed_source=source, decision=decision))
        if decision["accepted"]:
            break
    decision = attempts[-1]["decision"]
    return dict(candidate_id=row["candidate_id"], cell_id=row["cell_id"], action=copy.deepcopy(row["action"]),
                sources=copy.deepcopy(row["sources"]), accepted=decision["accepted"],
                status=decision["status"], reason=decision["reason"], decision=decision,
                witness_count=len(row["witnesses"]), attempted_seed_count=len(attempts),
                unneeded_witness_cases=[w["case_id"] for w in row["witnesses"][len(attempts):]] if row["witnesses"] else [],
                attempts=attempts)


def qualify_candidates(plan, scene, links, commonstarts, *, workers=1, progress=None,
                       solver=None, solver_protocol=None):
    """Run original CPU geometry admission for every candidate, without shots.

    ``commonstarts`` maps each TRAIN cell ID to both actual six-joint starts.
    For tests only, a custom original-signature solver may be supplied with an
    explicit protocol and workers=1. No rejected action is replaced or retried
    with a different physical action; finite historical IK branches are explicit.
    """
    value = validate_plan(plan)
    require(type(workers) is int and 1 <= workers <= 8, "Choose 1..8 CPU workers")
    require(solver is None or workers == 1 and solver_protocol is not None, "Custom solver needs explicit protocol and one worker")
    cells = {cell["id"]: cell for cell in value["cells"]}
    require(isinstance(commonstarts, dict) and set(commonstarts) == set(cells), "Common-start context membership changed")
    starts = plain(copy.deepcopy(commonstarts))
    for state in starts.values():
        require(isinstance(state, dict) and set(state) == set(ARMS), "Expected both common-start arm states")
        for q in state.values():
            arr = np.asarray(q, dtype=float)
            require(arr.shape == (6,) and np.isfinite(arr).all(), "Invalid common-start joints")
    jobs = [(scene, cells[row["cell_id"]], row, links, starts[row["cell_id"]], solver, solver_protocol)
            for row in value["records"]]
    begin = time.monotonic()
    pool = concurrent.futures.ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    records = []
    try:
        for row in (pool.map(_qualify_one, jobs) if pool else map(_qualify_one, jobs)):
            records.append(row)
            if progress:
                progress(dict(completed=len(records), total=len(jobs), admitted=sum(r["accepted"] for r in records),
                              elapsed_s=time.monotonic()-begin))
    finally:
        if pool:
            pool.shutdown(wait=True, cancel_futures=True)
    source_survival = {}
    for kind in KINDS:
        rows = [(row, alias) for row in records for alias in row["sources"] if alias["kind"] == kind]
        source_survival[kind] = dict(requested_aliases=len(rows), admitted_aliases=sum(row["accepted"] for row,_ in rows),
            declined_aliases=sum(not row["accepted"] for row,_ in rows),
            unique_candidates=len({row["candidate_id"] for row,_ in rows}),
            exclusion_reasons=dict(sorted(Counter(row["reason"] for row,_ in rows if not row["accepted"]).items())))
    history = [dict(case_id=alias["case_id"], candidate_id=row["candidate_id"], cell_id=row["cell_id"],
                    generator_retained=True, accepted=row["accepted"], status=row["status"], reason=row["reason"])
               for row in records for alias in row["sources"] if alias["kind"] == "historical_train_success"]
    per_cell = []
    for cell in value["cells"]:
        rows = [row for row in records if row["cell_id"] == cell["id"]]
        admitted = sum(row["accepted"] for row in rows)
        per_cell.append(dict(cell_id=cell["id"], requests=len(rows), admitted=admitted, declined=len(rows)-admitted,
                             status="supported" if admitted else "unresolved",
                             reasons=dict(sorted(Counter(row["reason"] for row in rows if not row["accepted"]).items()))))
    return signed(dict(schema=VERSION+"-qualification", plan_fingerprint=plan["fingerprint"],
        source_sha256=source_hashes(), commonstarts=starts, scene_sha256=canonical_hash(scene),
        records=records, requests=len(records), admitted=sum(row["accepted"] for row in records),
        source_survival=source_survival, history_survival=history, per_cell=per_cell,
        wall_s=time.monotonic()-begin, workers=workers, scope=SCOPE,
        optimizer_updates=0, native_shots=0, release_authorized=False))
