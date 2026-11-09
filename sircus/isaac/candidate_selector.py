"""Frozen seed-22 preferences over the separately audited TRAIN candidate bank.

This defines a new finite categorical preference law. The native comparison
uses its deterministic argmax, not stochastic PPO sampling or a conditioned
continuous Gaussian. No actor inference, optimizer or simulator runs here.
"""

import copy
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

from continuous_feasibility import ACTION_NAMES, HIGH, LOW, canonical_hash
from feasible_candidates import exact_action, plain, signed, train_cells, unsigned, validate_plan, values_of

VERSION = "sircus-frozen-candidate-comparison-v1"
PARENT_SHA256 = "57d7658e9ed2f72ad09cf4e8523ab497ccbe0708364f1b8ec4a4c5c1aef1fe9f"
PUBLISHED_SUMMARY = Path(__file__).resolve().parents[2] / "baselines/isaac-feasible-candidates-20261109/summary.json"
METHODS = ("parent", "candidate_bank")
ARMS = ("Arm10", "Arm14")
SCOPE = (
    "Prespecified deterministic TRAIN-only comparison: unchanged final parent versus "
    "argmax of its saved latent-Gaussian preference over qualified finite candidates. "
    "Historical TRAIN outcomes informed bank inclusion but never rank choices. "
    "New categorical probabilities are descriptive preferences, not old continuous "
    "PPO log probabilities or capture probabilities. Fresh native IK/path, settling "
    "and release guards remain mandatory; no optimizer, held-out search or retries."
)


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def serialized_sha256(value):
    """Producer bank artifacts use this exact readable JSON representation."""
    return hashlib.sha256((json.dumps(value, indent=2, allow_nan=False) + "\n").encode()).hexdigest()


def _read(path):
    return json.loads(Path(path).read_bytes(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError("Nonfinite JSON " + x)))


def load_bank(directory):
    """Load only the approved bank; verify actual file bytes before selection."""
    directory = Path(directory)
    values = {name: _read(directory/name) for name in ("plan.json", "qualification.json", "audit.json", "summary.json")}
    hashes = {name: digest(directory/name) for name in values}
    require(hashes["summary.json"] == digest(PUBLISHED_SUMMARY), "Bank summary differs from published preservation audit")
    for name in ("plan.json", "qualification.json", "audit.json"):
        require(hashes[name] == values["summary.json"]["artifacts_sha256"][name], "Bank file SHA mismatch: " + name)
    return values["plan.json"], values["qualification.json"], values["audit.json"], values["summary.json"], hashes


def preference_probabilities(mean, scale, candidates, parent_action):
    """A normalized finite law with latent-Gaussian density as its score.

    A qualified exact parent uses its saved latent mean directly. For other
    actions the original physical affine map and tanh are inverted. Saturated
    boundary actions fail closed rather than being clipped or silently omitted.
    """
    mean, scale = np.asarray(mean, float), np.asarray(scale, float)
    require(mean.shape == scale.shape == (5,) and np.isfinite(mean).all()
            and np.isfinite(scale).all() and (scale > 0).all(), "Invalid frozen parent Gaussian")
    require(candidates, "Cannot normalize empty support")
    parent_values = values_of(parent_action)
    scores, parent_indices = [], []
    for i, row in enumerate(candidates):
        vector = values_of(row["action"])
        if np.array_equal(vector, parent_values):
            latent = mean.copy()
            parent_indices.append(i)
        else:
            normalized = 2*(vector-LOW)/(HIGH-LOW)-1
            require((normalized > -1).all() and (normalized < 1).all(),
                    "Qualified action is on a saturated latent boundary; explicit new protocol required")
            latent = np.arctanh(normalized)
        score = float(-.5*np.square((latent-mean)/scale).sum()
                      - np.log(scale).sum() - 2.5*math.log(2*math.pi))
        require(math.isfinite(score), "Nonfinite finite-support preference score")
        scores.append(score)
    scores = np.asarray(scores)
    log_normalizer = float(scores.max() + np.log(np.exp(scores-scores.max()).sum()))
    log_probabilities = scores-log_normalizer
    probabilities = np.exp(log_probabilities)
    require(np.isfinite(log_probabilities).all() and np.isclose(probabilities.sum(), 1., atol=1e-12, rtol=0),
            "Finite candidate probabilities failed normalization")
    if parent_indices:
        require(len(parent_indices) == 1, "Exact parent duplicated in qualified support")
        selected = parent_indices[0]
        require(scores[selected] >= scores.max()-1e-12, "Qualified parent is not its own Gaussian maximum")
        reason = "qualified_parent_preserved"
    else:
        selected = min(range(len(candidates)), key=lambda i: (-float(scores[i]), candidates[i]["candidate_id"]))
        reason = "highest_frozen_parent_latent_preference"
    return dict(scores=scores.tolist(), probabilities=probabilities.tolist(),
                log_probabilities=log_probabilities.tolist(), log_normalizer=log_normalizer,
                selected_index=selected, reason=reason, parent_qualified=bool(parent_indices),
                law="categorical normalized latent-Gaussian density scores; not exact continuous conditional density",
                selection="deterministic argmax; qualified parent wins an exact numerical tie, otherwise candidate ID lexical order",
                stochastic_sampling=False)


def build_comparison(plan, qualification, audit, summary, *, artifact_sha256=None):
    """Freeze all 70 requests before native outcomes are observed."""
    bank = validate_plan(plan)
    qualified, audited, published = unsigned(qualification), unsigned(audit), unsigned(summary)
    hashes = artifact_sha256 or {name: serialized_sha256(value) for name, value in (
        ("plan.json", plan), ("qualification.json", qualification), ("audit.json", audit), ("summary.json", summary))}
    require(set(hashes) == {"plan.json", "qualification.json", "audit.json", "summary.json"}, "Bank artifact hash membership changed")
    require(hashes["summary.json"] == digest(PUBLISHED_SUMMARY), "Comparison requires the published candidate bank")
    for name, value in (("plan.json", plan), ("qualification.json", qualification), ("audit.json", audit), ("summary.json", summary)):
        require(hashes[name] == serialized_sha256(value), "Input object differs from its bank bytes: " + name)
    require(published["schema"] == "sircus-seed22-train-candidate-bank-v1" and published["status"] == "passed"
            and published["seed"] == 20261122 and audited["status"] == "passed", "Expected audited seed-22 bank")
    for name in ("plan.json", "qualification.json", "audit.json"):
        require(hashes[name] == published["artifacts_sha256"][name], "Published bank hash mismatch: " + name)
    require(qualified["plan_fingerprint"] == audited["plan_fingerprint"] == plan["fingerprint"]
            and audited["qualification_fingerprint"] == qualification["fingerprint"], "Bank fingerprint lineage mismatch")
    require(published["input_bindings"]["original"]["checkpoint_sha256"] == PARENT_SHA256,
            "Unexpected frozen parent checkpoint")
    require(published["optimizer_updates"] == published["new_native_shots"] == published["held_out_shot_files_read"] == 0,
            "Candidate preparation consumed training or held-out shots")
    require(qualified["optimizer_updates"] == qualified["native_shots"] == audited["optimizer_updates"] == audited["new_native_shots"] == 0,
            "Candidate artifacts contain optimizer/native work")
    cells = train_cells(bank["cells"])
    require(len(cells) == 35 and set(qualified["commonstarts"]) == {c["id"] for c in cells}, "Expected all 35 TRAIN start states")
    expected = {r["candidate_id"]: r for r in bank["records"]}
    rows = {r["candidate_id"]: r for r in qualified["records"]}
    require(len(expected) == len(bank["records"]) == len(rows) == len(qualified["records"])
            and set(rows) == set(expected), "Candidate qualification coverage changed")
    require(sum(r["accepted"] is True for r in rows.values()) == qualified["admitted"] == audited["admitted"] == published["admitted"],
            "Candidate admission total mismatch")
    by_cell = {cell["id"]: [] for cell in cells}
    for candidate_id, row in rows.items():
        original = expected[candidate_id]
        require(row["cell_id"] == original["cell_id"] and row["action"] == original["action"]
                and row["sources"] == original["sources"] and type(row["accepted"]) is bool,
                "Qualification changed an exact candidate")
        decision = row["decision"]
        require(decision["action"] == row["action"] and decision["accepted"] == row["accepted"]
                and decision["binding"]["context"]["id"] == row["cell_id"]
                and decision["binding"]["measured_starts_rad"] == qualified["commonstarts"][row["cell_id"]],
                "Qualification state/action binding changed")
        if row["accepted"]:
            require(row["reason"] is None and decision["qualification"]["accepted"] is True
                    and decision["path"]["accepted"] is True and decision["release_authorized"] is False,
                    "Qualified candidate lacks independent gates")
            by_cell[row["cell_id"]].append(row)
    cohorts = [dict(method=method, request_count=35, actions=[], native_enabled=[], decline_reasons=[], ik_seeds=[])
               for method in METHODS]
    selections = []
    for cell in cells:
        cell_id = cell["id"]
        distribution = bank["parent_distributions"][cell_id]
        parent = exact_action(cell, distribution["exact_action"], "parent")
        cohorts[0]["actions"].append(copy.deepcopy(parent))
        cohorts[0]["native_enabled"].append(True)
        cohorts[0]["decline_reasons"].append(None)
        cohorts[0]["ik_seeds"].append(None)
        support = sorted(by_cell[cell_id], key=lambda r: r["candidate_id"])
        if not support:
            # This is a disabled geometry placeholder, not a replacement shot.
            cohorts[1]["actions"].append(copy.deepcopy(parent))
            cohorts[1]["native_enabled"].append(False)
            cohorts[1]["decline_reasons"].append("no_qualified_candidate_support")
            cohorts[1]["ik_seeds"].append(None)
            selections.append(dict(cell_id=cell_id, selected_candidate_id=None, qualified_candidates=0,
                predeclined=True, decline_reason="no_qualified_candidate_support", parent_preserved=False,
                action_changed=False, reason="no_qualified_candidate_support", candidates=[]))
            continue
        preference = preference_probabilities(distribution["latent_mean"], distribution["latent_scale"], support, parent)
        selected = support[preference["selected_index"]]
        fits = selected["decision"]["qualification"]["ik"]
        require(set(fits) == set(ARMS) and all(fits[n]["solved"] is True for n in ARMS), "Selected candidate lacks both IK seed witnesses")
        cohorts[1]["actions"].append(copy.deepcopy(selected["action"]))
        cohorts[1]["native_enabled"].append(True)
        cohorts[1]["decline_reasons"].append(None)
        # Exact parent preservation includes its original fresh-IK protocol.
        # A different bank action may warm-start from its prepared endpoint fit.
        cohorts[1]["ik_seeds"].append(None if preference["parent_qualified"] else copy.deepcopy(fits))
        selections.append(dict(cell_id=cell_id, selected_candidate_id=selected["candidate_id"],
            qualified_candidates=len(support), predeclined=False, decline_reason=None,
            parent_preserved=preference["parent_qualified"],
            action_changed=not np.array_equal(values_of(selected["action"]), values_of(parent)),
            reason=preference["reason"], preference=preference,
            candidates=[dict(candidate_id=row["candidate_id"], score=preference["scores"][i],
                categorical_probability=preference["probabilities"][i],
                categorical_log_probability=preference["log_probabilities"][i]) for i,row in enumerate(support)]))
    requests = []
    for cohort_index, cohort in enumerate(cohorts):
        for slot, cell in enumerate(cells):
            requests.append(dict(request_index=len(requests), cohort=cohort_index, method=cohort["method"], slot=slot,
                cell_id=cell["id"], action=copy.deepcopy(cohort["actions"][slot]),
                predeclined=not cohort["native_enabled"][slot], decline_reason=cohort["decline_reasons"][slot],
                candidate_id=selections[slot]["selected_candidate_id"] if cohort_index else None))
    result = signed(dict(schema=VERSION, seed=20261122, cells=cells, cohorts=cohorts, requests=requests,
        request_count=70, cohort_order=list(METHODS), selections=selections,
        parent_preserved=sum(s["parent_preserved"] for s in selections), changed_bank_actions=sum(s["action_changed"] for s in selections),
        bank_predeclined=sum(s["predeclined"] for s in selections),
        input_bindings=dict(bank_artifact_sha256=hashes, bank_plan_fingerprint=plan["fingerprint"],
            bank_qualification_fingerprint=qualification["fingerprint"], bank_audit_fingerprint=audit["fingerprint"],
            original=copy.deepcopy(published["input_bindings"]["original"]),
            assets_sha256=copy.deepcopy(published["input_bindings"]["assets_sha256"]),
            common_start_sha256=canonical_hash(qualified["commonstarts"])),
        ik_seed_policy="Unchanged parent actions use native actual-start seeds in both cohorts; changed bank actions warm-start fresh native IK from their prepared fits",
        selector_source_sha256=digest(__file__), optimizer_updates=0, native_shots_observed=0,
        held_out_shot_files_read=0, release_authorized=False, scope=SCOPE))
    validate_comparison(result)
    return result


def validate_comparison(comparison):
    """Validate the immutable 70-request contract before an Isaac app starts.

    The caller may append a source-bound ``native_binding`` and resign the plan.
    The native harness still independently verifies those bytes and original
    checkpoint/scene/camera inputs. Nothing in this plan authorizes release.
    """
    value = unsigned(comparison)
    require(value.get("schema") == VERSION and value.get("seed") == 20261122
            and value.get("request_count") == 70 and value.get("cohort_order") == list(METHODS),
            "Unknown or unbounded comparison contract")
    require(value.get("optimizer_updates") == value.get("native_shots_observed") == value.get("held_out_shot_files_read") == 0
            and value.get("release_authorized") is False, "Comparison cannot train or bypass release guards")
    require(value.get("selector_source_sha256") == digest(__file__), "Selector source bytes changed")
    cells = train_cells(value["cells"])
    require(len(cells) == 35 and len(value["cohorts"]) == 2 and len(value["selections"]) == 35
            and len(value["requests"]) == 70, "Comparison needs 35 paired TRAIN positions and 70 requests")
    require(value["input_bindings"]["original"]["checkpoint_sha256"] == PARENT_SHA256,
            "Comparison changed the frozen model")
    require(set(value["input_bindings"]["bank_artifact_sha256"]) == {"plan.json","qualification.json","audit.json","summary.json"},
            "Comparison bank identity incomplete")
    require(all(isinstance(sha,str) and re.fullmatch(r"[a-f0-9]{64}",sha)
                for sha in value["input_bindings"]["bank_artifact_sha256"].values()), "Invalid bank SHA256")
    for cohort_index, cohort in enumerate(value["cohorts"]):
        require(cohort["method"] == METHODS[cohort_index] and cohort["request_count"] == 35,
                "Comparison cohort order/budget changed")
        require(all(len(cohort[k]) == 35 for k in ("actions","native_enabled","decline_reasons","ik_seeds")),
                "Comparison cohort field coverage changed")
        for slot, cell in enumerate(cells):
            action = exact_action(cell, cohort["actions"][slot], "comparison")
            enabled, decline, seeds = cohort["native_enabled"][slot], cohort["decline_reasons"][slot], cohort["ik_seeds"][slot]
            require(type(enabled) is bool and (decline is None if enabled else isinstance(decline,str) and decline),
                    "Comparison native mask/decline reason disagree")
            if cohort_index == 0:
                require(enabled and decline is None and seeds is None, "Raw parent protocol changed")
            if seeds is not None:
                require(cohort_index == 1 and enabled and set(seeds) == set(ARMS), "Unexpected cached seed roles")
                for fit in seeds.values():
                    q = np.asarray(fit["q"],float)
                    require(fit["solved"] is True and q.shape == (6,) and np.isfinite(q).all()
                            and type(fit["residual"]) in (int,float) and 0 <= fit["residual"] < 1e-4,
                            "Selected endpoint warm start is invalid")
            request_index = cohort_index*35+slot
            request = value["requests"][request_index]
            require(request["request_index"] == request_index and request["cohort"] == cohort_index
                    and request["method"] == cohort["method"] and request["slot"] == slot
                    and request["cell_id"] == cell["id"] and request["action"] == action
                    and request["predeclined"] is (not enabled) and request["decline_reason"] == decline,
                    "Comparison request identity/action changed")
            require(request["candidate_id"] == (value["selections"][slot]["selected_candidate_id"] if cohort_index else None),
                    "Comparison selected-candidate identity changed")
    for slot,(cell, selection) in enumerate(zip(cells,value["selections"])):
        require(selection["cell_id"] == cell["id"] and type(selection["predeclined"]) is bool
                and selection["predeclined"] is (not value["cohorts"][1]["native_enabled"][slot]),
                "Comparison selection state changed")
        parent = value["cohorts"][0]["actions"][slot]
        chosen = value["cohorts"][1]["actions"][slot]
        same_action = np.array_equal(values_of(parent),values_of(chosen))
        require(selection["action_changed"] is (not same_action), "Comparison changed-action flag disagrees")
        if selection["predeclined"]:
            require(selection["selected_candidate_id"] is None and selection["qualified_candidates"] == 0
                    and selection["decline_reason"] == "no_qualified_candidate_support"
                    and same_action and selection["parent_preserved"] is False and selection["candidates"] == [],
                    "Empty support must remain an explicit declined placeholder")
            continue
        preference = selection["preference"]
        candidates = selection["candidates"]
        require(len(candidates) == selection["qualified_candidates"] > 0
                and len({c["candidate_id"] for c in candidates}) == len(candidates), "Selection probability support changed")
        require(len(preference["scores"]) == len(preference["probabilities"]) == len(preference["log_probabilities"]) == len(candidates),
                "Selection probability field coverage changed")
        scores, probabilities, logp = (np.asarray(preference[k],float) for k in ("scores","probabilities","log_probabilities"))
        require(np.isfinite(scores).all() and np.isfinite(logp).all() and np.isfinite(probabilities).all()
                and np.all(probabilities >= 0) and np.isclose(probabilities.sum(),1.,atol=1e-12,rtol=0)
                and np.allclose(np.exp(logp),probabilities,atol=1e-14,rtol=1e-12)
                and np.allclose(scores-float(preference["log_normalizer"]),logp,atol=1e-12,rtol=0)
                and preference["stochastic_sampling"] is False, "Invalid categorical preference law")
        index = preference["selected_index"]
        require(type(index) is int and 0 <= index < len(candidates)
                and candidates[index]["candidate_id"] == selection["selected_candidate_id"]
                and scores[index] >= scores.max()-1e-12, "Comparison did not select its declared argmax")
        for i,candidate in enumerate(candidates):
            require(candidate["score"] == preference["scores"][i]
                    and candidate["categorical_probability"] == preference["probabilities"][i]
                    and candidate["categorical_log_probability"] == preference["log_probabilities"][i],
                    "Selection candidate probability metadata changed")
        require(selection["parent_preserved"] is preference["parent_qualified"], "Parent preservation state changed")
        if selection["parent_preserved"]:
            require(same_action and value["cohorts"][1]["ik_seeds"][slot] is None,
                    "Preserved parent must keep the raw action and native IK protocol")
        else:
            require(not same_action and value["cohorts"][1]["ik_seeds"][slot] is not None,
                    "Changed bank action lacks its prepared native IK warm start")
    require(value["parent_preserved"] == sum(s["parent_preserved"] for s in value["selections"])
            and value["changed_bank_actions"] == sum(s["action_changed"] for s in value["selections"])
            and value["bank_predeclined"] == sum(s["predeclined"] for s in value["selections"]),
            "Comparison aggregate accounting changed")
    native = value.get("native_binding")
    if native is not None:
        require(isinstance(native,dict) and native["checkpoint_sha256"] == PARENT_SHA256
                and isinstance(native["checkpoint_path"],str) and Path(native["checkpoint_path"]).is_absolute(),
                "Invalid native checkpoint binding")
        require(re.fullmatch(r"[a-f0-9]{40}",native["source_commit"]) is not None,
                "Native producer needs an exact source commit")
        for key in ("source_archive_sha256","geometry_fingerprint","source_scene_sha256"):
            require(isinstance(native[key],str) and re.fullmatch(r"[a-f0-9]{64}",native[key]), "Invalid native source SHA: "+key)
        require(native["original_source"]["geometry_fingerprint"] == native["geometry_fingerprint"]
                and isinstance(native["original_source"]["commit"],str)
                and re.fullmatch(r"[a-f0-9]{40}",native["original_source"]["commit"])
                and isinstance(native["bank_preprocessing"],dict)
                and type(native["selection_wall_s"]) in (int,float) and math.isfinite(native["selection_wall_s"])
                and native["selection_wall_s"] >= 0, "Native source/preprocessing identity changed")
    return value
