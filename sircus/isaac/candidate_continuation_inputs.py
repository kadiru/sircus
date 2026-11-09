"""Hash-bound TRAIN inputs for the no-update continuation adapter.

Exporting verifies the full approved bank and existing successful TRAIN evidence.
The compact export is not a training/native launch configuration or release permit.
"""

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from candidate_continuation_policy import FrozenCandidateSupport, freeze_support
from candidate_selector import load_bank, build_comparison, PARENT_SHA256
from continuous_feasibility import canonical_hash
from feasible_candidates import plain, train_cells, unsigned, values_of

SCHEMA = "sircus-candidate-continuation-inputs-v1"
STUDY_SHA = "135a05ba68295debdec7953703e3f027543bb04b53ae3e8bdec5042c9b67af1a"
GEOMETRY_SHA = "63362ef37a93881f769b39d1ee7456335f1cc7bf5db04ffa7e548ce9697455bf"
DOMAIN_SHA = "28937cf485815c0bd0ac6e0b8e0266e21d3b52efdeb7c35544c56b21b20855ae"
SOURCE = Path(__file__).resolve().parent
PUBLIC_SUMMARY = SOURCE.parents[1]/"baselines/isaac-feasible-candidates-20261109/summary.json"
MODULES = ("candidate_continuation_policy.py", "candidate_continuation_losses.py",
           "candidate_continuation_ledger.py", "candidate_continuation_inputs.py",
           "candidate_continuation_readiness.py", "continuous_policy.py",
           "candidate_selector.py", "continuous_feasibility.py", "feasible_candidates.py",
           "cylinder_farm.py")
ENCODING_CHECK_ATOL = 1e-12


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes(), parse_constant=lambda x: (_ for _ in ()).throw(
        ValueError("Nonfinite input JSON: "+x)))


def source_hashes():
    return {name:digest(SOURCE/name) for name in MODULES}


def checked_support(context_id, parent, raw):
    """Keep signed stored latents exact; tolerate only inverse-math roundoff.

    NumPy/libm versions can differ in the last binary digit of atanh. This
    consistency check never replaces or re-signs the exported latent values.
    The exact parent's original saved latent is checked without tolerance.
    """
    support = FrozenCandidateSupport.from_dict(raw)
    require(support.context_id == context_id, "Support/context order changed")
    rows = [dict(candidate_id=i, action=a) for i,a in zip(support.candidate_ids, support.actions)]
    expected = freeze_support(context_id, rows, parent["exact_action"], parent["latent_mean"])
    require(support.candidate_ids == expected.candidate_ids and support.actions == expected.actions
            and support.parent_index == expected.parent_index, "Fixed candidate action/order changed")
    require(np.allclose(np.asarray(support.latents), np.asarray(expected.latents),
                        atol=ENCODING_CHECK_ATOL, rtol=0), "Fixed candidate encoding changed")
    if support.parent_index is not None:
        require(support.latents[support.parent_index] == tuple(parent["latent_mean"]),
                "Exact original parent latent changed")
    return support


def teacher_choices(evidence, cells, supports, records):
    """Join exact successful TRAIN requests to their admitted bank choices.

    This generic join has no policy inference or optimizer. The exporter and
    compact validator additionally enforce the production 1534/485/21 counts.
    """
    unsigned(evidence)
    by_cell = {c["id"]:c for c in train_cells(cells)}
    require({c["id"]:c for c in train_cells(evidence["cells"])} == by_cell,
            "Demonstration TRAIN coordinates changed")
    support_by_cell = {s.context_id:s for s in supports}
    require(set(support_by_cell) == set(by_cell), "Demonstration support coverage changed")
    aliases = {}
    for record in records:
        for alias in record["sources"]:
            if alias["kind"] == "historical_train_success":
                key = alias["case_id"]
                require(key not in aliases, "Duplicate historical bank witness")
                aliases[key] = record
    demonstrations, keys, pairs = [], set(), set()
    for success in evidence["successes"]:
        cell, key = success["cell"], success["success_key"]
        require(key not in keys and cell["id"] in by_cell and cell == by_cell[cell["id"]],
                "Duplicate, changed or non-TRAIN demonstration")
        require(type(success["clean"]) is bool, "Invalid clean demonstration label")
        require(key in aliases, "Successful TRAIN request missing bank provenance")
        record = aliases[key]
        require(record["cell_id"] == cell["id"] and record["action"] == success["action"],
                "Historical demonstration action changed")
        support = support_by_cell[cell["id"]]
        require(record["candidate_id"] in support.candidate_ids,
                "Successful TRAIN demonstration was excluded from support")
        index = support.candidate_ids.index(record["candidate_id"])
        require(support.action(index) == success["action"], "Demonstration maps to a different action")
        pair = (cell["id"], record["candidate_id"])
        require(pair not in pairs, "Repeated physical success needs an explicit new weighting protocol")
        keys.add(key)
        pairs.add(pair)
        demonstrations.append(dict(context_id=cell["id"], candidate_id=record["candidate_id"],
            index=index, success_key=key, clean=success["clean"]))
    require(len(aliases) == len(keys) and set(aliases) == keys,
            "Historical bank/demo witness membership differs")
    return demonstrations


def export_inputs(bank_directory, train_evidence_path):
    """Verify full source evidence once and return a portable no-update export."""
    plan, qualification, audit, summary, hashes = load_bank(bank_directory)
    comparison = build_comparison(plan, qualification, audit, summary, artifact_sha256=hashes)
    expected_evidence = summary["input_bindings"]["train_evidence_sha256"]
    require(digest(train_evidence_path) == expected_evidence, "Successful TRAIN evidence bytes changed")
    evidence = read(train_evidence_path)
    require(evidence["schema"] == "sircus-seed22-paid-train-evidence-v1"
            and evidence["seed"] == 20261122
            and evidence["source_bindings"] == summary["input_bindings"]["original"]
            and evidence["new_native_shots"] == evidence["optimizer_updates"]
                == evidence["evaluated_trial_files_read"] == 0,
            "Successful TRAIN evidence lineage changed")
    qualified = {cell["id"]:[] for cell in plan["cells"]}
    for row in qualification["records"]:
        if row["accepted"]:
            qualified[row["cell_id"]].append(row)
    supports = []
    for cell in plan["cells"]:
        parent = plan["parent_distributions"][cell["id"]]
        supports.append(freeze_support(cell["id"], sorted(qualified[cell["id"]],
            key=lambda r:r["candidate_id"]), parent["exact_action"], parent["latent_mean"]))
    demonstrations = teacher_choices(evidence, plan["cells"], supports, plan["records"])
    value = dict(schema=SCHEMA, cells=copy.deepcopy(plan["cells"]),
        parent_distributions=copy.deepcopy(plan["parent_distributions"]),
        supports=[s.to_dict() for s in supports], demonstrations=demonstrations,
        successes_by_context=dict(Counter(d["context_id"] for d in demonstrations)),
        reference_choices=[s["selected_candidate_id"] for s in comparison["selections"]],
        input_bindings=dict(bank_artifact_sha256=hashes, train_evidence_sha256=expected_evidence,
            original=copy.deepcopy(summary["input_bindings"]["original"]),
            bank_plan_fingerprint=plan["fingerprint"], bank_qualification_fingerprint=qualification["fingerprint"],
            bank_audit_fingerprint=audit["fingerprint"], checkpoint_sha256=PARENT_SHA256,
            study_fingerprint=STUDY_SHA, geometry_fingerprint=GEOMETRY_SHA, domain_fingerprint=DOMAIN_SHA),
        source_sha256=source_hashes(), optimizer_updates=0, native_requests=0,
        held_out_shot_files_read=0, release_authorized=False,
        scope="Verified existing TRAIN bank/demo export only; no new proposals, model inference, optimizer or release.")
    value["fingerprint"] = canonical_hash(plain(value))
    validate_inputs(value)
    return value


def validate_inputs(value):
    """Validate a compact export; callers must also bind its actual file SHA."""
    unsigned(value)
    require(value.get("schema") == SCHEMA and value.get("source_sha256") == source_hashes(),
            "Adapter input schema/source identity changed")
    require(value.get("optimizer_updates") == value.get("native_requests")
            == value.get("held_out_shot_files_read") == 0 and value.get("release_authorized") is False,
            "Adapter export cannot authorize training or release")
    published = read(PUBLIC_SUMMARY)
    bindings = value["input_bindings"]
    require(bindings["original"] == published["input_bindings"]["original"]
            and bindings["checkpoint_sha256"] == PARENT_SHA256
            and bindings["train_evidence_sha256"] == published["input_bindings"]["train_evidence_sha256"]
            and bindings["study_fingerprint"] == STUDY_SHA
            and bindings["geometry_fingerprint"] == GEOMETRY_SHA and bindings["domain_fingerprint"] == DOMAIN_SHA,
            "Adapter parent/evidence identity changed")
    require(bindings["bank_artifact_sha256"]["summary.json"] == digest(PUBLIC_SUMMARY)
            and all(bindings["bank_artifact_sha256"][name] == published["artifacts_sha256"][name]
                    for name in ("plan.json", "qualification.json", "audit.json")), "Bank file identities changed")
    cells = train_cells(value["cells"])
    require(len(cells) == 35 and len(value["supports"]) == 35
            and set(value["parent_distributions"]) == {c["id"] for c in cells}, "Expected 35 original TRAIN contexts")
    supports = []
    for cell, raw in zip(cells, value["supports"]):
        parent = value["parent_distributions"][cell["id"]]
        support = checked_support(cell["id"], parent, raw)
        for identifier, action in zip(support.candidate_ids, support.actions):
            require(identifier == canonical_hash(dict(cell_id=cell["id"], values=values_of(action).tolist())),
                    "Candidate physical identity changed")
        supports.append(support)
    require(sum(len(s.candidate_ids) for s in supports) == 1814, "Expected complete 1814-candidate bank")
    require(len(value["reference_choices"]) == 35 and all(choice in support.candidate_ids
            for choice,support in zip(value["reference_choices"], supports)), "Reference choices changed")
    require(len(value["demonstrations"]) == 1534
            and len({d["success_key"] for d in value["demonstrations"]}) == 1534
            and len({(d["context_id"], d["candidate_id"]) for d in value["demonstrations"]}) == 1534
            and all(type(d["clean"]) is bool for d in value["demonstrations"])
            and sum(d["clean"] for d in value["demonstrations"]) == 485,
            "Expected all 1534 successful TRAIN examples including 485 clean")
    counts = Counter(d["context_id"] for d in value["demonstrations"])
    require(len(counts) == 21 and dict(counts) == value["successes_by_context"], "Success context weighting changed")
    lookup = {s.context_id:s for s in supports}
    for demonstration in value["demonstrations"]:
        require(demonstration["context_id"] in lookup, "Non-TRAIN demonstration context")
        support = lookup[demonstration["context_id"]]
        index = demonstration["index"]
        require(type(index) is int and 0 <= index < len(support.candidate_ids)
                and support.candidate_ids[index] == demonstration["candidate_id"], "Demonstration index/action changed")
    return tuple(supports)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--train-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Refuse to replace an adapter input export")
    value = export_inputs(args.bank, args.train_evidence)
    with args.output.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(status="passed", contexts=35, candidates=1814,
        demonstrations=1534, success_contexts=21, optimizer_updates=0, native_requests=0,
        output_sha256=digest(args.output))))


if __name__ == "__main__":
    main()
