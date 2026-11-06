"""Compare frozen broad/local pilots with a diverse-region, settled PPO pilot.

This exporter never starts physics or evaluates a policy. Every rejected active
request stays in the training denominator. Geometry preparation and changes to
the native setup controller are disclosed separately from policy learning.
"""

import argparse
import collections
import copy
import json
import math
from pathlib import Path

from continuous_pilot_report import export as native_export
from continuous_pilot_report import local_file, read, require, sha, summarize
from feasibility_guided_report import canonical_digest, compact_counts, layout_identity, stages

SCHEMA = "sircus-diverse-settled-comparison-v1"
STUDY_SCHEMA = "sircus-native-diverse-settled-pilot-v1"
SUPPORT_SCHEMA = "sircus-diverse-shot-support-v1"
DECODER = "geometry-only region choice + local affine residual-v1"
BASELINE_RESULT_SHA256 = "cf7e05910a22269330fc3ec89c10ed080ef0c5c67f3528b31c07e1b3735ca614"
GUIDED_RESULT_SHA256 = "cd90e6ec9c1e1bc3729ca5113b67d98c6ebf680f0964cec68065acb1f98138ca"
METHODS = ("broad", "guided", "diverse")
SPLITS = ("train", "validation", "test")


def compact_support(raw):
    """Keep every region's action definition, hashing omitted IK/probe records.

    This is a derived display receipt, never a replacement for the original
    fingerprinted runtime support contract.
    """
    unsigned = dict(raw)
    fingerprint = unsigned.pop("fingerprint", None)
    require(fingerprint == canonical_digest(unsigned), "Diverse support fingerprint mismatch")
    require(raw.get("schema") == SUPPORT_SCHEMA and raw.get("policy_geometry_only") is True,
            "Unknown or non-geometric diverse support")
    fields = ("schema", "source_fingerprint", "feasibility_fingerprint", "policy_geometry_only",
              "asset_sha256", "action_names", "global_low", "global_high", "base_half_width",
              "width_scales", "probe_count", "probe_seed", "minimum_probe_acceptance",
              "anchor_rank", "anchor_path_samples", "probe_solver", "seed_source", "elapsed_s",
              "probe_requests", "probe_accepted", "supported_cells", "total_regions", "scope",
              "selection", "max_regions", "max_regions_per_cell", "fixed_half_width", "region_order",
              "coverage", "candidate_requests", "region_count", "policy_decoder")
    out = {k: copy.deepcopy(raw[k]) for k in fields if k in raw}
    out.update(source_support_fingerprint=fingerprint, compact_receipt=True,
        omission_scope="Original IK joints, probe actions and geometry/path readbacks remain outside the report. Omitted fields are SHA256-bound. Region definitions are not capture witnesses or certified feasible interiors.")
    rows = []
    for cell in raw["cells"]:
        regions = cell.get("regions", [])
        require(cell.get("status") in ("supported", "unresolved")
                and (cell["status"] == "supported") == bool(regions),
                "Diverse support status/region count disagree")
        require(len({r["id"] for r in regions}) == len(regions), "Duplicate geometry region IDs")
        row = {k: copy.deepcopy(cell[k]) for k in ("cell_id", "cell_index", "bucket_xy_m", "status",
               "reason", "map_candidates", "candidate_regions", "rejected_regions", "coverage") if k in cell}
        row["regions"] = []
        for region in regions:
            low, high = region["low"], region["high"]
            require(len(low) == len(high) == 5 and all(type(x) in (int, float) and math.isfinite(x)
                    for x in low+high) and all(a < b for a,b in zip(low,high)),
                    "Invalid diverse region bounds")
            keep = {k: copy.deepcopy(region[k]) for k in ("id", "proposal", "low", "high", "anchor_action",
                    "probe_acceptance", "qualification_target_met", "selected_width_scale", "anchor_score",
                    "requests", "accepted", "map_proposal", "qualification_scope") if k in region}
            omitted = {k:v for k,v in region.items() if k not in keep}
            keep.update(omitted_fields=sorted(omitted), omitted_fields_sha256=canonical_digest(omitted))
            row["regions"].append(keep)
        omitted = {k:v for k,v in cell.items() if k not in row and k != "regions"}
        row.update(omitted_fields=sorted(omitted), omitted_fields_sha256=canonical_digest(omitted))
        row["region_count"] = len(regions)
        rows.append(row)
    out["cells"] = rows
    out["retained_region_count"] = sum(c["region_count"] for c in rows)
    out["retained_supported_cells"] = sum(c["status"] == "supported" for c in rows)
    return out


def method(report, label):
    training = compact_counts(report["training"])
    batches = report["study"].get("training_batches")
    expected = batches * 25 if type(batches) is int else None
    if report["status"] == "completed":
        require(expected == training["attempts"] == 1600,
                "Completed comparison requires the equal 1,600-attempt budget")
    return dict(label=label, status=report["status"], study_fingerprint=report["study_fingerprint"],
        source_result_sha256=report["source_sha256"].get("result.json"), training=training,
        planned_attempts=expected, elapsed_s=report["elapsed_s"],
        frozen={split:stages(report,split) for split in SPLITS},
        learning_curve=copy.deepcopy(report["learning_curve"]),
        training_failure_counts=copy.deepcopy(report["training"].get("failures", {})),
        setup_protocol=copy.deepcopy(report["study"].get("settling_protocol", report["study"].get("setup"))),
        audit_status={key:value["report"].get("status", "unknown")
                      for key,value in report.get("audits", {}).items()})


def comparison(diverse, broad, guided, *, baseline_report_sha256=None,
               guided_report_sha256=None, support=None):
    require(diverse["study"].get("schema") == STUDY_SCHEMA, "Expected a diverse-settled study")
    require(isinstance(diverse["study"].get("settling_protocol"), dict)
            and diverse["study"]["settling_protocol"], "Explicit native settling protocol required")
    for previous, expected, label in ((broad,BASELINE_RESULT_SHA256,"broad"),
                                      (guided,GUIDED_RESULT_SHA256,"guided")):
        require(previous["status"] == "completed", label+" baseline must stay completed and frozen")
        require(previous["source_sha256"].get("result.json") == expected,
                "Unexpected "+label+" baseline result")
        require(diverse["geometry_fingerprint"] == previous["geometry_fingerprint"],
                "Geometry fingerprint mismatch")
        require(layout_identity(diverse) == layout_identity(previous), "Layout or frozen split changed")
        for field in ("contract_sha256", "camera_sha256"):
            a,b = (r["study"].get("source",{}).get(field) for r in (diverse,previous))
            require(a is not None and a == b, "Comparison changed "+field)
        require(diverse["scoring"] == previous["scoring"], "Scoring definition changed")
    methods = dict(broad=method(broad,"Broad continuous domain"),
                   guided=method(guided,"One local geometry region"),
                   diverse=method(diverse,"Diverse regions + measured settling"))
    differences = {}
    for split in SPLITS:
        row = {}
        right = methods["diverse"]["frozen"][split]["final"]
        for key in ("broad","guided"):
            left = methods[key]["frozen"][split]["final"]
            delta = None
            if left is not None and right is not None:
                require(left["attempts"] == right["attempts"], "Frozen evaluation budgets differ")
                delta = 100*(right["capture_rate"]-left["capture_rate"])
            row["diverse_minus_"+key+"_percentage_points"] = delta
        differences[split] = row
    return dict(schema=SCHEMA,status=diverse["status"],methods=methods,final_policy_difference=differences,
        baseline_report_sha256=baseline_report_sha256, guided_report_sha256=guided_report_sha256,
        geometry_preparation=dict(original_sobol_requests=broad["feasibility"]["attempts"],
            original_map_admitted=broad["feasibility"]["admitted"],support=support,
            cost_scope="Geometry preprocessing is additional to the matched native shot budget. Native elapsed includes setup, filming and evaluation; this is not a matched headless compute benchmark."),
        settling_protocol=copy.deepcopy(diverse["study"]["settling_protocol"]),
        scientific_scope="Single-seed method comparison on inherited 25/5/5 coordinates, held out from optimizer reward updates, not globally unseen layouts. Geometry at held-out coordinates assists decoding. Identical nominal repeats measure numerical repeatability, not independent scene robustness. Proposal support, hybrid policy and native setup controller change together; this does not isolate PPO or certify table-wide performance.",
        action_scope="A categorical choice among geometry-qualified regions and a continuous residual select the shot. Requested actions still receive fresh IK/FK/path proxies and measured native release gates. Region bounds do not certify every interior action; unresolved sites are not proven impossible. Arm travel and initialized ball release remain scripted, not learned motor/grasp control.",
        inference_scope="Frozen deterministic inference uses its own saved checkpoint without per-scene capture search or held-out outcome selection.")


def verify_comparison_audit(receipt, value):
    """Bind a separately computed three-method count receipt to all raw results."""
    if "comparison" in receipt:
        require(receipt.get("status") == "passed" and receipt.get("source_result_sha256")
                == value["methods"]["diverse"]["source_result_sha256"],
                "Containing audit is not passed or belongs to another diverse result")
        receipt = receipt["comparison"]
    require(receipt.get("schema") == SCHEMA and receipt.get("status") == "completed",
            "Independent comparison incomplete or different schema")
    for key in METHODS:
        left, right = receipt["methods"][key], value["methods"][key]
        require(left.get("source_result_sha256") == right["source_result_sha256"],
                "Independent comparison belongs to different result bytes")
        def equal(a,b):
            require(a is not None and b is not None and all(a[k] == b[k]
                    for k in ("attempts","captures","clean","rejected")),
                    "Independent comparison count mismatch")
        equal(left["training"],right["training"])
        for split in SPLITS:
            for stage in ("initial","final"):
                equal(left["frozen"][split][stage],right["frozen"][split][stage])
    return receipt


def region_usage(run, support):
    """Count logged categorical requests, not certified motion or successful regions."""
    run = Path(run)
    cells = read(local_file(run,"study.json"))["cells"]
    known = {c["cell_id"]:c for c in support["cells"]}
    choices = {c["id"]:collections.Counter() for c in cells}
    batches, attempts, missing = 0,0,0
    for path in sorted((run/"trials").glob("*.json")):
        trial = read(local_file(run,"trials/"+path.name))
        if trial.get("mode") != "train":continue
        batches += 1
        indices = trial.get("region_indices")
        active = trial.get("active_cell_ids", [])
        if indices is None:
            missing += 1
            continue
        require(len(indices) == len(cells) and len(active) == len(set(active))
                and all(name in choices for name in active), "Invalid logged region-choice alignment")
        for cell, index in zip(cells,indices):
            if cell["id"] not in active:continue
            require(cell["split"] == "train", "Held-out region choice entered optimizer accounting")
            require(type(index) is int, "Region choice must be an integer")
            supported = known[cell["id"]]
            if supported["status"] == "supported":
                require(index in {r["id"] for r in supported["regions"]}, "Unknown requested geometry region")
            choices[cell["id"]][str(index)] += 1
            attempts += 1
    require(not missing or missing == batches, "Partially missing region-choice logs")
    return dict(status="not_logged" if missing else "logged",batches=batches,logged_training_attempts=attempts,
        cells=[dict(cell_id=c["id"],split=c["split"],available_regions=len(known[c["id"]]["regions"]),
            attempted_region_counts=dict(choices[c["id"]]),sampled_regions=len(choices[c["id"]]) if known[c["id"]]["status"] == "supported" else 0)
            for c in cells],
        scope="Active training categorical requests only, including rejected shots. These are region selections, not executed pose coverage or capture evidence. Evaluation and geometry coverage excluded.")


def add_comparison(summary, broad_path, guided_path, run, *, support_cost_path=None):
    run = Path(run)
    support_path = local_file(run,"support.json")
    raw = read(support_path)
    study = summary["study"]
    require(study.get("support_sha256") == sha(support_path)
            and study.get("support_fingerprint") == raw["fingerprint"]
            and study.get("domain",{}).get("support_fingerprint") == raw["fingerprint"],
            "Study support bytes or fingerprint changed")
    require(study.get("domain",{}).get("decoder") == DECODER, "Unknown diverse action decoder")
    require(study.get("feasibility_sha256") == summary["source_sha256"].get("feasibility.json")
            and raw.get("source_fingerprint") == summary["geometry_fingerprint"],
            "Support or map belongs to different geometry")
    expected_cells = {c["id"]:c["bucket_xy_m"] for c in summary["cells"]}
    support_cells = raw.get("cells", [])
    require(len(support_cells) == len(expected_cells)
            and {c["cell_id"]:c["bucket_xy_m"] for c in support_cells} == expected_cells,
            "Support cell membership or bucket coordinates differ from the native study")
    support = compact_support(raw)
    support["source_support_sha256"] = sha(support_path)
    summary["source_sha256"]["support.json"] = sha(support_path)
    c = comparison(summary,read(broad_path),read(guided_path),
        baseline_report_sha256=sha(broad_path),guided_report_sha256=sha(guided_path),support=support)
    c["region_usage"] = region_usage(run,raw)
    if c["region_usage"]["status"] == "logged":
        require(c["region_usage"]["logged_training_attempts"] == summary["training"]["attempts"],
                "Region-choice accounting differs from active training denominator")
    if support_cost_path is not None:
        cost = read(support_cost_path)
        require(cost.get("support_sha256") == sha(support_path)
                and cost.get("support_fingerprint") == raw["fingerprint"],
                "Resource receipt belongs to another support")
        c["geometry_preparation"]["process_resource_receipt"] = dict(sha256=sha(support_cost_path),report=cost)
    path = run/"comparison-audit.json"
    if path.is_file():
        path = local_file(run,"comparison-audit.json")
        receipt = read(path)
        verified = verify_comparison_audit(receipt,c)
        c["independent_audit"] = dict(sha256=sha(path),report=verified)
        summary["source_sha256"]["comparison-audit.json"] = sha(path)
    else:
        c["independent_audit"] = dict(status="not_attached")
    summary["comparison"] = c
    summary["exporter_wrapper_sha256"] = sha(__file__)
    return summary


def export(run, output, broad_report, guided_report, *, media=False,action_edit=False,support_cost_path=None):
    add_comparison(summarize(run),broad_report,guided_report,run,support_cost_path=support_cost_path)
    summary = native_export(run,output,media=media,action_edit=action_edit)
    add_comparison(summary,broad_report,guided_report,run,support_cost_path=support_cost_path)
    (Path(output)/"report.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    return summary


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run",type=Path)
    parser.add_argument("output",type=Path)
    parser.add_argument("--broad-report",type=Path,required=True)
    parser.add_argument("--guided-report",type=Path,required=True)
    parser.add_argument("--support-cost",type=Path)
    parser.add_argument("--media",action="store_true")
    parser.add_argument("--action-edit",action="store_true")
    args=parser.parse_args()
    value=export(args.run,args.output,args.broad_report,args.guided_report,media=args.media,
                 action_edit=args.action_edit,support_cost_path=args.support_cost)
    print(json.dumps(dict(status=value["status"],training=value["training"],
        comparison=value["comparison"]["final_policy_difference"],movie=value["movie"]),allow_nan=False))
