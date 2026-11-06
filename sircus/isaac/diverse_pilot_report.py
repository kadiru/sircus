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
SUPPORT_SCHEMA = "sircus-diverse-shot-support-v2"
DECODER = "geometry-only region choice + affine residual-v2"
GLOBAL_QUALIFICATION_SCOPE = "Full original action domain is unqualified. First geometry anchor supplies only an IK warm start; every requested action still needs fresh IK/path/settlement gates."
BASELINE_RESULT_SHA256 = "cf7e05910a22269330fc3ec89c10ed080ef0c5c67f3528b31c07e1b3735ca614"
GUIDED_RESULT_SHA256 = "cd90e6ec9c1e1bc3729ca5113b67d98c6ebf680f0964cec68065acb1f98138ca"
METHODS = ("broad", "guided", "diverse")
SPLITS = ("train", "validation", "test")
SETUP_CHECK_SCHEMA = "sircus-diverse-settling-setup-check-v1"


def setup_check_receipt(value, summary):
    """Check a compact setup-only ablation, never treating admission as capture."""
    study = summary["study"]
    require(value.get("schema") == SETUP_CHECK_SCHEMA and value.get("status") == "completed",
            "Setup check is incomplete or uses an unknown schema")
    require(value.get("source_guided_result_sha256") == GUIDED_RESULT_SHA256
            and value.get("geometry_fingerprint") == summary["geometry_fingerprint"]
            and value.get("support_fingerprint") == study["support_fingerprint"]
            and value.get("support_sha256") == study["support_sha256"],
            "Setup check belongs to different frozen requests, geometry or support")
    require(value.get("settling_protocol") == study["settling_protocol"],
            "Setup check used a different native settling protocol")
    fields = ("attempts", "before_admitted", "after_admitted", "repaired", "regressions", "ball_releases")
    require(all(type(value.get(k)) is int and value[k] >= 0 for k in fields)
            and value["attempts"] == 100 and value["ball_releases"] == 0,
            "Setup check requires 100 fixed requests and zero ball releases")
    n,before,after,repaired,regressions = (value[k] for k in fields[:-1])
    require(before <= n and after <= n and repaired <= n-before and regressions <= before
            and after-before == repaired-regressions, "Inconsistent paired setup counts")
    for name, admitted in (("before_failures",before),("after_failures",after)):
        counts = value.get(name)
        require(isinstance(counts,dict) and all(isinstance(k,str) and type(v) is int and v > 0
                for k,v in counts.items()) and sum(counts.values()) == n-admitted,
                "Inconsistent setup failure denominator")
    private_sha = value.get("source_setup_check_sha256")
    require(isinstance(private_sha,str) and len(private_sha) == 64
            and all(c in "0123456789abcdef" for c in private_sha),
            "Missing private setup-check source SHA256")
    # Native paired/joint/physics histories do not enter this public summary.
    keep = ("schema", "status", "source_guided_result_sha256", "geometry_fingerprint",
            "support_fingerprint", "support_sha256", "source_setup_check_sha256",
            "attempts", "before_admitted", "after_admitted", "repaired", "regressions",
            "before_failures", "after_failures", "ball_releases", "settling_protocol",
            "paired_summary", "selection", "scope")
    return {k:copy.deepcopy(value[k]) for k in keep if k in value}


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
              "coverage", "local_coverage", "local_region_count", "global_region_count",
              "candidate_requests", "region_count", "policy_decoder")
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
        locals_ = [r for r in regions if r.get("kind") == "local_geometry"]
        globals_ = [r for r in regions if r.get("kind") == "global_exploration"]
        require(len(locals_)+len(globals_) == len(regions)
                and len(globals_) == int(bool(regions)) and bool(locals_) == bool(regions)
                and cell.get("local_region_count") == len(locals_)
                and cell.get("global_region_count") == len(globals_),
                "Local anchor regions and unqualified global domains must be counted separately")
        require(all(type(r.get("proposal")) is int and r["proposal"] >= 0 for r in locals_),
                "A local geometry anchor cannot be an unqualified global domain")
        if globals_:
            escape=globals_[0]
            require(regions[-1] is escape and escape.get("proposal") == -1
                    and escape["low"] == raw["global_low"] and escape["high"] == raw["global_high"]
                    and escape.get("anchor_action") == locals_[0].get("anchor_action")
                    and escape.get("qualification_scope") == GLOBAL_QUALIFICATION_SCOPE,
                    "Global exploration must remain the unqualified full original domain")
        row = {k: copy.deepcopy(cell[k]) for k in ("cell_id", "cell_index", "bucket_xy_m", "status",
               "reason", "map_candidates", "candidate_regions", "rejected_regions", "coverage",
               "local_coverage", "local_region_count", "global_region_count") if k in cell}
        row["regions"] = []
        for region in regions:
            low, high = region["low"], region["high"]
            require(len(low) == len(high) == 5 and all(type(x) in (int, float) and math.isfinite(x)
                    for x in low+high) and all(a < b for a,b in zip(low,high)),
                    "Invalid diverse region bounds")
            require(region.get("interior_qualified") is False,
                    "Neither local interiors nor global exploration may be labeled qualified")
            keep = {k: copy.deepcopy(region[k]) for k in ("id", "kind", "proposal", "low", "high", "anchor_action",
                    "probe_acceptance", "qualification_target_met", "selected_width_scale", "anchor_score",
                    "requests", "accepted", "map_proposal", "qualification_scope", "interior_qualified") if k in region}
            keep["qualification_state"] = ("qualified_anchor_unqualified_local_interior"
                if region["kind"] == "local_geometry" else "unqualified_global_exploration")
            omitted = {k:v for k,v in region.items() if k not in keep}
            keep.update(omitted_fields=sorted(omitted), omitted_fields_sha256=canonical_digest(omitted))
            row["regions"].append(keep)
        omitted = {k:v for k,v in cell.items() if k not in row and k != "regions"}
        row.update(omitted_fields=sorted(omitted), omitted_fields_sha256=canonical_digest(omitted))
        row["region_count"] = len(regions)
        rows.append(row)
    out["cells"] = rows
    out["retained_region_count"] = sum(c["region_count"] for c in rows)
    out["retained_local_region_count"] = sum(c["local_region_count"] for c in rows)
    out["retained_global_region_count"] = sum(c["global_region_count"] for c in rows)
    out["retained_supported_cells"] = sum(c["status"] == "supported" for c in rows)
    require(raw.get("local_region_count") == out["retained_local_region_count"]
            and raw.get("global_region_count") == out["retained_global_region_count"]
            and raw.get("region_count") == out["retained_region_count"],
            "Diverse local/global aggregate counts changed")
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
                   diverse=method(diverse,"Local + global regions, measured settling"))
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
        action_scope="A categorical choice selects a local region around a qualified geometry anchor or a full original-domain exploration region; a continuous residual selects the requested shot inside it. All region interiors are unqualified. Requested actions still receive fresh IK/FK/path proxies and measured native release gates. Global exploration is not an extra qualified anchor, and unresolved sites are not proven impossible. Arm travel and initialized ball release remain scripted, not learned motor/grasp control.",
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
            attempted_region_counts=dict(choices[c["id"]]),sampled_regions=len(choices[c["id"]]) if known[c["id"]]["status"] == "supported" else 0,
            sampled_local_regions=sum(str(r["id"]) in choices[c["id"]] for r in known[c["id"]]["regions"] if r["kind"] == "local_geometry"),
            local_region_attempts=sum(choices[c["id"]][str(r["id"])] for r in known[c["id"]]["regions"] if r["kind"] == "local_geometry"),
            global_exploration_attempts=sum(choices[c["id"]][str(r["id"])] for r in known[c["id"]]["regions"] if r["kind"] == "global_exploration"))
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
    setup_path = run/"setup-check.json"
    if setup_path.is_file():
        setup_path = local_file(run,"setup-check.json")
        setup = setup_check_receipt(read(setup_path),summary)
        c["setup_check"] = dict(sha256=sha(setup_path),report=setup,
            scope="Paired setup-admission check of the first four frozen prior training batches, before learning. Zero ball releases; not captures, PPO data or held-out evaluation. Full per-request/native histories stay outside the report.")
        summary["source_sha256"]["setup-check.json"] = sha(setup_path)
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
