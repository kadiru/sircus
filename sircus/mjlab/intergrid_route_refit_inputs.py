"""Bind the frozen models, audited paired data and disjoint refit evaluation layouts."""

from __future__ import annotations

import json
from pathlib import Path

from .benchmark_verify import digest, read, sha
from .intergrid_matched_route_plan import make_layouts as make_matched_layouts
from .intergrid_matched_route_plan import sample_layouts as sample_matched_layouts
from .intergrid_matched_route_study import VERSION as MATCHED_VERSION, load_model_inputs
from .intergrid_paired_collection_study import VERSION as PAIRED_VERSION, collection_plan, collection_rows
from .intergrid_route_labels import LABEL_VERSION
from .intergrid_route_plan import ROUTES
from .intergrid_route_study import VERSION as MODEL_VERSION


PLAN_VERSION = "sircus-intergrid-route-refit-plan-v1"

_MATCHED_FILES = ("rows.jsonl", "choices.json", "layouts.json", "route-models.json", "training-prior.json")
_PAIRED_FILES = ("rows.jsonl", "collection-plan.json", "training-prior.json", "admitted.json",
                 "new-training-dataset.jsonl")


def _contract(study, version):
    contract = read(study / "contract.json")
    if (contract["version"], contract["physics_substeps"], contract["label_version"]) != (
        version, 64, LABEL_VERSION,
    ):
        raise ValueError(f"{study.name} uses a different study or native scoring contract")
    if digest({key: value for key, value in contract.items() if key != "fingerprint"}) != contract["fingerprint"]:
        raise ValueError(f"{study.name} contract fingerprint differs")
    return contract


def _audited_source(study, audit_path, version, files):
    """Check source bytes against the report and the independent audit receipts."""
    report, manifest, audit = (read(study / "report.json"), read(study / "evidence-manifest.json"),
                               read(audit_path))
    binding = {name: sha(study / name)
               for name in ("report.json", "contract.json", "evidence-manifest.json", *files)}
    binding["audit"] = sha(audit_path)
    if (
        report["status"] != "completed" or report["phase"] != "completed" or audit["status"] != "passed"
        or type(audit["checks_count"]) is not int or audit["checks_count"] <= 0
        or audit["report_sha256"] != binding["report.json"]
        or audit["evidence_manifest_sha256"] != binding["evidence-manifest.json"]
        or report["evidence_manifest_sha256"] != binding["evidence-manifest.json"]
        or any(manifest[name] != binding[name] for name in ("contract.json", *files))
    ):
        raise ValueError(f"{study.name} differs from its independent audit")
    return report, manifest, audit, binding, _contract(study, version)


def _check_new_rows(rows, positions, source):
    for row in rows:
        label = row["shot_route"]
        rebounds = row["table_rebounds_before_entry"]
        if (
            row["phase"] != "train" or row["source_study"] != source
            or positions[row["context_id"]] != row["bucket_xy_m"]
            or row["intended_route"] not in ROUTES
            or row["route_success"] is not row["standard_capture"]
            or (not row["route_success"] and (label is not None or rebounds is not None))
            or (row["route_success"] and (
                label not in (*ROUTES, None) or type(rebounds) is not int
                or (label == "table" and rebounds < 1) or (label != "table" and rebounds != 0)
            ))
        ):
            raise ValueError(f"Invalid paired training record: {row['id']}")


def load_inputs(history_root, model_study, model_audit, paired_study, paired_audit, matched_study, matched_audit):
    """Load the original training rows and the audited new native releases only."""
    if not Path(history_root).is_dir():
        raise ValueError("The historical placement archive is required")
    model_study, model_audit, paired_study, paired_audit, matched_study, matched_audit = map(
        Path, (model_study, model_audit, paired_study, paired_audit, matched_study, matched_audit)
    )
    checkpoint, original, _, previous, model_binding = load_model_inputs(model_study, model_audit)
    model_contract = _contract(model_study, MODEL_VERSION)
    if sha(model_study / "contract.json") != read(model_study / "evidence-manifest.json")["contract.json"]:
        raise ValueError("The original model contract differs from its manifest")

    matched_report, _, matched_receipt, matched_binding, matched_contract = _audited_source(
        matched_study, matched_audit, MATCHED_VERSION, _MATCHED_FILES
    )
    if (
        matched_contract["model_study"] != model_binding
        or matched_binding["route-models.json"] != matched_receipt["checkpoint_sha256"]
        or matched_binding["route-models.json"] != model_binding["checkpoint"]
        or matched_binding["training-prior.json"] != model_binding["geometry_prior"]
        or matched_binding["choices.json"] != matched_report["choices_sha256"]
        or matched_binding["choices.json"] != matched_receipt["choices_sha256"]
        or matched_receipt["comparison"] != matched_report["comparison"]
    ):
        raise ValueError("The matched evaluation does not bind the frozen original model")

    paired_report, paired_manifest, paired_receipt, paired_binding, paired_contract = _audited_source(
        paired_study, paired_audit, PAIRED_VERSION, _PAIRED_FILES
    )
    for key in ("parent", "identity", "scene"):
        if not model_contract[key] == matched_contract[key] == paired_contract[key]:
            raise ValueError(f"Source simulation contracts differ at {key}")

    expected_plan, source_binding = collection_plan(
        original, matched_study, matched_audit, model_binding, model_contract["parent"]
    )
    if (
        paired_contract["model_study"] != model_binding or paired_contract["matched_study"] != source_binding
        or paired_contract["frozen_inputs"] != {name: paired_binding[name]
                                                for name in ("collection-plan.json", "training-prior.json")}
        or paired_binding["training-prior.json"] != model_binding["geometry_prior"]
        or paired_binding["admitted.json"] != paired_report["admitted_sha256"]
        or paired_binding["new-training-dataset.jsonl"] != paired_receipt["training_dataset_sha256"]
        or paired_report["collection"] != paired_receipt["collection"]
        or read(paired_study / "collection-plan.json") != expected_plan
    ):
        raise ValueError("The paired collection does not match its audited source plan")

    # The matched evaluation supplied coordinates and planning seeds. Its outcomes
    # remain in their original ledger and are never copied into the refit rows.
    new, excluded = collection_rows(paired_study, expected_plan, paired_manifest)
    with (paired_study / "new-training-dataset.jsonl").open() as stream:
        exported = [json.loads(line) for line in stream]
    if (new != exported or len(new) != paired_receipt["valid_training_rows"]
            or len(new) != paired_report["valid_training_rows"]
            or excluded != paired_receipt["excluded_from_dataset"]
            or excluded != paired_report["excluded_from_dataset"]):
        raise ValueError("Paired training rows do not reproduce the audited native ledger")
    paired_layouts = [item["layout"] for item in expected_plan]
    _check_new_rows(new, {layout["id"]: layout["bucket_xy_m"] for layout in paired_layouts}, paired_study.name)
    if (len({row["id"] for row in (*original, *new)}) != len(original) + len(new)
            or any(lineage["source_study"] == matched_study.name
                   for row in (*original, *new) for lineage in row["lineage"])):
        raise ValueError("Duplicate shot identity or matched evaluation row in the training data")

    matched_layouts = read(matched_study / "layouts.json")["layouts"]
    binding = dict(model_study=model_binding, paired_study=paired_binding, matched_study=matched_binding)
    return checkpoint, original, new, previous + matched_layouts + paired_layouts, binding


def _refit_ids(layouts):
    renamed = []
    for index, layout in enumerate(layouts):
        identity = f"intergrid-route-refit-test-{index:02d}"
        renamed.append(dict(layout, id=identity, context_id=identity))
    return renamed


def sample_layouts(excluded, geometry, grid=None, *, seed):
    """Recreate the frozen layouts from their archived spatial exclusions."""
    return _refit_ids(sample_matched_layouts(excluded, geometry, grid, seed=seed))


def fresh_layouts(history_root, grid_path, scene, assets, previous_layouts, seed):
    """Freeze new test positions after excluding every previously observed position."""
    layouts, provenance = make_matched_layouts(
        history_root, read(grid_path), scene=scene, assets=assets, previous_layouts=previous_layouts, seed=seed
    )
    layouts = _refit_ids(layouts)
    provenance.update(version=PLAN_VERSION, layouts_sha256=digest(layouts))
    return layouts, provenance
