"""Bind the training releases, frozen refit and disjoint library comparison layouts."""

from __future__ import annotations

import json
from pathlib import Path

from .benchmark_verify import digest, read
from .intergrid_matched_route_plan import make_layouts as make_matched_layouts
from .intergrid_matched_route_plan import sample_layouts as sample_matched_layouts
from .intergrid_route_models import load_checkpoint
from .intergrid_route_refit_audit import VERSION as REFIT_AUDIT_VERSION
from .intergrid_route_refit_inputs import PLAN_VERSION as REFIT_PLAN_VERSION, _audited_source, load_inputs
from .intergrid_route_refit_study import POSITIONS as REFIT_POSITIONS, SPLIT_SEED as REFIT_SPLIT_SEED
from .intergrid_route_refit_study import VERSION as REFIT_VERSION, training_summary


PLAN_VERSION = "sircus-intergrid-table-library-plan-v1"
SPLIT_SEED = 2026112001

_REFIT_INPUTS = ("layouts.json", "training-prior.json", "training-dataset.jsonl")
_REFIT_FILES = (*_REFIT_INPUTS, "route-models-original.json", "route-models.json", "choices.json")


def load_library_inputs(
    history_root, model_study, model_audit, paired_study, paired_audit, matched_study, matched_audit,
    refit_study, refit_audit,
):
    """Authenticate the refit and return only the training rows used to fit it."""
    original_checkpoint, original_rows, new_rows, previous, binding = load_inputs(
        history_root, model_study, model_audit, paired_study, paired_audit, matched_study, matched_audit,
    )
    refit_study = Path(refit_study)
    report, _, receipt, refit_binding, contract = _audited_source(
        refit_study, Path(refit_audit), REFIT_VERSION, _REFIT_FILES,
    )
    parent = read(Path(model_study) / "contract.json")
    hashes = dict(original=refit_binding["route-models-original.json"], refit=refit_binding["route-models.json"])
    if (
        receipt["version"] != REFIT_AUDIT_VERSION
        or any(contract[key] != value for key, value in binding.items())
        or any(contract[key] != parent[key] for key in ("parent", "identity", "scene"))
        or contract["test_positions"] != REFIT_POSITIONS
        or receipt["test_positions"] != REFIT_POSITIONS
        or contract["split_seed"] != REFIT_SPLIT_SEED
        or contract["frozen_inputs"] != {name: refit_binding[name] for name in _REFIT_INPUTS}
        or refit_binding["training-prior.json"] != binding["model_study"]["geometry_prior"]
        or hashes["original"] != binding["model_study"]["checkpoint"]
        or receipt["checkpoint_sha256"] != hashes
        or report["choices_sha256"] != refit_binding["choices.json"]
        or receipt["choices_sha256"] != refit_binding["choices.json"]
        or receipt["comparison"] != report["comparison"]
    ):
        raise ValueError("Refit evidence differs from the audited source and frozen comparison")

    checkpoint = load_checkpoint(refit_study / "route-models.json")
    expected_rows = original_rows + new_rows
    with (refit_study / "training-dataset.jsonl").open(encoding="utf-8") as stream:
        training = [json.loads(line) for line in stream]
    summary = training_summary(original_rows, new_rows, checkpoint, hashes)
    if (
        training != expected_rows
        or checkpoint["seed"] != original_checkpoint["seed"]
        or checkpoint["routes"] != original_checkpoint["routes"]
        or checkpoint["training_rows"] != len(expected_rows)
        or checkpoint["dataset_sha256"] == original_checkpoint["dataset_sha256"]
        or summary != receipt["training"]
        or summary != report["training"]
    ):
        raise ValueError("Refit checkpoint differs from its audited training rows")

    layout_archive = read(refit_study / "layouts.json")
    layouts, provenance = layout_archive["layouts"], layout_archive["provenance"]
    coordinates = {tuple(layout["bucket_xy_m"]) for layout in layouts}
    previous_xy = {tuple(layout["bucket_xy_m"]) for layout in previous}
    if (
        len(layouts) != REFIT_POSITIONS or len(coordinates) != REFIT_POSITIONS or coordinates & previous_xy
        or provenance["version"] != REFIT_PLAN_VERSION or provenance["seed"] != REFIT_SPLIT_SEED
        or provenance["layouts_sha256"] != digest(layouts)
        or provenance["declared_previous_coordinates_m"] != [list(xy) for xy in sorted(previous_xy)]
        or any(
            layout["split"] != "test" or layout["id"] != f"intergrid-route-refit-test-{index:02d}"
            or layout["context_id"] != layout["id"]
            for index, layout in enumerate(layouts)
        )
    ):
        raise ValueError("Refit test layouts differ from the audited held-out positions")

    refit_binding["checkpoint_sha256"] = hashes["refit"]
    return checkpoint, original_rows, new_rows, previous + layouts, dict(binding, refit_study=refit_binding)


def _library_ids(layouts):
    renamed = []
    for index, layout in enumerate(layouts):
        identity = f"intergrid-table-library-test-{index:02d}"
        renamed.append(dict(layout, id=identity, context_id=identity))
    return renamed


def sample_layouts(excluded, geometry, grid=None, *, seed=SPLIT_SEED):
    """Reproduce the frozen layouts from archived coordinates and scene geometry."""
    return _library_ids(sample_matched_layouts(excluded, geometry, grid, seed=seed))


def fresh_layouts(history_root, grid_path, scene, assets, previous_layouts, seed=SPLIT_SEED):
    """Sample new test positions after excluding all inventoried and declared positions."""
    layouts, provenance = make_matched_layouts(
        history_root, read(grid_path), scene=scene, assets=assets, previous_layouts=previous_layouts, seed=seed,
    )
    layouts = _library_ids(layouts)
    provenance.update(version=PLAN_VERSION, layouts_sha256=digest(layouts))
    return layouts, provenance
