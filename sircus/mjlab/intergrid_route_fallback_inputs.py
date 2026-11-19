"""Bind the audited selector inputs and freeze 100 new test positions."""

from __future__ import annotations

from pathlib import Path

from .benchmark_verify import digest, read
from .intergrid_matched_route_plan import POSITION_COUNT as PASS_SIZE
from .intergrid_matched_route_plan import make_layouts as make_matched_layouts
from .intergrid_matched_route_plan import sample_layouts as sample_matched_layouts
from .intergrid_route_plan import ROUTES
from .intergrid_route_refit_inputs import _audited_source
from .intergrid_route_selector_audit import VERSION as SELECTOR_AUDIT_VERSION
from .intergrid_route_selector_inputs import PLAN_VERSION as SELECTOR_PLAN_VERSION
from .intergrid_route_selector_inputs import SPLIT_SEED as SELECTOR_SPLIT_SEED, load_selector_inputs
from .intergrid_route_selector_plan import METHODS as SELECTOR_METHODS
from .intergrid_route_selector_study import POSITIONS as SELECTOR_POSITIONS
from .intergrid_route_selector_study import VERSION as SELECTOR_VERSION
from .intergrid_route_study import PER_ROUTE_TEST


PLAN_VERSION = "sircus-intergrid-route-fallback-plan-v1"
SPLIT_SEED = 2026112201
POSITION_COUNT = 2 * PASS_SIZE

_SELECTOR_INPUTS = ("layouts.json", "training-prior.json", "route-models.json", "route-history.json")
_SELECTOR_FILES = (*_SELECTOR_INPUTS, "choices.json")


def load_fallback_inputs(
    history_root, model_study, model_audit, paired_study, paired_audit, matched_study, matched_audit,
    refit_study, refit_audit, library_study, library_audit, selector_study, selector_audit,
):
    """Reuse the selector's frozen weights, libraries and gate evidence."""
    checkpoint, original, new, previous, binding, route_history = load_selector_inputs(
        history_root, model_study, model_audit, paired_study, paired_audit, matched_study, matched_audit,
        refit_study, refit_audit, library_study, library_audit,
    )
    selector_study = Path(selector_study)
    report, _, receipt, selector_binding, contract = _audited_source(
        selector_study, Path(selector_audit), SELECTOR_VERSION, _SELECTOR_FILES,
    )
    parent = read(Path(library_study) / "contract.json")
    checkpoint_sha = binding["library_study"]["checkpoint_sha256"]
    attempts = SELECTOR_POSITIONS * len(SELECTOR_METHODS)
    if (
        receipt["version"] != SELECTOR_AUDIT_VERSION
        or any(contract[key] != value for key, value in binding.items())
        or any(contract[key] != parent[key] for key in ("parent", "identity", "scene"))
        or contract["test_positions"] != SELECTOR_POSITIONS
        or contract["split_seed"] != SELECTOR_SPLIT_SEED
        or contract["methods"] != list(SELECTOR_METHODS) or contract["routes"] != list(ROUTES)
        or contract["attempts_per_method_per_position"] != 1
        or contract["candidate_budget_per_route"] != PER_ROUTE_TEST
        or contract["frozen_inputs"] != {name: selector_binding[name] for name in _SELECTOR_INPUTS}
        or selector_binding["training-prior.json"] != binding["model_study"]["geometry_prior"]
        or selector_binding["route-models.json"] != checkpoint_sha
        or report["checkpoint_sha256"] != checkpoint_sha or receipt["checkpoint_sha256"] != checkpoint_sha
        or report["choices_sha256"] != selector_binding["choices.json"]
        or receipt["choices_sha256"] != selector_binding["choices.json"]
        or receipt["test_positions"] != SELECTOR_POSITIONS or report["test_prepared"] != SELECTOR_POSITIONS
        or receipt["test_attempts"] != attempts or report["test_slots"] != attempts
        or receipt["test_releases"] != report["test_releases"]
        or receipt["comparison"] != report["comparison"] or receipt["library"] != report["library"]
        or receipt["selection_source_positions"] != len(route_history)
        or report["selection_source_positions"] != len(route_history)
        or read(selector_study / "route-history.json") != route_history
    ):
        raise ValueError("Selector study does not match its audited checkpoint, gate history and test evidence")

    archive = read(selector_study / "layouts.json")
    layouts, provenance = archive["layouts"], archive["provenance"]
    coordinates = {tuple(layout["bucket_xy_m"]) for layout in layouts}
    previous_xy = {tuple(layout["bucket_xy_m"]) for layout in previous}
    if (
        len(layouts) != SELECTOR_POSITIONS or len(coordinates) != SELECTOR_POSITIONS or coordinates & previous_xy
        or provenance["version"] != SELECTOR_PLAN_VERSION or provenance["seed"] != SELECTOR_SPLIT_SEED
        or provenance["layouts_sha256"] != digest(layouts)
        or provenance["declared_previous_coordinates_m"] != [list(xy) for xy in sorted(previous_xy)]
        or any(
            layout["split"] != "test" or layout["id"] != f"intergrid-route-selector-test-{index:02d}"
            or layout["context_id"] != layout["id"]
            for index, layout in enumerate(layouts)
        )
    ):
        raise ValueError("Selector layouts do not match the 50 audited test positions")

    selector_binding["checkpoint_sha256"] = checkpoint_sha
    return checkpoint, original, new, previous + layouts, dict(binding, selector_study=selector_binding), route_history


def _second_pass(first, exclusions, geometry, grid, seed):
    used = [*exclusions, *(layout["bucket_xy_m"] for layout in first)]
    second = sample_matched_layouts(used, geometry, grid, seed=seed + 1)
    renamed = []
    for index, layout in enumerate(first + second):
        identity = f"intergrid-route-fallback-test-{index:02d}"
        renamed.append(dict(layout, id=identity, context_id=identity))
    return renamed


def sample_layouts(exclusions, geometry, grid=None, *, seed=SPLIT_SEED):
    """Reproduce both geometry-only passes, excluding the first from the second."""
    exclusions = list(exclusions)
    grid = None if grid is None else list(grid)
    first = sample_matched_layouts(exclusions, geometry, grid, seed=seed)
    return _second_pass(first, exclusions, geometry, grid, seed)


def fresh_layouts(history_root, grid_path, scene, assets, previous, seed=SPLIT_SEED):
    """Inventory earlier coordinates, then sample two disjoint sets of 50."""
    grid = read(grid_path)
    first, provenance = make_matched_layouts(
        history_root, grid, scene=scene, assets=assets, previous_layouts=previous, seed=seed,
    )
    layouts = _second_pass(first, provenance["excluded_coordinates_m"], provenance["geometry"], grid, seed)
    provenance.update(version=PLAN_VERSION, pass_seeds=[seed, seed + 1], layouts_sha256=digest(layouts))
    return layouts, provenance
