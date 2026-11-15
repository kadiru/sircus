"""Authenticate the selector's frozen models, route history and disjoint test layouts."""

from __future__ import annotations

import json
from pathlib import Path

from .benchmark_verify import digest, read, sha
from .intergrid_library_audit import VERSION as LIBRARY_AUDIT_VERSION
from .intergrid_library_inputs import PLAN_VERSION as LIBRARY_PLAN_VERSION
from .intergrid_library_inputs import SPLIT_SEED as LIBRARY_SPLIT_SEED, load_library_inputs
from .intergrid_library_plan import METHODS as LIBRARY_METHODS
from .intergrid_library_study import POSITIONS as LIBRARY_POSITIONS
from .intergrid_library_study import VERSION as LIBRARY_VERSION
from .intergrid_matched_route_plan import make_layouts as make_matched_layouts
from .intergrid_matched_route_plan import sample_layouts as sample_matched_layouts
from .intergrid_route_plan import ROUTES
from .intergrid_route_refit_inputs import _audited_source
from .intergrid_route_study import PER_ROUTE_TEST


PLAN_VERSION = "sircus-intergrid-route-selector-plan-v1"
SPLIT_SEED = 2026112101

_LIBRARY_INPUTS = ("layouts.json", "training-prior.json", "route-models.json")
_LIBRARY_FILES = (*_LIBRARY_INPUTS, "choices.json")
_HISTORY_METHOD = "specialist_refit"


def _capture_history(study, refit_binding):
    """Take only positions with both audited specialist routes actually released."""
    manifest_path = study / "evidence-manifest.json"
    ledger_path = study / "rows.jsonl"
    ledger_sha = sha(ledger_path)
    if (
        sha(manifest_path) != refit_binding["evidence-manifest.json"]
        or read(manifest_path)["rows.jsonl"] != ledger_sha
        or sha(study / "choices.json") != refit_binding["choices.json"]
        or sha(study / "layouts.json") != refit_binding["layouts.json"]
    ):
        raise ValueError("Refit route history differs from its independent audit")

    layouts = read(study / "layouts.json")["layouts"]
    positions = {layout["id"]: layout["bucket_xy_m"] for layout in layouts}
    choices = read(study / "choices.json")["slots"]
    slots = {}
    releases = {}
    for slot in choices:
        if slot["method"] != _HISTORY_METHOD:
            continue
        context, route, name = slot["context_id"], slot["intended_route"], slot["id"]
        if (
            context not in positions or route not in ROUTES or (context, route) in slots
            or slot["bucket_xy_m"] != positions[context] or name != f"{context}-{_HISTORY_METHOD}-{route}"
        ):
            raise ValueError(f"Refit selector history slot differs from its layout: {name}")
        slots[context, route] = slot
        request = slot["request"]
        if request is None:
            continue
        if any(request[key] != slot[key] for key in ("id", "method", "intended_route", "context_id", "bucket_xy_m")):
            raise ValueError(f"Refit selector history request differs from its slot: {name}")
        releases[name] = request
    if len(slots) != len(layouts) * len(ROUTES):
        raise ValueError("Refit selector history is missing a specialist route slot")

    captures = {}
    with ledger_path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            name = row["id"]
            if name not in releases and not (row["phase"] == "test" and row["method"] == _HISTORY_METHOD):
                continue
            request = releases.get(name)
            if (
                request is None or name in captures or row["phase"] != "test" or row["role"] != "test"
                or row["native_release"] is not True
                or any(row[key] != request[key] for key in (
                    "method", "context_id", "bucket_xy_m", "intended_route", "values", "proposal_id",
                ))
                or type(row["route_success"]) is not bool or type(row["invalid_physics"]) is not bool
            ):
                raise ValueError(f"Refit selector history release differs from its frozen request: {name}")
            captures[name] = row["route_success"] and not row["invalid_physics"]
    if captures.keys() != releases.keys():
        raise ValueError("Refit selector history is missing a specialist release")

    history = []
    for context, xy in positions.items():
        direct, table = (slots[context, route]["id"] for route in ("direct", "table"))
        if direct in captures and table in captures:
            history.append(dict(id=context, bucket_xy_m=xy,
                                direct_capture=captures[direct], table_capture=captures[table]))
    if not history:
        raise ValueError("Refit does not contain paired specialist releases to tune route choice")
    return history, ledger_sha


def load_selector_inputs(
    history_root, model_study, model_audit, paired_study, paired_audit, matched_study, matched_audit,
    refit_study, refit_audit, library_study, library_audit,
):
    """Reuse fitted weights and training rows, with prior refit releases used only for route choice."""
    checkpoint, original, new, previous, binding = load_library_inputs(
        history_root, model_study, model_audit, paired_study, paired_audit, matched_study, matched_audit,
        refit_study, refit_audit,
    )
    library_study, refit_study = Path(library_study), Path(refit_study)
    report, _, receipt, library_binding, contract = _audited_source(
        library_study, Path(library_audit), LIBRARY_VERSION, _LIBRARY_FILES,
    )
    parent = read(refit_study / "contract.json")
    checkpoint_sha = binding["refit_study"]["checkpoint_sha256"]
    if (
        receipt["version"] != LIBRARY_AUDIT_VERSION
        or any(contract[key] != value for key, value in binding.items())
        or any(contract[key] != parent[key] for key in ("parent", "identity", "scene"))
        or contract["test_positions"] != LIBRARY_POSITIONS
        or contract["split_seed"] != LIBRARY_SPLIT_SEED
        or contract["methods"] != list(LIBRARY_METHODS) or contract["routes"] != ["table"]
        or contract["attempts_per_method_per_route_per_position"] != 1
        or contract["candidate_budget_per_route"] != PER_ROUTE_TEST
        or contract["frozen_inputs"] != {name: library_binding[name] for name in _LIBRARY_INPUTS}
        or library_binding["training-prior.json"] != binding["model_study"]["geometry_prior"]
        or library_binding["route-models.json"] != checkpoint_sha
        or report["checkpoint_sha256"] != checkpoint_sha or receipt["checkpoint_sha256"] != checkpoint_sha
        or report["choices_sha256"] != library_binding["choices.json"]
        or receipt["choices_sha256"] != library_binding["choices.json"]
        or receipt["test_positions"] != LIBRARY_POSITIONS
        or receipt["comparison"] != report["comparison"] or receipt["library"] != report["library"]
    ):
        raise ValueError("Library study does not match its audited checkpoint and test evidence")

    archive = read(library_study / "layouts.json")
    layouts, provenance = archive["layouts"], archive["provenance"]
    coordinates = {tuple(layout["bucket_xy_m"]) for layout in layouts}
    previous_xy = {tuple(layout["bucket_xy_m"]) for layout in previous}
    if (
        len(layouts) != LIBRARY_POSITIONS or len(coordinates) != LIBRARY_POSITIONS or coordinates & previous_xy
        or provenance["version"] != LIBRARY_PLAN_VERSION or provenance["seed"] != LIBRARY_SPLIT_SEED
        or provenance["layouts_sha256"] != digest(layouts)
        or provenance["declared_previous_coordinates_m"] != [list(xy) for xy in sorted(previous_xy)]
        or any(
            layout["split"] != "test" or layout["id"] != f"intergrid-table-library-test-{index:02d}"
            or layout["context_id"] != layout["id"]
            for index, layout in enumerate(layouts)
        )
    ):
        raise ValueError("Library layouts do not match the 50 audited test positions")

    route_history, ledger_sha = _capture_history(refit_study, binding["refit_study"])
    binding["refit_study"] = dict(binding["refit_study"], **{"rows.jsonl": ledger_sha})
    library_binding["checkpoint_sha256"] = checkpoint_sha
    return checkpoint, original, new, previous + layouts, dict(binding, library_study=library_binding), route_history


def _selector_ids(layouts):
    renamed = []
    for index, layout in enumerate(layouts):
        identity = f"intergrid-route-selector-test-{index:02d}"
        renamed.append(dict(layout, id=identity, context_id=identity))
    return renamed


def sample_layouts(excluded, geometry, grid=None, *, seed=SPLIT_SEED):
    """Reproduce fresh selector positions from the frozen placement seed."""
    return _selector_ids(sample_matched_layouts(excluded, geometry, grid, seed=seed))


def fresh_layouts(history_root, grid_path, scene, assets, previous_layouts, seed=SPLIT_SEED):
    """Exclude earlier TEST, development and training coordinates before sampling."""
    layouts, provenance = make_matched_layouts(
        history_root, read(grid_path), scene=scene, assets=assets, previous_layouts=previous_layouts, seed=seed,
    )
    layouts = _selector_ids(layouts)
    provenance.update(version=PLAN_VERSION, layouts_sha256=digest(layouts))
    return layouts, provenance
