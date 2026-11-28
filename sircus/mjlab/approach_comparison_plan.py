"""Opt-in bounded expanded search with complete predicted bucket approach ranking.

This reuses the live hybrid planner's decision clock, exact setup/path gates,
warm worker lifecycle and early-return rule. No query outcome is used to propose
or rank a shot. A predicted bucket contact lowers a heuristic score; it never
forbids a shot or changes the native board-first capture evaluator.
"""
from __future__ import annotations

import copy
import math

from .benchmark_verify import digest
from .board_first_selector_plan import BoardFirstFlightRanker, VERSION as CONTACT_VERSION
from .bucket_approach import BucketApproach, VERSION as APPROACH_VERSION
from .expanded_actions import WIDE_BOUNDS
from .expanded_selector_plan import (
    VERSION as EXPANDED_MENU_VERSION, MENU_LIMIT as BASE_MENU_LIMIT,
    expanded_menu, key,
)
from .neural_shot_model import order
from .stalled_reference_plan import StalledReferencePlanner

VERSION = "sircus-expanded-bucket-approach-plan-v1"
SEED_LIMIT = 4
# (tilt degrees, release-gap metres), fixed before the fresh-position test.
# Heights/XY/heading/twist and original joint hints stay with each source.
LOCAL_DELTAS = (
    (-8., .02), (-8., .04),
    (-4., -.02), (-4., .02), (-4., .06),
    (0., -.04), (0., -.02), (0., .02), (0., .04), (0., .06), (0., .10),
    (4., -.02), (4., .02), (8., .02),
)
MENU_LIMIT = BASE_MENU_LIMIT + SEED_LIMIT * len(LOCAL_DELTAS)
MENU_KEYS = ("approach_menu_version", "approach_menu_checkpoint", "approach_menu_checkpoint_sha256")


def configure(libraries, ranker_config, neural_checkpoint):
    """Return independent, explicitly versioned inputs; safe to call twice.

    neural_checkpoint is the frozen expanded-selector network, not the separate
    route/shot regressor passed as source_checkpoint to the planner constructor.
    Configuration does no per-position planning, IK or trajectory assessment.
    """
    if libraries.get("expanded_menu_version") != EXPANDED_MENU_VERSION:
        raise ValueError("Bucket-approach comparison requires the expanded source library")
    if any(k in libraries.get("baseline_libraries", {}) for k in ("expanded_menu_version", *MENU_KEYS)):
        raise ValueError("Recursive approach/expanded library configuration")
    # Validate the checkpoint format without evaluating any query features.
    order(neural_checkpoint, [])
    checkpoint_hash = digest(neural_checkpoint)
    present = [name in libraries for name in MENU_KEYS]
    if any(present):
        if (not all(present) or libraries["approach_menu_version"] != VERSION
                or libraries["approach_menu_checkpoint_sha256"] != checkpoint_hash
                or digest(libraries["approach_menu_checkpoint"]) != checkpoint_hash):
            raise ValueError("Approach menu version or frozen selector checkpoint mismatch")
    if ranker_config.get("approach_ranking_version") not in (None, VERSION):
        raise ValueError("Unknown approach ranking version")
    if ranker_config.get("contact_ranking_version") not in (None, CONTACT_VERSION):
        raise ValueError("Approach ranking requires board-first contact semantics")
    result, config = copy.deepcopy(libraries), copy.deepcopy(ranker_config)
    result.update(approach_menu_version=VERSION,
                  approach_menu_checkpoint=copy.deepcopy(neural_checkpoint),
                  approach_menu_checkpoint_sha256=checkpoint_hash)
    config.update(approach_ranking_version=VERSION, contact_ranking_version=CONTACT_VERSION)
    return result, config


def comparison_menu(layout, libraries, source_checkpoint):
    """Keep the expanded menu and add bounded local alternatives around four seeds.

    The frozen network ranks the original proposals to choose seeds, then the
    unchanged planner ranks the complete menu with that same network. All this
    happens inside its decision clock. Native outcomes and solved query joints
    cannot enter the frozen neural features. Alternatives outside the declared
    bounds are omitted rather than silently clipped onto a different setting.
    """
    checkpoint = libraries.get("approach_menu_checkpoint")
    if (libraries.get("approach_menu_version") != VERSION
            or libraries.get("expanded_menu_version") != EXPANDED_MENU_VERSION
            or digest(checkpoint) != libraries.get("approach_menu_checkpoint_sha256")):
        raise ValueError("Approach menu requires its authenticated frozen checkpoint")
    base_libraries = {k: v for k, v in libraries.items() if k not in MENU_KEYS}
    base = expanded_menu(layout, base_libraries, source_checkpoint)
    if len(base) > BASE_MENU_LIMIT:
        raise ValueError("Expanded proposal menu exceeded its frozen bound")
    rows = copy.deepcopy(base)
    seen = {key(row) for row in rows}
    seeds = []
    seed_seen = set()
    for row in order(checkpoint, base):
        identity = key(row)
        if identity not in seed_seen:
            seeds.append(row)
            seed_seen.add(identity)
        if len(seeds) == SEED_LIMIT:
            break
    # Round-robin seeds for each delta avoids privileging the first source in
    # neural-score ties. The unmodified proposals retain their original order.
    for tilt_delta, gap_delta in LOCAL_DELTAS:
        for seed in seeds:
            values = list(seed["values"])
            values[3] += tilt_delta
            values[4] += gap_delta
            if any(not math.isfinite(v) or not lo <= v <= hi for v, lo, hi in
                   zip(values, WIDE_BOUNDS["low"], WIDE_BOUNDS["high"], strict=True)):
                continue
            candidate = copy.deepcopy(seed)
            candidate.update(values=values,
                id=f"{layout['id']}-approach-local-{len(rows)-len(base):03d}",
                method="expanded_approach_local", group="fixed_local_tilt_gap",
                approach_menu_version=VERSION,
                approach_parent_proposal_id=seed["id"],
                approach_parent_geometry_sha256=key(seed),
                approach_delta=dict(tilt_deg=tilt_delta, drop_gap_m=gap_delta),
                approach_seed_selector_sha256=libraries["approach_menu_checkpoint_sha256"])
            identity = key(candidate)
            if identity in seen:
                continue
            seen.add(identity)
            rows.append(candidate)
    # Final neural sorting overrides this order whenever probabilities differ.
    # Interleave only for deterministic ties so both original and local choices
    # remain represented under the unchanged 16-candidate exact-check budget.
    local = rows[len(base):]
    return [group[index] for index in range(max(len(base), len(local)))
            for group in (rows[:len(base)], local) if index < len(group)]


class ApproachFlightRanker(BoardFirstFlightRanker):
    """Rank exact admitted setups using a finite-horizon bucket contact forecast."""

    def __init__(self, scene, calibration, assets, rule):
        super().__init__(scene, calibration, assets, rule)
        self.bucket_approach = BucketApproach(scene)

    def predict(self, request):
        original = super().predict(request)
        assessment = self.bucket_approach.assess(request, self, prediction=original)
        prediction = copy.deepcopy(original)
        prediction.update(original_flight_score_m=original.get("score_m"),
            score_m=assessment["score_m"], predicted_route=assessment["predicted_route"],
            approach_ranking_version=VERSION,
            bucket_approach={k: copy.deepcopy(v) for k, v in assessment.items()
                             if k != "original_prediction"})
        return prediction


class ApproachComparisonPlanner(StalledReferencePlanner):
    """Same 30-second hybrid decision; only explicit menu/ranking inputs differ."""

    def __init__(self, method, libraries, source_checkpoint, history, ranker_config,
                 neural_checkpoint, **kwargs):
        libraries, ranker_config = configure(libraries, ranker_config, neural_checkpoint)
        super().__init__(method, libraries, source_checkpoint, history, ranker_config,
                         neural_checkpoint, **kwargs)


def description():
    """Serializable frozen-method details for manifests and the interface."""
    return dict(version=VERSION, bucket_approach_version=APPROACH_VERSION,
        expanded_base_menu_limit=BASE_MENU_LIMIT, local_seed_limit=SEED_LIMIT,
        local_deltas=[dict(tilt_deg=t, drop_gap_m=g) for t, g in LOCAL_DELTAS],
        total_menu_limit=MENU_LIMIT, initial_setup_check_limit=16,
        branch_fallback="Unchanged bounded hybrid arm-pose alternatives",
        seed_ranking="Frozen expanded neural checkpoint; no query outcome",
        decision_scope="Menu, neural order, exact setup/path gates and approach assessment all inside inherited decision clock",
        native_scoring_changed=False, predicted_contacts_forbidden=False,
        retrained=False, live_default_changed=False,
        limitations="Nominal release, one board and optional one table rebound; later robot/rim/board responses are not predicted")
