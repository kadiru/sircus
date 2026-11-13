"""Freeze 50 fresh bucket positions for the matched direct/table comparison."""
from __future__ import annotations

from .intergrid_proposer_plan import make_splits, sample_splits
from .learning_contract import digest

VERSION = "sircus-intergrid-matched-route-plan-v1"
SEED = 2026111801
POSITION_COUNT = 50


def _matched_ids(splits):
    layouts = []
    for index, layout in enumerate(splits["test"]):
        identity = f"intergrid-matched-route-test-{index:02d}"
        layouts.append(dict(layout, id=identity, context_id=identity))
    return layouts


def sample_layouts(excluded, geometry, grid=None, *, seed=SEED):
    """Reproduce the held-out layouts from frozen coordinates and scene geometry."""
    return _matched_ids(sample_splits(excluded, geometry, grid, seed=seed, include_train=False))


def make_layouts(history_root, grid=None, *, scene, assets, previous_layouts=(), seed=SEED):
    """Inventory historical placements and sample two per grid rectangle plus two near the fixed bases."""
    splits, provenance = make_splits(
        history_root, grid, scene=scene, assets=assets, previous_layouts=previous_layouts,
        seed=seed, include_train=False,
    )
    layouts = _matched_ids(splits)
    provenance.pop("splits_sha256")
    provenance.update(version=VERSION, layouts_sha256=digest(layouts))
    return layouts, provenance
