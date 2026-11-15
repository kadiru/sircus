"""Compare a direct-first fallback with the frozen bounce route selector."""
from __future__ import annotations

from .intergrid_route_fallback_inputs import fresh_layouts, load_fallback_inputs
from .intergrid_route_selector_plan import FALLBACK_METHODS, select_fallback_requests, summarize_fallback
from .intergrid_route_selector_study import parse_study_args, run_comparison


VERSION = "sircus-intergrid-route-fallback-study-v1"
SPLIT_SEED = 2026112201
SPLIT_SEEDS = (SPLIT_SEED, SPLIT_SEED + 1)
POSITIONS = 100


def main():
    extra_sources = ("selector",)
    args = parse_study_args(__doc__, extra_sources)
    run_comparison(args, version=VERSION, split_seeds=SPLIT_SEEDS, positions=POSITIONS,
                   methods=FALLBACK_METHODS, load_inputs=load_fallback_inputs, make_layouts=fresh_layouts,
                   select=select_fallback_requests, summarize_outcomes=summarize_fallback,
                   extra_source_names=extra_sources)


if __name__ == "__main__":
    main()
