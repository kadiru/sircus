"""Test the highest admitted release for each tilt at the case-33 bucket."""

from __future__ import annotations

import argparse
from pathlib import Path

from .intergrid_case33_tilt_plan import load_tilt_plan
from .intergrid_setup_drop_study import run_plan


VERSION = "sircus-intergrid-case33-tilt-v1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("preparation", "source-study", "source-audit", "parent-study", "parent-audit",
                 "reference", "assets", "gpu-lock", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--gpu-wait-s", type=int, default=7 * 3600)
    args = parser.parse_args()
    if args.output.exists() or args.gpu_wait_s <= 0:
        parser.error("Provide a fresh output directory and positive GPU wait bound")
    plan, binding = load_tilt_plan(args.preparation, args.source_study, args.source_audit,
                                   args.parent_study, args.parent_audit)
    if not any(case["requests"] for case in plan["cases"][1:]):
        parser.error("No tilt passed the frozen release search")
    run_plan(args, plan, binding, args.preparation, args.source_study / "training-prior.json",
             version=VERSION, source_key="tilt_source")


if __name__ == "__main__":
    main()
