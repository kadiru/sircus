"""Compare exact missed shots with alternative joint angles at the same poses."""

from __future__ import annotations

import argparse
from pathlib import Path

from .intergrid_joint_branch_search import load_branch_plan
from .intergrid_setup_drop_study import run_plan


VERSION = "sircus-intergrid-joint-branch-v1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("preparation", "source-study", "source-audit", "parent-study", "parent-audit",
                 "reference", "assets", "gpu-lock", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--gpu-wait-s", type=int, default=7 * 3600)
    args = parser.parse_args()
    if args.output.exists() or args.gpu_wait_s <= 0:
        parser.error("Provide a fresh output directory and positive GPU wait bound")
    plan, binding = load_branch_plan(args.preparation, args.source_study, args.source_audit,
                                     args.parent_study, args.parent_audit)
    if not any(len(case["requests"]) > 1 for case in plan["cases"]):
        parser.error("No distinct alternate joint configuration passed preparation")
    run_plan(args, plan, binding, args.preparation, args.source_study / "training-prior.json",
             version=VERSION, source_key="branch_source")


if __name__ == "__main__":
    main()
