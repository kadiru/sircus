"""Optional verified hybrid planner server; historical server remains unchanged."""
from __future__ import annotations
import argparse
from http.server import ThreadingHTTPServer
from pathlib import Path
import signal

from .interactive_physics_server import InteractivePhysicsState, PhysicsInterface, physics_handler


class StalledPhysicsState(InteractivePhysicsState):
    def __init__(self, planner, *, fast_planner=None, hybrid_planner=None, default_planner=None):
        baseline_default = default_planner if default_planner != "hybrid" else None
        super().__init__(planner, fast_planner=fast_planner, default_planner=baseline_default)
        if hybrid_planner is not None:
            if hybrid_planner.planning_bucket_bounds != planner.planning_bucket_bounds:
                raise ValueError("Planner bounds must match")
            self.planners["hybrid"] = hybrid_planner
        if default_planner == "hybrid":
            if hybrid_planner is None:
                raise ValueError("Verified hybrid default is unavailable")
            self.default_planner = "hybrid"

    def capabilities(self):
        caps = super().capabilities()
        if "hybrid" in self.planners:
            caps["planners"].insert(0, {"id": "hybrid", "label": "Checked planner with stalled-reference fallback"})
        return caps


def main():
    from .interactive_physics import PhysicsPlanner
    from .intergrid_flight_ranking_study import EXTRA_SOURCES
    from .intergrid_route_selector_study import SOURCE_NAMES

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent-study", "parent-audit", "reference", "assets", "gpu-lock", "history-root", "rule",
                 "system-study", "system-audit", "evidence-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in (*SOURCE_NAMES, *EXTRA_SOURCES, "prior-comparison"):
        for kind in ("study", "audit"):
            parser.add_argument(f"--{name}-{kind}", type=Path, required=True)
    parser.add_argument("--study-report", type=Path, help="Display a historical study alongside the planner")
    parser.add_argument("--scene-json", type=Path)
    parser.add_argument("--solver", help="Optional existing simulation solver for the historical modes")
    parser.add_argument("--gpu-wait-s", type=int, default=15, metavar="SECONDS",
                        help="Seconds to wait for the shared 4090, from 0 to 3600")
    parser.add_argument("--port", type=int, default=18781)
    for name in ("fast", "neural", "budget"):
        for kind in ("study", "audit"):
            parser.add_argument(f"--{name}-{kind}", type=Path)
    parser.add_argument("--stalled-study", type=Path)
    parser.add_argument("--stalled-audit", type=Path)
    parser.add_argument("--default-planner", choices=("hybrid", "fast", "classic"))
    args = parser.parse_args()
    fast_paths = [getattr(args, f"{name}_{kind}") for name in ("fast", "neural", "budget") for kind in ("study", "audit")]
    if any(fast_paths) and not all(fast_paths):
        parser.error("Enabling fast planning requires all fast/neural/budget study and audit paths")
    if args.default_planner == "fast" and not all(fast_paths):
        parser.error("Fast default requires verified fast planning inputs")
    stalled_paths = [args.stalled_study, args.stalled_audit]
    if any(stalled_paths) and (not all(stalled_paths) or not all(fast_paths)):
        parser.error("Hybrid requires audited stalled study and all original fast-planning inputs")
    if args.default_planner == "hybrid" and not all(stalled_paths):
        parser.error("Hybrid default requires its passed matched comparison")
    planner = PhysicsPlanner(args, args.evidence_dir)
    fast = None
    if all(fast_paths):
        from .interactive_fast_planner import FastPhysicsPlanner
        fast = FastPhysicsPlanner(args, args.evidence_dir / "fast", classic=planner)
    hybrid = None
    if all(stalled_paths):
        from .interactive_stalled_planner import StalledPhysicsPlanner
        hybrid = StalledPhysicsPlanner(args, args.evidence_dir / "hybrid", classic=planner)
    physics = StalledPhysicsState(planner, fast_planner=fast, hybrid_planner=hybrid,
                                  default_planner=args.default_planner)
    interface = PhysicsInterface(physics, args.study_report, args.scene_json, args.solver)
    def shutdown_signal(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, shutdown_signal)
    with ThreadingHTTPServer(("127.0.0.1", args.port), physics_handler(interface, physics)) as server:
        print(f"Sircus mjlab physics planner: http://127.0.0.1:{server.server_port}/", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            physics.close()


if __name__ == "__main__":
    main()
