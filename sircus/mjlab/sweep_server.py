"""Isolated simulator-only planner UI; never changes an existing service."""

import argparse
import json
import os
from pathlib import Path
from http.server import ThreadingHTTPServer

from .interface import DEFAULT_SCENE, InterfaceState, handler_for
from .sweep_worker import VERSION, atomic, load_reference


def public_state(ref):
    """Search readiness is independent of legacy learned-policy readiness."""
    public = ref["public"]
    cells = [
        dict(
            id=row["id"],
            bucket_xy_m=row["bucket_xy_m"],
            status=row["result_category"],
            cache_available=bool(row["qualified_settings"]),
        )
        for row in public["cells"]
    ]
    return dict(
        status="not_started",
        planner_only=True,
        scope="Independent sweep planner; no learned policy configured",
        planner=dict(
            ready=True,
            version=VERSION,
            physics=ref["contract"]["physics"],
            farm_cells=cells,
            qualified_positions=sum(row["cache_available"] for row in cells),
            grid_positions=len(cells),
            qualification="Exact published cache or fresh strict geometry, 3/3 nominal and >=18/20 fresh native captures plus sampled common-home path",
            arbitrary_query_scope="Any bucket center inside tabletop bounds can be requested; finite search may remain unresolved",
            engine="mjlab / corrected native MuJoCo Warp, prescribed held endpoints",
        ),
    )


def ui_scene(ref):
    scene = json.loads(json.dumps(DEFAULT_SCENE))
    first = ref["catalogue"]["sources"][0]["source"]
    scene["bucket"]["xy"] = first["bucket_xy_m"]
    scene["board"]["center"] = first["board_center_m"]
    scene["release_center"] = first["release_ball_center_m"]
    scene["pose_display"] = (
        "Native endpoint/trajectory evidence appears after Find shot; idle scene is schematic"
    )
    return scene


def main():
    parser = argparse.ArgumentParser()
    for name in ("reference", "assets", "gpu-lock", "cache", "state-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument(
        "--host", default="127.0.0.1", choices=["127.0.0.1", "localhost"]
    )
    parser.add_argument("--port", type=int, default=18786)
    args = parser.parse_args()
    ref = load_reference(args.reference, args.assets)
    args.state_dir.mkdir(parents=True, exist_ok=True)
    report, scene = (
        args.state_dir / "planner-report.json",
        args.state_dir / "ui-scene.json",
    )
    atomic(report, public_state(ref))
    atomic(scene, ui_scene(ref))
    for key, value in (
        ("REFERENCE", args.reference),
        ("ASSETS", args.assets),
        ("LOCK", args.gpu_lock),
        ("CACHE", args.cache),
    ):
        os.environ["SIRCUS_SWEEP_" + key] = str(value.resolve())
    state = InterfaceState(report, scene, "sircus.mjlab.sweep_interface:solve")
    server = ThreadingHTTPServer((args.host, args.port), handler_for(state))
    print(
        json.dumps(
            dict(
                url=f"http://{args.host}:{args.port}",
                planner=VERSION,
                cached_positions=public_state(ref)["planner"]["qualified_positions"],
                simulation_only=True,
            )
        ),
        flush=True,
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
