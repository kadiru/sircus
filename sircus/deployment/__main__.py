"""Offline, credential-free deployment-contract rehearsal; no Viam server needed."""
import argparse
import asyncio
import json
from .controller import DeploymentController


async def run(rehearse):
    controller = DeploymentController("dry_run" if rehearse else "blocked")
    if rehearse:
        for action in ("confirm_board", "confirm_bucket_placed", "scan_bucket", "review_bucket",
                       "go_loading", "confirm_loaded", "grip_drop"):
            await controller.command({"action": action, "confirmed": True})
    print(json.dumps(controller.status(), indent=2, allow_nan=False))
    await controller.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rehearse", action="store_true", help="Run explicit synthetic workflow fixture")
    asyncio.run(run(parser.parse_args().rehearse))
