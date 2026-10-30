"""Probe an already running isolated localhost module; never launches a server.

Requires its generated local config and rejects physical components, remotes,
cloud identity, unexpected services/resources, and non-loopback targets before
any rehearsal command. No credentials are read or accepted.
"""
from __future__ import annotations
import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import re

COORDINATOR = "sircus-operation-coordinator"
VISION = "sircus-bucket-vision"
REHEARSAL = "sircus-coordinator-rehearsal"
COORDINATOR_MODEL = "sircus:deployment:operation-coordinator"
VISION_MODEL = "sircus:deployment:bucket-vision-stub"
STAGES = (("confirm_board", "place_bucket"), ("confirm_bucket_placed", "ready_scan"),
          ("scan_bucket", "review_bucket"), ("review_bucket", "ready_loading"),
          ("go_loading", "insert_ball"), ("confirm_loaded", "ready_drop"), ("grip_drop", "complete"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_config(config, target, rehearsal=False):
    match = re.fullmatch(r"127\.0\.0\.1:([0-9]{1,5})", target)
    require(match is not None and 1 <= int(match.group(1)) <= 65535, "Only literal 127.0.0.1:PORT targets are supported")
    require(isinstance(config, dict), "Local config object required")
    require(set(config) <= {"modules", "services", "network"}, "Only isolated modules/services/network configuration is allowed; no cloud, remotes, components or auth")
    require(config.get("network") == {"bind_address": target}, "Local bind address must exactly match the probe target")
    modules = config.get("modules", [])
    require(len(modules) == 1 and modules[0].get("type") == "local", "Exactly one local stub module required")
    require(modules[0].get("name") == "sircus-deployment-stub", "Unexpected local module")
    require(set(modules[0]) <= {"name", "type", "executable_path", "env"}, "Unexpected module settings")
    executable = Path(modules[0].get("executable_path", "")).resolve()
    require(executable == Path(__file__).with_name("run-module.sh").resolve(), "Probe config must use this checkout's module launcher")
    require(set(modules[0].get("env", {})) <= {"SIRCUS_MODULE_PYTHON"}, "Module environment may contain only the Python interpreter path")
    expected = {COORDINATOR: ("rdk:service:generic", COORDINATOR_MODEL, {"mode": "blocked", "hardware_enabled": False}),
                VISION: ("rdk:service:vision", VISION_MODEL, {})}
    if rehearsal:
        expected[REHEARSAL] = ("rdk:service:generic", COORDINATOR_MODEL, {"mode": "dry_run", "hardware_enabled": False})
    services = config.get("services", [])
    require(len(services) == len(expected) and {s.get("name") for s in services} == set(expected), "Unexpected/missing service inventory")
    for service in services:
        require(set(service) == {"name", "api", "model", "attributes"}, "Dependencies or extra service settings are forbidden")
        api, model, attrs = expected[service["name"]]
        require((service["api"], service["model"], service["attributes"]) == (api, model, attrs), "Service must be an exact blocked/stub or separately named dry-run configuration")
    return expected


def validate_inventory(resources, rehearsal=False):
    allowed = {("rdk", "service", "generic", COORDINATOR), ("rdk", "service", "vision", VISION),
               ("rdk", "service", "motion", "builtin")}
    required = allowed - {("rdk", "service", "motion", "builtin")}
    if rehearsal:
        allowed.add(("rdk", "service", "generic", REHEARSAL)); required.add(("rdk", "service", "generic", REHEARSAL))
    actual = {(r["namespace"], r["type"], r["subtype"], r["name"]) for r in resources}
    require(required <= actual <= allowed and len(actual) == len(resources),
            "Unexpected resources/remotes/components: refusing all command probes")


async def probe(config, target="127.0.0.1:18768", *, rehearsal=False):
    validate_config(config, target, rehearsal)
    # SDK imports are delayed so invalid-target/config checks need no Viam setup.
    from google.protobuf.json_format import MessageToDict
    from viam.robot.client import RobotClient
    from viam.rpc.dial import DialOptions
    from viam.services.generic import Generic
    from viam.services.vision import Vision
    async def bounded(call):
        return await asyncio.wait_for(call, 15.)
    client = await bounded(RobotClient.at_address(target, RobotClient.Options(
        dial_options=DialOptions(insecure=True, disable_webrtc=True))))
    try:
        resources = [{"namespace": r.namespace, "type": r.type, "subtype": r.subtype, "name": r.name}
                     for r in client.resource_names]
        validate_inventory(resources, rehearsal)
        version = await bounded(client.get_version())
        blocked = Generic.from_robot(client, COORDINATOR)
        status = await bounded(blocked.do_command({"action": "status"}))
        require(status.get("mode") == "blocked" and status.get("hardware_enabled") is False, "Coordinator is not blocked")
        rejected = False
        try:
            await bounded(blocked.do_command({"action": "grip_drop"}))
        except Exception:
            rejected = True
        require(rejected, "Blocked coordinator unexpectedly accepted Grip & drop")
        # A fresh successful read distinguishes expected rejection from loss of server.
        status = await bounded(blocked.do_command({"action": "status"}))
        require(status.get("mode") == "blocked" and status.get("hardware_enabled") is False, "Blocked coordinator changed state")
        vision = Vision.from_robot(client, VISION)
        properties = await bounded(vision.get_properties())
        unknown = await bounded(vision.do_command({"action": "status"}))
        require(unknown.get("status") == "unknown" and unknown.get("estimate_available") is False
                and unknown.get("camera_read") is False and unknown.get("hardware_enabled") is False, "Vision falsely claims an available estimate")
        require(not any((properties.detections_supported, properties.detections_3d_supported,
                         properties.object_point_clouds_supported, properties.classifications_supported)), "Vision claims unsupported capabilities")
        stages, final = [], None
        if rehearsal:
            dry = Generic.from_robot(client, REHEARSAL)
            initial = await bounded(dry.do_command({"action": "status"}))
            require(initial.get("mode") == "dry_run" and initial.get("hardware_enabled") is False, "Rehearsal is not mock-only")
            await bounded(dry.do_command({"action": "reset"}))
            for action, phase in STAGES:
                final = await bounded(dry.do_command({"action": action, "confirmed": True}))
                require(final.get("mode") == "dry_run" and final.get("hardware_enabled") is False
                        and final.get("phase") == phase, "Synthetic stage failed")
                stages.append({"action": action, "phase": final["phase"]})
            require(final["trial"]["outcome"] == "unknown", "Synthetic release must not claim capture")
        return {"schema": "sircus-native-stub-validation-v2", "observed_at_utc": datetime.now(timezone.utc).isoformat(),
                "server_version": version.version, "target": target, "physically_verified": False,
                "scope": "isolated localhost RPC only; configuration checked and no component/remote resources allowed",
                "resources": resources, "blocked_grip_drop_rejected": rejected,
                "vision_properties": MessageToDict(properties), "vision_status": unknown,
                "stages": stages, "final_phase": final["phase"] if final else "not_requested",
                "trial_outcome": final["trial"]["outcome"] if final else "not_requested",
                "hardware_enabled": False}
    finally:
        await bounded(client.close())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Generated isolated local Viam config")
    parser.add_argument("--target", default="127.0.0.1:18768")
    parser.add_argument("--rehearsal", action="store_true", help="Also test the separately named dry-run resource")
    parser.add_argument("--output", type=Path, help="Optional new JSON result file; refuses overwrites")
    args = parser.parse_args()
    report = asyncio.run(probe(json.loads(args.config.read_text()), args.target, rehearsal=args.rehearsal))
    raw = json.dumps(report, indent=2, allow_nan=False)+"\n"
    if args.output:
        with args.output.open("x") as out:
            out.write(raw)
    print(raw, end="")


if __name__ == "__main__":
    main()
