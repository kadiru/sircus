"""Verify one planned pick/drop/reset on the explicitly local fake-only Viam server."""
import asyncio
import json
from pathlib import Path

from viam.components.generic import Generic
from viam.robot.client import RobotClient
from viam.rpc.dial import DialOptions

from viam_module import check_simulation_config


async def main():
    check_simulation_config()
    options = RobotClient.Options(dial_options=DialOptions(insecure=True, disable_webrtc=True))
    robot = await RobotClient.at_address("127.0.0.1:8080", options)
    try:
        demo = Generic.from_robot(robot, "sircus")
        probe = await demo.do_command({"probe": True})
        await demo.do_command({"run": True, "action": {"u_mm": 0, "v_mm": 0, "gap_mm": 200},
                               "seed": 0, "delay_s": 0.1, "playback_speed": 2})
        for _ in range(180):
            await asyncio.sleep(0.5)
            status = await demo.do_command({"status": True})
            if status.get("status") != "running":
                break
        else:
            await demo.do_command({"stop": True})
            raise RuntimeError("Native validation exceeded 90 seconds; trial cancelled")
        moves = [item for item in status.get("events", []) if item.get("motion_ok")]
        report = {"server_version": "1.14.3", "probe": probe, "status": status,
                  "planned_motions_passed": len(moves),
                  "max_tcp_error_mm": max((item["position_error_mm"] for item in moves), default=None),
                  "scope": "Viam fake-arm kinematics plus uncalibrated compliant-board dynamics; no physical verification"}
        output = Path(__file__).resolve().parent / "data" / "viam-native-validation.json"
        output.parent.mkdir(exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")
        assert status["status"] == "complete", status.get("error", status)
        assert len(moves) == 7, moves
        print(f"PASS: 7/7 planned motions; maximum TCP residual {report['max_tcp_error_mm']:.6f} mm; "
              f"simulated outcome {status['result']['outcome']}")
        print(output)
    finally:
        await robot.close()


if __name__ == "__main__":
    asyncio.run(main())
