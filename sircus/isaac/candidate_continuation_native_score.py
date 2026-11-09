"""CPU-only independent scoring of continuation TRAIN evidence.

This is an auditor, not a producer or launcher. Every admitted release is replayed
through the unchanged original physical scorer and terminal reward from recorded
240 Hz observations. Inactive parked slots are excluded from paid request counts;
the original scorer receives only an in-memory reward compatibility copy.
Geometry, motion history, source identity and journal linkage are separate audits.
"""

import copy
import math
from pathlib import Path

from shot_benchmark_audit import verify_native_trial

PACKET_SCHEMA = "sircus-candidate-continuation-native-trial-v1"
SLOTS = 35


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_score(native_directory, packet, scene, clip=None):
    """Rebuild physical labels/rewards, preserving the caller's packet exactly.

    ``native_directory`` contains the packet's basename-only physics NPZ. The
    outer audit binds compressed packet bytes, reservation, supports and sources.
    This function also checks the NPZ checksum through the unchanged scorer.
    Synthetic fixtures remain explicitly marked and are never performance data.
    """
    require(isinstance(packet, dict) and packet.get("schema") == PACKET_SCHEMA
            and packet.get("mode") == "train", "Expected continuation TRAIN native packet")
    require(type(packet.get("synthetic")) is bool, "Missing native evidence synthetic label")
    for name in ("cells", "actions", "choices", "enabled", "rows", "rewards", "reward_features"):
        require(type(packet.get(name)) is list and len(packet[name]) == SLOTS,
                "Native slot coverage changed: "+name)
    request_ids = packet.get("request_ids")
    require(type(request_ids) is list and 1 <= len(request_ids) <= SLOTS
            and all(type(r) is str and r for r in request_ids)
            and len(set(request_ids)) == len(request_ids), "Invalid or duplicate paid native request IDs")
    count = len(request_ids)
    require(all(type(v) is bool for v in packet["enabled"])
            and packet["enabled"] == [i < count for i in range(SLOTS)],
            "Native enabled slots differ from paid request prefix")
    original = copy.deepcopy(packet)
    temporary = copy.deepcopy(packet)
    for i, (cell, row, choice, action) in enumerate(zip(
            packet["cells"], packet["rows"], packet["choices"], packet["actions"])):
        require(isinstance(cell, dict) and isinstance(row, dict), "Invalid native slot metadata")
        invalid = row.get("invalid")
        require(invalid is None or type(invalid) is str and bool(invalid), "Invalid native decline reason")
        require(type(row.get("success")) is bool and type(row.get("route_success")) is bool
                and type(row.get("route_quality")) is int
                and row["route_quality"] in ((1, 2) if row["route_success"] else (0,)),
                "Contradictory native outcome flags")
        require(not row["route_success"] or row["success"], "Route capture lacks physical capture")
        if i < count:
            require(isinstance(choice, dict) and choice.get("request_id") == request_ids[i]
                    and choice.get("context_id") == cell.get("id")
                    and choice.get("action") == action
                    and cell.get("split") == row.get("split") == "train",
                    "Native choice/request/TRAIN association changed")
            value = packet["rewards"][i]
            require(type(value) in (int, float) and math.isfinite(value), "Invalid active native reward")
        else:
            require(choice is None and cell.get("split") == row.get("split") == "inactive"
                    and invalid is not None and packet["rewards"][i] is None
                    and packet["reward_features"][i] is None,
                    "Inactive native slot was charged, scored or released")
            # The unchanged whole-farm scorer expects one numeric reward per
            # parked slot. Never modify active rewards or original packet bytes.
            temporary["rewards"][i] = -1.
    physical = verify_native_trial(Path(native_directory), temporary, scene, clip)
    require(packet == original, "Continuation score audit mutated source packet")
    rows = packet["rows"][:count]
    results = [dict(request_id=request_ids[i], released=row["invalid"] is None,
                    decline_reason=row["invalid"], captured=row["route_success"],
                    clean=row["route_quality"] == 2, **{"return": packet["rewards"][i]})
               for i, row in enumerate(rows)]
    released = sum(r["released"] for r in results)
    require(physical["rescored_releases"] == released, "Paid release count differs from original scorer")
    return dict(synthetic=packet["synthetic"], requests=count, releases=released,
                rejections=count-released, captures=sum(r["captured"] for r in results),
                clean_captures=sum(r["clean"] for r in results), terminal_results=results,
                physical=physical,
                scope="Independent original 240 Hz scorer/reward replay; inactive parked slots excluded from paid counts. Geometry, motion, source and journal audits remain separate.")
