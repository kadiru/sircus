"""Transfer distinct successful settings, retaining their contact-route provenance.

This is a finite proposal menu, not a prediction of arbitrary robot rebounds.
Only authenticated original pre-IK hints enter features and proposals; fresh
exact arm/path checks remain required. Solved historical joints identify settings.
"""
from __future__ import annotations
from collections import defaultdict
import copy
import math

from .benchmark_verify import digest
from .expanded_actions import WIDE_BOUNDS

VERSION = "sircus-expanded-native-shot-menu-v1"
MENU_LIMIT = 56
NEW_LIMIT = 48


def key(row):
    return digest(dict(values=row["values"], heading=row["board_heading_offset_deg"] % 360,
        twist=row["board_twist_deg"] % 360, q10=row.get("seed_q10"), q14=row.get("seed_q14")))


def source_order(rows, target):
    """Round-robin route category and nearby position before repeated settings."""
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row.get("split") != "train":
            raise ValueError("Only TRAIN settings may enter the expanded candidate library")
        category = "+".join(sorted(row.get("observed_routes", []))) or row.get("observed_route", "other")
        grouped[category][tuple(row["bucket_xy_m"])].append(row)
    queues = {}
    for category, positions in grouped.items():
        nearest = sorted(positions, key=lambda xy:(math.dist(xy, target), xy))[:8]
        for xy in nearest:
            positions[xy].sort(key=lambda r:(-r.get("capture_probability", 0), r["id"]))
        queues[category] = [positions[xy][index] for index in range(4) for xy in nearest if index < len(positions[xy])]
    result = []
    for index in range(32):
        for category in sorted(queues):
            if index < len(queues[category]):
                result.append(queues[category][index])
    return result


def expanded_menu(layout, libraries, checkpoint):
    from .neural_shot_plan import menu
    baseline_libraries = libraries["baseline_libraries"]
    if "expanded_menu_version" in baseline_libraries:
        raise ValueError("Recursive expanded-menu configuration")
    target = layout["bucket_xy_m"]
    phase = "test" if layout["split"] == "test" else "train"
    ordered = source_order(libraries["expanded_sources"], target)[:16]
    rows, seen = [], set()
    # Fixed and translated board centers offer different arm reach tradeoffs.
    # Complete source height/tilt/gap/heading/twist stay fixed. No native outcome
    # at the queried target participates in this deterministic proposal menu.
    for shift in (1., 0., .5):
        for source in ordered:
            values = list(source["values"])
            values[:2] = [values[i] + shift*(target[i]-source["bucket_xy_m"][i]) for i in range(2)]
            if any(not lo <= v <= hi for v, lo, hi in zip(values, WIDE_BOUNDS["low"], WIDE_BOUNDS["high"], strict=True)):
                continue
            if math.dist(values[:2], target) <= .12:
                continue
            proposal = dict(values=values, bucket_xy_m=list(target),
                board_heading_offset_deg=source["board_heading_offset_deg"], board_twist_deg=source["board_twist_deg"],
                intended_route=source["intended_route"],
                phase=phase, role=phase, method="expanded_library", group=f"expanded_translation_{shift:g}",
                training_recipe_id=source["id"], training_bucket_xy_m=source["bucket_xy_m"],
                training_distance_m=math.dist(target, source["bucket_xy_m"]),
                source_observed_route=source.get("observed_routes", source.get("observed_route")), source_row_sha256=source.get("source_row_sha256"),
                source_setting_key=source["dedup_key"])
            for seed in ("seed_q10", "seed_q14"):
                if seed in source:proposal[seed] = list(source[seed])
            if ("seed_q10" in proposal) != ("seed_q14" in proposal):
                raise ValueError("Both original seed vectors must be present or absent")
            identity = key(proposal)
            if identity in seen:
                continue
            seen.add(identity)
            rows.append(proposal)
            if len(rows) >= NEW_LIMIT:
                break
        if len(rows) >= NEW_LIMIT:
            break
    for proposal in menu(layout, baseline_libraries, checkpoint):
        if key(proposal) not in seen:
            seen.add(key(proposal))
            rows.append(copy.deepcopy(proposal))
        if len(rows) >= MENU_LIMIT:
            break
    return [dict(row, id=f"{layout['id']}-expanded-menu-{i:03d}", context_id=layout["id"],
                 dataset_split=layout["split"]) for i, row in enumerate(rows)]
