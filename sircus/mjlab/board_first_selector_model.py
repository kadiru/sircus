"""Retrain the same selector using an explicit board-first label sidecar.

Original observations stay byte-for-byte unchanged in the dataset records.
Only TRAIN labels enter gradients; validation still selects an epoch. The
architecture and geometry-only inference are exactly the historical model's.
"""
from __future__ import annotations

import copy
from collections import Counter

from . import neural_shot_model as base
from . import board_first_score as score

VERSION = "sircus-board-first-selector-labels-v1"


def outcome(record):
    # Validate original setup/native provenance independently of the new label.
    previous = base.outcome(record)
    if previous in {"setup_rejected", "setup_only"}:
        if record.get("board_first_score") is not None:
            raise ValueError("An unobserved proposal cannot have a new capture label")
        return previous
    revised = record.get("board_first_score")
    if not isinstance(revised, dict) or revised.get("scoring_version") != score.VERSION:
        raise ValueError("Explicit audited board-first score required")
    for key in ("invalid_physics", "route_success"):
        if type(revised.get(key)) is not bool:
            raise ValueError("Board-first validity and capture must be booleans")
    if revised["invalid_physics"] != record["observation"]["invalid_physics"]:
        raise ValueError("Rescoring cannot alter physics validity")
    if revised["invalid_physics"]:
        return "invalid_physics"
    # Do not relabel an arm rebound as direct/table. The model's capture head
    # needs a binary target, while the sidecar retains the actual contact route.
    return "unclassified_capture" if revised["route_success"] else "flight_miss"


def attach_scores(records, revised_by_id):
    """Require one matching new label for every released TRAIN/VAL observation."""
    result = copy.deepcopy(records)
    for record in result:
        if record.get("split") not in {"train", "validation"}:
            raise ValueError("Historical TEST observations cannot enter fitting")
        native = record.get("observation")
        if native is not None:
            item = revised_by_id[native["id"]]
            original = item["original"]
            for key in ("id", "bucket_xy_m", "values"):
                if original.get(key) != native.get(key):
                    raise ValueError("Revised score identifies a different native command: " + key)
            for key in ("q10", "q14", "board_normal", "native_cache_sha256", "native_cache_row"):
                if key in original and original[key] != native.get(key):
                    raise ValueError("Revised score differs from exact native evidence: " + key)
            if item.get("status") != "scored":
                raise ValueError("A missing trace cannot become a new training label")
            if (item.get("native_sha256") != native.get("native_cache_sha256")
                    or item.get("native_row") != native.get("native_cache_row")):
                raise ValueError("Revised score must bind the exact native archive and column")
            record["board_first_score"] = copy.deepcopy(item["new_score"])
            record["board_first_evidence"] = {key: item[key] for key in ("native_sha256", "native_row")}
        outcome(record)
    return result


def fit(training_rows, validation_rows, *, seed=2026111707, epochs=base.DEFAULT_EPOCHS):
    return base.fit(training_rows, validation_rows, seed=seed, epochs=epochs,
                    labeler=outcome, label_version=VERSION)


def label_changes(rows):
    counts = Counter()
    positions = {}
    for record in rows:
        before, after = base.outcome(record), outcome(record)
        old_hit = before in {"direct", "table", "unclassified_capture"}
        new_hit = after == "unclassified_capture"
        label = "added" if new_hit and not old_hit else "removed" if old_hit and not new_hit else "unchanged"
        counts[label] += 1
        if label != "unchanged":
            xy = tuple(record["request"]["bucket_xy_m"])
            positions.setdefault(xy, Counter())[label] += 1
    return dict(counts=counts, positions=[dict(bucket_xy_m=list(xy), **value)
                                         for xy, value in sorted(positions.items())])
