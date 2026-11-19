"""Conservative trajectory warning; never replaces contact scoring or certifies direct shots."""

import hashlib
import json
import math


def low_external_rebound(points, bucket_xy, radius, outer_radius, tabletop_z=0.0):
    """Flag a sampled low Z reversal well outside the bucket's expanded footprint.

    Samples are only 30 Hz. Absence is inconclusive; collision identity is unknown.
    """
    for a, b, c in zip(points, points[1:], points[2:]):
        if (
            b[2] <= tabletop_z + radius + 0.030
            and a[2] - b[2] > 0.0001
            and c[2] - b[2] > 0.0001
            and math.dist(b[:2], bucket_xy) > outer_radius + radius + 0.010
        ):
            return True
    return False


def audit(report, trials_path):
    raw = trials_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    assert digest == report["artifact_sha256"]["trials.jsonl"], "Raw trial hash differs"
    expected = {
        r["trial_id"]: (e["distance_m"], r["success"])
        for e in report["distances"]
        for r in e["confirmation"]
    }
    scene = report["contract"]["parent_quality"]["parent"]["scene"]
    center = report["contract"]["parent_quality"]["parent"]["board_xy_m"]
    radius = scene["ball"]["radius_m"]
    outer_radius = scene["bucket"]["outer_top_diameter_m"] / 2
    tabletop_z = scene["table"]["center_m"][2] + scene["table"]["size_m"][2] / 2
    found = {}
    for line in raw.splitlines():
        r = json.loads(line)
        if r["stage"] != "confirmation":
            continue
        trial_id = r["trial_id"]
        assert trial_id in expected and trial_id not in found
        assert expected[trial_id] == (r["distance_m"], r["result"]["success"])
        points = r.get("trajectory_m", [])
        # Missing paths remain unknown, never treated as direct shots.
        found[trial_id] = (
            None
            if not points
            else low_external_rebound(
                points,
                [center[0], center[1] - r["distance_m"]],
                radius,
                outer_radius,
                tabletop_z,
            )
        )
    assert found.keys() == expected.keys(), "Missing confirmation records"
    distances = []
    for e in report["distances"]:
        success_ids = [r["trial_id"] for r in e["confirmation"] if r["success"]]
        distances.append(
            {
                "distance_m": e["distance_m"],
                "confirmed_captures": len(success_ids),
                "low_external_rebound_flags": sum(
                    found[i] is True for i in success_ids
                ),
                "missing_capture_paths": sum(found[i] is None for i in success_ids),
                "flagged_trial_ids": [i for i in success_ids if found[i] is True],
            }
        )
    return {
        "version": "sircus-sampled-external-rebound-warning-v1",
        "raw_trials_sha256": digest,
        "tabletop_z_m": tabletop_z,
        "sample_rate_hz": 1 / (scene["simulation"]["physics_dt_s"] * 8),
        "rule": "30Hz local Z minimum below tabletop+ball-radius+30mm, with >0.1mm fall/rise; XY beyond bucket outer-top radius+ball-radius+10mm",
        "scope": "Warning only: likely tabletop or another external contact. No table/arm contact sensor was recorded. No flag does not certify a direct board-to-bucket shot. Original capture and quality definitions are unchanged.",
        "distances": distances,
    }
