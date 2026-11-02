"""Engine-independent Sircus shots and scoring, reusing the Isaac study's scorer."""

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np

ISAAC = Path(__file__).resolve().parents[1] / "isaac"
if ISAAC.is_dir():
    sys.path.insert(0, str(ISAAC))
from board_route import BoardRoute, VERSION as ROUTE_VERSION
from entry_quality import EntryQuality
from rl_contract import ShotScore, VERSION as SCORER_VERSION

VERSION = "sircus-engine-shot-parity-v1"
FORCE_NAMES = ("board", "bucket", "table", "floor", "robot", "unattributed")
ACTION_KEYS = (
    "id",
    "azimuth_offset_deg",
    "azimuth_deg",
    "board_x_m",
    "board_y_m",
    "height_m",
    "gap_m",
    "tilt_deg",
)
CATEGORIES = ("clean_capture", "assisted_capture", "rim_contact_miss", "other_miss")


def canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False).encode()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def category(row):
    if row["route_success"]:
        return "clean_capture" if row["route_quality"] == 2 else "assisted_capture"
    return "rim_contact_miss" if row["rim_band_contact"] else "other_miss"


def verify_manifest(manifest):
    payload = dict(manifest)
    fingerprint = payload.pop("fingerprint")
    if (
        manifest["version"] != VERSION
        or hashlib.sha256(canonical(payload)).hexdigest() != fingerprint
    ):
        raise ValueError("Benchmark manifest or fingerprint changed")
    if manifest["dt_s"] != 1 / 240 or len(
        {s["shot_id"] for s in manifest["shots"]}
    ) != len(manifest["shots"]):
        raise ValueError("Wrong physics cadence or duplicate shot IDs")
    if manifest["scorer_sha256"] != {
        name: sha256(Path(sys.modules[name].__file__))
        for name in ("rl_contract", "entry_quality", "board_route")
    }:
        raise ValueError("Installed scoring code differs from the benchmark")
    return manifest


def select_benchmark(source, output=None, count=24, repeats=3, seed=20261102):
    """Freeze unique, outcome-stratified original actions without new optimization."""
    source = Path(source)
    spec = json.loads((source / "contract.json").read_text())
    parent = json.loads((source / "result.json").read_text())
    payload = dict(spec)
    fingerprint = payload.pop("fingerprint")
    if (
        hashlib.sha256(canonical(payload)).hexdigest() != fingerprint
        or parent["fingerprint"] != fingerprint
    ):
        raise ValueError("Original Isaac source contract changed")
    if (
        count <= 0
        or repeats <= 0
        or spec["scene"]["simulation"]["physics_dt_s"] != 1 / 240
    ):
        raise ValueError("Expected positive experiment size at 240 Hz")
    groups = {key: [] for key in CATEGORIES}
    seen = set()
    for clip in sorted(source.glob("clip-*.json")):
        clip_sha = sha256(clip)
        data = json.loads(clip.read_text())
        for row in data["rollout"]:
            if row["invalid"]:
                continue
            signature = canonical(
                [row["bucket_xy_m"], [row[k] for k in ACTION_KEYS if k != "id"]]
            )
            if signature in seen:
                continue
            seen.add(signature)
            key = category(row)
            groups[key].append(
                dict(
                    shot_id=f"{clip.stem}-{row['cell_id']}",
                    cell_id=row["cell_id"],
                    category=key,
                    action={k: row[k] for k in ACTION_KEYS},
                    bucket_xy_m=row["bucket_xy_m"],
                    release_ball_center_m=row["release_ball_center_m"],
                    settled_board_center_m=row["settled_board_center_m"],
                    settled_board_normal=row["settled_board_normal"],
                    final_joint_positions_rad=row["final_joint_positions_rad"],
                    isaac_reference=row,
                    source=dict(clip=clip.name, sha256=clip_sha),
                )
            )
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)
    available = {key: list(value) for key, value in groups.items() if value}
    if sum(map(len, available.values())) < count:
        raise ValueError("Insufficient distinct valid original shot configurations")
    chosen = []
    while len(chosen) < count:
        for group in available.values():
            if group and len(chosen) < count:
                chosen.append(group.pop())
    rng.shuffle(chosen)
    manifest = dict(
        version=VERSION,
        source_fingerprint=fingerprint,
        source_result_sha256=sha256(source / "result.json"),
        scoring_version=SCORER_VERSION,
        route_version=ROUTE_VERSION,
        scorer_sha256={
            name: sha256(Path(sys.modules[name].__file__))
            for name in ("rl_contract", "entry_quality", "board_route")
        },
        scene=spec["scene"],
        dt_s=1 / 240,
        horizon_steps=960,
        terminal_conditions="Freeze scoring at first capture success or ball-center Z below -0.2 m, as in ParallelShots.",
        repeats=repeats,
        seed=seed,
        source_unique_counts={k: len(v) for k, v in groups.items()},
        selection="Round-robin outcome strata, unique geometry/actions, seeded shuffle; diagnostic success rates are selected by source outcome.",
        repeat_protocol="Unperturbed resets; repeatability check, not physical robustness or generalization.",
        joint_state_timing="final_joint_positions_rad was measured at the end of the source throw; exact source release joint angles were not recorded.",
        comparison_protocol="fixed_endpoint_contact_replay; fresh paired engines use the same initialization and prescribed release, with joints held fixed.",
        shots=chosen,
    )
    manifest["fingerprint"] = hashlib.sha256(canonical(manifest)).hexdigest()
    if output is not None:
        Path(output).write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


class ScoreStream:
    """Feed local tabletop coordinates and attributed contact-force norms at 240 Hz."""

    def __init__(self, scene, bucket_xy_m, normal):
        self.dt = 1 / 240
        if scene["simulation"]["physics_dt_s"] != self.dt:
            raise ValueError("Scoring requires the original 240 Hz physics cadence")
        self.normal = np.asarray(normal, dtype=float)
        if (
            self.normal.shape != (3,)
            or not np.isfinite(self.normal).all()
            or not np.isclose(np.linalg.norm(self.normal), 1, atol=1e-5)
        ):
            raise ValueError("Expected a finite unit board normal")
        bucket = dict(scene["bucket"], center_xy_m=list(bucket_xy_m))
        radius = scene["ball"]["radius_m"]
        self.capture = ShotScore(bucket, radius)
        self.quality = EntryQuality(bucket, radius)
        self.route = BoardRoute(self.dt)
        self.steps = 0
        self.finished = False

    def step(self, position, velocity, forces):
        if self.finished:
            return
        p, v = np.asarray(position, dtype=float), np.asarray(velocity, dtype=float)
        if p.shape != (3,) or v.shape != (3,) or not np.isfinite([p, v]).all():
            raise ValueError("Invalid ball position or velocity")
        if set(forces) != set(FORCE_NAMES):
            raise ValueError("Contact attribution requires every force category")
        if any(not np.isfinite(value) or value < 0 for value in forces.values()):
            raise ValueError("Invalid contact force norm")
        self.capture.step(p, float(forces["board"]), self.dt, float(v @ self.normal))
        self.quality.step(p, float(forces["bucket"]), self.capture.entered)
        self.route.step(forces)
        self.steps += 1
        self.finished = bool(self.capture.success or p[2] < -0.2)

    def result(self):
        result = self.capture.result()
        quality = self.quality.result(result["success"])
        route = self.route.result(result["success"], quality["quality_score"])
        return dict(
            result,
            **quality,
            **route,
            complete_success=route["route_success"],
            scoring_version=SCORER_VERSION,
            physics_steps=self.steps,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    manifest = select_benchmark(args.source, args.output, args.count, args.repeats)
    print(
        json.dumps(
            dict(
                fingerprint=manifest["fingerprint"],
                shots=len(manifest["shots"]),
                repeats=manifest["repeats"],
            )
        )
    )
