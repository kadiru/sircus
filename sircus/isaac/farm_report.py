"""Verify PPO episode/clip provenance and export compact synthetic study evidence."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from farm_contract import FIXED, actions, contract, observation, reward, splits, statistics


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(run):
    r = read(run / "progress.json")
    if r["run"] != run.name or r["phase"] != "completed":
        raise ValueError("Not this run's completed training report")
    c = read(run / "contract.json")
    expected = contract(
        c["scene"], c["updates"], c["batches_per_update"], c["num_envs"]
    )
    if any(c[k] != v for k, v in expected.items()):
        raise ValueError("Contract changed")
    if read(run / "splits.json") != splits(c["num_envs"]):
        raise ValueError("Split provenance mismatch")
    for name, key in [
        ("episodes.jsonl", "episodes_sha256"),
        ("splits.json", "splits_sha256"),
        ("selected.pt", "selected_sha256"),
        ("latest.pt", "final_sha256"),
    ]:
        if digest(run / name) != r[key]:
            raise ValueError("Artifact changed: " + name)
    batches = [
        json.loads(line) for line in (run / "episodes.jsonl").read_text().splitlines()
    ]
    expected_order = [
        (i, b)
        for i in range(1, c["updates"] + 1)
        for b in range(c["batches_per_update"])
    ]
    if [(b["iteration"], b["batch"]) for b in batches] != expected_order:
        raise ValueError("Missing or reordered training batches")
    episode = 0
    before = r["initial_weight_hash"]
    for curve in r["learning_curve"]:
        if curve["optimization"]["before"] != before:
            raise ValueError("Broken policy-update hash chain")
        if curve["optimization"]["before"] == curve["optimization"]["after"]:
            raise ValueError("An advertised optimizer step made no change")
        before = curve["optimization"]["after"]
    if before != r["final_weight_hash"] or r["actor_change_l1"] <= 0:
        raise ValueError("No verified final actor change")
    for b in batches:
        if len(b["rows"]) != c["num_envs"]:
            raise ValueError("Wrong number of simultaneous cells")
        for row, cell in zip(b["rows"], splits(c["num_envs"])["train"]):
            episode += 1
            if row["episode"] != episode or row["cell_id"] != cell["id"]:
                raise ValueError("Episode/cell mismatch")
            np.testing.assert_allclose(
                row["observation"], observation(cell), atol=1e-7, rtol=0
            )
            chosen = actions(np.array(row["latent"], dtype=np.float32))[0]
            for k in ("height_m", "gap_m", "tilt_deg"):
                # NumPy float32 tanh can differ by a few ULPs across SIMD/library versions.
                if not np.isclose(row[k], chosen[k], rtol=2e-7, atol=1e-8):
                    raise ValueError("Action does not come from the sampled latent")
            if row["ppo_reward"] != reward(row):
                raise ValueError("Training reward mismatch")
    if episode != c["training_throws"] or episode != r["training_throws"]:
        raise ValueError("Training throw count mismatch")
    for curve in r["learning_curve"]:
        rows = [
            row
            for b in batches
            if b["iteration"] == curve["iteration"]
            for row in b["rows"]
        ]
        if any(curve[k] != v for k, v in statistics(rows).items()):
            raise ValueError("Learning curve mismatch")
    for clip in r["clips"]:
        path = run / clip["path"]
        m = read(path / "clip.json")
        samples = read(path / "samples.json")
        rows = next(
            b["rows"]
            for b in batches
            if (b["iteration"], b["batch"]) == (m["iteration"], 0)
        )
        if (
            m["rows"] != rows
            or m["weight_hash"]
            != r["learning_curve"][m["iteration"] - 1]["optimization"]["before"]
        ):
            raise ValueError("Video did not record the claimed training rollout")
        if m["policy_updates_applied"] != m["iteration"] - 1 or m[
            "samples_sha256"
        ] != digest(path / "samples.json"):
            raise ValueError("Clip provenance mismatch")
        if len(samples) != m["frames"] or not np.all(
            np.diff([s["step"] for s in samples]) == m["capture_stride"]
        ):
            raise ValueError("Missing or uneven camera samples")
        for camera in m["views"]:
            actual = sorted(p.name for p in path.glob(camera["name"] + "-*.jpg"))
            expected = [f"{camera['name']}-{i:04d}.jpg" for i in range(len(samples))]
            if actual != expected:
                raise ValueError("Camera frame sequence mismatch")
    val = [read(run / f"validation-{v['updates']:03d}.json") for v in r["validation"]]
    for meta, raw in zip(r["validation"], val):
        if any(meta[k] != v for k, v in statistics(raw["rows"]).items()):
            raise ValueError("Validation report mismatch")
        expected_hash = (
            r["initial_weight_hash"]
            if meta["updates"] == 0
            else r["learning_curve"][meta["updates"] - 1]["optimization"]["after"]
        )
        if raw["weight_hash"] != expected_hash or [
            row["cell_id"] for row in raw["rows"]
        ] != [cell["id"] for cell in splits(c["num_envs"])["validation"]]:
            raise ValueError("Validation policy or scenes changed")
    best = max(r["validation"], key=lambda v: (v["captures"], v["mean_reward"]))[
        "updates"
    ]
    if best != r["selected_update"]:
        raise ValueError("Selected policy does not match validation selection rule")
    test = {label: read(run / f"test-{label}.json") for label in r["evaluation"]}
    selected_hash = read(run / f"validation-{best:03d}.json")["weight_hash"]
    for label, expected_hash in (
        ("initial", r["initial_weight_hash"]),
        ("final", r["final_weight_hash"]),
        ("selected", selected_hash),
    ):
        if test[label]["weight_hash"] != expected_hash:
            raise ValueError("Frozen test does not use its advertised policy")
    for label, raw in test.items():
        if statistics(raw["rows"]) != r["evaluation"][label]:
            raise ValueError("Frozen test summary mismatch")
        if [row["cell_id"] for row in raw["rows"]] != [
            cell["id"] for cell in splits(c["num_envs"])["test"]
        ]:
            raise ValueError("Wrong final test scenes")
        if label == "fixed":
            for row in raw["rows"]:
                np.testing.assert_allclose(
                    [row[k] for k in ("height_m", "gap_m", "tilt_deg")],
                    FIXED,
                    rtol=0,
                    atol=1e-12,
                )
    if not all(p["all_passed"] for p in r["probes"]):
        raise ValueError("Contact validation failed")
    r["verification"] = dict(
        training_rows=episode,
        policy_hash_chain=True,
        actor_updated=True,
        video_rows_are_training_data=True,
        disjoint_splits=True,
        frozen_selection=True,
        validation_weights=True,
        fixed_actions=True,
        artifacts=True,
        contact_probes=True,
    )
    for p in r["probes"]:
        p.pop("peak_N", None)
    r["evaluation_by_distance"] = {}
    lookup = {c["id"]: c for c in splits(c["num_envs"])["test"]}
    for label, raw in test.items():
        group = {}
        for name, lo, hi in [("near", 0, 0.4), ("middle", 0.4, 0.5), ("far", 0.5, 2)]:
            rows = [
                row
                for row in raw["rows"]
                if lo <= lookup[row["cell_id"]]["distance_m"] < hi
            ]
            if rows:
                group[name] = statistics(rows)
        r["evaluation_by_distance"][label] = group
    return r


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    p.add_argument("output", type=Path)
    a = p.parse_args()
    r = verify(a.run)
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / "summary.json").write_text(json.dumps(r, indent=2) + "\n")
    print(json.dumps(r["verification"]))
