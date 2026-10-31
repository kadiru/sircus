"""Offline evaluation report/figure. Never imported by the sensor estimator."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from perception_worker import load_sensor
from validation_contract import paired_source, summarize, verify_pair


def build(run_dir, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir = Path(run_dir)
    output = Path(output)
    summary = json.loads((run_dir / "summary.json").read_text())
    if summary["state"] != "completed":
        raise ValueError("Report requires a completed study")
    (output / "result.json").write_text(
        json.dumps(
            {
                "passed": bool(summary["completed_observations"]),
                "state": "perception_validation_completed",
                "run": run_dir.name,
                "observations": summary["completed_observations"],
                "scope": "Infrastructure completion only; detector accuracy is in validation.json",
            },
            indent=2,
        )
    )
    records = json.loads((run_dir / "records.json").read_text())
    checked = summarize(
        records, summary["planned_observations"], summary["excluded_layouts"]
    )
    assert checked["conditions"] == summary["conditions"], (
        "Stored aggregate does not match raw records"
    )
    # Evaluation-only integrity/readback checks. These do not alter estimates.
    truths = [
        json.loads(p.read_text())
        for p in sorted((run_dir / "evaluation_truth").glob("*.json"))
    ]
    assert len(truths) == len(records)
    for t in truths:
        assert t["lighting_readback"]["intensity"] == t["condition"]["intensity"]
        np.testing.assert_allclose(
            t["lighting_readback"]["color"], t["condition"]["color"], atol=1e-6
        )
    paired_checks = 0
    for t in truths:
        source = paired_source(t["condition"]["name"])
        if not source:
            continue
        ident = f"{t['layout']['layout']:03d}"
        noisy = load_sensor(
            run_dir / "sensors" / f"{ident}-{t['condition']['name']}.npz"
        )
        base = load_sensor(run_dir / "sensors" / f"{ident}-{source}.npz")
        baseline_truth = json.loads(
            (run_dir / "evaluation_truth" / f"{ident}-{source}.json").read_text()
        )
        verify_pair(noisy, base, t, baseline_truth)
        paired_checks += 1
    diagnostics = {
        "exact_paired_sensor_checks": paired_checks,
        "worker_timeouts": sum(r["worker_timeout"] for r in records),
        "unique_layouts": len({r["layout"] for r in records}),
        "unique_frame_ids": len({r["frame_id"] for r in records}),
        "lighting_readbacks_verified": len(truths),
        "rgb_channel_means_by_condition": {
            name: np.mean(
                [
                    t["rendered_rgb_mean"]
                    for t in truths
                    if t["condition"]["name"] == name
                ],
                axis=0,
            ).tolist()
            for name in summary["conditions"]
        },
        "visible_ball_rule": "evaluation-only clean depth: >=20 sphere-agreeing pixels, 5mm tolerance",
    }
    names = list(summary["conditions"])
    objects = ["table", "board", "bucket", "ball"]
    counts = np.array(
        [
            [summary["conditions"][c]["objects"][o]["estimated"] for c in names]
            for o in objects
        ]
    )
    totals = np.array(
        [
            [summary["conditions"][c]["objects"][o]["total"] for c in names]
            for o in objects
        ]
    )
    fractions = np.divide(
        counts, totals, out=np.zeros_like(counts, dtype=float), where=totals > 0
    )
    fig, ax = plt.subplots(figsize=(11, 4.2), layout="constrained")
    im = ax.imshow(fractions, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
    for i in range(4):
        for j in range(len(names)):
            ax.text(
                j,
                i,
                f"{counts[i, j]}/{totals[i, j]}",
                ha="center",
                va="center",
                color="white" if fractions[i, j] > 0.55 else "#11252b",
                fontsize=13,
            )
    ax.set_xticks(range(len(names)), [n.replace("_", "\n") for n in names])
    ax.set_yticks(range(4), objects)
    ax.set_title(
        "Frozen RGB-D detector: detection availability across randomized layouts"
    )
    fig.colorbar(im, ax=ax, label="Fraction estimated (not recall)")
    fig.savefig(run_dir / "availability.png", dpi=150)
    plt.close(fig)
    (output / "validation-chart.tmp.png").write_bytes(
        (run_dir / "availability.png").read_bytes()
    )
    (output / "validation-chart.tmp.png").replace(output / "validation-chart.png")
    chart_meta = {"run": run_dir.name, "contract": summary["contract"]}
    (output / "validation-chart.tmp.json").write_text(json.dumps(chart_meta))
    (output / "validation-chart.tmp.json").replace(output / "validation-chart.json")
    (run_dir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
    manifest = {
        str(p.relative_to(run_dir)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(run_dir.rglob("*"))
        if p.is_file() and p.name != "manifest.json"
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    build(a.run, a.output)
