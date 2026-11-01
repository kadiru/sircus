"""Finalize a completed diagnostic report without changing original trial records."""

import argparse
import json
from pathlib import Path


def finalize(run):
    run = Path(run)
    summary = json.loads((run / "summary.json").read_text())
    rows = [json.loads(s) for s in (run / "shots.jsonl").read_text().splitlines()]
    if summary["state"] != "completed" or len(rows) != summary["planned"]:
        raise ValueError("Incomplete sweep")
    for d in summary["distances"]:
        selected = [
            r
            for r in rows
            if r["distance_m"] == d["distance_m"] and r["kind"] == "grid"
        ]
        d["grid_cells"] = [
            {
                "tilt_deg": r["action"][0],
                "gap_m": r["action"][2],
                "success": r["success"],
                "invalid": r.get("invalid"),
                "reward": r["reward"],
            }
            for r in selected
        ]
        expected = {
            (t, g)
            for t in [12, 16, 20, 24, 28, 36, 45]
            for g in [0.06, 0.10, 0.15, 0.20, 0.25]
        }
        actual = {(r["action"][0], r["action"][2]) for r in selected}
        if len(selected) != 35 or actual != expected:
            raise ValueError("Missing grid cells or duplicated actions")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--output", required=True)
    p.add_argument("--conclusion", required=True)
    args = p.parse_args()
    result = finalize(args.run)
    result["conclusion"] = args.conclusion
    path = Path(args.output)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=2, allow_nan=False))
    tmp.replace(path)
