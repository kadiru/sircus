"""Encode existing matched camera frames; does not run Isaac or alter trials."""

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path


def encode(run):
    if not re.fullmatch(r"parallel-world-\d{8}T\d{6}Z-n\d+", run.name):
        raise ValueError("Expected a named parallel-world run")
    summary = run / "summary.json"
    d = json.loads(summary.read_text())
    if d.get("run") != run.name:
        raise ValueError("Report belongs to a different run")
    if d["phase"] != "completed" or not d.get("verification"):
        raise ValueError("Export/verify completed evidence before publishing a video")
    frames = run / "frames"
    count = d["video_frames"]
    if count < 2:
        raise ValueError("No recorded batch")
    for view in ("overview", "close"):
        names = sorted(p.name for p in frames.glob(f"{view}-*.jpg"))
        if names != [f"{view}-{i:04d}.jpg" for i in range(count)]:
            raise ValueError("Missing, extra or unmatched camera frames")
    temp = run / "demo.tmp.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-threads",
            "2",
            "-framerate",
            "30",
            "-i",
            str(frames / "overview-%04d.jpg"),
            "-framerate",
            "30",
            "-i",
            str(frames / "close-%04d.jpg"),
            "-filter_complex",
            "[0:v][1:v]hstack=inputs=2[v]",
            "-map",
            "[v]",
            "-c:v",
            "libx264",
            "-threads",
            "2",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(temp),
        ],
        check=True,
    )
    temp.replace(run / "demo.mp4")
    d["video"] = {
        "run": d["run"],
        "url": f"/parallel-media/{d['run']}/demo.mp4",
        "sha256": hashlib.sha256((run / "demo.mp4").read_bytes()).hexdigest(),
        "frames": count,
        "fps": 30,
        "scope": "First search batch, overview and cell-0 camera at matched 240 Hz physics steps. Numerical successes are reported separately.",
    }
    summary.write_text(json.dumps(d, indent=2, allow_nan=False) + "\n")
    pointer = run.parent / "parallel-world.json"
    if pointer.exists() and json.loads(pointer.read_text())["run"] == d["run"]:
        tmp = pointer.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(d, allow_nan=False))
        tmp.replace(pointer)
    return d["video"]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    a = p.parse_args()
    print(json.dumps(encode(a.run)))
