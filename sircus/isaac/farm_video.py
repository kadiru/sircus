"""Compose real PPO training batches into a 32-table farm video."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from farm_report import verify
from overhead_video import project

W, H, FPS = 2560, 1600, 30


def encode(run):
    from PIL import Image, ImageDraw, ImageFont

    report = verify(run)
    if report["contract"]["num_envs"] != 32:
        raise ValueError("The farm video must contain all 32 workcells")
    fonts = {
        s: ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", s)
        for s in (19, 23, 28, 34, 46)
    }
    output = run / "sircus-32-table-ppo.mp4"
    temp = output.with_suffix(".tmp.mp4")
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{W}x{H}",
        "-r",
        str(FPS),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-threads",
        "4",
        "-preset",
        "fast",
        "-crf",
        "19",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(temp),
    ]
    total = 0
    with subprocess.Popen(command, stdin=subprocess.PIPE) as proc:
        for clip in report["clips"]:
            folder = run / clip["path"]
            meta = json.loads((folder / "clip.json").read_text())
            samples = json.loads((folder / "samples.json").read_text())
            speed = FPS * meta["dt_s"] * meta["capture_stride"]
            if not np.isclose(speed, 0.5):
                raise ValueError(
                    "Expected exact half-speed video without interpolation"
                )

            def frame(i):
                canvas = Image.new("RGB", (W, H), "#111c2b")
                d = ImageDraw.Draw(canvas)
                d.text(
                    (28, 16),
                    "SIRCUS  /  32 TABLES + 64 ARMS LEARNING TOGETHER",
                    fill="#f2f6fa",
                    font=fonts[46],
                )
                d.text(
                    (28, 72),
                    f"Isaac Lab / PhysX   |   PPO iteration {meta['iteration']}/{meta['total_updates']}   |   {meta['throws_before']:,} throws before this batch   |   0.5x playback",
                    fill="#aec1d3",
                    font=fonts[28],
                )
                for v in meta["views"]:
                    image = Image.open(folder / f"{v['name']}-{i:04d}.jpg").convert(
                        "RGB"
                    )
                    overlay = Image.new("RGBA", image.size)
                    pen = ImageDraw.Draw(overlay)
                    selected = range(32) if v["cell"] is None else [v["cell"]]
                    for j in selected:
                        if meta["rows"][j]["invalid"]:
                            continue
                        points = [
                            s["positions_m"][j] for s in samples[max(0, i - 60) : i + 1]
                        ]
                        uv = project(points, samples[i]["cameras"][v["name"]], v)
                        if len(uv) > 1:
                            pen.line(
                                [tuple(p) for p in uv],
                                fill=(240, 255, 250, 200),
                                width=2,
                            )
                        x, y = uv[-1]
                        pen.ellipse(
                            (x - 4, y - 4, x + 4, y + 4),
                            outline=(114, 228, 207, 255),
                            width=2,
                        )
                    image = Image.alpha_composite(
                        image.convert("RGBA"), overlay
                    ).convert("RGB")
                    if v["cell"] is None:
                        canvas.paste(image, (0, 120))
                        points = np.array(meta["cell_origins_m"]) + [0.67, -0.25, 1.1]
                        labels = project(points, samples[i]["cameras"][v["name"]], v)
                        for j, (x, y) in enumerate(labels):
                            if 0 <= x < 1920 and 0 <= y < 1440:
                                d.text(
                                    (x - 10, y + 120),
                                    str(j + 1).zfill(2),
                                    fill="#76e6d1",
                                    font=fonts[19],
                                    stroke_width=1,
                                    stroke_fill="#172838",
                                )
                    else:
                        slot = {"near": 0, "middle": 1, "far": 2}[v["name"]]
                        x, y = 1920, 120 + slot * 480
                        cell = meta["cells"][v["cell"]]
                        row = meta["rows"][v["cell"]]
                        d.text(
                            (x + 14, y + 8),
                            f"TABLE {v['cell'] + 1:02d}  /  {cell['distance_m'] * 1000:.0f} mm DISTANCE",
                            fill="#72e4cf",
                            font=fonts[23],
                        )
                        canvas.paste(image, (x, y + 40))
                        result = (
                            "IK REJECTED / NO RELEASE"
                            if row["invalid"]
                            else ("CAPTURE" if row["route_success"] else "MISS")
                        )
                        label = f"Batch outcome: {result}"
                        d.text((x + 14, y + 444), label, fill="#b4c4d7", font=fonts[23])
                counts = meta["statistics"]
                d.rectangle((20, 1484, 1876, 1544), fill="#1d3042")
                d.text(
                    (34, 1495),
                    f"BATCH RESULT  {counts['captures']}/32 captured   |   {counts['invalid']} IK rejects   |   simulation t = {samples[i]['time_s']:.2f} s",
                    fill="#f2f6fa",
                    font=fonts[28],
                )
                d.text(
                    (28, 1566),
                    "Actual training rollouts. Policy learns height / gap / tilt; arms reset to solved poses. Trails = simulator positions. No pickup or transit learning.",
                    fill="#aec1d3",
                    font=fonts[23],
                )
                return canvas

            order = [0] * 30 + list(range(len(samples))) + [len(samples) - 1] * 30
            previous = None
            for i in order:
                if previous != i:
                    image = frame(i)
                    data = image.tobytes()
                    previous = i
                    if i == 30:
                        image.save(
                            run / f"iteration-{meta['iteration']:03d}-poster.jpg",
                            quality=94,
                        )
                proc.stdin.write(data)
                total += 1
            print(
                json.dumps(dict(iteration=meta["iteration"], output_frames=total)),
                flush=True,
            )
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError("Video encoder failed")
    temp.replace(output)
    audit = dict(
        run=run.name,
        frames=total,
        fps=FPS,
        duration_s=total / FPS,
        resolution=[W, H],
        sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        recorded_iterations=[c["iteration"] for c in report["clips"]],
        playback_speed=0.5,
        scope="Actual on-policy PPO rollouts consumed by the recorded optimizer updates; not successful-shot replay.",
    )
    (run / "video.json").write_text(json.dumps(audit, indent=2) + "\n")
    return audit


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    print(json.dumps(encode(p.parse_args().run)))
