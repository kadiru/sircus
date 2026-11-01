"""Compose synchronized Isaac orthographic frames; never modifies simulated states."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

W, H, FPS = 1920, 1280, 30
INK, WHITE, MUTED, TEAL = "#101b29", "#f2f6fa", "#a6bacb", "#72e4cf"


def validate(run):
    meta = json.loads((run / "replay.json").read_text())
    if meta["run"] != run.name or meta["phase"] != "completed":
        raise ValueError("Expected this run's completed replay")
    if not meta.get("probes") or not all(p["all_passed"] for p in meta["probes"]):
        raise ValueError("Contact/cell-isolation verification did not pass")
    if any(r["invalid"] for r in meta["rows"]):
        raise ValueError("A replay endpoint failed settling/IK")
    for name, digest in meta["raw_sha256"].items():
        if hashlib.sha256((run / name).read_bytes()).hexdigest() != digest:
            raise ValueError("Raw replay evidence changed")
    samples = json.loads((run / "samples.json").read_text())
    if len(samples) != meta["frames"] or len(samples) < 2:
        raise ValueError("Incomplete recording")
    steps = np.diff([s["step"] for s in samples])
    if not np.all(steps == meta["capture_stride"]):
        raise ValueError("Uneven camera sampling; cannot claim a fixed playback speed")
    for c in meta["cameras"]:
        files = sorted(p.name for p in (run / "frames").glob(c["name"] + "-*.jpg"))
        if files != [f"{c['name']}-{i:04d}.jpg" for i in range(len(samples))]:
            raise ValueError("Camera sequences have missing/extra frames")
    return meta, samples


def project(points, camera, view):
    """True orthographic projection; ROS camera Y points down in the image."""
    local = (np.asarray(points) - camera["position_m"]) @ np.asarray(
        camera["rotation_ros"]
    )
    span = view["span_m"]
    return np.column_stack(
        (
            view["width"] * (0.5 + local[:, 0] / span),
            view["height"] * 0.5 + view["width"] * local[:, 1] / span,
        )
    )


def font(size, bold=False):
    from PIL import ImageFont

    base = Path("/usr/share/fonts/truetype/dejavu")
    return ImageFont.truetype(
        str(base / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")), size
    )


class Composer:
    def __init__(self, run):
        self.run = run
        self.meta, self.samples = validate(run)
        self.views = {v["name"]: v for v in self.meta["cameras"]}
        contacts = json.loads((run / "contact-samples.json").read_text())
        self.impacts = []
        for cell in range(4):
            self.impacts.append(
                [
                    next(
                        (
                            (c["step"] - self.samples[0]["step"]) * self.meta["dt_s"]
                            for c in contacts
                            if c["forces_N"][cell][kind] > 0.01
                        ),
                        None,
                    )
                    for kind in (0, 1)
                ]
            )
        self.f = {s: font(s) for s in (18, 22, 24, 28, 30, 34, 40, 48)}
        self.bold = font(40, True)

    def photo(self, view, i, trail=True):
        from PIL import Image, ImageDraw

        im = Image.open(self.run / "frames" / f"{view}-{i:04d}.jpg").convert("RGB")
        if trail and i:
            v = self.views[view]
            points = [s["positions_m"][v["cell"]] for s in self.samples[: i + 1]]
            uv = project(points, self.samples[i]["cameras"][view], v)
            # Measured simulator trail, explicitly labelled; not a predicted path.
            overlay = Image.new("RGBA", im.size)
            pen = ImageDraw.Draw(overlay)
            pen.line([tuple(p) for p in uv], fill=(255, 255, 255, 180), width=3)
            x, y = uv[-1]
            pen.ellipse(
                (x - 7, y - 7, x + 7, y + 7), outline=(22, 98, 101, 255), width=2
            )
            im = Image.alpha_composite(im.convert("RGBA"), overlay).convert("RGB")
        return im

    def state(self, cell, i):
        t = self.samples[i]["time_s"]
        board, bucket = self.impacts[cell]
        if i == len(self.samples) - 1:
            return "CAPTURED" if self.meta["rows"][cell]["route_success"] else "MISS"
        if bucket is not None and t >= bucket:
            return "BUCKET CONTACT / RETENTION CHECK"
        if board is not None and t >= board:
            return "BOARD BOUNCE"
        return "RELEASE / FALL"

    def frame(self, chapter, i):
        from PIL import Image, ImageDraw

        im = Image.new("RGB", (W, H), INK)
        d = ImageDraw.Draw(im)
        title = (
            "SIRCUS  /  FOUR TABLES, EIGHT ARMS"
            if chapter == "grid"
            else "SIRCUS  /  FOLLOW ONE BOUNCE"
        )
        d.text((28, 20), title, fill=WHITE, font=self.bold)
        speed = "0.5x" if chapter == "grid" else "0.25x"
        d.text(
            (28, 75),
            f"Orthographic Isaac Sim replay   |   {speed} slow motion   |   t = {self.samples[i]['time_s']:.3f} s",
            fill=MUTED,
            font=self.f[24],
        )
        if chapter == "grid":
            for j, (x, y) in enumerate(((24, 120), (972, 120), (24, 674), (972, 674))):
                photo = self.photo(f"cell{j}", i).resize(
                    (880, 550), getattr(Image, "Resampling", Image).LANCZOS
                )
                im.paste(photo, (x + 22, y))
                d.rectangle((x + 22, y, x + 902, y + 40), fill="#203242")
                c = self.meta["cells"][j]
                d.text(
                    (x + 36, y + 6),
                    f"{c['id'].upper()}  |  BUCKET {c['distance_m'] * 1000:.0f} mm AWAY",
                    fill=WHITE,
                    font=self.f[22],
                )
                d.rectangle((x + 22, y + 514, x + 902, y + 550), fill="#203242")
                d.text((x + 36, y + 519), self.state(j, i), fill=TEAL, font=self.f[22])
        else:
            im.paste(self.photo("cell0", i), (24, 190))
            for name, y, label in (
                ("top", 140, "TOP VIEW"),
                ("side", 600, "LOW FRONT VIEW"),
            ):
                im.paste(
                    self.photo(name, i).resize((576, 360), getattr(Image, "Resampling", Image).LANCZOS),
                    (1320, y + 40),
                )
                d.text((1320, y), label, fill=TEAL, font=self.f[28])
            d.text((38, 148), "DROPPER", fill=TEAL, font=self.f[28])
            d.text((1020, 148), "BOUNCER", fill=TEAL, font=self.f[28])
            d.text((32, 1030), self.state(0, i), fill=TEAL, font=self.f[34])
            c, a = self.meta["cells"][0], self.meta["actions"][0]
            release = self.meta["rows"][0]["release_ball_center_m"][2] * 1000
            d.text(
                (32, 1090),
                f"Board center {a['height_m'] * 1000:.0f} mm   |   release center {release:.1f} mm   |   tilt {a['tilt_deg']:.0f} degrees",
                fill=WHITE,
                font=self.f[30],
            )
            d.text(
                (32, 1140),
                f"Board-to-bucket distance {c['distance_m'] * 1000:.1f} mm   |   white trail = simulator ball position",
                fill=MUTED,
                font=self.f[24],
            )
        d.text(
            (28, 1240),
            "Simulation only. Arms start at solved poses; pickup and transit are not shown. Frozen settings, not new training.",
            fill=MUTED,
            font=self.f[22],
        )
        return im


def encode(run):
    c = Composer(run)
    dt = c.meta["dt_s"] * c.meta["capture_stride"]
    n = len(c.samples)
    output = run / "sircus-overhead.mp4"
    temp = output.with_suffix(".tmp.mp4")
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{W}x{H}",
        "-framerate",
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
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(temp),
    ]
    frames = 0
    with subprocess.Popen(cmd, stdin=subprocess.PIPE) as proc:
        for chapter, speed in (("grid", 0.5), ("detail", 0.25)):
            stride = round(speed / (FPS * dt))
            if not np.isclose(stride * FPS * dt, speed):
                raise ValueError("Requested speed cannot match captured timestamps")
            order = [0] * FPS + list(range(0, n, stride)) + [n - 1] * (2 * FPS)
            previous, data = None, None
            for i in order:
                if i != previous:
                    image = c.frame(chapter, i)
                    data = image.tobytes()
                    previous = i
                    if i == 0:
                        image.save(run / f"{chapter}-poster.jpg", quality=94)
                proc.stdin.write(data)
                frames += 1
            print(json.dumps({"chapter": chapter, "output_frames": frames}), flush=True)
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError("Video encoder failed")
    temp.replace(output)
    audit = {
        "run": c.meta["run"],
        "source_run": c.meta["source_run"],
        "output": output.name,
        "frames": frames,
        "fps": FPS,
        "duration_s": frames / FPS,
        "size": [W, H],
        "simulated_duration_s": c.samples[-1]["time_s"],
        "playback_speeds": [0.5, 0.25],
        "no_interpolation": True,
        "projection": c.meta["projection"],
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "endpoint_valid": [not r["invalid"] for r in c.meta["rows"]],
        "route_captured": [r["route_success"] for r in c.meta["rows"]],
        "board_contact_s": [t[0] for t in c.impacts],
        "bucket_contact_s": [t[1] for t in c.impacts],
    }
    (run / "video.json").write_text(json.dumps(audit, indent=2) + "\n")
    return audit


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    print(json.dumps(encode(p.parse_args().run)))
