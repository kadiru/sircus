"""Independent CPU estimator: reads sensor bundles only, never simulator truth."""

import argparse
import datetime
import json
import time
from pathlib import Path

import numpy as np
from perception_estimator import COLORS, estimate

KEYS = {"rgb", "depth_m", "intrinsics", "camera_to_world", "frame_id", "timestamp_utc"}


def load_sensor(path):
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != KEYS:
            raise ValueError("sensor bundle has forbidden or missing fields")
        return {k: data[k].copy() for k in KEYS}


def process(data):
    from PIL import Image, ImageDraw

    start = time.monotonic()
    result, masks = estimate(
        data["rgb"],
        data["depth_m"],
        data["intrinsics"],
        data["camera_to_world"],
        str(data["frame_id"]),
        str(data["timestamp_utc"]),
    )
    result["processing_ms"] = (time.monotonic() - start) * 1000
    rgb = data["rgb"].astype(float)
    for name, mask in masks.items():
        rgb[mask] = 0.65 * rgb[mask] + 0.35 * np.array(COLORS[name])
    image = Image.fromarray(np.uint8(rgb))
    draw = ImageDraw.Draw(image)
    camera_from_world = np.linalg.inv(data["camera_to_world"])
    k = data["intrinsics"]

    def pixel(p):
        v = camera_from_world[:3, :3] @ p + camera_from_world[:3, 3]
        if v[2] <= 0:
            return None
        return tuple((k @ v)[:2] / v[2])

    for name, obj in result["objects"].items():
        if obj["status"] != "estimated":
            continue
        center = np.array(obj["center_m"])
        p = pixel(center)
        if p is None:
            continue
        color = COLORS[name]
        draw.ellipse([p[0] - 4, p[1] - 4, p[0] + 4, p[1] + 4], outline=color, width=2)
        draw.text(
            (p[0] + 6, p[1] - 13),
            name.upper(),
            fill=color,
            stroke_width=1,
            stroke_fill="black",
        )
        if "normal" in obj:
            end = pixel(center + np.array(obj["normal"]) * 0.12)
            if end:
                draw.line([p, end], fill=color, width=3)
    draw.rectangle((0, 0, image.width, 22), fill=(12, 30, 35))
    draw.text(
        (6, 5),
        "RGB-D ESTIMATES | cyan: table  green: board  gold: bucket  pink: ball",
        fill="white",
    )
    return result, image


def atomic(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False))
    temp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--once", action="store_true")
    p.add_argument(
        "--input", help="Optional sensor bundle override for offline testing"
    )
    args = p.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    source = Path(args.input) if args.input else output / "sensors/latest.npz"
    last = None
    while True:
        try:
            data = load_sensor(source)
            if str(data["frame_id"]) != last:
                result, image = process(data)
                image.save(output / "perception.tmp.jpg")
                (output / "perception.tmp.jpg").replace(output / "perception.jpg")
                result["processed_at_utc"] = datetime.datetime.now(
                    datetime.UTC
                ).isoformat()
                atomic(output / "perception.json", result)
                last = str(data["frame_id"])
                if args.once:
                    print(json.dumps(result, indent=2))
                    return
        except (OSError, ValueError) as exc:
            if args.once:
                raise
            atomic(
                output / "perception-error.json",
                {
                    "error": str(exc),
                    "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
                },
            )
        time.sleep(0.05)


if __name__ == "__main__":
    main()
