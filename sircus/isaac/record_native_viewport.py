"""Record only the native Isaac X11 window; no image compositing or overlays."""

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path


def record(run):
    if not (run / "ready.json").exists():
        raise RuntimeError("Simulator has not reported ready")
    tree = subprocess.check_output(["xwininfo", "-root", "-tree"], text=True)
    windows = re.findall(r'(0x[0-9a-fA-F]+) "Isaac Sim 5\.1\.0":.*?(\d+)x(\d+)\+', tree)
    if len(windows) != 1:
        raise RuntimeError(f"Expected exactly one Isaac window, found {len(windows)}")
    handle, width, height = windows[0]
    output = run / "isaac-native-farm.mp4"
    if output.exists() or (run / "record").exists():
        raise RuntimeError("Refusing to overwrite an existing native recording")
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "x11grab",
        "-window_id",
        str(int(handle, 16)),
        "-draw_mouse",
        "0",
        "-framerate",
        "30",
        "-i",
        os.environ["DISPLAY"],
        "-an",
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
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
        str(output),
    ]
    started = time.monotonic()
    with (run / "recorder.log").open("w") as log:
        proc = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log)
        try:
            time.sleep(1)
            if proc.poll() is not None:
                raise RuntimeError("Window recorder failed to start")
            (run / "record").touch()
            while not (run / "result.json").exists():
                if (run / "failure.json").exists():
                    raise RuntimeError(
                        "Simulator failed; preserve partial footage as diagnostic only"
                    )
                if proc.poll() is not None or time.monotonic() - started > 180:
                    raise RuntimeError(
                        "Recording failed or exceeded its three-minute bound"
                    )
                time.sleep(0.5)
            time.sleep(2)
        finally:
            if proc.poll() is None:
                proc.stdin.write(b"q\n")
                proc.stdin.flush()
            if proc.wait(timeout=30) != 0:
                raise RuntimeError("ffmpeg did not finish successfully")
    result = dict(
        source="Direct X11 capture of the native Isaac Sim application window",
        window_id=handle,
        window_title="Isaac Sim 5.1.0",
        dimensions=[int(width), int(height)],
        fps=30,
        encoding_padding="At most one black pixel at right/bottom for even H.264 dimensions",
        compositor=False,
        overlays=False,
        interpolation=False,
        sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        bytes=output.stat().st_size,
        wall_s=time.monotonic() - started,
    )
    (run / "recording.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    record(parser.parse_args().run)
