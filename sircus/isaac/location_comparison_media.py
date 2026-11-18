"""Encode only the ten recordings of an externally pinned audited publication.

This is postprocessing, not a simulator or audit replacement. The input report
must be the completed output of location_comparison_publication, pinned by its
file SHA and by the independently verified audit SHA. Every packet, ordered
frame list and JPEG is checked again. Exact JPEG bytes are copied into a private
MJPEG stream before CPU-only ffmpeg encoding; no rendering, overlays, frame
selection, resampling, model loading or source-evidence writes are performed.

The destination must not exist and must be outside native evidence. All ten
videos must pass ffprobe and a full decode before a new report.json attaches
media once. Failure preserves the new directory and never publishes a partial
report or resumes an earlier export. Successful private stream snapshots are
removed only after their hash and provenance have been retained. Original
packets/JPEGs and the input report are never removed or modified.
"""

import argparse
import copy
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

from candidate_continuation_journal import write_new_json, _sync_directory
from location_comparison_publication import SCHEMA as REPORT_SCHEMA, digest, require, recovery_publication_verified

SCHEMA = "sircus-location-comparison-media-provenance-v1"
FPS = 30
MODELS = ("parent", "original_usable", "broader")
MAX_JSON_BYTES = 16 * 1024**2
MAX_JPEG_BYTES = 64 * 1024**2
SCHEDULE = [(model, "train", n, 35) for model in MODELS[1:] for n in (0, 2240, 4480, 6720)] + [
    ("parent", "test_broad", None, 24), ("parent", "test_local", None, 9)]


def _sha(value):
    require(type(value) is str and re.fullmatch(r"[a-f0-9]{64}", value), "Invalid SHA256 pin")
    return value


def _path(value, *, existing=True):
    path = Path(value).absolute()
    require(not any(p.is_symlink() for p in (path, *path.parents)), "Linked media path rejected")
    if existing:
        require(path.exists(), "Missing media path")
    return path


def _file(root, relative):
    require(type(relative) is str and relative and not Path(relative).is_absolute()
        and ".." not in Path(relative).parts, "Escaped media source path")
    root = _path(root)
    path = _path(root / relative)
    require(path.is_file() and path.resolve().is_relative_to(root.resolve()), "Invalid media source file")
    return path


def _hash(path):
    checksum = hashlib.sha256()
    with _path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            checksum.update(block)
    return checksum.hexdigest()


def _json(raw):
    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, "Duplicate media JSON key")
            value[key] = item
        return value
    return json.loads(raw, object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite media JSON")))


def _read(path, expected):
    import gzip
    path = _path(path)
    require(path.is_file() and _hash(path) == _sha(expected), "Changed pinned media input")
    opener = gzip.open if path.name.endswith(".gz") else open
    with opener(path, "rb") as stream:
        raw = stream.read(MAX_JSON_BYTES + 1)
    require(len(raw) <= MAX_JSON_BYTES, "Media JSON exceeds size bound")
    return _json(raw)


def _metrics(row, count):
    return (type(row) is dict and all(type(row.get(k)) is int and row[k] >= 0
        for k in ("requests", "captures", "clean", "rejected", "misses")) and row["requests"] == count
        and row["clean"] <= row["captures"] and row["captures"] + row["rejected"] + row["misses"] == count)


def _report_gate(report, expected_audit_sha256):
    recovered = recovery_publication_verified(report)
    require(report.get("schema") == REPORT_SCHEMA and (report.get("status") == "completed_audited" or recovered)
        and report.get("synthetic") is False and report.get("model_promoted") is False,
        "Completed audited publication required")
    audit = report.get("audit", {})
    require((audit.get("status") == "passed" or recovered) and audit.get("complete") is True and audit.get("synthetic") is False
        and audit.get("audit_sha256") == _sha(expected_audit_sha256), "Independent audit pin differs")
    for key in ("manifest_fingerprint", "comparison_fingerprint", "native_exit_fingerprint",
                "audit_exit_fingerprint", "trace_sha256"):
        _sha(audit.get(key))
    require(type(audit.get("session_id")) is str and re.fullmatch(r"[a-f0-9]{32}", audit["session_id"]), "Invalid audit session")
    require(all(_metrics(report.get("train", {}).get(m), 8960) for m in MODELS[1:])
        and all(_metrics(report.get("final_tests", {}).get(m, {}).get(p), n)
            for m in MODELS for p, n in (("test_broad", 70), ("test_local", 25))), "Incomplete audited request totals")
    videos = report.get("videos")
    require(type(videos) is list and len(videos) == 10, "Exactly ten prescheduled recordings required")
    require([(v.get("model"), v.get("phase"), v.get("train_requests_before"), v.get("active_slots"))
        for v in videos] == SCHEDULE, "Prescheduled cohort order or membership changed")
    require(len({v.get("id") for v in videos}) == 10, "Repeated recording identity")
    for video in videos:
        require(type(video.get("id")) is str and re.fullmatch(r"batch-[0-9]{5}", video["id"]), "Invalid recording id")
        expected = "comparison-native/trial-" + video["id"].split("-")[1] + ".json.gz"
        require(video.get("packet_file") == expected and video.get("fps") == FPS
            and type(video.get("frames")) is int and 0 < video["frames"] <= 1000000
            and video.get("parked_slots") == 35-video["active_slots"], "Invalid scheduled frame metadata")
        require(video.get("media") is None and video.get("media_status") == "verified_frames_not_encoded",
                "Media is already attached or source recording is unverified")
        label = video.get("label")
        require(type(label) is str and 0 < len(label) <= 512 and not any(ord(c) < 32 or ord(c) == 127 for c in label),
                "Invalid recording chapter label")
        require(video.get("chapters") == [dict(start_s=0, label=label, phase=video["phase"],
            train_requests_before=video["train_requests_before"])], "Publication chapter labels changed")
        _sha(video.get("packet_sha256")); _sha(video.get("frame_sequence_sha256"))
    return videos


def _jpeg(path, expected):
    path = _path(path)
    require(path.is_file() and path.stat().st_size <= MAX_JPEG_BYTES, "Missing or oversized JPEG")
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == _sha(expected), "Changed recorded JPEG bytes")
    require(len(raw) >= 4 and raw[:2] == b"\xff\xd8" and raw[-2:] == b"\xff\xd9", "Invalid JPEG framing")
    return raw


def _sources(native_output, report, videos):
    prepared, all_ids = [], []
    for video in videos:
        packet_path = _file(native_output, video["packet_file"])
        packet = _read(packet_path, video["packet_sha256"])
        require(packet.get("schema") == "sircus-location-comparison-native-trial-v1"
            and packet.get("synthetic") is False and packet.get("filming") is True
            and packet.get("comparison_fingerprint") == report["audit"]["comparison_fingerprint"],
            "Recording packet is not this audited native comparison")
        frames = packet.get("frames")
        require(type(frames) is list and len(frames) == video["frames"]
            and digest(frames) == video["frame_sequence_sha256"], "Ordered frame identity changed")
        resolution, sources = None, []
        ids = []
        for frame in frames:
            index = frame.get("frame")
            require(type(index) is int and 0 <= index <= 10000000, "Invalid source frame index")
            size = frame.get("resolution")
            require(type(size) is list and len(size) == 2 and all(type(n) is int and 0 < n <= 16384 and n % 2 == 0 for n in size),
                    "Invalid recorded JPEG dimensions")
            resolution = size if resolution is None else resolution
            require(size == resolution, "Recording dimensions changed mid-cohort")
            relative = f"native/frames/{index:05d}.jpg"
            path = _file(native_output, relative)
            _jpeg(path, frame["jpeg_sha256"])
            sources.append(dict(file=relative, sha256=frame["jpeg_sha256"], frame=index))
            ids.append(index)
        require(ids == list(range(ids[0], ids[0]+len(ids))), "Recorded frame order is not contiguous")
        all_ids.extend(ids)
        prepared.append(dict(video=video, packet_path=packet_path, sources=sources, resolution=resolution))
    require(all_ids == list(range(len(all_ids))), "Full prescheduled frame inventory changed")
    return prepared


def _tool(value):
    require(isinstance(value, (str, Path)) and str(value), "Missing encoder executable")
    found = shutil.which(str(value))
    require(found is not None, "Installed encoder executable not found: " + str(value))
    # Installed Homebrew/system executable links may resolve normally; source
    # evidence and output paths, in contrast, may never contain symlinks.
    path = Path(found).resolve()
    require(path.is_file() and os.access(path, os.X_OK), "Encoder executable is unavailable")
    return str(path)


def _run(command, timeout):
    return subprocess.run(command, check=True, shell=False, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="", NVIDIA_VISIBLE_DEVICES="void"))


def _version(tool, name):
    raw = _run([tool, "-version"], 30).stdout
    require(len(raw) <= 1024**2, "Unexpected encoder version response")
    first = raw.decode("utf-8", errors="strict").splitlines()[0]
    require(first.startswith(name + " version ") and len(first) <= 1024, "Wrong encoder executable")
    return first


def _metadata(label, frames):
    escaped = label
    for character in ("\\", "=", ";", "#"):
        escaped = escaped.replace(character, "\\" + character)
    return f";FFMETADATA1\n[CHAPTER]\nTIMEBASE=1/{FPS}\nSTART=0\nEND={frames}\ntitle={escaped}\n"


def _verify_probe(value, video, resolution):
    streams, chapters = value.get("streams"), value.get("chapters")
    require(type(streams) is list and 1 <= len(streams) <= 2, "Unexpected encoded stream count")
    videos = [s for s in streams if s.get("codec_type") == "video"]
    other = [s for s in streams if s.get("codec_type") != "video"]
    # ffmpeg's MP4 chapter writer may add one QuickTime text chapter data
    # track. This is metadata, not another image/audio/subtitle stream.
    require(len(videos) == 1 and all(s.get("codec_type") == "data" and s.get("codec_name") == "bin_data"
        and s.get("codec_tag_string") == "text" for s in other), "Unexpected encoded video/audio streams")
    stream, duration = videos[0], video["frames"] / FPS
    require(stream.get("codec_name") == "h264" and stream.get("pix_fmt") == "yuv420p"
        and Fraction(stream["avg_frame_rate"]) == FPS and int(stream["nb_read_frames"]) == video["frames"]
        and [stream.get("width"), stream.get("height")] == resolution, "Encoded frame count, rate or dimensions differ")
    for measured in (stream.get("duration"), value.get("format", {}).get("duration")):
        require(math.isfinite(float(measured)) and abs(float(measured)-duration) < .002, "Encoded duration differs")
    require(type(chapters) is list and len(chapters) == 1
        and abs(float(chapters[0]["start_time"])) < .002
        and abs(float(chapters[0]["end_time"])-duration) < .002
        and chapters[0].get("tags", {}).get("title") == video["label"], "Encoded chapter label or timing differs")
    return dict(frames=video["frames"], duration_s=duration, fps=FPS, resolution=resolution,
                codec="h264", pixel_format="yuv420p", audio_streams=0,
                chapter_data_streams=len(other), chapter_label=video["label"])


def _encode_one(native_output, destination, prepared, bindings, tools, timeout):
    video, sources = prepared["video"], prepared["sources"]
    stream_path = destination / "staging" / (video["id"] + ".mjpeg")
    metadata_path = destination / "staging" / (video["id"] + ".ffmeta")
    movie = destination / "media" / (video["id"] + ".mp4")
    require(not any(p.exists() or p.is_symlink() for p in (stream_path, metadata_path, movie)), "Media output would overwrite evidence")
    # A single fixed snapshot prevents ffmpeg from reading changed source paths
    # between checks, or discovering frames beyond this audited cohort.
    with stream_path.open("xb") as target:
        for item in sources:
            target.write(_jpeg(_file(native_output, item["file"]), item["sha256"]))
        target.flush(); os.fsync(target.fileno())
    stream_sha = _hash(stream_path)
    with metadata_path.open("x", encoding="utf-8") as target:
        target.write(_metadata(video["label"], video["frames"]))
        target.flush(); os.fsync(target.fileno())
    command = [tools["ffmpeg"], "-nostdin", "-hide_banner", "-loglevel", "error", "-xerror", "-n",
        "-hwaccel", "none", "-threads", "2", "-f", "image2pipe", "-vcodec", "mjpeg", "-framerate", str(FPS), "-i", str(stream_path),
        "-f", "ffmetadata", "-i", str(metadata_path), "-map", "0:v:0", "-map_metadata", "1", "-map_chapters", "1",
        "-an", "-sn", "-dn", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-threads", "2", "-preset", "medium",
        "-crf", "18", "-frames:v", str(video["frames"]), "-vsync", "0", "-movflags", "+faststart", str(movie)]
    began = time.monotonic()
    _run(command, timeout)
    require(_path(movie).is_file() and movie.stat().st_size > 0, "Encoder did not produce a video")
    movie_sha = _hash(movie)
    probe = _run([tools["ffprobe"], "-v", "error", "-count_frames", "-show_streams", "-show_chapters",
                  "-show_format", "-of", "json", str(movie)], timeout).stdout
    require(len(probe) <= MAX_JSON_BYTES, "Encoded video probe is oversized")
    checked = _verify_probe(_json(probe), video, prepared["resolution"])
    _run([tools["ffmpeg"], "-nostdin", "-v", "error", "-xerror", "-hwaccel", "none", "-threads", "2",
          "-i", str(movie), "-map", "0:v:0", "-f", "null", "-"], timeout)
    require(_hash(movie) == movie_sha and _hash(stream_path) == stream_sha, "Encoded video or frame snapshot changed during verification")
    require(_hash(prepared["packet_path"]) == video["packet_sha256"], "Recorded packet changed during encoding")
    for item in sources:
        _jpeg(_file(native_output, item["file"]), item["sha256"])
    provenance = dict(schema=SCHEMA, status="encoded_verified", synthetic=False, **bindings,
        id=video["id"], model=video["model"], phase=video["phase"], label=video["label"],
        train_requests_before=video["train_requests_before"], active_slots=video["active_slots"], parked_slots=video["parked_slots"],
        packet_file=video["packet_file"], packet_sha256=video["packet_sha256"],
        frame_sequence_sha256=video["frame_sequence_sha256"], source_frames=sources,
        input_mjpeg_sha256=stream_sha, source_bytes_rechecked=True, encoded_file="media/"+movie.name,
        encoded_sha256=movie_sha, verification=dict(checked, full_decode_passed=True),
        encoding=dict(codec="libx264", crf=18, preset="medium", threads=2, gpu=False,
            overlays=False, rerendering=False, resampling=False, vsync="0_passthrough", original_frame_order=True,
            ffmpeg_version=tools["ffmpeg_version"], ffprobe_version=tools["ffprobe_version"]),
        wall_s=time.monotonic()-began, cost_scope="Publication encoding/probing/decoding only; not native simulation or training compute.")
    receipt = destination / "provenance" / (video["id"] + ".json")
    write_new_json(receipt, provenance)
    # Only exact files created in this fresh private export are removed. On a
    # failure before this point the snapshot is preserved for diagnosis.
    stream_path.unlink(); metadata_path.unlink(); _sync_directory(stream_path.parent)
    return dict(url="/media/"+movie.name, sha256=movie_sha, verified=True, duration_s=checked["duration_s"],
        frames=checked["frames"], fps=FPS, resolution=checked["resolution"],
        provenance_file="provenance/"+receipt.name, provenance_sha256=_hash(receipt))


def encode_publication(report_path, native_output, destination, *, expected_report_sha256,
                       expected_audit_sha256, ffmpeg="ffmpeg", ffprobe="ffprobe",
                       timeout_s=1800, execute_approved=False):
    """Create one new private export, then attach all ten media descriptors once.

    Explicit binary paths are supported; no tools are installed. All subprocess
    calls use argument lists without a shell, software codecs and hidden GPUs.
    A failed/previous destination is never resumed or overwritten.
    """
    require(execute_approved is True, "Explicit post-audit encoding opt-in required")
    require(type(timeout_s) in (int, float) and math.isfinite(timeout_s) and 1 <= timeout_s <= 3600,
            "Invalid bounded encoder timeout")
    report_path, native_output = _path(report_path), _path(native_output)
    destination = _path(destination, existing=False)
    require(native_output.is_dir() and destination.parent.is_dir() and not destination.exists()
        and not destination.resolve().is_relative_to(native_output.resolve()), "New export outside native evidence required")
    report = _read(report_path, _sha(expected_report_sha256))
    videos = _report_gate(report, expected_audit_sha256)
    prepared = _sources(native_output, report, videos)
    tools = dict(ffmpeg=_tool(ffmpeg), ffprobe=_tool(ffprobe))
    tools.update(ffmpeg_version=_version(tools["ffmpeg"], "ffmpeg"), ffprobe_version=_version(tools["ffprobe"], "ffprobe"))
    bindings = dict(publication_sha256=expected_report_sha256, audit_sha256=expected_audit_sha256,
        manifest_fingerprint=report["audit"]["manifest_fingerprint"],
        comparison_fingerprint=report["audit"]["comparison_fingerprint"], session_id=report["audit"]["session_id"])
    destination.mkdir(); _sync_directory(destination.parent)
    for name in ("staging", "media", "provenance"):
        (destination / name).mkdir()
    write_new_json(destination / "encoding-start.json", dict(schema=SCHEMA, status="started", **bindings,
        scheduled_recordings=10, started_at=datetime.now(timezone.utc).isoformat(), source_evidence_writes=False))
    began = time.monotonic()
    try:
        result = copy.deepcopy(report)
        for index, item in enumerate(prepared):
            result["videos"][index]["media"] = _encode_one(native_output, destination, item, bindings, tools, timeout_s)
            result["videos"][index]["media_status"] = "encoded_verified"
        require(_hash(report_path) == expected_report_sha256, "Input publication changed during encoding")
        _sources(native_output, report, videos)
        for video in result["videos"]:
            media = video["media"]
            require(_hash(_file(destination, media["url"].lstrip("/"))) == media["sha256"]
                and _hash(_file(destination, media["provenance_file"])) == media["provenance_sha256"],
                "Completed media or provenance changed before attachment")
        result["media_publication"] = dict(schema=SCHEMA, status="all_prescheduled_recordings_encoded",
            **bindings, recordings=10, generated_at=datetime.now(timezone.utc).isoformat(), wall_s=time.monotonic()-began,
            scope="Derived native video encoding only. Numeric results and original evidence remain unchanged.")
        write_new_json(destination / "report.json", result)
        return result
    except Exception as error:
        write_new_json(destination / "encoding-failure.json", dict(schema=SCHEMA, status="failed_export_preserved_no_resume",
            **bindings, error_type=type(error).__name__, report_attached=False, wall_s=time.monotonic()-began))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--native-output", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--expected-report-sha256", required=True)
    parser.add_argument("--expected-audit-sha256", required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--timeout-s", type=float, default=1800)
    parser.add_argument("--encode-audited-recordings", action="store_true")
    args = vars(parser.parse_args(argv))
    args["report_path"] = args.pop("report")
    args["execute_approved"] = args.pop("encode_audited_recordings")
    result = encode_publication(**args)
    print(json.dumps(dict(status=result["media_publication"]["status"], recordings=len(result["videos"])), sort_keys=True))


if __name__ == "__main__":
    main()
