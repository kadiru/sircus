"""Publish verified saved-flight comparisons; no planning, physics or promotion.

Video bytes live only in an owned HTML JSON script, never in the polled study
report. Original rows, traces, checkpoints and other display panels are retained.
"""
from __future__ import annotations

import argparse
import base64
import copy
import json
import math
import os
from pathlib import Path
import re

from .benchmark_verify import contained, digest, read, sha
from .expanded_selector_publish import compose as parent_display

VERSION = "sircus-expanded-selector-replays-display-v1"
DIAGNOSIS_VERSION = "sircus-expanded-selector-saved-diagnosis-v1"
CASES = tuple(f"expanded-selector-test-{i:02d}" for i in (4, 10, 19, 47))
METHODS = ("current", "expanded")
MAX_VIDEO_BYTES = 50 * 1024 * 1024
MAX_TOTAL_VIDEO_BYTES = 140 * 1024 * 1024
ASSET_PATTERN = re.compile(r'(<script\b(?=[^>]*\bid=["\']expandedSelectorReplayVideoAssets["\'])[^>]*>)(.*?)(</script>)', re.S)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_vector(value, length):
    return isinstance(value, list) and len(value) == length and all(
        isinstance(n, (int, float)) and not isinstance(n, bool) and math.isfinite(n) for n in value)


def compose(study, audit_path, diagnosis_path):
    study, audit_path, diagnosis_path = map(Path, (study, audit_path, diagnosis_path))
    parent = parent_display(study, audit_path)
    diagnosis = read(diagnosis_path)
    require((diagnosis.get("version"), diagnosis.get("status"), diagnosis.get("cpu_only"),
             diagnosis.get("physics_substeps")) == (DIAGNOSIS_VERSION, "completed", True, 64),
            "Expected completed CPU diagnosis of native64 evidence")
    require(type(diagnosis.get("checks_count")) is int and diagnosis["checks_count"] > 0,
            "Diagnosis checks must pass")
    require(diagnosis.get("diagnostic_source_sha256") == sha(Path(__file__).with_name("expanded_selector_diagnosis.py")),
            "Diagnosis producer changed")
    source = diagnosis["source"]
    for key, path in (("report_sha256", study/"report.json"), ("audit_sha256", audit_path),
                      ("manifest_sha256", study/"evidence-manifest.json"), ("choices_sha256", study/"choices.json"),
                      ("contract_sha256", study/"contract.json")):
        require(source.get(key) == sha(path), "Diagnosis no longer binds " + key)
    manifest = read(study/"evidence-manifest.json")
    for name, expected in diagnosis["verified_artifacts"].items():
        require(manifest.get(name) == expected == sha(contained(study, name)), "Changed diagnosed evidence " + name)
    rows = [json.loads(line) for line in (study/"rows.jsonl").read_text().splitlines() if line.strip()]
    native = {r["id"]:r for r in rows}
    require(len(native) == len(rows), "Unique native identities required")
    slots = {s["id"]:s for s in read(study/"choices.json")["slots"]}
    layouts = {r["id"]:r for r in read(study/"layouts.json")["layouts"]}
    cases = diagnosis["cases"]
    require([c["id"] for c in cases] == list(CASES), "Expected four discordant pairs")
    require({p["id"] for p in parent["methods"]} == set(METHODS), "Expected both planners")
    captures = {m["id"]:m["captures"] for m in parent["methods"]}
    require(captures == {"current":44, "expanded":44}, "These replays explain the frozen 44/50 paired comparison")
    for index, case in enumerate(cases):
        require(case["bucket_xy_m"] == layouts[case["id"]]["bucket_xy_m"], "Wrong replay target")
        require(set(case["methods"]) == set(METHODS), "Each target needs both recorded methods")
        lost = index in (0, 2)
        require(case["change"] == ("expanded_lost" if lost else "expanded_gained"), "Wrong paired change")
        require(all(isinstance(case[k], list) and all(isinstance(s, str) for s in case[k])
                    for k in ("findings", "hypotheses")), "Diagnosis text must be plain strings")
        for method in METHODS:
            shot = case["methods"][method]
            identifier = case["id"] + "-" + method
            require(shot["id"] == identifier, "Wrong recorded method")
            row, slot = native[identifier], slots[identifier]
            require(slot["request"] is not None and slot["bucket_xy_m"] == case["bucket_xy_m"] == row["bucket_xy_m"],
                    "Replay must be an executed frozen attempt at this target")
            require(all(row[k] == v for k, v in dict(slot["request"], phase="test_"+method).items()), "Executed command differs from frozen choice")
            score = row["board_first_score"]
            require(shot["outcome"]["capture"] is (lost if method == "current" else not lost), "Wrong paired outcome")
            require(shot["outcome"]["capture"] == score["route_success"] and not score["invalid_physics"],
                    "Replay outcome differs from audited version-four score")
            for key, value in shot["outcome"].items():
                if key != "capture":
                    same = key in score and score[key] == value
                    if type(value) is float and type(score.get(key)) is float:
                        same = math.isclose(value, score[key], rel_tol=0., abs_tol=1e-12)
                    require(same, "Diagnosis score changed: " + key)
            s, v = shot["settings"], row["values"]
            expected = dict(command_board_center_m=v[:3], board_center_m=row["board_center_m"],
                board_height_m=v[2], tilt_deg=v[3], drop_gap_m=v[4],
                heading_offset_deg=row["board_heading_offset_deg"], twist_deg=row["board_twist_deg"],
                release_ball_center_m=row["release_ball_center_m"], q10_rad=row["native_initial_q10"],
                q14_rad=row["native_initial_q14"])
            require(all(s.get(k) == value for k, value in expected.items()), "Replay settings differ from original native command")
            require(all(finite_vector(s[k], n) for k, n in (("command_board_center_m",3), ("board_center_m",3),
                    ("release_ball_center_m",3), ("q10_rad",6), ("q14_rad",6))), "Finite 3D settings and six arm joints required")
            if shot["outcome"]["capture"]:
                shot["summary"] = "Captured and retained for one second (" + shot["outcome"]["route_class"].replace("_", " ") + ")."
            else:
                flight = shot.get("flight_measurements", {})
                bucket = flight.get("first_bucket_contact_geometry")
                if bucket and bucket.get("ascending") is True:
                    shot["summary"] = "The ball hit the bucket while still rising; it never fully entered."
                elif flight.get("first_robot_contact"):
                    shot["summary"] = "The ball contacted a robot after the board bounce; it never fully entered the bucket."
                else:
                    shot["summary"] = "The ball did not complete one continuous second fully inside the bucket."
            shot["record_sha256"] = digest(row)
            shot["calibrated_geometry_sha256"] = row["calibrated_geometry_sha256"]
            shot["video_source_hashes"] = {**{n:sha(study/n) for n in
                ("report.json", "contract.json", "rows.jsonl", "evidence-manifest.json", "choices.json")},
                "audit":sha(audit_path), "native_archive":row["native_cache_sha256"],
                "native_options":row["effective_options_sha256"]}
            trace = shot["trace"]
            name = row["adapter_directory"] + "/" + row["native_cache_file"]
            require((trace["relative_file"], trace["sha256"], trace["column"], trace["terminal_tick"]) ==
                    (name, row["native_cache_sha256"], row["native_cache_row"], row["physics_steps"]), "Wrong saved trajectory column")
            require(diagnosis["verified_artifacts"].get(name) == manifest.get(name) == sha(contained(study, name)),
                    "Replay trajectory changed or was not verified")
    # Only compact values consumed by this read-only panel enter the polled report.
    cases = [{**{key:copy.deepcopy(c[key]) for key in ("id","bucket_xy_m","change","findings","hypotheses")},
        "methods":{method:{key:copy.deepcopy(shot[key]) for key in
            ("id","summary","settings","selection","outcome","trace","record_sha256","calibrated_geometry_sha256","video_source_hashes")}
            for method,shot in c["methods"].items()}} for c in cases]
    return dict(version=VERSION, status="completed", audit_status="passed", physics_substeps=64,
        parent_summary=dict(positions=50, current_captures=44, expanded_captures=44),
        cases=copy.deepcopy(cases), provenance=dict(report_sha256=source["report_sha256"],
        audit_sha256=source["audit_sha256"], diagnosis_sha256=sha(diagnosis_path)),
        scope="Saved native64 evidence only; no new drops, training, planner change or robot motion.")


def media_assets(section, media_dir):
    """Read four fixed local names; refuse half-published, stale or swapped clips."""
    if media_dir is None:
        return {}
    media_dir, assets, total = Path(media_dir), {}, 0
    for case in section["cases"]:
        stem = "expanded-selector-case-" + case["id"].rsplit("-", 1)[1]
        movie, receipt_path = media_dir/(stem+".mp4"), media_dir/(stem+".json")
        require(movie.exists() and receipt_path.exists(), "Incomplete replay movie/receipt " + stem)
        require(movie.resolve().parent == media_dir.resolve() and receipt_path.resolve().parent == media_dir.resolve(),
                "Replay asset must remain inside its media directory")
        receipt = read(receipt_path)
        require(receipt.get("status") == "completed" and receipt.get("case_id") == case["id"]
                and receipt.get("native_steps_added") == 0 and receipt.get("held_arm_poses") is True,
                "Movie must replay the saved pair without stepping physics")
        require(receipt.get("version") == "sircus-expanded-selector-recorded-pairs-v1"
                and receipt.get("renderer_source_sha256") == sha(Path(__file__).with_name("expanded_selector_video.py")),
                "Bound paired-replay renderer required")
        require((receipt.get("playback_speed"), receipt.get("fps"), receipt.get("width"), receipt.get("height")) == (.25,30,1280,720)
                and receipt.get("interpolation") is False and receipt.get("sampled_ball_centers_only") is True
                and receipt.get("ball_spin_recorded") is False and receipt.get("historical_labels_unchanged") is True,
                "Expected labeled quarter-speed recorded samples without invented motion")
        require(set(receipt.get("methods", {})) == set(METHODS), "Paired movie must identify both methods")
        size = movie.stat().st_size
        require(16 <= size <= MAX_VIDEO_BYTES and receipt.get("video_bytes") == size, "Video size mismatch or inline limit exceeded")
        total += size
        require(total <= MAX_TOTAL_VIDEO_BYTES, "Paired movies exceed total inline limit")
        movie_hash, receipt_hash = sha(movie), sha(receipt_path)
        require(receipt.get("output_sha256") == movie_hash, "Movie bytes changed")
        methods = {}
        for method in METHODS:
            expected, saved = case["methods"][method], receipt["methods"][method]
            require(saved.get("record_id") == expected["id"] and saved.get("native_archive_sha256") == expected["trace"]["sha256"]
                    and saved.get("native_column") == expected["trace"]["column"], "Movie replayed a different source column")
            require(saved.get("record_sha256") == expected["record_sha256"]
                    and saved.get("input_sha256") == expected["video_source_hashes"]
                    and saved.get("calibrated_geometry_sha256") == expected["calibrated_geometry_sha256"],
                    "Movie must bind the same original record, scene and evidence")
            require(saved.get("capture") is expected["outcome"]["capture"]
                    and saved.get("scoring_version") == "sircus-board-first-native-capture-v4"
                    and saved.get("q10") == expected["settings"]["q10_rad"]
                    and saved.get("q14") == expected["settings"]["q14_rad"]
                    and saved.get("bucket_xy_m") == case["bucket_xy_m"]
                    and saved.get("values") == expected["settings"]["command_board_center_m"] +
                        [expected["settings"]["tilt_deg"],expected["settings"]["drop_gap_m"]]
                    and saved.get("source_interval_s") == [0,expected["trace"]["duration_s"]],
                    "Movie settings, outcome or sampled interval differ from diagnosis")
            methods[method] = dict(record_id=expected["id"], trace_sha256=expected["trace"]["sha256"], column=expected["trace"]["column"])
        frames = receipt.get("frames")
        require(isinstance(frames,list) and len(frames)>0 and receipt.get("duration_s") == len(frames)/30,
                "Complete frame-to-native-sample ledger required")
        end = max(case["methods"][m]["trace"]["duration_s"] for m in METHODS)
        expected_clock = [(0., "title")]*15 + [(min(end,i/30*.25), "slow")
            for i in range(math.ceil(end/.25*30))] + [(end, "outcome")]*30
        require(len(frames) == len(expected_clock), "Movie must include the complete paired scored intervals")
        for index, frame in enumerate(frames):
            require((frame.get("common_time_s"),frame.get("segment")) == expected_clock[index],
                    "Movie frame schedule differs from the complete quarter-speed replay")
            require(frame.get("frame") == index and frame.get("video_time_s") == index/30
                    and set(frame.get("methods",{})) == set(METHODS), "Frame sequence or paired methods differ")
            require(frame.get("segment") in ("title","slow","outcome") and frame.get("playback_speed") ==
                    (.25 if frame["segment"] == "slow" else 0), "Frame playback speed differs")
            common = frame.get("common_time_s")
            require(isinstance(common,(int,float)) and math.isfinite(common) and common>=0, "Finite shared native clock required")
            for method in METHODS:
                sample, trace = frame["methods"][method], case["methods"][method]["trace"]
                tick = sample.get("source_tick");when = sample.get("source_time_s")
                require(type(tick) is int and 0<=tick<=trace["terminal_tick"]
                        and isinstance(when,(int,float)) and math.isfinite(when)
                        and abs(when - tick*trace["time_step_s"]) < 1e-10
                        and abs(when - min(common,trace["duration_s"])) <= trace["time_step_s"]*.501 + 1e-10
                        and sample.get("terminal_hold") is (common>=trace["duration_s"]),
                        "Frame is not a matched saved sample or labeled terminal hold")
        payload = movie.read_bytes()
        require(payload[4:8] == b"ftyp", "Expected an MP4 container")
        assets[stem] = dict(url="data:video/mp4;base64,"+base64.b64encode(payload).decode("ascii"),
            case_id=case["id"], video_sha256=movie_hash, receipt_sha256=receipt_hash,
            native_steps_added=0, held_arm_poses=True, methods=methods)
        case["video"] = dict(asset_id=stem, video_sha256=movie_hash, receipt_sha256=receipt_hash)
    return assets


def embed_html(html, assets):
    require(len(ASSET_PATTERN.findall(html)) == 1, "Exactly one owned replay-video placeholder required")
    payload = json.dumps(assets, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    return ASSET_PATTERN.sub(lambda m:m[1]+payload+m[3], html, count=1)


def publish(study, audit, diagnosis, display, runtime_index, media_dir=None):
    study, audit, diagnosis, display, runtime_index = map(Path, (study, audit, diagnosis, display, runtime_index))
    outputs = [display, runtime_index]
    temporary = [p.with_name(p.name+".replays.tmp") for p in outputs]
    protected_dirs = [study.resolve(), diagnosis.resolve().parent]
    if media_dir is not None:
        protected_dirs.append(Path(media_dir).resolve())
    require(len({p.resolve() for p in outputs+temporary}) == 4, "Distinct derived output paths required")
    require(not any(p.resolve() == audit.resolve() or any(p.resolve().is_relative_to(d) for d in protected_dirs)
                    for p in outputs+temporary), "Derived outputs cannot replace immutable evidence")
    section = compose(study, audit, diagnosis)
    assets = media_assets(section, media_dir)
    html = embed_html(runtime_index.read_text(), assets)
    data = read(display) if display.exists() else {}
    data["expanded_selector_replays"] = section
    serialized = json.dumps(data, indent=2, allow_nan=False)+"\n"
    # HTML first: a reader never receives new video references before the bytes.
    for target, content in ((runtime_index, html), (display, serialized)):
        temp = target.with_name(target.name+".replays.tmp")
        temp.write_text(content)
        os.replace(temp, target)
    return dict(status=section["status"], cases=len(section["cases"]), videos=len(assets))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("study", "audit", "diagnosis", "display-report", "runtime-index"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--media-dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(publish(args.study, args.audit, args.diagnosis, args.display_report, args.runtime_index, args.media_dir)))


if __name__ == "__main__":
    main()
