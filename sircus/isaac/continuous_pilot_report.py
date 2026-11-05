"""Summarize the continuous-shot pilot without running physics or a policy.

Every rejected active shot remains in the denominator. Endpoint-map samples,
on-policy training outcomes and frozen evaluation are kept in separate groups.
The native trajectory/likelihood audit is a separate deliverable: this report
checks scalar outcome consistency, source hashes and optional video decoding.
"""

import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess


SCHEMA = "sircus-continuous-pilot-report-v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    def reject(value):
        raise ValueError("Non-finite JSON: " + value)
    return json.loads(Path(path).read_text(), parse_constant=reject)


def local_file(root, name):
    require(isinstance(name, str) and name and not Path(name).is_absolute()
            and ".." not in Path(name).parts, "Unsafe evidence filename")
    path = Path(root) / name
    require(path.is_file() and not path.is_symlink()
            and path.resolve().is_relative_to(Path(root).resolve()),
            "Missing or unsafe evidence file: " + name)
    return path


def interval(captures, attempts):
    if not attempts:
        return None
    p, z = captures / attempts, 1.959963984540054
    den = 1 + z*z/attempts
    center = (p + z*z/(2*attempts))/den
    radius = z*math.sqrt(p*(1-p)/attempts + z*z/(4*attempts*attempts))/den
    return [max(0., center-radius), min(1., center+radius)]


def failure(row):
    if row.get("invalid"):
        return str(row["invalid"])
    if row.get("route_success"):
        return "captured"
    if row.get("external_contact_steps", 0):
        return "external_contact"
    if row.get("board_bounce") is False:
        return "no_board_bounce"
    if row.get("whole_ball_entry") and row.get("max_retention_s", 0) < 1:
        return "retention_under_one_second"
    return "released_miss"


def counts(rows):
    values = list(rows)
    for row in values:
        require(type(row.get("route_success")) is bool,
                "Every active shot needs an explicit native capture result")
        require("invalid" in row, "Every active shot needs an explicit admission result")
        if row["route_success"]:
            require(not row["invalid"] and row.get("board_bounce")
                    and row.get("whole_ball_entry")
                    and row.get("max_retention_s", 0) >= 1 - 1e-8
                    and not row.get("external_contact_steps", 0),
                    "Capture does not meet the board-route scoring contract")
    n = len(values)
    captures = sum(row["route_success"] for row in values)
    return dict(attempts=n, captures=captures,
                clean=sum(row.get("route_quality") == 2 for row in values),
                single_bounce=sum(bool(row.get("single_bounce_success")) for row in values),
                invalid=sum(bool(row["invalid"]) for row in values),
                capture_rate=captures/n if n else None,
                wilson95=interval(captures, n),
                failures=dict(collections.Counter(failure(row) for row in values)))


def active_rows(trial, cells):
    rows = trial.get("rows", trial.get("rollout"))
    require(isinstance(rows, list), "Trial has no native outcome rows")
    known = {cell["id"]: cell for cell in cells}
    require(len(known) == len(cells), "Duplicate cell IDs")
    ids = trial.get("active_cell_ids")
    enabled = trial.get("enabled", trial.get("release_enabled"))
    if ids is None and enabled is not None:
        require(len(enabled) == len(cells) and all(type(v) is bool for v in enabled),
                "Invalid active-cell mask")
        ids = [cell["id"] for cell, use in zip(cells, enabled) if use]
    require(isinstance(ids, list) and len(set(ids)) == len(ids)
            and all(name in known for name in ids), "Explicit active-cell IDs required")
    result, seen = [], set()
    for row in rows:
        name = row.get("cell_id")
        require(name in known and name not in seen, "Missing, unknown or duplicate outcome cell")
        seen.add(name)
        if name in ids:
            value = dict(row)
            split = value.setdefault("split", known[name]["split"])
            require(split == known[name]["split"], "Native row changed the frozen spatial split")
            result.append(value)
    require({row["cell_id"] for row in result} == set(ids), "An active attempted shot disappeared")
    return result


def numeric_ranges(actions):
    grouped = collections.defaultdict(list)
    for action in actions:
        for key, value in action.items():
            if type(value) in (int, float) and math.isfinite(value):
                grouped[key].append(value)
    return {key: dict(min=min(values), max=max(values), samples=len(values))
            for key, values in grouped.items()}


def summarize(run):
    run = Path(run)
    study = read(local_file(run, "study.json"))
    cells = study["cells"]
    ids = {cell["id"] for cell in cells}
    require(len(ids) == len(cells) and all(cell.get("split") in ("train", "validation", "test")
            for cell in cells), "Missing or invalid frozen spatial split")
    xy = [tuple(cell["bucket_xy_m"]) for cell in cells]
    require(len(set(xy)) == len(xy), "Bucket positions are not distinct")
    source_hashes = {"study.json": sha(run/"study.json")}
    feasibility = read(local_file(run, "feasibility.json"))
    source_hashes["feasibility.json"] = sha(run/"feasibility.json")
    map_rows = [dict(row, valid=row.get("valid", row.get("accepted")))
                for row in feasibility.get("rows", feasibility.get("records", []))]
    require(map_rows, "Missing sampled feasibility records")
    require(all(row.get("cell_id") in ids and type(row.get("valid")) is bool for row in map_rows),
            "Invalid endpoint-map record")
    batches, train_rows, coverage_rows, evaluation, trial_rows = [], [], [], collections.defaultdict(list), []
    evaluation_deterministic = collections.defaultdict(list)
    all_actions = []
    paths = sorted((run/"trials").glob("*.json"))
    for path in paths:
        require(not path.is_symlink(), "Trial symlink is not evidence")
        trial = read(path)
        rows = active_rows(trial, cells)
        mode = trial["mode"]
        require(mode in ("train", "eval", "assessment", "coverage"), "Unknown trial mode")
        update = trial.get("update_count", trial.get("update_count_before"))
        require(type(update) is int and update >= 0, "Missing sampling-policy update count")
        if mode == "train":
            require(all(row["split"] == "train" for row in rows), "Held-out layout entered training")
            batch = trial["batch"]
            require(type(batch) is int and batch >= 0, "Invalid training batch")
            summary = dict(batch=batch, update_count=update, **counts(rows))
            rewards = trial.get("rewards", [])
            if rewards:
                if len(rewards) == len(cells):
                    rewards = [value for value, cell in zip(rewards, cells)
                               if cell["id"] in {row["cell_id"] for row in rows}]
                require(len(rewards) == len(rows), "Rewards do not match active attempted shots")
                require(all(type(x) in (int, float) and math.isfinite(x) for x in rewards), "Invalid rewards")
                summary["mean_reward"] = sum(rewards)/len(rewards)
            batches.append(summary)
            train_rows.extend(rows)
            actions = trial.get("actions", [])
            if len(actions) == len(cells):
                actions = [a for a, cell in zip(actions, cells) if cell["id"] in {r["cell_id"] for r in rows}]
            all_actions.extend(actions)
        elif mode == "coverage":
            require(trial.get("optimizer_samples", 0) == 0, "Coverage shots entered the optimizer")
            coverage_rows.extend(rows)
        else:
            require(trial.get("optimizer_samples", 0) == 0, "Evaluation contains optimizer samples")
            require(isinstance(trial.get("policy_hash"), str) and trial["policy_hash"], "Frozen policy hash required")
            for row in rows:
                key = (update, row["split"], trial["policy_hash"])
                evaluation[key].append(row)
                evaluation_deterministic[key].append(trial.get("deterministic"))
        trial_rows.append(dict(file="trials/"+path.name, mode=mode, update_count=update, rows=rows))
        source_hashes["trials/"+path.name] = sha(path)
    batches.sort(key=lambda item: item["batch"])
    require(len({item["batch"] for item in batches}) == len(batches), "Duplicate training batch")
    cumulative, cumulative_captures = 0, 0
    for index, point in enumerate(batches):
        cumulative += point["attempts"]
        cumulative_captures += point["captures"]
        recent = batches[max(0, index-7):index+1]
        point.update(completed_attempts=cumulative, cumulative_captures=cumulative_captures,
                     rolling_capture_rate=sum(p["captures"] for p in recent)/sum(p["attempts"] for p in recent),
                     rolling_batches=len(recent))
    checks = [dict(update_count=update, split=split, policy_hash=policy_hash, **counts(rows))
              for (update, split, policy_hash), rows in sorted(evaluation.items())]
    for check in checks:
        key = (check["update_count"], check["split"], check["policy_hash"])
        values = evaluation[key]
        check["unique_layouts"] = len({row["cell_id"] for row in values})
        check["layouts_with_capture"] = len({row["cell_id"] for row in values if row["route_success"]})
        check["deterministic"] = all(value is True for value in evaluation_deterministic[key])
        if check["deterministic"]:
            # Replaying one deterministic prediction eight times is not eight
            # independent layouts. Do not draw misleading binomial error bars.
            check["wilson95"] = None
            check["interval_scope"] = "No binomial interval: repeated deterministic trials share layouts and are not independent generalization samples."
    require(len({(item["update_count"], item["split"]) for item in checks}) == len(checks),
            "One evaluation stage contains multiple policy hashes")
    cell_results = []
    for cell in cells:
        name = cell["id"]
        mapped = [row for row in map_rows if row["cell_id"] == name]
        heldout = [dict(update_count=update, **counts(row for row in rows if row["cell_id"] == name))
                   for (update, split, _), rows in sorted(evaluation.items())
                   if split == cell["split"]]
        cell_results.append(dict(cell, feasibility=dict(attempts=len(mapped),
            admitted=sum(row["valid"] for row in mapped),
            rejected=sum(not row["valid"] for row in mapped),
            reasons=dict(collections.Counter(str(row.get("reason") or "admitted") for row in mapped)),
            action_ranges=numeric_ranges(row["action"] for row in mapped if row["valid"])),
            training=counts(row for row in train_rows if row["cell_id"] == name),
            coverage=counts(row for row in coverage_rows if row["cell_id"] == name),
            evaluation=[item for item in heldout if item["attempts"]]))
    result_path = run/"result.json"
    result = read(result_path) if result_path.is_file() else {}
    if result:
        source_hashes["result.json"] = sha(result_path)
    audit_path = run/"audit.json"
    native_audit = read(audit_path) if audit_path.is_file() else None
    if native_audit is not None:
        source_hashes["audit.json"] = sha(audit_path)
    return dict(schema=SCHEMA, status=result.get("status", "in_progress"),
                study_fingerprint=study.get("fingerprint"),
                geometry_fingerprint=study.get("geometry_fingerprint"),
                scope="One-step privileged-geometry continuous-action PPO pilot. Endpoint feasibility is not collision-certified motion planning. Scripted arm transit and initialized ball release; grasp and joint paths are not learned.",
                scoring="Board bounce + whole-ball entry + one continuous second retained; no detected external route contact. Rim-assisted captures count; clean captures reported separately.",
                uncertainty="Bucket reported height175mm remains unresolved overall-height versus interior-depth. Contact properties and rigid board mount remain provisional.",
                study=study, cells=cell_results, feasibility=dict(attempts=len(map_rows),
                    admitted=sum(row["valid"] for row in map_rows),
                    rejected=sum(not row["valid"] for row in map_rows),
                    reasons=dict(collections.Counter(str(row.get("reason") or "admitted") for row in map_rows))),
                training=counts(train_rows), coverage=counts(coverage_rows),
                coverage_scope="Prescheduled geometry-coverage attempts; outcomes excluded from policy updates and frozen evaluation.",
                learning_curve=batches, assessments=checks,
                sampled_training_action_ranges=numeric_ranges(all_actions),
                trial_files=len(paths), elapsed_s=result.get("elapsed_s", result.get("seconds")),
                native_audit=native_audit,
                audit_scope="Report counts and capture invariants checked against active native scalar rows. Raw trajectory and checkpoint verification are separately labeled.",
                source_sha256=source_hashes, clips=[], movie=None)


def inspect_movie(path):
    probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-count_frames",
        "-show_streams", "-show_format", "-of", "json", str(path)]))
    videos = [stream for stream in probe["streams"] if stream["codec_type"] == "video"]
    require(len(videos) == 1, "Exactly one native video stream is required")
    stream = videos[0]
    require(stream["codec_name"] == "h264", "Native clip must be H.264")
    require(stream["width"] == 1920 and stream["height"] == 1080,
            "Chosen-camera native resolution changed")
    numerator, denominator = map(int, stream["avg_frame_rate"].split("/"))
    fps = numerator/denominator
    require(fps > 0, "Invalid native frame rate")
    subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-i", str(path), "-map", "0:v:0",
                    "-f", "null", "-"], check=True, stdout=subprocess.DEVNULL)
    return dict(sha256=sha(path), frames=int(stream["nb_read_frames"]),
                duration_s=float(stream.get("duration", probe["format"]["duration"])),
                fps=fps, resolution=[stream["width"], stream["height"]], decode_verified=True)


def native_frame_manifest(run, clips):
    """Bind unique chronological JPEGs to the exact saved camera and real steps.

    This is a movie-lineage check. Joint readback establishes visible movement,
    not collision-safe transit, learned joint control or physical calibration.
    """
    run = Path(run)
    camera = read(local_file(run, "selected-camera.json"))
    require(camera.get("resolution") == [1920, 1080] and camera.get("up_axis") == "Z",
            "Chosen-camera metadata changed")
    seen, previous, result = set(), -1, []
    for clip in clips:
        if "trial_file" in clip:
            trial = read(local_file(run, clip["trial_file"]))
            require(all(clip[key] == trial[key] for key in
                        ("mode", "batch", "update_count", "actions", "rows", "release_step")),
                    "Native film does not match its recorded policy/coverage trial")
        frames = clip["frames"]
        require(isinstance(frames, list) and frames, "A recorded trial needs native frame metadata")
        require(clip.get("fps", 30) == 30, "Native clip frame rate changed")
        indices, physics_steps, hashes, joints = [], [], {}, collections.defaultdict(list)
        for frame in frames:
            index = frame["frame"]
            step = frame["physics_step"]
            require(type(index) is int and index > previous and index not in seen,
                    "Native camera frames duplicated or reordered")
            require(type(step) is int and (not physics_steps or step > physics_steps[-1]),
                    "Native frame physics steps are not increasing")
            require(frame["phase"] in ("setup", "move", "release"), "Unknown native camera phase")
            # RTX readback is float32 even when the saved requested camera is
            # double precision. Preserve the source bytes and use the original
            # native2e-4m readback gate; target is the requested double vector.
            require(len(frame["eye_m"]) == 3
                    and all(type(x) in (int, float) and math.isfinite(x) for x in frame["eye_m"])
                    and math.dist(frame["eye_m"], camera["eye_m"]) <= .0002
                    and frame["target_m"] == camera["target_m"],
                    "Native camera moved away from the selected fixed angle")
            filename = f"frames/{index:05d}.jpg"
            path = local_file(run, filename)
            hashes[filename] = sha(path)
            readbacks = frame.get("all_joints_rad", {})
            require(set(readbacks) == {"Arm10", "Arm14"}, "Missing actual arm-joint readbacks")
            for arm, values in readbacks.items():
                require(len(values) == 35 and all(len(v) >= 6 for v in values)
                        and all(type(x) in (int, float) and math.isfinite(x)
                                for v in values for x in v), "Invalid actual arm-joint readback")
                joints[arm].append(values[1][:6])
            seen.add(index)
            previous = index
            indices.append(index)
            physics_steps.append(step)
        require(indices == list(range(indices[0], indices[-1]+1)),
                "Native JPEG clip has gaps; do not hide setup or flight frames")
        if "start_frame" in clip:
            require(clip["start_frame"] == indices[0] and clip["end_frame"] == indices[-1]+1,
                    "Native recorded clip boundaries changed")
        setup = [frame for frame in frames if frame["phase"] in ("setup", "move")]
        require(setup, "Complete visible setup frames are required")
        excursions = {arm: max(max(v[j] for v in values)-min(v[j] for v in values)
                              for j in range(6)) for arm, values in joints.items()}
        result.append(dict(first_frame=indices[0], frames=len(indices), fps=30,
            first_physics_step=physics_steps[0], last_physics_step=physics_steps[-1],
            native_frame_sha256=hashes, setup_frames=len(setup),
            release_frames=sum(frame["phase"] == "release" for frame in frames),
            foreground_joint_excursion_rad=excursions,
            camera_sha256=sha(run/"selected-camera.json"),
            camera_readback_position_tolerance_m=.0002))
    return result


def encode_native_frames(run, output, clips):
    audits = native_frame_manifest(run, clips)
    encoded = []
    for index, (clip, audit) in enumerate(zip(clips, audits)):
        name = f"trial-{index:03d}.mp4"
        movie = Path(output)/name
        from presentation_export import caption_font
        font = caption_font()
        require(font is not None, "An installed caption font is required")
        mode = clip["mode"]
        require(mode in ("train", "coverage", "eval", "assessment"), "Unknown native film mode")
        caption = ("GEOMETRY COVERAGE - no policy updates" if mode == "coverage" else
                   "PPO TRAINING" if mode == "train" else "FROZEN POLICY EVALUATION")
        batch, update = clip["batch"], clip.get("update_count")
        require(type(batch) is int and (update is None or type(update) is int), "Invalid native film identity")
        caption += f" | batch {batch+1}"
        if update is not None:
            caption += f" | sampling update {update}"
        caption_path = Path(output)/f"caption-{index:03d}.txt"
        caption_path.write_text(caption)
        def filter_path(path):
            return str(Path(path).resolve()).replace("\\", "\\\\").replace("'", "\\'").replace(":", "\\:")
        video_filter = ("drawbox=x=18:y=16:w=1220:h=54:color=0x17202bcc:t=fill,"
                        f"drawtext=fontfile='{filter_path(font)}':textfile='{filter_path(caption_path)}':"
                        "x=32:y=28:fontsize=25:fontcolor=0xeff5fa")
        subprocess.run(["ffmpeg", "-v", "error", "-n", "-framerate", "30",
            "-start_number", str(audit["first_frame"]), "-i", str(Path(run)/"frames/%05d.jpg"),
            "-frames:v", str(audit["frames"]), "-vf", video_filter, "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-threads", "4", "-preset", "fast", "-crf", "18", "-movflags", "+faststart", str(movie)], check=True)
        data = inspect_movie(movie)
        require(data["frames"] == audit["frames"] and data["fps"] == 30,
                "Encoded video differs from native JPEG frame manifest")
        metadata = {key: value for key, value in clip.items()
                    if key not in ("frames", "rows", "rollout", "actions")}
        encoded.append(dict(metadata, file=name, **data, native_frame_audit=audit,
                            non_simulation_content="Data-derived trial mode/batch/update text caption"))
    require(all(any(clip["native_frame_audit"]["foreground_joint_excursion_rad"][arm] > .02
                    for clip in encoded) for arm in ("Arm10", "Arm14")),
            "No measured foreground movement of both arms in the recorded trials")
    return encoded


def export(run, output, *, media=False):
    run, output = Path(run), Path(output)
    require(not output.exists() and not output.resolve().is_relative_to(run.resolve()),
            "Export must use a new directory outside raw evidence")
    summary = summarize(run)
    output.mkdir(parents=True)
    if media and (run/"clips.json").is_file():
        manifest = read(run/"clips.json")
        clips = manifest if isinstance(manifest, list) else manifest["clips"]
        summary["source_sha256"]["clips.json"] = sha(run/"clips.json")
        if clips and isinstance(clips[0].get("frames"), list):
            summary["clips"] = encode_native_frames(run, output, clips)
        else:
            for index, clip in enumerate(clips):
                filename = clip.get("file", clip.get("video_file"))
                path = local_file(run, filename)
                expected = clip.get("sha256", clip.get("video_sha256"))
                require(expected == sha(path), "Native clip hash changed")
                data = inspect_movie(path)
                if "frames" in clip:
                    require(clip["frames"] == data["frames"], "Native clip frame count changed")
                name = f"trial-{index:03d}.mp4"
                shutil.copyfile(path, output/name)
                summary["clips"].append(dict(clip, file=name, **data))
        if summary["clips"]:
            listing = output/"concat.txt"
            listing.write_text("".join(f"file '{item['file']}'\n" for item in summary["clips"]))
            movie = output/"trials.mp4"
            subprocess.run(["ffmpeg", "-v", "error", "-n", "-f", "concat", "-safe", "1",
                "-i", str(listing), "-map", "0:v:0", "-an", "-c", "copy", "-movflags", "+faststart", str(movie)], check=True)
            data = inspect_movie(movie)
            require(data["frames"] == sum(clip["frames"] for clip in summary["clips"]),
                    "Concatenation lost native frames")
            summary["movie"] = dict(file=movie.name, **data,
                selection="All native clips in recorded manifest order; no outcome selection or retiming")
    (output/"report.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--media", action="store_true")
    args = parser.parse_args()
    result = export(args.run, args.output, media=args.media)
    print(json.dumps({key: result[key] for key in ("status", "training", "assessments", "movie")}, allow_nan=False))
