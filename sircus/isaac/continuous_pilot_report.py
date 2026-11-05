"""Summarize the continuous-shot pilot without running physics or a policy.

Every rejected active shot remains in the denominator. Endpoint-map samples,
on-policy training outcomes and frozen evaluation are kept in separate groups.
The native trajectory/likelihood audit is a separate deliverable: this report
checks scalar outcome consistency, source hashes and optional video decoding.
"""

import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess


SCHEMA = "sircus-continuous-pilot-report-v1"
FOREGROUND_CELL = 1


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


def requested_parameters(action, ball_radius_m=None):
    """Requested shot geometry only; never call an initialized ball a sensor."""
    result = {}
    for key in ("board_x_m", "board_y_m", "height_m", "gap_m", "radius_m",
                "tilt_rad", "tilt_deg", "bearing_rad", "bearing_offset_rad"):
        if key in action:
            value = action[key]
            require(type(value) in (int, float) and math.isfinite(value), "Non-finite requested shot parameter")
            result[key] = value
    if "tilt_deg" not in result and "tilt_rad" in result:
        result["tilt_deg"] = math.degrees(result["tilt_rad"])
    if "bearing_rad" in result:
        result["bearing_deg"] = math.degrees(result["bearing_rad"]) % 360
    if "bearing_offset_rad" in result:
        result["bearing_offset_deg"] = math.degrees(result["bearing_offset_rad"])
    if "height_m" in result and "gap_m" in result:
        result["release_ball_bottom_height_m"] = result["height_m"]+result["gap_m"]
        if ball_radius_m is not None:
            require(type(ball_radius_m) in (int, float) and math.isfinite(ball_radius_m)
                    and ball_radius_m > 0, "Invalid declared ball radius")
            result["release_ball_center_height_m"] = result["release_ball_bottom_height_m"]+ball_radius_m
            if "board_x_m" in result and "board_y_m" in result:
                result["release_ball_center_m"] = [result["board_x_m"], result["board_y_m"],
                                                   result["release_ball_center_height_m"]]
    return result


def outcome_status(row):
    if "invalid" not in row:
        state = "unknown"
    elif row["invalid"]:
        state = "release_withheld"
    elif row.get("route_success") is True:
        state = "captured"
    elif row.get("route_success") is False:
        state = "released_no_qualified_capture"
    else:
        state = "unknown"
    return dict(state=state, invalid_reason=row.get("invalid"),
                admitted=not row["invalid"] if "invalid" in row else None,
                captured=row.get("route_success"), clean=row.get("route_quality") == 2,
                board_bounce=row.get("board_bounce"), whole_ball_entry=row.get("whole_ball_entry"),
                max_retention_s=row.get("max_retention_s"),
                external_contact_steps=row.get("external_contact_steps"))


def prediction_record(trial, row, action, geometry, filename):
    radius = (geometry or {}).get("ball", {}).get("radius_m")
    return dict(update_count=trial.get("update_count", trial.get("update_count_before")),
                policy_hash=trial["policy_hash"], repeat=trial.get("batch"), source_trial=filename,
                requested=requested_parameters(action, radius),
                measured_board=dict(center_m=row.get("settled_board_center_m"),
                    normal=row.get("settled_board_normal"), settling_errors=row.get("settled")),
                outcome=outcome_status(row),
                coordinate_frame="Table-local metres; board normal follows common world/table axes",
                release_provenance="Requested initialized ball-center pose, not measured gripper/ball localization")


def foreground_metadata(clip, geometry):
    require(len(clip["actions"]) == len(clip["rows"]) == 35, "Foreground settings need the original35 native cells")
    action, row = clip["actions"][FOREGROUND_CELL], clip["rows"][FOREGROUND_CELL]
    radius = geometry["ball"]["radius_m"]
    requested = requested_parameters(action, radius)
    require(all(key in requested for key in ("height_m", "gap_m", "release_ball_center_height_m",
                "radius_m", "tilt_deg", "bearing_offset_deg")), "Incomplete requested foreground geometry")
    return dict(cell_id=row["cell_id"], requested=requested, outcome=outcome_status(row),
                parameter_provenance="Commanded/requested geometry; ball is initialized at release. Not measured gripper or perception output.")


def foreground_caption(foreground):
    p = foreground["requested"]
    return (f"Requested FG (mm): board H {p['height_m']*1000:.0f} | gap {p['gap_m']*1000:.0f} | "
            f"ball-center release {p['release_ball_center_height_m']*1000:.1f} | radius {p['radius_m']*1000:.0f} "
            f"| tilt {p['tilt_deg']:.1f}° | bearing offset {p['bearing_offset_deg']:+.1f}°")


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
    geometry = None
    if (run/"contract.json").is_file():
        contract = read(local_file(run, "contract.json"))
        scene = contract.get("scene", {})
        geometry = {key: scene[key] for key in ("table", "bucket", "ball", "board") if key in scene}
        source_hashes["contract.json"] = sha(run/"contract.json")
    feasibility = read(local_file(run, "feasibility.json"))
    if "source_fingerprint" in feasibility:
        require(feasibility["source_fingerprint"] == study.get("geometry_fingerprint"),
                "Feasibility map belongs to a different scene")
    source_hashes["feasibility.json"] = sha(run/"feasibility.json")
    map_rows = [dict(row, valid=row.get("valid", row.get("accepted")))
                for row in feasibility.get("rows", feasibility.get("records", []))]
    require(map_rows, "Missing sampled feasibility records")
    require(all(row.get("cell_id") in ids and type(row.get("valid")) is bool for row in map_rows),
            "Invalid endpoint-map record")
    batches, train_rows, coverage_rows, evaluation, trial_rows = [], [], [], collections.defaultdict(list), []
    evaluation_deterministic = collections.defaultdict(list)
    predictions = {}
    all_actions = []
    paths = sorted((run/"trials").glob("*.json"))
    for path in paths:
        require(not path.is_symlink(), "Trial symlink is not evidence")
        trial = read(path)
        if "study_fingerprint" in trial:
            require(trial["study_fingerprint"] == study.get("fingerprint"),
                    "Trial belongs to a different continuous study")
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
                if trial.get("deterministic") is True:
                    action = row
                    if len(trial.get("actions", [])) == len(cells):
                        index = next(i for i, cell in enumerate(cells) if cell["id"] == row["cell_id"])
                        action = trial["actions"][index]
                    predictions[(row["cell_id"], update)] = prediction_record(
                        trial, row, action, geometry, "trials/"+path.name)
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
            poses=[dict(proposal=row.get("proposal"), action=row["action"],
                        qualification="Geometry, independent FK and coarse clearance proxies only; no native capture label")
                   for row in mapped if row["valid"]],
            action_ranges=numeric_ranges(row["action"] for row in mapped if row["valid"])),
            training=counts(row for row in train_rows if row["cell_id"] == name),
            coverage=counts(row for row in coverage_rows if row["cell_id"] == name),
            frozen_predictions=[predictions[key] for key in sorted(predictions) if key[0] == name],
            evaluation=[item for item in heldout if item["attempts"]]))
    result_path = run/"result.json"
    result = read(result_path) if result_path.is_file() else {}
    if result:
        require(result.get("study_fingerprint") == study.get("fingerprint"),
                "Native result belongs to a different study")
        source_hashes["result.json"] = sha(result_path)
        if result.get("status") == "completed":
            expected = study.get("training_batches")
            if expected is not None:
                require([item["batch"] for item in batches] == list(range(expected)),
                        "Completed study has missing or reordered training attempts")
            declared = [entry for key in ("batches", "assessments", "coverage")
                        for entry in result.get(key, [])]
            require({entry["file"] for entry in declared} == {"trials/"+path.name for path in paths},
                    "Completed result does not bind every native trial")
            for entry in declared:
                require(entry["sha256"] == sha(local_file(run, entry["file"])),
                        "Native trial changed after the completed result was saved")
    audits = {}
    for label, filename in (("native", "native-audit.json"), ("checkpoint", "checkpoint-audit.json"),
                            ("feasibility", "feasibility-audit.json"), ("other", "audit.json")):
        audit_path = run/filename
        if audit_path.is_file():
            value = read(local_file(run, filename))
            if label != "feasibility" and "study_fingerprint" in value:
                require(value["study_fingerprint"] == study.get("fingerprint"), "Audit belongs to a different study")
            if "source_result_sha256" in value:
                audited = run/"feasibility.json" if label == "feasibility" else result_path
                require(audited.is_file() and value["source_result_sha256"] == sha(audited), "Audit belongs to different result bytes")
            audits[label] = dict(file=filename, sha256=sha(audit_path), report=value)
            source_hashes[filename] = sha(audit_path)
    native_audit = audits.get("native", audits.get("other", {})).get("report")
    return dict(schema=SCHEMA, status=result.get("status", "in_progress"),
                exporter_code_sha256=sha(__file__),
                study_fingerprint=study.get("fingerprint"),
                geometry_fingerprint=study.get("geometry_fingerprint"),
                scope="One-step privileged-geometry continuous-action PPO pilot on 35 inherited bucket sites; current 25/5/5 optimization split is not globally unseen-layout evaluation. Endpoint feasibility is not collision-certified motion planning. Scripted arm transit and initialized ball release; grasp and joint paths are not learned.",
                scoring="Board bounce + whole-ball entry + one continuous second retained; no detected external route contact. Rim-assisted captures count; clean captures reported separately.",
                uncertainty="Bucket reported height 175 mm remains unresolved overall-height versus interior-depth. Contact properties and rigid board mount remain provisional.",
                study=study, cells=cell_results, feasibility=dict(attempts=len(map_rows),
                    admitted=sum(row["valid"] for row in map_rows),
                    rejected=sum(not row["valid"] for row in map_rows),
                    ik_protocol=feasibility.get("ik_protocol"), scope=feasibility.get("scope"),
                    reasons=dict(collections.Counter(str(row.get("reason") or "admitted") for row in map_rows))),
                geometry=geometry,
                training=counts(train_rows), coverage=counts(coverage_rows),
                coverage_scope="Prescheduled geometry-coverage attempts; outcomes excluded from policy updates and frozen evaluation.",
                learning_curve=batches, assessments=checks,
                sampled_training_action_ranges=numeric_ranges(all_actions),
                trial_files=len(paths), elapsed_s=result.get("elapsed_s", result.get("seconds")),
                native_audit=native_audit, audits=audits,
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
        home = None
        if "trial_file" in clip:
            trial = read(local_file(run, clip["trial_file"]))
            require(all(clip[key] == trial[key] for key in
                        ("mode", "batch", "update_count", "actions", "rows", "release_step")),
                    "Native film does not match its recorded policy/coverage trial")
            home = trial.get("home")
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
            # native 2e-4 m readback gate; target is the requested double vector.
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
            camera_readback_position_tolerance_m=.0002,
            episode_home=home))
    return result


def encode_native_frames(run, output, clips):
    audits = native_frame_manifest(run, clips)
    geometry = read(local_file(run, "contract.json"))["scene"]
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
        home = audit["episode_home"] or {}
        reset = home.get("episode_reset") is True
        if reset:
            require(home.get("protocol") == "canonical-episode-home-reset-before-visible-drive-v1",
                    "Unknown episode-reset protocol cannot be labeled in the video")
        setup_caption = ("Episode reset before visible goal travel; scripted joint drives; initialized ball release."
                         if reset else "Native articulated setup; scripted joint drives; initialized ball release.")
        setup_caption_path = Path(output)/f"setup-caption-{index:03d}.txt"
        setup_caption_path.write_text(setup_caption)
        foreground = foreground_metadata(clip, geometry)
        requested_caption_path = Path(output)/f"requested-caption-{index:03d}.txt"
        requested_caption_path.write_text(foreground_caption(foreground))
        def filter_path(path):
            return str(Path(path).resolve()).replace("\\", "\\\\").replace("'", "\\'").replace(":", "\\:")
        video_filter = ("drawbox=x=18:y=16:w=1220:h=116:color=0x17202bcc:t=fill,"
                        f"drawtext=fontfile='{filter_path(font)}':textfile='{filter_path(caption_path)}':"
                        "x=32:y=28:fontsize=25:fontcolor=0xeff5fa,"
                        f"drawtext=fontfile='{filter_path(font)}':textfile='{filter_path(setup_caption_path)}':"
                        "x=32:y=67:fontsize=18:fontcolor=0xc5d4e4,"
                        f"drawtext=fontfile='{filter_path(font)}':textfile='{filter_path(requested_caption_path)}':"
                        "x=32:y=96:fontsize=18:fontcolor=0xedcf96")
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
                            foreground=foreground,
                            non_simulation_content="Data-derived trial mode/batch/update, episode-setup provenance and requested foreground geometry captions; no future outcome displayed"))
    require(all(any(clip["native_frame_audit"]["foreground_joint_excursion_rad"][arm] > .02
                    for clip in encoded) for arm in ("Arm10", "Arm14")),
            "No measured foreground movement of both arms in the recorded trials")
    return encoded


def action_frame_indices(frames, *, context_frames=12, joint_change_rad=.0001):
    """Remove only setup dwells with no motion above conservative readback gates.

    Keep every release frame, first/last context and every trial. These gates
    are declared editorial thresholds, not a statement that motors are exactly
    stationary. Full normal-time native clips are always preserved separately.
    """
    require(frames and type(context_frames) is int and context_frames >= 0
            and joint_change_rad > 0, "Invalid action-edit thresholds")
    count = len(frames)
    keep = set(range(min(count, context_frames+1)))
    keep.update(range(max(0, count-context_frames-1), count))
    for index, frame in enumerate(frames):
        if frame["phase"] == "release":
            keep.add(index)
        if not index:
            continue
        before = frames[index-1]
        window_before = frames[max(0, index-15)]
        joints = max(abs(x-y) for arm in ("Arm10", "Arm14")
                     for a, b in zip(before["all_joints_rad"][arm], frame["all_joints_rad"][arm])
                     for x, y in zip(a, b))
        slow_joints = max(abs(x-y) for arm in ("Arm10", "Arm14")
                          for a, b in zip(window_before["all_joints_rad"][arm], frame["all_joints_rad"][arm])
                          for x, y in zip(a, b)) >= .0005
        # During setup the balls are parked, unscored initialization objects;
        # escaped setup balls must not keep an otherwise idle arm dwell alive.
        # Every recorded release frame above is retained regardless of motion.
        if joints >= joint_change_rad or slow_joints:
            keep.update(range(max(0, index-1-context_frames), min(count, index+context_frames+1)))
    result = sorted(keep)
    require(result and result[0] == 0 and result[-1] == count-1
            and all(i in keep for i, frame in enumerate(frames) if frame["phase"] == "release"),
            "Action edit lost a trial boundary or ball-flight frame")
    return result


def contiguous_spans(indices):
    require(indices and all(b > a for a, b in zip(indices, indices[1:])),
            "Action frame indices must be unique and chronological")
    result = []
    first = previous = indices[0]
    for value in indices[1:]:
        if value > previous+1:
            result.append([first, previous])
            first = value
        previous = value
    result.append([first, previous])
    return result


def make_action_edit(output, source_clips, encoded):
    output = Path(output)
    require(len(source_clips) == len(encoded), "Missing chronological action-edit source clips")
    result = []
    for index, (source, clip) in enumerate(zip(source_clips, encoded)):
        indices = action_frame_indices(source["frames"])
        spans = contiguous_spans(indices)
        select = "+".join(f"between(n,{first},{last})" for first, last in spans)
        name = f"action-trial-{index:03d}.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-n", "-i", str(output/clip["file"]),
            "-vf", f"select='{select}',setpts=N/(30*TB)", "-an", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", "-threads", "4", "-preset", "fast", "-crf", "18",
            "-r", "30", "-movflags", "+faststart", str(output/name)], check=True)
        data = inspect_movie(output/name)
        require(data["frames"] == len(indices), "Action edit added or lost selected native frames")
        result.append(dict(file=name, **data, mode=source["mode"], batch=source["batch"],
            original_clip=clip["file"], original_frames=clip["frames"],
            retained_native_global_frames=[source["frames"][i]["frame"] for i in indices],
            source_local_spans_inclusive=spans,
            removed_setup_dwell_frames=clip["frames"]-len(indices)))
    listing = output/"action-concat.txt"
    listing.write_text("".join(f"file '{item['file']}'\n" for item in result))
    movie = output/"action-trials.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-n", "-f", "concat", "-safe", "1", "-i", str(listing),
        "-map", "0:v:0", "-an", "-c", "copy", "-movflags", "+faststart", str(movie)], check=True)
    data = inspect_movie(movie)
    require(data["frames"] == sum(clip["frames"] for clip in result), "Action movie lost native chapters")
    return dict(file=movie.name, **data, clips=result,
        selection="All recorded trials in order; only setup dwell below declared arm-joint motion thresholds removed. Parked/unscored setup-ball motion is ignored; every release frame preserved, no speedup.",
        thresholds=dict(joint_change_rad_per_frame=.0001,
                        slow_window_frames=15, joint_change_rad_per_window=.0005,
                        before_after_context_frames=12, fps=30,
                        setup_ball_motion="Ignored: balls are parked/unscored during setup; all release frames are kept"),
        full_untrimmed_movie="trials.mp4")


def write_settings_csv(summary, path):
    """One row per cell; frozen predictions only, never a best-shot search."""
    fields = ["cell_id", "split", "study_status", "bucket_x_m", "bucket_y_m",
              "map_requests", "map_admitted", "initial_update", "initial_attempts",
              "initial_captures", "initial_invalid", "latest_update", "latest_attempts",
              "latest_captures", "latest_invalid", "latest_admitted", "latest_invalid_reason",
              "latest_outcome", "requested_board_x_m", "requested_board_y_m", "requested_board_height_m",
              "requested_gap_m", "requested_ball_center_release_height_m", "requested_radius_m",
              "requested_tilt_deg", "requested_bearing_deg", "requested_bearing_offset_deg",
              "measured_board_x_m", "measured_board_y_m", "measured_board_z_m",
              "measured_board_normal_x", "measured_board_normal_y", "measured_board_normal_z",
              "board_position_error_mm", "board_normal_error_deg", "joint_speed_rad_s",
              "max_retention_s", "external_contact_steps", "prediction_trial", "release_provenance"]
    with Path(path).open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for cell in summary["cells"]:
            checks = sorted(cell["evaluation"], key=lambda value: value["update_count"])
            initial, latest = (checks[0], checks[-1]) if checks else ({}, {})
            predictions = sorted(cell.get("frozen_predictions", []), key=lambda value: value["update_count"])
            prediction = predictions[-1] if predictions else {}
            requested = prediction.get("requested", {})
            outcome = prediction.get("outcome", {})
            measured = prediction.get("measured_board", {})
            center, normal = measured.get("center_m"), measured.get("normal")
            errors = measured.get("settling_errors") or {}
            row = dict(cell_id=cell["id"], split=cell["split"], study_status=summary["status"],
                bucket_x_m=cell["bucket_xy_m"][0], bucket_y_m=cell["bucket_xy_m"][1],
                map_requests=cell["feasibility"]["attempts"], map_admitted=cell["feasibility"]["admitted"],
                latest_admitted=outcome.get("admitted"), latest_invalid_reason=outcome.get("invalid_reason"),
                latest_outcome=outcome.get("state"), prediction_trial=prediction.get("source_trial"),
                max_retention_s=outcome.get("max_retention_s"),
                external_contact_steps=outcome.get("external_contact_steps"),
                board_position_error_mm=errors.get("position_error_mm"),
                board_normal_error_deg=errors.get("normal_error_deg"), joint_speed_rad_s=errors.get("joint_speed_rad_s"),
                release_provenance=prediction.get("release_provenance"))
            for prefix, value in (("initial", initial), ("latest", latest)):
                for target, source in (("update", "update_count"), ("attempts", "attempts"),
                                       ("captures", "captures"), ("invalid", "invalid")):
                    row[f"{prefix}_{target}"] = value.get(source)
            for target, source in (("board_x_m", "board_x_m"), ("board_y_m", "board_y_m"),
                ("board_height_m", "height_m"), ("gap_m", "gap_m"),
                ("ball_center_release_height_m", "release_ball_center_height_m"), ("radius_m", "radius_m"),
                ("tilt_deg", "tilt_deg"), ("bearing_deg", "bearing_deg"), ("bearing_offset_deg", "bearing_offset_deg")):
                row["requested_"+target] = requested.get(source)
            for axis, index in zip("xyz", range(3)):
                row["measured_board_"+axis+"_m"] = center[index] if center is not None else None
                row["measured_board_normal_"+axis] = normal[index] if normal is not None else None
            writer.writerow(row)
    return dict(file=Path(path).name, sha256=sha(path), text=Path(path).read_bytes().decode("utf-8"),
                cells=len(summary["cells"]),
                scope="Initial and latest frozen-policy outcomes and requested geometry; no per-cell search or measured gripper pose. Latest is final only after completed collection.")


def export(run, output, *, media=False, action_edit=False):
    run, output = Path(run), Path(output)
    require(not output.exists() and not output.resolve().is_relative_to(run.resolve()),
            "Export must use a new directory outside raw evidence")
    summary = summarize(run)
    output.mkdir(parents=True)
    summary["settings_csv"] = write_settings_csv(summary, output/"settings.csv")
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
            if action_edit:
                require(all(isinstance(clip.get("frames"), list) for clip in clips),
                        "Action edit requires actual native frame/joint/ball metadata")
                summary["action_movie"] = make_action_edit(output, clips, summary["clips"])
    (output/"report.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--media", action="store_true")
    parser.add_argument("--action-edit", action="store_true", help="Also remove only measured setup dwells, preserving full clips")
    args = parser.parse_args()
    result = export(args.run, args.output, media=args.media, action_edit=args.action_edit)
    print(json.dumps({key: result[key] for key in ("status", "training", "assessments", "movie")}, allow_nan=False))
