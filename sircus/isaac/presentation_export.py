"""Audit fresh native presentation trials and make one minute of action footage.

No physics, policy, renderer or hardware is launched here. Movie imagery comes
from original Isaac camera JPEGs, with an optional provenance caption. Prefix
selection is chronological and duration-based, never capture-based. Quiet
outcome tails are the only cuts.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np

from action_training_export import (
    BACKDROP, DT, FORCE_THRESHOLD_N, FORCES, MOVE_FRAMES, RELEASE_STRIDE,
    close_record, finite, score_focus, terminal_cut, verify_action_geometry,
    verify_probes,
)
from camera_options import preview_origins
from dark_export import inside_frame
from expanded_farm import validate_action, verify_camera_binding
from expanded_pose_menu import load_native_menu
from export_cylinder_film import matches_action, read, require
from farm_contract import statistics
from presentation_recording import load_source, make_plan
from silver_style import stage_settings
from settling_contract import accepted


VERSION = "sircus-native-presentation-export-v1"
REPORT_SCHEMA = "sircus-native-presentation-recording-v1"
CLIP_SCHEMA = "sircus-native-presentation-clip-v1"
FPS = 30
TARGET_FRAMES = 1800
MAX_SPEED_MULTIPLE = 1.10
MIN_FOREGROUND_DROPS = 10
FOCUS = 1
SCOPE = ("Fresh native Isaac Sim/PhysX presentation trials using frozen qualified "
         "shot configurations. Scripted driven arm transit and initialized ball "
         "release; no new optimizer updates, learned grasp/path, held-out policy "
         "evaluation or physical validation. Native camera imagery with an optional text caption.")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def signed(value, label):
    value = dict(value)
    actual = value.pop("fingerprint", None)
    require(actual == fingerprint(value), label + " fingerprint changed")
    return actual


def choose_prefix(chapters, target_frames=TARGET_FRAMES, maximum_speed=MAX_SPEED_MULTIPLE,
                  minimum_drops=MIN_FOREGROUND_DROPS):
    """First complete chronological prefix, with no outcome-based skipping.

    Only decimation is permitted: source content must already be at least one
    minute, so encoding cannot invent duplicate/frozen frames to stretch it.
    """
    require(target_frames > 0 and maximum_speed >= 1 and minimum_drops > 0,
            "Invalid presentation timing limits")
    selected, count, drops = [], 0, 0
    for index, chapter in enumerate(chapters):
        require(chapter["cycle"] == index, "Presentation cycles are not chronological")
        require(type(chapter["source_frames"]) is int and chapter["source_frames"] > 0,
                "Missing complete action chapter")
        selected.append(chapter)
        count += chapter["source_frames"]
        drops += bool(chapter.get("board_rebound"))
        if count >= target_frames and drops >= minimum_drops:
            require(count / target_frames <= maximum_speed + 1e-12,
                    "Complete prefix needs excessive retiming; record more usable action footage")
            return selected
    raise ValueError("Not enough complete native action footage or distinct foreground rebounds")


def source_frame_map(chapters, target_frames=TARGET_FRAMES):
    source, cycle = [], []
    for chapter in chapters:
        first, last = chapter["source_start_frame"], chapter["source_end_frame_inclusive"]
        require(last - first + 1 == chapter["source_frames"], "Action cut span changed")
        source.extend(range(first, last + 1))
        cycle.extend([chapter["cycle"]] * chapter["source_frames"])
    require(len(source) >= target_frames and len(set(source)) == len(source),
            "Native footage cannot be looped, repeated or padded")
    picks = np.rint(np.linspace(0, len(source)-1, target_frames)).astype(int)
    require(np.all(np.diff(picks) > 0), "Retiming duplicated a camera frame")
    mapping = [dict(encoded_frame=i, source_frame=source[p], cycle=cycle[p],
                    source_action_frame=int(p)) for i, p in enumerate(picks)]
    require(len(mapping) == target_frames, "One-minute frame count changed")
    encoded = []
    for chapter in chapters:
        members = [m for m in mapping if m["cycle"] == chapter["cycle"]]
        require(members, "A complete action chapter disappeared during retiming")
        encoded.append(dict(chapter, start_frame=members[0]["encoded_frame"], frames=len(members),
                            title=f"Native trial {chapter['cycle']+1} — {chapter['terminal_reason']}"))
    return mapping, encoded, len(source)/target_frames


def check_frames(clip, camera, start, count=35):
    frames = clip["frames"]
    move = [f for f in frames if f["phase"] == "move"]
    release = [f for f in frames if f["phase"] == "release"]
    require(len(move) == MOVE_FRAMES and frames[:MOVE_FRAMES] == move and release
            and frames[MOVE_FRAMES:] == release, "Complete driven setup/release frames required")
    require(clip["physics_dt_s"] == DT and clip["release_stride"] == RELEASE_STRIDE,
            "Native release cadence changed")
    for i, frame in enumerate(frames):
        require(frame["frame"] == start+i, "Native frame numbering changed")
        require(np.allclose(finite(frame["eye_m"], (3,), "Bad camera eye"), camera["eye_m"], atol=2e-4, rtol=0)
                and np.allclose(finite(frame["target_m"], (3,), "Bad camera aim"), camera["target_m"], atol=2e-5, rtol=0),
                "The saved camera moved or followed a ball")
        pos = finite(frame["all_ball_world_m"], (count, 3), "Missing farm ball observations")
        require(np.allclose(frame["focus_ball_world_m"], pos[FOCUS], atol=1e-7, rtol=0),
                "Foreground identity changed")
    require(all(b["physics_step"] > a["physics_step"] for a, b in zip(move, move[1:])),
            "Measured setup time is not increasing")
    require(release[0]["physics_step"] == clip["release_step"]
            and all(b["physics_step"]-a["physics_step"] == RELEASE_STRIDE for a,b in zip(release, release[1:])),
            "Ball frame cadence changed")
    motion = {}
    for name in ("Arm10", "Arm14"):
        q = finite([f["focus_joints_rad"][name] for f in move], (MOVE_FRAMES,6), "Missing actual setup joints")
        motion[name] = float(np.max(np.ptp(q, axis=0)))
    return move, release, motion


def background_rebounds(clip, records):
    release = [f for f in clip["frames"] if f["phase"] == "release"]
    visible = set()
    for item in clip["focus_telemetry"]:
        activity = item["activity"]
        require(set(activity) == set(FORCES) and all(type(i) is int and 0 <= i <35 for group in activity.values() for i in group),
                "Bad background contact indices")
        require(all((FOCUS in activity[name]) == (item["forces_N"][name] > FORCE_THRESHOLD_N) for name in FORCES),
                "Foreground contact/activity identity changed")
        for index in activity["board"]:
            if index != FOCUS and not records[index]["invalid"] and records[index].get("board_bounce"):
                nearest = min(release, key=lambda f:abs(f["physics_step"]-item["physics_step"]))
                if inside_frame(nearest, nearest["all_ball_world_m"][index]):
                    visible.add(index)
    return sorted(visible)


def verify_physics(run, clip, records, scene, origins, cells):
    """Reconstruct all35 scalar outcomes from full-rate native observations."""
    name=clip["physics_file"]
    require(isinstance(name,str) and Path(name).name==name and name.endswith(".npz"),"Unsafe native physics filename")
    path=run/name
    require(not path.is_symlink() and clip["physics_sha256"]==sha(path),"Native physics bytes changed")
    with np.load(path,allow_pickle=False)as raw:
        keys=("time_step","position_m","velocity_m_s","forces_N", "joint_position_rad",
              "joint_velocity_rad_s", "board_position_m", "board_quaternion_wxyz")
        require(all(k in raw.files for k in keys),"Incomplete native state packet")
        arrays={k:raw[k]for k in keys}
    steps=arrays["time_step"]
    require(steps.ndim==1 and np.array_equal(steps,np.arange(clip["release_step"]+1,clip["capture_end_step"]+1)),
            "Native physics/clock coverage changed")
    count=len(steps)
    require(clip["physics_cell_ids"] == clip["physics_columns"]["cell_ids"]
            == [c["id"] for c in cells], "Native packet cell ordering changed")
    require(clip["physics_columns"]["force_names"] == list(FORCES)
            and clip["physics_columns"]["arm_names"] == ["Arm10", "Arm14"]
            and clip["physics_columns"]["coordinate_frame"] == "world"
            and clip["physics_columns"]["position_units"] == "metres",
            "Native state packet column meanings changed")
    for key,tail in(("position_m",(3,)),("velocity_m_s",(3,)),("forces_N",(6,))):
        finite(arrays[key],(count,35,*tail),"Native packet shape/nonfinite state: "+key)
    require((arrays["forces_N"]>=0).all(),"Native force magnitude became negative")
    for key,tail in (("joint_position_rad", (2,6)), ("joint_velocity_rad_s", (2,6)),
                     ("board_position_m", (3,)), ("board_quaternion_wxyz", (4,))):
        finite(arrays[key], (count,35,*tail), "Incomplete native arm/board state: " + key)
    require(np.allclose(np.linalg.norm(arrays["board_quaternion_wxyz"], axis=-1), 1, atol=1e-3),
            "Native board quaternions are not normalized")
    for frame in clip["frames"]:
        if frame["phase"] != "release" or frame["physics_step"] == clip["release_step"]:
            continue
        i = frame["physics_step"] - clip["release_step"] - 1
        require(0 <= i < count and np.array_equal(frame["all_ball_world_m"], arrays["position_m"][i]),
                "Camera ball readback differs from native packet")
        for arm,name in enumerate(("Arm10", "Arm14")):
            require(np.array_equal(frame["all_joints_rad"][name], arrays["joint_position_rad"][i,:,arm]),
                    "Camera articulation readback differs from native packet")
    telemetry=clip["focus_telemetry"]
    require(len(telemetry)==count,"Foreground JSON/native packet horizon changed")
    for i,item in enumerate(telemetry):
        require(item["physics_step"]==int(steps[i])
                and np.array_equal(item["ball_world_m"],arrays["position_m"][i,FOCUS])
                and np.array_equal(item["ball_velocity_m_s"],arrays["velocity_m_s"][i,FOCUS])
                and np.array_equal([item["forces_N"][n]for n in FORCES],arrays["forces_N"][i,FOCUS]),
                "Foreground JSON observations differ from native packet")
        require(all(item["activity"][n]==np.flatnonzero(arrays["forces_N"][i,:,j]>FORCE_THRESHOLD_N).tolist()
                    for j,n in enumerate(FORCES)),"Background activity differs from native forces")
    audit=[]
    for world,row in enumerate(records):
        if row["invalid"]:
            require(not row["route_success"],"Rejected release cannot capture")
            audit.append(dict(skipped_invalid=row["invalid"]));continue
        require(accepted(row["settled"]) and row["settled"]["dropper_position_error_mm"]<=2,
                "Native endpoint release guard failed")
        per_world=[dict(physics_step=int(step),ball_world_m=arrays["position_m"][i,world].tolist(),
                        ball_velocity_m_s=arrays["velocity_m_s"][i,world].tolist(),
                        forces_N=dict(zip(FORCES,arrays["forces_N"][i,world].tolist())))
                   for i,step in enumerate(steps)]
        audit.append(score_focus(dict(clip,focus_telemetry=per_world),row,scene,origins[world]))
    board = arrays["board_position_m"][:, FOCUS]
    return dict(file=name, sha256=sha(path), native_steps=count,
                rescored_worlds=sum(not r["invalid"] for r in records),
                focus_board_max_translation_from_release_mm=float(1000*np.max(np.linalg.norm(board-board[0],axis=1)))), audit


def verify(run, source):
    run, source = Path(run), Path(source)
    report, spec, plan = (read(run/name) for name in ("result.json","contract.json","plan.json"))
    signed(spec,"Source contract")
    require(plan.get("schema") == "sircus-native-presentation-plan-v1", "Unknown filming plan")
    require(report.get("schema") == REPORT_SCHEMA and report.get("status") == "completed",
            "Completed native presentation recording required")
    require(report["source_fingerprint"] == spec["fingerprint"] and report["filming_plan"] == plan
            and report["plan_sha256"] == fingerprint(plan), "Source or filming plan changed")
    require(sha(source/"contract.json") == sha(run/"contract.json"), "Original frozen expanded contract bytes changed")
    source_binding = report["source"]
    require(source_binding["contract_sha256"] == sha(source/"contract.json")
            and source_binding["result_sha256"] == sha(source/"result.json")
            and source_binding["menu_sha256"] == sha(run/"expanded-menu-qualified.json")
            and source_binding["camera_sha256"] == sha(run/"selected-camera.json")
            and source_binding["camera_binding_sha256"] == sha(source/"camera-binding.json")
            and source_binding["probes_sha256"] == sha(source/"probes.json")
            and source_binding["rim_rays_sha256"] == sha(source/"rim-rays.json")
            and report["camera_binding_sha256"] == sha(run/"camera-binding.json"), "Pinned source bytes changed")
    require(plan["source"] == source_binding and plan["source_fingerprint"] == spec["fingerprint"]
            and plan["original_training_batches"] == 10 and plan["optimizer_updates"] == 0
            and report["optimizer_updates"] == 0, "Presentation must retain frozen source provenance")
    old_hashes = source_binding["original_source_sha256"]
    current_hashes = source_binding["current_original_dependencies_sha256"]
    require(old_hashes == read(source/"result.json")["source_sha256"]
            and set(current_hashes) == set(old_hashes)
            and source_binding["intentionally_updated_entrypoints"] == ["film_cylinder_farm.py"]
            and all(current_hashes[n] == h for n,h in old_hashes.items() if n != "film_cylinder_farm.py"),
            "Original physics/control dependencies changed")
    recorded = report["source_sha256"]
    require(all(recorded.get(n) == h for n,h in current_hashes.items())
            and all(n in recorded for n in ("presentation_recording.py", "film_cylinder_farm.py")),
            "Native recorder source hashes are incomplete or inconsistent")
    require(spec["version"] == "sircus-expanded-cylinder-film-v1" and spec["training_batches"] == 10,
            "Presentation must not rewrite the source training contract")
    cells = spec["cells"]
    require(len(cells) == report["workcells"] == 35 and report["articulations"] == 70
            and len({tuple(c["bucket_xy_m"]) for c in cells}) == 35,
            "Expected 35 unchanged distinct bucket workcells and70arms")
    origins = finite(report["origins_m"], (35,3), "Bad display origins")
    require(np.array_equal(origins,preview_origins()), "Whole-cell display permutation changed")
    camera,binding = verify_camera_binding(run,spec,origins)
    require(report["camera"] == camera and report["resolution"] == [1920,1080]
            and report["fps"] == FPS and report["physics_dt_s"] == DT
            and plan["camera"] == camera and plan["camera_binding"] == binding,
            "Camera/cadence changed")
    expected_stage = stage_settings();expected_stage["camera_backdrop"] = BACKDROP
    require(report["stage"] == expected_stage, "Silver-gray lighting or floor changed")
    menus,menu_audit = load_native_menu(run/"expanded-menu-qualified.json",spec)
    require(len(menus) ==35 and all(len(m)>=2 for m in menus), "All farm sites need qualified source options")
    verify_probes(run,35)
    batches=report["batches"]
    require(len(batches) == len(plan["cycles"]) == plan["presentation_batches"], "Incomplete filming plan")
    bundle = load_source(source, run/"selected-camera.json")
    require(plan == make_plan(bundle, len(batches)), "Geometry-only filming plan changed")
    require(all(recorded.get(n) == sha(Path(__file__).parent/n)
                for n in ("presentation_recording.py", "film_cylinder_farm.py")),
            "Native recording code differs from the recorded source")
    chapters,allrows,clip_hashes,total=[],[],{},0
    native_frames = {}
    for cycle,summary in enumerate(batches):
        path=run/f"clip-{cycle:02d}.json";clip=read(path);clip_hashes[path.name]=sha(path)
        require(clip["schema"] == CLIP_SCHEMA and clip["cycle"] == cycle and clip["focus"] == FOCUS
                and clip["source_fingerprint"] == spec["fingerprint"]
                and clip["plan_sha256"] == report["plan_sha256"]
                and clip["source"] == source_binding and clip["optimizer_updates"] == 0
                and clip["source_cycle"] == plan["cycles"][cycle]
                and clip["actions"] == plan["cycles"][cycle]["actions"], "Clip source/identity changed")
        records,actions=clip["rollout"],clip["actions"]
        require(len(records) ==len(actions)==35, "Incomplete farm trial")
        for index,(cell,row,action) in enumerate(zip(cells,records,actions)):
            validate_action(cell,action)
            require(row["cell_id"]==cell["id"] and matches_action(row,action), "Workcell/action association changed")
            require(any(matches_action(action,a) for a in menus[index]), "Shot is outside frozen qualified geometry menu")
            verify_action_geometry(row,action,cell,expanded=True)
            if row["route_success"]:
                require(not row["invalid"] and row["board_bounce"] and row["whole_ball_entry"]
                        and row["max_retention_s"]>=1-1e-8 and not row["external_contact_steps"], "Unqualified capture")
        move,release,motion=check_frames(clip,camera,total)
        require(summary["start_frame"]==total and summary["frames"]==len(clip["frames"]), "Clip frame summary changed")
        require(clip["telemetry_end_step"]==clip["capture_end_step"]==clip["focus_telemetry"][-1]["physics_step"]
                and clip["score_end_step"]<=clip["capture_end_step"], "Full scoring horizon unavailable")
        physics,world_audits=verify_physics(run,clip,records,spec["scene"],origins,cells)
        audit = world_audits[FOCUS]
        if records[FOCUS]["invalid"]:
            audit = score_focus(clip,records[FOCUS],spec["scene"],origins[FOCUS])
        cut=terminal_cut(clip,records[FOCUS],audit,origins[FOCUS])
        for frame in clip["frames"]:
            native_frames[frame["frame"]] = dict(phase=frame["phase"], physics_step=frame["physics_step"],
                native_physics_time_s=frame["physics_step"]*DT,
                native_time_since_release_s=(frame["physics_step"]-clip["release_step"])*DT)
        background=background_rebounds(clip,records)
        for index,row in enumerate(records):
            if not row["invalid"]:
                require(np.allclose(np.asarray(release[0]["all_ball_world_m"])[index]-origins[index],row["release_ball_center_m"],atol=.003,rtol=0),
                        "Camera release frame differs from native initialization")
        chapters.append(dict(cycle=cycle,focus=cells[FOCUS]["id"],board_rebound=bool(records[FOCUS]["board_bounce"]),
            native_release=not bool(records[FOCUS]["invalid"]),route_success=bool(records[FOCUS]["route_success"]),
            motion_excursion_rad=motion,background_visible_board_contacts=len(background),background_cell_indices=background,
            native_physics_audit=physics,
            source_physics_start_step=clip["frames"][0]["physics_step"],
            source_physics_end_step=native_frames[cut["source_end_frame_inclusive"]]["physics_step"],
            first_board_contact_step=audit["first_board_contact_step"],**cut))
        allrows.extend(records);total+=len(clip["frames"])
    require(report["frames"]==total and [p.name for p in sorted((run/"frames").glob("*.jpg"))]
            ==[f"{i:05d}.jpg"for i in range(total)], "Native frames incomplete or repeated")
    require(not report.get("updates") and not report.get("ppo_updates"), "Presentation capture must not claim optimizer updates")
    close_record(report["counts"],statistics(allrows),"Native run counts changed")
    selected=choose_prefix(chapters)
    require(all(c["background_visible_board_contacts"]>=2 for c in selected), "Missing concurrently visible background rebounds")
    require(all(any(c["motion_excursion_rad"][a]>.01 for c in selected)for a in("Arm10","Arm14")),
            "Both arms need measured configuration changes")
    mapping,encoded,speed=source_frame_map(selected)
    for item in mapping:
        item.update(native_frames[item["source_frame"]], encoded_time_s=item["encoded_frame"]/FPS,
                    original_camera_index_time_s=item["source_frame"]/FPS)
    return dict(schema=VERSION,status="verified",source_fingerprint=spec["fingerprint"],
        source_result_sha256=sha(run/"result.json"),source_contract_sha256=sha(run/"contract.json"),
        plan_sha256=fingerprint(plan),plan_file_sha256=sha(run/"plan.json"),source_clip_sha256=clip_hashes,
        source_code_sha256=recorded,
        exporter_code_sha256=sha(__file__),
        selected_camera_sha256=sha(run/"selected-camera.json"),camera=camera,camera_binding=binding,
        source=source_binding,geometry_menu=menu_audit,workcells=35,articulations=70,
        all_cycles=len(chapters),retained_complete_prefix_cycles=len(selected),
        omitted_cycles_reason="Chronological duration-only suffix; all source cycles and outcomes remain in raw evidence",
        native_counts=statistics(allrows),raw_frames=total,raw_duration_s=total/FPS,
        native_contact_probe_sha256=sha(run/"probes.json"),native_rim_probe_sha256=sha(run/"rim-rays.json"),
        fps=FPS,resolution=[1920,1080],duration_s=60.,frames=TARGET_FRAMES,
        source_action_frames=sum(c["source_frames"]for c in selected),playback_multiple=speed,
        retiming="Uniform source-frame decimation only; no duplicates, interpolation, loops or frozen padding",
        setup_timing="36 measured joint-progress frames per driven setup; compressed timing, not real-time controller execution",
        release_timing="Native release frames every8/240s; output applies declared uniform playback multiple",
        scope=SCOPE,all_chapters=chapters,chapters=encoded,frame_map=mapping)


def caption_font():
    """Use an installed font; download or generate no visual assets."""
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    )
    return next((Path(p) for p in candidates if Path(p).is_file()), None)


def encode_presentation_movie(frames_dir, movie, chapters, resolution, caption=True):
    """H.264 encode actual camera JPEGs with a small, optional provenance label."""
    require(resolution == [1920,1080], "Native 1080p imagery required")
    count, metadata = 0, [";FFMETADATA1"]
    for chapter in chapters:
        require(chapter["start_frame"] == count, "Encoded chapters are not contiguous")
        metadata.extend(("[CHAPTER]", "TIMEBASE=1/30", f"START={count}",
                         f"END={count+chapter['frames']}", f"title={chapter['title']}"))
        count += chapter["frames"]
    require(count == TARGET_FRAMES, "Exactly 1800 encoded native frames required")
    movie = Path(movie)
    metadata_path = movie.with_suffix(".ffmeta")
    metadata_path.write_text("\n".join(metadata)+"\n")
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-n", "-framerate", "30",
               "-start_number", "0", "-i", str(frames_dir/"%05d.jpg"), "-i", str(metadata_path),
               "-map", "0:v:0", "-map_metadata", "1", "-map_chapters", "1", "-an"]
    with tempfile.TemporaryDirectory(prefix="sircus-film-caption-") as temp:
        text_files, font = [], None
        if caption:
            font = caption_font()
            require(font is not None, "No installed caption font; install DejaVu Sans or use --no-caption")
            lines = ("SIRCUS · Native Isaac Sim", "Saved configuration trials · action edit")
            for index,line in enumerate(lines):
                path = Path(temp)/f"caption-{index}.txt"
                path.write_text(line)
                text_files.append(path)
            # These paths are tool-created fixed names; escaping protects filter syntax.
            def escape(path):
                return str(path).replace("\\", "\\\\").replace(":", "\\:").replace("'", "'\\''")
            filters = []
            for index,path in enumerate(text_files):
                filters.append(
                    f"drawtext=fontfile='{escape(font)}':textfile='{escape(path)}':"
                    f"fontsize={22 if index == 0 else 16}:fontcolor={('0xEDF1F4' if index == 0 else '0xD1D8E0')}@0.94:"
                    f"x=28:y={24 if index == 0 else 54}:shadowcolor=0x1C222A@0.6:shadowx=1:shadowy=1")
            command.extend(("-vf", ",".join(filters)))
        command.extend(("-c:v", "libx264", "-frames:v", str(count), "-pix_fmt", "yuv420p",
                        "-threads", "4", "-preset", "medium", "-crf", "17", "-movflags", "+faststart", str(movie)))
        subprocess.run(command, check=True)
    data = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-count_frames", "-show_streams", "-show_chapters",
        "-show_format", "-of", "json", str(movie)]))
    videos = [s for s in data["streams"] if s["codec_type"] == "video"]
    require(len(videos) == 1, "Movie has an unexpected video stream")
    stream = videos[0]
    require(int(stream["nb_read_frames"]) == count and stream["avg_frame_rate"] == "30/1"
            and stream["codec_name"] == "h264" and stream["pix_fmt"] == "yuv420p"
            and [stream["width"],stream["height"]] == resolution
            and abs(float(stream["duration"])-60) < 0.002
            and abs(float(data["format"]["duration"])-60) < 0.002
            and len(data["chapters"]) == len(chapters), "Encoded movie differs from the verified frame map")
    for chapter,encoded in zip(chapters,data["chapters"]):
        require(abs(float(encoded["start_time"])-chapter["start_frame"]/FPS) < 0.002
                and abs(float(encoded["end_time"])-(chapter["start_frame"]+chapter["frames"])/FPS) < 0.002
                and encoded.get("tags",{}).get("title") == chapter["title"], "Encoded chapter map changed")
    subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-i", str(movie), "-map", "0:v:0",
                    "-f", "null", "-"], check=True, stdout=subprocess.DEVNULL)
    return dict(file=movie.name, sha256=sha(movie), frames=count, duration_s=60.,
                codec="h264", fps=30, resolution=resolution, decode_verified=True,
                non_simulation_content=["text_caption"] if caption else [],
                caption_lines=list(lines) if caption else [], caption_font=str(font) if caption else None)


def render_movie(run,output,summary,caption=True):
    """Symlink1800 different originals; compress native imagery and optional text."""
    run = Path(run)
    with tempfile.TemporaryDirectory(prefix="sircus-presentation-cut-")as temp:
        root=Path(temp)
        for item in summary["frame_map"]:
            source = run/"frames"/f"{item['source_frame']:05d}.jpg"
            require(not source.is_symlink(), "Native frame source must be an actual camera file")
            item["source_frame_sha256"] = sha(source)
            (root/f"{item['encoded_frame']:05d}.jpg").symlink_to(source.resolve())
        movie=encode_presentation_movie(root,output,summary["chapters"],summary["resolution"],caption=caption)
    require(movie["frames"]==1800 and movie["duration_s"]==60,"Encoded presentation is not one minute")
    return movie


def poster_and_contact_sheet(run,out,summary):
    from PIL import Image,ImageDraw

    sources=[]
    for chapter in summary["chapters"][:6]:
        clip=read(run/f"clip-{chapter['cycle']:02d}.json")
        release=[f for f in clip["frames"]if f["phase"]=="release"]
        for step in(clip["release_step"],chapter["first_board_contact_step"] or clip["release_step"],chapter["terminal_physics_step"]):
            sources.append(min(release,key=lambda f:abs(f["physics_step"]-step))["frame"])
    require(sources,"No native inspection frames")
    poster=sources[1]if len(sources)>1 else sources[0]
    shutil.copyfile(run/"frames"/f"{poster:05d}.jpg",out/"farm-poster.jpg")
    sheet=Image.new("RGB",(1152,240*math.ceil(len(sources)/3)),(29,33,38));draw=ImageDraw.Draw(sheet)
    for i,source in enumerate(sources):
        image=Image.open(run/"frames"/f"{source:05d}.jpg").convert("RGB");image.thumbnail((384,216))
        x,y=(i%3)*384,(i//3)*240;sheet.paste(image,(x,y));draw.text((x+8,y+220),f"Native frame {source}",fill=(235,239,242))
    sheet.save(out/"contact-sheet.jpg",quality=94)
    return dict(poster_frame=poster,poster_sha256=sha(out/"farm-poster.jpg"),
                contact_sheet_source_frames=sources,contact_sheet_sha256=sha(out/"contact-sheet.jpg"))


def export(run,out,source,caption=True):
    run,out=Path(run),Path(out)
    require(not out.exists() and not out.resolve().is_relative_to(run.resolve()),"New export directory outside raw evidence required")
    summary=verify(run,source)
    out.mkdir(parents=True)
    summary["movie"]=render_movie(run,out/"farm-one-minute.mp4",summary,caption=caption)
    summary["inspection_images"]=poster_and_contact_sheet(run,out,summary)
    (out/"film.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:summary[k]for k in("status","all_cycles","retained_complete_prefix_cycles","frames","duration_s","playback_multiple","movie")}),flush=True)
    return summary


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run",type=Path);parser.add_argument("output",type=Path)
    parser.add_argument("--source",type=Path,required=True)
    parser.add_argument("--no-caption", action="store_true", help="Encode native imagery without the small provenance label")
    args=parser.parse_args();export(args.run,args.output,args.source,caption=not args.no_caption)
