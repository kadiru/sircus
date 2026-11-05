"""Audit a genuine categorical-PPO run and edit its scheduled native RGB trials.

This module performs CPU analysis and software video encoding only. It never
launches Isaac, updates a policy, chooses a successful shot, or controls hardware.
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
    DT, FOCUS, close_record, finite, score_focus, terminal_cut,
    verify_action_geometry, verify_probes,
)
from camera_options import preview_origins
from expanded_farm import validate_action, verify_camera_binding
from expanded_pose_menu import load_native_menu
from export_cylinder_film import matches_action, read, require
from farm_contract import reward, statistics
from presentation_export import (
    background_rebounds, caption_font, check_frames, sha, verify_physics,
    verify_stage,
)
from presentation_recording import load_source
from settling_contract import accepted


SCHEMA = "sircus-native-rl-film-export-v1"
RESULT_SCHEMA = "sircus-native-rl-film-result-v1"
TRIAL_SCHEMA = "sircus-native-rl-film-trial-v1"
CLIP_SCHEMA = "sircus-native-rl-film-clip-v1"
AUDIT_SCHEMA = "sircus-native-rl-film-checkpoint-audit-v1"
POLICY_VERSION = "sircus-categorical-shot-policy-v1"
SETUP_PROTOCOL = "scripted-visual-transit-then-common-canonical-endpoint-init-v1"
FILMED = tuple(range(8)) + tuple(range(60, 68)) + tuple(range(120, 128))
ASSESSMENT_STAGES = (0, 32, 64, 128)
FPS = 30
SCOPE = (
    "Fresh native Isaac Sim categorical-PPO training collection. The actor "
    "selects among 24 native-qualified configurations per known bucket layout. "
    "Filmed scripted transit is a visualization before common canonical episode "
    "initialization; grasping and joint paths are not learned. Native camera "
    "pixels receive explicitly data-derived text/curve overlays. Assessments "
    "use the same 35 known layouts with no optimizer updates; they are not "
    "held-out spatial generalization or physical validation."
)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def signed(value, label):
    unsigned = dict(value)
    actual = unsigned.pop("fingerprint", None)
    require(actual == fingerprint(unsigned), label + " fingerprint changed")
    return actual


def safe_file(run, filename, suffix):
    require(isinstance(filename, str) and Path(filename).name == filename
            and filename.endswith(suffix), "Unsafe evidence filename")
    path = Path(run) / filename
    require(path.is_file() and not path.is_symlink(), "Missing evidence file: " + filename)
    return path


def rates(rows, window=8):
    """Use every attempted slot, including rejected releases, in capture rates."""
    require(len(rows) == 128 and window == 8, "Expected the fixed 128-batch study")
    result, captures, attempts = [], 0, 0
    for batch, records in enumerate(rows):
        require(len(records) == 35, "Incomplete training batch")
        summary = statistics(records)
        recent = [r for group in rows[max(0, batch-window+1):batch+1] for r in group]
        recent_summary = statistics(recent)
        captures += summary["captures"]
        attempts += 35
        result.append(dict(
            batch=batch, completed_train_batches=batch+1,
            optimizer_update_before=batch//2, attempts=35,
            captures=summary["captures"], invalid=summary["invalid"],
            mean_reward=summary["mean_reward"], batch_capture_rate=summary["captures"]/35,
            cumulative_captures=captures, cumulative_attempts=attempts,
            cumulative_capture_rate=captures/attempts,
            rolling_window_batches=min(batch+1, window),
            rolling_captures=recent_summary["captures"],
            rolling_attempts=recent_summary["attempts"],
            rolling_capture_rate=recent_summary["captures"]/recent_summary["attempts"],
        ))
    return result


def wilson(captures, attempts, z=1.959963984540054):
    require(type(captures) is int and type(attempts) is int
            and 0 <= captures <= attempts and attempts > 0, "Invalid capture counts")
    p = captures / attempts
    denominator = 1 + z*z/attempts
    midpoint = (p + z*z/(2*attempts)) / denominator
    radius = z*math.sqrt(p*(1-p)/attempts + z*z/(4*attempts*attempts))/denominator
    return [0. if captures == 0 else max(0., midpoint-radius),
            1. if captures == attempts else min(1., midpoint+radius)]


def phase(batch):
    require(batch in FILMED, "A non-scheduled batch cannot enter the film")
    return "Early training" if batch < 8 else "Middle training" if batch < 68 else "Late training"


def frame_map(chapters):
    """Retain all scheduled chapters in order, without looping or padding."""
    require([c["batch"] for c in chapters] == list(FILMED), "Filmed schedule changed")
    frames, batches, totals = [], [], []
    for chapter in chapters:
        first, last = chapter["source_start_frame"], chapter["source_end_frame_inclusive"]
        require(last-first+1 == chapter["source_frames"] and last >= first,
                "Incomplete scheduled action chapter")
        frames.extend(range(first, last+1))
        batches.extend([chapter["batch"]] * chapter["source_frames"])
        totals.append(chapter["source_frames"])
    require(len(set(frames)) == len(frames) and all(b > a for a,b in zip(frames, frames[1:])),
            "Camera footage is duplicated or reordered")
    # Preserve 1x flight whenever possible; shorten only a >90-second edit.
    target = min(len(frames), 90*FPS)
    require(target > 0 and len(frames)/target <= 1.10,
            "Scheduled footage needs excessive retiming")
    picks = np.rint(np.linspace(0, len(frames)-1, target)).astype(int)
    require(np.all(np.diff(picks) > 0), "Retiming duplicated an original frame")
    mapping = [dict(encoded_frame=k, source_frame=frames[i], batch=batches[i])
               for k,i in enumerate(picks)]
    encoded = []
    for chapter in chapters:
        members = [m for m in mapping if m["batch"] == chapter["batch"]]
        require(members, "A scheduled trial disappeared")
        encoded.append(dict(chapter, start_frame=members[0]["encoded_frame"],
                            frames=len(members), phase=phase(chapter["batch"])))
    return mapping, encoded, len(frames)/target


def overlay_state(curve, assessments, batch, reveal_current=False):
    """Before outcome, reveal only completed prior batches, never future scores."""
    require(len(curve) == 128 and 0 <= batch < 128, "Incomplete learning curve")
    completed = batch + int(bool(reveal_current))
    previous = curve[batch-1] if batch else None
    visible_curve = curve[:completed]
    known = [a for a in assessments if a["after_train_batches"] <= completed]
    assessment = max(known, key=lambda a:a["after_train_batches"]) if known else None
    return dict(batch=batch, sampling_update=batch//2,
                previous=previous, visible_curve=visible_curve,
                current=curve[batch] if reveal_current else None,
                assessment=assessment)


def verify_trial(run, trial, spec, origins, menus):
    require(trial.get("schema") in (TRIAL_SCHEMA, CLIP_SCHEMA), "Unknown RL trial schema")
    require(trial["source_fingerprint"] == spec["fingerprint"]
            and trial["setup_protocol"] == SETUP_PROTOCOL,
            "Scene or canonical episode setup changed")
    require(trial["policy_version"] == POLICY_VERSION, "Unsupported categorical policy")
    indices = np.asarray(trial["selected_indices"])
    require(indices.shape == (35,) and np.issubdtype(indices.dtype, np.integer)
            and np.all((indices >= 0) & (indices < 24)), "Invalid executed category")
    finite(trial["old_logp"], (35,), "Missing categorical likelihoods")
    finite(trial["entropy"], (35,), "Missing action entropy")
    finite(trial["observations"], (35,5), "Missing scene observations")
    finite(trial["candidate_features"], (35,24,10), "Missing qualified menu features")
    rewards = finite(trial["rewards"], (35,), "Incomplete RL rewards")
    records, actions = trial["rollout"], trial["actions"]
    require(len(records) == len(actions) == 35 and trial["release_enabled"] == [True]*35,
            "All 35 distinct sites must remain enabled")
    for index,(cell,row,action,choice,value) in enumerate(zip(spec["cells"],records,actions,indices,rewards)):
        validate_action(cell, action)
        require(row["cell_id"] == cell["id"] and matches_action(action, menus[index][choice])
                and matches_action(row, action), "Executed action differs from sampled category")
        verify_action_geometry(row, action, cell, expanded=True)
        require(np.isclose(value, reward(row), atol=1e-6, rtol=0), "RL reward differs from scored outcome")
        if not row["invalid"]:
            require(accepted(row["settled"]) and row["settled"]["dropper_position_error_mm"] <= 2,
                    "Original endpoint admission failed")
        if row["route_success"]:
            require(not row["invalid"] and row["board_bounce"] and row["whole_ball_entry"]
                    and row["max_retention_s"] >= 1-1e-8 and not row["external_contact_steps"],
                    "Unqualified capture")
    checkpoint = safe_file(run, trial["checkpoint_file"], ".pt")
    require(sha(checkpoint) == trial["checkpoint_sha256"], "Sampling checkpoint bytes changed")
    physics, audits = verify_physics(run, trial, records, spec["scene"], origins, spec["cells"])
    return records, physics, audits


def verify_checkpoint_audit(run, report, study, trial_hashes):
    path = safe_file(run, "checkpoint-audit.json", ".json")
    audit = read(path)
    require(audit.get("schema") == AUDIT_SCHEMA and audit.get("status") == "passed",
            "Passing independent categorical checkpoint audit required")
    require(audit["source_result_sha256"] == sha(run/"result.json")
            and audit["source_study_sha256"] == sha(run/"study.json")
            and audit["source_contract_sha256"] == sha(run/"contract.json")
            and audit["menu_sha256"] == sha(run/"expanded-menu-qualified.json")
            and audit["study_fingerprint"] == study["fingerprint"]
            and audit["source_trial_sha256"] == trial_hashes,
            "Checkpoint audit is stale or belongs to another run")
    require(audit["source_fingerprint"] == report["source_fingerprint"]
            and audit["training_batches_checked"] == 128
            and audit["assessment_batches_checked"] == 32
            and audit["updates_checked"] == 64
            and audit["final_policy_hash"] == report["final_policy_hash"]
            and len(audit["checkpoint_sha256"]) == 133,
            "Checkpoint audit does not cover the complete RL study")
    for filename,expected in audit["checkpoint_sha256"].items():
        require(sha(safe_file(run,filename,".pt")) == expected,
                "Checkpoint changed after its independent audit")
    return dict(file=path.name, sha256=sha(path), audit=audit)


def verify(run, source):
    from rl_film_study import make_study

    run, source = Path(run), Path(source)
    report, spec, study = (read(run/name) for name in ("result.json","contract.json","study.json"))
    signed(spec,"Geometry"); signed(study,"RL study")
    require(report.get("schema") == RESULT_SCHEMA and report.get("status") == "completed",
            "A complete genuine RL study is required")
    bundle = load_source(source, run/"selected-camera.json")
    require(study == make_study(bundle) and spec == bundle["spec"]
            and sha(run/"contract.json") == sha(source/"contract.json")
            and report["source_fingerprint"] == spec["fingerprint"]
            and report["study_fingerprint"] == study["fingerprint"]
            and report["source"] == bundle["source"], "Frozen scene or RL protocol changed")
    require(len(spec["cells"]) == report["workcells"] == 35 and report["articulations"] == 70
            and len({tuple(c["bucket_xy_m"]) for c in spec["cells"]}) == 35,
            "The 35 distinct bucket positions changed")
    origins = finite(report["origins_m"], (35,3), "Missing actual cell origins")
    require(np.array_equal(origins,preview_origins()), "Chosen-camera cell permutation changed")
    camera,binding = verify_camera_binding(run,spec,origins)
    require(report["camera"] == camera and report["camera_binding"] == binding
            and report["selected_camera_sha256"] == sha(run/"selected-camera.json")
            and report["camera_binding_sha256"] == sha(run/"camera-binding.json")
            and report["resolution"] == [1920,1080] and report["fps"] == FPS
            and report["physics_dt_s"] == DT, "Native camera/cadence changed")
    stage = verify_stage(report["stage"], bundle["report"]["stage"])
    verify_probes(run,35)
    menus,menu_audit = load_native_menu(run/"expanded-menu-qualified.json",spec)
    require(len(menus) == 35 and all(len(m) == 24 for m in menus)
            and sha(run/"expanded-menu-qualified.json") == bundle["source"]["menu_sha256"],
            "RL must use all original qualified menus")
    summaries = report["training_batches"]
    require(len(summaries) == 128 and len(report["updates"]) == 64
            and report["filmed_batches"] == list(FILMED) and len(report["clips"]) == 24,
            "Incomplete RL budget or predeclared film schedule")
    rows, trials, hashes, native = [], {}, {}, []
    for batch,summary in enumerate(summaries):
        name = f"train-batch-{batch:03}.json"; path=safe_file(run,name,".json"); trial=read(path)
        require(summary["file"] == name and summary["batch"] == trial["batch"] == batch
                and trial["mode"] == "train" and trial["schema"] == TRIAL_SCHEMA
                and trial["study_fingerprint"] == study["fingerprint"]
                and trial["update_count_before"] == batch//2,
                "Training batch, study or sampling policy reordered")
        require(trial["setup_visualization"] == ("driven" if batch in FILMED else "not-recorded"),
                "Episode setup provenance changed")
        records,physics,_ = verify_trial(run,trial,spec,origins,menus)
        close_record(summary,statistics(records),"Training curve counts changed")
        rows.append(records);trials[batch]=trial;hashes[name]=sha(path);native.append(physics)
    curve = rates(rows)
    assessments=[]
    require(len(report["assessments"]) == 4, "Missing prespecified no-update checks")
    for stage,summary in zip(ASSESSMENT_STAGES,report["assessments"]):
        require(summary["after_train_batches"] == stage, "Assessment schedule changed")
        assessment_rows=[]; checkpoints=set()
        for rep in range(8):
            name=f"assessment-{stage:03}-{rep:02}.json"; path=safe_file(run,name,".json"); trial=read(path)
            require(trial["mode"] == "assessment" and trial["schema"] == TRIAL_SCHEMA
                    and trial["after_train_batches"] == stage and trial["repeat"] == rep
                    and trial["study_fingerprint"] == study["fingerprint"]
                    and trial["update_count_before"] == stage//2
                    and trial["setup_visualization"] == "not-recorded",
                    "Assessment identity or no-update stage changed")
            records,physics,_=verify_trial(run,trial,spec,origins,menus)
            assessment_rows.extend(records);hashes[name]=sha(path);native.append(physics)
            checkpoints.add((trial["policy_hash"],trial["checkpoint_sha256"]))
        require(len(checkpoints) == 1, "Assessment changed its frozen policy")
        counts=statistics(assessment_rows);close_record(summary,counts,"Assessment counts changed")
        assessments.append(dict(after_train_batches=stage, policy_update=stage//2,
            attempts=280,captures=counts["captures"],invalid=counts["invalid"],
            capture_rate=counts["captures"]/280, wilson95=wilson(int(counts["captures"]),280),
            policy_hash=next(iter(checkpoints))[0],
            scope="No-update stochastic assessment on the same 35 known bucket layouts."))
    checkpoint_audit=verify_checkpoint_audit(run,report,study,hashes)
    recorded=report["source_sha256"]
    require(all(recorded.get(name)==value
                for name,value in bundle["source"]["current_original_dependencies_sha256"].items()),
            "Recorded physics, control or camera dependencies changed")
    require(all(recorded.get(name)==sha(Path(__file__).parent/name)
                for name in ("rl_film_study.py","rl_film_policy.py","film_cylinder_farm.py")),
            "Native study or policy source differs from the frozen recording")
    chapters,total,clip_hashes=[],0,{}
    identity=("mode","batch","policy_version","update_count_before","policy_hash",
              "checkpoint_file","checkpoint_sha256","selected_indices","old_logp",
              "entropy","observations","candidate_features","actions","rewards","rollout",
              "physics_file","physics_sha256","release_step","score_end_step",
              "capture_end_step","focus_telemetry","frames","setup_protocol","setup_visualization")
    for cycle,(batch,summary) in enumerate(zip(FILMED,report["clips"])):
        name=f"clip-{cycle:02}.json"; path=safe_file(run,name,".json");clip=read(path);trial=trials[batch]
        require(clip["schema"] == CLIP_SCHEMA and clip["cycle"] == cycle
                and clip["batch"] == summary["batch"] == batch and summary["file"] == name
                and clip["study_fingerprint"] == study["fingerprint"]
                and all(clip[key] == trial[key] for key in identity),
                "Filmed native collection differs from the actual optimizer batch")
        move,release,motion=check_frames(clip,camera,total)
        require(summary["start_frame"] == total and summary["frames"] == len(clip["frames"]),
                "Filmed source frame counts changed")
        record=clip["rollout"][FOCUS]
        audit=score_focus(clip,record,spec["scene"],origins[FOCUS])
        cut=terminal_cut(clip,record,audit,origins[FOCUS])
        background=background_rebounds(clip,clip["rollout"])
        chapters.append(dict(cycle=cycle,batch=batch,focus=spec["cells"][FOCUS]["id"],
            sampling_update=clip["update_count_before"],policy_hash=clip["policy_hash"],
            foreground_action=clip["actions"][FOCUS],motion_excursion_rad=motion,
            batch_counts=statistics(clip["rollout"]),background_projected_board_contacts=len(background),
            first_board_contact_step=audit["first_board_contact_step"],**cut))
        total+=len(clip["frames"]);clip_hashes[name]=sha(path)
    require(report["frames"] == total and [p.name for p in sorted((run/"frames").glob("*.jpg"))]
            == [f"{i:05d}.jpg" for i in range(total)],"Native RGB sequence incomplete")
    require(all(any(c["motion_excursion_rad"][arm]>.01 for c in chapters)
                for arm in ("Arm10","Arm14")),"No measured foreground setup motion")
    mapping,encoded,speed=frame_map(chapters)
    counts=statistics([r for group in rows for r in group])
    return dict(schema=SCHEMA,status="verified",scope=SCOPE,
        study_fingerprint=study["fingerprint"],source_fingerprint=spec["fingerprint"],
        source_result_sha256=sha(run/"result.json"),source_study_sha256=sha(run/"study.json"),
        source_contract_sha256=sha(run/"contract.json"),source_trial_sha256=hashes,
        source_clip_sha256=clip_hashes,checkpoint_audit=checkpoint_audit,
        exporter_code_sha256=sha(__file__),source_code_sha256=report["source_sha256"],
        camera=camera,camera_binding=binding,selected_camera_sha256=sha(run/"selected-camera.json"),
        ball_radius_m=spec["scene"]["ball"]["radius_m"],
        stage=stage,workcells=35,articulations=70,geometry_menu=menu_audit,
        actual_training_counts=counts,learning_curve=curve,known_layout_assessments=assessments,
        native_physics_audits=native,raw_frames=total,filmed_batches=list(FILMED),
        frames=len(mapping),duration_s=len(mapping)/FPS,fps=FPS,resolution=[1920,1080],
        playback_multiple=speed,chapters=encoded,frame_map=mapping,
        setup_protocol=SETUP_PROTOCOL,
        setup_timing="Driven setup is a compressed visualization followed by the common canonical endpoint initialization used by all scored episodes.",
        learning_scope="Choice among24qualified local configurations per known layout; global pose-space coverage is not established.",
        overlay_timing="Prior completed training data during setup/flight; current batch result only during final0.25s. Assessments separately labeled known-layout no-update checks.")


def make_overlay(summary,batch,reveal_current):
    """Small data-derived overlays; source camera remains fixed underneath."""
    from PIL import Image,ImageDraw,ImageFont

    state=overlay_state(summary["learning_curve"],summary["known_layout_assessments"],batch,reveal_current)
    overlay=Image.new("RGBA",(1920,1080),(0,0,0,0));draw=ImageDraw.Draw(overlay)
    font=caption_font(); require(font is not None,"An installed DejaVu/Arial font is required")
    fonts={size:ImageFont.truetype(str(font),size)for size in(14,16,19,22,24)}
    draw.rounded_rectangle((20,16,790,153),radius=12,fill=(24,31,40,218))
    draw.text((34,25),"SIRCUS · Actual PPO training",font=fonts[24],fill=(238,244,248,255))
    draw.text((34,60),f"{phase(batch)} · batch {batch+1}/128 · sampling update {state['sampling_update']}",font=fonts[19],fill=(205,220,234,255))
    previous=state["previous"]
    text=(f"Previous {previous['rolling_window_batches']} completed batches: "
          f"{previous['rolling_captures']}/{previous['rolling_attempts']} captures "
          f"({100*previous['rolling_capture_rate']:.1f}%)")if previous else"No completed training batches yet"
    draw.text((34,88),text,font=fonts[16],fill=(222,231,239,255))
    chapter=next(c for c in summary["chapters"]if c["batch"]==batch);a=chapter["foreground_action"]
    release_mm=1000*(a["height_m"]+a["gap_m"]+summary["ball_radius_m"])
    draw.text((34,115),f"Front targets: board {1000*a['height_m']:.0f} mm · ball release {release_mm:.0f} mm · tilt {a['tilt_deg']:.1f}°",font=fonts[16],fill=(187,207,223,255))
    draw.rounded_rectangle((20,824,485,1060),radius=12,fill=(24,31,40,224))
    draw.text((34,834),"Whole-farm capture rate · all attempts",font=fonts[16],fill=(230,239,247,255))
    x0,y0,x1,y1=69,876,465,996
    for val in(0,.5,1):
        y=y1-val*(y1-y0);draw.line((x0,y,x1,y),fill=(83,99,116,150),width=1)
        draw.text((28,y-8),f"{val*100:.0f}%",font=fonts[14],fill=(161,179,197,255))
    for count in(0,64,128):
        x=x0+(x1-x0)*count/128;draw.text((x-8,y1+4),str(count),font=fonts[14],fill=(161,179,197,255))
    raw=[];rolling=[]
    for p in state["visible_curve"]:
        x=x0+(x1-x0)*p["completed_train_batches"]/128
        raw.append((x,y1-p["batch_capture_rate"]*(y1-y0)))
        rolling.append((x,y1-p["rolling_capture_rate"]*(y1-y0)))
    if len(raw)>1:draw.line(raw,fill=(118,139,157,130),width=1)
    if len(rolling)>1:draw.line(rolling,fill=(85,211,238,255),width=3)
    if rolling:draw.ellipse((rolling[-1][0]-3,rolling[-1][1]-3,rolling[-1][0]+3,rolling[-1][1]+3),fill=(85,211,238,255))
    draw.text((220,1020),"completed training batches",font=fonts[14],fill=(170,189,205,255))
    assessment=state["assessment"]
    if assessment:
        draw.text((34,1040),f"Known-layout check U{assessment['policy_update']}: {assessment['captures']}/280",font=fonts[14],fill=(203,217,231,255))
    if reveal_current:
        c=state["current"]
        draw.rounded_rectangle((1090,20,1900,72),radius=10,fill=(24,31,40,225))
        draw.text((1107,32),f"Completed batch result: {c['captures']}/35 captures · {c['invalid']} rejected",font=fonts[22],fill=(139,230,208,255))
    return overlay


def learning_plot_data(summary):
    """Recompute plotted rates from verified integer counts, including rejects."""
    require(summary.get("status") == "verified", "Plot requires verified study data")
    curve = summary["learning_curve"]
    assessments = summary["known_layout_assessments"]
    require(len(curve) == 128 and len(assessments) == 4, "Incomplete learning plot data")
    raw, rolling = [], []
    for batch, point in enumerate(curve):
        require(point["batch"] == batch and point["completed_train_batches"] == batch+1
                and point["attempts"] == 35 and type(point["captures"]) is int
                and 0 <= point["captures"] <= 35,
                "Learning plot training counts changed")
        rate = point["captures"]/35
        recent = curve[max(0, batch-7):batch+1]
        smoothed = sum(p["captures"] for p in recent)/(35*len(recent))
        require(np.isclose(point["batch_capture_rate"], rate, atol=1e-12, rtol=0)
                and np.isclose(point["rolling_capture_rate"], smoothed, atol=1e-12, rtol=0),
                "Learning plot rates differ from attempted-shot counts")
        raw.append((batch+1, rate))
        rolling.append((batch+1, smoothed))
    checks = []
    for stage, point in zip(ASSESSMENT_STAGES, assessments):
        require(point["after_train_batches"] == stage and point["attempts"] == 280
                and type(point["captures"]) is int and 0 <= point["captures"] <= 280,
                "Learning plot assessment counts changed")
        rate = point["captures"]/280
        require(np.isclose(point["capture_rate"], rate, atol=1e-12, rtol=0),
                "Learning plot assessment rate differs from counts")
        checks.append(dict(completed_train_batches=stage, capture_rate=rate,
                           captures=point["captures"], attempts=280,
                           wilson95=wilson(point["captures"], 280)))
    return dict(training=raw, rolling8=rolling, known_layout_checks=checks)


def render_learning_curve(summary, output):
    """Save a standalone plot with final results readable beyond the video tail."""
    from PIL import Image, ImageDraw, ImageFont

    output = Path(output)
    require(not output.exists(), "Learning plot output must be new")
    data = learning_plot_data(summary)
    font = caption_font()
    require(font is not None, "An installed DejaVu/Arial font is required")
    fonts = {size: ImageFont.truetype(str(font), size) for size in (19, 21, 24, 34)}
    image = Image.new("RGB", (1600, 1000), (24, 31, 40))
    draw = ImageDraw.Draw(image)
    draw.text((55, 30), "SIRCUS · Actual PPO learning results", font=fonts[34], fill=(238, 244, 248))
    draw.text((57, 82), "35 known bucket layouts · 4,480 training attempts · 64 PPO updates",
              font=fonts[21], fill=(186, 207, 224))
    draw.line((64, 131, 109, 131), fill=(134, 153, 171), width=2)
    draw.text((120, 118), "Training batch (35 attempts)", font=fonts[19], fill=(195, 210, 224))
    draw.line((505, 131, 550, 131), fill=(85, 211, 238), width=4)
    draw.text((562, 118), "Previous 8 completed batches", font=fonts[19], fill=(195, 210, 224))
    draw.ellipse((1009, 125, 1021, 137), fill=(245, 199, 95))
    draw.text((1034, 118), "Frozen-policy known-layout check", font=fonts[19], fill=(195, 210, 224))
    x0, y0, x1, y1 = 110, 218, 1510, 805

    def position(batch, rate):
        return x0+(x1-x0)*batch/128, y1-(y1-y0)*rate

    draw.text((55, 172), "Capture rate (% of all attempted shots)", font=fonts[21], fill=(213, 225, 236))
    for percent in (0, 25, 50, 75, 100):
        _, y = position(0, percent/100)
        draw.line((x0, y, x1, y), fill=(63, 79, 96), width=1)
        draw.text((49, y-12), f"{percent}%", font=fonts[21], fill=(170, 190, 208))
    for batch in (0, 32, 64, 96, 128):
        x, _ = position(batch, 0)
        draw.line((x, y0, x, y1), fill=(44, 60, 77), width=1)
        label = str(batch)
        box = draw.textbbox((0, 0), label, font=fonts[21])
        draw.text((x-(box[2]-box[0])/2, y1+12), label, font=fonts[21], fill=(170, 190, 208))
    draw.line([position(x, y) for x, y in data["training"]], fill=(134, 153, 171), width=2)
    draw.line([position(x, y) for x, y in data["rolling8"]], fill=(85, 211, 238), width=4)
    for point in data["known_layout_checks"]:
        batch, rate = point["completed_train_batches"], point["capture_rate"]
        x, y = position(batch, rate)
        low, high = point["wilson95"]
        _, low_y = position(batch, low)
        _, high_y = position(batch, high)
        draw.line((x, low_y, x, high_y), fill=(245, 199, 95), width=3)
        draw.line((x-8, low_y, x+8, low_y), fill=(245, 199, 95), width=3)
        draw.line((x-8, high_y, x+8, high_y), fill=(245, 199, 95), width=3)
        draw.ellipse((x-6, y-6, x+6, y+6), fill=(245, 199, 95))
        label = f"{100*rate:.1f}% · {point['captures']}/280"
        box = draw.textbbox((0, 0), label, font=fonts[19])
        label_width = box[2]-box[0]
        label_x = max(x0+8, min(x-label_width/2, x1-label_width-8))
        label_y = max(y0+5, high_y-31)
        draw.rectangle((label_x-5, label_y-2, label_x+label_width+5, label_y+25), fill=(24, 31, 40))
        draw.text((label_x, label_y), label, font=fonts[19], fill=(250, 214, 133))
    draw.text((614, 852), "Completed training batches", font=fonts[24], fill=(211, 225, 236))
    draw.text((56, 905), "Checkpoint checks: 280 independently sampled attempts each; error bars are descriptive 95% Wilson intervals.",
              font=fonts[19], fill=(180, 201, 219))
    draw.text((56, 938), "Same 35 known layouts; no held-out-layout claim. Rejected releases remain in every attempt denominator.",
              font=fonts[19], fill=(180, 201, 219))
    image.save(output)
    return dict(file=output.name, sha256=sha(output), resolution=[1600, 1000],
                training_curve_points=128, known_layout_assessment_points=4,
                x_axis="Completed training batches (35 attempts each)",
                y_axis="Capture rate among all attempted shots", rolling_window_batches=8,
                assessment_interval="Descriptive pooled 95% Wilson interval; same 35 known layouts",
                data_source="Verified learning_curve and known_layout_assessments integer counts")


def render(run,out,summary):
    from PIL import Image

    count=summary["frames"]; duration=count/FPS
    metadata=[";FFMETADATA1"]
    for c in summary["chapters"]:
        metadata.extend(("[CHAPTER]","TIMEBASE=1/30",f"START={c['start_frame']}",
            f"END={c['start_frame']+c['frames']}",f"title={c['phase']} — batch {c['batch']+1}"))
    metadata_path=out/"film.ffmeta";metadata_path.write_text("\n".join(metadata)+"\n")
    with tempfile.TemporaryDirectory(prefix="sircus-rl-film-")as temp:
        rendered=Path(temp); overlays={}
        for chapter in summary["chapters"]:
            overlays[(chapter["batch"],False)]=make_overlay(summary,chapter["batch"],False)
            overlays[(chapter["batch"],True)]=make_overlay(summary,chapter["batch"],True)
        chapters={c["batch"]:c for c in summary["chapters"]}
        for item in summary["frame_map"]:
            source=run/"frames"/f"{item['source_frame']:05d}.jpg"
            require(source.is_file() and not source.is_symlink(),"Missing original native camera image")
            item["source_frame_sha256"]=sha(source)
            chapter=chapters[item["batch"]]
            reveal=item["encoded_frame"]>=chapter["start_frame"]+chapter["frames"]-math.ceil(.25*FPS)
            item["overlay_current_batch_result"]=reveal
            original=Image.open(source).convert("RGBA")
            require(original.size==(1920,1080),"Camera source resolution changed")
            composed=Image.alpha_composite(original,overlays[(item["batch"],reveal)]).convert("RGB")
            composed.save(rendered/f"{item['encoded_frame']:05d}.png",compress_level=1)
        movie=out/"rl-training.mp4"
        subprocess.run(["ffmpeg","-hide_banner","-loglevel","error","-n","-framerate","30",
            "-i",str(rendered/"%05d.png"),"-i",str(metadata_path),"-map","0:v:0","-map_metadata","1",
            "-map_chapters","1","-an","-c:v","libx264","-pix_fmt","yuv420p","-threads","4",
            "-preset","medium","-crf","17","-frames:v",str(count),"-movflags","+faststart",str(movie)],check=True)
        shutil.copyfile(rendered/"00000.png",out/"poster.png")
    probe=json.loads(subprocess.check_output(["ffprobe","-v","error","-count_frames","-show_streams","-show_chapters","-show_format","-of","json",str(movie)]))
    videos=[s for s in probe["streams"]if s["codec_type"]=="video"]
    require(len(videos)==1,"Unexpected video streams")
    stream=videos[0]
    require(stream["codec_name"]=="h264" and stream["avg_frame_rate"]=="30/1"
        and int(stream["nb_read_frames"])==count and stream["width"]==1920 and stream["height"]==1080
        and abs(float(stream["duration"])-duration)<.002 and abs(float(probe["format"]["duration"])-duration)<.002
        and len(probe["chapters"])==24,"Encoded video differs from native frame map")
    for actual,expected in zip(probe["chapters"],summary["chapters"]):
        require(abs(float(actual["start_time"])-expected["start_frame"]/FPS)<.002
            and abs(float(actual["end_time"])-(expected["start_frame"]+expected["frames"])/FPS)<.002,
            "Video chapter timing changed")
    subprocess.run(["ffmpeg","-v","error","-xerror","-i",str(movie),"-map","0:v:0","-f","null","-"],check=True,stdout=subprocess.DEVNULL)
    return dict(file=movie.name,sha256=sha(movie),frames=count,duration_s=duration,
        resolution=[1920,1080],fps=30,codec="h264",decode_verified=True,
        non_simulation_content=["provenance_text","data_derived_learning_curve","data_derived_result_labels"])


def export(run,out,source):
    run,out=Path(run),Path(out)
    require(not out.exists() and not out.resolve().is_relative_to(run.resolve()),"A new export directory outside raw evidence is required")
    summary=verify(run,source);out.mkdir(parents=True)
    summary["movie"]=render(run,out,summary)
    summary["learning_curve_image"]=render_learning_curve(summary,out/"learning-curve.png")
    (out/"film.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:summary[k]for k in("status","frames","duration_s","playback_multiple","movie")}),flush=True)
    return summary


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run",type=Path);parser.add_argument("output",type=Path)
    parser.add_argument("--source",type=Path,required=True)
    args=parser.parse_args();export(args.run,args.output,args.source)
