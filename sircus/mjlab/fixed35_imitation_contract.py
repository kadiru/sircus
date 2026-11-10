"""Fixed canonical grid, historical action supervision, and explicit yaw behavior."""
from __future__ import annotations

import copy
from pathlib import Path
import numpy as np

from .benchmark_contract import RESIDUAL_SCALE, observation, action_from_residual
from .benchmark_verify import contained, digest, read, sha
from .corrected_contract import grid_layouts
from .corrected_imitation_contract import authenticate_study, require
from .table_training_contract import captured, clean, statistics
from .table_assisted_score import reward
from .table_correction_native import META
from .yaw_tilt_gap_contract import FAR, NEAR

VERSION = "sircus-fixed35-imitation-v1"
AUDIT_VERSION = "sircus-fixed35-imitation-audit-v1"
METHODS = ("fixed35_bc", "current_bc")
INPUT_KEYS = ("corrected-study", "corrected-audit", "yaw-study", "yaw-audit", "reference", "assets")
SCOPE = ("Continuous five-coordinate residual imitation with a shared geometry-anchor prior. "
         "Both frozen actors use bucket-facing release yaw on the historical fixed 7x5 grid. "
         "This is coverage/regression, not unseen-position generalization. No perturbations, "
         "PPO, flight search at inference, learned transit, hardware, or policy activation.")
ROTATION = "Rz(atan2(bucket_y-board_y,bucket_x-board_x)) @ diag(1,-1,-1)"


def protocol():
    return dict(substeps=64, methods=list(METHODS), test_layouts=35, test_requests=70,
                control_requests=18, maximum_requests=88, base_teacher_count=97,
                correction_teacher_count=2, new_teacher_count=2, teacher_count=99,
                actor_steps=800, critic_steps=0, actor_lr=.001, noise=False,
                bucket_perturbations=False, grid_source="corrected_contract.grid_layouts",
                evaluation_scope="historical fixed-grid coverage/regression; not held out",
                residual_scale=RESIDUAL_SCALE.tolist(), worker_limit_s=3600,
                queue_limit_s=900, global_limit_s=6300, audit_reserve_s=1800,
                storage_reserve_bytes=8*1024**3, rotation=ROTATION,
                release_behavior="Same bucket-facing gripper yaw for both policies; controls retain original joints",
                fitting="800 actor-only full-batch Adam steps at .001; 99 unique layouts, equal per-layout weight",
                teacher_semantics="97 historical successful action targets plus two confirmed yaw targets; historical labels do not prove new-yaw success",
                selection="Far grid22 repeat0 and near reference repeat0; each already confirmed 3/3, one teacher per context",
                test="Both checkpoints and all 70 continuous predictions frozen before first grid outcome; no retries or adaptation",
                geometry_prior="Unchanged 19 geometry anchors; prior is not a discrete policy action bank",
                control_prior="Original 17 control prior and original proof/joints", scope=SCOPE)


def fixed_grid():
    return [dict(id="fixed35-"+r["id"], canonical_id=r["id"], split="test",
                 evaluation_scope="historical_grid_coverage", bucket_xy_m=r["bucket_xy_m"],
                 grid_cell=[r["column"],r["row"]], row=r["row"], column=r["column"])
            for r in grid_layouts()]


def yaw_teacher(group, name):
    """One independently repeated setting per layout, without repeat upweighting."""
    group = sorted(group, key=lambda r:r["repeat"])
    require(len(group)==3 and [r["repeat"] for r in group]==[0,1,2]
            and all(captured(r) for r in group), "Three captured yaw repeats required")
    first = group[0]
    for r in group:
        require(all(r[k]==first[k] for k in (*META, "values", "bucket_xy_m", "q10", "q14", "connection_sha256")),
                "Yaw teacher repeats changed action, geometry, or observation")
        require(r["adapter_directory"]=="native-yaw" and not r.get("invalid_physics")
                and not r.get("diagnostic_robot_contact_steps"), "Confirmed valid bucket-facing recipe required")
    obs = observation(first["bucket_xy_m"], first["effective_source_hint"]).tolist()
    require(obs==first["observation"], "Yaw source observation differs")
    residual = (np.asarray(first["values"])-np.asarray(first["effective_source_hint"]["values"]))/RESIDUAL_SCALE
    require(np.isfinite(residual).all() and (np.abs(residual)<1).all(), "Teacher outside actor support; do not clip")
    require(np.allclose(action_from_residual(first["effective_source_hint"]["values"],residual),first["values"],atol=1e-12,rtol=0),
            "Yaw teacher reconstruction differs")
    teacher = copy.deepcopy(first)
    teacher.update(id="fixed35-yaw-teacher-"+name, role="train", original_role=first["role"],
                   source_correction_role=first["role"], development_reclassified=True,
                   normalized_residual=residual.tolist(), observation=obs, learning_reward=reward(first),
                   source_correction_record_id=first["id"], source_correction_record_sha256=digest(first),
                   correction_repeat_ids=[r["id"] for r in group],
                   correction_repeat_sha256={r["id"]:digest(r) for r in group},
                   supervision_provenance="confirmed bucket-facing native64 recipe")
    return teacher


def load_inputs(a, root):
    root = Path(root)
    require(read(root/"sircus/mjlab/fixed35-imitation-rule.json")==protocol(), "Frozen fixed35 protocol differs")
    public = read(root/"baselines/mjlab-corrected-imitation-20261109/summary.json")
    corrected = authenticate_study(a.corrected_study, a.corrected_audit,
        dict(report_sha256=public["report_sha256"],proof_sha256=public["audit"]["proof_sha256"]),checks_count=55633)
    public_yaw = read(root/"baselines/mjlab-yaw-tilt-gap-20261109/summary.json")
    yaw = authenticate_study(a.yaw_study, a.yaw_audit,
        dict(report_sha256=public_yaw["report_sha256"],proof_sha256=public_yaw["audit"]["proof_sha256"]),checks_count=50615)
    require(len(corrected["rows"])==158 and len(yaw["rows"])==106, "Complete input studies required")
    for key in ("scene_sha256", "asset_sha256"):
        require(corrected["spec"]["identity"][key]==yaw["spec"]["identity"][key], "Shared scene identity differs")
    out = {}
    for key,name in (("prior","prior.json"),("old_prior","old-prior.json"),("controls","controls.json"),
                     ("control_replays","control-replays.json"),("calibration","calibration.json"),
                     ("home","setup-home.json"),("teachers","bootstrap/teachers.json")):
        out[key] = read(corrected["bound"](name))
    corrected["bound"]("verification.json")
    require(len(out["teachers"])==97 and len({r["context_id"] for r in out["teachers"]})==97
            and all(r["role"]=="train" and captured(r) for r in out["teachers"]), "Original 97 successful historical action targets required")
    require(len(out["prior"]["records"])==19 and len(out["old_prior"]["records"])==17, "Unchanged 19/17 geometry priors required")
    artifact = copy.deepcopy(corrected["report"]["artifacts"]["current_bc"])
    artifact["source_path"] = str(corrected["bound"](artifact["path"]))
    require(sha(artifact["source_path"])==artifact["sha256"] and artifact["samples"]==92, "Preserved current 40/70 actor required")
    groups = {
        "far": [r for r in yaw["rows"] if r.get("phase")=="selected_repeats" and r.get("grid_index")==22],
        "near": [r for r in yaw["rows"] if r.get("phase")=="references" and r.get("variant")=="near_success_reference"],
    }
    require(all(r["context_id"]==FAR for r in groups["far"]) and all(r["context_id"]==NEAR for r in groups["near"]), "Exact near/far corrections required")
    additions = [yaw_teacher(groups[name],name) for name in ("far","near")]
    for group in groups.values():
        for r in group:
            for key,hkey in (("context_resolution_file","context_resolution_sha256"),("connection_file","connection_sha256"),("native_cache_file","native_cache_sha256")):
                require(sha(yaw["bound"]("native-yaw/"+r[key]))==r[hkey], "Yaw teacher evidence differs")
    for r in out["control_replays"]:
        for key,hkey in (("context_resolution_file","context_resolution_sha256"),("connection_file","connection_sha256")):
            require(sha(corrected["bound"]("control-parent/native/"+r[key]))==r[hkey], "Control proof changed")
    for name in ("control-parent/native/effective-options.json", "control-parent/native/native-compiled-geometry.json",
                 "native/effective-options.json", "native/native-compiled-geometry.json"):
        corrected["bound"](name)
    all_teachers = sorted(copy.deepcopy(out["teachers"])+additions,key=lambda r:r["context_id"])
    require(len(all_teachers)==99 and len({r["context_id"] for r in all_teachers})==99 and len({tuple(r["bucket_xy_m"]) for r in all_teachers})==99, "One teacher per 99 unique layouts required")
    lineage = [dict(context_id=r["context_id"],teacher_id=r["id"],teacher_sha256=digest(r),
                    correction_record_id=r["source_correction_record_id"],correction_record_sha256=r["source_correction_record_sha256"],
                    confirmation_record_ids=r["correction_repeat_ids"],confirmation_record_sha256=r["correction_repeat_sha256"])
               for r in additions]
    out.update(parent=corrected["spec"],parent_report=corrected["report"],rows=corrected["rows"],
               artifacts={"current_bc":artifact}, corrected_teachers=additions, all_teachers=all_teachers,
               teacher_lineage=lineage, sources=[],
               bindings=dict(corrected=corrected["bindings"], yaw=yaw["bindings"], teacher_lineage=lineage,
                             old_teacher_file_sha256=sha(corrected["bound"]("bootstrap/teachers.json")),
                             old_teacher_sha256={r["id"]:digest(r) for r in out["teachers"]},
                             teacher_semantics=protocol()["teacher_semantics"],
                             preserved_yaw_recipes=[dict(id=r["id"],sha256=digest(r),values=r["values"],bucket_xy_m=r["bucket_xy_m"])
                                                    for r in yaw["rows"] if r.get("phase")=="selected_repeats" and r.get("repeat")==0]))
    return out
