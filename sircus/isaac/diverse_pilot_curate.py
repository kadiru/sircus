"""Curate an already completed, audited diverse pilot without running a model.

Only Python's standard library and the read-only report helpers are imported.
No simulator, Torch, SSH, subprocess, GPU, service or hardware action exists.
Raw native observations, pictures, clips and logs are read/hash-bound as needed,
but never copied into the output. Full signed support is retained unchanged:
its anchors, warm-start joints and path proxies are offline geometry, not native
observations. This is explicitly authorized study data, not host calibration.
"""

import argparse
import collections
import copy
import csv
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil
import statistics
import tempfile

from diverse_pilot_report import (SCHEMA as COMPARISON_SCHEMA, STUDY_SCHEMA,
    BASELINE_RESULT_SHA256, GUIDED_RESULT_SHA256, canonical_digest, compact_support,
    setup_check_receipt, verify_comparison_audit)

SCHEMA = "sircus-diverse-pilot-curation-v1"
FINAL_PRODUCER = "40dd24ffdfeec0cd521fdfbf409c31c68c69a2d7"
FINAL_ARCHIVE = "7f2b1fa1daf0da75a26445be940be1c46b2898939eddd25bc89b746f4706f893"
CAMERA_SHA256 = "a7967c072b8f949318b43a510235d503c81c21e05f433670a0ce92a7af385ede"
SHA_RE = re.compile(r"[0-9a-f]{64}\Z")
COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")
MAX_JSON = 256*1024**2
MAX_MODEL = 8*1024**2
SENSITIVE = re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----|\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b|\bAKIA[0-9A-Z]{16}\b|\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{24,}\b|\bBearer\s+[A-Za-z0-9._~-]{12,}",re.I)
SECRET_KEYS = {"password","passwd","api_key","apikey","api_token","access_token","refresh_token",
               "client_secret","private_key","authorization","cookie","cookies","credentials"}
RAW_FIELDS = {"records","paired","physics_samples","trajectory_m","all_joints_rad","all_ball_world_m",
              "joint_position_rad","joint_velocity_rad_s","all_articulation_joint_velocity",
              "board_position_m","board_quaternion_wxyz","position_m","velocity_m_s","forces_N",
              "native_frame_sha256","episode_home","retained_native_global_frames"}
CSV_FIELDS = ("cell_id","split","study_status","bucket_x_m","bucket_y_m","map_requests","map_admitted",
    "initial_update","initial_attempts","initial_captures","initial_invalid","latest_update","latest_attempts",
    "latest_captures","latest_invalid","latest_admitted","latest_invalid_reason","latest_outcome",
    "requested_board_x_m","requested_board_y_m","requested_board_height_m","requested_gap_m",
    "requested_ball_center_release_height_m","requested_radius_m","requested_tilt_deg","requested_bearing_deg",
    "requested_bearing_offset_deg","board_position_error_mm","board_normal_error_deg","prediction_trial","release_provenance")


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    return h.hexdigest()


def safe_file(root,name,limit=MAX_JSON):
    root=Path(root).resolve()
    require(isinstance(name,str) and name and "\\" not in name,"Invalid evidence filename")
    rel=PurePosixPath(name)
    require(not rel.is_absolute() and str(rel)==name and not set(rel.parts)&{".",".."},"Unsafe evidence path")
    p=root.joinpath(*rel.parts)
    require(p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(root)
            and all(not parent.is_symlink() for parent in p.parents if parent!=root and parent.is_relative_to(root)),
            "Missing or linked evidence file: "+name)
    require(0<p.stat().st_size<=limit,"Evidence file exceeds bounded size: "+name)
    return p


def read(path):
    def unique(pairs):
        d={}
        for k,v in pairs:
            require(k not in d,"Duplicate JSON key")
            d[k]=v
        return d
    return json.loads(Path(path).read_text(),object_pairs_hook=unique,
        parse_constant=lambda _:(_ for _ in ()).throw(ValueError("Nonfinite JSON")))


def inspect(value):
    if isinstance(value,dict):
        for k,v in value.items():
            require(str(k).lower() not in SECRET_KEYS,"Sensitive configuration field")
            inspect(v)
    elif isinstance(value,list):
        for v in value:inspect(v)
    elif isinstance(value,str):
        require(not SENSITIVE.search(value.encode()),"Credential pattern in curated metadata")


def signed(value,label):
    unsigned=dict(value);fingerprint=unsigned.pop("fingerprint",None)
    require(isinstance(fingerprint,str) and SHA_RE.fullmatch(fingerprint)
            and canonical_digest(unsigned)==fingerprint,label+" fingerprint mismatch")
    return fingerprint


def compact(value):
    if isinstance(value,list):return [compact(v) for v in value]
    if isinstance(value,dict):
        return {k:dict(omitted_raw_field=True,entries=len(v),canonical_json_sha256=canonical_digest(v))
                if k in RAW_FIELDS and isinstance(v,(dict,list)) else compact(v) for k,v in value.items()}
    return value


def identity(run,producer,archive):
    launch_path=Path(str(run)+".launch.json")
    status_path=Path(str(run)+".supervisor.json")
    require(launch_path.is_file() and status_path.is_file(),"Completed launch/exit receipts required")
    launch,status=read(launch_path),read(status_path)
    require(launch.get("source_commit")==status.get("source_commit")==producer,
            "Native producer identity differs from the explicitly selected source")
    require(status.get("status")=="completed" and status.get("returncode")==0,
            "Native process did not complete successfully")
    require(launch.get("integrity",{}).get("archive_sha256")==archive,"Native source archive binding changed")
    return dict(source_commit=producer,source_archive_sha256=archive,launch_receipt_sha256=sha(launch_path),
        exit_receipt_sha256=sha(status_path),source_members_sha256=copy.deepcopy(launch["integrity"].get("source_members_sha256",{})))


def preflight_evidence(setup_run,audit_path,expected_audit,study,producer,archive):
    """Require the directly measured final producer, never relabel an old check."""
    setup_run=Path(setup_run)
    raw=safe_file(setup_run,"setup-check.json")
    raw_history=safe_file(setup_run,"setup-check-raw.json")
    value,audit=read(raw),read(audit_path)
    require(sha(audit_path)==expected_audit and audit.get("status")=="passed"
            and audit.get("schema")=="sircus-fixed-request-setup-audit-v1","Preflight independent audit mismatch")
    require(audit.get("source_setup_sha256")==sha(raw) and audit.get("source_raw_sha256")==sha(raw_history),
            "Preflight audit belongs to different setup bytes")
    require(value.get("schema")=="sircus-fixed-request-setup-check-v1" and value.get("status")=="completed"
            and value.get("geometry_fingerprint")==study["geometry_fingerprint"]
            and value.get("settling_protocol")==study["settling_protocol"],"Preflight geometry/controller differs")
    for key in ("attempts","before_admitted","after_admitted","before_failures","after_failures","repaired","regressions","ball_releases"):
        require(value.get(key)==audit.get(key),"Preflight paired audit count mismatch")
    require(value["attempts"]==100 and value["ball_releases"]==0,"Require100 fixed setup-only requests, no task releases")
    provenance=identity(setup_run,producer,archive)
    summary={k:copy.deepcopy(v) for k,v in value.items() if k!="paired"}
    pairs=value.get("paired",[])
    require(len(pairs)==100,"Incomplete setup paired summary")
    metrics={}
    for key in ("position_error_mm","position_drift_mm","joint_speed_rad_s"):
        numbers=[row[key] for row in pairs]
        require(all(type(v) in (int,float) and math.isfinite(v) for v in numbers),"Invalid setup aggregate")
        metrics[key]=dict(count=len(numbers),minimum=min(numbers),maximum=max(numbers),mean=statistics.mean(numbers),median=statistics.median(numbers))
    return dict(schema="sircus-direct-final-producer-preflight-summary-v1",status="passed",producer=provenance,
        source_setup_sha256=sha(raw),source_raw_sha256=sha(raw_history),source_audit_sha256=sha(audit_path),
        counts=summary,paired_aggregates=metrics,paired_count=100,paired_canonical_json_sha256=canonical_digest(pairs),
        scope="Direct final-producer setup-only replay of100 fixed previous training requests. No task ball releases, PPO samples, captures or held-out evaluation. Startup contact probes are separate; raw paired/transit/physics histories excluded."),audit


def film_receipt(run,export,report):
    manifest=read(safe_file(run,"clips.json",limit=2*1024**3))
    native=manifest["clips"] if isinstance(manifest,dict) else manifest
    full,action=report.get("movie"),report.get("action_movie")
    require(isinstance(full,dict) and isinstance(action,dict),"Both full and action-edited native movies required")
    clips=report["clips"];edits=action["clips"]
    require(len(clips)==len(native)==len(edits) and clips,"Missing native film chapters")
    expected_train=report["study"]["filmed_training_batches"]
    require([c["batch"] for c in clips if c["mode"]=="train"]==expected_train,
            "Film omits or reorders prescheduled training batches")
    require(sum(c["mode"]=="coverage" for c in clips)==report["study"]["coverage_batches"],"Missing prescheduled coverage film")
    release_frames=removed=0
    foreground=collections.Counter()
    for c,e,n in zip(clips,edits,native):
        require(all(c[k]==n[k] for k in ("mode","batch","trial_file","camera_sha256")),"Film chapter identity changed")
        require(c["camera_sha256"]==CAMERA_SHA256 and c["decode_verified"] and e["decode_verified"],"Unverified or wrong-camera film")
        require(c["frames"]==len(n["frames"])==c["native_frame_audit"]["frames"],"Native clip frame count changed")
        kept=e["retained_native_global_frames"]
        require(kept==sorted(set(kept)) and len(kept)==e["frames"],"Edited frame indices duplicated or reordered")
        require(set(kept)<={f["frame"] for f in n["frames"]},"Edited frame does not belong to its native chapter")
        releases={f["frame"] for f in n["frames"] if f["phase"]=="release"}
        require(releases<=set(kept),"A native release frame was removed")
        require(e["original_clip"]==c["file"] and e["original_frames"]==c["frames"],"Action edit belongs to another chapter")
        for media in (c,e):
            require(sha(safe_file(export,media["file"],limit=8*1024**3))==media["sha256"],"Encoded media bytes changed")
        if c["mode"]=="train":
            prefix=c["prior_training"]
            require(prefix["completed_batches"]==c["batch"] and prefix["attempts"]==c["batch"]*25
                    and prefix["batch_ids"]==list(range(c["batch"])),"Film caption uses current/future/missing outcomes")
        outcome=c.get("foreground",{}).get("outcome",{})
        foreground[str(outcome.get("state","unknown"))]+=1
        release_frames+=len(releases);removed+=c["frames"]-e["frames"]
    for movie,children in ((full,clips),(action,edits)):
        require(movie["decode_verified"] and movie["frames"]==sum(c["frames"] for c in children)
                and sha(safe_file(export,movie["file"],limit=8*1024**3))==movie["sha256"],"Full encoded film checksum/count changed")
        require(movie["resolution"]==[1920,1080] and abs(movie["fps"]-30)<.01,"Final native video format changed")
    return dict(schema="sircus-diverse-compact-film-v1",status="passed",source_clips_sha256=sha(run/"clips.json"),
        movie=compact(full),action_movie=compact(action),clips=compact(clips),chapters=len(clips),
        retained_release_frames=release_frames,removed_setup_dwell_frames=removed,foreground_outcome_counts=dict(foreground),
        scope="Actual native fixed-camera frames, all scheduled chapters in order. Only declared setup dwells removed; every release frame retained. Large articulation travel is scripted, not learned motor control. Foreground outcomes are descriptive, not whole-farm frozen performance. Media and native readbacks stay outside Git.")


def requested_csv(raw,expected_cells=None):
    reader=csv.DictReader(io.StringIO(raw.decode()))
    require(reader.fieldnames and len(set(reader.fieldnames))==len(reader.fieldnames),"Invalid or duplicate CSV fields")
    require(set(CSV_FIELDS)<=set(reader.fieldnames),"Missing requested-settings fields")
    rows=list(reader)
    require(len(rows)==35 and len({r["cell_id"] for r in rows})==35,"Expected35 distinct settings rows")
    if expected_cells is not None:
        expected={c["id"]:c for c in expected_cells}
        require(set(expected)=={r["cell_id"] for r in rows},"CSV workcell membership changed")
        for row in rows:
            cell=expected[row["cell_id"]]
            require(row["split"]==cell["split"] and all(math.isclose(float(row[k]),cell["bucket_xy_m"][i],rel_tol=0,abs_tol=1e-12)
                    for i,k in enumerate(("bucket_x_m","bucket_y_m"))),"CSV split or bucket geometry changed")
    output=io.StringIO(newline="")
    writer=csv.DictWriter(output,fieldnames=CSV_FIELDS,lineterminator="\n")
    writer.writeheader();writer.writerows({k:r[k] for k in CSV_FIELDS} for r in rows)
    return output.getvalue().encode()


def contact_receipts(run):
    probes=read(safe_file(run,"probes.json"))
    rows=probes.get("probes",[]) if isinstance(probes,dict) else probes
    per_cell=("bucket","rim","table","floor","robot")
    foreign="foreign_cell_collision_filter"
    require(isinstance(rows,list) and len(rows)==6 and len({r.get("kind") for r in rows})==6
            and {r.get("kind") for r in rows}==set(per_cell)|{foreign},
            "Expected the five35-cell contact kinds and one foreign-cell isolation probe")
    require(all(row.get("all_passed") is True for row in rows),"Contact/isolation probes not passed")
    by_kind={r["kind"]:r for r in rows}
    summaries=[]
    for kind in (*per_cell,foreign):
        row=by_kind[kind]
        if kind in per_cell:
            flags=row.get("per_cell_passed")
            require(isinstance(flags,list) and len(flags)==35 and all(v is True for v in flags),
                    "Every per-cell contact kind requires35 explicit passed Boolean flags")
            peaks=row.get("peak_N")
            require(isinstance(peaks,list) and len(peaks)==35 and all(isinstance(p,list) and len(p)==6
                    and all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in p) for p in peaks),
                    "Invalid35-by-six native contact-peak readbacks")
            target={"bucket":1,"rim":1,"table":2,"floor":3,"robot":4}[kind]
            actual=[p[target]>.01 and (kind not in ("table","floor") or p[0]<=.01 and p[1]<=.01) for p in peaks]
            require(actual==flags,"Passed contact flags disagree with native force gates")
            count=dict(per_cell_count=35,per_cell_passed_count=35,single_isolation_probe_count=None,
                       single_isolation_passed_count=None)
        else:
            require("per_cell_passed" not in row,"Foreign-cell isolation is one probe, not35 per-cell checks")
            peak=row.get("peak_N");z=row.get("foreign_ball_final_z_m")
            require(isinstance(peak,list) and len(peak)==6 and all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in peak)
                    and type(z) in (int,float) and math.isfinite(z),"Invalid single foreign-cell isolation readback")
            require(max(peak)<=.01 and z<-1,"Foreign-cell isolation flags disagree with native force/fall gates")
            count=dict(per_cell_count=None,per_cell_passed_count=None,single_isolation_probe_count=1,
                       single_isolation_passed_count=1)
        summaries.append(dict(kind=kind,all_passed=True,**count,
            omitted_fields_sha256=canonical_digest({k:v for k,v in row.items() if k not in ("kind","all_passed","per_cell_passed")})))
    contact=dict(source_sha256=sha(run/"probes.json"),probes=summaries,
        per_cell_probe_kinds=5,per_cell_check_count=175,single_isolation_probe_count=1,
        scope="Compact startup contact/isolation checks: five kinds at35 cells plus one foreign-cell isolation probe. Force readbacks omitted and hash-bound. Not task-shot or capture counts.")
    rays=read(safe_file(run,"rim-rays.json"))
    require(len(rays["passed"])==35 and all(v is True for v in rays["passed"]),"Rim rays did not pass at all35 cells")
    rim=dict(source_sha256=sha(run/"rim-rays.json"),all35_passed=True,rays=len(rays.get("rays",[])),clearance_m=rays.get("clearance_m"),
        raw_readbacks_sha256=canonical_digest(rays),scope="Startup rim-ray contact diagnostics; native per-ray force arrays excluded.")
    return contact,rim


def curate(run,export,output,*,setup_run,setup_audit,support_cost,expected_report_sha256,
           expected_native_audit_sha256,expected_checkpoint_audit_sha256,expected_setup_audit_sha256,
           auditor_commit,exporter_commit,producer=FINAL_PRODUCER,archive=FINAL_ARCHIVE,diagnostics=()):
    run,export,output=map(Path,(run,export,output))
    require(not output.exists(),"Curated output must be a new directory")
    require(not any(output.resolve().is_relative_to(p.resolve()) for p in (run,export,Path(setup_run))),"Output must be outside immutable source evidence")
    require(all(COMMIT_RE.fullmatch(v) for v in (producer,auditor_commit,exporter_commit))
            and SHA_RE.fullmatch(archive),"Explicit full source identities required")
    trusted=(expected_report_sha256,expected_native_audit_sha256,expected_checkpoint_audit_sha256,expected_setup_audit_sha256)
    require(all(SHA_RE.fullmatch(v) for v in trusted),"Explicit trusted final receipt hashes required")
    report_path=safe_file(export,"report.json")
    require(sha(report_path)==expected_report_sha256,"Final report bytes changed")
    report=read(report_path);result=read(safe_file(run,"result.json"));study=read(safe_file(run,"study.json"))
    require(report.get("status")==result.get("status")=="completed"
            and study.get("schema")==result.get("schema")==STUDY_SCHEMA,"Completed diverse native study required")
    contract=read(safe_file(run,"contract.json"))
    require(signed(study,"Study")==report["study_fingerprint"]
            and signed(contract,"Geometry")==study["geometry_fingerprint"],"Signed study/geometry identity changed")
    require(report["study"]==study and report["study_fingerprint"]==result["study_fingerprint"]==study["fingerprint"]
            and report["geometry_fingerprint"]==result["geometry_fingerprint"]==study["geometry_fingerprint"],"Study/result/report binding mismatch")
    require(report["source_sha256"]["result.json"]==sha(run/"result.json")
            and report["source_sha256"]["study.json"]==sha(run/"study.json"),"Report source bytes changed")
    require(study["training_batches"]==64 and report["training"]["attempts"]==1600,"Matched1600-attempt study required")
    require(study["settling_protocol"]["version"]=="sircus-driven-settlement-v5","Final full-DOF controller protocol required")
    native_producer=identity(run,producer,archive)
    require(native_producer["source_members_sha256"].get("sircus/isaac/settled_transit.py")
            == result.get("source_sha256",{}).get("settled_transit.py"),"Native controller source differs from archived source")
    receipts={}
    for key,expected in (("native",expected_native_audit_sha256),("checkpoint",expected_checkpoint_audit_sha256)):
        p=safe_file(run,key+"-audit.json");value=read(p)
        require(sha(p)==expected==report["audits"][key]["sha256"] and value.get("status")=="passed"
                and value.get("source_result_sha256")==sha(run/"result.json"),"Final independent "+key+" audit mismatch")
        receipts[key]=value
    feasibility=read(safe_file(run,"feasibility-audit.json"))
    require(feasibility.get("status")=="passed" and sha(run/"feasibility-audit.json")==report["audits"]["feasibility"]["sha256"]
            and feasibility.get("feasibility_sha256",feasibility.get("source_result_sha256"))
            == study["feasibility_sha256"]==sha(safe_file(run,"feasibility.json")),"Feasibility audit mismatch")
    c=report["comparison"]
    require(c.get("schema")==COMPARISON_SCHEMA and c.get("status")=="completed","Completed three-method comparison required")
    for key,expected in (("broad",BASELINE_RESULT_SHA256),("guided",GUIDED_RESULT_SHA256),("diverse",sha(run/"result.json"))):
        require(c["methods"][key]["source_result_sha256"]==expected,"Comparison baseline/result identity changed")
    independent=c.get("independent_audit",{})
    comparison_file=safe_file(run,"comparison-audit.json")
    require(independent.get("sha256")==sha(comparison_file),"Independent comparison receipt bytes changed")
    verify_comparison_audit(read(comparison_file),c)
    support_path=safe_file(run,"support.json")
    support=read(support_path);definitions=compact_support(support)
    inspect(support)
    require(study["support_sha256"]==sha(support_path) and study["support_fingerprint"]==support["fingerprint"],"Full signed support identity changed")
    require([(c["cell_id"],c["bucket_xy_m"]) for c in support["cells"]]
            == [(c["id"],c["bucket_xy_m"]) for c in study["cells"]],"Full support context order or geometry changed")
    require(receipts["checkpoint"].get("updates")==16 and receipts["checkpoint"].get("optimizer_samples")==1600
            and receipts["checkpoint"].get("support_sha256")==sha(support_path),"Checkpoint budget or support binding changed")
    preflight,preflight_audit=preflight_evidence(setup_run,Path(setup_audit),expected_setup_audit_sha256,study,producer,archive)
    require(preflight["producer"]["source_members_sha256"].get("sircus/isaac/settled_transit.py")
            == native_producer["source_members_sha256"]["sircus/isaac/settled_transit.py"],"Direct preflight controller source differs from main native producer")
    setup_attached=read(safe_file(run,"setup-check.json"))
    setup_check_receipt(setup_attached,report)
    require(setup_attached["source_setup_check_sha256"]==preflight["source_setup_sha256"],"Attached preflight differs from direct final-producer check")
    require(sha(run/"setup-check.json")==c["setup_check"]["sha256"],"Report attached setup receipt changed")
    cost=read(support_cost)
    require(sha(support_cost)==c["geometry_preparation"]["process_resource_receipt"]["sha256"]
            and cost["support_sha256"]==sha(support_path) and cost["support_fingerprint"]==support["fingerprint"],"Support cost receipt changed")
    require(sha(safe_file(run,"selected-camera.json"))==study["source"]["camera_sha256"]==CAMERA_SHA256,"User-selected camera changed")
    contact,rim=contact_receipts(run)
    film=film_receipt(run,export,report)
    csv_path=safe_file(export,"settings.csv")
    csv_raw=csv_path.read_bytes()
    require(sha(csv_path)==report["settings_csv"]["sha256"],"Native CSV bytes changed")
    requested=requested_csv(csv_raw,study["cells"])
    checkpoints={}
    for stage in ("initial","final"):
        p=safe_file(run,f"policy-{stage}.pt",MAX_MODEL)
        require(receipts["checkpoint"]["checkpoints"].get(p.name)==sha(p),"Model bytes differ from audited checkpoint")
        checkpoints[p.name]=p
    summary=dict(schema=SCHEMA,status="completed",study_fingerprint=study["fingerprint"],geometry_fingerprint=study["geometry_fingerprint"],
        domain_fingerprint=study["domain"]["fingerprint"],support_fingerprint=support["fingerprint"],support_sha256=sha(support_path),
        source_result_sha256=sha(run/"result.json"),training=report["training"],comparison=c,
        source_identity=dict(native_producer=native_producer,preflight_producer=preflight["producer"],auditor_commit=auditor_commit,exporter_commit=exporter_commit),
        elapsed_s_including_setup_film_evaluation=report["elapsed_s"],model_updates=16,
        scoring=report["scoring"],uncertainty=report["uncertainty"],
        settings_source_sha256=sha(csv_path),requested_settings_sha256=hashlib.sha256(requested).hexdigest(),
        settings_scope="Requested values, per-site outcomes and derived pose errors only; direct measured board/joint/retention/contact readbacks excluded.",
        scope=c["scientific_scope"]+" No automatic model promotion, physical readiness or collision-safe-path claim. Full signed support contains offline geometry and IK warm starts only, not native observations; every requested shot still needs fresh qualification.")
    # Keep the comparison data but omit raw/per-request fields defensively.
    data={"study.json":study,"contract.json":contract,"summary.json":compact(summary),
        "comparison.json":compact(c),"learning-curve.json":report["learning_curve"],"region-usage.json":c["region_usage"],
        "support-definitions.json":definitions,"native-audit.json":compact(receipts["native"]),
        "checkpoint-audit.json":compact(receipts["checkpoint"]),"comparison-audit.json":compact(read(comparison_file)),
        "feasibility-audit.json":compact(feasibility),"setup-summary.json":compact(preflight),
        "setup-audit.json":compact(preflight_audit),"setup-attached.json":compact(setup_attached),
        "contact-probes.json":contact,"rim-rays.json":rim,"support-cost.json":cost,"film.json":film,
        "camera-binding.json":dict(source_sha256=sha(safe_file(run,"camera-binding.json")),selected_camera_sha256=CAMERA_SHA256,
            raw_metadata_sha256=canonical_digest(read(run/"camera-binding.json")),scope="Original camera binding metadata remains private; every recorded frame/readback was separately verified by the native audit.")}
    history=[]
    for path in map(Path,diagnostics):
        value=read(path)
        history.append(dict(source_sha256=sha(path),status=value.get("status"),source_commit=value.get("source_commit"),
            training_batches=value.get("training_batches"),ppo_updates=value.get("ppo_updates"),scope=value.get("scope"),
            use="Historical abort/queue provenance only; excluded from this study's optimizer, evaluation and admission totals."))
    if history:data["historical-diagnostics.json"]=history
    for value in data.values():inspect(value)
    inputs={"report":sha(report_path),"result":sha(run/"result.json"),"study":sha(run/"study.json"),
        "native_audit":expected_native_audit_sha256,"checkpoint_audit":expected_checkpoint_audit_sha256,
        "setup_audit":expected_setup_audit_sha256,"full_settings_csv":sha(csv_path),"full_support":sha(support_path)}
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=Path(tempfile.mkdtemp(prefix="."+output.name+"-",dir=output.parent))
    try:
        for name,value in data.items():(temporary/name).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n")
        (temporary/"requested-settings.csv").write_bytes(requested)
        for name,p in checkpoints.items():shutil.copyfile(p,temporary/name)
        shutil.copyfile(support_path,temporary/"support.json")
        shutil.copyfile(run/"selected-camera.json",temporary/"selected-camera.json")
        (temporary/"README.md").write_text(readme(summary,preflight,film,output.name))
        for p in temporary.iterdir():
            require(not SENSITIVE.search(p.read_bytes()),"Credential pattern in selected output")
        manifest=dict(schema=SCHEMA,study_fingerprint=study["fingerprint"],geometry_fingerprint=study["geometry_fingerprint"],
            domain_fingerprint=study["domain"]["fingerprint"],support_fingerprint=support["fingerprint"],source_inputs_sha256=inputs,
            native_source_commit=producer,native_source_archive_sha256=archive,auditor_commit=auditor_commit,exporter_commit=exporter_commit,
            curator_code_sha256=sha(__file__),
            files={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(temporary.iterdir())},
            external_media={x["file"]:x["sha256"] for x in (report["movie"],report["action_movie"])},
            scope="Manifest excludes itself. Full signed geometry support and byte-identical audited weights retained. No raw native observations, host calibration, credentials, frames, media, logs or runtime launch configuration copied.")
        (temporary/"manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
        require(not output.exists(),"Output appeared while curating; refusing replacement")
        temporary.rename(output)
        return manifest
    finally:
        if temporary.exists():shutil.rmtree(temporary)


def readme(summary,preflight,film,baseline_name):
    c=summary["comparison"]
    rows=[]
    for split in ("train","validation","test"):
        cells=[]
        for method in ("broad","guided","diverse"):
            r=c["methods"][method]["frozen"][split]["final"]
            cells.append(f"{r['captures']}/{r['attempts']} ({100*r['capture_rate']:.1f}%)")
        rows.append("| "+split+" | "+" | ".join(cells)+" |")
    return f'''# Diverse-region PPO and measured settling

Completed simulation-only matched pilot: 1,600 attempted training shots, 16 PPO
updates, 25 training/five validation/five test coordinates. Every rejected active
attempt remains in the optimizer and denominator. Both older studies stay frozen.

| Frozen final subset | Broad | One local region | Diverse + settling |
|---|---:|---:|---:|
{chr(10).join(rows)}

This is one seed on inherited layouts, with geometry assistance at held-out
coordinates. Nominal repeats measure numerical repeatability; they are not new
layouts or independent noise trials. Proposal support, hybrid policy and setup
controller change together. No isolated PPO, arbitrary-table, physical-readiness
or collision-safe-path claim follows from these numbers.

Success requires a board bounce, whole-ball bucket entry and one continuous
second inside without intervening external route contact. Rim-assisted captures
count; clean captures remain separate. Historical definitions are unchanged.

The signed `support.json` is preserved byte-for-byte with its original fingerprint.
It contains offline geometry anchors, IK warm-start joints and path-proxy records,
not measured native trajectories. Local regions have qualified anchors and
unqualified interiors. Each supported site also has an explicitly unqualified
global exploration domain. Neither a box nor its envelope certifies feasibility.

The direct final-producer setup check admitted {preflight['counts']['before_admitted']}
before and {preflight['counts']['after_admitted']} after on 100 fixed prior training
requests, with {preflight['counts']['repaired']} repairs and {preflight['counts']['regressions']}
regressions. It made zero task-shot releases and measures admission, not capture
or held-out policy performance. Its producer identity remains separate from all
earlier prototype/queued runs.

The actual fixed-camera film is {film['movie']['duration_s']:.2f}s full and
{film['action_movie']['duration_s']:.2f}s after setup-dwell trimming. All
{film['retained_release_frames']} recorded release frames are retained. Large
visible arm travel is scripted; grasp/loading/joint paths are not learned. Film
digests/cuts are in `film.json`; media stays outside Git.

For CPU prediction with the repository's documented Torch environment, set
`PYTHONPATH=sircus/isaac` and `CUDA_VISIBLE_DEVICES=''`:

```python
import json
from pathlib import Path
import torch
from diverse_policy import load_checkpoint, build_observations, sample
from diverse_support import physical_actions

base = Path({('baselines/'+baseline_name)!r})
m = json.loads((base/'manifest.json').read_text())
support = json.loads((base/'support.json').read_text())
policy, saved = load_checkpoint(base/'policy-final.pt', support=support,
    study_fingerprint=m['study_fingerprint'], geometry_fingerprint=m['geometry_fingerprint'],
    domain_fingerprint=m['domain_fingerprint'], support_fingerprint=m['support_fingerprint'], device='cpu')
study = json.loads((base/'study.json').read_text())
contexts = [c for c in study['cells'] if c['id'] == 'r2c3']
picked = sample(policy, torch.as_tensor(build_observations(contexts), dtype=torch.float32), deterministic=True)
print(physical_actions(picked['actions_normalized'], support, contexts, picked['region_indices']))
```

Ordinary inference rejects unknown/unresolved geometry contexts. Predicted
parameters are not native-qualified motion commands. Do not bypass the saved
study/domain/support identities or native release/IK/path guards.

`manifest.json` checksums every selected file. `requested-settings.csv` retains
requests, outcomes and derived errors; direct native pose/readback columns stay
private. Native/checkpoint/setup audits, exact source hashes and aggregate
provenance are retained. Raw physics/paired/transit observations, RGB/depth,
generated runtime configuration, credentials, queues, logs and media are excluded.
The 175 mm bucket-height interpretation and fixed material assumptions remain
provisional; see the unchanged signed contract.
'''


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("run","export","output"):p.add_argument(name,type=Path)
    for name in ("setup-run","setup-audit","support-cost"):p.add_argument("--"+name,type=Path,required=True)
    for name in ("report-sha256","native-audit-sha256","checkpoint-audit-sha256","setup-audit-sha256","auditor-commit","exporter-commit"):
        p.add_argument("--"+name,required=True)
    p.add_argument("--producer",default=FINAL_PRODUCER)
    p.add_argument("--archive",default=FINAL_ARCHIVE)
    p.add_argument("--diagnostic-record",type=Path,action="append",default=[])
    a=p.parse_args()
    value=curate(a.run,a.export,a.output,setup_run=a.setup_run,setup_audit=a.setup_audit,support_cost=a.support_cost,
        expected_report_sha256=a.report_sha256,expected_native_audit_sha256=a.native_audit_sha256,
        expected_checkpoint_audit_sha256=a.checkpoint_audit_sha256,expected_setup_audit_sha256=a.setup_audit_sha256,
        auditor_commit=a.auditor_commit,exporter_commit=a.exporter_commit,producer=a.producer,archive=a.archive,diagnostics=a.diagnostic_record)
    print(json.dumps(dict(status="curated",files=len(value["files"]),source_result_sha256=value["source_inputs_sha256"]["result"])))
