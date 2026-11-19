"""Checked compact publication of calibrated benchmark evidence.

No Torch deserialization, physics, hardware, SSH, Git or source-data mutation.
Study and audit versions are deliberately distinct. Synthetic test evidence is
refused unless the caller explicitly requests fixture export.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import statistics
import tempfile
import zipfile

from .calibrated_benchmark_contract import VERSION as STUDY_VERSION, AUDIT_VERSION, SEEDS, BUDGET, MILESTONES

VERSION = "sircus-calibrated-benchmark-compact-export-v2"
MAX_FILE_BYTES = 16 * 1024**2
MAX_BUNDLE_BYTES = 64 * 1024**2
RESOURCE_FIELDS = ("wall_s","cpu_s","geometry_s","path_s","context_resolution_s","admission_wall_s","postprocess_s","gpu_s","native_init_s","setup_s","evaluation_wall_s",
                   "gpu_reservation_s","peak_rss_mb","peak_vram_mb","torch_allocated_mb","torch_reserved_mb")
SECRET_PATTERNS = (rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",rb"AKIA[0-9A-Z]{16}",
                   rb"AIza[0-9A-Za-z_-]{30,}",rb"gh[pousr]_[A-Za-z0-9]{25,}")
HOST_PATTERN = re.compile(r"(?:/Users/[^/\s]+/|/home/[^/\s]+/|/private/|/data/|\b(?:192\.168\.|10\.\d+\.\d+\.))")


def sha_bytes(raw): return hashlib.sha256(raw).hexdigest()
def digest(value): return sha_bytes(json.dumps(value,sort_keys=True,allow_nan=False).encode())
def require(value, message):
    if not value: raise ValueError(message)


def safe_member(root, name):
    require(isinstance(name,str) and bool(name),"Unsafe member path type")
    p=PurePosixPath(name)
    require(not p.is_absolute() and ".." not in p.parts and "\\" not in name,"Unsafe member path")
    target=Path(root)/name
    require(not target.is_symlink() and target.resolve().is_relative_to(Path(root).resolve()),"Unsafe member target")
    return target


def raw_file(path):
    path=Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size<=MAX_FILE_BYTES,"Missing, unsafe or oversized compact input")
    return path.read_bytes()


def signed(value, label):
    unsigned=dict(value);actual=unsigned.pop("fingerprint",None)
    require(actual==digest(unsigned),label+" fingerprint mismatch")
    return actual


def json_safety(value):
    if isinstance(value,dict):
        for key,item in value.items():
            if str(key).lower() in {"api_key","api_key_id","password","secret","access_token","refresh_token"}:
                require(not isinstance(item,str) or not item or item.startswith(("REPLACE","YOUR_","<")),"Credential-like metadata field")
            json_safety(item)
    elif isinstance(value,list):
        for item in value:json_safety(item)
    elif isinstance(value,str):
        require(not HOST_PATTERN.search(value),"Host path or private endpoint in compact metadata")
    elif isinstance(value,float):
        require(math.isfinite(value),"Nonfinite compact metadata")


def payload_safety(name, raw):
    require(not any(re.search(pattern,raw) for pattern in SECRET_PATTERNS),"Credential-like payload bytes")
    if name.endswith(".json"):
        json_safety(json.loads(raw))
    if name.endswith(".pt"):
        require(zipfile.is_zipfile(io.BytesIO(raw)),"Selected checkpoint is not a safe ZIP container")
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            require(sum(m.file_size for m in archive.infolist())<=MAX_FILE_BYTES,"Oversized checkpoint archive")
            for m in archive.infolist():
                p=PurePosixPath(m.filename)
                require(not p.is_absolute() and ".." not in p.parts and "\\" not in m.filename,"Unsafe checkpoint ZIP member")
                require((m.external_attr>>16)&0o170000 != 0o120000,"Checkpoint ZIP symlink")
                data=archive.read(m)
                require(not any(re.search(pattern,data) for pattern in SECRET_PATTERNS),"Credential-like checkpoint bytes")


def csv_bytes(rows):
    fields=[]
    for row in rows:
        for key in row:
            if key not in fields:fields.append(key)
    stream=io.StringIO(newline="")
    writer=csv.DictWriter(stream,fieldnames=fields,lineterminator="\n")
    writer.writeheader();writer.writerows(rows)
    return stream.getvalue().encode()


def aggregate(rows, keys, fields):
    result=[]
    groups={tuple(r[k] for k in keys) for r in rows}
    for group in sorted(groups):
        selected=[r for r in rows if tuple(r[k] for k in keys)==group]
        row=dict(zip(keys,group));row["seeds"]=len(selected)
        for field in fields:
            values=[float(r[field]) for r in selected if r.get(field) is not None]
            row[field+"_observed_seeds"]=len(values)
            row[field+"_mean"]=statistics.mean(values) if values else None
            row[field+"_sample_sd"]=statistics.stdev(values) if len(values)>1 else None
        result.append(row)
    return result


def export(study,audit_path,repository,output,*,expected_contract_fingerprint,expected_audit_sha256,allow_synthetic=False):
    study,audit_path,repository,output=(Path(p).resolve() for p in (study,audit_path,repository,output))
    require(not output.exists() and not output.is_relative_to(study),"New compact output must be outside source study")
    audit_raw=raw_file(audit_path)
    require(sha_bytes(audit_raw)==expected_audit_sha256,"Pinned audit bytes changed")
    audit=json.loads(audit_raw)
    require(audit.get("version")==AUDIT_VERSION and audit.get("status")=="passed" and not audit.get("failures")
            and type(audit.get("checks")) is int and audit["checks"]>0,"Completed matching independent audit required")
    report_raw,contract_raw,manifest_raw=(raw_file(study/name) for name in ("report.json","contract.json","manifest.json"))
    require(sha_bytes(report_raw)==audit["report_sha256"] and sha_bytes(manifest_raw)==audit["study_manifest_sha256"],"Audit/report/manifest byte mismatch")
    contract,report,manifest=(json.loads(raw) for raw in (contract_raw,report_raw,manifest_raw))
    require(contract.get("version")==STUDY_VERSION and report.get("version")==STUDY_VERSION,"Study version mismatch")
    fp=signed(contract,"Contract");signed(report,"Report")
    require(fp==expected_contract_fingerprint==audit["contract_fingerprint"]==report["contract_fingerprint"],"Frozen contract binding mismatch")
    require(report.get("fingerprint")==audit["report_fingerprint"] and report.get("status")=="completed" and report.get("phase")=="completed","Complete immutable report required")
    p=contract["protocol"];kind=p.get("study_kind")
    require(kind in {"native_current_ball_v2","synthetic_fixture"},"Unknown evidence kind")
    require(kind!="synthetic_fixture" or allow_synthetic,"Synthetic evidence refused by native exporter")
    require(bool(audit.get("synthetic_fixture"))==(kind=="synthetic_fixture"),"Audit/study evidence kind mismatch")
    if kind=="native_current_ball_v2":
        require(p["seeds"]==SEEDS and p["sample_budget"]==BUDGET and p["milestones"]==MILESTONES,"Canonical native protocol changed")
    require(report["protocol"]==p and audit["source_sha256"]==contract["identity"]["source_sha256"],"Protocol/execution source changed")
    for name,expected in audit["source_sha256"].items():
        require(sha_bytes(raw_file(safe_member(repository,name)))==expected,"Execution source bytes changed")
    require(sha_bytes(raw_file(repository/"sircus/mjlab/calibrated_benchmark_verify.py"))==audit["auditor_sha256"],"Auditor source bytes changed")
    for name,expected in audit.get("auditor_dependencies",{}).items():
        require(sha_bytes(raw_file(safe_member(repository,name)))==expected,"Auditor dependency changed")
    pairs={(method,seed) for method in ("sweep","ppo") for seed in p["seeds"]}
    require(len(pairs)==6 and len(report["runs"])==6 and {(r["method"],r["seed"]) for r in report["runs"]}==pairs,"Six unique complete workers required")
    expected_trials=len(pairs)*(p["sample_budget"]+36*len(p["milestones"])+72)
    require(audit["requested_trials"]==expected_trials,"Full requested-trial audit coverage required")
    expected_roles={"train":len(pairs)*p["sample_budget"],"validation":len(pairs)*36*len(p["milestones"]),"test":len(pairs)*72}
    require({role:audit["role_counts"][role]["attempts"] for role in expected_roles}==expected_roles,"Audit phase budgets differ")

    copies={"report.json":report_raw,"contract.json":contract_raw,"audit.json":audit_raw,"original-manifests/study.json":manifest_raw}
    def copy_member(name,target=None):
        raw=raw_file(safe_member(study,name))
        require(manifest.get(name)==sha_bytes(raw),"Original manifest does not bind copied payload")
        copies[target or name]=raw
        return raw
    for name in ("calibration.json","verification.json","setup-home.json","prior.json","splits.json"):
        copy_member(name)
    curves,finals,selections=[],[],[]
    for worker in sorted(report["runs"],key=lambda r:(r["method"],r["seed"])):
        method,seed=worker["method"],worker["seed"]
        prefix=f"runs/{method}-seed{seed}"
        require(worker.get("status")=="completed" and worker.get("phase")=="completed" and worker.get("train_requests")==p["sample_budget"],"Worker training budget incomplete")
        raw=raw_file(safe_member(study,prefix+"/report.json"))
        require(manifest.get(prefix+"/report.json")==sha_bytes(raw) and json.loads(raw)==worker,"Overall/worker report mismatch")
        signed(worker,"Worker")
        run_manifest_raw=copy_member(prefix+"/manifest.json",f"original-manifests/{method}-seed{seed}.json")
        run_manifest=json.loads(run_manifest_raw)
        require([r["samples"] for r in worker["curves"]]==p["milestones"],"Missing curve milestone")
        for point in worker["curves"]:
            count=point["validation"]
            require(count["attempts"]==36 and 0<=count["captures"]<=count["native_releases"]<=36,"Invalid validation counts")
            curves.append(dict(method=method,seed=seed,samples=point["samples"],captures=count["captures"],requested=36,
                               capture_rate=count["captures"]/36,native_releases=count["native_releases"],invalid=count["invalid"],
                               **{k:point["resources"].get(k) for k in RESOURCE_FIELDS}))
        selected_path=prefix+"/selected.json";selection_raw=copy_member(selected_path,f"selected/{method}-seed{seed}/selected.json")
        selection=json.loads(selection_raw);artifact=selection["artifact"]
        require(selection.get("frozen_before_test") is True and artifact["samples"]==worker["selected_samples"],"Selected artifact was not frozen")
        selected_name=prefix+"/"+artifact["path"]
        artifact_raw=raw_file(safe_member(study,selected_name))
        require(manifest.get(selected_name)==artifact["sha256"]==run_manifest.get(artifact["path"])==sha_bytes(artifact_raw)==worker["checkpoint_sha256"],"Selected artifact hash chain changed")
        target=f"selected/{method}-seed{seed}/"+PurePosixPath(artifact["path"]).name
        copies[target]=artifact_raw
        final=worker["final_test"];count=final["counts"]
        require(final["frozen_artifact_sha256"]==artifact["sha256"] and count["attempts"]==72 and 0<=count["captures"]<=count["native_releases"]<=72,"Invalid final frozen cohort")
        finals.append(dict(method=method,seed=seed,spent_train_requests=worker["train_requests"],train_native_releases=worker["train_native_releases"],
                           selected_train_requests=worker["selected_samples"],captures=count["captures"],requested=72,capture_rate=count["captures"]/72,
                           native_releases=count["native_releases"],invalid=count["invalid"],clean=count["clean"],
                           cohort_prediction_preparation_s=final["query_s"],pre_prediction_plus_policy_s=final["pre_prediction_plus_policy_s"],selected_artifact_bytes=len(artifact_raw),artifact_sha256=artifact["sha256"],
                           **{k:worker["resources"].get(k) for k in RESOURCE_FIELDS}))
        selections.append(dict(method=method,seed=seed,source_path=selected_name,bundle_path=target,sha256=artifact["sha256"],samples=artifact["samples"]))
    require(len(curves)==6*len(p["milestones"]) and len(finals)==6,"Complete curve/final cohort extraction required")
    copies["per-seed-curves.csv"]=csv_bytes(curves)
    copies["per-seed-final.csv"]=csv_bytes(finals)
    copies["aggregate-curves.csv"]=csv_bytes(aggregate(curves,("method","samples"),("capture_rate",*RESOURCE_FIELDS)))
    copies["aggregate-final.csv"]=csv_bytes(aggregate(finals,("method",),("capture_rate",*RESOURCE_FIELDS,"cohort_prediction_preparation_s","pre_prediction_plus_policy_s","selected_artifact_bytes")))
    provenance=dict(version=VERSION,study_version=STUDY_VERSION,audit_version=AUDIT_VERSION,evidence_kind=kind,
                    contract_fingerprint=fp,report_sha256=sha_bytes(report_raw),audit_sha256=sha_bytes(audit_raw),
                    source_manifest_sha256=sha_bytes(manifest_raw),exporter_sha256=sha_bytes(Path(__file__).read_bytes()),
                    selections=selections,scope="Compact exact metadata/model copies plus derived descriptive CSVs; raw traces remain external and are covered by the original audit.")
    copies["provenance.json"]=(json.dumps(provenance,indent=2,allow_nan=False)+"\n").encode()
    summary=["# Calibrated current-ball sweep/PPO benchmark — v2", "",f"Evidence kind: **{kind}**. Independent audit passed {audit['checks']:,} checks over {audit['requested_trials']:,} requests.","",
             "Both methods share the 17 historical recipes and the same geometry-only pre-prediction source/staging resolver. Every requested training decision counts, including rejections. Geometry search is separately timed; held-out predictions do not perform ball-flight outcome search.","",
             "| Method | Paid TRAIN requests | Frozen test captures / requests |","|---|---:|---:|"]
    for method in ("sweep","ppo"):
        rows=[r for r in finals if r["method"]==method]
        summary.append(f"| {method} | {sum(r['spent_train_requests'] for r in rows)} | {sum(r['captures'] for r in rows)}/{sum(r['requested'] for r in rows)} |")
    summary += ["", "CSV aggregates are descriptive three-seed means and sample SDs; paired layouts/repetitions are dependent. Setup/TRAIN time excludes evaluation; reservation and lifetime memory include it. Query timing is cohort prediction/preparation, not an isolated actor-call latency; pre_prediction_plus_policy_s adds geometry resolver wall time. Same-action endpoint/path admission and native rollout remain separately timed, not included in that prediction-only sum. PPO checkpoint bytes include critic/optimizer state.","",
                "Calibration/verification/home records are copied exactly and their digests are frozen in the contract. Dense native traces were independently rescored by the passed audit; this subset export does not repeat physics or rehash unbundled raw arrays. Original manifests under original-manifests reference unbundled source evidence. SHA256SUMS covers only this compact bundle.","",
                "Historical v1 uses different scoring, setup preprocessing and evaluation cohorts; do not pool its counts or attribute a v1/v2 rate difference solely to learning. These are privileged-geometry held-endpoint simulator results, not physical grasp, driven transit, impact compliance, perception or new-ball adaptation validation. No checkpoint is activated by this bundle."]
    if kind=="synthetic_fixture":summary += ["", "**Synthetic software-test evidence only. No native dynamics or experimental performance is established.**"]
    copies["README.md"]=("\n".join(summary)+"\n").encode()
    require(len(copies)==33,"Unexpected compact payload count")
    for name,raw in copies.items():payload_safety(name,raw)
    require(sum(map(len,copies.values()))<=MAX_BUNDLE_BYTES,"Compact bundle too large")
    with tempfile.TemporaryDirectory(prefix=".calibrated-export-",dir=output.parent) as temp:
        staged=Path(temp)/"bundle";staged.mkdir()
        for name,raw in copies.items():
            target=safe_member(staged,name);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
        (staged/"SHA256SUMS").write_text("".join(f"{sha_bytes(raw)}  {name}\n" for name,raw in sorted(copies.items())))
        require(sha_bytes(raw_file(audit_path))==expected_audit_sha256 and sha_bytes(raw_file(study/"report.json"))==sha_bytes(report_raw)
                and sha_bytes(raw_file(study/"manifest.json"))==sha_bytes(manifest_raw),"Immutable source changed during export")
        require(not output.exists(),"Output appeared during export")
        staged.rename(output)
    return dict(status="exported",files=34,contract_fingerprint=fp,selected_artifacts=len(selections),evidence_kind=kind)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("study","audit","repository","output"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--expected-contract-fingerprint",required=True)
    p.add_argument("--expected-audit-sha256",required=True)
    # No fixture option in the production CLI. Unit tests call export directly.
    a=p.parse_args()
    print(json.dumps(export(a.study,a.audit,a.repository,a.output,expected_contract_fingerprint=a.expected_contract_fingerprint,
                            expected_audit_sha256=a.expected_audit_sha256),allow_nan=False))


if __name__=="__main__":main()
