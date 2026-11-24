"""Build a versioned development library from authenticated, rescored shot records.

This is a CPU metadata operation. It never edits or deletes source evidence.
The model receives original pre-IK request features; solved joints are used only
for deduplication and as historical provenance. Proposals keep the original pre-IK source seeds.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
from hashlib import sha256
import json
import math
from pathlib import Path

from . import board_first_score
from .neural_shot_model import feature

VERSION = "sircus-expanded-deduplicated-shot-library-v1"
SEED = 2026112409
NOMINAL_SCENE = "9a50295d9908be8634c2df1c0b460ec6011710cf8e14ce33cbc67a75749ff1c8"
NOMINAL_CALIBRATION = "ab0d2b5b0bcb9122b0ca3e996501d33a6ebe1069e57aebd11d7643a597f55a20"
PROTOCOL = dict(substeps=64, board_mode="articulated", gravity_mode="passive_only")


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_sha(path):
    h = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def vector(value, count):
    require(isinstance(value, (list, tuple)) and len(value) == count, "Missing command vector")
    require(all(type(x) in (float, int) and math.isfinite(x) for x in value), "Nonfinite command")
    return list(map(float, value))


def position(value):
    return tuple(round(x, 8) for x in vector(value, 2))


def split_for(xy, *, seed=SEED, forced_train_positions=()):
    key = position(xy)
    if key in {position(p) for p in forced_train_positions}:
        return "train"
    return "validation" if int(digest([seed, key])[:8], 16) % 5 == 0 else "train"


def request_features(request):
    """Preserve explicit proposal fields; never synthesize seeds from query q."""
    result = {key: copy.deepcopy(request[key]) for key in (
        "values", "bucket_xy_m", "board_heading_offset_deg", "board_twist_deg")}
    for arm in (10, 14):
        key = f"seed_q{arm}"
        if key in request:
            result[key] = copy.deepcopy(request[key])
    # Intent is a pre-release generator choice. Never infer it from the outcome.
    result["intended_route"] = request.get("intended_route", "direct")
    if result["intended_route"] not in {"direct", "table"}:
        result["intended_route"] = "direct"
    feature(result)
    return result


def command_key(request, native, provenance):
    """Sub-micrometer/radian numerical identity, not a broad similarity cluster.

    Joints are deliberately not angle-wrapped: whole-turn representations can
    alter feasible paths. Different elbows, tool yaw, or release points remain
    different settings even when their five task parameters coincide.
    """
    vec = lambda xs, n, digits: [round(x, digits) for x in vector(xs, n)]
    identity = dict(
        bucket=vec(request["bucket_xy_m"], 2, 8), values=vec(request["values"], 5, 8),
        heading=round(float(request["board_heading_offset_deg"]) % 360., 7),
        twist=round(float(request["board_twist_deg"]) % 360., 7),
        q10=vec(native["q10"], 6, 6), q14=vec(native["q14"], 6, 6),
        release=vec(native["release_ball_center_m"], 3, 8),
        scene=provenance["scene_sha256"], calibration=provenance["calibration_fingerprint"],
        protocol=native["contact_protocol"],
    )
    return digest(identity)


def group_native(observations, *, seed=SEED, forced_train_positions=()):
    """Each exact setting gets one training weight while retaining every outcome."""
    grouped = defaultdict(list)
    native_seen = set()
    for item in observations:
        token = (item["evidence"]["native_sha256"], item["evidence"]["native_row"])
        require(token not in native_seen, "Same native evidence supplied twice")
        native_seen.add(token)
        grouped[command_key(item["request"], item["native"], item["provenance"])].append(item)
    rows, candidates = [], []
    for key, repeats in sorted(grouped.items()):
        repeats.sort(key=lambda r: (r["evidence"]["native_sha256"], r["evidence"]["native_row"]))
        first = repeats[0]
        valid = [r for r in repeats if not r["score"]["invalid_physics"]]
        captures = [r for r in valid if r["score"]["route_success"]]
        request = request_features(first["request"])
        request["id"] = "expanded-shot-" + key
        split = split_for(request["bucket_xy_m"], seed=seed, forced_train_positions=forced_train_positions)
        kinds = Counter(r["score"]["route_class"] for r in captures)
        row = dict(split=split, dedup_key=key, request=request, setup_probability=1.,
                   capture_probability=len(captures)/len(valid) if valid else None,
                   valid_native_count=len(valid), repeats_count=len(repeats),
                   invalid_native_count=len(repeats)-len(valid), capture_count=len(captures),
                   observed_routes=dict(kinds), evidence=[r["evidence"] for r in repeats],
                   pre_ik_feature_variants=len({digest(request_features(r["request"])) for r in repeats}),
                   feature_source=first["evidence"],
                   source_phases=dict(Counter(r["evidence"]["historical_phase"] for r in repeats)))
        rows.append(row)
        if captures and split == "train":
            # Keep one demonstrated source per command, but expose its full
            # empirical rate. An occasional capture is not a reliability claim.
            successful = captures[0]
            actual = successful["score"]
            native = successful["native"]
            source = dict(id=request["id"], values=request["values"], bucket_xy_m=request["bucket_xy_m"],
                          q10=native["q10"], q14=native["q14"],
                          board_heading_offset_deg=request["board_heading_offset_deg"],
                          board_twist_deg=request["board_twist_deg"],
                          intended_route=request["intended_route"],
                          observed_routes=dict(kinds), observed_route=actual["route_class"],
                          capture_probability=row["capture_probability"],
                          valid_native_count=len(valid), repeats_count=len(repeats),
                          route_quality=actual.get("route_quality", 0),
                          source_evidence=successful["evidence"], dedup_key=key,
                          phase="train", role="train", split="train", lineage=[r["evidence"] for r in repeats])
            for arm in (10, 14):
                field=f"seed_q{arm}"
                if field in request:
                    source[field]=copy.deepcopy(request[field])
            candidates.append(source)
    return rows, candidates


class SourceReader:
    """Authenticated line/packet lookup with bounded, per-file in-memory caches."""
    def __init__(self, root):
        self.root = Path(root)
        self.inputs = {}
        self.line_cache = {}
        self.enriched = {}
        self.packets = {}
        self.study_manifests = {}

    def path(self, relative):
        path = (self.root / relative).resolve()
        require(path.is_relative_to(self.root.resolve()), "Source escapes evidence root")
        return path

    def bind(self, path, expected=None):
        path = Path(path)
        key = str(path)
        if key not in self.inputs:
            self.inputs[key] = file_sha(path)
        if expected is not None:
            require(self.inputs[key] == expected, "Source hash differs: " + str(path))
        return path

    def source_row(self, source):
        path = self.bind(self.path(source["file"]), source["metadata_sha256"])
        if path not in self.line_cache:
            self.line_cache[path] = lines(path)
        return self.line_cache[path][source["line"]-1]

    def lookup(self, item):
        root = self.path(item["source_study"])
        if root not in self.enriched:
            manifest_path = root/"evidence-manifest.json"
            if not manifest_path.exists():
                return None, "missing_original_evidence_manifest"
            self.study_manifests[root]=read(self.bind(manifest_path))
            rows = {}
            for name in ("records.jsonl", "rows.jsonl"):
                path = root / name
                if path.exists():
                    expected=self.study_manifests[root].get(name)
                    if not isinstance(expected,str):
                        continue
                    self.bind(path, expected)
                    for row in lines(path):
                        rows[row["id"]] = row
            self.enriched[root] = rows
        enriched = self.enriched[root].get(item["id"])
        if enriched is None:
            return None, "missing_original_request_record"
        packet = enriched.get("request_packet")
        if packet is None:
            return None, "missing_frozen_request_packet"
        path = (root / packet).resolve()
        require(path.is_relative_to(root), "Request packet escapes study")
        require(self.study_manifests[root].get(packet)==enriched["request_packet_sha256"],
                "Request packet differs from original study manifest")
        self.bind(path, enriched["request_packet_sha256"])
        if path not in self.packets:
            data = read(path)
            rows = data.get("rows", data.get("requests", [])) if isinstance(data, dict) else data
            self.packets[path] = {row["id"]: row for row in rows}
            require(len(rows) == len(self.packets[path]), "Duplicate request identity in packet")
        request = self.packets[path].get(item["id"])
        require(request is not None, "Recorded native row has no matching packet request")
        for key in ("id", "bucket_xy_m", "values"):
            require(request[key] == item["original"][key], "Request differs from scored command: " + key)
        # These optional fields identify the released observation independently
        # from similar IDs in controls or other methods.
        for key in ("native_cache_sha256", "native_cache_row"):
            if key in enriched:
                require(enriched[key] == item["original"][key], "Enriched native binding differs")
        if any(key not in request for key in ("board_heading_offset_deg", "board_twist_deg")):
            return (request, dict(file=str(path), sha256=self.inputs[str(path)], request_id=request["id"])), "orientation_not_explicit_in_original_request"
        for key in ("impact_offset_m", "release_offset_uv_m"):
            if key in request and any(abs(float(v)) > 1e-10 for v in request[key]):
                return None, "unrepresented_noncentral_release"
        return (request, dict(file=str(path), sha256=self.inputs[str(path)], request_id=request["id"])), None


def nominal_orientation_error(request, native, scene):
    """Independently check the historical zero-heading/zero-twist full board pose."""
    def multiply(a,b):
        w,x,y,z=a;v,p,q,r=b
        return [w*v-x*p-y*q-z*r,w*p+x*v+y*r-z*q,w*q-x*r+y*v+z*p,w*r+x*q-y*p+z*v]
    def rotation(axis,angle):
        q=[math.cos(angle/2),0.,0.,0.];q[axis+1]=math.sin(angle/2);return q
    values=vector(request["values"],5);bucket=vector(request["bucket_xy_m"],2)
    azimuth=math.atan2(bucket[1]-values[1],bucket[0]-values[0])+math.pi/2
    expected=multiply(multiply(rotation(2,azimuth),rotation(0,math.radians(values[3]))),
                      rotation(2,float(scene["board"]["mount_yaw_rad"])))
    actual=vector(native["board_quaternion_wxyz"],4)
    norm=math.sqrt(sum(x*x for x in actual))
    require(norm>0.,"Invalid native board quaternion")
    cosine=abs(sum(a*b/norm for a,b in zip(expected,actual)))
    return 2*math.acos(min(1.,max(0.,cosine)))


def recover_nominal(reader, item, native, original_result, reason):
    """Recover only historically nominal commands whose measured pose agrees.

    Requests with a frozen packet may omit nominal angle defaults. Older fixed
    grid studies instead explicitly bind their bucket-facing behavior in the
    contract. Neither path inserts solved query joints into pre-IK features.
    """
    if reason not in {"orientation_not_explicit_in_original_request", "missing_frozen_request_packet"}:
        return None,None
    provenance=item["provenance"]
    path=reader.bind(reader.path(provenance["contract_file"]),provenance["contract_sha256"])
    contract=read(path)
    scene=contract.get("scene")
    if not isinstance(scene,dict):
        return None,None
    actual_scene_hash=sha256(json.dumps(scene,sort_keys=True,allow_nan=False).encode()).hexdigest()
    require(actual_scene_hash==provenance["scene_sha256"],"Nominal recovery scene differs")
    if original_result is not None:
        request,binding=original_result
        request=copy.deepcopy(request)
        if any(float(request.get(key,0.))!=0. for key in ("board_heading_offset_deg","board_twist_deg")):
            return None,None
    else:
        behavior="Rz(atan2(bucket_y-board_y,bucket_x-board_x)) @ diag(1,-1,-1)"
        if contract.get("release_behavior")!=behavior:
            return None,None
        request={key:copy.deepcopy(native[key]) for key in ("id","bucket_xy_m","values")}
        request["intended_route"]="direct"
        binding=dict(file=str(reader.path(item["source"]["file"])),
                     sha256=item["source"]["metadata_sha256"],request_id=item["id"],
                     reconstruction="Original native command metadata; source seeds not available")
    if any(abs(float(v))>1e-10 for v in request.get("impact_offset_m",[])):
        return None,None
    try:
        error=nominal_orientation_error(request,native,scene)
    except (ValueError,KeyError,TypeError):
        return None,None
    if error>=4e-4:
        return None,None
    request.update(board_heading_offset_deg=0.,board_twist_deg=0.)
    proof=dict(kind="verified_nominal_orientation",quaternion_error_rad=error,
               threshold_rad=4e-4,contract_file=str(path),contract_sha256=provenance["contract_sha256"],
               definition="Rz(bucket bearing + pi/2) * Rx(board tilt) * Rz(scene board mount yaw)",
               original_schema=reason,seed_policy="Only explicit original pre-IK seeds; missing stays missing")
    return (request,binding),proof


def build(rescore, neural, output, *, root=Path("/"), seed=SEED, forced_train_positions=(),
          extra_excluded_positions=()):
    rescore, neural, output = map(Path, (rescore, neural, output))
    require(not output.exists(), "Expanded library output must be new")
    audit, manifest = read(rescore/"audit.json"), read(rescore/"manifest.json")
    require(audit.get("status") == "passed" and audit.get("full_checks_verified") is True,
            "A completed independent historical rescore audit is required")
    require(audit["manifest_sha256"] == file_sha(rescore/"manifest.json"), "Rescore manifest changed")
    require(audit["summary_sha256"] == file_sha(rescore/"summary.json"), "Rescore summary changed")
    for name, expected in manifest["artifacts"].items():
        require(file_sha(rescore/name) == expected, "Rescore artifact changed: " + name)
    reader = SourceReader(root)
    records = lines(rescore/"records.jsonl")
    observations, excluded = [], Counter()
    all_positions = {position(p) for p in extra_excluded_positions}
    for item in records:
        original = item["original"]
        if original.get("bucket_xy_m") is not None:
            all_positions.add(position(original["bucket_xy_m"]))
        if item["status"] != "scored":
            excluded["non_native_or_unscored"] += 1
            continue
        if item.get("substeps") != 64:
            excluded["not_native64"] += 1
            continue
        provenance = item["provenance"]
        if (provenance["scene_sha256"] != NOMINAL_SCENE
                or provenance["calibration_fingerprint"] != NOMINAL_CALIBRATION
                or original.get("contact_protocol") != PROTOCOL):
            excluded["different_scene_calibration_or_contact_protocol"] += 1
            continue
        native = reader.source_row(item["source"])
        result, why = reader.lookup(item)
        recovery = None
        if why:
            original_reason=why
            result, recovery = recover_nominal(reader, item, native, result, why)
            if result is None:
                excluded[original_reason] += 1
                continue
        request, packet_binding = result
        for key in ("id", "bucket_xy_m", "values", "native_cache_sha256", "native_cache_row"):
            require(native[key] == original[key], "Raw row differs from authenticated rescore: " + key)
        score = item["new_score"]
        require(score["scoring_version"] == board_first_score.VERSION, "Unexpected scoring version")
        request_features(request)
        # Do not include any unsupported release offset hidden outside five values.
        expected_release = [*request["values"][:2], request["values"][2] + request["values"][4]]
        if math.dist(native["release_ball_center_m"][:2], expected_release[:2]) > 1e-7:
            excluded["unrepresented_noncentral_release"] += 1
            continue
        observations.append(dict(request=request, native=native, provenance=provenance, score=score,
            evidence=dict(id=item["id"], source_study=item["source_study"], source=item["source"],
                          request_packet=packet_binding, nominal_reconstruction=recovery, native_sha256=item["native_sha256"],
                          native_row=item["native_row"], historical_phase=item.get("phase"),
                          capture=score["route_success"], invalid=score["invalid_physics"],
                          route=score["route_class"])))
    rows, candidates = group_native(observations, seed=seed, forced_train_positions=forced_train_positions)
    # Original proposal records add failed and admitted-but-not-released setup
    # evidence. Already released rows were handled above, without duplicating them.
    setup_groups = defaultdict(list)
    neural_manifest = read(reader.bind(neural/"evidence-manifest.json"))
    for name in ("training-dataset.jsonl", "validation-dataset.jsonl"):
        path = reader.bind(neural/name, neural_manifest[name])
        for line, record in enumerate(lines(path), 1):
            request = record["request"]
            all_positions.add(position(request["bucket_xy_m"]))
            if record.get("observation") is not None:
                continue
            proposal = request_features(request)
            key = digest(["setup", proposal])
            setup_groups[key].append(dict(setup=bool(record["setup_ok"]), request=proposal,
                                           evidence=dict(file=str(path), line=line, sha256=reader.inputs[str(path)])))
    for key, attempts in sorted(setup_groups.items()):
        first = attempts[0]
        request = dict(first["request"], id="expanded-setup-"+key)
        rows.append(dict(split=split_for(request["bucket_xy_m"], seed=seed,
                                         forced_train_positions=forced_train_positions),
                         dedup_key=key, request=request,
                         setup_probability=sum(a["setup"] for a in attempts)/len(attempts),
                         capture_probability=None, valid_native_count=0, repeats_count=len(attempts),
                         invalid_native_count=0, capture_count=0, observed_routes={},
                         evidence=[a["evidence"] for a in attempts], pre_ik_feature_variants=1,
                         source_phases={"old_train_or_validation_setup":len(attempts)}))
    rows.sort(key=lambda r:r["dedup_key"])
    splits = {key:[row for row in rows if row["split"]==key] for key in ("train", "validation")}
    require(all(splits.values()), "Both training and validation must have records")
    require(not ({position(r["request"]["bucket_xy_m"]) for r in splits["train"]}
                 & {position(r["request"]["bucket_xy_m"]) for r in splits["validation"]}), "Split leakage")
    validation_positions={position(r["request"]["bucket_xy_m"]) for r in splits["validation"]}
    require(all(position(c["bucket_xy_m"]) not in validation_positions for c in candidates),
            "Validation settings cannot enter the candidate library")
    summary = dict(version=VERSION, scoring_version=board_first_score.VERSION, seed=seed,
        status="built", source_records=len(records), included_native_attempts=len(observations),
        included_capture_attempts=sum(o["score"]["route_success"] for o in observations),
        nominal_reconstructed_attempts=sum(o["evidence"]["nominal_reconstruction"] is not None for o in observations),
        native_settings=sum(r["valid_native_count"]+r["invalid_native_count"] > 0 for r in rows),
        native_repeat_rows_removed=len(observations)-sum(r["valid_native_count"]+r["invalid_native_count"] > 0 for r in rows),
        unique_setting_capture_count=sum(r["capture_count"] > 0 for r in rows),
        conflicting_settings=sum(0 < r["capture_count"] < r["valid_native_count"] for r in rows),
        candidate_settings=len(candidates),
        candidate_observed_routes=dict(Counter(c["observed_route"] for c in candidates)),
        capture_attempt_routes=dict(Counter(o["score"]["route_class"] for o in observations if o["score"]["route_success"])),
        settings_by_observed_route=dict(Counter(route for r in rows for route in r["observed_routes"])),
        forced_train_positions_represented=sum(position(p) in {position(r["request"]["bucket_xy_m"]) for r in splits["train"]} for p in forced_train_positions),
        candidate_positions=len({position(c["bucket_xy_m"]) for c in candidates}),
        all_historical_positions=len(all_positions), exclusions=dict(excluded),
        splits={key:dict(rows=len(items),positions=len({position(r["request"]["bucket_xy_m"]) for r in items}),
                        capture_settings=sum(r["capture_count"]>0 for r in items)) for key,items in splits.items()},
        forced_train_positions=[list(position(p)) for p in forced_train_positions],
        deduplication="Bucket+requested geometry+both actual arm joints+release point+scene/calibration/protocol. Meters/task values rounded8decimals; joints6decimals; orientation7decimals. IDs/repeat count do not define a setting.",
        targets="Each deduplicated setting has one loss weight; capture target is successes / valid native repeats. Invalid outcomes masked. Equal total loss weight per bucket position, applied during fitting.",
        split_policy="Entire bucket coordinate belongs to one split; deterministic80/20 hash except declared fixed-gridTRAIN positions. Historical TEST is explicitly reused as development; new TEST must exclude all historical positions.",
        input_scope="Original pre-IK geometry and source seeds only. Solved query joints excluded from training features, preserved for dedup and candidate provenance only. Candidate ranking uses original pre-IK seeds.",
        limitations="Bounded60-source corpus; nominal commands require measured full-quaternion confirmation and a frozen request or explicit bucket-facing contract. Unsupported command schemas, noncentral releases, other scenes and non64steps excluded. Later contacts accepted byv4; contact presence does not prove a causal rebound.")
    output.mkdir(parents=True)
    for key, items in splits.items():
        (output/(("train" if key=="train" else "validation")+"-dataset.jsonl")).write_text("".join(json.dumps(r,sort_keys=True,allow_nan=False)+"\n" for r in items))
    for name, value in (("candidate-sources.json", candidates),("all-positions.json", sorted(all_positions)),
                        ("library-summary.json", summary)):
        (output/name).write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n")
    inputs=dict(reader.inputs)
    for name in ("audit.json","manifest.json","summary.json","records.jsonl","contract.json"):
        inputs[str(rescore/name)] = file_sha(rescore/name)
    artifacts={p.name:file_sha(p) for p in output.iterdir() if p.is_file()}
    (output/"library-manifest.json").write_text(json.dumps(dict(version=VERSION,inputs=inputs,artifacts=artifacts),sort_keys=True,indent=2)+"\n")
    return summary


def verify_library(path, *, verify_inputs=True):
    """Verify frozen bytes, split separation, ratios and source-candidate linkage."""
    path=Path(path)
    manifest=read(path/"library-manifest.json")
    require(manifest.get("version")==VERSION,"Unexpected library version")
    expected={"train-dataset.jsonl","validation-dataset.jsonl","candidate-sources.json",
              "all-positions.json","library-summary.json"}
    require(set(manifest["artifacts"])==expected,"Library artifact inventory differs")
    for name, checksum in manifest["artifacts"].items():
        require(file_sha(path/name)==checksum,"Library artifact checksum differs: "+name)
    if verify_inputs:
        for name, checksum in manifest["inputs"].items():
            require(file_sha(name)==checksum,"Library source checksum differs: "+name)
    datasets={split:lines(path/(split+"-dataset.jsonl")) for split in ("train","validation")}
    groups={}
    positions={split:set() for split in datasets}
    native_evidence=set()
    for split, rows in datasets.items():
        for row in rows:
            require(row["split"]==split and row["dedup_key"] not in groups,"Duplicate key or incorrect split")
            groups[row["dedup_key"]]=row
            positions[split].add(position(row["request"]["bucket_xy_m"]))
            feature(row["request"])
            require(row["repeats_count"]==len(row["evidence"])>0,"Repeat evidence count differs")
            require(0<=row["setup_probability"]<=1,"Invalid setup target")
            native=[e for e in row["evidence"] if "native_sha256" in e]
            if native:
                require(len(native)==len(row["evidence"]),"Mixed native/setup-only group")
                valid=[e for e in native if not e["invalid"]]
                captures=[e for e in valid if e["capture"]]
                require(row["valid_native_count"]==len(valid),"Valid count differs")
                require(row["invalid_native_count"]==len(native)-len(valid),"Invalid count differs")
                require(row["capture_count"]==len(captures),"Capture count differs")
                expected_target=len(captures)/len(valid) if valid else None
                require(row["capture_probability"]==expected_target,"Repeat target differs")
                require(row["observed_routes"]==dict(Counter(e["route"] for e in captures)),"Observed route count differs")
                require(row["feature_source"] in native,"Canonical feature source lacks native binding")
                for evidence in native:
                    key=(evidence["native_sha256"],evidence["native_row"])
                    require(key not in native_evidence,"Native observation counted more than once")
                    native_evidence.add(key)
            else:
                require(row["valid_native_count"]==0 and row["capture_probability"] is None,
                        "Setup-only data cannot have capture label")
    require(not positions["train"] & positions["validation"],"Train/validation positions overlap")
    all_positions={position(p) for p in read(path/"all-positions.json")}
    require(positions["train"] | positions["validation"] <= all_positions,"Historical positions omitted")
    candidates=read(path/"candidate-sources.json")
    require(len({r["dedup_key"] for r in candidates})==len(candidates),"Candidate settings duplicated")
    expected_candidates={key for key,row in groups.items() if row["split"]=="train" and row["capture_count"]>0}
    require({r["dedup_key"] for r in candidates}==expected_candidates,"Candidate set differs from TRAIN captures")
    for source in candidates:
        row=groups[source["dedup_key"]]
        require(source["split"]=="train" and source["capture_probability"]==row["capture_probability"],
                "Candidate split or capture fraction differs")
        require(source["source_evidence"] in row["evidence"] and source["source_evidence"]["capture"] is True,
                "Candidate lacks demonstrated capture")
        for key in ("values","bucket_xy_m","board_heading_offset_deg","board_twist_deg","seed_q10","seed_q14","intended_route"):
            require(source.get(key)==row["request"].get(key),"Candidate geometry or original seed differs from setting")
    summary=read(path/"library-summary.json")
    require(summary["included_native_attempts"]==len(native_evidence),"Native count differs")
    require(summary["candidate_settings"]==len(candidates),"Candidate count differs")
    for split,rows in datasets.items():
        require(summary["splits"][split]["rows"]==len(rows)
                and summary["splits"][split]["positions"]==len(positions[split]),"Split summary differs")
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rescore",type=Path,required=True)
    parser.add_argument("--neural",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--root",type=Path,default=Path("/"))
    parser.add_argument("--forced-train-positions",type=Path)
    parser.add_argument("--extra-excluded-positions",type=Path)
    args=parser.parse_args()
    print(json.dumps(build(args.rescore,args.neural,args.output,root=args.root,
        forced_train_positions=read(args.forced_train_positions) if args.forced_train_positions else (),
        extra_excluded_positions=read(args.extra_excluded_positions) if args.extra_excluded_positions else ()),indent=2))


if __name__=="__main__":main()
