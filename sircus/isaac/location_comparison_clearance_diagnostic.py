"""Saved-data arithmetic diagnostic; never a replacement for a native audit.

The original score/quality/route helpers and 1e-7 comparison are imported from
SHA-pinned producer source. No model, optimizer or simulator is imported. The
three replay paths isolate world-origin subtraction from scorer input dtype;
none changes, repairs, accepts or republishes the failed original audit.
"""

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import importlib
import io
import json
import math
from pathlib import Path
import platform
import re
import stat
import sys
import time
from types import SimpleNamespace

import numpy as np

SCHEMA = "sircus-location-comparison-clearance-diagnostic-v1"
TOLERANCE = 1e-7  # Original field comparison, including clearance measured in mm.
SOURCE_COMMIT = "300ab5c6d161b51577d37d2a70f7499fc3f782e9"
PATHS = ("auditor_float64", "producer_float32", "producer_float32_then_float64")
FORCES = ("board", "bucket", "table", "floor", "robot", "unattributed")
KEYS = {"time_step", "position_m", "velocity_m_s", "forces_N", "joint_position_rad",
        "joint_velocity_rad_s", "board_position_m", "board_quaternion_wxyz"}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return sha_bytes(json.dumps(value, sort_keys=True, allow_nan=False).encode())


def _sha(value):
    require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value), "Invalid SHA256 pin")
    return value


def _object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON field")
        result[key] = value
    return result


def read_json_bytes(data):
    def nonfinite(_):
        raise ValueError("Nonfinite JSON value")
    return json.loads(data, object_pairs_hook=_object, parse_constant=nonfinite)


def safe_file(root, relative):
    require(type(relative) is str and relative and not Path(relative).is_absolute()
            and Path(relative).as_posix() == relative
            and all(part not in (".", "..") for part in Path(relative).parts), "Unsafe artifact path")
    root = Path(root).resolve(strict=True)
    current = root
    for part in Path(relative).parts:
        current = current / part
        require(not current.is_symlink(), "Linked artifact")
    require(stat.S_ISREG(current.stat().st_mode), "Not a regular artifact")
    require(current.resolve().is_relative_to(root), "Artifact escaped root")
    return current


def pinned_json(path, expected_sha):
    path = Path(path)
    require(not path.is_symlink() and stat.S_ISREG(path.stat().st_mode), "Linked or special pinned input")
    data = path.read_bytes()
    require(sha_bytes(data) == _sha(expected_sha), "Pinned input SHA changed: " + path.name)
    return read_json_bytes(data)


class SealedFiles:
    """Validate only the numerical subset opened here; never claim full inventory audit."""

    def __init__(self, root, inventory):
        self.root, self.files, self.used = Path(root), inventory.get("files"), {}
        require(type(self.files) is dict and inventory.get("file_count") == len(self.files),
                "Invalid sealed inventory count")
        for name, receipt in self.files.items():
            require(type(name) is str and type(receipt) is dict and set(receipt) == {"bytes", "sha256"}
                    and type(receipt["bytes"]) is int and receipt["bytes"] >= 0,
                    "Invalid sealed inventory member")
            _sha(receipt["sha256"])
        require(sum(v["bytes"] for v in self.files.values()) == inventory.get("total_bytes"),
                "Invalid sealed inventory byte total")

    def read(self, name, expected_sha=None):
        require(name in self.files, "Unsealed numerical artifact: " + name)
        path = safe_file(self.root, name)
        data = path.read_bytes()
        receipt = dict(bytes=len(data), sha256=sha_bytes(data))
        require(receipt == self.files[name], "Sealed numerical bytes changed: " + name)
        if expected_sha is not None:
            require(receipt["sha256"] == _sha(expected_sha), "Receipt/inventory SHA differs: " + name)
        self.used[name] = receipt
        return data

    def finish(self):
        for name, expected in self.used.items():
            path = safe_file(self.root, name)
            require(path.stat().st_size == expected["bytes"] and sha_file(path) == expected["sha256"],
                    "Numerical artifact changed during diagnostic: " + name)
        return dict(scope="Only opened numerical artifacts; excluded media and model bytes are not re-audited here",
                    files=len(self.used), bytes=sum(v["bytes"] for v in self.used.values()),
                    files_fingerprint=digest(self.used), before_after_verified=True,
                    complete_inventory_verified=False)


def runtime_info(*, synthetic=False, cross_version_diagnostic=False):
    require(type(synthetic) is bool and type(cross_version_diagnostic) is bool, "Invalid runtime mode")
    require(not (synthetic and cross_version_diagnostic), "Choose one diagnostic exception mode")
    matches = np.__version__ == "1.26.0"
    if not synthetic and not cross_version_diagnostic:
        require(matches, "Native diagnostic requires exact NumPy 1.26.0; use explicit cross-version-diagnostic for exploratory replay")
    return dict(python=platform.python_version(), numpy=np.__version__, platform=platform.platform(),
                runtime_matches_producer=False, full_producer_runtime_verified=False,
                numpy_matches_producer=matches, cross_platform=platform.system() != "Linux",
                cross_version_diagnostic=cross_version_diagnostic, synthetic=synthetic,
                producer_arithmetic_parity_claimed=False,
                warning=("Cross-version replay: scalar promotion can differ; producer_float32 is not established producer parity"
                         if not matches else "CPU reconstruction; platform and tensor/scalar arithmetic are not a full native audit"))


def load_original_helpers(source, source_sha256):
    """Refuse shadowed helpers; run this diagnostic in a fresh Python process."""
    source = Path(source).resolve(strict=True)
    require(type(source_sha256) is dict and bool(source_sha256), "Missing producer source pins")
    for name, expected in source_sha256.items():
        require(sha_file(safe_file(source, name)) == _sha(expected), "Producer source SHA changed: " + name)
    required = ("rl_contract", "entry_quality", "board_route", "continuous_native_audit",
                "farm_contract", "settling_contract", "location_comparison_audit", "camera_options", "dark_stage")
    for name in required:
        require(name + ".py" in source_sha256, "Missing imported helper source pin: " + name)
    for filename in source_sha256:
        if Path(filename).parent == Path(".") and filename.endswith(".py"):
            name = Path(filename).stem
            if name in sys.modules:
                require(Path(sys.modules[name].__file__).resolve() == source / filename,
                        "Producer helper shadowed; use a fresh process: " + name)
    sys.path.insert(0, str(source))
    bytecode_setting = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        modules = {name: importlib.import_module(name) for name in required}
    finally:
        sys.path.pop(0)
        sys.dont_write_bytecode = bytecode_setting
    for name, module in modules.items():
        require(Path(module.__file__).resolve() == source / (name + ".py"), "Producer helper resolved elsewhere")
    return SimpleNamespace(ShotScore=modules["rl_contract"].ShotScore,
        EntryQuality=modules["entry_quality"].EntryQuality, BoardRoute=modules["board_route"].BoardRoute,
        same=modules["continuous_native_audit"].same, audit_trace=modules["location_comparison_audit"].audit_trace,
        core_state=modules["location_comparison_audit"].CORE_STATE,
        preview_origins=modules["camera_options"].preview_origins)


def local_coordinates(world, origins):
    require(world.dtype == np.dtype("float32"), "Saved producer world positions must be float32")
    origins64 = np.asarray(origins, dtype=np.float64)
    require(origins64.shape == world.shape[-2:] and np.isfinite(origins64).all(), "Malformed origins")
    origins32 = origins64.astype(np.float32)
    require(np.array_equal(origins64, origins32.astype(np.float64)), "Origins do not round-trip exactly through float32")
    a = world - origins64
    b = world - origins32  # Same-dtype NumPy subtraction, matching elemental producer rounding.
    require(a.dtype == np.float64 and b.dtype == np.float32, "Unexpected subtraction dtype")
    delta = np.abs(a - b.astype(np.float64))
    return {PATHS[0]: a, PATHS[1]: b, PATHS[2]: b.astype(np.float64)}, dict(
        scope="All recorded steps in this batch, paid cells only; including steps after individual score freeze",
        coordinates_identical=bool(np.array_equal(a, b.astype(np.float64))),
        differing_components=int(np.count_nonzero(delta)), components=int(delta.size),
        maximum_absolute_difference_m=float(delta.max(initial=0)))


def validate_physics(data, packet):
    require(set(data) == KEYS, "Physics columns changed")
    steps = data["time_step"]
    require(steps.ndim == 1 and np.issubdtype(steps.dtype, np.integer), "Invalid physics clock")
    count, length = len(packet["cells"]), len(steps)
    require(1 <= length <= 960 and type(packet["physics"]["samples"]) is int
            and packet["physics"]["samples"] == length, "Invalid physics sample count")
    require(type(packet["release_step"]) is int and type(packet["score_end_step"]) is int
            and np.array_equal(steps, np.arange(packet["release_step"] + 1, packet["score_end_step"] + 1)),
            "Incomplete 240 Hz physics clock coverage")
    for name, tail in (("position_m", (3,)), ("velocity_m_s", (3,)), ("forces_N", (6,)),
                       ("joint_position_rad", (2, 6)), ("joint_velocity_rad_s", (2, 6)),
                       ("board_position_m", (3,)), ("board_quaternion_wxyz", (4,))):
        value = data[name]
        require(value.shape == (length, count, *tail) and np.issubdtype(value.dtype, np.number)
                and not np.issubdtype(value.dtype, np.complexfloating) and np.isfinite(value).all(),
                "Malformed physics array: " + name)
    require((data["forces_N"] >= 0).all(), "Negative force")
    require(np.allclose(np.linalg.norm(data["board_quaternion_wxyz"], axis=2), 1, atol=1e-3, rtol=0),
            "Invalid physics quaternion")


def replay_cell(scene, cell, row, data, coordinates, slot, helpers):
    dt = float(scene["simulation"]["physics_dt_s"])
    require(abs(dt - 1 / 240) < 1e-15, "Changed physics dt")
    bucket = dict(scene["bucket"], center_xy_m=cell["bucket_xy_m"])
    score = helpers.ShotScore(bucket, scene["ball"]["radius_m"])
    quality = helpers.EntryQuality(bucket, scene["ball"]["radius_m"])
    route = helpers.BoardRoute(dt)
    consumed, reason, minimum = 0, "rejected_no_replay", None
    if not row["invalid"]:
        require(data is not None and coordinates is not None, "Released cell has no full-rate physics")
        normal = np.asarray(row.get("scoring_board_normal", row["settled_board_normal"]))
        require(normal.shape == (3,) and np.isfinite(normal).all()
                and abs(np.linalg.norm(normal) - 1) <= 1e-4, "Invalid scoring normal")
        reason = "recording_ended_before_freeze"
        for j in range(len(data["time_step"])):
            p = coordinates[j, slot]
            forces, velocity = data["forces_N"][j, slot], data["velocity_m_s"][j, slot]
            score.step(p, float(forces[0]), dt, float(velocity @ normal))
            before = quality.margin
            quality.step(p, float(forces[1]), score.entered)
            route.step(dict(zip(FORCES, map(float, forces))))
            consumed += 1
            if quality.margin < before:
                minimum = dict(sample_index=j, physics_step=int(data["time_step"][j]),
                               local_position_m=p.tolist(), clearance_mm=float(quality.margin * 1000),
                               position_dtype=str(p.dtype))
            if score.success or p[2] < -.2:
                reason = "capture" if score.success else "fell_below_minus_0_2m"
                break
        if consumed == 960 and reason == "recording_ended_before_freeze":
            reason = "full_960_step_horizon"
    outcome = score.result()
    q = quality.result(outcome["success"])
    result = dict(**outcome, **q, **route.result(outcome["success"], q["quality_score"]))
    return dict(fields=result, consumed_steps=consumed, freeze_reason=reason,
                freeze_rule_satisfied=reason != "recording_ended_before_freeze", minimum_clearance_step=minimum)


def mismatches(recorded, replayed, helpers):
    """Every returned top-level scorer field, with the unchanged original comparison."""
    result = []
    for key, expected in replayed.items():
        if key not in recorded or not helpers.same(recorded[key], expected, TOLERANCE):
            item = dict(field=key, recorded=recorded.get(key), replayed=expected,
                        recorded_field_present=key in recorded)
            if type(recorded.get(key)) in (int, float) and type(expected) in (int, float):
                item["absolute_difference"] = abs(recorded[key] - expected)
            result.append(item)
    return result


def labels(fields, rejected):
    captured = bool(fields["route_success"]) and not rejected
    return dict(outcome="rejected" if rejected else "capture" if captured else "miss",
                clean=bool(captured and fields["route_quality"] == 2))


def diagnose_packet(packet, reservation, terminal, scene, data, helpers):
    """Pure numerical core. Caller binds packet/data SHA and source before using it."""
    paid = len(reservation["cells"])
    require(1 <= paid <= 35 and len(packet.get("cells", [])) == 35
            and len(packet.get("raw_rows", [])) == 35 and len(packet.get("rewards", [])) == 35,
            "Invalid paid/padded coverage")
    require(packet.get("enabled") == [i < paid for i in range(35)]
            and all(type(v) is bool for v in packet["enabled"]), "Paid mask changed")
    require(packet.get("normalized_results") == terminal["rows"] and len(terminal["rows"]) == paid,
            "Terminal/raw normalized join changed")
    require(packet.get("dt_s") == scene["simulation"]["physics_dt_s"], "Packet dt changed")
    for i, target in enumerate(reservation["cells"]):
        cell, raw, saved = packet["cells"][i], packet["raw_rows"][i], terminal["rows"][i]
        require(cell.get("id") == cell.get("request_id") == target["request_id"]
                and all(cell.get(k) == v for k, v in target.items() if k != "id")
                and raw.get("cell_id") == target["request_id"]
                and saved.get("request_id") == target["request_id"]
                and saved.get("target_id") == target["target_id"], "Paid request identity changed")
        require(labels(raw, bool(raw["invalid"])) == {k: saved[k] for k in ("outcome", "clean")}
                and packet["rewards"][i] == saved["reward"]
                and saved["rejection_reason"] == (str(raw["invalid"]) if raw["invalid"] else None),
                "Saved labels/reward do not match raw rows")
    for i in range(paid, 35):
        require(bool(packet["raw_rows"][i]["invalid"]) and packet["cells"][i]["target_id"] == "inactive",
                "Padding is not an inactive rejected slot")
    origins = np.asarray(packet["origins_m"], dtype=np.float64)
    require(origins.shape == (35, 3) and np.isfinite(origins).all()
            and np.array_equal(origins, origins.astype(np.float32).astype(np.float64)),
            "Origins do not round-trip exactly through float32")
    if data is None:
        require(packet["physics"] is None and packet["score_end_step"] == packet["release_step"]
                and all(r["invalid"] for r in packet["raw_rows"]), "Missing physics for released cell")
        paths = dict.fromkeys(PATHS)
    else:
        validate_physics(data, packet)
        paths, _ = local_coordinates(data["position_m"], origins)
    records = []
    for i, target in enumerate(reservation["cells"]):
        row, saved = packet["raw_rows"][i], terminal["rows"][i]
        record = dict(request_id=target["request_id"], target_id=target["target_id"], slot=i,
            batch_id=reservation["batch_id"], model=reservation["model"], phase=reservation["phase"],
            milestone=reservation["milestone"], checkpoint_sha256=packet["proposal"]["checkpoint"]["sha256"],
            saved={k: saved[k] for k in ("outcome", "clean", "rejection_reason")}, paths={})
        if data is not None and not row["invalid"]:
            a, b = paths[PATHS[0]][:, i], paths[PATHS[2]][:, i]
            delta = np.abs(a - b)
            record["coordinate_comparison"] = dict(coordinates_identical=bool(np.array_equal(a, b)),
                differing_components=int(np.count_nonzero(delta)), components=int(delta.size),
                maximum_absolute_difference_m=float(delta.max(initial=0)))
        else:
            record["coordinate_comparison"] = None
        replays = {}
        for name, coordinates in paths.items():
            replay = replay_cell(scene, packet["cells"][i], row, data, coordinates, i, helpers)
            label = labels(replay["fields"], bool(row["invalid"]))
            differences = mismatches(row, replay["fields"], helpers)
            record["paths"][name] = dict(**label, consumed_steps=replay["consumed_steps"],
                freeze_reason=replay["freeze_reason"], freeze_rule_satisfied=replay["freeze_rule_satisfied"],
                mismatches=differences, capture_flip=(label["outcome"] == "capture") != (saved["outcome"] == "capture"),
                clean_flip=label["clean"] != saved["clean"], minimum_clearance_step=replay["minimum_clearance_step"])
            replays[name] = replay["fields"]
        record["path_differences"] = {
            "auditor_vs_producer": mismatches(replays[PATHS[0]], replays[PATHS[1]], helpers),
            "auditor_vs_float32_then_float64": mismatches(replays[PATHS[0]], replays[PATHS[2]], helpers),
            "float32_vs_float32_then_float64": mismatches(replays[PATHS[1]], replays[PATHS[2]], helpers)}
        records.append(record)
    return records


def _summary(records):
    result = dict(requests=len(records), saved=dict(Counter(r["saved"]["outcome"] for r in records)), paths={})
    result["saved"]["clean_captures"] = sum(r["saved"]["clean"] for r in records)
    for path in PATHS:
        selected = [r["paths"][path] for r in records]
        fields = Counter(d["field"] for r in selected for d in r["mismatches"])
        residuals = {}
        for r in selected:
            for d in r["mismatches"]:
                if "absolute_difference" in d:
                    residuals[d["field"]] = max(residuals.get(d["field"], 0), d["absolute_difference"])
        result["paths"][path] = dict(Counter(r["outcome"] for r in selected),
            clean_captures=sum(r["clean"] for r in selected),
            requests_with_field_mismatch=sum(bool(r["mismatches"]) for r in selected),
            mismatched_fields=dict(fields), maximum_absolute_residual_by_field=residuals,
            capture_flips=sum(r["capture_flip"] for r in selected), clean_flips=sum(r["clean_flip"] for r in selected),
            incomplete_freeze_rules=sum(not r["freeze_rule_satisfied"] for r in selected))
    coords = [r["coordinate_comparison"] for r in records if r["coordinate_comparison"] is not None]
    result["coordinates"] = dict(released_requests_measured=len(coords),
        requests_with_different_coordinates=sum(not c["coordinates_identical"] for c in coords),
        differing_components=sum(c["differing_components"] for c in coords),
        components=sum(c["components"] for c in coords),
        maximum_absolute_difference_m=max((c["maximum_absolute_difference_m"] for c in coords), default=0),
        scope="All saved steps for paid released cells, including samples after individual score freeze")
    result["cross_path_field_mismatches"] = {name: dict(Counter(d["field"] for r in records
        for d in r["path_differences"][name])) for name in (
            "auditor_vs_producer", "auditor_vs_float32_then_float64", "float32_vs_float32_then_float64")}
    return result


def summarize(records):
    groups = defaultdict(list)
    for row in records:
        groups[(row["model"], row["phase"], row["milestone"], row["checkpoint_sha256"])].append(row)
    grouped = [dict(model=k[0], phase=k[1], milestone=k[2], checkpoint_sha256=k[3], **_summary(rows))
               for k, rows in sorted(groups.items())]
    final = {model: {phase: _summary([r for r in records if r["model"] == model and r["phase"] == phase])
                    for phase in ("test_broad", "test_local")}
             for model in ("parent", "original_usable", "broader")}
    return dict(overall=_summary(records), grouped=grouped, final_tests=final)


def diagnose_saved(*, output, inventory_path, inventory_sha256, trace_sha256, manifest_path,
                   manifest_sha256, plan_path, plan_sha256, source, synthetic=False,
                   cross_version_diagnostic=False, progress=None):
    """Read-only complete-trace diagnosis. No acceptance path or original artifact writes."""
    started = time.monotonic()
    runtime = runtime_info(synthetic=synthetic, cross_version_diagnostic=cross_version_diagnostic)
    manifest = pinned_json(manifest_path, manifest_sha256)
    inventory = pinned_json(inventory_path, inventory_sha256)
    plan = pinned_json(plan_path, plan_sha256)
    require(manifest.get("source_commit") == SOURCE_COMMIT and manifest.get("synthetic") is synthetic,
            "Wrong original producer manifest")
    require(manifest["files"]["comparison"]["sha256"] == plan_sha256, "Plan is not bound to manifest")
    require(inventory.get("schema") == "sircus-comparison-closed-evidence-inventory-v1"
            and inventory.get("status") == "hash_inventory_only_not_a_scientific_audit"
            and inventory.get("manifest_sha256") == manifest_sha256
            and inventory.get("source_commit") == SOURCE_COMMIT, "Sealed inventory identity changed")
    files = SealedFiles(output, inventory)
    trace = read_json_bytes(files.read("comparison-trace.json", trace_sha256))
    require(trace.get("mode") == ("synthetic" if synthetic else "native"), "Wrong trace mode")
    helpers = load_original_helpers(source, manifest["source_sha256"])
    scene = pinned_json(Path(source) / "scene.json", manifest["files"]["scene"]["sha256"])
    accounting = helpers.audit_trace(plan, trace)
    require(accounting["status"] == ("synthetic_verified" if synthetic else "native_accounting_verified_physics_pending"),
            "Trace accounting is incomplete")
    expected_origins = helpers.preview_origins().tolist()
    records, batches, proposal, reservation = [], 0, None, None
    seen_requests, trials, physics = set(), set(), set()
    for event in trace["events"]:
        payload = event["payload"]
        if event["kind"] == "reserve":
            reservation = payload
        elif event["kind"] == "proposal":
            proposal = payload["packet"]
        elif event["kind"] == "terminal":
            receipt = payload["evidence"]
            expected_file = f"comparison-native/trial-{batches:05d}.json.gz"
            require(receipt.get("file") == expected_file and receipt.get("evidence_root") == "native"
                    and receipt.get("synthetic") is synthetic, "Terminal native evidence identity changed")
            packet = read_json_bytes(gzip.decompress(files.read(expected_file, receipt["sha256"])))
            require(packet.get("schema") == "sircus-location-comparison-native-trial-v1"
                    and packet.get("comparison_fingerprint") == plan["fingerprint"]
                    and packet.get("synthetic") is synthetic and packet.get("evidence_root") == "native"
                    and packet.get("proposal") == proposal and packet.get("filming") is reservation["film"],
                    "Native packet/proposal identity changed")
            require(packet.get("origins_m") == expected_origins, "Producer origins changed")
            checkpoint = proposal["checkpoint"]
            require(checkpoint["file"] in files.files and files.files[checkpoint["file"]]["sha256"] == checkpoint["sha256"]
                    and all(checkpoint["state"][field] == payload["state_before"][field] for field in helpers.core_state),
                    "Checkpoint receipt identity changed")
            require(packet["physics"] == receipt["physics"], "Physics receipt join changed")
            data = None
            if packet["physics"] is not None:
                relative = "native/" + packet["physics"]["file"]
                require(relative not in physics, "Repeated full-rate physics artifact")
                with np.load(io.BytesIO(files.read(relative, packet["physics"]["sha256"])), allow_pickle=False) as arrays:
                    data = {name: arrays[name] for name in arrays.files}
                physics.add(relative)
            batch = diagnose_packet(packet, reservation, payload, scene, data, helpers)
            require(not seen_requests.intersection(r["request_id"] for r in batch), "Repeated paid request")
            seen_requests.update(r["request_id"] for r in batch)
            records.extend(batch)
            trials.add(expected_file)
            batches += 1
            if progress is not None:
                progress(dict(batches=batches, paid_requests=len(records)))
    require(batches == 539 and len(records) == len(seen_requests) == 18520, "Incomplete 539-batch / 18,520-paid-request coverage")
    require(trials == {k for k in files.files if k.startswith("comparison-native/")}, "Trial inventory coverage differs")
    require(physics == {k for k in files.files if re.fullmatch(r"native/trial-[0-9]+\.npz", k)},
            "Full-rate physics inventory coverage differs")
    summary = summarize(records)
    for model in ("parent", "original_usable", "broader"):
        require(summary["final_tests"][model]["test_broad"]["requests"] == 70
                and summary["final_tests"][model]["test_local"]["requests"] == 25, "Final TEST cohort coverage differs")
    numerical = files.finish()
    # Re-read all external bindings and source bytes after the numerical replay.
    for path, expected in ((inventory_path, inventory_sha256), (manifest_path, manifest_sha256), (plan_path, plan_sha256)):
        require(sha_file(path) == expected, "External diagnostic binding changed")
    for name, expected in manifest["source_sha256"].items():
        require(sha_file(safe_file(source, name)) == expected, "Producer source changed during diagnostic")
    result = dict(schema=SCHEMA, status="diagnostic_only_original_audit_failed", synthetic=synthetic,
        original_audit_passed=False, recovery_audit_passed=False, clean_shutdown=False, model_promoted=False,
        runtime=runtime, source_commit=SOURCE_COMMIT, source_sha256=manifest["source_sha256"],
        diagnostic_source_sha256=sha_file(__file__), manifest_sha256=manifest_sha256,
        manifest_fingerprint=manifest["fingerprint"], comparison_sha256=plan_sha256,
        comparison_fingerprint=plan["fingerprint"], inventory_sha256=inventory_sha256,
        native_session_id=inventory["session_id"], trace_sha256=trace_sha256,
        tolerance=TOLERANCE, tolerance_scope="Unchanged original same() field comparison; clearance is in millimetres",
        paths={PATHS[0]: "Saved float32 world minus JSON float64 origin; unchanged scorer receives float64",
               PATHS[1]: "Saved float32 world minus float32 origin; unchanged scorer receives float32; reconstructed arithmetic, not proved producer parity",
               PATHS[2]: "Same float32 subtraction then float64 scorer input; dtype-isolation experiment, not presumed old-NumPy emulation"},
        batches=batches, paid_requests=len(records), padding_excluded=True, numerical_inventory=numerical,
        scope="Full-rate score/quality/route arithmetic only; no policy likelihood, PPO updates, release guards, body/scene physics, camera or shutdown re-audit",
        first_difference_scope="Diagnostic paid-slot order, not the unchanged audit's rejected-first verification order",
        duration_wall_s=time.monotonic() - started, **summary, requests=records)
    result["fingerprint"] = digest(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("output", "inventory-path", "inventory-sha256", "trace-sha256", "manifest-path",
                 "manifest-sha256", "plan-path", "plan-sha256", "source", "report"):
        parser.add_argument("--" + flag, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--synthetic", action="store_true")
    mode.add_argument("--cross-version-diagnostic", action="store_true")
    args = vars(parser.parse_args())
    report = Path(args.pop("report"))
    require(not report.exists() and not report.is_symlink(), "Diagnostic report must be fresh")
    for protected in (args["output"], args["source"]):
        require(not report.resolve().is_relative_to(Path(protected).resolve()), "Report cannot modify evidence or producer source")
    result = diagnose_saved(**args, progress=lambda p: print(json.dumps(p), flush=True) if p["batches"] % 25 == 0 else None)
    with report.open("x") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(status=result["status"], report_sha256=sha_file(report), paid_requests=result["paid_requests"])))


if __name__ == "__main__":
    main()
