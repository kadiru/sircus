"""Read-only exact shallow-board candidate evidence; never starts a simulator."""
import argparse
import copy
import datetime
import hashlib
import json
import math
from pathlib import Path
import time

from .prepare_expanded_report import optional_json, select
from .prepare_lowboard_report import compact as lowboard_compact
from .prepare_neighborhood_report import TERMINAL

KEY = "shallow_evaluation"
VERSION = "sircus-shallow-board-candidate-v1"


def full_audit_verified(raw, audit):
    checks, count = audit.get("checks"), audit.get("checks_count")
    fingerprint = raw.get("contract_fingerprint")
    return bool(raw.get("version") == VERSION and raw.get("status") == "completed" and raw.get("phase") == "completed" and
        isinstance(fingerprint, str) and fingerprint and audit.get("status") == "passed" and
        audit.get("contract_fingerprint") == fingerprint and not audit.get("failures") and
        isinstance(checks, list) and checks and isinstance(count, int) and not isinstance(count, bool) and count > 0 and
        count == len(checks) and all(isinstance(c, dict) and c.get("passed") is True for c in checks))


def compact(raw):
    value = lowboard_compact(raw)
    if value is None:
        return None
    value["settings"]["candidate"] = select(raw.get("settings", {}).get("candidate", {}), (
        "board_height_m", "tilt_deg", "gap_m", "release_center_height_m", "ball_radius_m", "release_headroom_m", "gap_definition", "input_precision", "scope"))
    value["active_policy"] = "Read-only exact-candidate feasibility and engineering repeatability; parent studies and active solver routes remain unchanged"
    return value


def prepare(base_display, study, output):
    base_display, study, output = map(Path, (base_display, study, output))
    if output.resolve() == base_display.resolve() or output.resolve().is_relative_to(study.resolve()):
        raise ValueError("Write a separate derived report outside original evidence")
    base = optional_json(base_display)
    if not isinstance(base, dict):
        raise ValueError("A valid existing base display report is required")
    result = copy.deepcopy(base)
    raw = optional_json(study / "report.json")
    value = compact(raw)
    result[KEY] = value
    if value is not None:
        audit_path = study / "audit-shallow.json"
        audit = optional_json(audit_path)
        audit = audit if isinstance(audit, dict) else {}
        value["audit"] = select(audit, ("status", "contract_fingerprint", "checks_count", "reason", "limitations"))
        value["audit"]["full_checks_verified"] = full_audit_verified(raw, audit)
        if audit_path.is_file():
            value["audit"]["source_sha256"] = hashlib.sha256(audit_path.read_bytes()).hexdigest()
        if audit.get("status") == "passed" and not value["audit"]["full_checks_verified"]:
            value["audit"]["verification_reason"] = "Completed exact-version evidence and a nonempty matching strictly passed full audit are required"
        value["display_updated_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(payload)
    temporary.replace(output)
    return result


def watch(base_display, study, output, seconds=0):
    if not math.isfinite(seconds) or not 0 <= seconds <= 30000:
        raise ValueError("Bounded watch must be 0–30,000 seconds")
    started = time.monotonic()
    while True:
        result = prepare(base_display, study, output)
        value = result[KEY]
        if value and value.get("status") in TERMINAL:
            return result
        remaining = seconds - (time.monotonic() - started)
        if remaining <= 0:
            return result
        time.sleep(min(5, remaining))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base-display", "study", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--watch-seconds", type=float, default=0)
    args = parser.parse_args()
    watch(args.base_display, args.study, args.output, args.watch_seconds)
