"""Durable simulation evidence plus disposable Viam Data sync copies.

Viam's additional_sync_paths removes files after upload. Only sync-outbox is
watched; archive is the authoritative local evidence and is never removed here.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import threading
from datetime import datetime, timezone
import uuid

ROOT = Path(__file__).resolve().parent
SCHEMA = "sircus-two-arm-trial-v1"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def source_hashes():
    names = ("physics.py", "bucket_geometry.py", "bucket_clearance.py", "training_loop.py", "kinematics.py", "two_arm_kinematics.py",
             "models/xarm850_kinematics.json")
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in names if (ROOT / name).is_file()}


def fingerprint(config):
    value = {"config": config, "source_hashes": source_hashes()}
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                    allow_nan=False).encode()).hexdigest()


def _json_bytes(payload):
    return (json.dumps(payload, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode()


def _atomic_file(path, raw, staging):
    """Publish complete files; temporary bytes never enter the watched folder."""
    fd, temp = tempfile.mkstemp(prefix="record-", suffix=".tmp", dir=staging)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


class TrialStore:
    def __init__(self, config, root=None, session_id=None, purpose="training", metadata=None):
        self.config = copy.deepcopy(config)
        self.root = Path(root) if root is not None else ROOT / "data"
        self.session_id = session_id or (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid.uuid4().hex[:8])
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", self.session_id):
            raise ValueError("Invalid session ID")
        self.archive = self.root / "archive" / self.session_id
        self.outbox = self.root / "sync-outbox"
        self.staging = self.root / ".staging"
        self.archive.mkdir(parents=True, exist_ok=False)
        self.outbox.mkdir(parents=True, exist_ok=True)
        self.staging.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.fingerprint = fingerprint(self.config)
        self.sources = source_hashes()
        self.counts = {}
        self._publish("session", {"purpose": purpose, "metadata": metadata or {},
                                 "scene_config": self.config, "source_hashes": self.sources})

    def _publish(self, kind, payload, suffix=None):
        if not re.fullmatch(r"[a-z0-9_-]+", kind):
            raise ValueError("Invalid record kind")
        name = kind + ("-" + suffix if suffix is not None else "")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError("Invalid record name")
        record = {"schema": SCHEMA, "record_type": kind,
                  "session_id": self.session_id, "record_id": self.session_id + "-" + name,
                  "created_utc": utc_now(), "config_fingerprint": self.fingerprint,
                  "mode": "simulation", "physically_verified": False,
                  "drop_robot_id": 10, "board_robot_id": 14,
                  "data_origin": "synthetic_compliant_contact_model",
                  **copy.deepcopy(payload)}
        raw = _json_bytes(record)
        archive_file = self.archive / (name + ".json")
        outbox_file = self.outbox / (record["record_id"] + ".json")
        with self.lock:
            if archive_file.exists():
                raise FileExistsError("Evidence is immutable: " + str(archive_file))
            _atomic_file(archive_file, raw, self.staging)
            _atomic_file(outbox_file, raw, self.staging)
            self.counts[kind] = self.counts.get(kind, 0) + 1
        return record

    def record_trial(self, phase, trial, action, seed, result, feasibility=None, scene_config=None):
        if phase not in ("train", "test", "native", "manual"):
            raise ValueError("Invalid trial phase")
        if isinstance(trial, bool) or not isinstance(trial, int) or trial < 1:
            raise ValueError("trial must be a positive integer")
        evaluation_scene = self.config if scene_config is None else scene_config
        impact = result.get("impact_uv_mm")
        size = evaluation_scene["board"]["size_mm"]
        normalized = None if impact is None else max(
            abs(impact[0]) / (size[0] / 2), abs(impact[1]) / (size[1] / 2))
        region = ("no_contact" if normalized is None else
                  "center" if normalized <= .35 else "edge" if normalized >= .7 else "middle")
        telemetry_names = ("peak_board_force_n", "max_support_displacement_mm",
                           "max_support_tilt_deg", "max_penetration_mm", "incoming_mm_s",
                           "outgoing_mm_s", "contacts", "duration_s", "initial_energy_j",
                           "max_energy_j", "final_speed_mm_s", "model_limit")
        feasibility_summary = copy.deepcopy(feasibility or {})
        for key in ("arm_sequence", "frames", "board_arm_sequence"):
            feasibility_summary.pop(key, None)
        payload = {"phase": phase, "trial": trial, "action": action, "seed": seed,
                   "scene_config": evaluation_scene, "source_hashes": self.sources,
                   "evaluation_config_fingerprint": fingerprint(evaluation_scene),
                   "reward": result.get("reward"), "retained": bool(result.get("retained", False)),
                   "outcome": result.get("outcome"), "impact_uv_mm": impact,
                   "impact_region": region, "normalized_impact_radius": normalized,
                   "landing_xy_mm": result.get("landing_xy_mm"),
                   "board_pose": {"center_mm": evaluation_scene["board"]["center_mm"],
                                  "rpy_deg": evaluation_scene["board"]["rpy_deg"]},
                   "feasibility": feasibility_summary,
                   "telemetry": {key: result.get(key) for key in telemetry_names},
                   "result": result,
                   "valid_for_training": bool(result.get("valid_for_training", phase != "native" or result.get("native_replay_verified", False))),
                   "native_replay_verified": bool(result.get("native_replay_verified", False)),
                   "validation_scope": (("native Viam motion replay; modeled contacts" if result.get("native_replay_verified") else "native execution failed; predicted contacts not validated")
                                        if phase == "native" else
                                        "synthetic contact; kinematic endpoint checks; no native motion path check")}
        return self._publish("trial", payload, f"{phase}-{trial:04d}")

    def save_policy(self, policy):
        return self._publish("policy", {"policy": policy})

    def save_summary(self, summary):
        return self._publish("summary", {"summary": summary})

    def status(self):
        pending = len(list(self.outbox.glob(self.session_id + "-*.json")))
        return {"session_id": self.session_id, "archive_dir": str(self.archive),
                "sync_outbox_dir": str(self.outbox), "records_written": dict(self.counts),
                "pending_sync_files": pending,
                "cloud_upload_verified": False,
                "sync_note": "Viam deletes outbox copies after upload; verify DATA > Files for cloud evidence."}
