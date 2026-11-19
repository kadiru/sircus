"""Read-only native TRAIN evidence audit, separate from producer and learning.

Replays the request ledger, binds each durable dispatch to its exact reserved
actions, and independently rechecks saved guards and physics scores. No policy,
optimizer, simulator, subprocess or GPU backend is constructed. A native pass
does not certify learning arithmetic, fresh assessment, or physical execution.
"""

import copy
import gzip
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from candidate_continuation_audit import _chain
from candidate_continuation_journal import read_events
from candidate_continuation_ledger import Ledger, Outcome, canonical_hash
from candidate_continuation_native import (PACKET_SCHEMA, make_binding,
    prepare_packet, _frozen_identity)
from candidate_continuation_native_guards import verify_guards
from candidate_continuation_native_score import verify_score
from candidate_native_audit import verify_controls
from prepare_assets import chain

SCHEMA = "sircus-candidate-continuation-native-audit-v1"
MAX_PACKET_BYTES = 64 * 1024**2
RESULT_FIELDS = {"request_id", "released", "decline_reason", "source_sha256",
                 "return", "captured", "clean"}
PADDING_VIEW_SCHEMA = "sircus-native-train-padding-guard-view-v1"


def require(value, message):
    if not value:
        raise ValueError(message)


def _json(raw):
    def pairs(items):
        value = {}
        for key, child in items:
            require(key not in value, "Duplicate native JSON key")
            value[key] = child
        return value
    def invalid(_):
        raise ValueError("Nonfinite native JSON")
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


def _file(root, name):
    require(type(name) is str and Path(name).name == name and name not in ("", ".", ".."),
            "Native evidence requires a plain filename")
    path = root/name
    require(path.is_file() and not path.is_symlink() and path.resolve().parent == root.resolve(),
            "Missing, linked or escaped native evidence: " + name)
    return path


def _digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def _input_bytes(path, expected, label):
    raw = Path(path).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected, "Native asset, pose or camera bytes changed: " + label)
    return raw


def _packet(native, stem):
    path = _file(native, stem+".json.gz")
    receipt = _json(_file(native, stem+".receipt.json").read_bytes())
    require(set(receipt) == {"file", "sha256", "bytes", "uncompressed_sha256"}
        and receipt["file"] == path.name and type(receipt["bytes"]) is int
        and receipt["bytes"] == path.stat().st_size and receipt["sha256"] == _digest(path),
        "Native packet byte receipt changed")
    with gzip.open(path, "rb") as stream:
        raw = stream.read(MAX_PACKET_BYTES+1)
    require(len(raw) <= MAX_PACKET_BYTES, "Native packet exceeds bounded audit size")
    require(hashlib.sha256(raw).hexdigest() == receipt["uncompressed_sha256"],
            "Native uncompressed receipt changed")
    return _json(raw), receipt["sha256"]


def _padding_guard_view(packet):
    """Strictly adapt only unpaid native padding in a private guard copy.

    The original ParallelShots boundary emits no_selected_setting after prepare
    records inactive_split. This reader never changes a paid row or the raw
    packet passed to the unchanged scorer. Its caller first authenticates bytes,
    reservation, slot mapping and any saved physics file.
    """
    names=("cells","actions","choices","enabled","rows","proposal_ik_seeds",
           "preflight_records","rewards","reward_features","trajectory_m")
    require(all(type(packet.get(name)) is list and len(packet[name]) == 35 for name in names),
            "Native padding requires complete 35-slot evidence")
    identifiers=packet.get("request_ids")
    require(type(identifiers) is list and 1 <= len(identifiers) <= 35
        and all(type(value) is str and value for value in identifiers)
        and len(set(identifiers)) == len(identifiers),"Native paid request prefix changed")
    paid=len(identifiers)
    require(all(type(value) is bool for value in packet["enabled"])
        and packet["enabled"] == [i < paid for i in range(35)],"Native padding differs from paid prefix")
    padding=list(range(paid,35))
    for i in padding:
        row,cell=packet["rows"][i],packet["cells"][i]
        require(type(row) is dict and type(cell) is dict and row.get("invalid") == "no_selected_setting"
            and row.get("cell_id") == cell.get("id") and row.get("split") == cell.get("split") == "inactive"
            and row.get("request_id") is None and packet["choices"][i] is None
            and packet["proposal_ik_seeds"][i] is None and packet["preflight_records"][i] is None
            and type(row.get("qualification")) is dict and row["qualification"].get("accepted") is False
            and row["qualification"] == dict(accepted=False,reason="inactive_split")
            and row.get("path_proxy") is None and row.get("path_rejection") is None
            and packet["rewards"][i] is None and packet["reward_features"][i] is None
            and row.get("success") is False and row.get("route_success") is False
            and type(row.get("route_quality")) is int and row["route_quality"] == 0
            and packet["trajectory_m"][i] == [],
            "Unpaid native padding was executed, relabelled, scored or omitted")
    guarded=copy.deepcopy(packet)
    for i in padding:guarded["rows"][i]["invalid"]="inactive_split"
    proof=dict(schema=PADDING_VIEW_SCHEMA,paid_requests=paid,unpaid_slots=35-paid,
        normalized_slots=padding,raw_label="no_selected_setting",guard_view_label="inactive_split",
        paid_rows_unchanged=guarded["rows"][:paid] == packet["rows"][:paid],
        scope="Unpaid parked slots only; raw packet, actions, original scoring and paid counts stay unchanged.")
    require(proof["paid_rows_unchanged"],"Padding adapter changed a paid native row")
    return guarded,proof


def audit_native_train(directory, *, expected_head_sha256, expected_bindings,
                       expected_native_binding, scene, cells, supports,
                       prepared_records, assets_paths, poses_path, camera_path,
                       parent_checkpoint_path=None):
    """Audit complete TRAIN evidence against externally trusted input identities.

    Interrupted, missing or extra evidence fails closed; reserved requests cannot
    be refunded. Run ``candidate_continuation_verify.verify_run`` separately for
    checkpoint/learning integrity. This function never opens held-out files and
    cannot promote a model or authorize execution.
    """
    directory = Path(directory)
    native = directory/"native"
    require(native.is_dir() and not native.is_symlink(), "Missing native evidence directory")
    journal = directory/"journal"
    require(journal.is_dir() and not journal.is_symlink()
        and all(not p.is_symlink() and p.is_file() for p in journal.iterdir()),
        "Missing or linked journal evidence")
    events = read_events(journal)
    _chain(events, expected_head_sha256)
    require(events[0]["kind"] == "start" and events[-1]["kind"] == "final"
        and events[0]["payload"].get("bindings") == expected_bindings
        and type(events[0]["payload"].get("synthetic")) is bool,
        "Incomplete native TRAIN history or changed external binding")
    synthetic = events[0]["payload"]["synthetic"]
    b = expected_native_binding
    reconstructed = make_binding(scene=scene, geometry_fingerprint=b["geometry_fingerprint"],
        camera_sha256=b["camera_sha256"], assets_sha256=b["assets_sha256"],
        poses_sha256=b["poses_sha256"], cells=cells, canonical_home_rad=b["canonical_home_rad"],
        supports=supports, prepared_records=prepared_records,
        startup_control_sha256=b["startup_control_sha256"])
    require(b == reconstructed and expected_bindings.get("source_sha256") == b["source_sha256"]
        and expected_bindings.get("support_sha256") == b["support_sha256"],
        "Native source, controller, preparation or input identities changed")
    if not synthetic:
        require(all(b.get(k) == v for k,v in _frozen_identity().items()),
                "Original frozen scene/assets/camera identities changed")
    require(set(assets_paths) == set(b["assets_sha256"]) == {"Arm10", "Arm14"}, "Native asset names changed")
    links = {name:chain(ET.fromstring(_input_bytes(path, b["assets_sha256"][name], name)))
             for name,path in assets_paths.items()}
    poses = _json(_input_bytes(poses_path, b["poses_sha256"], "poses"))
    _input_bytes(camera_path, b["camera_sha256"], "camera")
    require(b["canonical_home_rad"] == {"Arm10":poses["dropper_joint_rad"], "Arm14":poses["board_joint_rad"]},
            "Bound canonical home differs from verified pose bytes")
    parent_verified = parent_checkpoint_path is not None
    require(synthetic or parent_verified, "Original parent checkpoint bytes required")
    if parent_verified:
        require(_digest(Path(parent_checkpoint_path)) == b["parent_checkpoint_sha256"],
                "Original parent checkpoint bytes changed")
    for name in ("probes.json", "rim-rays.json"):
        _file(directory, name)
    controls = verify_controls(directory, scene, b["startup_control_sha256"])
    ledger = Ledger.from_dict(events[0]["payload"]["ledger"])
    require(not ledger.batches and ledger.phase == "train" and ledger.stopped_reason is None
        and ledger.train_context_ids == tuple(s.context_id for s in supports)
        and (synthetic or ledger.bindings.complete), "Native TRAIN must start with the bound empty ledger")
    require(ledger.bindings.parent_checkpoint_sha256 == b["parent_checkpoint_sha256"]
        and ledger.bindings.train_support_sha256 == b["support_sha256"]
        and all(expected_bindings.get(key) == value for key,value in asdict(ledger.bindings).items()
            if value is not None and key not in ("parent_checkpoint_sha256", "train_support_sha256"))
        and (ledger.bindings.producer_source_sha256 is None
            or ledger.bindings.producer_source_sha256 == b["source_sha256"]),
        "Native ledger source/support bindings differ from external inputs")
    known, expected_files, summaries = {}, set(), []
    seen_requests = set()
    costs = dict(wall_s=0., process_cpu_s=0.)
    for event in events[1:]:
        kind, payload = event["kind"], event["payload"]
        if kind == "reserve":
            batch = ledger.next_batch()
            require(batch is not None and ledger.pending is None, "Native reservation after cap or before outcomes")
            ledger = ledger.reserve_batch(batch)
            require(payload["ledger"] == ledger.to_dict(), "Native reservation ledger changed")
            mapped = prepare_packet(payload["choices"], ledger, cells, supports, prepared_records, cells)
            stem = f"train-{batch.batch_index:06d}"
            packet, packet_sha = _packet(native, stem)
            expected_files.update(stem+suffix for suffix in (".dispatch.json", ".json.gz", ".receipt.json"))
            claim = _json(_file(native, stem+".dispatch.json").read_bytes())
            require(claim == dict(reservation_sha256=event["fingerprint"],
                ledger_fingerprint=ledger.to_dict()["fingerprint"], request_ids=mapped["request_ids"],
                native_binding_fingerprint=b["fingerprint"], status="claimed_before_native_call"),
                "Native dispatch is not the exact durable reservation")
            require(packet.get("schema") == PACKET_SCHEMA and packet.get("mode") == "train"
                and type(packet.get("batch")) is int and packet["batch"] == batch.batch_index
                and packet.get("synthetic") is synthetic and packet.get("native_binding") == b
                and packet.get("reservation_sha256") == event["fingerprint"]
                and packet.get("policy_weight_hash") == payload["policy_weight_hash"]
                and packet.get("setup_controller") == b["setup_controller"]
                and packet.get("startup_controls") == controls, "Native packet reservation/source identity changed")
            fields = dict(cells="cells", actions="actions", choices="selected", enabled="enabled",
                          request_ids="request_ids", proposal_ik_seeds="ik_seeds")
            require(all(packet.get(key) == mapped[value] for key,value in fields.items()),
                    "Native candidate/action/coordinate/padding differs from reserved support")
            physics = packet.get("physics")
            if physics is not None:
                require(physics.get("file") == stem+".npz", "Native physics filename changed")
                require(_digest(_file(native, physics["file"])) == physics["sha256"],
                        "Native physics bytes changed")
                expected_files.add(physics["file"])
            raw_fingerprint=canonical_hash(packet)
            guarded,padding_proof = _padding_guard_view(packet)
            guards = verify_guards(guarded, scene, packet["cells"], links, packet["proposal_ik_seeds"])
            scored = verify_score(native, packet, scene)
            require(canonical_hash(packet) == raw_fingerprint,"Native padding audit mutated the raw packet")
            require(len(scored["terminal_results"]) == len(batch.requests), "Scorer omitted reserved requests")
            for row in scored["terminal_results"]:
                identifier = row["request_id"]
                require(identifier not in known, "Native request repeated across packets")
                known[identifier] = dict(row, source_sha256=packet_sha)
            for field in costs:
                value = packet[field]
                require(type(value) in (int,float) and math.isfinite(value) and value >= 0,
                        "Invalid native process cost")
                costs[field] += value
            summaries.append(dict(batch=batch.batch_index, packet_sha256=packet_sha,
                guards=guards, scores={k:v for k,v in scored.items() if k != "terminal_results"},
                padding_guard_view=padding_proof))
        elif kind == "outcomes":
            rows = payload["results"]
            require(type(rows) is list and bool(rows), "Empty native outcome prefix")
            outcomes = []
            for row in rows:
                require(type(row) is dict and set(row) == RESULT_FIELDS, "Native terminal schema changed")
                identifier = row["request_id"]
                require(identifier in known and identifier not in seen_requests and row == known[identifier],
                        "Learning terminal reward/labels/source differ from independent native audit")
                outcomes.append(Outcome(identifier, row["released"], row["decline_reason"], row["source_sha256"]))
                seen_requests.add(identifier)
            ledger = ledger.record_outcomes(tuple(outcomes))
            require(payload["ledger"] == ledger.to_dict(), "Native outcome ledger changed")
        elif kind == "final":
            require(event is events[-1] and payload["ledger"] == ledger.to_dict(),
                    "Native final ledger changed")
        else:
            require(kind in ("imitation_update", "ppo_start", "ppo_update"),
                    "Failed or unexpected native history; no promotion")
    require(known and set(known) == seen_requests and ledger.pending is None and ledger.next_batch() is None,
            "Incomplete native TRAIN coverage or budget not reached; no promotion")
    require({p.name for p in native.iterdir()} == expected_files
        and all(not p.is_symlink() and p.is_file() for p in native.iterdir()),
        "Unlisted, partial or repeated native evidence")
    captures = sum(r["captured"] for r in known.values())
    clean = sum(r["clean"] for r in known.values())
    released = sum(r["released"] for r in known.values())
    result = dict(schema=SCHEMA, status="passed", synthetic=synthetic,
        journal_head_sha256=expected_head_sha256, native_binding_fingerprint=b["fingerprint"],
        raw_assets_poses_camera_verified=True, parent_checkpoint_bytes_verified=parent_verified,
        requests=len(known), releases=released, rejections=len(known)-released,
        captures=captures, clean_captures=clean, batches=summaries, ledger=ledger.summary(),
        native_batch_costs=costs, startup_controls=controls,
        auditor_source_sha256=_digest(Path(__file__)),
        padding_guard_view=dict(schema=PADDING_VIEW_SCHEMA,
            normalized_unpaid_slots=sum(row["padding_guard_view"]["unpaid_slots"] for row in summaries),
            raw_label="no_selected_setting",guard_view_label="inactive_split",
            raw_packets_unchanged=True,paid_rows_unchanged=True,
            scope="Strict read-only guard views after byte/reservation/physics joins; no paid result or saved evidence is relabelled."),
        learning_checkpoint_audit_required=True, assessment_audit_implemented=False,
        startup_schedule_counts_verified=False, model_promoted=False,
        execution_authorized=False, release_authorized=False,
        scope="Native TRAIN ledger/guards/physics/reward only; costs are sums of per-batch producer wall/CPU, not total startup, learning, GPU or audit cost. No held-out results or optimizer arithmetic certified.")
    result["fingerprint"] = canonical_hash(result)
    return result
