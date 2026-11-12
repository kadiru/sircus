"""Restore scheduler state from a stopped grid study's copied evidence."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .benchmark_verify import EvidenceError, contained, digest, read, sha
from .table_training_study import control_gate


PHASES = ("controls_pre", "search", "controls_post")


def require(condition, message):
    if not condition:
        raise EvidenceError(message)


def ledger(path):
    if path.exists():
        with path.open() as stream:
            for line in stream:
                require(line.endswith("\n"), f"Incomplete ledger row in {path.name}")
                yield json.loads(line)


def restore_preparations(output, summaries, pending, generators, seen, dry_preparations, geometry_key):
    drafts = []
    draft_cells = set()
    for path in sorted((output / "preparations").glob("*.json")):
        packet = read(path)
        proposals = packet["proposals"]
        require(isinstance(proposals, list) and proposals, f"Empty preparation {path.name}")
        cell = proposals[0]["context_id"]
        require(cell not in draft_cells, f"A draft is followed by another preparation at {cell}")
        require(path.name == f"batch-{proposals[0]['id']}.json", f"Unexpected preparation filename {path.name}")
        summary = summaries[cell]
        warm = packet["warm_start"]
        require(type(warm) is bool and warm == (dry_preparations[cell] < 2),
                f"Warm-start sequence changed for {path.name}")
        for proposal in proposals:
            index = summary["attempts"]
            generated = dict(next(generators[cell]),
                             id=f"parallel-grid-{cell[-5:]}-{index:06d}",
                             phase="search", role="development", context_id=cell,
                             method="parallel_capture_grid")
            require(digest(proposal) == digest(generated), f"Proposal stream differs at {proposal['id']}")
            key = geometry_key(proposal)
            require(key not in seen[cell], f"Duplicate historical or prepared geometry at {proposal['id']}")
            seen[cell].add(key)
            summary["attempts"] += 1

        if "records" not in packet or "admitted" not in packet:
            require(set(packet) == {"proposals", "warm_start"}, f"Ambiguous incomplete preparation {path.name}")
            drafts.append((path, proposals, warm))
            draft_cells.add(cell)
            continue

        records, admitted = packet["records"], packet["admitted"]
        require(len(records) == len(proposals), f"Preparation count mismatch in {path.name}")
        expected_branch_ids = []
        selected = []
        by_id = {request["id"]: request for request in admitted}
        require(len(by_id) == len(admitted), f"Duplicate prepared request in {path.name}")
        for proposal, record in zip(proposals, records, strict=True):
            identifier = proposal["id"]
            require((record["proposal_id"], record["proposal_sha256"]) == (identifier, digest(proposal)),
                    f"Preparation proof does not bind {identifier}")
            branch_ids = record["admitted_request_ids"]
            require(branch_ids == [f"{identifier}-b{i}" for i in range(record["admitted"])],
                    f"Noncontiguous prepared branches for {identifier}")
            expected_branch_ids.extend(branch_ids)
            for branch_id in branch_ids:
                request = by_id[branch_id]
                require(request["proposal_id"] == identifier and request["native_prepared"] is True
                        and all(request[field] == proposal[field] for field in
                                ("context_id", "bucket_xy_m", "values", "board_heading_offset_deg", "board_twist_deg")),
                        f"Prepared request changed {identifier}")
                for path_field, sha_field in (
                    ("context_resolution_file", "context_resolution_sha256"),
                    ("prepared_connection_file", "prepared_connection_sha256"),
                ):
                    proof_path = contained(output / "native-heading", request[path_field])
                    require(sha(proof_path) == request[sha_field], f"Changed prepared proof for {branch_id}")
            if branch_ids:
                selected.append(by_id[branch_ids[0]])
            else:
                summary["preparation_declines"] += 1
        require(list(by_id) == expected_branch_ids, f"Unaccounted prepared branches in {path.name}")
        pending[cell].extend(selected)
        dry_preparations[cell] = 0 if selected else dry_preparations[cell] + 1
    return drafts


def verify_archive(output, directory, packet_rows, index):
    name = f"native-{index:04d}.npz"
    path = output / directory / name
    checksum = sha(path)
    ids = [row["id"] for row in packet_rows]
    with np.load(path, allow_pickle=False) as archive:
        require(list(archive["record_ids"]) == ids, f"Native archive order differs in {directory}/{name}")
    for column, row in enumerate(packet_rows):
        require((row["adapter_directory"], row["native_cache_file"], row["native_cache_row"],
                 row["native_cache_sha256"]) == (directory, name, column, checksum),
                f"Native evidence differs for {row['id']}")
    return name


def restore_releases(output, report, summaries, outcomes, pending, order, native_batch):
    rows = iter(ledger(output / "rows.jsonl"))
    timings = iter(ledger(output / "timing.jsonl"))
    controls = {phase: [] for phase in ("controls_pre", "controls_post")}
    native_indices = {"native-heading": 0, "native-controls": 0}
    archived_ids = {directory: [] for directory in native_indices}
    committed_archives = {directory: set() for directory in native_indices}
    observed_ids = set()
    phase_index = 0
    next_packet = 0
    for packet_index, path in enumerate(sorted((output / "requests").glob("*.json"))):
        require(path.name == f"batch-{packet_index:05d}.json", f"Noncontiguous request packet {path.name}")
        packet = read(path)
        phase, requests = packet["phase"], packet["rows"]
        current_phase = PHASES.index(phase)
        require(current_phase >= phase_index, f"Out-of-order phase {phase}")
        phase_index = current_phase
        require(0 < len(requests) <= native_batch, f"Invalid native packet size in {path.name}")
        directory = "native-heading" if phase == "search" else "native-controls"
        relative = str(path.relative_to(output))
        packet_hash = sha(path)
        measured = []
        for request in requests:
            row = next(rows, None)
            name = request["id"]
            require(row is not None, f"Missing observation for {name}")
            require(name not in observed_ids and row["id"] == name, f"Duplicate or out-of-order observation {name}")
            observed_ids.add(name)
            require((row["phase"], row["request_packet"], row["request_packet_sha256"], row["native_release"])
                    == (phase, relative, packet_hash, True), f"Observation does not bind packet for {name}")
            require(all(row[field] == request[field] for field in ("context_id", "bucket_xy_m", "values")),
                    f"Observed action differs for {name}")
            if phase == "search":
                cell = request["context_id"]
                require(pending[cell] and pending[cell][0] == request,
                        f"Frozen request is not the next prepared branch for {name}")
                pending[cell].popleft()
            measured.append(row)
        timing = next(timings, None)
        require(timing is not None and timing["packet"] == relative and timing["phase"] == phase
                and timing["timing"]["native_releases"] == len(measured),
                f"Incomplete or mismatched packet timing for {path.name}")
        archive = verify_archive(output, directory, measured, native_indices[directory])
        committed_archives[directory].add(archive)
        native_indices[directory] += 1
        archived_ids[directory].extend(row["id"] for row in measured)
        if phase == "search":
            packet_cells = {row["context_id"] for row in measured}
            report["search_packets"] += 1
            report["mixed_packets"] += int(len(packet_cells) > 1)
            report["max_cells_in_packet"] = max(report["max_cells_in_packet"], len(packet_cells))
            for row in measured:
                cell = row["context_id"]
                outcomes[cell][row["experimental_route"]["route_class"]] += 1
                summaries[cell]["search_releases"] += 1
            report["search_releases"] += len(measured)
            order.rotate(-order.index(measured[-1]["context_id"]) - 1)
        else:
            controls[phase].extend(measured)
        next_packet = packet_index + 1
    require(next(rows, None) is None and next(timings, None) is None, "Unmatched observation or timing tail")
    for directory, names in committed_archives.items():
        actual = {path.name for path in (output / directory).glob("*.npz")}
        require(actual == names, f"Unmatched native archive tail in {directory}")
        adapter_ids = [row["id"] for row in ledger(output / directory / "shots.jsonl")]
        require(adapter_ids == archived_ids[directory], f"Adapter ledger differs in {directory}")
    for phase, control_rows in controls.items():
        if control_rows:
            gate = control_gate(control_rows)
            require(gate["status"] == "passed", f"Saved {phase} failed its original gate")
            report["controls"][phase] = gate
    require(not report["search_releases"] or "controls_pre" in report["controls"],
            "Search prefix lacks passed pre-controls")
    return next_packet


def recover(output, report, summaries, outcomes, pending, order, generators, seen, dry_preparations, geometry_key):
    """Restore a committed prefix and return (next packet index, unexecuted drafts).

    Each draft is a (path, proposals, warm start) tuple. Its proposals already
    count toward attempts and the generator cursor, so callers reissue it
    exactly and write its first finalized preparation result.
    """
    output = Path(output)
    contract = read(output / "contract.json")
    saved_report = read(output / "report.json")
    require(contract["fingerprint"] == digest({k: v for k, v in contract.items() if k != "fingerprint"}),
            "Copied grid contract fingerprint changed")
    for name, key in (("plan.json", "plan_sha256"), ("historical-geometries.json", "history_sha256")):
        require(sha(output / name) == contract[key], f"Copied grid input changed: {name}")
    require(set(summaries) == set(outcomes) == set(pending) == set(order) == set(generators) == set(seen)
            == set(contract["cells"]), "Recovery context inventory differs")
    require(all(not summaries[cell]["attempts"] and not summaries[cell]["search_releases"]
                and not summaries[cell]["preparation_declines"] and not pending[cell] and not outcomes[cell]
                and not dry_preparations[cell] for cell in summaries),
            "Recovery requires fresh in-memory counters and queues")
    require(not report["controls"] and not report["search_releases"] and not report["search_packets"]
            and not report["mixed_packets"] and not report["max_cells_in_packet"],
            "Recovery requires fresh report counters")
    drafts = restore_preparations(output, summaries, pending, generators, seen, dry_preparations, geometry_key)
    next_packet = restore_releases(output, report, summaries, outcomes, pending, order, contract["native_batch"])
    target = contract["target_per_cell"]
    for cell, summary in summaries.items():
        require(summary["search_releases"] + len(pending[cell]) <= target, f"Saved quota exceeded for {cell}")
        summary["route_counts"] = dict(outcomes[cell])
        saved = saved_report["cells"][cell]
        for field in ("attempts", "search_releases", "preparation_declines"):
            require(saved[field] <= summary[field], f"Saved report exceeds evidence for {cell}/{field}")
        require(all(count <= summary["route_counts"].get(route, 0) for route, count in saved["route_counts"].items()),
                f"Saved report route totals exceed evidence for {cell}")
    for field in ("search_releases", "search_packets", "mixed_packets", "max_cells_in_packet"):
        require(saved_report[field] <= report[field], f"Saved report exceeds durable {field}")
    for phase, gate in saved_report["controls"].items():
        require(report["controls"].get(phase) == gate, f"Saved control gate differs for {phase}")
    if "controls_post" in report["controls"]:
        require(report["search_releases"] == len(summaries) * target and not drafts,
                "Post-controls were saved before the full quota")
    return next_packet, drafts
