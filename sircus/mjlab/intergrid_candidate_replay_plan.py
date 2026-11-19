"""Freeze diagnostic replays from already admitted, independently audited shots."""

from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path

from .benchmark_verify import contained, digest, read, sha
from .intergrid_matched_route_study import outcome
from .intergrid_route_fallback_audit import VERSION as FALLBACK_AUDIT_VERSION
from .intergrid_route_fallback_study import POSITIONS, VERSION as FALLBACK_VERSION
from .intergrid_route_plan import ROUTES
from .intergrid_route_refit_inputs import _audited_source
from .intergrid_route_selector_plan import (
    FALLBACK_METHODS, STANDARD_CAPTURES, select_fallback_requests, summarize_fallback,
)


VERSION = "sircus-intergrid-candidate-replay-plan-v1"
FLIGHT_MISSES = 16
SETUP_FAILURES = 5
_FROZEN_INPUTS = ("layouts.json", "training-prior.json", "route-models.json", "route-history.json")
_SOURCE_FILES = (*_FROZEN_INPUTS, "choices.json", "rows.jsonl")


def command_key(request):
    """Compare native actions and joint branches independently of planner and proposal IDs."""
    return (
        tuple(request["bucket_xy_m"]), tuple(request["values"]),
        request["board_heading_offset_deg"] % 360, request["board_twist_deg"] % 360,
        tuple(request["selected_q10"]), tuple(request["selected_q14"]),
    )


def _check_artifact(artifact, layout):
    proposals, admitted, records = (artifact[key] for key in ("proposals", "admitted", "records"))
    identity, xy = layout["id"], layout["bucket_xy_m"]
    proposal_ids = {row["id"] for row in proposals}
    by_id = {row["id"]: row for row in admitted}
    if len(proposals) != len(records) or len(proposal_ids) != len(proposals) or len(by_id) != len(admitted):
        raise ValueError(f"Fallback preparation has duplicate or missing records: {identity}")
    ordered = []
    for proposal, record in zip(proposals, records, strict=True):
        name, route = proposal["id"], proposal["intended_route"]
        ids = record["admitted_request_ids"]
        if (
            proposal["context_id"] != identity or proposal["bucket_xy_m"] != xy
            or proposal["phase"] != "test" or proposal["role"] != "test" or proposal["method"] != route
            or route not in ROUTES or (record["proposal_id"], record["proposal_sha256"]) != (name, digest(proposal))
            or record["admitted"] != len(ids) or len(ids) > 1
            or (ids and record["reason"] is not None)
            or (not ids and (not isinstance(record["reason"], str) or not record["reason"]))
        ):
            raise ValueError(f"Invalid fallback preparation record: {name}")
        for branch_id in ids:
            request = by_id[branch_id]
            if (
                branch_id != name + "-b0" or request["proposal_id"] != name
                or request["native_prepared"] is not True or request["command_values"] != proposal["values"]
                or any(request[key] != value for key, value in proposal.items() if key != "id")
            ):
                raise ValueError(f"Fallback admitted action differs from its proposal: {branch_id}")
            ordered.append(branch_id)
    if ordered != list(by_id):
        raise ValueError(f"Fallback admission order differs from preparation records: {identity}")


def _read_test_releases(study, slots):
    requests = {slot["id"]: slot["request"] for slot in slots if slot["request"] is not None}
    observed = {}
    with (study / "rows.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            name = row["id"]
            if row["phase"] != "test" and name not in requests:
                continue
            request = requests.get(name)
            if (
                request is None or name in observed or row["native_release"] is not True
                or any(row[key] != request[key] for key in (
                    "phase", "role", "context_id", "method", "bucket_xy_m", "intended_route", "values", "proposal_id",
                ))
                or type(row["invalid_physics"]) is not bool or type(row["route_success"]) is not bool
            ):
                raise ValueError(f"Fallback release differs from its frozen request: {name}")
            observed[name] = row
    if observed.keys() != requests.keys():
        raise ValueError("A frozen fallback request lacks its test release")
    return observed


def _check_proofs(study, manifest, request, binding):
    for role in ("context_resolution", "prepared_connection"):
        relative = request[role + "_file"]
        name = f"native-heading/{relative}"
        expected = request[role + "_sha256"]
        if manifest[name] != expected:
            raise ValueError(f"Fallback proof is absent from the audited manifest: {name}")
        if name not in binding:
            binding[name] = sha(contained(study / "native-heading", relative))
        if binding[name] != expected:
            raise ValueError(f"Fallback proof differs from its audited bytes: {name}")


def _setup_failure(layout, proposals, records):
    reasons = {route: Counter() for route in ROUTES}
    for proposal, record in zip(proposals, records, strict=True):
        reasons[proposal["intended_route"]][record["reason"]] += 1
    return dict(layout=layout, reason="all_proposals_rejected" if proposals else "no_proposals",
                routes={route: dict(proposed=sum(counts.values()), reasons=dict(counts))
                        for route, counts in reasons.items()})


def _replay_case(layout, selector, direct_first, observed, admitted):
    chosen = selector["request"]
    control = direct_first["request"]
    control_key = command_key(control)
    chosen_key = command_key(chosen)
    control_capture = outcome(observed[direct_first["id"]]) in STANDARD_CAPTURES
    seen = {chosen_key}
    commands = [chosen]
    for request in admitted:
        key = command_key(request)
        if key not in seen:
            commands.append(request)
            seen.add(key)

    requests, candidates = [], []
    for index, command in enumerate(commands):
        name = f"{layout['id']}-replay-{index:02d}"
        already_direct_first = command_key(command) == control_key
        requests.append(dict(command, id=name, phase="test", role="test", stratum=layout["stratum"],
                             method="alternative" if index else "baseline_repeat"))
        candidates.append(dict(id=name, already_direct_first=already_direct_first,
                               prior_direct_first_capture=control_capture if already_direct_first else None))
    return dict(layout=layout, prior_selector_id=selector["id"], prior_direct_first_id=direct_first["id"],
                requests=requests, candidates=candidates)


def load_replay_plan(study, audit_path):
    """Select the complete ordinary-miss cohort and preserve all valid archived candidates."""
    study = Path(study)
    report, manifest, receipt, binding, contract = _audited_source(
        study, Path(audit_path), FALLBACK_VERSION, _SOURCE_FILES,
    )
    attempts = POSITIONS * len(FALLBACK_METHODS)
    checkpoint = binding["route-models.json"]
    choices = read(study / "choices.json")
    if (
        receipt["version"] != FALLBACK_AUDIT_VERSION
        or contract["test_positions"] != POSITIONS or contract["methods"] != list(FALLBACK_METHODS)
        or contract["routes"] != list(ROUTES) or contract["attempts_per_method_per_position"] != 1
        or contract["frozen_inputs"] != {name: binding[name] for name in _FROZEN_INPUTS}
        or checkpoint != choices["checkpoint_sha256"] or checkpoint != report["checkpoint_sha256"]
        or checkpoint != receipt["checkpoint_sha256"]
        or receipt["choices_sha256"] != binding["choices.json"] or report["choices_sha256"] != binding["choices.json"]
        or receipt["test_positions"] != POSITIONS or report["test_prepared"] != POSITIONS
        or receipt["test_attempts"] != attempts or report["test_slots"] != attempts or len(choices["slots"]) != attempts
        or receipt["test_releases"] != report["test_releases"] or report["test_releases"] != report["test_admitted"]
        or receipt["comparison"] != report["comparison"]
    ):
        raise ValueError("Fallback study does not match its independent audit")

    archive = read(study / "layouts.json")
    layouts = archive["layouts"]
    identities = {layout["id"] for layout in layouts}
    if (
        len(layouts) != POSITIONS or len(identities) != POSITIONS
        or len({tuple(layout["bucket_xy_m"]) for layout in layouts}) != POSITIONS
        or archive["provenance"]["layouts_sha256"] != digest(layouts)
        or any(layout["id"] != f"intergrid-route-fallback-test-{index:02d}"
               or layout["context_id"] != layout["id"] or layout["split"] != "test"
               or len(layout["bucket_xy_m"]) != 2 or not all(map(math.isfinite, layout["bucket_xy_m"]))
               for index, layout in enumerate(layouts))
    ):
        raise ValueError("Fallback replay layouts differ from the audited positions")
    preparation_names = {f"test/{identity}.json" for identity in identities}
    if (
        preparation_names != {name for name in manifest if name.startswith("test/")}
        or preparation_names != {path.relative_to(study).as_posix() for path in (study / "test").glob("*.json")}
    ):
        raise ValueError("Expected exactly one audited preparation file per fallback position")

    history = read(study / "route-history.json")
    prepared, slots = {}, []
    for layout in layouts:
        identity = layout["id"]
        name = f"test/{identity}.json"
        path = contained(study, name)
        binding[name] = sha(path)
        if binding[name] != manifest[name]:
            raise ValueError(f"Fallback preparation differs from its audit: {name}")
        artifact = read(path)
        _check_artifact(artifact, layout)
        prepared[identity] = artifact
        slots.extend(select_fallback_requests(layout, artifact["proposals"], artifact["admitted"], history))
    if slots != choices["slots"]:
        raise ValueError("Fallback preparations do not reproduce the audited choices")
    observed = _read_test_releases(study, slots)
    if len(observed) != receipt["test_releases"] or summarize_fallback(layouts, slots, observed) != receipt["comparison"]:
        raise ValueError("Fallback ledger does not reproduce the audited comparison")

    slot_index = {(slot["context_id"], slot["method"]): slot for slot in slots}
    cases, setup_failures = [], []
    for layout in layouts:
        identity = layout["id"]
        artifact = prepared[identity]
        selector = slot_index[identity, "route_selector"]
        result = outcome(observed.get(selector["id"]))
        if result == "setup_rejected":
            if artifact["admitted"]:
                raise ValueError(f"Setup refusal had an admitted command: {identity}")
            setup_failures.append(_setup_failure(layout, artifact["proposals"], artifact["records"]))
        elif result == "flight_miss":
            direct_first = slot_index[identity, "direct_first"]
            case = _replay_case(layout, selector, direct_first, observed, artifact["admitted"])
            for request in case["requests"]:
                _check_proofs(study, manifest, request, binding)
            cases.append(case)
    if len(cases) != FLIGHT_MISSES or len(setup_failures) != SETUP_FAILURES:
        raise ValueError("Expected exactly sixteen ordinary selector misses and five setup refusals")
    return dict(version=VERSION, cases=cases, setup_failures=setup_failures), binding
