"""Pure, hash-bound display contract for the 35-cell sweep development study.

Saved catalogue evidence and this study's new trials remain separate. Search
statistics may aggregate work, while repeat rates always belong to one exact
selected recipe. Missing native counts never become successes or releases.
"""

from __future__ import annotations

from collections.abc import Mapping
import copy
import math
from numbers import Real

from .corrected_contract import PHYSICS, grid_layouts
from .learning_contract import digest

VERSION = "sircus-sweep-grid35-development-v1"
IDENTITY_FIELDS = (
    "contract_fingerprint",
    "scene_sha256",
    "asset_sha256",
    "source_sha256",
    "physics",
)
STATUSES = {
    "queued",
    "searching",
    "verified_fresh",
    "saved_verified",
    "unresolved",
    "blocked",
    "failed",
}


def _hash(value, label):
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError(f"Invalid {label} SHA256")
    return value


def validate_identity(identity):
    """Do not mix source, asset, scene or corrected-contact contracts."""
    if not isinstance(identity, Mapping):
        raise ValueError("Study identity must be a mapping")
    value = {name: copy.deepcopy(identity.get(name)) for name in IDENTITY_FIELDS}
    for name in ("contract_fingerprint", "scene_sha256"):
        _hash(value[name], name)
    for name in ("asset_sha256", "source_sha256"):
        if not isinstance(value[name], Mapping) or not value[name]:
            raise ValueError(f"Nonempty {name} is required")
        for path, expected in value[name].items():
            if not isinstance(path, str) or not path:
                raise ValueError(f"Invalid {name} path")
            _hash(expected, name)
    if value["physics"] != PHYSICS:
        raise ValueError("Study physics differs from corrected native contact")
    return value


def _number(value, label):
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
    ):
        raise ValueError(f"{label} must be finite")
    return float(value)


def _vector(value, size, label):
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(f"{label} requires {size} coordinates")
    return [_number(x, label) for x in value]


def _count(value, label, *, optional=False):
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def statistics(captures=None, attempts=None):
    """Requested-trial denominator, including any geometric/path rejects."""
    captures = _count(captures, "captures", optional=True)
    attempts = _count(attempts, "attempts", optional=True)
    if captures is not None and (attempts is None or captures > attempts):
        raise ValueError("Capture count exceeds or lacks its requested denominator")
    return dict(
        captures=captures,
        attempts=attempts,
        rate=(captures / attempts if captures is not None and attempts else None),
        denominator_scope="All requested repeats, including geometry/path rejects",
    )


def _cohort(result, prefix):
    metrics = result.get("metrics", {})
    return statistics(
        metrics.get(prefix + "_captures"), metrics.get(prefix + "_attempts")
    )


def _saved(result):
    return bool(
        result.get("provenance", {}).get("saved_evidence") is True
        or result.get("method")
        in {"catalog_lookup", "catalogue", "cached", "dataset_lookup"}
    )


def _configuration(result, bucket, *, source):
    config = result.get("configuration")
    if not isinstance(config, Mapping):
        raise ValueError("Selected shot lacks its configuration")
    board = _vector(config.get("board_center"), 3, "board center")
    release = _vector(config.get("release_center"), 3, "ball-center release")
    tilt = _number(config.get("tilt_deg"), "tilt")
    gap = _number(config.get("gap_m"), "drop gap")
    if not 0 <= tilt < 90 or gap <= 0 or release[2] <= board[2]:
        raise ValueError("Selected shot violates vertical shot geometry")
    return dict(
        board_center_m=board,
        tilt_deg=tilt,
        gap_m=gap,
        release_ball_center_m=release,
        distance_m=math.hypot(board[0] - bucket[0], board[1] - bucket[1]),
        source=source,
        request_id=result.get("request_id"),
        configuration_sha256=digest(config),
    )


def _qualified(result):
    nominal, fresh = _cohort(result, "nominal"), _cohort(result, "fresh")
    metrics = result.get("metrics", {})
    return bool(
        result.get("success") is True
        and result.get("qualified") is True
        and result.get("status_native") == "verified_shot"
        and result.get("motion", {}).get("passed") is True
        and metrics.get("scalar_score_audit_passed") is True
        and not metrics.get("invalid")
        and nominal["captures"] == nominal["attempts"] == 3
        and fresh["attempts"] == 20
        and fresh["captures"] is not None
        and fresh["captures"] >= 18
    )


def validate_attempt(attempt, layout, identity):
    """Validate exact bucket/request/provenance before accepting display evidence.

    File-byte hashes are retained as provenance, while canonical request/result
    payload hashes are computed here. The native request key is a separate
    identity supplied by the worker; it is never relabeled as a file checksum.
    """
    if not isinstance(attempt, Mapping):
        raise ValueError("Study attempt must be a mapping")
    request, result = attempt.get("request"), attempt.get("result")
    if not isinstance(request, Mapping) or not isinstance(result, Mapping):
        raise ValueError("Study attempt requires request and result mappings")
    if not isinstance(attempt.get("stage"), str) or not attempt["stage"]:
        raise ValueError("Study attempt stage is required")
    request_id = request.get("request_id")
    if (
        not isinstance(request_id, str)
        or not request_id
        or result.get("request_id") != request_id
    ):
        raise ValueError("Study request/result association mismatch")
    if (
        request.get("bucket_xy") != layout["bucket_xy_m"]
        or result.get("bucket_xy_m") != layout["bucket_xy_m"]
    ):
        raise ValueError("Study attempt belongs to another bucket position")
    provenance = result.get("provenance", {})
    if any(provenance.get(name) != identity[name] for name in IDENTITY_FIELDS):
        raise ValueError("Study attempt source/scene/asset/physics identity mismatch")
    if provenance.get("policy_used") is not False:
        raise ValueError("Sweep study cannot include learned-policy inference")
    if result.get("success") is True and not _qualified(result):
        raise ValueError("Successful study shot lacks qualification evidence")
    for key, payload in (
        ("request_payload_sha256", request),
        ("result_payload_sha256", result),
    ):
        if attempt.get(key) is not None and attempt[key] != digest(payload):
            raise ValueError(f"Changed study {key}")
    for key in ("request_sha256", "result_sha256"):
        if attempt.get(key) is not None:
            _hash(attempt[key], key)
    result_url = attempt.get("result_url")
    if (
        result_url is not None
        and result_url != "/api/planner-grid/result/" + layout["id"]
    ):
        raise ValueError("Study result URL must identify this grid cell")
    evidence = attempt.get("evidence_sha256", {})
    if not isinstance(evidence, Mapping):
        raise ValueError("Study evidence manifest must be a mapping")
    for name, expected in evidence.items():
        if (
            not isinstance(name, str)
            or not name
            or name.startswith("/")
            or ".." in name.split("/")
        ):
            raise ValueError("Study evidence paths must be relative and contained")
        _hash(expected, "evidence")
    if provenance.get("request_sha256") is not None:
        _hash(provenance["request_sha256"], "native request")
    value = copy.deepcopy(dict(attempt))
    value.update(
        request_payload_sha256=digest(request),
        result_payload_sha256=digest(result),
        saved_evidence=_saved(result),
    )
    return value


def frozen_validation(raw, selected):
    """A separate, frozen 3+30 repeat set never affects recipe selection."""
    if raw is None:
        return None
    if not isinstance(raw, Mapping) or raw.get("status") not in {
        "pending",
        "running",
        "completed",
        "blocked",
        "failed",
    }:
        raise ValueError("Unknown frozen-validation status")
    value = copy.deepcopy(dict(raw))
    if (
        selected is None
        or value.get("configuration_sha256") != selected["configuration_sha256"]
    ):
        raise ValueError("Frozen validation belongs to another selected recipe")
    for name, limit in (("nominal", 3), ("fresh", 30)):
        cohort = value.get(name, {})
        normalized = statistics(cohort.get("captures"), cohort.get("attempts"))
        if normalized["attempts"] is not None and normalized["attempts"] > limit:
            raise ValueError("Frozen validation exceeds its fixed repeat budget")
        if value["status"] == "completed" and (
            normalized["attempts"] != limit or normalized["captures"] is None
        ):
            raise ValueError("Completed frozen validation lacks all requested repeats")
        value[name] = normalized
    quality = value.get("quality")
    if quality is not None:
        clean = _count(quality.get("clean"), "frozen clean")
        assisted = _count(quality.get("assisted"), "frozen assisted")
        if (
            value["fresh"]["captures"] is None
            or clean + assisted != value["fresh"]["captures"]
        ):
            raise ValueError("Frozen fresh quality counts differ from captures")
        value["quality"] = dict(
            clean=clean, assisted=assisted, cohort="fresh", source="frozen_validation"
        )
    if value.get("seed") is not None:
        _count(value["seed"], "frozen seed")
    value["scope"] = (
        "Separate frozen recipe validation; no reselection from these outcomes"
    )
    return value


def summarize_cell(layout, attempts, *, identity, state=None, validation=None):
    """One cell, one selected recipe; fresh and saved repeat cohorts stay apart."""
    identity = validate_identity(identity)
    bucket = _vector(layout.get("bucket_xy_m"), 2, "bucket center")
    rows = [validate_attempt(row, layout, identity) for row in attempts]
    ids = [row["request"]["request_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Repeated study request ID would double-count work")
    search = dict(
        cpu_candidates=0,
        rejected=0,
        native_requests=0,
        confirmation_requests=0,
        captures=0,
    )
    unknown_search_captures = False
    for row in rows:
        if row["saved_evidence"]:
            continue
        metrics = row["result"].get("metrics", {})
        for out, field in (
            (
                "cpu_candidates",
                (
                    "search_cpu_candidates"
                    if "search_cpu_candidates" in metrics
                    else "cpu_candidates"
                ),
            ),
            (
                "rejected",
                (
                    "search_geometry_rejected"
                    if "search_geometry_rejected" in metrics
                    else "geometry_rejected"
                ),
            ),
            ("native_requests", "search_requests"),
            ("confirmation_requests", "confirmation_requests"),
        ):
            search[out] += _count(metrics.get(field, 0), field)
        captured = row.get("search_captures", metrics.get("search_captures"))
        requested = metrics.get("search_requests", 0)
        if captured is None and requested:
            unknown_search_captures = True
        elif captured is not None:
            search["captures"] += statistics(captured, requested)["captures"]
    if search["rejected"] > search["cpu_candidates"]:
        raise ValueError("Geometry rejects exceed checked CPU candidates")
    if unknown_search_captures:
        search["captures"] = None
    fresh_ready = [
        r for r in rows if not r["saved_evidence"] and _qualified(r["result"])
    ]
    saved_ready = [r for r in rows if r["saved_evidence"] and _qualified(r["result"])]
    selected_row = (fresh_ready or saved_ready or [None])[-1]
    result = selected_row["result"] if selected_row else None
    selected = None
    nominal, fresh, saved_nominal, saved_fresh = (statistics() for _ in range(4))
    quality = dict(clean=None, assisted=None, scope="Unknown repeat quality breakdown")
    if result:
        saved = selected_row["saved_evidence"]
        selected = _configuration(
            result, bucket, source="saved_reference" if saved else "fresh_study"
        )
        if saved:
            saved_nominal, saved_fresh = _cohort(result, "nominal"), _cohort(
                result, "fresh"
            )
            # A later failed fresh retest may report rates for the same recipe,
            # but only with an explicit configuration binding.
            matching = [
                r
                for r in rows
                if not r["saved_evidence"]
                and r["result"]
                .get("metrics", {})
                .get("confirmation_configuration_sha256")
                == selected["configuration_sha256"]
            ]
            if matching:
                nominal, fresh = _cohort(matching[-1]["result"], "nominal"), _cohort(
                    matching[-1]["result"], "fresh"
                )
        else:
            nominal, fresh = _cohort(result, "nominal"), _cohort(result, "fresh")
        supplied_quality = selected_row.get("quality")
        if supplied_quality is not None:
            clean = _count(supplied_quality.get("clean"), "clean")
            assisted = _count(supplied_quality.get("assisted"), "assisted")
            cohort = supplied_quality.get("cohort")
            stats = _cohort(result, cohort) if cohort in {"nominal", "fresh"} else None
            if (
                stats is None
                or stats["captures"] is None
                or clean + assisted != stats["captures"]
            ):
                raise ValueError("Repeat quality breakdown lacks a matching cohort")
            quality = dict(
                clean=clean,
                assisted=assisted,
                cohort=cohort,
                source=selected["source"],
                scope="Explicit recorded repeat cohort",
            )
    last = rows[-1]["result"] if rows else {}
    status = (
        "verified_fresh"
        if fresh_ready
        else (
            "saved_verified"
            if saved_ready
            else (
                "queued"
                if not rows
                else (
                    "blocked"
                    if last.get("status_native") == "common_home_unresolved"
                    else "unresolved"
                )
            )
        )
    )
    if state is not None:
        if state not in STATUSES:
            raise ValueError("Unknown explicit cell state")
        if state in {"verified_fresh", "saved_verified"} and state != status:
            raise ValueError("Explicit state cannot fabricate shot qualification")
        if state in {"queued", "searching", "blocked", "failed", "unresolved"}:
            status = state
    return dict(
        id=layout["id"],
        bucket_xy_m=bucket,
        row=layout.get("row"),
        column=layout.get("column"),
        status=status,
        stage=rows[-1]["stage"] if rows else "pending",
        search=search,
        selected=selected,
        nominal=nominal,
        fresh=fresh,
        saved_nominal=saved_nominal,
        saved_fresh=saved_fresh,
        quality=quality,
        reason=last.get("message", "Awaiting this study's trials"),
        result_url=selected_row.get("result_url") if selected_row else None,
        frozen_validation=frozen_validation(validation, selected),
        attempts=[
            dict(
                stage=r["stage"],
                request_id=r["request"]["request_id"],
                request_payload_sha256=r["request_payload_sha256"],
                result_payload_sha256=r["result_payload_sha256"],
                request_sha256=r.get("request_sha256"),
                result_sha256=r.get("result_sha256"),
                evidence_sha256=r.get("evidence_sha256", {}),
                native_request_sha256=r["result"]
                .get("provenance", {})
                .get("request_sha256"),
                saved_evidence=r["saved_evidence"],
                status_native=r["result"].get("status_native"),
                success=r["result"].get("success") is True,
            )
            for r in rows
        ],
    )


def _summary(cells):
    return dict(
        grid_positions=len(cells),
        verified_fresh=sum(
            c["selected"] is not None and c["selected"]["source"] == "fresh_study"
            for c in cells
        ),
        saved_verified=sum(
            c["selected"] is not None and c["selected"]["source"] == "saved_reference"
            for c in cells
        ),
        unresolved=sum(c["status"] == "unresolved" for c in cells),
        blocked=sum(c["status"] == "blocked" for c in cells),
        pending=sum(c["status"] in {"queued", "searching"} for c in cells),
        failed=sum(c["status"] == "failed" for c in cells),
        cpu_candidates=sum(c["search"]["cpu_candidates"] for c in cells),
        geometry_rejected=sum(c["search"]["rejected"] for c in cells),
        search_native_requests=sum(c["search"]["native_requests"] for c in cells),
        confirmation_requests=sum(c["search"]["confirmation_requests"] for c in cells),
        frozen_validation_completed=sum(
            bool(
                c.get("frozen_validation")
                and c["frozen_validation"]["status"] == "completed"
            )
            for c in cells
        ),
        frozen_nominal_captures=sum(
            (c.get("frozen_validation") or {}).get("nominal", {}).get("captures") or 0
            for c in cells
        ),
        frozen_nominal_attempts=sum(
            (c.get("frozen_validation") or {}).get("nominal", {}).get("attempts") or 0
            for c in cells
        ),
        frozen_fresh_captures=sum(
            (c.get("frozen_validation") or {}).get("fresh", {}).get("captures") or 0
            for c in cells
        ),
        frozen_fresh_attempts=sum(
            (c.get("frozen_validation") or {}).get("fresh", {}).get("attempts") or 0
            for c in cells
        ),
    )


def make_report(
    attempts_by_cell,
    *,
    identity,
    protocol,
    status="running",
    states=None,
    started_at=None,
    updated_at=None,
    validation_by_cell=None,
):
    """Always publish all 35 fixed distinct development targets, including unknowns."""
    identity = validate_identity(identity)
    layouts = grid_layouts()
    known = {r["id"] for r in layouts}
    if not isinstance(attempts_by_cell, Mapping) or set(attempts_by_cell) - known:
        raise ValueError("Study contains unknown grid targets")
    if states is not None and (not isinstance(states, Mapping) or set(states) - known):
        raise ValueError("Study states contain unknown grid targets")
    if validation_by_cell is not None and (
        not isinstance(validation_by_cell, Mapping) or set(validation_by_cell) - known
    ):
        raise ValueError("Frozen validation contains unknown grid targets")
    if status not in {
        "running",
        "validating",
        "completed",
        "failed",
        "blocked",
        "cancelled",
    }:
        raise ValueError("Unknown study status")
    cells = [
        summarize_cell(
            layout,
            attempts_by_cell.get(layout["id"], []),
            identity=identity,
            state=(states or {}).get(layout["id"]),
            validation=(validation_by_cell or {}).get(layout["id"]),
        )
        for layout in layouts
    ]
    summary = _summary(cells)
    if status == "completed" and summary["pending"]:
        raise ValueError("Completed study still has pending cells")
    report = dict(
        version=VERSION,
        status=status,
        started_at=started_at,
        updated_at=updated_at,
        identity=identity,
        protocol=copy.deepcopy(protocol),
        cells=cells,
        summary=summary,
        scope="Searched development settings, not RL inference or continuous-table coverage. Prescribed held endpoints and sampled geometric paths; no impact compliance or grasp/transit dynamics.",
    )
    report["fingerprint"] = digest(report)
    return report


def validate_report(report):
    """Reject changed snapshots and grid identities before serving display data."""
    if not isinstance(report, Mapping) or report.get("version") != VERSION:
        raise ValueError("Unknown grid study report")
    raw = copy.deepcopy(dict(report))
    expected = raw.pop("fingerprint", None)
    if expected != digest(raw):
        raise ValueError("Grid study report fingerprint mismatch")
    if raw.get("status") not in {
        "running",
        "validating",
        "completed",
        "failed",
        "blocked",
        "cancelled",
    }:
        raise ValueError("Unknown grid study status")
    validate_identity(raw.get("identity"))
    layouts = grid_layouts()
    cells = raw.get("cells")
    if not isinstance(cells, list) or len(cells) != len(layouts):
        raise ValueError("Grid study must preserve all 35 cells")
    for layout, cell in zip(layouts, cells):
        if (
            cell.get("id") != layout["id"]
            or cell.get("bucket_xy_m") != layout["bucket_xy_m"]
        ):
            raise ValueError("Grid study target identity changed")
        if cell.get("status") not in STATUSES:
            raise ValueError("Unknown grid cell status")
        for name in ("nominal", "fresh", "saved_nominal", "saved_fresh"):
            cohort = cell.get(name, {})
            if cohort != statistics(cohort.get("captures"), cohort.get("attempts")):
                raise ValueError("Grid cohort rate or denominator changed")
        search = cell.get("search", {})
        for name in (
            "cpu_candidates",
            "rejected",
            "native_requests",
            "confirmation_requests",
        ):
            _count(search.get(name), name)
        if search["rejected"] > search["cpu_candidates"]:
            raise ValueError("Grid rejects exceed checked proposals")
        captures = _count(search.get("captures"), "search captures", optional=True)
        if captures is not None and captures > search["native_requests"]:
            raise ValueError("Grid captures exceed native search requests")
        selected = cell.get("selected")
        if selected is not None:
            if not isinstance(selected, Mapping) or selected.get("source") not in {
                "fresh_study",
                "saved_reference",
            }:
                raise ValueError("Unknown selected evidence source")
            _hash(selected.get("configuration_sha256"), "selected configuration")
            board = _vector(selected.get("board_center_m"), 3, "selected board")
            release = _vector(
                selected.get("release_ball_center_m"), 3, "selected release"
            )
            distance = math.hypot(
                board[0] - layout["bucket_xy_m"][0], board[1] - layout["bucket_xy_m"][1]
            )
            if selected.get("distance_m") != distance or release[2] <= board[2]:
                raise ValueError("Selected geometry relationship changed")
            for name in ("gap_m", "tilt_deg"):
                _number(selected.get(name), name)
            if selected["gap_m"] <= 0 or not 0 <= selected["tilt_deg"] < 90:
                raise ValueError("Selected shot geometry is invalid")
            prefix = "saved_" if selected["source"] == "saved_reference" else ""
            nominal, fresh = cell[prefix + "nominal"], cell[prefix + "fresh"]
            if (
                nominal["captures"] != 3
                or nominal["attempts"] != 3
                or fresh["attempts"] != 20
                or fresh["captures"] is None
                or fresh["captures"] < 18
            ):
                raise ValueError("Selected evidence lacks repeated qualification")
            matches = [
                r
                for r in cell.get("attempts", [])
                if r.get("request_id") == selected.get("request_id")
            ]
            if (
                len(matches) != 1
                or matches[0].get("success") is not True
                or matches[0].get("saved_evidence") != (prefix == "saved_")
            ):
                raise ValueError(
                    "Selected evidence lacks its unique successful attempt"
                )
        elif cell["status"] in {"verified_fresh", "saved_verified"} or cell.get(
            "result_url"
        ):
            raise ValueError("Ready grid cell lacks a selected shot")
        expected_status = (
            "verified_fresh"
            if selected and selected["source"] == "fresh_study"
            else "saved_verified"
        )
        if (
            cell["status"] in {"verified_fresh", "saved_verified"}
            and cell["status"] != expected_status
        ):
            raise ValueError("Ready cell changed evidence source")
        if (
            cell.get("result_url") is not None
            and cell["result_url"] != "/api/planner-grid/result/" + layout["id"]
        ):
            raise ValueError("Grid result URL changed cell identity")
        request_ids = [r.get("request_id") for r in cell.get("attempts", [])]
        if len(request_ids) != len(set(request_ids)):
            raise ValueError("Duplicate display attempt")
        for attempt in cell.get("attempts", []):
            for name in ("request_payload_sha256", "result_payload_sha256"):
                _hash(attempt.get(name), name)
        frozen = cell.get("frozen_validation")
        if frozen != frozen_validation(frozen, selected):
            raise ValueError("Frozen validation normalization changed")
    if raw.get("summary") != _summary(cells):
        raise ValueError("Grid summary differs from its cells")
    if raw.get("status") == "completed" and raw["summary"]["pending"]:
        raise ValueError("Completed grid report still has pending cells")
    return copy.deepcopy(dict(report))
