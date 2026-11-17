"""Check the interactive one-drop mjlab API on fresh bucket placements."""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import median
import time
from urllib.error import HTTPError
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


VERSION = "sircus-mjlab-interactive-acceptance-v1"
STUDY_VERSION = "sircus-intergrid-system-planner-v1"
AUDIT_VERSION = "sircus-intergrid-system-independent-audit-v1"
SEED = 2026112701
API = "/api/interactive-physics"
PLANNING = {"queued", "planning"}
RELEASING = {"queued", "running"}
UNAVAILABLE = {"failed", "expired"}
CAPTURES = {"direct", "table", "unclassified_capture"}
OUTCOMES = CAPTURES | {"arm_assisted_capture", "flight_miss", "invalid_physics"}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _vector(value, length):
    return isinstance(value, list) and len(value) == length and all(map(_finite, value))


def _checksum(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _digest(value):
    return sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _save(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def historical_layouts(study, audit_path):
    """Bind the coordinate inventory to the independently audited system benchmark."""
    source = {name: (study / name).read_bytes() for name in
              ("report.json", "contract.json", "evidence-manifest.json", "layouts.json")}
    hashes = {name: sha256(raw).hexdigest() for name, raw in source.items()}
    report, contract, manifest, archived = (json.loads(raw) for raw in source.values())
    audit_bytes = audit_path.read_bytes()
    audit = json.loads(audit_bytes)
    fingerprint = contract.pop("fingerprint")
    _require((contract["version"], contract["physics_substeps"], report["status"], report["phase"],
              audit["version"], audit["status"]) == (STUDY_VERSION, 64, "completed", "completed", AUDIT_VERSION, "passed")
             and fingerprint == _digest(contract), "Expected a completed, audited native64 planner comparison")
    _require(report["evidence_manifest_sha256"] == audit["evidence_manifest_sha256"] == hashes["evidence-manifest.json"]
             and audit["report_sha256"] == hashes["report.json"]
             and audit["comparison"] == report["comparison"]
             and manifest["contract.json"] == hashes["contract.json"]
             and manifest["layouts.json"] == contract["frozen_inputs"]["layouts.json"] == hashes["layouts.json"],
             "The historical layout bytes differ from the audited, frozen comparison")
    _require(set(archived) == {"train", "test"}, "Expected archived train and test placements")
    metadata = archived["test"]["provenance"]
    geometry, grid = metadata["geometry"], metadata["usable_grid"]
    minimum = metadata["minimum_separation_m"]
    _require(minimum == .005 and geometry["scene_sha256"] == contract["identity"]["scene_sha256"],
             "Expected five-millimeter novelty and the audited scene geometry")
    known = set()
    for split, part in archived.items():
        layouts, provenance = part["layouts"], part["provenance"]
        coordinates = provenance["excluded_coordinates_m"]
        _require(len(layouts) == 50 and _digest(layouts) == provenance["layouts_sha256"]
                 and _digest(coordinates) == provenance["excluded_coordinates_sha256"]
                 and (provenance["geometry"], provenance["minimum_separation_m"]) == (geometry, minimum),
                 f"Historical {split} coordinates failed provenance checks")
        known.update(tuple(xy) for xy in coordinates)
        known.update(tuple(row["bucket_xy_m"]) for row in layouts)
    known.update(map(tuple, grid.values()))
    return geometry, grid, sorted(known), dict(hashes, audit_sha256=sha256(audit_bytes).hexdigest())


def sample_positions(geometry, grid, known, bounds, seed):
    """Select four interior, four perimeter and two near-base samples using geometry alone."""
    from .intergrid_matched_route_plan import sample_layouts

    available = [row for row in sample_layouts(known, geometry, seed=seed)
                 if all(lo <= point <= hi for point, (lo, hi) in zip(row["bucket_xy_m"], bounds, strict=True))]
    gx, gy = zip(*grid.values(), strict=True)
    x0, x1, y0, y1 = min(gx), max(gx), min(gy), max(gy)
    targets = [
        ("southwest", "rectangle_center", .25, .25), ("southeast", "rectangle_center", .75, .25),
        ("northwest", "rectangle_center", .25, .75), ("northeast", "rectangle_center", .75, .75),
        ("west grid edge", "grid_perimeter", 0., .5), ("east grid edge", "grid_perimeter", 1., .5),
        ("south grid edge", "grid_perimeter", .5, 0.), ("north grid edge", "grid_perimeter", .5, 1.),
    ]
    selected = []
    for region, stratum, u, v in targets:
        eligible = [row for row in available if row["stratum"] == stratum]
        _require(bool(eligible), f"No fresh {stratum} sample lies within the server placement limits")
        target = [x0 + u * (x1 - x0), y0 + v * (y1 - y0)]
        row = min(eligible, key=lambda item: math.dist(item["bucket_xy_m"], target))
        selected.append(dict(region=region, bucket_xy_m=row["bucket_xy_m"]))
        available.remove(row)
    near = [row for row in available if row["stratum"] == "near_fixed_base"]
    _require(len(near) == 2, "Both near-base samples must lie within the server placement limits")
    selected.extend(dict(region=f"near {name} fixed base", bucket_xy_m=row["bucket_xy_m"])
                    for name, row in zip(("dropper", "bouncer"), near, strict=True))
    occupied = list(known)
    for index, row in enumerate(selected, 1):
        xy = row["bucket_xy_m"]
        _require(all(math.dist(xy, other) >= .005 for other in occupied), "Placement is not distinct from known coordinates")
        row["case"] = f"{index:02d}"
        occupied.append(xy)
    return selected


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class Client:
    def __init__(self, origin):
        url = urlsplit(origin)
        _require(url.scheme == "http" and url.hostname in {"127.0.0.1", "::1", "localhost"}
                 and url.username is None and url.password is None and url.port is not None
                 and url.path in {"", "/"} and not url.query and not url.fragment,
                 "Specify the mjlab service's loopback HTTP origin and port")
        self.origin = origin.rstrip("/")
        self.opener = build_opener(ProxyHandler({}), _NoRedirect())

    def request(self, path="", *, body=None, expected=200):
        data = None if body is None else json.dumps(body, allow_nan=False).encode()
        request = Request(self.origin + API + path, data=data,
                          headers={"Accept": "application/json", "Content-Type": "application/json"})
        try:
            response = self.opener.open(request, timeout=10)
        except HTTPError as exc:
            response = exc
        with response:
            raw = response.read(2_000_001)
            _require(response.status == expected,
                     f"{request.get_method()} {path or '/'} returned HTTP {response.status}, expected {expected}")
            _require(len(raw) <= 2_000_000, "Interactive response exceeds the compact evidence limit")
        if expected == 409:
            return None
        value = json.loads(raw)
        _require(isinstance(value, dict), "Expected a JSON object from the interactive simulation endpoint")
        return value


def _bounds(capability):
    _require(capability.get("enabled") is True, "Interactive native mjlab simulation is disabled")
    bounds = [capability["bucket_bounds"][axis] for axis in ("x", "y")]
    _require(all(_vector(span, 2) and span[0] < span[1] for span in bounds), "Invalid tabletop placement limits")
    return bounds


def _poll(client, identity, xy, active, final, timeout, interval):
    deadline = time.monotonic() + timeout
    path = "/" + quote(identity, safe="")
    while True:
        status = client.request(path)
        _require((status["id"], status["bucket_xy_m"]) == (identity, xy), "Plan identity or requested bucket moved")
        state = status["state"]
        if state in final:
            return status
        _require(state in active, f"Unexpected interactive plan state {state!r}")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"Plan {identity!r} stayed {state!r} for {timeout:g} s")
        time.sleep(min(interval, remaining))


def verify_preview(preview, xy):
    _require(preview["bucket_xy_m"] == xy and preview["status"] == "feasible", "Feasible preview must use the requested coordinates")
    config = preview["configuration"]
    for key, size in (("board_center_m", 3), ("board_normal", 3), ("board_quaternion_wxyz", 4),
                      ("release_center_m", 3), ("q10", 6), ("q14", 6)):
        _require(_vector(config[key], size), f"Invalid or missing preview pose field {key}")
    for key in ("board_normal", "board_quaternion_wxyz"):
        _require(math.isclose(math.hypot(*config[key]), 1, abs_tol=2e-6), f"Preview {key} must specify a unit orientation")
    for key in ("link_positions10", "link_positions14"):
        links = config[key]
        _require(isinstance(links, list) and len(links) >= 2 and all(_vector(link, 3) for link in links),
                 f"Preview {key} must show the arm's link positions")
    _require(config["release_center_m"][2] > config["board_center_m"][2] and _finite(config["tilt_deg"])
             and _finite(config["gap_m"]) and config["gap_m"] > 0, "Expected a finite board tilt and release above the board")
    _require(preview["candidate_pool"] in {"base", "fallback"} and preview["route_intent"] in {"direct", "table"},
             "Preview should show the selected candidate pool and intended bounce route")
    _require(_checksum(preview["provenance"]["request_sha256"]), "Preview must bind its exact release request")


def verify_result(result, preview, xy):
    _require((result["status"], result["bucket_xy_m"], result["configuration"]) == ("completed", xy, preview["configuration"]),
             "Measured release must match the accepted preview and exact coordinates")
    label, capture, retention = result["outcome"], result["capture"], result["retention_s"]
    _require(label in OUTCOMES and capture is (label in CAPTURES) and _finite(retention) and retention >= 0,
             "Expected a consistent measured native outcome, capture label and retention time")
    _require(not capture or retention >= 1 - 1e-9, "A capture requires one continuous second inside the bucket")
    provenance = result["provenance"]
    expected = preview["provenance"]
    _require(all(provenance.get(key) == value for key, value in expected.items()),
             "The measured shot lost its frozen request or study provenance")
    release_id = provenance["native_release_id"]
    _require(provenance["substeps"] == 64 and _checksum(provenance["native_cache_sha256"])
             and isinstance(release_id, str) and bool(release_id), "Expected native64 evidence with a trace checksum and release ID")
    _require(result["scalar_score_audit_passed"] is True and result["scoring_version"] == expected["scoring_version"],
             "The measured score must pass its scalar audit with the advertised scoring version")
    timeline = result["trajectory"]
    _require(isinstance(timeline, list) and len(timeline) >= 2, "A release must provide a measured ball timeline")
    previous = -1
    for point in timeline:
        stamp = point["time"]
        _require(_finite(stamp) and 0 <= stamp <= 4 + 1e-9 and stamp > previous and _vector(point["ball"], 3),
                 "Expected finite, strictly ordered native ball samples in the four-second trial")
        previous = stamp
    _require(timeline[0]["time"] == 0
             and math.dist(timeline[0]["ball"], preview["configuration"]["release_center_m"]) <= 3e-6,
             "Ball timeline must start at time zero at the selected release center")


def _latency(values):
    return dict(samples=len(values), median_s=median(values), max_s=max(values)) if values else dict(samples=0)


def summarize(rows):
    counts = Counter(row.get("outcome", row["state"]) for row in rows)
    drops = sum(row.get("drop_sent", False) for row in rows)
    complete = sum(row["state"] == "completed" for row in rows)
    captures = sum(row.get("outcome") in CAPTURES for row in rows)
    return dict(positions=len(rows), drop_requests=drops, measured_drops=complete, captures=captures,
                no_feasible_setup=counts["no_setup"], outcomes=dict(counts),
                planning_wall_time=_latency([row["planning_s"] for row in rows if "planning_s" in row]),
                drop_wall_time=_latency([row["drop_s"] for row in rows if "drop_s" in row]))


def check_position(client, row, seen, expected_provenance, save, timeout, interval, wait_for_gpu=None):
    """Plan once and send at most one drop request. A miss is final."""
    xy = row["bucket_xy_m"]
    started = time.monotonic()
    created = client.request("/plan", body={"bucket_xy_m": xy}, expected=202)
    identity = created["id"]
    _require(isinstance(identity, str) and identity and identity not in seen, "New placement needs a unique plan identity")
    seen.add(identity)
    row.update(plan_id=identity, state="planning")
    save()
    status = _poll(client, identity, xy, PLANNING, {"ready", "no_setup"} | UNAVAILABLE, timeout, interval)
    row.update(state=status["state"], planning_s=time.monotonic() - started, message=status.get("message"))
    if status["state"] != "ready":
        if status["state"] == "no_setup" and status.get("preview") is not None:
            preview = status["preview"]
            _require(preview["status"] in {"unresolved", "blocked"} and preview["bucket_xy_m"] == xy
                     and preview["configuration"] is None, "A declined setup must not issue an arm configuration")
        return
    preview = status["preview"]
    verify_preview(preview, xy)
    _require(all(preview["provenance"].get(key) == value for key, value in expected_provenance.items()),
             "The preview does not match the intended simulation-only native64 study")
    path = f"/{quote(identity, safe='')}"
    row["preview"] = preview
    if wait_for_gpu is not None:
        row["state"] = "waiting_for_4090"
        save()
        row["gpu_wait_s"] = wait_for_gpu()
        ready = client.request(path)
        _require(ready["state"] == "ready" and ready["preview"] == preview,
                 "The planned shot changed before the 4090 became available")
    row.update(state="drop_requested", drop_sent=True)
    save()
    started = time.monotonic()
    client.request(path + "/drop", body={}, expected=202)
    terminal = _poll(client, identity, xy, RELEASING, {"completed"} | UNAVAILABLE, timeout, interval)
    row.update(state=terminal["state"], drop_s=time.monotonic() - started, message=terminal.get("message"))
    if terminal["state"] == "completed":
        result = terminal["result"]
        verify_result(result, preview, xy)
        row.update(result=result, outcome=result["outcome"])


def verify_duplicate(client, row):
    identity = quote(row["plan_id"], safe="")
    client.request(f"/{identity}/drop", body={}, expected=409)
    after = client.request(f"/{identity}")
    _require((after["id"], after["state"], after["bucket_xy_m"], after["result"])
             == (row["plan_id"], "completed", row["bucket_xy_m"], row["result"]),
             "The duplicate request changed the completed plan")


def run(args):
    geometry, grid, known, source = historical_layouts(args.study, args.audit)
    client = Client(args.base_url)
    capability = client.request()
    placements = sample_positions(geometry, grid, known, _bounds(capability), args.seed)
    expected_provenance = dict(scene_sha256=geometry["scene_sha256"], study_audit_sha256=source["audit_sha256"],
                               simulation_only=True, substeps=64, required_retention_s=1.)
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    frozen = dict(version=VERSION, seed=args.seed, historical_source=source, history_positions=len(known),
                  history_sha256=_digest(known), placements_sha256=_digest(placements),
                  scene_sha256=geometry["scene_sha256"], minimum_separation_m=.005, placements=placements)
    _save(output / "positions.json", frozen)
    position_hash = sha256((output / "positions.json").read_bytes()).hexdigest()
    report = dict(version=VERSION, status="running", positions_sha256=position_hash,
                  results=[], summary=summarize([]), duplicate_drop_rejected=False)

    def save():
        report["summary"] = summarize(report["results"])
        _save(output / "report.json", report)

    def wait_for_gpu():
        from .gpu_slot import gpu_slot

        started = time.monotonic()
        with gpu_slot(args.gpu_lock, timeout_s=args.gpu_wait_s, poll_s=5):
            return time.monotonic() - started

    seen = set()
    try:
        for placement in placements:
            row = dict(placement, state="pending")
            report["results"].append(row)
            check_position(client, row, seen, expected_provenance, save, args.timeout_s, args.poll_s,
                           wait_for_gpu if args.gpu_lock else None)
            if row["state"] == "completed" and not report["duplicate_drop_rejected"]:
                verify_duplicate(client, row)
                report["duplicate_drop_rejected"] = True
            save()
            print(json.dumps({key: value for key, value in row.items() if key not in {"preview", "result"}}), flush=True)
            _require(row["state"] in {"completed", "no_setup"}, "The plan or drop failed or expired")
        _require(report["duplicate_drop_rejected"], "No placement produced a measured release")
        release_ids = [row["result"]["provenance"]["native_release_id"] for row in report["results"] if row["state"] == "completed"]
        _require(len(set(release_ids)) == len(release_ids), "Each completed placement needs its own measured release evidence")
        _require(sha256((output / "positions.json").read_bytes()).hexdigest() == position_hash,
                 "Bucket placements changed after freezing")
        report["status"] = "passed"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        save()
    return report["summary"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18781")
    for name in ("study", "audit", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--timeout-s", type=float, default=600.)
    parser.add_argument("--poll-s", type=float, default=.75)
    parser.add_argument("--gpu-lock", type=Path, help="Wait for the 4090 before sending each drop request")
    parser.add_argument("--gpu-wait-s", type=float, default=3600.)
    args = parser.parse_args()
    if args.seed < 0 or not all(_finite(value) and value > 0 for value in (args.timeout_s, args.poll_s, args.gpu_wait_s)):
        parser.error("Require a nonnegative seed and positive finite timeouts")
    if args.output.exists() or args.output.resolve().is_relative_to(args.study.resolve()):
        parser.error("Choose a new output directory outside the historical study")
    print(json.dumps(run(args)), flush=True)


if __name__ == "__main__":
    main()
