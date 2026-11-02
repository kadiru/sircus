"""Read-only, loopback report for the bounded 35-table shot-learning study.

Reads only named aggregate JSON files. It never imports Isaac, Torch, Viam or
robot-control modules. Raw telemetry, credentials and arbitrary paths are not
served. Missing observations remain unknown rather than becoming zero scores.
"""

import argparse
import copy
import datetime as dt
import json
import math
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

SCHEMA = "sircus-farm-learning-view-v1"
CELL_IDS = tuple(f"r{row}c{column}" for row in range(1, 6) for column in range(1, 8))
STAGES = (
    "sweep",
    "train",
    "best_repeat",
    "candidate_initial_stochastic",
    "candidate_initial_deterministic",
    "candidate_final_stochastic",
    "candidate_final_deterministic",
    "legacy_stochastic",
    "legacy_deterministic",
)
COUNT_KEYS = ("attempts", "releases", "invalid", "captures", "clean", "single_bounce")
MEAN_KEYS = ("mean_learning_reward", "mean_historical_reward")
ACTION_KEYS = (
    "board_x_m",
    "board_y_m",
    "height_m",
    "gap_m",
    "tilt_deg",
    "radius_m",
    "azimuth_deg",
    "azimuth_offset_deg",
)
DEFAULT_BASELINE = (
    Path(__file__).resolve().parents[2]
    / "baselines/isaac-expanded-farm-20261102/cell-summary.json"
)
HTML = Path(__file__).with_name("farm_learning.html")
MAX_BYTES = 16 * 1024 * 1024


def finite_number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Invalid {label}")
    if not math.isfinite(value):
        raise ValueError(f"Nonfinite {label}")
    return value


def ratio(captures, attempts):
    """An absent/empty sample has no measured capture rate."""
    if captures is None or attempts in (None, 0):
        return None
    return captures / attempts


def clean_stats(raw):
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("Stage statistics must be a mapping")
    result = {}
    for key in COUNT_KEYS:
        value = raw.get(key)
        if value is not None:
            if type(value) is not int or not 0 <= value <= 10**9:
                raise ValueError(f"Invalid {key} count")
            result[key] = value
    for key in MEAN_KEYS:
        value = raw.get(key)
        if value is not None:
            result[key] = finite_number(value, key)
    if all(key in result for key in ("attempts", "releases", "invalid")):
        if result["releases"] + result["invalid"] != result["attempts"]:
            raise ValueError("Release/invalid counts do not match attempts")
    for child, parent in (
        ("releases", "attempts"),
        ("captures", "releases"),
        ("clean", "captures"),
        ("single_bounce", "captures"),
    ):
        if child in result and parent in result and result[child] > result[parent]:
            raise ValueError(f"{child} exceeds {parent}")
    result["capture_rate"] = ratio(result.get("captures"), result.get("attempts"))
    return result


def coordinates(value):
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("Expected two bucket coordinates")
    return [finite_number(v, "bucket coordinate") for v in value]


def clean_action(raw):
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("Expected a reported action mapping")
    result = {
        key: finite_number(raw[key], f"action {key}")
        for key in ACTION_KEYS
        if key in raw
    }
    if "height_m" in result and "gap_m" in result:
        # This is the configured ball-center release height, not measured FK.
        result["ball_center_release_m"] = result["height_m"] + result["gap_m"] + 0.0375
    return result or None


def baseline_cells(document):
    if not isinstance(document, dict) or not isinstance(document.get("cells"), list):
        raise ValueError("Baseline is missing its cell list")
    result = {}
    for raw in document["cells"]:
        if not isinstance(raw, dict):
            raise ValueError("Baseline cells must be mappings")
        identity = raw.get("cell_id")
        if identity not in CELL_IDS or identity in result:
            raise ValueError("Baseline has an unknown or repeated table ID")
        attempts, releases = raw.get("attempts"), raw.get("releases")
        stats = clean_stats(
            dict(
                attempts=attempts,
                releases=releases,
                invalid=(
                    attempts - releases
                    if type(attempts) is int and type(releases) is int
                    else None
                ),
                captures=raw.get("qualified_captures"),
                clean=raw.get("clean_captures"),
                single_bounce=raw.get("single_bounce_captures"),
            )
        )
        result[identity] = dict(
            bucket_xy_m=coordinates(raw.get("bucket_xy_m")), stats=stats
        )
    if set(result) != set(CELL_IDS):
        raise ValueError("Baseline must contain all 35 table IDs")
    if len({tuple(row["bucket_xy_m"]) for row in result.values()}) != 35:
        raise ValueError("Baseline bucket coordinates are not distinct")
    return result


def normalize_state(document, baseline, complete=False, updated_utc=None):
    """Keep aggregate report fields; reject mismatched table/bucket identities."""
    if not isinstance(document, dict):
        raise ValueError("Expected a study aggregate mapping")
    previous = baseline_cells(baseline)
    expected_source = baseline.get("source_fingerprint")
    if (
        expected_source is not None
        and document.get("source_fingerprint") is not None
        and document["source_fingerprint"] != expected_source
    ):
        raise ValueError("Study scene fingerprint differs from its comparison baseline")
    raw_cells = document.get("per_cell", [])
    if not isinstance(raw_cells, list):
        raise ValueError("Expected per_cell aggregates")
    current = {}
    for raw in raw_cells:
        if not isinstance(raw, dict):
            raise ValueError("Study cells must be mappings")
        identity = raw.get("cell_id")
        if identity not in CELL_IDS or identity in current:
            raise ValueError("Unknown or repeated study table ID")
        xy = coordinates(raw.get("bucket_xy_m"))
        if xy != previous[identity]["bucket_xy_m"]:
            raise ValueError("Study bucket position differs from its baseline")
        stages = raw.get("stages", {})
        if not isinstance(stages, dict):
            raise ValueError("Expected per-stage aggregate mappings")
        current[identity] = dict(
            stages={
                name: clean_stats(stages[name]) for name in STAGES if name in stages
            },
            best_sweep_action=clean_action(raw.get("best_sweep_action")),
        )
    if complete and set(current) != set(CELL_IDS):
        raise ValueError("Completed report is missing tables")
    cells = [
        dict(
            cell_id=identity,
            bucket_xy_m=copy.deepcopy(previous[identity]["bucket_xy_m"]),
            baseline=previous[identity]["stats"],
            stages=current.get(identity, {}).get("stages", {}),
            best_sweep_action=current.get(identity, {}).get("best_sweep_action"),
        )
        for identity in CELL_IDS
    ]
    aggregates = document.get("summary", document.get("summaries", {}))
    if not isinstance(aggregates, dict):
        raise ValueError("Expected stage summary mappings")
    summary = {
        name: clean_stats(aggregates[name]) for name in STAGES if name in aggregates
    }
    phase = document.get("phase", "completed" if complete else "waiting")
    if not isinstance(phase, str) or len(phase) > 300:
        raise ValueError("Invalid study phase")
    fingerprints = {}
    for key in ("study_fingerprint", "source_fingerprint", "version"):
        value = document.get(key)
        if value is not None:
            if not isinstance(value, str) or len(value) > 160:
                raise ValueError("Invalid study identity")
            fingerprints[key] = value
    return dict(
        schema=SCHEMA,
        complete=bool(complete),
        phase=phase,
        updated_utc=updated_utc,
        identity=fingerprints,
        summary=summary,
        per_cell=cells,
        scope="Simulation development study. Search/best-setting repeats are separate from frozen learned-policy evaluation. Same bucket coordinates do not establish unseen-layout generalization.",
    )


def read_json(path, parent=None):
    path = Path(path)
    if parent is not None and path.resolve().parent != Path(parent).resolve():
        raise ValueError("Report file resolves outside its explicit output directory")
    if path.stat().st_size > MAX_BYTES:
        raise ValueError("Aggregate JSON exceeds the report size limit")
    return json.loads(path.read_text())


def read_state(output, baseline=DEFAULT_BASELINE):
    output = Path(output).resolve()
    previous = read_json(baseline)
    result = output / "farm-study-result.json"
    progress = output / "progress.json"
    contract = output / "farm-study-contract.json"
    path = result if result.is_file() else progress if progress.is_file() else None
    document = read_json(path, output) if path is not None else {}
    if contract.is_file():
        identity = read_json(contract, output)
        document = dict(document)
        for key in ("version", "source_fingerprint"):
            if document.get(key) is not None and document[key] != identity.get(key):
                raise ValueError("Aggregate identity differs from its study contract")
            document.setdefault(key, identity.get(key))
        if document.get("study_fingerprint") is not None and document[
            "study_fingerprint"
        ] != identity.get("fingerprint"):
            raise ValueError("Aggregate fingerprint differs from its study contract")
        document.setdefault("study_fingerprint", identity.get("fingerprint"))
    updated = (
        dt.datetime.fromtimestamp(path.stat().st_mtime, dt.timezone.utc).isoformat()
        if path is not None
        else None
    )
    return normalize_state(
        document, previous, complete=path == result, updated_utc=updated
    )


def render_page(state):
    # Escape the HTML parser's script terminator, not just JavaScript quotes.
    payload = json.dumps(state, allow_nan=False).replace("<", "\\u003c")
    payload = payload.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    return HTML.read_text().replace("__SIRCUS_STATE__", payload).encode()


def make_handler(output, baseline=DEFAULT_BASELINE):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.respond(False)

        def do_HEAD(self):
            self.respond(True)

        def respond(self, head):
            path = urlsplit(self.path).path
            if path not in ("/", "/farm_learning.html", "/api/state"):
                self.send_error(404)
                return
            try:
                state = read_state(output, baseline)
                body = (
                    json.dumps(state, allow_nan=False).encode()
                    if path == "/api/state"
                    else render_page(state)
                )
            except (OSError, ValueError, TypeError, KeyError):
                self.send_error(503, "Aggregate report unavailable or inconsistent")
                return
            self.send_response(200)
            self.send_header(
                "Content-Type",
                (
                    "application/json; charset=utf-8"
                    if path == "/api/state"
                    else "text/html; charset=utf-8"
                ),
            )
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; img-src 'self' data:; base-uri 'none'; frame-ancestors 'none'",
            )
            self.end_headers()
            if not head:
                self.wfile.write(body)

        def log_message(self, *_args):
            pass

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--port", type=int, default=18785)
    parser.add_argument(
        "--export", type=Path, help="Write a self-contained HTML snapshot and exit"
    )
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error("Port must be in0..65535")
    if args.export is not None:
        args.export.write_bytes(render_page(read_state(args.output, args.baseline)))
        return
    server = ThreadingHTTPServer(
        ("127.0.0.1", args.port), make_handler(args.output, args.baseline)
    )
    print(f"Read-only farm report: http://127.0.0.1:{server.server_port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
