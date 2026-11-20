"""Screen a pre-release flight ranker on complete menus from audited replays."""

from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import json
from pathlib import Path

from .benchmark_verify import digest, read, sha
from .intergrid_candidate_replay_audit import STUDY_VERSION, VERSION as AUDIT_VERSION
from .intergrid_flight_ranker import COEFFICIENTS, RULE_VERSION, FlightRanker, rank_key
from .intergrid_matched_route_study import outcome
from .intergrid_rebound_calibration import fit_rebound_calibration
from .intergrid_route_refit_inputs import _audited_source
from .intergrid_route_selector_plan import STANDARD_CAPTURES
from .parallel_capture_grid_audit import observed_rows
from .sweep_worker import atomic


VERSION = "sircus-intergrid-flight-ranking-screen-v1"


def evaluate(cases, predictions, observations):
    report = []
    uniform = Fraction()
    within_position_agreement = []
    for case, scores in zip(cases, predictions, strict=True):
        menu = case["requests"]
        if [request["id"] for request in menu] != [row["id"] for row in scores]:
            raise ValueError("Frozen predictions differ from the replay menu")
        ranked = sorted(scores, key=lambda row: rank_key(row, row["score_m"]))
        ranked = [dict(row, outcome=outcome(observations[row["id"]])) for row in ranked]
        positive = [row for row in ranked if row["outcome"] in STANDARD_CAPTURES]
        negative = [row for row in ranked if row["outcome"] not in STANDARD_CAPTURES]
        uniform += Fraction(len(positive), len(ranked))
        if positive and negative:
            wins = 0.
            for a in positive:
                a_key = rank_key(a, a["score_m"])[:2]
                for b in negative:
                    b_key = rank_key(b, b["score_m"])[:2]
                    wins += (a_key < b_key) + .5 * (a_key == b_key)
            within_position_agreement.append(wins / (len(positive) * len(negative)))
        report.append(dict(layout=case["layout"], candidates=ranked, selected=ranked[0]["id"],
                           selected_outcome=ranked[0]["outcome"], possible_captures=len(positive)))
    selected = Counter(row["selected_outcome"] for row in report)
    captures = sum(selected[label] for label in STANDARD_CAPTURES)
    return dict(
        positions=len(report), attempts=sum(len(row["candidates"]) for row in report),
        captures=captures, uniform_expected_captures=float(uniform),
        mixed_positions=len(within_position_agreement),
        mean_within_position_auc=(sum(within_position_agreement) / len(within_position_agreement)
                                  if within_position_agreement else None),
        selected_outcomes=dict(selected), advance=captures > uniform, cases=report,
    )


def run(args):
    replay, receipt = args.replay_study, args.replay_audit
    _, _, audit, binding, contract = _audited_source(
        replay, receipt, STUDY_VERSION, ("replay-plan.json", "calibration.json", "rows.jsonl"),
    )
    if audit["version"] != AUDIT_VERSION or audit["replay_plan_sha256"] != binding["replay-plan.json"]:
        raise ValueError("Replay receipt does not bind the screened study")
    if contract["fallback_study"]["audit"] != sha(args.fallback_audit):
        raise ValueError("Calibration must use the source of the archived replay")
    plan = read(replay / "replay-plan.json")
    calibration = fit_rebound_calibration(args.fallback_study, args.fallback_audit, plan)
    rule = dict(version=RULE_VERSION, coefficients={key: calibration[key] for key in COEFFICIENTS})
    ranker = FlightRanker(contract["scene"], read(replay / "calibration.json")["score_config"],
                         args.assets, rule)
    args.output.mkdir(parents=True, exist_ok=False)
    predictions = [[dict(id=request["id"], **ranker.predict(request)) for request in case["requests"]]
                   for case in plan["cases"]]
    replay_binding = dict(evidence_manifest_sha256=binding["evidence-manifest.json"],
                          report_sha256=binding["report.json"], audit_sha256=binding["audit"])
    frozen = dict(version=VERSION, rule=rule, rebound_calibration=calibration,
                  replay_binding=replay_binding, predictions=predictions)
    atomic(args.output / "predictions.json", frozen)
    rows = [row for row in observed_rows(replay / "rows.jsonl") if row["phase"] == "test"]
    observations = {row["id"]: row for row in rows}
    if sha(replay / "rows.jsonl") != binding["rows.jsonl"] or len(rows) != len(observations) or observations.keys() != {
        row["id"] for case in plan["cases"] for row in case["requests"]
    }:
        raise ValueError("Replay observations differ from the audited menu")
    report = dict(version=VERSION, rule=rule, replay_binding=replay_binding,
                  predictions_sha256=sha(args.output / "predictions.json"), rule_sha256=digest(rule),
                  **evaluate(plan["cases"], predictions, observations))
    atomic(args.output / "report.json", report)
    return {key: value for key, value in report.items() if key != "cases"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("replay-study", "replay-audit", "fallback-study", "fallback-audit", "assets", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
