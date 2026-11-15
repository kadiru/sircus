"""Account for repeated and alternative shots at previously missed positions."""
from __future__ import annotations

from collections import Counter

from .intergrid_matched_route_study import outcome
from .intergrid_route_plan import ROUTES
from .intergrid_route_selector_plan import STANDARD_CAPTURES


def _counts(shots):
    outcomes = Counter(shot["outcome"] for shot in shots)
    return dict(releases=len(shots), captures=sum(outcomes[label] for label in STANDARD_CAPTURES),
                outcomes=dict(outcomes))


def summarize(plan, observations):
    """Compare novel alternatives with freshly repeated baselines, by position."""
    cases = plan["cases"]
    requests = [request for case in cases for request in case["requests"]]
    identifiers = {request["id"] for request in requests}
    if len(identifiers) != len(requests) or observations.keys() != identifiers:
        raise ValueError("Expected one native release for every frozen replay request")
    if len({case["layout"]["id"] for case in cases}) != len(cases):
        raise ValueError("Replay positions must be unique")

    baselines, novel_shots, cross_checks, case_rows = [], [], [], []
    novel_positions = Counter()
    no_alternatives = no_novel_alternatives = 0
    labels = {(True, True): "both_capture", (True, False): "repeat_only",
              (False, True): "novel_only", (False, False): "neither_capture"}
    paired_novel = dict.fromkeys(labels.values(), 0)
    for case in cases:
        layout = case["layout"]
        candidates = {candidate["id"]: candidate for candidate in case["candidates"]}
        if (len(candidates) != len(case["candidates"])
                or candidates.keys() != {request["id"] for request in case["requests"]}):
            raise ValueError(f"Replay metadata does not match frozen requests at {layout['id']}")
        baseline = None
        alternatives = []
        for request in case["requests"]:
            name, method, route = (request[key] for key in ("id", "method", "intended_route"))
            row, candidate = observations[name], candidates[name]
            if (row["native_release"] is not True or row["id"] != name or row["method"] != method
                    or route not in ROUTES or row["intended_route"] != route):
                raise ValueError(f"Native outcome does not match frozen request {name}")
            already = candidate["already_direct_first"]
            prior_capture = candidate["prior_direct_first_capture"]
            if type(already) is not bool or (type(prior_capture) is not bool if already else prior_capture is not None):
                raise ValueError(f"Invalid prior direct-first comparison metadata for {name}")
            shot = dict(id=name, intended_route=route, outcome=outcome(row), already_direct_first=already,
                        prior_direct_first_capture=prior_capture)
            if method == "baseline_repeat" and baseline is None:
                baseline = shot
            elif method == "alternative":
                alternatives.append(shot)
                (cross_checks if already else novel_shots).append(shot)
            else:
                raise ValueError(f"Unexpected replay method or duplicate baseline at {layout['id']}")
        if baseline is None:
            raise ValueError(f"No frozen baseline repeat at {layout['id']}")
        baselines.append(baseline)
        novel = [shot for shot in alternatives if not shot["already_direct_first"]]
        no_alternatives += int(not alternatives)
        no_novel_alternatives += int(not novel)
        if novel:
            routes = {shot["intended_route"] for shot in novel if shot["outcome"] in STANDARD_CAPTURES}
            original_route = baseline["intended_route"]
            novel_positions["positions_tested"] += 1
            novel_positions["positions_captured"] += int(bool(routes))
            novel_positions["same_route_capture_positions"] += int(original_route in routes)
            novel_positions["other_route_capture_positions"] += int(bool(routes - {original_route}))
            paired_novel[labels[baseline["outcome"] in STANDARD_CAPTURES, bool(routes)]] += 1
        case_rows.append(dict(layout=layout, prior_selector_id=case["prior_selector_id"],
                              prior_direct_first_id=case["prior_direct_first_id"], baseline_repeat=baseline,
                              alternatives=alternatives))

    return dict(
        positions=len(cases), no_alternative_positions=no_alternatives,
        no_novel_alternative_positions=no_novel_alternatives,
        baseline_repeat=dict(_counts(baselines), matches_prior_direct_first=sum(
            shot["already_direct_first"] for shot in baselines)),
        novel_alternatives=dict(_counts(novel_shots), **{
            key: novel_positions[key] for key in ("positions_tested", "positions_captured",
                                                 "same_route_capture_positions", "other_route_capture_positions")
        }),
        prior_direct_first_replays=dict(_counts(cross_checks), previous_captures=sum(
            shot["prior_direct_first_capture"] for shot in cross_checks), capture_disagreements=sum(
            shot["prior_direct_first_capture"] != (shot["outcome"] in STANDARD_CAPTURES) for shot in cross_checks)),
        paired_novel=paired_novel, setup_failures=plan["setup_failures"], cases=case_rows,
    )
