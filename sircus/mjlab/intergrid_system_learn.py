"""Fit a capture ranker on complete development menus and validate by bucket site."""

from __future__ import annotations

from hashlib import sha256
import json
import math


VERSION = "sircus-intergrid-pairwise-flight-capture-v1"
FOLDS = 5
L2 = 0.15
ITERATIONS = 160
MARGIN_SCALE_M = 0.035
ROUTES = ("direct", "table")
PERIODIC = ("heading", "twist") + tuple(
    f"arm{arm}_joint{joint}" for arm in (10, 14) for joint in range(1, 7)
)
FEATURE_NAMES = (
    "relative_bucket_x", "relative_bucket_y", "board_bucket_distance",
    "board_height", "tilt", "release_gap", "table_intent", "table_prediction",
) + tuple(
    f"{route}_{kind}_{feature}"
    for route in ROUTES
    for kind in ("rim", "arm")
    for feature in ("margin", "missing")
) + tuple(f"{function}_{angle}" for angle in PERIODIC for function in ("sin", "cos"))


def _digest(value):
    return sha256(json.dumps(value, allow_nan=False, sort_keys=True).encode()).hexdigest()


def _units(value, scale):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Expected finite pitch, distance, clearance and release geometry")
    return max(-4., min(4., value / scale))


def _features(request, prediction):
    x, y, height, tilt, gap = request["values"]
    bucket_x, bucket_y = request["bucket_xy_m"]
    dx, dy = bucket_x - x, bucket_y - y
    vector = [
        _units(dx, .30), _units(dy, .30), _units(math.hypot(dx, dy), .30),
        _units(height, .30), _units(tilt, 30), _units(gap, .30),
        float(request["intended_route"] == "table"), float(prediction["predicted_route"] == "table"),
    ]
    for route in ROUTES:
        for kind in ("rim", "arm"):
            margin = (prediction[route] or {}).get(f"{kind}_clearance_m")
            vector.extend((0. if margin is None else _units(margin, MARGIN_SCALE_M), float(margin is None)))
    q10, q14 = request["selected_q10"], request["selected_q14"]
    if len(q10) != 6 or len(q14) != 6:
        raise ValueError("A candidate must describe both six-joint poses")
    angles = (
        math.radians(request["board_heading_offset_deg"]),
        math.radians(request["board_twist_deg"]),
        *q10, *q14,
    )
    for angle in angles:
        vector.extend((math.sin(angle), math.cos(angle)))
    if not all(map(math.isfinite, vector)):
        raise ValueError("Nonfinite pre-release feature")
    return vector


def _prior(score_m):
    # Preserve the order of finite predictions and give missing predictions a
    # distinct fallback. Compress extreme margins before forming pair losses.
    return -5. if score_m is None else 3. * math.atan(float(score_m) / MARGIN_SCALE_M)


def _analytic_key(identifier, score):
    return (score is None, -score if score is not None else 0., identifier)


def _encoded(admitted, predictions):
    data = []
    for request, prediction in zip(admitted, predictions, strict=True):
        identifier = request["id"]
        if identifier != prediction["id"] or not isinstance(identifier, str) or not identifier:
            raise ValueError("Frozen physics predictions must match admitted candidate IDs in order")
        score = prediction["score_m"]
        if score is not None and not math.isfinite(score):
            raise ValueError("Expected a finite analytic margin")
        data.append((_analytic_key(identifier, score), _prior(score), _features(request, prediction)))
    if len({item[0][2] for item in data}) != len(data):
        raise ValueError("A candidate menu must use distinct request IDs")
    return data


def _learned_key(candidate, weights):
    analytic_key, prior, features = candidate
    correction = math.fsum(w * x for w, x in zip(weights, features, strict=True))
    return (-(prior + correction), analytic_key)


def learned_order(admitted, predictions, checkpoint):
    """Apply the frozen weights to requests without using any rollout observations."""
    if (checkpoint["version"] != VERSION or checkpoint["feature_names"] != list(FEATURE_NAMES)
            or checkpoint["margin_scale_m"] != MARGIN_SCALE_M
            or len(checkpoint["weights"]) != len(FEATURE_NAMES)
            or not all(map(math.isfinite, checkpoint["weights"]))):
        raise ValueError("Unsupported or invalid learned ranker checkpoint")
    annotated = zip(admitted, _encoded(admitted, predictions), strict=True)
    return [request for request, _ in sorted(
        annotated, key=lambda item: _learned_key(item[1], checkpoint["weights"]),
    )]


def _pairs(menus):
    """Construct capture-vs-miss pairs and give each informative site equal influence."""
    comparisons = []
    for menu in menus:
        positive = [candidate for candidate, label in menu if label is True]
        negative = [candidate for candidate, label in menu if label is False]
        n_pairs = len(positive) * len(negative)
        if not n_pairs:
            continue
        for _, win_prior, win in positive:
            for _, loss_prior, loss in negative:
                delta = [a - b for a, b in zip(win, loss, strict=True)]
                comparisons.append((win_prior - loss_prior, delta, 1 / n_pairs))
    return comparisons


def _fit(menus):
    pairs = _pairs(menus)
    weights = [0.] * len(FEATURE_NAMES)
    effective_sites = math.fsum(weight for _, _, weight in pairs)
    if not pairs:
        return weights, 0, 0, None, None
    pairs = [(prior, delta, weight / effective_sites) for prior, delta, weight in pairs]
    # A global upper bound on the curvature gives a fixed, stable descent step.
    step = 1 / (L2 + .25 * math.fsum(
        weight * math.fsum(x * x for x in delta) for _, delta, weight in pairs
    ))
    initial = final = 0.
    for iteration in range(ITERATIONS + 1):
        gradient = [L2 * value for value in weights]
        weighted_losses = []
        for prior, delta, weight in pairs:
            margin = prior + math.fsum(w * x for w, x in zip(weights, delta, strict=True))
            bounded_margin = max(-100., min(100., margin))
            dloss = -weight / (1 + math.exp(bounded_margin))
            for j, x in enumerate(delta):
                gradient[j] += dloss * x
            if iteration in (0, ITERATIONS):
                weighted_losses.append(weight * (max(-margin, 0.) + math.log1p(math.exp(-abs(margin)))))
        if iteration == 0:
            initial = math.fsum(weighted_losses)
        elif iteration == ITERATIONS:
            final = math.fsum(weighted_losses) + .5 * L2 * math.fsum(w * w for w in weights)
        else:
            weights = [w - step * derivative for w, derivative in zip(weights, gradient, strict=True)]
    return ([round(w, 12) for w in weights], len(pairs), round(effective_sites),
            round(initial, 12), round(final, 12))


def _labels(observations):
    labels = {}
    for identifier, row in observations.items():
        if (row["id"] != identifier or row["native_release"] is not True
                or type(row["invalid_physics"]) is not bool or type(row["route_success"]) is not bool):
            raise ValueError("Expected a scored native release for each admitted development command")
        labels[identifier] = None if row["invalid_physics"] else row["route_success"]
    return labels


def fit_ranker(cases, predictions, observations):
    """Select a ranker using site-grouped out-of-fold capture, then refit development only.

    Each candidate was released in the development simulator. Invalid physics
    provides no training label and counts as zero capture when selected.
    A missing candidate menu remains a setup refusal for both selectors.
    """
    labels = _labels(observations)
    sites, ids, seen = [], set(), set()
    for case, predicted in zip(cases, predictions, strict=True):
        layout, requests = case["layout"], case["requests"]
        xy = tuple(round(float(value), 10) for value in layout["bucket_xy_m"])
        identity = layout["id"]
        if (len(xy) != 2 or not all(map(math.isfinite, xy)) or xy in seen
                or not isinstance(identity, str) or not identity or identity in ids
                or any(tuple(row["bucket_xy_m"]) != tuple(layout["bucket_xy_m"]) for row in requests)):
            raise ValueError("Development menus require one distinct matched layout per bucket position")
        seen.add(xy)
        ids.add(identity)
        sites.append((layout, xy, _encoded(requests, predicted)))
    manifest = [row[0][2] for _, _, candidates in sites for row in candidates]
    if len(set(manifest)) != len(manifest) or set(manifest) != labels.keys():
        raise ValueError("Development releases must cover every admitted menu exactly once")
    if len(sites) < FOLDS:
        raise ValueError(f"At least {FOLDS} distinct development positions are required")

    sites.sort(key=lambda item: (_digest(item[1]), item[1]))
    menus = [[(candidate, labels[candidate[0][2]]) for candidate in candidates]
             for _, _, candidates in sites]
    fold_rows, results = [], []
    for fold in range(FOLDS):
        training = [menu for index, menu in enumerate(menus) if index % FOLDS != fold]
        weights, n_pairs, n_sites, initial, final = _fit(training)
        fold_rows.append(dict(fold=fold, training_positions=len(training), informative_positions=n_sites,
                              training_pairs=n_pairs, initial_loss=initial, final_loss=final))
        for index in range(fold, len(sites), FOLDS):
            layout, _, candidates = sites[index]
            analytic_id = min(candidates, key=lambda candidate: candidate[0])[0][2] if candidates else None
            learned_id = min(candidates, key=lambda candidate: _learned_key(candidate, weights))[0][2] if candidates else None
            results.append(dict(
                id=layout["id"], bucket_xy_m=layout["bucket_xy_m"], fold=fold,
                candidates=len(candidates), analytic_id=analytic_id, learned_id=learned_id,
                analytic_capture=labels.get(analytic_id) is True, learned_capture=labels.get(learned_id) is True,
            ))
    analytic_captures = sum(row["analytic_capture"] for row in results)
    learned_captures = sum(row["learned_capture"] for row in results)
    input_digest = _digest(dict(cases=cases, predictions=predictions, labels=labels))
    validation = dict(
        version=VERSION, positions=len(sites), releases=len(labels),
        invalid_physics=sum(label is None for label in labels.values()),
        analytic_captures=analytic_captures, learned_captures=learned_captures,
        advance=learned_captures > analytic_captures, folds=fold_rows,
        evidence_sha256=input_digest, cases=sorted(results, key=lambda row: row["id"]),
    )
    checkpoint = None
    if validation["advance"]:
        weights, n_pairs, n_sites, initial, final = _fit(menus)
        checkpoint = dict(
            version=VERSION, feature_names=list(FEATURE_NAMES), margin_scale_m=MARGIN_SCALE_M,
            iterations=ITERATIONS, l2=L2, weights=weights,
            development_positions=len(sites), development_pairs=n_pairs, informative_positions=n_sites,
            initial_loss=initial, final_loss=final, evidence_sha256=input_digest,
        )
    return dict(checkpoint=checkpoint, validation=validation)
