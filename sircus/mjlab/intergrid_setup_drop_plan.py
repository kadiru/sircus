"""Freeze exact native drop requests from the five audited setup recoveries."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from .benchmark_verify import contained, sha
from .intergrid_candidate_replay_plan import command_key
from .intergrid_setup_recovery import VERSION as RECOVERY_VERSION, verify


VERSION = "sircus-intergrid-setup-drop-plan-v1"
PLAN_SHA256 = "f03f2033e895a1c7358589258cb344f4b5cb0c66dde5b4f735568cbcc912b105"
AUDIT_SHA256 = "50b27bbf23bf260e56f4c77e33f02c5dee39db3244fece646857f4d1658630ad"
PREPARATION_SHA256 = {
    "intergrid-route-fallback-test-29": "67a725abd3826d0990658663661d69689eb2bff94941ca7b92fdefe5a9a7ef61",
    "intergrid-route-fallback-test-31": "ba1c886317dcfc0c27858b354c28646ab1f046115f609d000f778fe4c8873773",
    "intergrid-route-fallback-test-33": "df11a463a49706280970eee7e5e55780e9c57c4de39d2dde2d34251c00af6aaf",
    "intergrid-route-fallback-test-81": "75d5c098a40d6862475c13424353ce7665f2bc20fb4a87cf87fc1b70ab7607c2",
    "intergrid-route-fallback-test-90": "2fb1f4eb4e2f35ecb446ef2c813de26a6e5ae56903588ef1928c7e1704efa91e",
}
REQUESTS = 43


def _bound_json(path, expected):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"Setup recovery input differs from its bound bytes: {path.name}")
    return json.loads(data)


def load_setup_drop_plan(setup_study, setup_audit, fallback_study, parent_study):
    """Return the full, ordered admission menu after verifying the CPU evidence."""
    setup_study, setup_audit, fallback_study, parent_study = map(
        Path, (setup_study, setup_audit, fallback_study, parent_study),
    )
    archived = _bound_json(contained(setup_study, "plan.json"), PLAN_SHA256)
    receipt = _bound_json(setup_audit, AUDIT_SHA256)
    layouts = [case["previous"]["layout"] for case in archived["cases"]]
    if (archived["version"] != RECOVERY_VERSION
            or [layout["id"] for layout in layouts] != list(PREPARATION_SHA256)
            or any(case["layout"] != dict(layout, split="train")
                   for case, layout in zip(archived["cases"], layouts, strict=True))):
        raise ValueError("Expected the original five previously blocked layouts")

    parent_manifest = _bound_json(contained(parent_study, "evidence-manifest.json"), archived["parent"]["manifest"])
    home = _bound_json(contained(parent_study, "setup-home.json"), parent_manifest["setup-home.json"])
    fallback_manifest = _bound_json(
        contained(fallback_study, "evidence-manifest.json"), archived["fallback"]["evidence-manifest.json"],
    )
    prior_sha = archived["fallback"]["training-prior.json"]
    if prior_sha != fallback_manifest["training-prior.json"]:
        raise ValueError("Archived setup used a different fallback geometry prior")
    prior = _bound_json(contained(fallback_study, "training-prior.json"), prior_sha)

    recreated = verify(SimpleNamespace(output=setup_study), archived, dict(prior=prior, home=home))
    if recreated != receipt or receipt["status"] != "passed":
        raise ValueError("Saved preparation and path proofs disagree with the passed CPU audit")

    cases, proof_manifest = [], {}
    for layout in layouts:
        identity = layout["id"]
        artifact = _bound_json(
            contained(setup_study, f"preparations/{identity}.json"), PREPARATION_SHA256[identity],
        )
        requests = [dict(row, phase="test", role="test", stratum=layout["stratum"])
                    for row in artifact["admitted"]]
        for request in requests:
            for key in ("context_resolution", "prepared_connection"):
                relative = "native-heading/" + request[f"{key}_file"]
                expected = request[f"{key}_sha256"]
                if sha(contained(setup_study, relative)) != expected:
                    raise ValueError(f"Admitted path proof differs from its bound bytes: {relative}")
                if proof_manifest.setdefault(relative, expected) != expected:
                    raise ValueError(f"Shared setup artifact has inconsistent bindings: {relative}")
        cases.append(dict(layout=layout, requests=requests))

    requests = [row for case in cases for row in case["requests"]]
    if len(requests) != REQUESTS or len({row["id"] for row in requests}) != REQUESTS or len(
        {command_key(row) for row in requests}
    ) != REQUESTS:
        raise ValueError("Expected all 43 distinct archived native commands")

    binding = dict(
        plan_sha256=PLAN_SHA256, audit_sha256=AUDIT_SHA256,
        preparation_sha256=dict(PREPARATION_SHA256), proof_manifest=proof_manifest,
        training_prior_sha256=prior_sha, parent=archived["parent"],
    )
    return dict(version=VERSION, cases=cases), binding
