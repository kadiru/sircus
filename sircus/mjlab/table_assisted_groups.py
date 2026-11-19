"""Conservative cohort labels for saved-shot rescoring, never outcome inference.

Prefer recorded fields. Fallbacks match literal ID grammars used by the pinned
study producers; an arbitrary occurrence of 'ppo' or 'test' is not a label.
Unknown phases/substeps stay unknown, including rejected rows with no protocol.
"""
from __future__ import annotations

from pathlib import Path
import re

VERSION = "sircus-saved-shot-cohort-labels-v1"


def grouping(file, row, source):
    ident = row.get("id") or ""
    inferred_method, inferred_phase = None, None
    # calibrated_benchmark_run.py, literal seed/method/request grammar.
    benchmark = (re.fullmatch(r"(ppo|sweep)-\d+-(train)-\d+", ident)
                 or re.fullmatch(r"(ppo|sweep)-\d+-(validation|test)-\d+-calibrated-\2-\d+-r\d+", ident))
    if benchmark:
        inferred_method, inferred_phase = benchmark.group(1, 2)
    # policy64_study.py: behavioral cloning is stored as 'bc', not PPO.
    policy = re.fullmatch(r"(validation|test)-(prior|nearest|bc|ppo)-\d+-policy64-(validation|test)-\d+", ident)
    if policy and policy.group(1) == policy.group(3):
        inferred_phase, inferred_method = policy.group(1, 2)
    if re.fullmatch(r"policy64-train-\d+", ident):
        inferred_method, inferred_phase = "ppo", "train"
    # Later comparison producers repeat their own namespace in context IDs.
    for family, methods in (
        ("physics-ensemble", "bc|ppo|ensemble"),
        ("consistent-flight", "bc|ppo|ensemble|consistent"),
        ("gap-coverage", "consistent|gap_coverage"),
        ("arm-clearance", "consistent|arm_clearance"),
    ):
        if family == "physics-ensemble":
            pattern = rf"{family}-(validation|test)-({methods})-{family}-\1-\d+"
            found = re.fullmatch(pattern, ident)
            if found:
                inferred_phase, inferred_method = found.group(1, 2)
        else:
            found = re.fullmatch(rf"{family}-({methods})-{family}-(test)-\d+", ident)
            if found:
                inferred_method, inferred_phase = found.group(1, 2)
    control = re.fullmatch(r"(?:(?:physics-ensemble|consistent-flight|gap-coverage|arm-clearance)-)?(controls_pre|controls_post)-(positive_control|negative_control)-r\d+", ident)
    if control:
        inferred_method, inferred_phase = "control", control.group(1)

    method = row.get("method") or inferred_method
    if not method:
        parts = Path(file).parts
        matches = [re.fullmatch(r"(ppo|sweep)-seed\d+", part) for part in parts]
        named = {m.group(1) for m in matches if m}
        if len(named) == 1:
            method = named.pop()
    phase = row.get("phase") or row.get("role") or inferred_phase or "unspecified"
    substeps = row.get("substeps") or (row.get("contact_protocol") or {}).get("substeps")
    return dict(source_study=source, method=method or "unspecified", phase=phase,
                substeps=substeps if type(substeps) is int and substeps > 0 else "unknown")
