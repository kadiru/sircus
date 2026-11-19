"""Independent CPU evidence checks for a geometry-guided continuous PPO pilot.

The historical chain/membership auditor is reused through an explicit backend.
The new decoder is checked separately in physical coordinates, while PPO
likelihoods stay in normalized residual coordinates. No GPU or simulator starts.
"""

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np

import continuous_pilot_audit as chain

SCHEMA = "sircus-feasibility-guided-checkpoint-audit-v1"
STUDY_SCHEMA = "sircus-native-feasibility-guided-pilot-v1"
DECODER = "geometry-conditioned local affine residual-v1"
SCORING_FILES = ("rl_contract.py", "board_route.py", "entry_quality.py",
                 "farm_contract.py", "settling_contract.py")


def validate_support_evidence(support, feasibility, spec):
    """Rebuild anchor ranking, Sobol requests and width selection independently.

    This verifies provenance/arithmetic, not a second simulation or full mesh
    collision certificate. Native release gates are independently audited later.
    """
    from continuous_feasibility import LOW, HIGH, ACTION_NAMES, decode_action
    from scipy.stats import qmc

    chain.signed(support, "Guided support")
    chain.signed(feasibility, "Geometry map")
    chain.require(support.get("feasibility_fingerprint") == feasibility["fingerprint"]
                  and support.get("asset_sha256") == feasibility.get("asset_sha256")
                  and feasibility.get("source_fingerprint") == spec["fingerprint"]
                  and feasibility.get("ik_protocol") == "full_robust_v2",
                  "Guided support/map geometry binding changed")
    half = np.asarray([math.radians(5), .025, .025, .060, math.radians(6)])
    chain.require(support.get("base_half_width") == half.tolist()
                  and support.get("width_scales") == [1., .5, .25, .125]
                  and support.get("probe_count") == 16
                  and support.get("probe_seed") == 20261113
                  and support.get("minimum_probe_acceptance") == .75
                  and support.get("anchor_path_samples") == 16,
                  "Guided geometry calibration protocol changed")
    home = support.get("canonical_home_joint_rad", {})
    chain.require(set(home) == {"Arm10", "Arm14"}, "Missing guided canonical home provenance")
    home = {n: chain.finite(v, (6,), "canonical home") for n, v in home.items()}
    by_cell = {c["id"]: [] for c in spec["cells"]}
    for record in feasibility["records"]:
        chain.require(record["cell_id"] in by_cell, "Unexpected map support cell")
        if record["accepted"]:
            by_cell[record["cell_id"]].append(record)
    probe_requests = probe_accepted = supported = 0
    rows = support["cells"]
    chain.require(len(rows) == len(spec["cells"]) == 35, "Guided support cell count changed")
    for i, (row, cell) in enumerate(zip(rows, spec["cells"])):
        chain.require(row.get("cell_id") == cell["id"] and row.get("cell_index") == i
                      and row.get("bucket_xy_m") == cell["bucket_xy_m"],
                      "Guided support cell identity changed")
        options = by_cell[cell["id"]]
        chain.require(row.get("map_candidates") == len(options)
                      and len(row.get("anchor_checks", [])) == len(options),
                      "Guided anchor candidates omitted")
        distance = np.linalg.norm(np.array([.67, -.10])-cell["bucket_xy_m"])
        prior = np.array([0., np.clip(distance, .29, 1.19), .35, .24, math.radians(29)])
        ranked = []
        for option, check in zip(options, row["anchor_checks"]):
            action = option["action"]
            values = np.asarray([action[k] for k in ACTION_NAMES])
            chain.require(action == decode_action(cell, values, action["id"]),
                          "Map anchor action includes non-geometric fields or inconsistent geometry")
            travel = np.mean([np.square(np.asarray(option["ik"][n]["q"])-home[n]).sum()
                              for n in ("Arm10", "Arm14")])
            score = float(np.square((values-prior)/(HIGH-LOW)).sum() + .01*travel)
            chain.require(check.get("proposal") == option["proposal"]
                          and check.get("action_id") == action["id"]
                          and abs(check.get("score", math.inf)-score) <= 2e-7
                          and type(check.get("path", {}).get("accepted")) is bool,
                          "Geometry-only anchor rank changed")
            if check["path"]["accepted"]:
                ranked.append((score, int(option["proposal"]), option))
        if not ranked:
            chain.require(row.get("status") == "unresolved" and not row.get("calibration")
                          and row.get("anchor_action") is None and row.get("anchor_ik") is None
                          and row.get("low") == LOW.tolist() and row.get("high") == HIGH.tolist(),
                          "Empty geometry support was relabeled or assigned an anchor")
            continue
        supported += 1
        score, proposal, chosen = min(ranked, key=lambda x: x[:2])
        chain.require(row.get("status") == "supported" and row.get("anchor_proposal") == proposal
                      and row.get("anchor_action") == chosen["action"]
                      and row.get("anchor_ik") == chosen["ik"]
                      and abs(row.get("anchor_score", math.inf)-score) <= 2e-7,
                      "Chosen anchor differs from geometry-only ranking")
        center = np.asarray([chosen["action"][k] for k in ACTION_NAMES])
        unit = qmc.Sobol(d=5, scramble=True, seed=20261113+i).random_base2(4)
        calibration = row["calibration"]
        chain.require(1 <= len(calibration) <= 4, "Missing or excessive geometry probes")
        for level, (probe_set, scale) in enumerate(zip(calibration, (1., .5, .25, .125))):
            lo, hi = np.maximum(LOW, center-half*scale), np.minimum(HIGH, center+half*scale)
            chain.require(probe_set.get("scale") == scale and probe_set.get("low") == lo.tolist()
                          and probe_set.get("high") == hi.tolist()
                          and probe_set.get("requests") == len(probe_set.get("records", [])) == 16,
                          "Guided local bounds or probe budget changed")
            accepted_count = 0
            for j, (record, values) in enumerate(zip(probe_set["records"], lo+unit*(hi-lo))):
                expected = decode_action(cell, values, f"support-{i:02d}-{scale}-{j:02d}")
                qualification, path = record["qualification"], record["path"]
                accepted = bool(qualification["accepted"] and path and path["accepted"])
                chain.require(record.get("probe") == j and record.get("action") == expected
                              and qualification.get("action") == expected
                              and record.get("accepted") is accepted
                              and (qualification["accepted"] or path is None),
                              "Geometry probe request/qualification association changed")
                accepted_count += accepted
            rate = accepted_count/16
            chain.require(probe_set.get("accepted") == accepted_count
                          and probe_set.get("acceptance") == rate,
                          "Geometry probe denominator changed")
            chain.require(level == len(calibration)-1 or rate < .75,
                          "Support shrank after already meeting declared geometry target")
            probe_requests += 16
            probe_accepted += accepted_count
        selected = calibration[-1]
        chain.require(selected["acceptance"] >= .75 or len(calibration) == 4,
                      "Support calibration stopped before target/minimum width")
        chain.require(row.get("low") == selected["low"] and row.get("high") == selected["high"]
                      and row.get("selected_width_scale") == selected["scale"]
                      and row.get("probe_acceptance") == selected["acceptance"]
                      and row.get("qualification_target_met") is (selected["acceptance"] >= .75),
                      "Guided selected local support or qualification status changed")
    chain.require(support.get("supported_cells") == supported
                  and support.get("probe_requests") == probe_requests
                  and support.get("probe_accepted") == probe_accepted,
                  "Geometry support aggregate accounting changed")
    return dict(supported_cells=supported, unresolved_cells=35-supported,
                probe_requests=probe_requests, probe_accepted=probe_accepted,
                geometry_only_anchor_selection_verified=True,
                scope="Independent anchor ranking/probe arithmetic/provenance checks; "
                      "no replay of IK or capsule checks and no complete box feasibility certificate")


def independent_physical_actions(normalized, support, cells):
    """Reconstruct the per-context affine map independently of producer code."""
    if hasattr(normalized, "detach"):
        normalized = normalized.detach().cpu().numpy()
    values = np.asarray(normalized, dtype=float)
    chain.require(values.shape == (len(cells), 5) and np.isfinite(values).all()
                  and (np.abs(values) <= 1).all(), "Invalid guided normalized action")
    by_id = {row["cell_id"]: row for row in support["cells"]}
    chain.require(len(by_id) == len(support["cells"]), "Duplicate guided support cells")
    lo = np.asarray([by_id[c["id"]]["low"] for c in cells], dtype=float)
    hi = np.asarray([by_id[c["id"]]["high"] for c in cells], dtype=float)
    chain.require(lo.shape == hi.shape == values.shape and np.isfinite(lo).all()
                  and np.isfinite(hi).all() and (hi > lo).all(), "Invalid guided support widths")
    return lo + (values + 1.) * (hi - lo) / 2.


def physical_log_density_offset(support, cells):
    """Fixed-context Jacobian term; it cancels in old/new PPO likelihood ratios."""
    by_id = {row["cell_id"]: row for row in support["cells"]}
    widths = np.asarray([np.asarray(by_id[c["id"]]["high"]) -
                         np.asarray(by_id[c["id"]]["low"]) for c in cells])
    chain.require(widths.shape == (len(cells), 5) and np.isfinite(widths).all()
                  and (widths > 0).all(), "Invalid guided physical density Jacobian")
    return -np.log(widths / 2.).sum(axis=1)


def validate_guided_study(study, spec, support, support_sha256, feasibility_sha256):
    """Preserve every original budget/split/gate guard with new typed identity."""
    from continuous_policy import ARCHITECTURE as OLD_ARCH, POLICY_VERSION as OLD_VERSION
    from feasibility_guided_policy import ARCHITECTURE, POLICY_VERSION
    from feasibility_guided_support import validate_support

    study_hash = chain.signed(study, "Guided study")
    geometry_hash = chain.signed(spec, "Original geometry")
    domain_hash = chain.signed(study["domain"], "Guided domain")
    support_hash = chain.signed(support, "Guided support")
    validate_support(support, spec["cells"])
    chain.require(study.get("schema") == STUDY_SCHEMA
                  and study.get("architecture") == ARCHITECTURE
                  and study.get("policy_version") == POLICY_VERSION,
                  "Guided study policy identity changed")
    chain.require(study.get("support_fingerprint") == support_hash
                  and study.get("support_sha256") == support_sha256
                  and study.get("feasibility_sha256") == feasibility_sha256
                  and study["domain"].get("decoder") == DECODER
                  and study["domain"].get("support_fingerprint") == support_hash,
                  "Guided support/domain binding changed")
    chain.require(support.get("source_fingerprint") == geometry_hash
                  and support.get("policy_geometry_only") is True,
                  "Guided support is not geometry-only or source changed")
    # Deliberately translate only the three protocol identity fields. The old
    # validator still checks every shared split, seed, budget, gate, timing,
    # global bound, filming schedule and PPO hyperparameter unchanged.
    shared = copy.deepcopy(study)
    shared.update(schema="sircus-native-continuous-pilot-v1",
                  architecture=OLD_ARCH, policy_version=OLD_VERSION)
    shared.pop("fingerprint")
    shared["fingerprint"] = chain.canonical_hash(shared)
    chain.validate_study(shared, spec)
    return study_hash, geometry_hash, domain_hash


class _Backend:
    def __init__(self, run):
        from feasibility_guided_policy import GuidedPolicy, load_checkpoint
        from feasibility_guided_support import physical_actions

        self.run = Path(run)
        self.support_path = chain.evidence_file(run, "support.json")
        self.feasibility_path = chain.evidence_file(run, "feasibility.json")
        self.support = chain.read(self.support_path)
        self.feasibility = chain.read(self.feasibility_path)
        self.spec = chain.read(chain.evidence_file(run, "contract.json"))
        chain.signed(self.feasibility, "Geometry feasibility map")
        chain.require(self.support.get("feasibility_fingerprint") == self.feasibility["fingerprint"],
                      "Guided support came from a different feasibility map")
        self.support_evidence = validate_support_evidence(self.support, self.feasibility, self.spec)
        self.policy_factory = GuidedPolicy
        self._load_checkpoint = load_checkpoint
        self._physical_actions = physical_actions

    def validate_study(self, study, spec):
        chain.require(spec == self.spec, "Guided audit source changed")
        return validate_guided_study(study, spec, self.support,
                                    chain.digest(self.support_path), chain.digest(self.feasibility_path))

    def load_checkpoint(self, path, **kwargs):
        return self._load_checkpoint(path, support_fingerprint=self.support["fingerprint"], **kwargs)

    def physical_actions(self, normalized, _low, _high):
        expected = independent_physical_actions(normalized, self.support, self.spec["cells"])
        actual = self._physical_actions(normalized, self.support, self.spec["cells"])
        chain.require(np.array_equal(expected, actual), "Guided producer affine decode changed")
        return expected


def backend_for_run(run):
    """Explicit adapter shared by policy audit and native outcome audit."""
    return _Backend(run)


def audit(run, executed_source=None, *, first_update_only=False):
    backend = backend_for_run(run)
    fn = chain.verify_first_update if first_update_only else chain.verify_checkpoint_chain
    result = fn(run, executed_source, policy_backend=backend)
    result["schema"] = SCHEMA if not first_update_only else SCHEMA.replace("checkpoint", "first-update")
    result["support_fingerprint"] = backend.support["fingerprint"]
    result["support_sha256"] = chain.digest(backend.support_path)
    result["feasibility_sha256"] = chain.digest(backend.feasibility_path)
    result["support_evidence"] = backend.support_evidence
    for name in result["trials"]:
        trial = chain.read(chain.evidence_file(run, name, trial=True))
        chain.require(trial.get("support_fingerprint") == backend.support["fingerprint"],
                      "Trial guided support metadata changed")
        for arm, home in backend.support["canonical_home_joint_rad"].items():
            target = chain.finite(trial.get("home", {}).get("target_joint_rad", {}).get(arm),
                                  (35, 6), "guided canonical target")
            chain.require(np.allclose(target, np.asarray(home)[None], atol=1e-7, rtol=0),
                          "Guided support home differs from commanded episode home")
    result["physical_log_density_offsets"] = physical_log_density_offset(
        backend.support, backend.spec["cells"]).tolist()
    result["coordinate_scope"] = (
        "Likelihoods/entropy use normalized residual actions. The fixed per-context "
        "affine physical-density Jacobian cancels in PPO ratios. Independent affine "
        "decoding verifies every requested action; no retries/projection remove failures.")
    result["held_out_scope"] = (
        "Held out from reward optimization only. Geometry-only support uses each "
        "known bucket coordinate, including validation/test; not globally unseen contexts.")
    return result


def compare_runs(old_run, guided_run):
    """Compare frozen per-split outcomes without changing either raw result."""
    old_run, guided_run = Path(old_run), Path(guided_run)
    old_spec, new_spec = (chain.read(p / "contract.json") for p in (old_run, guided_run))
    old_study, new_study = (chain.read(p / "study.json") for p in (old_run, guided_run))
    old_result, new_result = (chain.read(p / "result.json") for p in (old_run, guided_run))
    chain.validate_study(old_study, old_spec)
    backend_for_run(guided_run).validate_study(new_study, new_spec)
    for result, study in ((old_result, old_study), (new_result, new_study)):
        chain.require(result.get("status") == "completed" and result.get("schema") == study["schema"]
                      and result.get("study_fingerprint") == study["fingerprint"]
                      and result.get("geometry_fingerprint") == study["geometry_fingerprint"],
                      "Comparison requires completed bound results")
    chain.require(old_spec == new_spec and old_study["cells"] == new_study["cells"],
                  "Comparison geometry or held-out membership changed")
    for key in ("training_batches", "batches_per_update", "assessment_batches", "seed",
                "ppo", "measurement_hold_s", "measurement_window_s", "home_hold_s",
                "episode_start", "ik_protocol", "filmed_training_batches"):
        chain.require(old_study.get(key) == new_study.get(key), "Comparison protocol changed: " + key)
    chain.require(old_study["source"]["camera_sha256"] == new_study["source"]["camera_sha256"],
                  "Comparison camera changed")
    chain.require(all(old_result["source_sha256"].get(k) == new_result["source_sha256"].get(k)
                      == chain.digest(Path(__file__).with_name(k))
                      for k in SCORING_FILES), "Comparison scoring/release definitions changed")
    from farm_contract import statistics

    def stats(run, result):
        batches = result["batches"]
        def rows_for(summary, mode):
            path = chain.evidence_file(run, summary["file"], trial=True)
            chain.require(chain.digest(path) == summary["sha256"], "Comparison trial checksum changed")
            trial = chain.read(path)
            chain.require(trial.get("mode") == summary.get("mode") == mode
                          and len(trial.get("rows", [])) == 35
                          and [(r["cell_id"], r["split"]) for r in trial["rows"]]
                          == [(c["id"], c["split"]) for c in old_study["cells"]],
                          "Comparison trial membership changed")
            return trial["rows"]
        train = [row for summary in batches for row in rows_for(summary, "train")
                 if row["split"] == "train"]
        chain.require(len(train) == 25*old_study["training_batches"]
                      and len(result["assessments"]) == 2*old_study["assessment_batches"],
                      "Comparison sample budget changed")
        values = dict(training=statistics(train), frozen={})
        repeats = len(result["assessments"]) // 2
        for stage, label in enumerate(("initial", "final")):
            rows = [row for summary in result["assessments"][stage*repeats:(stage+1)*repeats]
                    for row in rows_for(summary, "eval")]
            values["frozen"][label] = {split: statistics([r for r in rows if r["split"] == split])
                                       for split in ("train", "validation", "test")}
        return values

    return dict(schema="sircus-feasibility-guided-comparison-v1", status="completed",
                original_result_sha256=chain.digest(old_run/"result.json"),
                guided_result_sha256=chain.digest(guided_run/"result.json"),
                original=stats(old_run, old_result), guided=stats(guided_run, new_result),
                scope="Matched one-seed nominal-layout comparison; geometry-informed action support "
                      "changes the proposal distribution. Include geometry precompute cost. "
                      "No new-layout/noise robustness or physical validation claim.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--executed-source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--first-update-only", action="store_true")
    parser.add_argument("--compare-original", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output; previous evidence is preserved")
    value = audit(args.run, args.executed_source, first_update_only=args.first_update_only)
    if args.compare_original:
        chain.require(not args.first_update_only, "Partial pilot cannot produce final comparison")
        value["comparison"] = compare_runs(args.compare_original, args.run)
    with args.output.open("x") as f:
        json.dump(value, f, allow_nan=False)
    print(json.dumps(value), flush=True)


if __name__ == "__main__":
    main()
