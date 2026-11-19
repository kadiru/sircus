"""Train-derived relative proposals for EXPLICIT native search, never inference.

A translated/rotated source capture is only a proposal. Changed robot poses and
finite table boundaries can introduce contacts, so no success label is copied.
All proposals still require endpoint IK, collision gates and a fresh trajectory.
"""

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .learning_contract import HIGH, LOW, digest
from .learning_reach import feasible_mask

VERSION = "sircus-relative-search-bank-v1"
ROTATION_STEP_DEG = 22.5


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_bank(catalogue, train_layouts, catalogue_sha256, contract_fingerprint):
    """Freeze a finite proposal bank using TRAIN successes and confirmations."""
    if not isinstance(catalogue, list):
        raise RuntimeError("Training catalogue must contain a list of cells")
    train = np.asarray(train_layouts, dtype=float)
    if train.ndim != 2 or train.shape[1] != 2 or not np.isfinite(train).all():
        raise RuntimeError("Malformed train-layout provenance")
    templates = []
    for cell in catalogue:
        if cell.get("split") != "train":
            raise RuntimeError(
                "Explicit search bank must contain only TRAIN catalogue cells"
            )
        xy = np.asarray(cell.get("bucket_xy_m"), dtype=float)
        if (
            xy.shape != (2,)
            or not np.isfinite(xy).all()
            or not np.any(np.all(np.abs(train - xy) <= 1e-12, axis=1))
        ):
            raise RuntimeError(
                "Catalogue cell does not belong to the frozen training split"
            )
        source = cell.get("best")
        if source is None:
            continue
        if source.get("phase") != "train-exploration":
            raise RuntimeError(
                "Search teachers must come from TRAIN exploration, never evaluation"
            )
        if source.get("invalid") or not source.get("route_success"):
            continue
        values = np.asarray(source.get("values"), dtype=float)
        if (
            values.shape != (5,)
            or not np.isfinite(values).all()
            or not ((values >= LOW) & (values <= HIGH)).all()
        ):
            raise RuntimeError("Malformed training teacher action")
        if not np.allclose(source.get("bucket_xy_m"), xy, atol=1e-12, rtol=0):
            raise RuntimeError("Teacher bucket differs from its training cell")
        source_id = source.get("id")
        if not isinstance(source_id, str) or not source_id:
            raise RuntimeError("Teacher is missing its original trial ID")
        confirmation = cell.get("confirmation", {})
        attempts = int(confirmation.get("attempts", 0))
        captures = int(confirmation.get("captures", 0))
        invalid = int(confirmation.get("invalid", 0))
        if (
            not 0 <= captures <= attempts
            or not 0 <= invalid <= attempts
            or captures + invalid > attempts
        ):
            raise RuntimeError("Malformed teacher confirmation counts")
        templates.append(
            dict(
                source_trial_id=source_id,
                source_cell_id=cell.get("id"),
                relative_board_xy_m=(values[:2] - xy).tolist(),
                height_m=float(values[2]),
                tilt_deg=float(values[3]),
                gap_m=float(values[4]),
                source_route_quality=int(source.get("route_quality", 0)),
                confirmation=dict(
                    attempts=attempts, captures=captures, invalid=invalid
                ),
                qualification=(
                    "train_confirmation_3_of_3"
                    if attempts == 3 and captures == 3 and invalid == 0
                    else "train_capture_lower_confirmation"
                ),
            )
        )
    strong = [
        row for row in templates if row["qualification"] == "train_confirmation_3_of_3"
    ]
    # Do not silently call weak teachers reliable. An empty bank simply leaves
    # the explicit search's continuous proposal family available.
    selected = strong if strong else templates
    selected.sort(
        key=lambda row: (
            -row["confirmation"]["captures"] / max(1, row["confirmation"]["attempts"]),
            -row["source_route_quality"],
            row["source_trial_id"],
        )
    )
    bank = dict(
        version=VERSION,
        contract_fingerprint=contract_fingerprint,
        training_catalogue_sha256=catalogue_sha256,
        rotation_step_deg=ROTATION_STEP_DEG,
        source_scope="Frozen TRAIN exploration teachers; never a policy or query success label",
        qualification=(
            "3/3 confirmed teachers only"
            if strong
            else "TRAIN captures; lower/unavailable confirmation explicitly retained"
        ),
        templates=selected,
    )
    bank["sha256"] = digest(bank)
    return bank


def load_training_bank(study, contract_fingerprint):
    """Read-only, completed-study provenance check before making proposals."""
    study = Path(study)
    report = json.loads((study / "report.json").read_text())
    if report.get("status") != "completed" or not report.get("frozen_evaluation"):
        raise RuntimeError("Explicit-search bank requires the completed frozen study")
    contract = json.loads((study / "contract.json").read_text())
    payload = dict(contract)
    recorded_fingerprint = payload.pop("fingerprint")
    if (
        digest(payload) != recorded_fingerprint
        or report.get("contract_fingerprint") != contract_fingerprint
        or recorded_fingerprint != contract_fingerprint
    ):
        raise RuntimeError("Search catalogue belongs to another scene contract")
    provenance = json.loads((study / "artifact_provenance.json").read_text())
    if provenance.get("contract_fingerprint") != contract_fingerprint:
        raise RuntimeError("Search catalogue provenance belongs to another study")
    catalogue = study / "catalogue.json"
    actual = _sha256(catalogue)
    if provenance.get("training_catalogue_sha256") != actual:
        raise RuntimeError("Training catalogue checksum is missing or changed")
    return build_bank(
        json.loads(catalogue.read_text()),
        contract["train_layouts"],
        actual,
        contract_fingerprint,
    )


def bank_candidates(bank, bucket_xy, scene):
    """Translate/rotate proposals, filtering bounds and conservative wrist reach.

    Rotation is an exact multiple of the frozen 64-wall bucket's rotational
    period. Full native contact/IK checks remain authoritative. The ordering
    interleaves teachers so one template does not consume the whole first batch.
    """
    payload = dict(bank)
    expected = payload.pop("sha256")
    if bank.get("version") != VERSION or digest(payload) != expected:
        raise RuntimeError("Relative search bank fingerprint changed")
    xy = np.asarray(bucket_xy, dtype=float)
    if xy.shape != (2,) or not np.isfinite(xy).all():
        raise ValueError("Expected one finite bucket XY")
    if scene["bucket"]["segments"] != 64:
        raise RuntimeError(
            "22.5-degree relative bank rotations require the frozen 64-wall bucket"
        )
    groups = []
    for template in bank["templates"]:
        candidates, metadata = [], []
        relative = np.asarray(template["relative_board_xy_m"])
        for turn in range(16):
            angle = math.radians(turn * ROTATION_STEP_DEG)
            rotation = np.array(
                [
                    [math.cos(angle), -math.sin(angle)],
                    [math.sin(angle), math.cos(angle)],
                ]
            )
            point = xy + rotation @ relative
            value = [
                *point,
                template["height_m"],
                template["tilt_deg"],
                template["gap_m"],
            ]
            candidates.append(value)
            metadata.append(
                dict(
                    source_trial_id=template["source_trial_id"],
                    rotation_deg=turn * ROTATION_STEP_DEG,
                    qualification=template["qualification"],
                )
            )
        values = np.asarray(candidates)
        within = ((values >= LOW - 1e-9) & (values <= HIGH + 1e-9)).all(axis=1)
        prior = feasible_mask(
            values[:, :3],
            values[:, 3],
            np.repeat(xy[None], 16, axis=0),
            values[:, 4],
            scene,
        )
        groups.append(
            [
                (value, meta)
                for value, meta, valid in zip(values, metadata, within & prior)
                if valid
            ]
        )
    values, metadata = [], []
    for offset in range(max((len(group) for group in groups), default=0)):
        for group in groups:
            if offset < len(group):
                value, meta = group[offset]
                values.append(value)
                metadata.append(meta)
    return np.asarray(values, dtype=float).reshape(-1, 5), metadata
