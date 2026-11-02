"""CPU normal-impact calibration independent of bucket outcomes.

Keep the board's actual 5 mm thickness while checking both 240 and 960 Hz.
Report velocity restitution, apex restitution and maximum geometric penetration.
"""

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

import mujoco
import numpy as np

TRAIN_GAPS = (0.1, 0.3, 0.6)
TEST_GAPS = (0.2, 0.4)
TIMECONSTS = (0.008, 0.01, 0.015, 0.02, 0.03)
DAMPING = (0.1, 0.136, 0.18, 0.2, 0.25)
TARGET = 0.65
RADIUS = 0.0375
THICKNESS = 0.005


def probe(timeconst, damping_ratio, gap, dt):
    xml = f"""<mujoco><option timestep="{dt}" integrator="implicitfast"
      solver="Newton" iterations="50" cone="elliptic" gravity="0 0 -9.81"/>
      <worldbody><geom name="board" type="box" size="1 1 {THICKNESS/2}"
      pos="0 0 {-THICKNESS/2}" condim="3" margin="0" gap="0"/>
      <body pos="0 0 {gap+RADIUS}"><freejoint/><geom name="ball" type="sphere"
      size="{RADIUS}" mass=".2" priority="1" friction=".35 .005 .0001"
      condim="3" margin="0" gap="0" solref="{timeconst} {damping_ratio}"
      solimp=".95 .99 .001 .5 2"/></body></worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    incoming = outgoing = None
    penetration = apex = 0.0
    contact_steps = 0
    peak_force = 0.0
    escaped = False
    for _ in range(math.ceil(2.0 / dt)):
        before = float(data.qvel[2])
        mujoco.mj_step(model, data)
        z = float(data.qpos[2])
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise RuntimeError("Non-finite flat-probe state")
        penetration = max(penetration, RADIUS - z)
        if data.ncon:
            if incoming is None:
                incoming = abs(before)
            contact_steps += 1
            for c in range(data.ncon):
                force = np.zeros(6)
                mujoco.mj_contactForce(model, data, c, force)
                peak_force = max(peak_force, abs(float(force[0])))
        if z < -RADIUS - THICKNESS:
            escaped = True
            break
        if incoming is not None and not data.ncon and data.qvel[2] > 0:
            if outgoing is None:
                outgoing = float(data.qvel[2])
            apex = max(apex, z - RADIUS)
        if outgoing is not None:
            apex = max(apex, z - RADIUS)
            if data.qvel[2] <= 0:
                break
    measured = outgoing / incoming if outgoing is not None and incoming else 0.0
    return dict(
        gap_m=gap,
        restitution_velocity=measured,
        restitution_height=math.sqrt(max(0, apex) / gap),
        rebound_height_m=apex,
        incoming_m_s=incoming,
        outgoing_m_s=outgoing,
        peak_penetration_mm=1000 * max(0, penetration),
        contact_steps=contact_steps,
        peak_contact_force_N=peak_force,
        escaped_through_board=escaped,
    )


def study(timeconsts=TIMECONSTS, damping=DAMPING, timesteps=(1 / 240, 1 / 960)):
    candidates = []
    for dt in timesteps:
        for timeconst, zeta in itertools.product(timeconsts, damping):
            train = [probe(timeconst, zeta, gap, dt) for gap in TRAIN_GAPS]
            test = [probe(timeconst, zeta, gap, dt) for gap in TEST_GAPS]
            candidates.append(
                dict(
                    physics_dt_s=dt,
                    timeconst_s=timeconst,
                    damping_ratio=zeta,
                    train=train,
                    heldout=test,
                    train_mse=float(
                        np.mean(
                            [(p["restitution_velocity"] - TARGET) ** 2 for p in train]
                        )
                    ),
                    max_train_penetration_mm=max(
                        p["peak_penetration_mm"] for p in train
                    ),
                    any_train_escape=any(p["escaped_through_board"] for p in train),
                )
            )
    safe = [
        c
        for c in candidates
        if not c["any_train_escape"] and c["max_train_penetration_mm"] <= 2
    ]
    selected = min(safe, key=lambda x: x["train_mse"]) if safe else None
    best_fit = min(candidates, key=lambda x: x["train_mse"])
    return dict(
        version="sircus-mujoco-normal-substep-calibration-v1",
        scope="CPU normal drops on a 5 mm flat board; selected using train heights only, no bucket results",
        target_restitution=TARGET,
        plate_thickness_m=THICKNESS,
        ball_radius_m=RADIUS,
        contact_friction=0.35,
        solimp=[0.95, 0.99, 0.001, 0.5, 2],
        train_gaps_m=TRAIN_GAPS,
        heldout_gaps_m=TEST_GAPS,
        penetration_limit_mm=2,
        selection="Lowest train velocity restitution MSE subject to <=2mm train penetration and no escape",
        mujoco_version=mujoco.__version__,
        candidates=candidates,
        selected=selected,
        best_fit_unconstrained=best_fit,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--fine", action="store_true", help="Stiffer contact scan at 960/1920 Hz"
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = (
        study(
            (0.0021, 0.003, 0.004, 0.005, 0.006, 0.008),
            (0.1, 0.12, 0.136, 0.15, 0.16, 0.18, 0.2),
            (1 / 960, 1 / 1920),
        )
        if args.fine
        else study()
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "candidates"}))
