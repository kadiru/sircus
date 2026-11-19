"""Export measured frozen simulator transforms for a camera-only browser scout."""

import hashlib
from pathlib import Path

from parallel_runtime import atomic


def write_snapshot(env):
    data = dict(
        schema="sircus-native-camera-scout-state-v1",
        scope="Frozen native Isaac transforms; static camera scouting, no rollout or control API.",
        physics_step=env.steps,
        config=env.cfg,
        origins_m=env.origins.tolist(),
        source_cells=[c["id"] for c in env.spec["cells"]],
        arms={
            name: dict(
                body_names=arm.body_names,
                positions_m=arm.data.body_pos_w.cpu().numpy().tolist(),
                quaternions_wxyz=arm.data.body_quat_w.cpu().numpy().tolist(),
            )
            for name, arm in env.arms.items()
        },
        objects={
            name: dict(
                positions_m=env.scene[name].data.root_pos_w.cpu().numpy().tolist(),
                quaternions_wxyz=env.scene[name]
                .data.root_quat_w.cpu()
                .numpy()
                .tolist(),
            )
            for name in ("Table", "Bucket", "Ball")
        },
        exporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    path = env.dest / "scout-state.json"
    atomic(path, data)
    return dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
