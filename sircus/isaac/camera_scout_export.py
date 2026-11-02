"""Build a local camera scout from native transforms and the original URDF visuals."""

import argparse
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from camera_options import camera_options, preview_origins
from prepare_assets import matrix, vec
from scipy.spatial.transform import Rotation


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pose(position, quaternion):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(
        np.asarray(quaternion)[[1, 2, 3, 0]]
    ).as_matrix()
    result[:3, 3] = position
    return result


def origin(node):
    return (
        np.eye(4)
        if node is None
        else matrix(vec(node.get("xyz", "0 0 0")), vec(node.get("rpy", "0 0 0")))
    )


def visual_transform(link, transforms, root):
    """Use measured native bodies; expand only rigid merged children if necessary."""
    if link in transforms:
        return transforms[link]
    joint = next(
        (j for j in root.findall("joint") if j.find("child").get("link") == link), None
    )
    if joint is None or joint.get("type") != "fixed":
        raise ValueError(f"No measured transform or fixed ancestor for {link}")
    return visual_transform(
        joint.find("parent").get("link"), transforms, root
    ) @ origin(joint.find("origin"))


def export(snapshot, manifest, assets, meshes, output):
    state, capture = json.loads(snapshot.read_text()), json.loads(manifest.read_text())
    if state["schema"] != "sircus-native-camera-scout-state-v1":
        raise ValueError("Unknown snapshot")
    if digest(snapshot) != capture["scout_snapshot"]["sha256"]:
        raise ValueError("Native snapshot digest mismatch")
    if len(state["origins_m"]) != 35:
        raise ValueError("Expected 35 workcells")
    output.mkdir(exist_ok=False)
    (output / "meshes").mkdir()
    cfg, geometries, instances, asset_hashes = state["config"], {}, [], {}
    canonical = preview_origins()
    display = sorted(canonical.tolist(), key=lambda p: (p[1], p[0]))

    def geometry(value):
        key = hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[
            :12
        ]
        geometries[key] = value
        return key

    def add(gid, transform, color, semantic, index):
        slot = next(
            i
            for i, p in enumerate(display)
            if np.allclose(p, state["origins_m"][index], atol=1e-5)
        )
        instances.append(
            dict(
                geometry=gid,
                matrix=transform.T.flatten().tolist(),
                color=color,
                semantic=semantic,
                source_cell=state["source_cells"][index],
                display_slot=f"r{slot // 7 + 1}c{slot % 7 + 1}",
            )
        )

    for role, filename, semantic in (
        ("Arm10", "dropper.urdf", "arm10"),
        ("Arm14", "board_arm.urdf", "arm14"),
    ):
        path = assets / filename
        root = ET.parse(path).getroot()
        asset_hashes[filename] = digest(path)
        body = state["arms"][role]
        for index in range(35):
            transforms = {
                name: pose(p, q)
                for name, p, q in zip(
                    body["body_names"],
                    body["positions_m"][index],
                    body["quaternions_wxyz"][index],
                )
            }
            for link in root.findall("link"):
                for visual in link.findall("visual"):
                    shape = visual.find("geometry")
                    transform = visual_transform(
                        link.get("name"), transforms, root
                    ) @ origin(visual.find("origin"))
                    color_node = visual.find("material/color")
                    color = (
                        vec(color_node.get("rgba"))[:3]
                        if color_node is not None
                        else [1, 1, 1]
                    )
                    if shape.find("mesh") is not None:
                        mesh = shape.find("mesh")
                        name = Path(mesh.get("filename")).name
                        src = meshes / name
                        if not (output / "meshes" / name).exists():
                            shutil.copyfile(src, output / "meshes" / name)
                            asset_hashes[name] = digest(src)
                        transform = transform @ np.diag(
                            [*vec(mesh.get("scale", "1 1 1")), 1]
                        )
                        value = dict(type="stl", url="meshes/" + name)
                    elif shape.find("box") is not None:
                        value = dict(
                            type="box", size_m=vec(shape.find("box").get("size"))
                        )
                    elif shape.find("sphere") is not None:
                        value = dict(
                            type="sphere",
                            radius_m=float(shape.find("sphere").get("radius")),
                        )
                    elif shape.find("cylinder") is not None:
                        c = shape.find("cylinder")
                        value = dict(
                            type="cylinder",
                            radius_m=float(c.get("radius")),
                            length_m=float(c.get("length")),
                        )
                    else:
                        raise ValueError("Unsupported URDF visual")
                    add(
                        geometry(value),
                        transform,
                        color,
                        "board" if link.get("name") == "impact_board" else semantic,
                        index,
                    )
    table = geometry(dict(type="box", size_m=cfg["table"]["size_m"]))
    ball = geometry(dict(type="sphere", radius_m=cfg["ball"]["radius_m"]))
    b = cfg["bucket"]
    bucket = geometry(
        dict(
            type="bucket",
            **{
                k: b[k]
                for k in (
                    "inner_top_diameter_m",
                    "outer_top_diameter_m",
                    "inner_bottom_diameter_m",
                    "outer_bottom_diameter_m",
                )
            },
            height_m=b["assumed_interior_depth_m"] + b["assumed_bottom_thickness_m"],
            bottom_thickness_m=b["assumed_bottom_thickness_m"],
        )
    )
    for index in range(35):
        for name, gid, color, semantic in (
            ("Table", table, [0.4, 0.44, 0.47], "table"),
            ("Ball", ball, [1, 0.04, 0.60], "ball"),
            ("Bucket", bucket, [1, 0.68, 0.02], "bucket"),
        ):
            item = state["objects"][name]
            add(
                gid,
                pose(item["positions_m"][index], item["quaternions_wxyz"][index]),
                color,
                semantic,
                index,
            )
    selected = capture["variants"][0]
    camera = {
        k: capture[k]
        for k in ("focal_length_mm", "horizontal_aperture_mm", "vertical_aperture_mm")
    }
    camera.update(eye_m=selected["eye_m"], target_m=selected["target_m"])
    data = dict(
        schema="sircus-camera-scout-v1",
        units="metres",
        up_axis="Z",
        scope="Camera-only scout using frozen native Isaac transforms and original visual meshes. Browser lighting approximates the final Isaac render.",
        source_fingerprint=capture["source_scene_fingerprint"],
        snapshot_sha256=digest(snapshot),
        default_camera=camera,
        presets=[
            dict(id="selected", name="Selected view 06 · Z 2 m", **camera),
            *camera_options(),
        ],
        geometries=geometries,
        instances=instances,
        floor=dict(z_m=-0.775, color=[0.22, 0.23, 0.24]),
        asset_sha256=asset_hashes,
        workcells=35,
        arms=70,
        bucket_height_status=b["height_interpretation"],
    )
    (output / "scene.json").write_text(json.dumps(data, allow_nan=False))
    return dict(
        workcells=35,
        arms=70,
        visual_instances=len(instances),
        unique_geometries=len(geometries),
        mesh_files=len(list((output / "meshes").glob("*.stl"))),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("snapshot", "manifest", "assets", "meshes", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(export(a.snapshot, a.manifest, a.assets, a.meshes, a.output)))
