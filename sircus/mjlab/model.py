"""Build the Sircus two-arm MuJoCo scene from the shared geometry and UF850 URDFs.

MuJoCo solref/solimp are recorded engine settings, not a claim that its contact
response equals PhysX restitution. The bucket remains a set of separate convex
wall wedges so that its mouth and interior are empty collision space.
"""

import itertools
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from sircus.isaac.geometry import bucket_wedge

BALL, BOARD, BUCKET, TABLE, ROBOT, FLOOR = 1, 2, 4, 8, 16, 32
ALL = BALL | BOARD | BUCKET | TABLE | ROBOT | FLOOR


def numbers(values):
    return " ".join(format(float(value), ".17g") for value in values)


def vector(text, default=(0, 0, 0)):
    return list(default) if text is None else [float(value) for value in text.split()]


def quaternion(rpy):
    roll, pitch, yaw = (x / 2 for x in rpy)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def transform(element):
    origin = element.find("origin")
    if origin is None:
        return {}
    return dict(
        pos=numbers(vector(origin.get("xyz"))),
        quat=numbers(quaternion(vector(origin.get("rpy")))),
    )


def _mesh_path(filename, assets):
    path = Path(filename)
    if filename.startswith("/workspace/"):
        path = assets.parent / filename.removeprefix("/workspace/")
    elif not path.is_absolute():
        path = assets / path
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.resolve()


def _shape(geo, asset, name, assets):
    shape = next(iter(geo))
    if shape.tag == "box":
        return dict(type="box", size=numbers(x / 2 for x in vector(shape.get("size"))))
    if shape.tag == "sphere":
        return dict(type="sphere", size=shape.get("radius"))
    if shape.tag == "cylinder":
        return dict(
            type="cylinder",
            size=numbers([float(shape.get("radius")), float(shape.get("length")) / 2]),
        )
    if shape.tag == "mesh":
        values = dict(name=name, file=str(_mesh_path(shape.get("filename"), assets)))
        if shape.get("scale"):
            values["scale"] = shape.get("scale")
        ET.SubElement(asset, "mesh", **values)
        return dict(type="mesh", mesh=name)
    raise ValueError(f"Unsupported URDF collision geometry: {shape.tag}")


def _arm(world, asset, contact, actuator, role, name, assets, pos, quat):
    urdf = ET.parse(assets / f"{role}.urdf").getroot()
    links = {link.get("name"): link for link in urdf.findall("link")}
    children = {}
    child_links = set()
    for joint in urdf.findall("joint"):
        parent, child = joint.find("parent").get("link"), joint.find("child").get(
            "link"
        )
        children.setdefault(parent, []).append(joint)
        child_links.add(child)
    roots = set(links) - child_links
    if len(roots) != 1:
        raise ValueError("Expected one fixed UF850 URDF root")
    body_names = []

    def add(parent, link_name, placement, joint=None):
        link = links[link_name]
        body_name = f"{name}/{link_name}"
        body_names.append(body_name)
        body = ET.SubElement(parent, "body", name=body_name, gravcomp="1", **placement)
        inertial = link.find("inertial")
        if inertial is not None:
            inertia = inertial.find("inertia")
            values = [
                float(inertia.get(k))
                for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")
            ]
            xx, yy, zz, xy, xz, yz = values
            tensor = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
            origin = inertial.find("origin")
            rpy = vector(origin.get("rpy")) if origin is not None else [0, 0, 0]
            rot = Rotation.from_euler("xyz", rpy).as_matrix()
            tensor = rot @ tensor @ rot.T
            ET.SubElement(
                body,
                "inertial",
                mass=inertial.find("mass").get("value"),
                pos=numbers(
                    vector(origin.get("xyz")) if origin is not None else [0, 0, 0]
                ),
                fullinertia=numbers(
                    [
                        tensor[0, 0],
                        tensor[1, 1],
                        tensor[2, 2],
                        tensor[0, 1],
                        tensor[0, 2],
                        tensor[1, 2],
                    ]
                ),
            )
        if joint is not None and joint.get("type") != "fixed":
            kind = joint.get("type")
            if kind not in ("revolute", "continuous", "prismatic"):
                raise ValueError(f"Unsupported joint {kind}")
            attrs = dict(
                name=f"{name}/{joint.get('name')}",
                type="slide" if kind == "prismatic" else "hinge",
                axis=joint.find("axis").get("xyz"),
                damping="0",
                frictionloss="0",
                armature="0",
            )
            limit = joint.find("limit")
            if kind != "continuous":
                attrs.update(
                    limited="true",
                    range=numbers(
                        [float(limit.get("lower")), float(limit.get("upper"))]
                    ),
                )
            ET.SubElement(body, "joint", **attrs)
            is_finger = kind == "prismatic"
            force = 30 if is_finger else 100
            ET.SubElement(
                actuator,
                "position",
                name=f"{name}/{joint.get('name')}/servo",
                joint=attrs["name"],
                kp="1000" if is_finger else "1500",
                kv="50" if is_finger else "100",
                forcelimited="true",
                forcerange=numbers([-force, force]),
            )
        for i, collision in enumerate(link.findall("collision")):
            label = f"{body_name}/collision/{i}"
            category = BOARD if link_name == "impact_board" else ROBOT
            color = (
                (0.16, 0.55, 0.55, 1) if category == BOARD else (0.83, 0.84, 0.86, 1)
            )
            ET.SubElement(
                body,
                "geom",
                name=label,
                contype=str(category),
                conaffinity=str(ALL),
                rgba=numbers(color),
                **transform(collision),
                **_shape(collision.find("geometry"), asset, label + "/mesh", assets),
            )
        for next_joint in children.get(link_name, []):
            add(
                body,
                next_joint.find("child").get("link"),
                transform(next_joint),
                next_joint,
            )

    add(world, next(iter(roots)), dict(pos=numbers(pos), quat=numbers(quat)))
    # PhysX source has self-collision disabled for each articulation, while the
    # two independent arms can contact each other and the shared environment.
    for first, second in itertools.combinations(body_names, 2):
        ET.SubElement(contact, "exclude", body1=first, body2=second)


def build_scene(
    cfg,
    assets,
    *,
    bucket_xy=None,
    ball_position=None,
    solref=None,
    contact_friction=None,
):
    """Return MJCF for two fully articulated UF850 arms and the shared shot geometry.

    The caller sets joint qpos/ctrl from a shared shot and calls mj_forward before
    release. Arms have no free joint. Ball position is its center in table frame.
    """
    assets = Path(assets)
    scene = ET.Element("mujoco", model="sircus-two-arm-bounce")
    ET.SubElement(
        scene, "compiler", angle="radian", inertiafromgeom="false", autolimits="true"
    )
    ET.SubElement(
        scene,
        "option",
        timestep=str(cfg["simulation"]["physics_dt_s"]),
        gravity="0 0 -9.81",
        integrator="implicitfast",
        solver="Newton",
        iterations="50",
        cone="elliptic",
    )
    ET.SubElement(scene, "size", nconmax="512", njmax="2048")
    default = ET.SubElement(scene, "default")
    contact_attrs = dict(
        condim="3",
        friction=numbers([0.5, 0.005, 0.0001]),
        margin="0",
        gap="0",
        solimp="0.95 0.99 0.001 0.5 2",
    )
    restitution = min(max(float(cfg["ball"]["restitution"]), 1e-6), 1 - 1e-6)
    log_e = math.log(restitution)
    damping_ratio = -log_e / math.sqrt(math.pi**2 + log_e**2)
    contact_attrs["solref"] = numbers(
        solref if solref is not None else (0.01, damping_ratio)
    )
    ET.SubElement(default, "geom", **contact_attrs)
    asset = ET.SubElement(scene, "asset")
    world = ET.SubElement(scene, "worldbody")
    ET.SubElement(
        world, "light", pos="1 -2 4", dir="-0.2 0.3 -1", diffuse="0.8 0.8 0.8"
    )
    camera = cfg.get("camera")
    if camera:
        eye = np.asarray(camera["position_m"], dtype=float)
        look = np.asarray(camera["look_at_m"], dtype=float) - eye
        look /= np.linalg.norm(look)
        right = np.cross(look, [0.0, 0.0, 1.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, look)
        vertical_aperture = (
            camera["horizontal_aperture_mm"] * camera["height"] / camera["width"]
        )
        fovy = math.degrees(
            2 * math.atan(vertical_aperture / (2 * camera["focal_length_mm"]))
        )
        ET.SubElement(
            world,
            "camera",
            name="front",
            pos=numbers(eye),
            xyaxes=numbers([*right, *up]),
            fovy=str(fovy),
        )
        visual = ET.SubElement(scene, "visual")
        ET.SubElement(
            visual,
            "global",
            offwidth=str(camera["width"]),
            offheight=str(camera["height"]),
        )
    contact = ET.SubElement(scene, "contact")
    actuator = ET.SubElement(scene, "actuator")
    _arm(
        world,
        asset,
        contact,
        actuator,
        "dropper",
        "Arm10",
        assets,
        (0, 0, 0),
        (1, 0, 0, 0),
    )
    _arm(
        world,
        asset,
        contact,
        actuator,
        "board_arm",
        "Arm14",
        assets,
        (cfg["arms"]["base_spacing_m"], 0, 0),
        quaternion((0, 0, cfg["arms"]["board_base_yaw_rad"])),
    )
    for name, size, pos, cat, color in (
        (
            "table",
            [v / 2 for v in cfg["table"]["size_m"]],
            cfg["table"]["center_m"],
            TABLE,
            (0.23, 0.31, 0.34, 1),
        ),
        ("floor", (2, 2, 0.025), (0.67, -0.25, -0.8), FLOOR, (0.006, 0.006, 0.006, 1)),
    ):
        ET.SubElement(
            world,
            "geom",
            name=name,
            type="box",
            size=numbers(size),
            pos=numbers(pos),
            contype=str(cat),
            conaffinity=str(ALL),
            rgba=numbers(color),
        )
    bucket = cfg["bucket"]
    center = bucket["center_xy_m"] if bucket_xy is None else bucket_xy
    body = ET.SubElement(
        world, "body", name="bucket", pos=numbers([*center, 0]), mocap="true"
    )
    for i in range(bucket["segments"]):
        mesh = f"bucket/wedge{i:02}"
        ET.SubElement(
            asset,
            "mesh",
            name=mesh,
            vertex=numbers(x for p in bucket_wedge(bucket, i) for x in p),
        )
        ET.SubElement(
            body,
            "geom",
            name=f"bucket/wall{i:02}",
            type="mesh",
            mesh=mesh,
            contype=str(BUCKET),
            conaffinity=str(ALL),
            rgba="0.91 0.52 0.12 1",
        )
    thickness = bucket["assumed_bottom_thickness_m"]
    ET.SubElement(
        body,
        "geom",
        name="bucket/base",
        type="cylinder",
        size=numbers([bucket["outer_bottom_diameter_m"] / 2, thickness / 2]),
        pos=numbers([0, 0, thickness / 2]),
        contype=str(BUCKET),
        conaffinity=str(ALL),
        rgba="0.91 0.52 0.12 1",
    )
    b = cfg["ball"]
    if ball_position is None:
        ball_position = (0.1, -0.8, 0.04)
    body = ET.SubElement(world, "body", name="ball", pos=numbers(ball_position))
    ET.SubElement(body, "freejoint", name="ball/free")
    ET.SubElement(
        body,
        "inertial",
        pos="0 0 0",
        mass=str(b["mass_kg"]),
        diaginertia=numbers([0.4 * b["mass_kg"] * b["radius_m"] ** 2] * 3),
    )
    ET.SubElement(
        body,
        "geom",
        name="ball/geom",
        priority="1",
        friction=numbers(
            [
                (
                    0.5 * (cfg["ball"]["friction"] + 0.5)
                    if contact_friction is None
                    else contact_friction
                ),
                0.005,
                0.0001,
            ]
        ),
        type="sphere",
        size=str(b["radius_m"]),
        mass=str(b["mass_kg"]),
        contype=str(BALL),
        conaffinity=str(ALL),
        rgba="0.9 0.1 0.45 1",
    )
    ET.indent(scene)
    return ET.tostring(scene, encoding="unicode")
