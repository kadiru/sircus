"""Expand pinned upstream UF850 descriptions; generate private USD-import inputs.

The upstream model is downloaded, not redistributed in Git. Tool geometries,
fixed board mounting and IK target poses are Sircus provisional approximations.
"""

import argparse
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from geometry import board_world_transform
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def matrix(xyz=(0, 0, 0), rpy=(0, 0, 0)):
    t = np.eye(4)
    t[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    t[:3, 3] = xyz
    return t


def vec(s):
    return [float(x) for x in s.split()]


def chain(root):
    return [
        (
            matrix(vec(j.find("origin").get("xyz")), vec(j.find("origin").get("rpy"))),
            np.array(vec(j.find("axis").get("xyz"))),
            float(j.find("limit").get("lower")),
            float(j.find("limit").get("upper")),
        )
        for i in range(1, 7)
        for j in [root.find(f"joint[@name='joint{i}']")]
    ]


def fk(q, links):
    t = np.eye(4)
    for angle, (origin, axis, *_) in zip(q, links):
        r = np.eye(4)
        r[:3, :3] = Rotation.from_rotvec(axis * angle).as_matrix()
        t = t @ origin @ r
    return t


def solve(target, links):
    lower = np.array([l[2] for l in links])
    upper = np.array([l[3] for l in links])

    def residual(q):
        cur = fk(q, links)
        return np.r_[
            cur[:3, 3] - target[:3, 3],
            0.25 * Rotation.from_matrix(target[:3, :3] @ cur[:3, :3].T).as_rotvec(),
        ]

    rng = np.random.default_rng(42)
    global_rng = np.random.default_rng(42)
    best = None
    for attempt in range(48):
        q = np.clip(
            np.array([0, -0.5, -1.5, 0, 0.5, 0]) + rng.normal(0, 0.7, 6),
            lower + 0.001,
            upper - 0.001,
        )
        # Facing the opposing arm may need the other elbow configuration.
        if attempt >= 12:
            q = global_rng.uniform(lower + 0.001, upper - 0.001)
        r = least_squares(
            residual,
            q,
            bounds=(lower, upper),
            max_nfev=500,
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
        )
        if best is None or np.linalg.norm(r.fun) < np.linalg.norm(best.fun):
            best = r
        if np.linalg.norm(r.fun) < 1e-6:
            break
    if np.linalg.norm(best.fun) > 1e-4:
        raise RuntimeError(f"IK residual too large: {best.fun}")
    return best.x.tolist(), fk(best.x, links).tolist()


def rigid_link(root, name, size, mass, color):
    link = ET.SubElement(root, "link", name=name)
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "mass", value=str(mass))
    ET.SubElement(inertial, "origin", xyz="0 0 0", rpy="0 0 0")
    x, y, z = size
    ET.SubElement(
        inertial,
        "inertia",
        ixx=str(mass * (y * y + z * z) / 12),
        iyy=str(mass * (x * x + z * z) / 12),
        izz=str(mass * (x * x + y * y) / 12),
        ixy="0",
        ixz="0",
        iyz="0",
    )
    for tag in ("visual", "collision"):
        item = ET.SubElement(link, tag)
        geo = ET.SubElement(item, "geometry")
        ET.SubElement(geo, "box", size=" ".join(map(str, size)))
        if tag == "visual":
            mat = ET.SubElement(item, "material", name=name)
            ET.SubElement(mat, "color", rgba=color)
    return link


def fixed(root, name, parent, child, xyz, rpy="0 0 0"):
    j = ET.SubElement(root, "joint", name=name, type="fixed")
    ET.SubElement(j, "parent", link=parent)
    ET.SubElement(j, "child", link=child)
    ET.SubElement(j, "origin", xyz=xyz, rpy=rpy)


def main():
    import xacro

    p = argparse.ArgumentParser()
    p.add_argument("--upstream", default="/workspace/vendor/xarm_ros2/xarm_description")
    p.add_argument("--output", default="/workspace/assets")
    a = p.parse_args()
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    src = Path(a.upstream)
    cfg = json.loads((Path(__file__).parent / "scene.json").read_text())
    # Stage only the macros; replace ROS package lookup with explicit known path.
    macros = out / "xacro"
    shutil.copytree(src / "urdf", macros, dirs_exist_ok=True)
    for f in macros.rglob("*.xacro"):
        f.write_text(f.read_text().replace("$(find xarm_description)", str(src)))
    wrapper = f'''<robot xmlns:xacro="http://ros.org/wiki/xacro" name="uf850">
      <xacro:property name="use_xacro_load_yaml" value="true"/>
      <xacro:property name="mesh_suffix" value="stl"/>
      <xacro:property name="mesh_path" value="{src}/meshes"/>
      <xacro:include filename="{macros}/common/common.material.xacro"/>
      <xacro:include filename="{macros}/common/common.link.xacro"/>
      <xacro:include filename="{macros}/uf850/uf850.urdf.xacro"/>
      <xacro:uf850_urdf prefix="" kinematics_params_filename="{src}/config/kinematics/default/uf850_default_kinematics.yaml"/>
    </robot>'''
    path = out / "uf850.xacro"
    path.write_text(wrapper)
    xml = xacro.process_file(str(path)).toxml()
    base = ET.fromstring(xml)
    links = chain(base)
    # Board surface normal points toward the operator-side bucket.
    b = cfg["board"]
    board_world = board_world_transform(b)
    mount = matrix([0, 0, b["tool_offset_m"]], [0, 0, b["mount_yaw_rad"]])
    base14 = matrix([cfg["arms"]["base_spacing_m"], 0, 0], [0, 0, np.pi])
    target14 = np.linalg.inv(base14) @ board_world @ np.linalg.inv(mount)
    q14, actual14 = solve(target14, links)
    target10 = matrix(cfg["dropper"]["flange_target_m"], [np.pi, 0, 0])
    q10, actual10 = solve(target10, links)
    for role in ("dropper", "board_arm"):
        root = ET.fromstring(xml)
        if role == "board_arm":
            rigid_link(
                root, "suction_proxy", [0.07, 0.07, 0.07], 0.1, "0.25 0.25 0.3 1"
            )
            fixed(root, "suction_mount", "link6", "suction_proxy", "0 0 0.04")
            rigid_link(
                root, "impact_board", b["size_m"], b["mass_kg"], "0.1 0.65 0.65 1"
            )
            fixed(
                root,
                "board_mount",
                "link6",
                "impact_board",
                f"0 0 {b['tool_offset_m']}",
                f"0 0 {b['mount_yaw_rad']}",
            )
        else:
            rigid_link(
                root, "gripper_proxy", [0.065, 0.065, 0.04], 0.3, "0.2 0.25 0.3 1"
            )
            fixed(root, "gripper_mount", "link6", "gripper_proxy", "0 0 .03")
            for side in (-1, 1):
                name = "finger_left" if side == 1 else "finger_right"
                rigid_link(root, name, [0.018, 0.018, 0.085], 0.03, "0.25 0.3 0.35 1")
                j = ET.SubElement(root, "joint", name=name + "_joint", type="prismatic")
                ET.SubElement(j, "parent", link="link6")
                ET.SubElement(j, "child", link=name)
                ET.SubElement(j, "origin", xyz=f"0 {side * 0.025} .095", rpy="0 0 0")
                ET.SubElement(j, "axis", xyz=f"0 {side} 0")
                ET.SubElement(
                    j, "limit", lower="0", upper=".035", effort="30", velocity=".1"
                )
        ET.indent(root)
        ET.ElementTree(root).write(
            out / f"{role}.urdf", encoding="utf-8", xml_declaration=True
        )
        for mesh in root.findall(".//mesh"):
            if not Path(mesh.get("filename")).is_file():
                raise FileNotFoundError(mesh.get("filename"))
    report = {
        "dropper_joint_rad": q10,
        "board_joint_rad": q14,
        "dropper_flange_local_matrix": actual10,
        "board_flange_local_matrix": actual14,
        "board_world_matrix": board_world.tolist(),
        "status": "offline endpoint IK only; trajectories and grasp not validated",
    }
    (out / "poses.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
