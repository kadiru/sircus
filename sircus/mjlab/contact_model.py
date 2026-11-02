"""Versioned ideal-held board contact model, never an implicit baseline change."""

import xml.etree.ElementTree as ET

from .model import build_scene

VERSION = "sircus-held-board-contact-v2"


def board_xml(xml, mode):
    if mode not in ("articulated", "kinematic"):
        raise ValueError("Unknown board contact mode")
    if mode == "articulated":
        return xml
    root = ET.fromstring(xml)
    board = root.find(".//body[@name='Arm14/impact_board']")
    if board is None or board.find("joint") is not None:
        raise ValueError("Expected one fixed attached impact board")
    parent = next(p for p in root.iter() if board in list(p))
    parent.remove(board)
    board.set("pos", "0 0 0")
    board.set("quat", "1 0 0 0")
    board.set("mocap", "true")
    root.find("worldbody").append(board)
    # Exclusions remain valid by body name. All geometries retain names,
    # dimensions, contact masks and material parameters.
    return ET.tostring(root, encoding="unicode")


def make_model(scene, assets, mode):
    import mujoco
    xml = build_scene(scene, assets, solref=(.004, .15), contact_friction=.35)
    attached = mujoco.MjModel.from_xml_string(xml)
    return mujoco.MjSpec.from_string(board_xml(xml, mode)), attached
