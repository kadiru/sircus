"""Compact 35-table clone layout and camera-visible, non-colliding stage."""

import numpy as np
from cylinder_farm import COLS, ROWS

COL_PITCH_M = 2.50
ROW_PITCH_M = 2.10
FLOOR_Z_M = -0.775  # Top of the original 50 mm thick floor collider at -0.8 m.
SEAM_WIDTH_M = 0.009
FLOOR_RGB = (0.003, 0.003, 0.003)
SEAM_RGB = (0.019, 0.019, 0.019)
STAGE_HALF_M = 120.0


def origins():
    return np.array(
        [
            [(c - (COLS - 1) / 2) * COL_PITCH_M, (r - (ROWS - 1) / 2) * ROW_PITCH_M, 0]
            for r in range(ROWS)
            for c in range(COLS)
        ],
        dtype=np.float32,
    )


def stage_settings():
    return dict(
        columns=COLS,
        rows=ROWS,
        column_pitch_m=COL_PITCH_M,
        row_pitch_m=ROW_PITCH_M,
        floor_surface_z_m=FLOOR_Z_M,
        floor_linear_rgb=FLOOR_RGB,
        seam_linear_rgb=SEAM_RGB,
        seam_width_m=SEAM_WIDTH_M,
        floor_visual_span_m=2 * STAGE_HALF_M,
        floor_collision="Original filtered 4 m square per-cell colliders; invisible to the camera.",
    )


def _bands(count, pitch):
    """Split one axis into the cell interiors, the thin seams, and the backdrop."""
    edges = [-STAGE_HALF_M, -count * pitch / 2]
    for i in range(1, count):
        center = (i - count / 2) * pitch
        edges.extend((center - SEAM_WIDTH_M / 2, center + SEAM_WIDTH_M / 2))
    return edges + [count * pitch / 2, STAGE_HALF_M]


def floor_mesh(center_xy):
    """Return a watertight tiling of the visible plane, with separate seam faces."""
    xs, ys = _bands(COLS, COL_PITCH_M), _bands(ROWS, ROW_PITCH_M)
    cx, cy = center_xy
    points = [(cx + x, cy + y, FLOOR_Z_M) for y in ys for x in xs]
    floor, seams = [], []
    for iy in range(len(ys) - 1):
        for ix in range(len(xs) - 1):
            inside_grid = 0 < ix < len(xs) - 2 and 0 < iy < len(ys) - 2
            seam = inside_grid and (ix % 2 == 0 or iy % 2 == 0)
            i = iy * len(xs) + ix
            (seams if seam else floor).extend((i, i + 1, i + len(xs) + 1, i + len(xs)))
    return points, floor, seams


def configure_stage(scfg, cfg):
    """Call after SimulationContext creation, before cloning the InteractiveScene."""
    import isaaclab.sim as sim_utils
    from pxr import Sdf, UsdGeom, UsdShade

    # Visibility is independent of PhysX and of the per-ball contact sensor filter.
    scfg.Floor.spawn.visible = False
    stage = sim_utils.get_current_stage()
    root = "/World/SircusDarkStage"
    UsdGeom.Xform.Define(stage, root)
    points, floor, seams = floor_mesh(cfg["table"]["center_m"][:2])
    for name, faces, color in (("Floor", floor, FLOOR_RGB), ("Seams", seams, SEAM_RGB)):
        mesh = UsdGeom.Mesh.Define(stage, f"{root}/{name}")
        mesh.CreatePointsAttr(points)
        mesh.CreateFaceVertexCountsAttr([4] * (len(faces) // 4))
        mesh.CreateFaceVertexIndicesAttr(faces)
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateDisplayColorAttr([color])
        material_path = f"{root}/Look{name}"
        mat = UsdShade.Material.Define(stage, material_path)
        shader = UsdShade.Shader.Define(stage, f"{material_path}/Surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color)
        shader.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(1)
        shader.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set((0, 0, 0))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
        surface = shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        mat.CreateSurfaceOutput().ConnectToSource(surface)
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
