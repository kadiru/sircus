"""Selected silver-gray surfaces and slightly brighter, softer studio lighting."""

import dark_stage

FLOOR_RGB = (0.22, 0.23, 0.24)
TABLE_RGB = (0.40, 0.44, 0.47)


def stage_settings():
    settings = dark_stage.stage_settings()
    settings.update(
        palette="silver-soft-v1",
        floor_linear_rgb=FLOOR_RGB,
        seam_linear_rgb=FLOOR_RGB,
        table_linear_rgb=TABLE_RGB,
        table_roughness=0.72,
        ambient_occlusion=False,
        lighting=dict(
            dome_fill=1050,
            key=dict(
                name="SilverKey",
                intensity=320,
                angle=22.0,
                rotation_xyz_deg=[-25, -35, 0],
            ),
            fills=[
                dict(
                    name="SilverLeftFill",
                    intensity=200,
                    angle=40.0,
                    rotation_xyz_deg=[25, 30, 150],
                ),
                dict(
                    name="SilverRightFill",
                    intensity=160,
                    angle=45.0,
                    rotation_xyz_deg=[-20, 25, -70],
                ),
            ],
        ),
    )
    return settings


def configure_stage(scfg, cfg):
    """Call after SimulationContext creation and before InteractiveScene cloning."""
    import isaaclab.sim as sim_utils
    from pxr import Sdf, UsdShade
    from scipy.spatial.transform import Rotation

    dark_stage.configure_stage(scfg, cfg)
    settings = stage_settings()
    lighting = settings["lighting"]
    scfg.Table.spawn.visual_material = sim_utils.PreviewSurfaceCfg(
        diffuse_color=TABLE_RGB,
        roughness=settings["table_roughness"],
    )
    scfg.light.spawn.intensity = lighting["dome_fill"]
    scfg.light.spawn.color = (1.0, 1.0, 1.0)

    stage = sim_utils.get_current_stage()
    root = "/World/SircusSilverLook/FloorMaterial"
    mat = UsdShade.Material.Define(stage, root)
    shader = UsdShade.Shader.Define(stage, root + "/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(FLOOR_RGB)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.88)
    shader.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(1)
    shader.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set(
        (0.04, 0.04, 0.04)
    )
    mat.CreateSurfaceOutput().ConnectToSource(
        shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    )
    for name in ("Floor", "Seams"):
        prim = stage.GetPrimAtPath("/World/SircusDarkStage/" + name)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)

    for source in (lighting["key"], *lighting["fills"]):
        light = sim_utils.DistantLightCfg(
            intensity=source["intensity"],
            angle=source["angle"],
            color=(1.0, 1.0, 1.0),
        )
        q = Rotation.from_euler(
            "xyz", source["rotation_xyz_deg"], degrees=True
        ).as_quat()
        light.func(
            "/World/" + source["name"],
            light,
            orientation=tuple(map(float, q[[3, 0, 1, 2]])),
        )
