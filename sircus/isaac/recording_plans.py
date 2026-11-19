"""Versioned camera plans keep historical captures reproducible."""


def recording_plan(name, camera=None):
    if name == "fixed-station-v1":
        from station_flight import SHOTS, camera_pose
    elif name == "dark-tour-v1":
        from dark_flight import SHOTS, camera_pose
    elif name == "chosen-farm-v1":
        from functools import partial
        from chosen_flight import SHOTS, camera_pose

        if camera is None:
            raise ValueError("The chosen film requires its saved camera")
        return SHOTS, partial(camera_pose, camera=camera)
    else:
        raise ValueError(f"Unknown recording plan: {name}")
    return SHOTS, camera_pose


def plan_origins(name):
    if name == "chosen-farm-v1":
        from camera_options import preview_origins as origins
    else:
        from dark_stage import origins
    return origins()
