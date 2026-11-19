"""Versioned camera plans keep historical captures reproducible."""


def recording_plan(name):
    if name == "fixed-station-v1":
        from station_flight import SHOTS, camera_pose
    elif name == "dark-tour-v1":
        from dark_flight import SHOTS, camera_pose
    else:
        raise ValueError(f"Unknown recording plan: {name}")
    return SHOTS, camera_pose
