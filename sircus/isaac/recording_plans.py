"""Versioned camera plans keep historical captures reproducible."""


def setup_capture_progress(step, total_steps, measured_progress, training=False):
    """Schedule training RGB by command time, independent of tiny pose residuals.

    Frames still contain measured joint angles. This only schedules the camera;
    joint readiness and the final pose/velocity/drift guards remain separate.
    Historical replay keeps its original measured-progress schedule.
    """
    return min(1.0, step / total_steps) if training else measured_progress


def transition_group_ready(
    supported_ready, foreground_ready, background_count, require_all=False
):
    """Expanded training waits for every qualified cell; legacy filming keeps its gate."""
    if require_all:
        return foreground_ready and all(supported_ready)
    return foreground_ready and background_count >= 2


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
