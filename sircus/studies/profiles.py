"""Explicit scene profiles; no source geometry or historical fingerprint is rewritten."""
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROFILE_ID = 'front-negative-y-v1'
HISTORICAL_PROFILE_ID = 'historical-positive-y-v1'
DISTRIBUTION = {
    'board_center_mm': [[570., 630.], [-180., -120.], [330., 375.]],
    'bucket_relative_xy_mm': [[-120., 120.], [-360., -140.]],
    'bucket_absolute_y_mm': [-600., -150.],
    'scope': 'front-bucket core; challenges extend beyond this trained layout distribution',
}


def historical_scene():
    """Unchanged historical positive-Y asset definition, not a corrected profile."""
    return json.loads((ROOT/'sircus/two-arm/scene.json').read_text())


def front_scene():
    scene = historical_scene()
    scene['scene_id'] = 'sircus-front-negative-y-provisional-v1'
    scene['board']['center_mm'] = [600., -150., 350.]
    scene['board']['rpy_deg'] = [0., -30., 90.]
    scene['bin']['center_xy_mm'] = [600., -400.]
    scene['table']['center_mm'] = [668.655, -300., -5.]
    scene['table']['size_mm'] = [1537.31, 1000., 40.]
    # This inherited numerical field uses a ball-at-TCP proxy. It is NOT a newly
    # calibrated grasp/ball pose derived from the user's screenshot.
    scene['pickup_mm'] = [150., -600., 200.]
    scene['measurement_status'].update(
        coordinate_profile=PROFILE_ID,
        coordinate_axes='Dropper base origin; +X toward board-arm base; +Z up; operator toward -Y',
        board_placement='Provisional nominal center near dropper release XY [600,-150]',
        board_orientation='Provisional negative-Y-facing normal: Rz(90) Ry(-30) Rx(0)',
        table_placement='Provisional expanded tabletop supporting front-side bucket layouts; unmeasured',
        bucket_placement='User clarified negative-Y/operator side; exact placement remains provisional',
        loading_pose='Screenshot TCP [150,-600,200] mm; numerical ball-at-TCP proxy uncalibrated',
        dropping_pose='Screenshot TCP [600,-150,600] mm; not a measured ball center or calibrated board gap')
    return scene


def profile_metadata():
    return {
        'profile_id': PROFILE_ID, 'historical_profile_id': HISTORICAL_PROFILE_ID,
        'geometry_status': 'provisional placements, orientations, tool and contact parameters',
        'coordinate_frame': {'origin': 'dropper arm base', 'positive_x': 'toward arm14 base',
                             'positive_z': 'up', 'operator_y_approx_mm': -900.,
                             'user_bucket_y_interval_mm': [-600., -150.]},
        'measured': {'board_size_mm': [200., 300., 5.], 'arm_spacing_mm': 1337.31,
                     'bucket_inner_top_bottom_diameter_mm': [177., 150.],
                     'bucket_outer_top_bottom_diameter_mm': [200., 158.],
                     'bucket_reported_height_mm': 175.},
        'unresolved': ['175 mm overall height versus interior depth',
                       'base spacing reference/full rigid transform', 'tool-to-board mount',
                       'ball center relative to gripper TCP', 'contact and compliance dynamics'],
        'screenshot_tcp_targets_mm': {'loading': [150., -600., 200.], 'dropping': [600., -150., 600.]},
        'screenshot_status': 'user-reported TCP targets, not ball-center or calibrated task poses',
        'distribution': copy.deepcopy(DISTRIBUTION), 'hardware_enabled': False,
        'native_paths_checked': False,
    }
