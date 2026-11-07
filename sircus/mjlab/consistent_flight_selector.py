"""Frozen-model flight ranking without observed trajectories or native probes.

The source network's rebound-velocity heads are combined with the requested
board's nominal sphere/plane tangent point and gravity.  That contact origin is
an approximation: the source targets were measured after finite native contact.
The resulting aperture margins are geometric ranking proxies, not calibrated
capture probabilities, collision guarantees, or retention predictions.

Rim-assisted captures remain possible for negative margins.  Every candidate
remains in the ordering; this module does not alter native admission or scoring.
"""
from __future__ import annotations

import copy
import math

import numpy as np

from .calibrated_score import _geometry, validate_calibration


VERSION = "sircus-consistent-flight-ranking-v1"
MEMBERS = 5
CANDIDATES = 16
SCOPE = (
    "Frozen velocity heads plus requested-geometry tangent origin and gravity. "
    "Conservative rim-band clearance is a nominal ranking proxy, not capture. "
    "No observed rebound, new fit, native flight search, robot-flight collision "
    "proof or rim-contact/retention prediction. Negative margins do not exclude "
    "rim-assisted capture."
)


def _vector(value, size, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError("Finite " + name + " required")
    return result


def geometry(scene, calibration, *, gravity=(0., 0., -9.81)):
    """Use the unchanged compiled bucket facets; never fit geometric tolerances."""
    cfg = validate_calibration(calibration)
    _, _, _, radius, floor, rim, planes, horizontal = _geometry(
        scene, [0., 0.], cfg, np.zeros((1, 3)))
    size = _vector(scene['board']['size_m'], 3, 'board dimensions')
    grav = _vector(gravity, 3, 'gravity')
    yaw = float(scene['board']['mount_yaw_rad'])
    if (size <= 0).any() or not math.isfinite(yaw):
        raise ValueError('Positive board dimensions and finite mount yaw required')
    if not np.allclose(grav[:2], 0., rtol=0, atol=0) or grav[2] >= 0:
        raise ValueError('The frozen vertical-gravity model requires gravity [0,0,-g]')
    return dict(ball_radius_m=float(radius), board_size_m=size.tolist(),
        mount_yaw_rad=yaw, rim_m=float(rim), floor_m=float(floor),
        inward_rim_halfplanes=np.column_stack((planes[:, :2]/horizontal[:, None],
            (planes[:, 2]*rim+planes[:, 3])/horizontal)).tolist(),
        gravity_m_s2=grav.tolist(), calibration_fingerprint=cfg['fingerprint'],
        bucket_reported_height_m=scene['bucket'].get('reported_height_m'),
        bucket_height_interpretation=scene['bucket'].get('height_interpretation'))


def nominal_origin(values, bucket_xy, geom):
    """First tangent point for the *center* of a vertically falling sphere.

    The board center is at its mid-thickness plane.  Its normal points toward
    the bucket.  The ball's XY remains the requested board-center XY; adding
    radius along the tilted normal would incorrectly shift this vertical ray.
    """
    action = _vector(values, 5, 'five shot values')
    bucket = _vector(bucket_xy, 2, 'bucket XY')
    delta = bucket-action[:2]
    distance = float(np.linalg.norm(delta))
    tilt, gap = float(action[3]), float(action[4])
    if distance <= 1e-8 or not 0 <= tilt < 90 or gap <= 0:
        raise ValueError('Separated board/bucket, 0<=tilt<90 and positive gap required')
    radius = float(geom['ball_radius_m'])
    size = _vector(geom['board_size_m'], 3, 'board dimensions')
    theta = math.radians(tilt); sine, cosine = math.sin(theta), math.cos(theta)
    direction = delta/distance
    lateral = np.array([-direction[1], direction[0], 0.])
    forward = np.r_[cosine*direction, -sine]
    normal = np.r_[sine*direction, cosine]
    origin = action[:3].copy()
    origin[2] += (radius+size[2]/2)/cosine
    release = action[:3].copy(); release[2] += radius+gap
    contact = origin-radius*normal
    yaw = float(geom['mount_yaw_rad']); c, s = math.cos(yaw), math.sin(yaw)
    rotation = np.column_stack((lateral, -forward, normal)) @ np.array([
        [c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    local = rotation.T @ (contact-action[:3])
    margin = size[:2]/2-np.abs(local[:2])
    support = bool((margin >= 0.).all() and release[2] >= origin[2])
    return dict(origin_m=origin.tolist(), release_center_m=release.tolist(),
        board_normal=normal.tolist(), forward_xy=direction.tolist(),
        contact_point_m=contact.tolist(), board_local_contact_m=local.tolist(),
        contact_edge_margin_m=float(margin.min()), supported_planar_contact=support,
        release_above_tangent=bool(release[2] >= origin[2]),
        board_to_bucket_distance_m=distance,
        provenance='requested_geometry_nominal_sphere_plane_tangent_not_observed')


def _descending_time(origin_z, velocity_z, height, gravity_z):
    """Later (descending) positive root, including a launch below the plane."""
    discriminant = velocity_z*velocity_z-2*gravity_z*(origin_z-height)
    if discriminant < 0:
        return None
    result = (-velocity_z-math.sqrt(max(0., discriminant)))/gravity_z
    if result <= 0 or velocity_z+gravity_z*result >= 0:
        return None
    return float(result)


def trajectory_for_member(origin, velocity, times, *, gravity=(0., 0., -9.81)):
    """Finite contact-free trajectory; callers choose the stated time horizon."""
    p = _vector(origin, 3, 'origin'); v = _vector(velocity, 3, 'velocity')
    g = _vector(gravity, 3, 'gravity'); t = np.asarray(times, dtype=float)
    if t.ndim != 1 or not len(t) or not np.isfinite(t).all() or (t < 0).any() or (np.diff(t) < 0).any():
        raise ValueError('Finite nonnegative ordered trajectory times required')
    return (p+t[:, None]*v+.5*t[:, None]**2*g).tolist()


def member_flight(origin, velocity_world, bucket_xy, geom):
    """Calculate rim diagnostics, with no subsequent-contact or outcome labels.

    During a full downward passage, the sphere intersects the rim plane while
    its center moves from rim+radius to rim-radius.  Using the *full* radius
    throughout that band is conservative.  The minimum polygon half-plane
    margin occurs at one of its two endpoints because horizontal motion is
    linear.  A positive margin is sufficient only for this ideal rim aperture;
    it says nothing about earlier robot/wall contacts or later retention.
    """
    p = _vector(origin, 3, 'origin'); v = _vector(velocity_world, 3, 'world velocity')
    bucket = _vector(bucket_xy, 2, 'bucket XY')
    gravity = _vector(geom['gravity_m_s2'], 3, 'gravity')
    if np.any(gravity[:2] != 0) or gravity[2] >= 0:
        raise ValueError('Vertical downward gravity required')
    radius, rim = float(geom['ball_radius_m']), float(geom['rim_m'])
    planes = np.asarray(geom['inward_rim_halfplanes'], dtype=float)
    if planes.ndim != 2 or planes.shape[1] != 3 or len(planes) < 3 or not np.isfinite(planes).all():
        raise ValueError('Finite polygon rim half-planes required')
    apex_t = max(0., -v[2]/gravity[2])
    apex_z = float(p[2]+v[2]*apex_t+.5*gravity[2]*apex_t*apex_t)
    times = [_descending_time(p[2], v[2], z, gravity[2])
             for z in (rim+radius, rim, rim-radius)]

    def margin(time):
        point = p[:2]+v[:2]*time
        return float((planes[:, :2] @ (point-bucket)+planes[:, 2]).min()-radius)

    center_t = times[1]
    center = None if center_t is None else dict(time_s=center_t,
        xy_m=(p[:2]+center_t*v[:2]).tolist(), clearance_m=margin(center_t))
    complete = bool(all(t is not None for t in times) and times[0] < times[1] < times[2])
    band_margin = min(margin(times[0]), margin(times[2])) if complete else None
    # No-root is never synthesized into a fake crossing or captured label.
    return dict(origin_m=p.tolist(), velocity_world_m_s=v.tolist(),
        apex_height_m=apex_z, apex_time_s=float(apex_t),
        apex_above_full_sphere_rim_m=float(apex_z-rim-radius),
        starts_below_rim=bool(p[2] < rim),
        ascending_before_descent=bool(v[2] > 0),
        first_descending_center_rim=center,
        full_downward_rim_band_available=complete,
        rim_band_start_s=times[0] if complete else None,
        rim_band_end_s=times[2] if complete else None,
        conservative_rim_band_clearance_m=band_margin,
        nominal_clear_aperture=bool(complete and band_margin >= 0),
        missing_reason=None if complete else 'no_future_complete_descending_sphere_rim_band',
        contacts_modeled=False, capture_predicted=False)


def predict_candidate(scene, calibration, bucket_xy, values, frozen_prediction,
                      *, gravity=(0., 0., -9.81), prepared_geometry=None):
    """Read only the five raw velocity heads; never consume measured outcomes."""
    geom = prepared_geometry or geometry(scene, calibration, gravity=gravity)
    launch = nominal_origin(values, bucket_xy, geom)
    raw = np.asarray(frozen_prediction.get('member_raw_output'), dtype=float)
    if raw.shape != (MEMBERS, 9) or not np.isfinite(raw).all():
        raise ValueError('Exact five frozen finite nine-output members required')
    d = np.asarray(launch['forward_xy']); side = np.array([-d[1], d[0]])
    members = []
    for index, velocity in enumerate(raw[:, 6:9]):
        world = np.r_[velocity[0]*d+velocity[1]*side, velocity[2]]
        result = member_flight(launch['origin_m'], world, bucket_xy, geom)
        result.update(member=index, velocity_forward_lateral_vertical_m_s=velocity.tolist())
        members.append(result)
    available = [r['conservative_rim_band_clearance_m'] for r in members
                 if r['full_downward_rim_band_available']]
    original = float(frozen_prediction.get('ranking_score', float('nan')))
    if not math.isfinite(original):
        raise ValueError('Finite frozen original ranking score required')
    return dict(version=VERSION, launch=launch, members=members,
        complete_rim_band_members=len(available),
        nominal_clear_aperture_members=sum(r['nominal_clear_aperture'] for r in members),
        worst_clearance_m=min(available) if available else None,
        mean_clearance_m=float(np.mean(available)) if available else None,
        minimum_apex_margin_m=min(r['apex_above_full_sphere_rim_m'] for r in members),
        original_ranking_score=original, scope=SCOPE,
        new_native_queries=0, uses_observed_rebound=False)


def rank(scene, calibration, context, items, predictions, *, gravity=(0., 0., -9.81)):
    """Deterministic lexicographic geometry ranking of the SAME 16 proposals.

    Supported nominal board contact first, then the number of members with a
    complete future rim band, worst member margin, mean margin, frozen original
    ranking and candidate ID.  No learned classification score is interpreted
    as probability.  Missing margins sort after available margins; no shot is
    removed, substituted or executed here.
    """
    if len(items) != CANDIDATES or len(predictions) != CANDIDATES:
        raise ValueError('Exactly sixteen candidates and matching frozen predictions required')
    ids = [r.get('id') for r in items]
    if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != CANDIDATES:
        raise ValueError('Sixteen unique nonempty candidate IDs required')
    geom = geometry(scene, calibration, gravity=gravity)
    rows = [predict_candidate(scene, calibration, context['bucket_xy_m'], item['values'], prediction,
                gravity=gravity, prepared_geometry=geom)
            for item, prediction in zip(items, predictions)]
    for item, row in zip(items, rows):
        row['candidate_id'] = item['id']

    def key(i):
        row = rows[i]
        worst = row['worst_clearance_m']; mean = row['mean_clearance_m']
        # Infinity is internal to sorting only; JSON evidence retains None.
        return (-int(row['launch']['supported_planar_contact']), -row['complete_rim_band_members'],
            -(worst if worst is not None else -math.inf),
            -(mean if mean is not None else -math.inf), -row['original_ranking_score'], ids[i])

    return dict(version=VERSION, order=sorted(range(CANDIDATES), key=key),
        predictions=copy.deepcopy(predictions), diagnostics=rows, geometry=geom,
        candidate_count=CANDIDATES, ball_flight_search_at_inference=False,
        ranking_rule=['supported_planar_contact_descending', 'complete_rim_band_members_descending',
            'worst_clearance_descending', 'mean_clearance_descending',
            'original_rank_descending', 'candidate_id_ascending'],
        capture_probability_calibrated=False, uses_observed_rebound=False,
        scope=SCOPE)
