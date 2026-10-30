"""Sircus compliant-board sandbox, derived from the imported Viam scene idea.

Coordinates exposed to clients are mm / degrees. Dynamics use SI internally.
This is an uncalibrated, reduced-order model, NOT a hardware force predictor.

A rigid board is supported at a configurable point by one normal translation
spring and two tilt springs. The nominal pose is its preloaded equilibrium;
gravity sag is therefore already included in that pose. Small tilt dynamics
use diagonal inertia about the support. Tangential mount displacement, yaw
compliance, board flexural modes, ball spin, and robot compliance are omitted.
Sphere/box and exact axisymmetric bucket contacts use a positive normal penalty force,
plus dissipative sliding friction. The board can move DURING an impact, so
off-centre torque changes the outgoing trajectory without an edge-specific
restitution rule. Point contact is not a model of the ball's detailed shape.

Reference: https://drake.mit.edu/doxygen_cxx/group__compliant__contact.html
All board/support/contact parameters below are provisional, not measurements.
"""
import copy
import math
import random


def default_config():
    return {
        "units": "mm, degrees, kg, seconds",
        "model": "sircus_compliant_board_v1",
        "calibration": "provisional; board, support and contact parameters unmeasured",
        "roles": {"drop_robot_id": 10, "board_robot_id": 14},
        "ball": {"diameter_mm": 75.0, "mass_kg": 0.2,
                 "contact_stiffness_n_m": 18000.0,
                 "contact_damping_s_m": 0.12, "friction": 0.2},
        "board": {"center_mm": [350.0, -30.0, 190.0],
                  "size_mm": [280.0, 260.0, 12.0],
                  "rpy_deg": [-30.0, 0.0, 0.0], "mass_kg": 0.4,
                  "support_offset_mm": [0.0, 0.0, 0.0],
                  "normal_stiffness_n_m": 12000.0,
                  "normal_damping_ns_m": 18.0,
                  "tilt_stiffness_nm_rad": [30.0, 30.0],
                  "tilt_damping_nms_rad": [0.15, 0.15]},
        "table": {"center_mm": [425.0, 0.0, -5.0],
                  "size_mm": [500.0, 700.0, 40.0], "contact_damping_s_m": 0.3},
        "bin": {"center_xy_mm": [350.0, 220.0],
                "inner_size_mm": [220.0, 220.0], "floor_top_mm": 27.0,
                "wall_height_mm": 100.0, "wall_thickness_mm": 8.0,
                "contact_damping_s_m": 0.6},
        "simulation": {"dt_s": 0.0005, "duration_s": 3.0,
                       "frame_dt_s": 0.02, "noise_xy_mm": 0.0,
                       "noise_gap_mm": 0.0},
    }


def _number(value, name, low=None, high=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(name + " must be a finite number")
    if low is not None and value < low or high is not None and value > high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return float(value)


def _vector(value, length, name, low=None, high=None):
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{name} must have {length} coordinates")
    return [_number(v, name, low, high) for v in value]


def rotation(rpy_deg):
    """Matrix Rz(yaw) Ry(pitch) Rx(roll), row-major, local -> world."""
    r, p, y = [math.radians(v) for v in rpy_deg]
    sr, cr, sp, cp, sy, cy = math.sin(r), math.cos(r), math.sin(p), math.cos(p), math.sin(y), math.cos(y)
    return ((cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr),
            (sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr),
            (-sp, cp*sr, cp*cr))


def matvec(a, v):
    return (a[0][0]*v[0]+a[0][1]*v[1]+a[0][2]*v[2],
            a[1][0]*v[0]+a[1][1]*v[1]+a[1][2]*v[2],
            a[2][0]*v[0]+a[2][1]*v[1]+a[2][2]*v[2])


def transposevec(a, v):
    return (a[0][0]*v[0]+a[1][0]*v[1]+a[2][0]*v[2],
            a[0][1]*v[0]+a[1][1]*v[1]+a[2][1]*v[2],
            a[0][2]*v[0]+a[1][2]*v[1]+a[2][2]*v[2])


def _multiply(a, b):
    return tuple(tuple(sum(a[i][k]*b[k][j] for k in range(3)) for j in range(3)) for i in range(3))


def matrix_rpy(a):
    """Euler angles for display; consumers should prefer the exact matrix."""
    return [math.degrees(math.atan2(a[2][1], a[2][2])),
            math.degrees(math.asin(max(-1.0, min(1.0, -a[2][0])))),
            math.degrees(math.atan2(a[1][0], a[0][0]))]


def _overlap_obb_aabb(center, half, r, other_center, other_half, clearance=0.0):
    """Separating-axis test, optionally requiring a small clearance."""
    board_axes = [tuple(r[i][j] for i in range(3)) for j in range(3)]
    world_axes = [(1,0,0),(0,1,0),(0,0,1)]
    axes = board_axes+world_axes+[_cross(a,b) for a in board_axes for b in world_axes]
    delta = tuple(center[i]-other_center[i] for i in range(3))
    for axis in axes:
        if sum(v*v for v in axis) < 1e-14: continue
        projected_board = sum(half[j]*abs(sum(board_axes[j][i]*axis[i] for i in range(3))) for j in range(3))
        projected_other = sum(other_half[i]*abs(axis[i]) for i in range(3))
        if abs(sum(delta[i]*axis[i] for i in range(3))) >= projected_board+projected_other+clearance*math.sqrt(sum(v*v for v in axis))-1e-7:
            return False
    return True



def _overlap_obb_obb(center, half, rotation_a, other_center, other_half,
                     other_rotation=None, clearance=0.0):
    """Run existing SAT in the second box's local frame."""
    if other_rotation is None:
        return _overlap_obb_aabb(center, half, rotation_a, other_center, other_half, clearance)
    delta = tuple(center[i] - other_center[i] for i in range(3))
    local_center = transposevec(other_rotation, delta)
    transpose = tuple(tuple(other_rotation[j][i] for j in range(3)) for i in range(3))
    local_rotation = _multiply(transpose, rotation_a)
    return _overlap_obb_aabb(local_center, half, local_rotation, (0, 0, 0), other_half, clearance)

def validate_config(config):
    if not isinstance(config, dict):
        raise ValueError("config must be an object")
    try:
        b, p, t, box, s = (config[k] for k in ("ball", "board", "table", "bin", "simulation"))
        _number(b["diameter_mm"], "ball diameter", 1, 300)
        _number(b["mass_kg"], "ball mass", .001, 10)
        _number(b["contact_stiffness_n_m"], "contact stiffness", 100, 200000)
        _number(b["contact_damping_s_m"], "contact dissipation", 0, 2)
        _number(b["friction"], "friction", 0, 2)
        _vector(p["center_mm"], 3, "board center", -3000, 3000)
        size = _vector(p["size_mm"], 3, "board size", 1, 2000)
        angles = _vector(p["rpy_deg"], 3, "board Euler angles", -360, 360)
        _number(p["mass_kg"], "board mass", .05, 20)
        offset = _vector(p["support_offset_mm"], 3, "support offset", -1000, 1000)
        if any(abs(offset[i]) > size[i]/2 for i in range(3)):
            raise ValueError("support point must lie inside or on the board")
        _number(p["normal_stiffness_n_m"], "support normal stiffness", 100, 200000)
        _number(p["normal_damping_ns_m"], "support normal damping", 0, 200)
        _vector(p["tilt_stiffness_nm_rad"], 2, "tilt stiffness", .1, 2000)
        _vector(p["tilt_damping_nms_rad"], 2, "tilt damping", 0, 20)
        _vector(t["center_mm"], 3, "table center", -3000, 3000)
        _vector(t["size_mm"], 3, "table size", 1, 6000)
        _number(t.get("contact_damping_s_m", b["contact_damping_s_m"]), "table contact dissipation", 0, 2)
        _vector(box["center_xy_mm"], 2, "bin center", -3000, 3000)
        shape = box.get("shape", "rectangular")
        if shape not in ("rectangular", "tapered_bucket"):
            raise ValueError("unsupported target container shape")
        _number(box["floor_top_mm"], "bin floor top", -1000, 3000)
        if is_bucket(config):
            for field in ("inner_top_diameter_mm", "outer_top_diameter_mm",
                          "inner_bottom_diameter_mm", "outer_bottom_diameter_mm"):
                _number(box[field], field, 10, 2000)
            _number(box["interior_depth_mm"], "assumed bucket interior depth", 10, 1000)
            _number(box["bottom_thickness_mm"], "assumed bucket base thickness", .1, 100)
            _number(box["reported_height_mm"], "reported bucket height", 10, 1200)
            if box.get("height_interpretation") not in ("unresolved", "overall_height", "interior_depth"):
                raise ValueError("bucket height_interpretation must be explicit")
            if box["inner_top_diameter_mm"] <= b["diameter_mm"] or box["inner_bottom_diameter_mm"] <= b["diameter_mm"]:
                raise ValueError("bucket interior must be wider than the ball")
            if box["inner_top_diameter_mm"] < box["inner_bottom_diameter_mm"] or box["outer_top_diameter_mm"] < box["outer_bottom_diameter_mm"]:
                raise ValueError("bucket must taper outward toward the opening")
            if box["outer_top_diameter_mm"] <= box["inner_top_diameter_mm"] or box["outer_bottom_diameter_mm"] <= box["inner_bottom_diameter_mm"]:
                raise ValueError("bucket outside must enclose its inside")
            if box["interior_depth_mm"] <= b["diameter_mm"]:
                raise ValueError("bucket depth must contain the entire ball")
        else:
            inside = _vector(box["inner_size_mm"], 2, "bin interior", 10, 2000)
            if min(inside) <= b["diameter_mm"]:
                raise ValueError("bin opening must be wider than the ball")
            _number(box["wall_height_mm"], "bin wall height", 10, 1000)
            _number(box["wall_thickness_mm"], "bin wall thickness", 1, 100)
        _number(box.get("contact_damping_s_m", b["contact_damping_s_m"]), "bin contact dissipation", 0, 2)
        dt = _number(s["dt_s"], "time step", .00005, .001)
        _number(s["duration_s"], "duration", .25, 10)
        _number(s["frame_dt_s"], "frame interval", dt, .2)
        _number(s["noise_xy_mm"], "release XY noise", 0, 30)
        _number(s["noise_gap_mm"], "release gap noise", 0, 50)
        normal = matvec(rotation(angles), (0, 0, 1))
        if normal[2] < .35:
            raise ValueError("board top normal must point upward (tilt below about 69 degrees)")
        # Conservative explicit-integrator resolution check, not merely a dt cap.
        dims = [v/1000 for v in size]
        off = [v/1000 for v in offset]
        inertias = [p["mass_kg"]*((dims[1]**2+dims[2]**2)/12+off[1]**2+off[2]**2),
                    p["mass_kg"]*((dims[0]**2+dims[2]**2)/12+off[0]**2+off[2]**2)]
        frequencies = [math.sqrt(b["contact_stiffness_n_m"]*(1/b["mass_kg"]+1/p["mass_kg"])),
                       math.sqrt(p["normal_stiffness_n_m"]/p["mass_kg"])]
        frequencies += [math.sqrt(k/i) for k, i in zip(p["tilt_stiffness_nm_rad"], inertias)]
        if dt*max(frequencies) > .3:
            raise ValueError("time step is too large for these masses/stiffnesses; reduce dt_s")
        damping_rates = [p["normal_damping_ns_m"]/p["mass_kg"]]
        damping_rates += [d/i for d, i in zip(p["tilt_damping_nms_rad"], inertias)]
        if dt*max(damping_rates) > .5:
            raise ValueError("time step is too large for support damping; reduce dt_s")
        for solid in clearance_boxes(config):
            if _overlap_obb_obb(p["center_mm"], [v/2 for v in size], rotation(angles),
                                 solid["center_mm"], [v/2 for v in solid["size_mm"]],
                                 solid.get("rotation_matrix"), clearance=.1):
                raise ValueError("nominal board intersects " + solid["name"] + " (or has less than 0.1 mm clearance)")
    except (KeyError, TypeError) as exc:
        raise ValueError("missing or malformed config field: " + str(exc)) from exc
    return copy.deepcopy(config)


def static_boxes(config):
    """Legacy rectangular ball-contact solids, or only table for analytic bucket."""
    t, b = config["table"], config["bin"]
    if is_bucket(config):
        # Bucket contacts are analytic. Clearance proxies must never become
        # ball-contact boxes or close the open top.
        return [{"name": "table", "center_mm": list(t["center_mm"]), "size_mm": list(t["size_mm"])}]
    x, y = b["center_xy_mm"]
    w, d = b["inner_size_mm"]
    h, thick, floor = b["wall_height_mm"], b["wall_thickness_mm"], b["floor_top_mm"]
    return [
        {"name": "table", "center_mm": list(t["center_mm"]), "size_mm": list(t["size_mm"])},
        {"name": "bin-floor", "center_mm": [x,y,floor-thick/2], "size_mm": [w+2*thick,d+2*thick,thick]},
        {"name": "bin-left", "center_mm": [x-w/2-thick/2,y,floor+h/2], "size_mm": [thick,d+2*thick,h]},
        {"name": "bin-right", "center_mm": [x+w/2+thick/2,y,floor+h/2], "size_mm": [thick,d+2*thick,h]},
        {"name": "bin-front", "center_mm": [x,y-d/2-thick/2,floor+h/2], "size_mm": [w,thick,h]},
        {"name": "bin-back", "center_mm": [x,y+d/2+thick/2,floor+h/2], "size_mm": [w,thick,h]},
    ]



def is_bucket(config):
    return config["bin"].get("shape") == "tapered_bucket"


def bucket_profile(config):
    """Physical cross-section in millimeters; reported height is never inferred.

    interior_depth_mm and bottom_thickness_mm are explicit simulation assumptions
    while height_interpretation is unresolved. Outer taper extends from the
    underside to the rim; inner taper extends from the interior floor to rim.
    """
    b = config["bin"]
    floor = b["floor_top_mm"]
    return {"center_xy_mm": list(b["center_xy_mm"]),
            "inner_bottom_radius_mm": b["inner_bottom_diameter_mm"] / 2,
            "inner_top_radius_mm": b["inner_top_diameter_mm"] / 2,
            "outer_bottom_radius_mm": b["outer_bottom_diameter_mm"] / 2,
            "outer_top_radius_mm": b["outer_top_diameter_mm"] / 2,
            "floor_z_mm": floor, "bottom_z_mm": floor - b["bottom_thickness_mm"],
            "rim_z_mm": floor + b["interior_depth_mm"],
            "interior_depth_mm": b["interior_depth_mm"],
            "outer_height_mm": b["interior_depth_mm"] + b["bottom_thickness_mm"]}


def clearance_boxes(config):
    """Outer-envelope proxies shared with Viam, never used for ball contacts."""
    if not is_bucket(config):
        return static_boxes(config)
    from bucket_geometry import clearance_boxes as outer_clearance
    return outer_clearance(config)



def bucket_contact(position_mm, radius_mm, config):
    """Exact sphere contact with a straight-sided axisymmetric bucket solid.

    Return (surface_name, (penetration_mm, outward_normal, surface_point_mm))
    or None. Closest-distance projection in (radial distance,z) is exact for a
    solid of revolution. The axis closing edge is omitted: it is not a physical
    surface. No disk spans the opening. At multiple contacts this reduced-order
    penalty solver uses the nearest solid surface normal.
    """
    p = bucket_profile(config)
    dx, dy = (position_mm[i] - p["center_xy_mm"][i] for i in range(2))
    rho, z = math.hypot(dx, dy), position_mm[2]
    er = (dx / rho, dy / rho) if rho > 1e-12 else (1.0, 0.0)
    rb, rt = p["outer_bottom_radius_mm"], p["outer_top_radius_mm"]
    ib, it = p["inner_bottom_radius_mm"], p["inner_top_radius_mm"]
    bottom, floor, top = p["bottom_z_mm"], p["floor_z_mm"], p["rim_z_mm"]
    vertices = [(0.0, bottom), (rb, bottom), (rt, top), (it, top), (ib, floor), (0.0, floor)]
    names = ("bucket-underside", "bucket-outer-wall", "bucket-rim", "bucket-inner-wall", "bucket-floor")
    closest = None
    for name, a, b in zip(names, vertices, vertices[1:]):
        dr, dz = b[0] - a[0], b[1] - a[1]
        u = max(0.0, min(1.0, ((rho - a[0]) * dr + (z - a[1]) * dz) / (dr * dr + dz * dz)))
        near = (a[0] + u * dr, a[1] + u * dz)
        distance = math.hypot(rho - near[0], z - near[1])
        if closest is None or distance < closest[0]:
            closest = (distance, name, near, dr, dz)
    distance, name, near, dr, dz = closest
    outer_radius = rb + (rt - rb) * (z - bottom) / (top - bottom)
    inner_radius = ib + (it - ib) * (z - floor) / (top - floor)
    inside_outer = bottom <= z <= top and rho <= outer_radius
    in_cavity = floor <= z <= top and rho < inner_radius
    inside_solid = inside_outer and not in_cavity
    if not inside_solid and distance >= radius_mm:
        return None
    penetration = radius_mm + distance if inside_solid else radius_mm - distance
    if distance > 1e-12:
        sign = -1 if inside_solid else 1
        nr, nz = sign * (rho - near[0]) / distance, sign * (z - near[1]) / distance
    else:
        # Polygon traversal leaves the material to the left of each edge.
        length = math.hypot(dr, dz)
        nr, nz = dz / length, -dr / length
    normal = (nr * er[0], nr * er[1], nz)
    point = (p["center_xy_mm"][0] + near[0] * er[0],
             p["center_xy_mm"][1] + near[0] * er[1], near[1])
    return name, (penetration, normal, point)


def bucket_contains_sphere(config, position_mm, radius_mm, tolerance_mm=0.0):
    """Sphere containment uses only the inner taper, floor and opening.

    The sloped-wall margin is radius*sqrt(1+slope²), not a radius subtracted from
    the opening circle. A small caller-selected tolerance can accommodate the
    penalty solver's normal contact penetration; geometry tests use zero.
    """
    p = bucket_profile(config)
    z = position_mm[2]
    if z - radius_mm < p["floor_z_mm"] - tolerance_mm or z + radius_mm > p["rim_z_mm"] + tolerance_mm:
        return False
    slope = (p["inner_top_radius_mm"] - p["inner_bottom_radius_mm"]) / p["interior_depth_mm"]
    radial_limit = p["inner_bottom_radius_mm"] + slope * (z - p["floor_z_mm"]) - radius_mm * math.sqrt(1 + slope * slope)
    rho = math.hypot(*(position_mm[i] - p["center_xy_mm"][i] for i in range(2)))
    return rho <= radial_limit + tolerance_mm

def release_point(config, action):
    """U/V specify the nominal contact point on the board top, not world XY.

    Ball starts one radius along the board normal, then gap_mm vertically up.
    With a stationary board and zero noise, a vertical fall first touches U/V.
    """
    if not isinstance(action, dict):
        raise ValueError("action must be an object")
    p, b = config["board"], config["ball"]
    try:
        u = _number(action["u_mm"], "impact U", -p["size_mm"][0]/2+2, p["size_mm"][0]/2-2)
        v = _number(action["v_mm"], "impact V", -p["size_mm"][1]/2+2, p["size_mm"][1]/2-2)
        gap = _number(action["gap_mm"], "vertical release gap", 20, 1000)
    except KeyError as exc:
        raise ValueError("action needs u_mm, v_mm and gap_mm") from exc
    local = (u, v, p["size_mm"][2]/2+b["diameter_mm"]/2)
    world = matvec(rotation(p["rpy_deg"]), local)
    return [p["center_mm"][i]+world[i]+(gap if i == 2 else 0) for i in range(3)]


def _box_contact(local, half, radius):
    """Return penetration, outward normal, nearest surface point; or None."""
    nearest = tuple(max(-h, min(h, q)) for q, h in zip(local, half))
    delta = tuple(q-n for q,n in zip(local,nearest))
    dsq = sum(v*v for v in delta)
    if dsq >= radius*radius:
        return None
    if dsq > 1e-20:
        dist = math.sqrt(dsq)
        return radius-dist, tuple(v/dist for v in delta), nearest
    axis = min(range(3), key=lambda i: half[i]-abs(local[i]))
    sign = 1.0 if local[axis] >= 0 else -1.0
    normal = tuple(sign if i == axis else 0.0 for i in range(3))
    point = list(local)
    point[axis] = sign*half[axis]
    return radius+half[axis]-abs(local[axis]), normal, tuple(point)


def _cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def simulate(config, action, seed=0, frames=True):
    """One stationary release; no network, SDK, arm actuation or hidden targets.

    'retained' requires a slow ball wholly inside the bin for 0.2 s near its
    floor. First floor contact alone is explicitly a different outcome.
    Failures and missed board contacts are valid observations. If tilt exceeds
    20 degrees or support deflection exceeds 80 mm the model reports its limit
    instead of silently clipping it into a physically implausible trajectory.
    """
    c = validate_config(config)
    _number(seed, "seed", 0, 2**32-1)
    if int(seed) != seed:
        raise ValueError("seed must be an integer")
    rng = random.Random(int(seed))
    b, p, s, box = (c[k] for k in ("ball", "board", "simulation", "bin"))
    nominal_release = release_point(c, action)
    release = [nominal_release[0]+rng.uniform(-s["noise_xy_mm"],s["noise_xy_mm"]),
               nominal_release[1]+rng.uniform(-s["noise_xy_mm"],s["noise_xy_mm"]),
               nominal_release[2]+rng.uniform(-s["noise_gap_mm"],s["noise_gap_mm"])]
    position = [v/1000 for v in release]
    velocity = [0.0,0.0,0.0]
    radius, mass = b["diameter_mm"]/2000, b["mass_kg"]
    mb, dt = p["mass_kg"], s["dt_s"]
    r0 = rotation(p["rpy_deg"])
    normal0 = matvec(r0,(0,0,1))
    center0 = tuple(v/1000 for v in p["center_mm"])
    offset = tuple(v/1000 for v in p["support_offset_mm"])
    world_offset = matvec(r0,offset)
    pivot0 = tuple(center0[i]+world_offset[i] for i in range(3))
    size = tuple(v/1000 for v in p["size_mm"])
    half = tuple(v/2 for v in size)
    inertia = (mb*((size[1]**2+size[2]**2)/12+offset[1]**2+offset[2]**2),
               mb*((size[0]**2+size[2]**2)/12+offset[0]**2+offset[2]**2))
    solids = [(a["name"],tuple(v/1000 for v in a["center_mm"]),tuple(v/2000 for v in a["size_mm"])) for a in static_boxes(c)]
    clearance_solids = [(a["name"], tuple(v/1000 for v in a["center_mm"]),
                         tuple(v/2000 for v in a["size_mm"]), a.get("rotation_matrix")) for a in clearance_boxes(c)]
    tapered = is_bucket(c)
    if _box_contact(transposevec(r0,tuple(position[i]-center0[i] for i in range(3))),half,radius):
        raise ValueError("noisy release starts in contact with the board; increase gap or reduce noise")
    for name, solid_center, solid_half in solids:
        if _box_contact(tuple(position[i]-solid_center[i] for i in range(3)),solid_half,radius):
            raise ValueError("release starts in contact with " + name)
    if tapered and bucket_contact(release, radius * 1000, c):
        raise ValueError("release starts in contact with bucket")
    q, qdot, alpha, beta, adot, bdot = 0.0,0.0,0.0,0.0,0.0,0.0
    k, dissipation, mu = b["contact_stiffness_n_m"], b["contact_damping_s_m"], b["friction"]
    trace, contacts = [], []
    impact, first_floor, landing, incoming, outgoing = None, False, None, None, None
    board_touching, floor_recorded, retained = False, False, False
    settled_for, max_q, max_tilt, max_pen, peak_force = 0.0,0.0,0.0,0.0,0.0
    initial_energy = mass*9.81*position[2]
    max_energy = initial_energy
    next_frame = 0.0
    model_limit = None
    rotation_now, center = r0, center0

    def record(t):
        trace.append({"t": round(t,6), "ball_xyz_mm": [round(v*1000,5) for v in position],
                      "board_center_mm": [round(v*1000,5) for v in center],
                      "board_rpy_deg": matrix_rpy(rotation_now),
                      "board_rotation": [list(row) for row in rotation_now]})

    def force_for(contact, normal, relative_velocity, damping=dissipation):
        penetration = contact[0]
        vn = sum(relative_velocity[i]*normal[i] for i in range(3))
        fn = k*penetration*max(0.0,1.0-damping*vn)
        tangent = tuple(relative_velocity[i]-vn*normal[i] for i in range(3))
        speed = math.sqrt(sum(v*v for v in tangent))
        # Stop tangential relative motion without reversing it in one step.
        friction = min(mu*fn, mass*speed/dt) if speed > 1e-12 else 0.0
        result = tuple(fn*normal[i]-(friction*tangent[i]/speed if speed > 1e-12 else 0.0) for i in range(3))
        return result,fn,.5*k*penetration*penetration

    for step in range(int(math.ceil(s["duration_s"]/dt))+1):
        t = step*dt
        # Exact displayed pose; two rotational dynamics coordinates are small-angle.
        ca,sa,cb,sb = math.cos(alpha),math.sin(alpha),math.cos(beta),math.sin(beta)
        relative_rotation = ((cb,0,sb),(sa*sb,ca,-sa*cb),(-ca*sb,sa,ca*cb))
        rotation_now = _multiply(r0,relative_rotation)
        pivot = tuple(pivot0[i]+normal0[i]*q for i in range(3))
        moved_offset = matvec(rotation_now,offset)
        center = tuple(pivot[i]-moved_offset[i] for i in range(3))
        if abs(q)+abs(alpha)+abs(beta) > 1e-6:
            # Board/scene contact is outside this mount model. Broad phase avoids
            # SAT work for distant solids; reject instead of letting it pass
            # through a tabletop or box wall and invent a useful trajectory.
            extents = tuple(sum(abs(rotation_now[i][j])*half[j] for j in range(3)) for i in range(3))
            for name, solid_center, solid_half, solid_rotation in clearance_solids:
                solid_extents = (solid_half if solid_rotation is None else
                                 tuple(sum(abs(solid_rotation[i][j])*solid_half[j] for j in range(3)) for i in range(3)))
                if any(abs(center[i]-solid_center[i]) >= extents[i]+solid_extents[i] for i in range(3)):
                    continue
                if _overlap_obb_obb(center, half, rotation_now, solid_center, solid_half, solid_rotation):
                    model_limit = "moving board contacted " + name + "; board/scene contact is not modeled"
                    break
            if model_limit:
                if frames: record(t)
                break
        if frames and (t+1e-10 >= next_frame or step == 0):
            record(t)
            next_frame += s["frame_dt_s"]
        force = [0.0,0.0,-mass*9.81]
        board_force, torques = 0.0,[0.0,0.0]
        elastic_energy = 0.0
        touching_floor = False
        local = transposevec(rotation_now,tuple(position[i]-center[i] for i in range(3)))
        board_contact = _box_contact(local,half,radius)
        if board_contact:
            normal = matvec(rotation_now,board_contact[1])
            cp = matvec(rotation_now,board_contact[2])
            arm = tuple(center[i]+cp[i]-pivot[i] for i in range(3))
            # Angular velocity for R0 * Rx(alpha) * Ry(beta).
            omega = matvec(r0,(adot,ca*bdot,sa*bdot))
            rot_velocity = _cross(omega,arm)
            contact_velocity = tuple(normal0[i]*qdot+rot_velocity[i] for i in range(3))
            relvel = tuple(velocity[i]-contact_velocity[i] for i in range(3))
            f,fn,elastic = force_for(board_contact,normal,relvel)
            elastic_energy += elastic
            for i in range(3): force[i] += f[i]
            opposite = tuple(-v for v in f)
            board_force += sum(opposite[i]*normal0[i] for i in range(3))
            torque_local = transposevec(r0,_cross(arm,opposite))
            torques[0] += torque_local[0]
            torques[1] += ca*torque_local[1]+sa*torque_local[2]
            max_pen = max(max_pen,board_contact[0])
            peak_force = max(peak_force,fn)
            if not board_touching:
                contacts.append({"t":round(t,6),"object":"board","point_uv_mm":[board_contact[2][0]*1000,board_contact[2][1]*1000]})
                if impact is None:
                    impact = [board_contact[2][0]*1000,board_contact[2][1]*1000]
                    incoming = [v*1000 for v in velocity]
            board_touching = True
        elif board_touching:
            if outgoing is None: outgoing = [v*1000 for v in velocity]
            board_touching = False
        for name, solid_center, solid_half in solids:
            relative = tuple(position[i]-solid_center[i] for i in range(3))
            hit = _box_contact(relative,solid_half,radius)
            if not hit: continue
            surface = c["table"] if name == "table" else box
            f,fn,elastic = force_for(hit,hit[1],velocity,surface.get("contact_damping_s_m",dissipation))
            elastic_energy += elastic
            for i in range(3): force[i] += f[i]
            max_pen = max(max_pen,hit[0])
            if name == "bin-floor":
                touching_floor = True
                first_floor = True
                if not floor_recorded:
                    contacts.append({"t":round(t,6),"object":name,"ball_xyz_mm":[v*1000 for v in position]})
                    floor_recorded = True
            elif not contacts or contacts[-1]["object"] != name:
                contacts.append({"t":round(t,6),"object":name,"ball_xyz_mm":[v*1000 for v in position]})
        if tapered:
            contact = bucket_contact([v * 1000 for v in position], radius * 1000, c)
            if contact:
                name, raw = contact
                hit = (raw[0] / 1000, raw[1], tuple(v / 1000 for v in raw[2]))
                f, fn, elastic = force_for(hit, hit[1], velocity, box.get("contact_damping_s_m", dissipation))
                elastic_energy += elastic
                for i in range(3): force[i] += f[i]
                max_pen = max(max_pen, hit[0])
                if name == "bucket-floor":
                    touching_floor = True
                    first_floor = True
                    if not floor_recorded:
                        contacts.append({"t": round(t,6), "object": name, "ball_xyz_mm": [v*1000 for v in position]})
                        floor_recorded = True
                elif not contacts or contacts[-1]["object"] != name:
                    contacts.append({"t": round(t,6), "object": name, "ball_xyz_mm": [v*1000 for v in position]})
        energy = mass*(sum(v*v for v in velocity)/2+9.81*position[2])+elastic_energy
        energy += .5*mb*qdot*qdot+.5*p["normal_stiffness_n_m"]*q*q
        energy += sum(.5*i*w*w+.5*stiff*a*a for i,w,stiff,a in zip(inertia,(adot,bdot),p["tilt_stiffness_nm_rad"],(alpha,beta)))
        max_energy = max(max_energy,energy)
        if landing is None and impact is not None and position[2]*1000 <= box["floor_top_mm"]+radius*1000 and velocity[2] < 0:
            landing = [position[0]*1000,position[1]*1000]
        inside = (bucket_contains_sphere(c, [v*1000 for v in position], radius*1000, tolerance_mm=.5)
                  if tapered else all(abs(position[i]*1000-box["center_xy_mm"][i])+radius*1000 <= box["inner_size_mm"][i]/2 for i in (0,1)))
        low = abs(position[2]*1000-(box["floor_top_mm"]+radius*1000)) < 3
        slow = sum(v*v for v in velocity) < .06**2
        settled_for = settled_for+dt if inside and low and slow and first_floor else 0.0
        if settled_for >= .2:
            retained = True
            if frames: record(t)
            break
        if position[2] < -1.0 or abs(position[0]) > 4 or abs(position[1]) > 4:
            if frames: record(t)
            break
        max_q = max(max_q,abs(q))
        max_tilt = max(max_tilt,abs(alpha),abs(beta))
        if max_tilt > math.radians(20) or max_q > .08:
            model_limit = "small-deflection support model exceeded 20 degrees or 80 mm"
            if frames: record(t)
            break
        if step*dt >= s["duration_s"]: break
        qdd = (board_force-p["normal_stiffness_n_m"]*q-p["normal_damping_ns_m"]*qdot)/mb
        add = (torques[0]-p["tilt_stiffness_nm_rad"][0]*alpha-p["tilt_damping_nms_rad"][0]*adot)/inertia[0]
        bdd = (torques[1]-p["tilt_stiffness_nm_rad"][1]*beta-p["tilt_damping_nms_rad"][1]*bdot)/inertia[1]
        qdot += qdd*dt
        adot += add*dt
        bdot += bdd*dt
        q += qdot*dt
        alpha += adot*dt
        beta += bdot*dt
        for i in range(3):
            velocity[i] += force[i]/mass*dt
            position[i] += velocity[i]*dt
        if not all(math.isfinite(v) for v in (*position,*velocity,q,qdot,alpha,beta,adot,bdot)):
            raise ValueError("dynamics diverged; reduce timestep and review parameters")

    if frames and (not trace or trace[-1]["t"] < t-1e-7): record(t)
    final = [v*1000 for v in position]
    landing = landing or final[:2]
    error = math.dist(landing,box["center_xy_mm"])
    outcome = "model_limit" if model_limit else "retained" if retained else "board_miss" if impact is None else "floor_contact_not_retained" if first_floor else "miss"
    return {
        "model": c.get("model","sircus_compliant_board_v1"), "calibrated": False,
        "physically_verified": False, "action": dict(action), "seed": int(seed),
        "release_xyz_mm": release, "nominal_release_xyz_mm": nominal_release,
        "frames": trace, "outcome": outcome, "retained": retained,
        "first_floor_contact": first_floor, "impact_uv_mm": impact,
        "landing_xy_mm": landing, "final_ball_xyz_mm": final,
        "final_speed_mm_s": math.sqrt(sum(v*v for v in velocity))*1000,
        "incoming_mm_s": incoming, "outgoing_mm_s": outgoing,
        "center_error_mm": error,
        "reward": (1.0 if retained else .25 if first_floor else 0.0)-error/1000-(1 if model_limit else 0),
        "contacts": contacts, "duration_s": t,
        "max_support_displacement_mm": max_q*1000,
        "max_support_tilt_deg": math.degrees(max_tilt),
        "max_penetration_mm": max_pen*1000, "peak_board_force_n": peak_force,
        "initial_energy_j": initial_energy, "max_energy_j": max_energy,
        "model_limit": model_limit,
        "success_definition": ("ball wholly inside inner bucket frustum (0.5mm numerical contact tolerance), near floor, speed <60mm/s for0.2s"
                               if tapered else "ball wholly inside bin, near floor, speed <60 mm/s for 0.2 s"),
        "container_shape": box.get("shape", "rectangular"),
    }
