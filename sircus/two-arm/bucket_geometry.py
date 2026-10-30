"""Conservative native-Viam clearance geometry for an open tapered bucket.

Ball contacts use the analytic inner geometry in physics.py, never these boxes.
The wall's 24 OBB sectors enclose each curved frustum sector, aligned to the
outside straight taper. This can slightly shrink the usable opening and wall
interior; it never underestimates outer wall clearance or creates a top lid.
The floor is a conservative square bounding box using the outside radius there.
All dimensions and positions are millimeters.
"""
import math

WALL_SECTORS = 24


def clearance_boxes(config):
    b = config['bin']
    if b.get('shape') != 'tapered_bucket':
        from physics import static_boxes
        return static_boxes(config)
    t = config['table']
    result = [{'name': 'table', 'center_mm': list(t['center_mm']),
               'size_mm': list(t['size_mm'])}]
    x, y = b['center_xy_mm']
    floor, depth, bottom = (b[k] for k in ('floor_top_mm', 'interior_depth_mm', 'bottom_thickness_mm'))
    inner0, inner1 = b['inner_bottom_diameter_mm']/2, b['inner_top_diameter_mm']/2
    outer0, outer1 = b['outer_bottom_diameter_mm']/2, b['outer_top_diameter_mm']/2
    k = (outer1-outer0)/(depth+bottom)
    outer_floor = outer0+k*bottom
    norm = math.sqrt(1+k*k)
    half_angle = math.pi/WALL_SECTORS
    # Each radial/height vertex of the wall is swept over one angular sector.
    # Projection extrema occur at its vertices and delta in {-half_angle,0,+half_angle}.
    projections = []
    for radius, z in ((inner0, 0), (outer_floor, 0), (inner1, depth), (outer1, depth)):
        for delta in (-half_angle, 0, half_angle):
            radial, tangent = radius*math.cos(delta), radius*math.sin(delta)
            projections.append(((radial-k*z)/norm, tangent, (k*radial+z)/norm))
    low = [min(p[i] for p in projections) for i in range(3)]
    high = [max(p[i] for p in projections) for i in range(3)]
    center = [(lo+hi)/2 for lo, hi in zip(low, high)]
    size = [hi-lo for lo, hi in zip(low, high)]
    for i in range(WALL_SECTORS):
        theta = i*2*math.pi/WALL_SECTORS
        c, s = math.cos(theta), math.sin(theta)
        # Columns: outward normal, tangent, outer-wall generatrix. Right-handed.
        rotation = [[c/norm, -s, k*c/norm], [s/norm, c, k*s/norm], [-k/norm, 0, 1/norm]]
        xyz = [sum(rotation[row][col]*center[col] for col in range(3))
               for row in range(3)]
        result.append({'name': f'bucket-wall-{i:02d}',
                       'center_mm': [x+xyz[0], y+xyz[1], floor+xyz[2]],
                       'size_mm': size.copy(), 'rotation_matrix': rotation,
                       'geometry_scope': 'conservative outer-taper wall OBB'})
    result.append({'name': 'bucket-floor-clearance',
                   'center_mm': [x, y, floor-bottom/2],
                   'size_mm': [2*outer_floor, 2*outer_floor, bottom],
                   'geometry_scope': 'conservative outer-floor bounding box'})
    return result
