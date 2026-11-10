"""Read-only CPU audit of the two fixed-grid bucket/fixed-base blockers.

No physics steps, IK, GPU, collision-mask changes, guard overrides or file writes.
MuJoCo mesh colliders are convex hulls. Independent half-space intersection
certificates distinguish their actual overlap from an enclosing-box artefact.
The bottom cylinder is queried natively; the independent overlap certificate
uses an *inscribed* prism, then checks its witness against the exact cylinder.
Failure to obtain a certificate is never interpreted as proven separation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull

VERSION = 'sircus-fixed35-six-fixed-base-geometry-v1'
DEFAULT_CASES = (
    ('fixed35-dev-grid-03-00', [.1, .025000000000000022]),
    ('fixed35-dev-grid-03-06', [1.24, .025000000000000022]),
    ('fixed35-dev-grid-02-06', [1.24, -.2]),
)
TOL = 1e-8


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def polyhedron(points):
    points = np.asarray(points, float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError('Finite three-dimensional convex collider vertices required')
    hull = ConvexHull(points)
    # Compiled mesh vertices, rather than rendered/visual STL triangles.
    planes = np.unique(np.round(hull.equations, 13), axis=0)
    normals = np.linalg.norm(planes[:, :3], axis=1)
    planes /= normals[:, None]
    return dict(kind='convex_polyhedron', points=points, planes=planes,
                center=points.mean(0), normals=planes[:, :3])


def cylinder(center, rotation, radius, half_height, segments=96):
    center, rotation = np.asarray(center, float), np.asarray(rotation, float)
    if (center.shape != (3,) or rotation.shape != (3, 3) or not np.isfinite(center).all()
            or not np.isfinite(rotation).all() or radius <= 0 or half_height <= 0 or segments < 8
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-9)
            or not np.isclose(np.linalg.det(rotation), 1., atol=1e-9)):
        raise ValueError('A finite rigid cylinder with positive dimensions is required')
    theta = (np.arange(segments) + .5) * 2 * np.pi / segments
    # Faces tangent to the inscribed polygon, so all its points lie in cylinder.
    normals = np.column_stack((np.cos(theta), np.sin(theta), np.zeros(segments))) @ rotation.T
    normals = np.r_[normals, rotation[:, 2][None], -rotation[:, 2][None]]
    limits = np.r_[np.full(segments, radius*np.cos(np.pi/segments)), half_height, half_height]
    planes = np.column_stack((normals, -normals @ center-limits))
    return dict(kind='cylinder', center=center, rotation=rotation, radius=float(radius),
                half_height=float(half_height), planes=planes, normals=normals,
                certificate_segments=segments,
                inscribed_radial_error_m=float(radius*(1-np.cos(np.pi/segments))))


def interior_slack(shape, point):
    """Radius of a sphere centred here contained in this exact convex collider."""
    point = np.asarray(point, float)
    if shape['kind'] == 'convex_polyhedron':
        return float(-(shape['planes'][:, :3] @ point+shape['planes'][:, 3]).max())
    local = shape['rotation'].T @ (point-shape['center'])
    return float(min(shape['radius']-np.linalg.norm(local[:2]), shape['half_height']-abs(local[2])))


def support(shape, axes):
    axes = np.asarray(axes, float)
    if shape['kind'] == 'convex_polyhedron':
        return (shape['points'] @ axes.T).max(0)
    local = axes @ shape['rotation']
    return axes @ shape['center'] + shape['radius']*np.linalg.norm(local[:, :2], axis=1)+shape['half_height']*abs(local[:, 2])


def separation_bound(first, second):
    """Conservative positive gap using exact collider support, not proxy AABBs."""
    delta = second['center']-first['center']
    axes = np.r_[first['normals'], second['normals'], delta[None]]
    axes = axes[np.linalg.norm(axes, axis=1) > 1e-12]
    axes /= np.linalg.norm(axes, axis=1)[:, None]
    axes = np.unique(np.round(axes, 12), axis=0)
    axes /= np.linalg.norm(axes, axis=1)[:, None]
    gaps = np.maximum(-support(first, -axes)-support(second, axes),
                      -support(second, -axes)-support(first, axes))
    i = int(np.argmax(gaps))
    return dict(lower_bound_m=max(0., float(gaps[i])), axis=axes[i].tolist(),
                meaning='A positive separating-axis gap is a lower bound, not exact Euclidean clearance')


def overlap_certificate(first, second):
    """Find a common interior sphere; positive verified slack proves overlap.

    The LP's objective value alone is not evidence. Recheck the centre against
    each exact collider. A zero/negative radius is inconclusive, not clearance.
    """
    planes = np.r_[first['planes'], second['planes']]
    result = linprog([0., 0., 0., -1.],
        A_ub=np.column_stack((planes[:, :3], np.ones(len(planes)))),
        b_ub=-planes[:, 3], bounds=[(None, None)]*4, method='highs')
    if not result.success:
        return dict(verified_overlap=False, solver_status=int(result.status), common_interior_radius_m=0.)
    point = result.x[:3]
    slacks = [interior_slack(s, point) for s in (first, second)]
    # Subtract numerical tolerance before making a strict geometric assertion.
    certified = max(0., min(slacks)-TOL)
    return dict(verified_overlap=bool(certified > TOL), solver_status=int(result.status),
                witness_world_m=point.tolist(), common_interior_radius_m=certified,
                exact_collider_slacks_m=slacks,
                meaning='Common interior sphere proves geometric overlap; radius is not a minimum translation distance')


def compiled_shape(m, d, index):
    typ = int(m.geom_type[index]);rotation = d.geom_xmat[index].reshape(3, 3)
    if typ == 7:  # mjGEOM_MESH: native collision is its compiled convex hull.
        mesh = int(m.geom_dataid[index]);start = int(m.mesh_vertadr[mesh])
        points = np.asarray(m.mesh_vert[start:start+int(m.mesh_vertnum[mesh])], float)
        return polyhedron(points @ rotation.T+d.geom_xpos[index])
    if typ == 5:  # mjGEOM_CYLINDER
        return cylinder(d.geom_xpos[index], rotation, *m.geom_size[index, :2])
    raise ValueError(f'Unsupported fixed-base/bucket collision type {typ}; no proxy fallback')


def signed_query(mj, m, d, i, j, first, second):
    witness = np.zeros(6);reverse = np.zeros(6);limit = .5
    distance = float(mj.mj_geomDistance(m, d, i, j, limit, witness))
    back = float(mj.mj_geomDistance(m, d, j, i, limit, reverse))
    saturated = distance >= limit-1e-10 or back >= limit-1e-10
    norm_error = max(abs(np.linalg.norm(witness[:3]-witness[3:])-abs(distance)),
                     abs(np.linalg.norm(reverse[:3]-reverse[3:])-abs(back)))
    surface_error = max(abs(interior_slack(first, witness[:3])), abs(interior_slack(second, witness[3:])),
                        abs(interior_slack(second, reverse[:3])), abs(interior_slack(first, reverse[3:])))
    valid = (not saturated and np.isfinite([distance, back, norm_error, surface_error]).all()
             and abs(distance-back) <= 1e-5 and norm_error <= 1e-5 and surface_error <= 1e-5)
    return dict(distance_m=distance, reverse_distance_m=back, witness_world_m=witness.tolist(),
                reverse_witness_world_m=reverse.tolist(), query_limit_m=limit, truncated=bool(saturated),
                witness_norm_error_m=float(norm_error), witness_surface_error_m=float(surface_error),
                validated_signed_query=bool(valid))


def fixed_ancestry(m, body):
    path=[]
    while body:
        path.append(int(body))
        if int(m.body_dofnum[body]):return False, path
        body = int(m.body_parentid[body])
    return True, path


def audit_geometry(scene, assets, cases=None, clearance_m=.005):
    """Build the unchanged native model, audit fixed pairs, return JSON data.

    Callable by producer and independent verifier. Does not import mjlab,
    initialize Torch/CUDA, invoke a simulation environment, or write anything.
    """
    import mujoco as mj
    from .model import build_scene
    if clearance_m != .005:raise ValueError('The existing 5 mm guard remains unchanged')
    assets = Path(assets)
    xml = build_scene(scene, assets, solref=(.004, .15), contact_friction=.35)
    m = mj.MjModel.from_xml_string(xml);d = mj.MjData(m)
    names = [mj.mj_id2name(m, mj.mjtObj.mjOBJ_GEOM, i) or str(i) for i in range(m.ngeom)]
    bases = [i for i,n in enumerate(names) if n.startswith(('Arm10/', 'Arm14/')) and fixed_ancestry(m, int(m.geom_bodyid[i]))[0]]
    bucket = [i for i,n in enumerate(names) if n.startswith('bucket/')]
    if [names[i] for i in bases] != ['Arm10/link_base/collision/0', 'Arm14/link_base/collision/0'] or len(bucket) != 65:
        raise ValueError('Expected two exact fixed-base colliders and unchanged64-wall/bottom bucket')
    hashes={}
    for role in ('dropper', 'board_arm'):
        urdf = assets/(role+'.urdf');hashes[role+'.urdf'] = sha(urdf)
    for mesh in ET.fromstring(xml).findall('./asset/mesh'):
        if mesh.get('file'):
            hashes['collision_mesh:'+mesh.get('name')] = sha(mesh.get('file'))
    reports=[]
    for case in (DEFAULT_CASES if cases is None else cases):
        identity,xy = (case['context_id'],case['bucket_xy_m']) if isinstance(case,dict) else case
        if len(xy)!=2 or not np.isfinite(xy).all():raise ValueError('Finite fixed bucket XY required')
        d.qpos[:] = m.qpos0;d.qvel[:] = 0
        mocap = int(m.body_mocapid[m.body('bucket').id])
        d.mocap_pos[mocap] = [*xy, 0];d.mocap_quat[mocap] = [1, 0, 0, 0]
        mj.mj_forward(m, d)  # Kinematics/collision queries only: no mj_step.
        shapes = {i:compiled_shape(m,d,i) for i in bases+bucket}
        geometry=[]
        for i,s in shapes.items():
            g=dict(name=names[i],type=int(m.geom_type[i]),position_world_m=d.geom_xpos[i].tolist(),
                   rotation_world=d.geom_xmat[i].reshape(3,3).tolist(),size=m.geom_size[i].tolist(),
                   contype=int(m.geom_contype[i]),conaffinity=int(m.geom_conaffinity[i]),
                   fixed_base=i in bases,body_has_movable_ancestor=not fixed_ancestry(m,int(m.geom_bodyid[i]))[0])
            if s['kind']=='convex_polyhedron':
                g.update(compiled_vertices_sha256=digest(s['points'].tolist()),
                         convex_halfspaces_sha256=digest(s['planes'].tolist()),vertices=len(s['points']))
            else:g.update(certificate_segments=s['certificate_segments'],inscribed_radial_error_m=s['inscribed_radial_error_m'])
            geometry.append(g)
        pairs=[]
        for i in bases:
            for j in bucket:
                a,b=shapes[i],shapes[j]
                bound=separation_bound(a,b)
                query=signed_query(mj,m,d,i,j,a,b)
                certificate=overlap_certificate(a,b) if bound['lower_bound_m']<clearance_m else None
                contacts=[dict(distance_m=float(c.dist),position_world_m=c.pos.tolist())
                    for c in d.contact[:d.ncon] if set(map(int,c.geom))=={i,j}]
                pairs.append(dict(pair=[names[i],names[j]],native_signed_query=query,separation=bound,
                                  overlap_certificate=certificate,active_native_contacts=contacts))
        overlaps=[p for p in pairs if p['overlap_certificate'] and p['overlap_certificate']['verified_overlap']]
        validated=[p for p in pairs if p['native_signed_query']['validated_signed_query']]
        minimum=min(validated,key=lambda p:p['native_signed_query']['distance_m']) if validated else None
        proven_clear=all(p['separation']['lower_bound_m']>=clearance_m for p in pairs)
        reports.append(dict(context_id=identity,bucket_xy_m=list(xy),pairs=pairs,geometry=geometry,
            classification='actual_fixed_base_overlap' if overlaps else 'proven_clearance' if proven_clear else 'unresolved',
            certified_overlap_pairs=len(overlaps),minimum_validated_signed_distance_m=minimum['native_signed_query']['distance_m'] if minimum else None,
            minimum_validated_pair=minimum['pair'] if minimum else None,
            all_pairs_prove_5mm_clearance=proven_clear,
            active_fixed_base_bucket_contacts=sum(len(p['active_native_contacts']) for p in pairs)))
    return dict(version=VERSION,scene_sha256=digest(scene),model_xml_sha256=hashlib.sha256(xml.encode()).hexdigest(),
        source_sha256={Path(__file__).name:sha(__file__), 'model.py':sha(Path(__file__).with_name('model.py'))},
        assets_sha256=hashes,mujoco_version=mj.__version__,clearance_m=clearance_m,guard_changed=False,
        physics_steps=0,gpu_used=False,cases=reports,
        scope='Exact compiled convex collision shapes; geometric query and common-interior proof. '
              'Native contact activation is reported separately: an empty solver contact list does not negate geometric overlap. '
              'Per-feature signed penetration is not the translation required to clear the entire assembly. '
              'No physical calibration, arm-pose change, guard override, native drop or policy improvement.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract',type=Path,required=True)
    parser.add_argument('--assets',type=Path,required=True)
    args=parser.parse_args()
    scene=json.loads(args.contract.read_text())['scene']
    print(json.dumps(audit_geometry(scene,args.assets),sort_keys=True,allow_nan=False))


if __name__=='__main__':main()
