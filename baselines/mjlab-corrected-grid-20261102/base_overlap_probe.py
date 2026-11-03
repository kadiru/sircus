"""Supplementary CPU collision-hull intersection proof; no source/study mutation."""
import hashlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull

from sircus.mjlab.model import build_scene


study=Path(sys.argv[1]);assets=Path(sys.argv[2]);output=Path(sys.argv[3])
spec=json.loads((study/'contract.json').read_text());home=json.loads((study/'common-home.json').read_text())
xml=build_scene(spec['scene'],assets,solref=(.004,.15),contact_friction=.35)
m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m)
for role in (10,14):
 for i,q in enumerate(home[f'q{role}'],1):d.qpos[m.jnt_qposadr[m.joint(f'Arm{role}/joint{i}').id]]=q
bucket=m.body('bucket').id;mocap=int(m.body_mocapid[bucket])

def points(geom):
 assert int(m.geom_type[geom])==int(mujoco.mjtGeom.mjGEOM_MESH)
 mesh=int(m.geom_dataid[geom]);start=int(m.mesh_vertadr[mesh]);count=int(m.mesh_vertnum[mesh])
 return np.asarray(m.mesh_vert[start:start+count],float)@d.geom_xmat[geom].reshape(3,3).T+d.geom_xpos[geom]

def intersection(a,b):
 equations=np.r_[ConvexHull(a).equations,ConvexHull(b).equations]
 normals=equations[:,:3];offsets=equations[:,3]
 norms=np.linalg.norm(normals,axis=1)
 # Chebyshev ball: its center lies in both convex collision hulls and every
 # hull facet has at least radius clearance. Positive radius proves volumetric
 # intersection; this is not a negative-distance inference from a zero bound.
 result=linprog([0,0,0,-1],A_ub=np.c_[normals,norms],b_ub=-offsets,
  bounds=[(None,None)]*3+[(0,None)],method='highs')
 if not result.success:return dict(positive_volume_overlap=False,linear_program_status=int(result.status))
 center=result.x[:3];radius=float(result.x[3]);slack=-(normals@center+offsets)/norms
 return dict(positive_volume_overlap=radius>1e-6 and float(slack.min())>1e-6,
  inscribed_intersection_radius_m=radius,witness_center_m=center.tolist(),
  independently_checked_minimum_facet_clearance_m=float(slack.min()),
  maximum_constraint_violation_m=float(np.maximum(radius-slack,0).max()))

cases=[]
for cell_id,role in [('dev-grid-03-00',10),('dev-grid-03-06',14)]:
 cell=next(x for x in spec['grid_layouts'] if x['id']==cell_id)
 d.mocap_pos[mocap]=[*cell['bucket_xy_m'],0];d.mocap_quat[mocap]=[1,0,0,0]
 mujoco.mj_forward(m,d)
 base=m.geom(f'Arm{role}/link_base/collision/0').id;body=int(m.geom_bodyid[base]);ancestors=[];p=body
 while p:
  ancestors.append(dict(body=m.body(p).name,joints=int(m.body_jntnum[p])));p=int(m.body_parentid[p])
 tested=[]
 for index in range(spec['scene']['bucket']['segments']):
  wall=m.geom(f'bucket/wall{index:02d}').id
  result=intersection(points(base),points(wall))
  if result['positive_volume_overlap']:tested.append(dict(pair=[m.geom(base).name,m.geom(wall).name],**result))
 cases.append(dict(cell_id=cell_id,bucket_xy_m=cell['bucket_xy_m'],fixed_base_ancestors=ancestors,
  base_is_fixed_to_world=all(x['joints']==0 for x in ancestors),tested_bucket_wall_segments=spec['scene']['bucket']['segments'],
  positive_intersections=tested,strongest=max(tested,key=lambda x:x['inscribed_intersection_radius_m']) if tested else None,
  scope='Convex hulls of actual compiled MuJoCo mesh collision geometries; not a real-hardware claim'))

result=dict(version='sircus-cpu-fixed-base-convex-intersection-probe-v1',mujoco_version=mujoco.__version__,
 model_xml_sha256=hashlib.sha256(xml.encode()).hexdigest(),contract_sha256=hashlib.sha256((study/'contract.json').read_bytes()).hexdigest(),
 probe_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),cases=cases,
 limitations=['Supplementary evidence only; running source, masks, records and original classifications were not changed.',
 'Positive-radius hull intersection proves overlap of these modeled convex collision solids, not general unreachability elsewhere.',
 'The witness radius is not a signed penetration-depth estimate.'])
with output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
print(json.dumps(dict(output=str(output),cases=[dict(cell_id=x['cell_id'],fixed=x['base_is_fixed_to_world'],
 intersections=len(x['positive_intersections']),strongest=x['strongest']) for x in cases])))
