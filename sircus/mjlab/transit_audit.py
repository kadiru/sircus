"""Finite, CPU-only geometric transit planning for the two simulated UF850s.

This opt-in planner does not step physics or command a robot. Historical native
shot endpoint initialization, collision filters and scorers remain unchanged.
"""

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from scipy.spatial import ConvexHull
from itertools import product

VERSION = "sircus-serialized-low-board-high-dropper-transit-v1"


def joint_samples(first, second, max_step_rad=np.deg2rad(.5)):
    a, b = np.asarray(first, float), np.asarray(second, float)
    if a.shape != (12,) or b.shape != (12,) or not np.isfinite([a, b]).all():
        raise ValueError("Two finite 12-joint configurations required")
    if not 0 < max_step_rad <= .05:
        raise ValueError("Finite sampling step in (0,.05] rad required")
    n = max(1, int(np.ceil(np.max(np.abs(b - a)) / max_step_rad)))
    return np.linspace(a, b, n + 1)


def pose_samples(first, second, translation_step_m=.025, angle_step_deg=5.):
    a, b = np.asarray(first, float), np.asarray(second, float)
    n = max(1, int(np.ceil(np.linalg.norm(b[:3, 3] - a[:3, 3]) / translation_step_m)),
        int(np.ceil(np.degrees(Rotation.from_matrix(b[:3, :3] @ a[:3, :3].T).magnitude()) / angle_step_deg)))
    t = np.linspace(0, 1, n + 1)
    rotations = Slerp([0, 1], Rotation.from_matrix([a[:3, :3], b[:3, :3]]))(t).as_matrix()
    out = np.broadcast_to(np.eye(4), (n + 1, 4, 4)).copy()
    out[:, :3, :3] = rotations
    out[:, :3, 3] = a[:3, 3] + t[:, None] * (b[:3, 3] - a[:3, 3])
    return out[1:]


class GeometricGate:
    """Native convex collision distances with 5mm clearance and strict limits.

    A separate validation model retains only mechanical adjacent/welded link
    exclusions. Same-arm non-adjacent pairs and a carried-ball proxy are checked.
    It never changes the historical dynamics model. Mounted base/table interfaces
    and ball/gripping-jaw contact are intentional exclusions.
    """
    def __init__(self, scene, assets, bucket_xy, clearance_m=.005):
        import mujoco as mj
        from .model import build_scene
        from .stable_geometry import NativeEndpointGate
        self.native = NativeEndpointGate(scene, assets)
        self.mj, self.m, self.d = mj, self.native.model, self.native.data
        self.clearance = clearance_m
        xml = ET.fromstring(build_scene(scene, Path(assets), solref=(.004,.15), contact_friction=.35))
        contact = xml.find("contact")
        for child in list(contact): contact.remove(child)
        m = self.m
        names = [mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, i) or "world" for i in range(m.nbody)]
        self.structural_exclusions = []
        def chain_distance(a,b):
            def ancestors(i):
                out={i:0}; cost=0
                while i:
                    cost += int(m.body_jntnum[i] > 0)
                    i=int(m.body_parentid[i]);out[i]=cost
                return out
            aa,bb=ancestors(a),ancestors(b)
            return min(aa[x]+bb[x] for x in aa.keys() & bb.keys())
        for i in range(1,m.nbody):
            for j in range(i+1,m.nbody):
                if names[i].startswith(("Arm10/","Arm14/")) and names[i].split('/')[0]==names[j].split('/')[0] and chain_distance(i,j)<=1:
                    ET.SubElement(contact,"exclude",body1=names[i],body2=names[j])
                    self.structural_exclusions.append([names[i],names[j]])
        for role in ("Arm10","Arm14"):
            ET.SubElement(contact,"exclude",body1=f"{role}/link_base",body2="world")
        self.carried_exclusions=[]
        for name in names:
            if name.startswith("Arm10/") and ("finger" in name or name.endswith("/link6")):
                ET.SubElement(contact,"exclude",body1="ball",body2=name)
                self.carried_exclusions.append(["ball",name])
        for geom in xml.findall(".//geom"): geom.set("margin",str(clearance_m))
        validation_xml=ET.tostring(xml,encoding="unicode")
        self.validation_model_sha256=hashlib.sha256(validation_xml.encode()).hexdigest()
        self.m=m=mj.MjModel.from_xml_string(validation_xml)
        self.d=mj.MjData(m)
        m = self.m
        self.names = [mj.mj_id2name(m, mj.mjtObj.mjOBJ_GEOM, i) or str(i) for i in range(m.ngeom)]
        self.body_names = [mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, int(i)) or "world" for i in m.geom_bodyid]
        self.robot = [i for i, name in enumerate(self.names) if name.startswith(("Arm10/", "Arm14/"))]
        self.moving_geoms=self.robot+[i for i,name in enumerate(self.names) if name.startswith("ball/")]
        self.pairs, self.mount_exclusions = [], []
        for i in self.moving_geoms:
            for j in range(m.ngeom):
                if i >= j and j in self.moving_geoms:
                    continue
                if i == j:
                    continue
                if j in self.robot and self.names[i].split('/')[0]==self.names[j].split('/')[0] and chain_distance(int(m.geom_bodyid[i]),int(m.geom_bodyid[j]))<=1:
                    continue
                if any(set(pair)=={self.body_names[i],self.body_names[j]} for pair in self.carried_exclusions):
                    continue
                # Base/table attachment is intentional; no movable-link/table exclusions.
                if self.names[j] == "table" and self.body_names[i].endswith(("/link_base", "/base_link")):
                    self.mount_exclusions.append([self.names[i], self.names[j]])
                    continue
                self.pairs.append((i, j))
        self.bucket_xy = np.asarray(bucket_xy, float)
        self.radius=scene["ball"]["radius_m"]
        self.local_points,self.local_axes=[],[]
        for i in range(m.ngeom):
            if int(m.geom_type[i])==7:
                mesh=int(m.geom_dataid[i]);start=int(m.mesh_vertadr[mesh]);count=int(m.mesh_vertnum[mesh])
                points=np.asarray(m.mesh_vert[start:start+count],float)
            else:
                # Conservative enclosing box for analytic primitives; exact box.
                points=np.asarray(list(product((-1.,1.),repeat=3)))*m.geom_aabb[i,3:]+m.geom_aabb[i,:3]
            self.local_points.append(points)
            axes=ConvexHull(points).equations[:,:3]
            for axis in axes:
                if axis[np.flatnonzero(np.abs(axis)>1e-8)[0]]<0:axis*=-1
            self.local_axes.append(np.unique(np.round(axes,8),axis=0))

    def state(self, q):
        m, d, n = self.m, self.d, self.native
        d.qpos[:] = m.qpos0
        d.qpos[n.qi10], d.qpos[n.qi14] = q[:6], q[6:]
        d.qpos[n.fingers] = .025
        d.qpos[n.ball:n.ball + 3] = [0, 0, 2]
        d.qpos[n.ball + 3:n.ball + 7] = [1, 0, 0, 0]
        d.qvel[:] = 0
        d.mocap_pos[n.mocap] = [*self.bucket_xy, 0]
        d.mocap_quat[n.mocap] = [1, 0, 0, 0]
        self.mj.mj_forward(m, d)
        # Follow the downward-facing flange as a grasped ball proxy. No dynamics
        # or fictitious free fall during transit; jaw/grasp-assembly contacts excluded.
        d.qpos[n.ball:n.ball+3]=d.xpos[n.flange]+d.xmat[n.flange].reshape(3,3) @ [0,0,.095]
        self.mj.mj_forward(m,d)

    def aabb(self,i):
        points=self.world_points[i]
        return points.min(0),points.max(0)

    def separation_lower_bound(self,i,j):
        """A positive separating-axis gap is a conservative Euclidean bound.

        Missing edge-cross axes can only reject a feasible path, never produce
        a false clearance pass. Primitives use enclosing boxes.
        """
        a,b=self.world_points[i],self.world_points[j]
        axes=np.r_[self.local_axes[i] @ self.d.geom_xmat[i].reshape(3,3).T,
            self.local_axes[j] @ self.d.geom_xmat[j].reshape(3,3).T]
        axes/=np.linalg.norm(axes,axis=1)[:,None]
        aa,bb=a@axes.T,b@axes.T
        return max(0.,float(np.maximum(aa.min(0)-bb.max(0),bb.min(0)-aa.max(0)).max()))

    def check(self, q, *, corridor=None):
        q = np.asarray(q, float)
        lower = np.r_[self.native.limits10[:, 0], self.native.limits14[:, 0]]
        upper = np.r_[self.native.limits10[:, 1], self.native.limits14[:, 1]]
        if q.shape != (12,) or not np.isfinite(q).all() or (q < lower).any() or (q > upper).any():
            return dict(passed=False, reason="joint_limits_or_nonfinite")
        self.state(q)
        minimum, pair = 1., None
        unreliable=0
        self.world_points=[p @ self.d.geom_xmat[i].reshape(3,3).T+self.d.geom_xpos[i] for i,p in enumerate(self.local_points)]
        boxes=[self.aabb(i) for i in range(self.m.ngeom)]
        unresolved=[]
        for i, j in self.pairs:
            gap=np.maximum(np.maximum(boxes[i][0]-boxes[j][1],boxes[j][0]-boxes[i][1]),0)
            conservative=float(np.linalg.norm(gap))
            distance=conservative
            if conservative<self.clearance:
                witness=np.zeros(6)
                distance = float(self.mj.mj_geomDistance(self.m, self.d, i, j, .25, witness))
                # MuJoCo3.11 mesh/box sometimes returns zero with widely separated
                # witnesses. Do not mislabel this as measured collision or clearance.
                if distance<.249 and abs(np.linalg.norm(witness[:3]-witness[3:])-abs(distance))>1e-5:
                    unreliable+=1;distance=conservative
                else: distance=max(distance,conservative)
                if distance<self.clearance:
                    distance=max(distance,self.separation_lower_bound(i,j))
            if distance<self.clearance:
                unresolved.append(dict(pair=[self.names[i],self.names[j]],lower_bound_m=distance))
            if distance < minimum:
                minimum, pair = distance, [self.names[i], self.names[j]]
        bad=[]
        for c in self.d.contact[:self.d.ncon]:
            a,b=map(int,c.geom)
            if not (a in self.moving_geoms or b in self.moving_geoms):continue
            # Only mechanically attached gripper/ball interactions are excluded.
            bodies={self.body_names[a],self.body_names[b]}
            if any(set(pair)==bodies for pair in self.carried_exclusions):continue
            if c.dist < self.clearance:
                bad.append(dict(pair=[self.names[a],self.names[b]],distance_m=float(c.dist)))
        board = self.d.xpos[self.native.board].copy()
        rot = self.d.xmat[self.native.board].reshape(3, 3)
        # Full box extent, not merely center height; landscape rotation already in model.
        size = np.asarray(self.m.geom_size[self.m.geom("Arm14/impact_board/collision/0").id])
        board_top = float(board[2] + np.abs(rot[2]) @ size)
        drop_h = float(self.d.xpos[self.native.flange, 2])
        height_ok = not corridor or ((corridor.get("board_top_max_m") is None or board_top <= corridor["board_top_max_m"])
            and (corridor.get("dropper_flange_min_m") is None or drop_h >= corridor["dropper_flange_min_m"]))
        return dict(passed=not bad and not unresolved and height_ok,
            reason="clearance" if bad else ("unproven_clearance" if unresolved else ("corridor" if not height_ok else None)),
            minimum_clearance_m=minimum, closest_pair=pair, board_top_m=board_top, dropper_flange_z_m=drop_h,
            margin_contacts=bad, inconsistent_distance_queries=unreliable, unresolved_clearance_pairs=unresolved)

    def frame(self, q, phase):
        self.state(q)
        m, d = self.m, self.d
        def links(role):
            return [d.xpos[m.body(f"{role}/link{i}").id].tolist() for i in range(1, 7)]
        return dict(phase=phase, q10=q[:6].tolist(), q14=q[6:].tolist(),
            ball_center_m=d.qpos[self.native.ball:self.native.ball+3].tolist(),
            link_positions10=[[0, 0, 0], *links("Arm10")],
            link_positions14=[[float(m.body_pos[m.body("Arm14/link_base").id][0]), 0, 0], *links("Arm14")],
            board_center_m=d.xpos[self.native.board].tolist(),
            board_quaternion_wxyz=d.xquat[self.native.board].tolist())


def audit_segment(gate, first, second, *, corridor=None, max_step_rad=np.deg2rad(.5)):
    samples = joint_samples(first, second, max_step_rad)
    minimum, closest = 1., None
    heights=[]
    for index, q in enumerate(samples):
        result = gate.check(q, corridor=corridor)
        if "board_top_m" in result:heights.append([result["board_top_m"],result["dropper_flange_z_m"]])
        if result.get("minimum_clearance_m", 1.) < minimum:
            minimum, closest = result["minimum_clearance_m"], result.get("closest_pair")
        if not result["passed"]:
            return dict(passed=False, sample=index, total_samples=len(samples), **{k:v for k,v in result.items() if k != "passed"})
    return dict(passed=True, total_samples=len(samples), minimum_clearance_m=minimum, closest_pair=closest,
        maximum_actual_joint_sample_step_rad=float(np.abs(np.diff(samples,axis=0)).max()),
        board_top_range_m=[float(np.min(np.asarray(heights)[:,0])),float(np.max(np.asarray(heights)[:,0]))] if heights else None,
        dropper_flange_z_range_m=[float(np.min(np.asarray(heights)[:,1])),float(np.max(np.asarray(heights)[:,1]))] if heights else None)


def plan_cycle(scene, assets, row, *, low_board_m=.24, high_dropper_m=.95, clearance_m=.005,
    lift_mode="vertical", board_shift=(.15,.04)):
    """Round-trip reconfiguration, starting/ending at one exact saved shot pair.

    First lift and park dropper, lower/transfer board while dropper is parked,
    restore board, then transfer dropper high and descend. Each phase moves one
    arm only; no live controller or native actuation is performed.
    """
    from .stable_ik import StableShotIK
    gate = GeometricGate(scene, assets, row["bucket_xy_m"], clearance_m)
    ik = StableShotIK(scene, assets)
    q = np.r_[row["q10"], row["q14"]]
    initial = gate.check(q)
    report = dict(source_id=row["id"], bucket_xy_m=row["bucket_xy_m"], values=row["values"],
        q10=row["q10"], q14=row["q14"], initial=initial, phases=[], passed=False,
        requested_low_board_center_m=low_board_m, requested_high_dropper_flange_m=high_dropper_m,
        lift_mode=lift_mode, board_retraction_xy_m=list(board_shift), low_transfer_xy_m=[.05,.02],
        clearance_m=clearance_m, maximum_joint_sample_step_rad=float(np.deg2rad(.5)),
        cartesian_waypoint_translation_step_m=.025, cartesian_waypoint_rotation_step_deg=5.,
        mount_exclusions=gate.mount_exclusions,
        structural_exclusions=gate.structural_exclusions, carried_ball_exclusions=gate.carried_exclusions,
        validation_model_sha256=gate.validation_model_sha256, mujoco_version=gate.mj.mj_versionString(),
        limitations=["Geometric samples only; no actuator tracking, velocity, acceleration or torque certification",
            "Same-arm adjacent/welded mechanical pairs excluded; all non-adjacent pairs enabled in separate validation model",
            "Carried-ball sphere follows flange; intentional jaw/grasp-assembly contacts excluded; grasp stability untested",
            "Finite sampled sweep is not a formal continuous collision proof"])
    if not initial["passed"]:
        return report, []
    def flange(which):
        gate.state(q)
        role = "Arm10" if which == 10 else "Arm14"
        body = gate.m.body(f"{role}/link6").id
        result = np.eye(4); result[:3, :3] = gate.d.xmat[body].reshape(3, 3)
        result[:3, 3] = gate.d.xpos[body]
        return result
    original10, original14 = flange(10), flange(14)
    lifted = original10.copy(); lifted[2, 3] = max(min(high_dropper_m,.88), original10[2, 3] + .06)
    if lift_mode=="retract":
        lifted[:2,3]=[min(.45,original10[0,3]),original10[1,3]*.75]
        lifted[2,3]=max(.86,original10[2,3]+.04)
    elif lift_mode!="vertical":raise ValueError("Unknown finite lift mode")
    parked = lifted.copy(); parked[:3, 3] = [.25, -.10,max(high_dropper_m,lifted[2,3])]
    # Retract towards Arm14's base while descending; vertical descent at the
    # fully extended shooting XY can itself be unreachable.
    low = original14.copy(); low[2, 3] -= row["values"][2] - low_board_m
    low[:2,3] += board_shift
    away = low.copy(); away[:2, 3] += [.05, .02]
    phases = [
        ("dropper_lift", 10, lifted, {}),
        ("dropper_high_park", 10, parked, {"dropper_flange_min_m": lifted[2, 3] - .01}),
        ("board_lower_retract", 14, low, {"dropper_flange_min_m": parked[2, 3] - .01}),
        ("board_low_transfer", 14, away, {"dropper_flange_min_m": parked[2, 3] - .01, "board_top_max_m": low_board_m + .13}),
        ("board_low_return", 14, low, {"dropper_flange_min_m": parked[2, 3] - .01, "board_top_max_m": low_board_m + .13}),
        ("board_restore", 14, original14, {"dropper_flange_min_m": parked[2, 3] - .01}),
        ("dropper_high_transfer", 10, lifted, {"dropper_flange_min_m": lifted[2, 3] - .01}),
        ("dropper_descend", 10, original10, {})]
    frames = [gate.frame(q, "initial")]
    for label, role, target_world, corridor in phases:
        phase = dict(name=label, moving_arm=role, target_flange_world=target_world.tolist(), corridor=corridor,
            segments=[], passed=False)
        report["phases"].append(phase)
        current_world = flange(role)
        for world in pose_samples(current_world, target_world):
            target = world.copy()
            arm = ik.dropper if role == 10 else ik.board
            offset = 0 if role == 10 else 6
            if role == 14:
                base = np.eye(4); base[:3, :3] = Rotation.from_euler("z", scene["arms"]["board_base_yaw_rad"]).as_matrix()
                base[0, 3] = scene["arms"]["base_spacing_m"]
                target = np.linalg.inv(base) @ world
            pool = arm.solve_candidates(target[None], known=q[offset:offset+6], starts=4, iterations=100)
            candidates = pool["q"][0].cpu().numpy()
            valid = pool["valid"][0].cpu().numpy()
            order = sorted(np.flatnonzero(valid), key=lambda j: np.linalg.norm(candidates[j] - q[offset:offset+6]))
            accepted = None
            for branch in order:
                proposed = q.copy(); proposed[offset:offset+6] = candidates[branch]
                if np.max(np.abs(proposed - q)) > .5:
                    continue
                checked = audit_segment(gate, q, proposed, corridor=corridor)
                if checked["passed"]:
                    accepted = proposed, checked, int(branch)
                    break
                phase["last_rejection"] = checked
            if accepted is None:
                phase["reason"] = "no_continuous_clearance_valid_ik_branch"
                phase["valid_ik_branches"] = int(valid.sum())
                return report, frames
            proposed, checked, branch = accepted
            phase["segments"].append(dict(**checked, branch=branch, q10=proposed[:6].tolist(), q14=proposed[6:].tolist()))
            q = proposed
            frames.append(gate.frame(q, label))
        phase["passed"] = True
    exact=np.r_[row["q10"],row["q14"]]
    restore=audit_segment(gate,q,exact)
    report["exact_nominal_restore"]=restore
    if not restore["passed"]:
        return report,frames
    q=exact
    frames.append(gate.frame(q,"exact_nominal_restore"))
    # Endpoint equality is checked geometrically; tiny IK polishing differences disclosed.
    report["returned_joint_max_error_rad"] = float(np.max(np.abs(q - np.r_[row["q10"], row["q14"]])))
    report["passed"] = True
    report["checked_joint_configurations"] = restore["total_samples"]+sum(s["total_samples"] for p in report["phases"] for s in p["segments"])
    report["minimum_clearance_m"] = min(restore["minimum_clearance_m"],min(s["minimum_clearance_m"] for p in report["phases"] for s in p["segments"]))
    indices = np.unique(np.linspace(0, len(frames)-1, min(100, len(frames))).astype(int))
    return report, [frames[i] for i in indices]


def plan_bounded_cycle(scene, assets, row, *, low_board_m=.24, high_dropper_m=.95):
    """Small declared waypoint search; every failed attempt remains evidence."""
    attempts=[]
    variants=[("vertical",low_board_m,(.15,.04)),
        ("retract",low_board_m,(.15,.04)),
        ("retract",.28,(.15,.04)),("retract",.32,(.15,.04)),
        ("retract",.28,(.10,.02)),("retract",.32,(.10,.02))]
    for mode,height,shift in variants:
        report,frames=plan_cycle(scene,assets,row,low_board_m=height,
            high_dropper_m=high_dropper_m,lift_mode=mode,board_shift=shift)
        attempts.append(report)
        if report["passed"] or not report["initial"]["passed"]:
            return report,frames,attempts
    return report,frames,attempts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", type=Path, required=True)
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--source-id", action="append", default=[])
    p.add_argument("--low-board", type=float, default=.24)
    p.add_argument("--high-dropper", type=float, default=.95)
    p.add_argument("--bounded",action="store_true",help="Try six declared low/high waypoint variants, retaining rejections")
    a = p.parse_args()
    if a.output.exists():
        raise ValueError("Output must be new; preserve previous evidence")
    a.output.mkdir(parents=True)
    contract = json.loads((a.study / "contract.json").read_text())
    wanted = a.source_id or ["stable-shot-0004711"]
    records = [r for r in map(json.loads, (a.study / "shots.jsonl").open()) if r["id"] in wanted]
    if len(records) != len(set(wanted)):
        raise ValueError("Every source ID must resolve exactly once")
    reports, previews = [], []
    for row in records:
        if a.bounded:
            report,frames,attempts=plan_bounded_cycle(contract["scene"],a.assets,row,
                low_board_m=a.low_board,high_dropper_m=a.high_dropper)
            report["attempt_history"]=[r for r in attempts[:-1]]
        else:
            report, frames = plan_cycle(contract["scene"], a.assets, row,
                low_board_m=a.low_board, high_dropper_m=a.high_dropper)
        reports.append(report)
        previews.append(dict(source_id=row["id"], passed=report["passed"], frames=frames))
        print(row["id"], report["passed"], flush=True)
    result = dict(version=VERSION, scope="CPU native geometry; no physics stepping or training", scene=contract["scene"],
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_contract_sha256=hashlib.sha256((a.study / "contract.json").read_bytes()).hexdigest(),
        trials=reports, passed=sum(r["passed"] for r in reports), total=len(reports))
    (a.output / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    (a.output / "preview.json").write_text(json.dumps(dict(version=VERSION, paths=previews), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
