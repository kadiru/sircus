"""Opt-in contact diagnostics; original environment and scoring remain unchanged.

Kinematic board mode makes the prescribed rigid board pose truly fixed during
contact. Passive-only mode removes the extra qfrc_bias force because every arm
body already has gravcomp=1. These are separate, explicit model interventions.
"""

import time
import mujoco
import numpy as np
import torch
from .learning_env import ShotEnvironment
from .learning_score import BatchScore
from .contract import ScoreStream, FORCE_NAMES
from .expanded_actions import checked_bounds, project, OLD_BOUNDS, VERSION


class ContactEnvironment(ShotEnvironment):
    def __init__(self, scene, assets, n=64, bounds=None, *, substeps=8,
                 board_mode="articulated", gravity_mode="legacy"):
        from .contact_model import make_model
        from mjlab.sim import Simulation, SimulationCfg, MujocoCfg
        from .run import sensor_config
        from .learning_ik import ShotIK
        import json
        if substeps not in (8, 16, 32, 64, 128):
            raise ValueError("Controlled substeps must be 8,16,32,64,128")
        if board_mode not in ("articulated", "kinematic") or gravity_mode not in ("legacy", "passive_only"):
            raise ValueError("Unknown contact protocol")
        if torch.cuda.device_count() != 1 or "4090" not in torch.cuda.get_device_name(0):
            raise RuntimeError("Select ONLY the RTX4090 by UUID")
        self.substeps, self.board_mode, self.gravity_mode = substeps, board_mode, gravity_mode
        self.scene, self.n, self.device = scene, n, "cuda:0"
        self.action_bounds = OLD_BOUNDS if bounds is None else dict(bounds)
        checked_bounds(self.action_bounds)
        spec, self.attached_model = make_model(scene, assets, board_mode)
        self.sensors = sensor_config(spec, spec.compile())
        self.m = spec.compile()
        self.sim = Simulation(num_envs=n, cfg=SimulationCfg(mujoco=MujocoCfg(timestep=1/(240*substeps)),
            nconmax=512, njmax=2048), model=self.m, device=self.device)
        for sensor in self.sensors:
            sensor.initialize(self.sim.mj_model, self.sim.model, self.sim.data, self.device)
        self.ik = ShotIK(scene, assets, device=self.device)
        self.seeds = json.loads((assets / "poses.json").read_text())
        m=self.m
        joint=lambda name:m.joint(name).id
        self.ball_q=int(m.jnt_qposadr[joint("ball/free")]);self.ball_v=int(m.jnt_dofadr[joint("ball/free")])
        self.board=m.body("Arm14/impact_board").id;self.flange=m.body("Arm10/link6").id
        self.bucket=m.body("bucket").id;self.mocap=int(m.body_mocapid[self.bucket])
        self.board_mocap=int(m.body_mocapid[self.board])
        self.q10=[int(m.jnt_qposadr[joint(f"Arm10/joint{k}")]) for k in range(1,7)]
        self.q14=[int(m.jnt_qposadr[joint(f"Arm14/joint{k}")]) for k in range(1,7)]
        self.arm_q=self.q10+self.q14
        self.arm_v=[int(m.jnt_dofadr[joint(f"Arm{role}/joint{k}")]) for role in (10,14) for k in range(1,7)]
        self.actuator_q=[int(m.jnt_qposadr[j]) for j in m.actuator_trnid[:,0]]
        self.fingers=[int(m.jnt_qposadr[joint(f"Arm10/finger_{side}_joint")]) for side in ("left","right")]
        self.cpu=mujoco.MjData(m)
        self.attached_data=mujoco.MjData(self.attached_model)
        self.links={role:[m.body(f"Arm{role}/link{k}").id for k in range(1,7)] for role in (10,14)}
        self.bucket_geoms=[i for i in range(m.ngeom) if (mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,i) or "").startswith("bucket/")]
        self.last_arrays=None

    def _board_poses(self, q):
        m,d=self.attached_model,self.attached_data
        poses=[];quats=[]
        # Removing the fixed board changes no joint ordering; assert instead of assuming.
        assert m.nq==self.m.nq and m.nv==self.m.nv
        for row in q:
            d.qpos[:]=row;d.qvel[:]=0
            mujoco.mj_forward(m,d)
            poses.append(d.xpos[m.body("Arm14/impact_board").id].copy())
            quats.append(d.xquat[m.body("Arm14/impact_board").id].copy())
        return np.asarray(poses),np.asarray(quats)

    def _collision_gate(self, q, xy):
        # Original attached model is authoritative for all endpoint gates.
        original_m,original_d=self.m,self.cpu
        self.m,self.cpu=self.attached_model,self.attached_data
        try:
            return super()._collision_gate(q,xy)
        finally:
            self.m,self.cpu=original_m,original_d

    def project(self, values, locks, radius):
        return project(values,locks,radius,self.action_bounds)

    def run(self, bucket, values, locks, *, record=False):
        start = time.monotonic()
        bucket = np.asarray(bucket, dtype=float)
        count = len(bucket)
        if not 0 < count <= self.n:
            raise ValueError("Batch outside configured environment count")
        values, allowed = self.project(values, locks, self.scene["ball"]["radius_m"])
        # Avoid zero-distance heading in rejected input rows.
        geometry = (np.linalg.norm(bucket-values[:,:2],axis=1) > .15)
        allowed &= geometry
        safe = values.copy()
        safe[~allowed] = [.65,-.04,.36,25,.24]
        safe_bucket = bucket.copy()
        close = np.linalg.norm(safe_bucket-safe[:,:2],axis=1) < .01
        safe_bucket[close,1] -= .4
        # Pad to fixed CUDA graph shapes. Padded samples are never counted.
        pad = lambda a: np.concatenate([a, np.repeat(a[-1:],self.n-count,axis=0)])
        bucket_p, values_p = pad(safe_bucket), pad(safe)
        ik = self.ik.solve(bucket_p, values_p[:,:3], values_p[:,3], values_p[:,4],
                           initial10=self.seeds["dropper_joint_rad"],initial14=self.seeds["board_joint_rad"],starts=4,iterations=90)
        q = np.tile(self.m.qpos0,(self.n,1))
        q[:,self.q10] = ik["q10"].cpu().numpy()
        q[:,self.q14] = ik["q14"].cpu().numpy()
        q[:,self.fingers] = .025
        # Fingers stay open; release is prescribed, no grasp actuation learned.
        q[:,self.ball_q:self.ball_q+3] = ik["targets"]["release_center_m"]
        q[:,self.ball_q+3:self.ball_q+7] = [1,0,0,0]
        collision = self._collision_gate(q[:count], bucket_p[:count])
        valid = ik["valid"][:count].cpu().numpy() & allowed & np.array([not c for c in collision])
        sim = self.sim
        sim.reset()
        targets = torch.tensor(np.column_stack([bucket_p,np.zeros(self.n)]),device=self.device,dtype=torch.float32)
        sim.data.mocap_pos[:,self.mocap,:] = targets
        sim.data.mocap_quat[:,self.mocap,:] = torch.tensor([1,0,0,0],device=self.device)
        qt = torch.tensor(q,device=self.device,dtype=torch.float32)
        sim.data.qpos[:] = qt
        sim.data.qvel[:] = 0
        sim.data.ctrl[:] = qt[:,self.actuator_q]
        if self.board_mode=="kinematic":
            bp,bq=self._board_poses(q)
            sim.data.mocap_pos[:,self.board_mocap,:]=torch.tensor(bp,device=self.device,dtype=torch.float32)
            sim.data.mocap_quat[:,self.board_mocap,:]=torch.tensor(bq,device=self.device,dtype=torch.float32)
        sim.forward()
        expected = targets[:,None,:] + torch.tensor(self.m.geom_pos[self.bucket_geoms],device=self.device,dtype=torch.float32)
        collider_error = float(torch.linalg.vector_norm(sim.data.geom_xpos[:,self.bucket_geoms]-expected,dim=-1).max())
        if collider_error > 1e-5:
            raise RuntimeError("Actual bucket collision transforms differ from scoring")
        centers = sim.data.xpos[:,self.board,:].clone()
        quaternions = sim.data.xquat[:,self.board,:].cpu().tolist()
        links = {role:sim.data.xpos[:,ids,:].cpu().tolist() for role,ids in self.links.items()}
        normals = sim.data.xmat[:,self.board,:].reshape(self.n,3,3)[:,:,2].clone()
        poserr = torch.linalg.vector_norm(centers-torch.tensor(values_p[:,:3],device=self.device),dim=-1).cpu().numpy()
        normerr = np.rad2deg(np.arccos(np.clip((normals.cpu().numpy()*ik["targets"]["board_normal_world"]).sum(1),-1,1)))
        droperr = torch.linalg.vector_norm(sim.data.xpos[:,self.flange]-torch.tensor(ik["targets"]["release_center_m"]+[0,0,.095],device=self.device),dim=-1).cpu().numpy()
        valid &= (poserr[:count]<.002)&(normerr[:count]<.5)&(droperr[:count]<.002)
        for sensor in self.sensors:
            sensor.reset()
        score = BatchScore(self.scene,bucket_p,normals,device=self.device)
        trajectories, sampled_forces, sampled_velocity, sampled_positions = [], [], [], []
        drift = torch.zeros(self.n,device=self.device)
        excursion=torch.zeros(self.n,device=self.device)
        peak_joint_velocity=torch.zeros(self.n,device=self.device)
        # Full 4s horizon; terminal score freezes at capture or fall, as before.
        for step in range(960):
            forces = torch.zeros((self.n,6),device=self.device)
            for _ in range(self.substeps):
                sim.data.qpos[:,:self.ball_q] = qt[:,:self.ball_q]
                sim.data.qvel[:,:self.ball_v] = 0
                sim.data.qfrc_applied[:,:self.ball_v] = (sim.data.qfrc_bias[:,:self.ball_v] if self.gravity_mode=="legacy" else 0)
                sim.step()
                excursion=torch.maximum(excursion,(sim.data.qpos[:,self.arm_q]-qt[:,self.arm_q]).abs().amax(1))
                peak_joint_velocity=torch.maximum(peak_joint_velocity,sim.data.qvel[:,self.arm_v].abs().amax(1))
                for sensor in self.sensors:
                    sensor.update(1/(240*self.substeps))
                vectors = [s.data.force for s in self.sensors]
                norms = [torch.linalg.vector_norm(v,dim=-1).sum(1) for v in vectors[:5]]
                residual = torch.linalg.vector_norm(vectors[-1].sum(1)+sum(v.sum(1) for v in vectors[:5]),dim=-1)
                forces = torch.maximum(forces,torch.stack([*norms,residual],-1))
            xyz = sim.data.qpos[:,self.ball_q:self.ball_q+3]
            vel = sim.data.qvel[:,self.ball_v:self.ball_v+3]
            score.step(xyz,vel,forces)
            drift = torch.maximum(drift,torch.linalg.vector_norm(sim.data.xpos[:,self.board]-centers,dim=-1))
            if record and (step+1)%4==0:
                trajectories.append(xyz[:count].clone())
            if record:
                sampled_forces.append(forces[:count].clone())
                sampled_velocity.append(vel[:count].clone())
                sampled_positions.append(xyz[:count].clone())
        rows = score.rows()[:count]
        if record:
            fp=torch.stack(sampled_forces).cpu().numpy()
            vp=torch.stack(sampled_velocity).cpu().numpy()
            pp=torch.stack(sampled_positions).cpu().numpy()
            self.last_arrays=dict(forces_N=fp,velocity_m_s=vp,position_m=pp)
            for i in range(count):
                reference=ScoreStream(self.scene,bucket_p[i],normals[i].cpu().numpy())
                for p,v,f in zip(pp[:,i],vp[:,i],fp[:,i]):
                    reference.step(p,v,dict(zip(FORCE_NAMES,f)))
                actual=reference.result()
                for key in ("success","complete_success","board_bounce","whole_ball_entry","quality_score","board_contact_episodes","external_contact_steps","physics_steps"):
                    if actual[key]!=rows[i][key]:
                        raise RuntimeError(f"Native GPU/scalar scorer mismatch: {i}/{key}")
                rows[i]["scalar_score_audit_passed"]=True
        trails = torch.stack(trajectories,dim=1).cpu().tolist() if trajectories else None
        for i,row in enumerate(rows):
            invalid = None if valid[i] else ("constraint_or_distance" if not allowed[i] else "endpoint_collision" if collision[i] else "endpoint_ik_or_pose")
            if invalid:
                row.update(route_success=False,complete_success=False,route_quality=0,single_bounce_success=False)
            row.update(invalid=invalid,bucket_xy_m=bucket[i].tolist(),values=values[i].tolist(),
                       board_center_m=centers[i].cpu().tolist(),board_normal=normals[i].cpu().tolist(),
                       release_ball_center_m=ik["targets"]["release_center_m"][i].tolist(),
                       q10=q[i,self.q10].tolist(),q14=q[i,self.q14].tolist(),
                       pose_gate=dict(board_position_error_mm=float(poserr[i]*1000),board_normal_error_deg=float(normerr[i]),dropper_error_mm=float(droperr[i]*1000)),
                       endpoint_collision_pairs=collision[i],max_board_drift_mm=float(drift[i]*1000),bucket_collider_error_m=collider_error)
            row["max_integrated_joint_excursion_rad"]=float(excursion[i])
            row["peak_integrated_joint_velocity_rad_s"]=float(peak_joint_velocity[i])
            row["contact_protocol"]=dict(substeps=self.substeps,board_mode=self.board_mode,gravity_mode=self.gravity_mode)
            row["board_quaternion_wxyz"]=quaternions[i]
            row["link_positions10"]=[[0,0,0],*links[10][i]]
            row["link_positions14"]=[[self.scene["arms"]["base_spacing_m"],0,0],*links[14][i]]
            if record:
                row["trajectory_m"] = trails[i]
                row["trajectory_dt_s"] = 1/60
        return rows, dict(seconds=time.monotonic()-start,batch_worlds=self.n,actual_shots=count)
