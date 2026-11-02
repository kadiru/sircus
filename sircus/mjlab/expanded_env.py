"""Versioned action bounds with the original native shot integration unchanged.

Only action projection differs. Constructor, assets, physics cadence, force
attribution, native FK/collision gates and scalar score audit remain inherited
or byte-identical to the original environment. Original source is not edited.
"""

import time
import mujoco
import numpy as np
import torch
from .learning_env import ShotEnvironment
from .learning_score import BatchScore
from .contract import ScoreStream, FORCE_NAMES
from .expanded_actions import checked_bounds, project, OLD_BOUNDS, VERSION


class ExpandedEnvironment(ShotEnvironment):
    def __init__(self, scene, assets, n=128, bounds=None):
        checked_bounds(bounds)
        self.action_bounds = OLD_BOUNDS if bounds is None else dict(bounds)
        super().__init__(scene,assets,n)

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
        # Full 4s horizon; terminal score freezes at capture or fall, as before.
        for step in range(960):
            forces = torch.zeros((self.n,6),device=self.device)
            for _ in range(8):
                sim.data.qpos[:,:self.ball_q] = qt[:,:self.ball_q]
                sim.data.qvel[:,:self.ball_v] = 0
                sim.data.qfrc_applied[:,:self.ball_v] = sim.data.qfrc_bias[:,:self.ball_v]
                sim.step()
                for sensor in self.sensors:
                    sensor.update(1/1920)
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
                sampled_forces.append(forces[:min(count,8)].clone())
                sampled_velocity.append(vel[:min(count,8)].clone())
                sampled_positions.append(xyz[:min(count,8)].clone())
        rows = score.rows()[:count]
        if record:
            fp=torch.stack(sampled_forces).cpu().numpy()
            vp=torch.stack(sampled_velocity).cpu().numpy()
            pp=torch.stack(sampled_positions).cpu().numpy()
            for i in range(min(count,8)):
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
            row["board_quaternion_wxyz"]=quaternions[i]
            row["link_positions10"]=[[0,0,0],*links[10][i]]
            row["link_positions14"]=[[self.scene["arms"]["base_spacing_m"],0,0],*links[14][i]]
            if record:
                row["trajectory_m"] = trails[i]
                row["trajectory_dt_s"] = 1/60
        return rows, dict(seconds=time.monotonic()-start,batch_worlds=self.n,actual_shots=count)
