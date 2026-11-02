"""Batched mjlab shot trials with endpoint and contact gates, no physical APIs."""

import time
import json
import mujoco
import numpy as np
import torch
from mjlab.sim import Simulation, SimulationCfg, MujocoCfg
from .model import build_scene
from .run import sensor_config
from .learning_ik import ShotIK
from .learning_score import BatchScore
from .learning_contract import project
from .contract import ScoreStream, FORCE_NAMES


class ShotEnvironment:
    def __init__(self, scene, assets, n=128):
        if torch.cuda.device_count() != 1 or "4090" not in torch.cuda.get_device_name(0):
            raise RuntimeError("Select ONLY the RTX4090 by UUID")
        self.scene, self.n, self.device = scene, n, "cuda:0"
        spec = mujoco.MjSpec.from_string(build_scene(scene, assets, solref=(.004, .15), contact_friction=.35))
        self.sensors = sensor_config(spec, spec.compile())
        self.m = spec.compile()
        self.sim = Simulation(num_envs=n, cfg=SimulationCfg(mujoco=MujocoCfg(timestep=1/1920),
                              nconmax=512, njmax=2048), model=self.m, device=self.device)
        for sensor in self.sensors:
            sensor.initialize(self.sim.mj_model, self.sim.model, self.sim.data, self.device)
        self.ik = ShotIK(scene, assets, device=self.device)
        self.seeds = json.loads((assets / "poses.json").read_text())
        ident = lambda kind, name: mujoco.mj_name2id(self.m, kind, name)
        joint = lambda name: ident(mujoco.mjtObj.mjOBJ_JOINT, name)
        self.ball_q = int(self.m.jnt_qposadr[joint("ball/free")])
        self.ball_v = int(self.m.jnt_dofadr[joint("ball/free")])
        self.board = ident(mujoco.mjtObj.mjOBJ_BODY, "Arm14/impact_board")
        self.flange = ident(mujoco.mjtObj.mjOBJ_BODY, "Arm10/link6")
        self.bucket = ident(mujoco.mjtObj.mjOBJ_BODY, "bucket")
        self.mocap = int(self.m.body_mocapid[self.bucket])
        self.q10 = [int(self.m.jnt_qposadr[joint(f"Arm10/joint{k}")]) for k in range(1,7)]
        self.q14 = [int(self.m.jnt_qposadr[joint(f"Arm14/joint{k}")]) for k in range(1,7)]
        self.actuator_q = [int(self.m.jnt_qposadr[j]) for j in self.m.actuator_trnid[:,0]]
        self.fingers = [int(self.m.jnt_qposadr[joint(f"Arm10/finger_{side}_joint")]) for side in ("left","right")]
        self.cpu = mujoco.MjData(self.m)
        self.links = {role:[ident(mujoco.mjtObj.mjOBJ_BODY,f"Arm{role}/link{k}") for k in range(1,7)] for role in (10,14)}
        self.bucket_geoms = [i for i in range(self.m.ngeom) if (mujoco.mj_id2name(self.m,mujoco.mjtObj.mjOBJ_GEOM,i) or "").startswith("bucket/")]

    def _collision_gate(self, q, xy):
        """Reject endpoint penetration between robot/board and world/other arm.

        Ball is deliberately initialized at release and excluded here. Contact
        during the flight is still attributed by all six original categories.
        """
        errors = []
        for qi, bucket in zip(q, xy):
            self.cpu.qpos[:] = qi
            self.cpu.qvel[:] = 0
            self.cpu.mocap_pos[self.mocap] = [*bucket,0]
            self.cpu.mocap_quat[self.mocap] = [1,0,0,0]
            mujoco.mj_forward(self.m, self.cpu)
            bad = []
            for c in self.cpu.contact[:self.cpu.ncon]:
                names = [mujoco.mj_id2name(self.m,mujoco.mjtObj.mjOBJ_GEOM,int(g)) or "" for g in c.geom]
                if c.dist < -.002 and not any(x.startswith("ball/") for x in names):
                    bad.append(names)
            errors.append(bad)
        return errors

    def run(self, bucket, values, locks, *, record=False):
        start = time.monotonic()
        bucket = np.asarray(bucket, dtype=float)
        count = len(bucket)
        if not 0 < count <= self.n:
            raise ValueError("Batch outside configured environment count")
        values, allowed = project(values, locks, self.scene["ball"]["radius_m"])
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
