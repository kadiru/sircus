"""Opt-in driven setup: finish transit before correcting steady Cartesian bias.

Original holding gains, gravity compensation, release tolerances and scoring
remain unchanged. Episode home reset precedes filming; no endpoint state writes.
"""
import copy
import math

import numpy as np
from scipy.spatial.transform import Rotation

from continuous_pilot import ContinuousShots
from settling_contract import accepted, correction, pose_errors

VERSION = "sircus-driven-settlement-v5"


def protocol():
    return dict(version=VERSION, transit_velocity_rad_s=.85,
                transit_stiffness=6000., transit_damping=180.,
                transit_effort="Original URDF joint caps", transit_tail_max_s=8.,
                gain_restore_ramp_s=1., holding_gains="Original native gains/caps restored before measurement and release",
                initial_stabilization_max_s=40., correction_stabilization_max_s=8.,
                total_stabilization_max_s=64., minimum_hold_s=1.75, measurement_window_s=.25,
                stability_early_exit="After minimum hold, every currently eligible cell is release-ready or quiet with joint lag <=0.02 rad",
                corrections_max=3, correction_joint_lag_limit_rad=.02,
                release_position_mm=2., release_normal_deg=.5,
                release_joint_velocity_rad_s=.005, release_board_drift_mm=.1,
                release_normal_drift_deg=.05,
                velocity_gate_scope="Legacy numeric maximum over ALL articulation DOFs, including both simulated gripper fingers; revolute rad/s and prismatic native m/s. Six-axis FK/lag state stays separate.",
                scope="Bounded driven setup and steady-bias feedback for both arms; no selected-endpoint joint-state reset. Readiness and release are distinct. Original score/impact holding drives unchanged.")


def equivalent_fit(target, fit, links, reference, normal_only):
    """Remove legal full turns/null wrist spin, then independently verify FK."""
    from expanded_pose_native import verified_fit
    from prepare_assets import fk
    if not fit["solved"]:
        return fit
    q, reference = np.asarray(fit["q"], float).copy(), np.asarray(reference, float)
    before = q.copy()
    for j, link in enumerate(links):
        lo, hi = link[2:4]
        choices = q[j] + 2*np.pi*np.arange(-3, 4)
        choices = choices[(choices >= lo) & (choices <= hi)]
        if len(choices):
            q[j] = choices[np.argmin(np.abs(choices-reference[j]))]
    if normal_only and links[-1][2] <= reference[-1] <= links[-1][3]:
        candidate = q.copy()
        candidate[-1] = reference[-1]
        a, b = fk(q, links), fk(candidate, links)
        if np.linalg.norm(a[:3, 3]-b[:3, 3]) < 1e-10 and np.linalg.norm(a[:3, 2]-b[:3, 2]) < 1e-10:
            q = candidate
    result = dict(fit, q=q.tolist(), equivalent_representation=dict(
        original_q=before.tolist(), reference_q=reference.tolist(),
        max_change_rad=float(np.abs(q-before).max())))
    verified_fit(target, result, links, normal_only)
    return result


def quiet(metric):
    return (metric["joint_speed_rad_s"] <= .005
            and metric["position_drift_mm"] <= .1
            and metric["normal_drift_deg"] <= .05
            and metric["dropper_drift_mm"] <= .1)


def release_ready(metric):
    return accepted(metric) and metric["dropper_position_error_mm"] <= 2


def feedback_ready(metric):
    lag=max(abs(p-t) for state in metric["arms"].values()
            for p,t in zip(state["position_rad"],state["target_rad"]))
    return quiet(metric) and lag <= .02


class SettledTransitShots(ContinuousShots):
    def board_poses(self):
        # Unlike the old mixin, each read is an instantaneous readback. All waits
        # are explicit and the measured stability window is exactly 60 ticks.
        from parallel_runtime import ParallelShots
        return ParallelShots.board_poses(self)

    def _ik(self, target, name, seed=None):
        from robust_ik import solve
        key = (name, tuple(target.ravel()))
        if seed is None and key in self.ik_cache:
            return self.ik_cache[key]
        reference = getattr(self, "solve_reference", {}).get(name)
        if reference is None:
            reference = self.poses["board_joint_rad" if name == "Arm14" else "dropper_joint_rad"]
        fit = solve(target, self.continuous_links[name], reference if seed is None else seed, name == "Arm14")
        return equivalent_fit(target, fit, self.continuous_links[name], reference, name == "Arm14")

    def _state(self):
        return {n: dict(position_rad=a.data.joint_pos[:, :6].cpu().tolist(),
                        velocity_rad_s=a.data.joint_vel[:, :6].cpu().tolist(),
                        target_rad=a.data.joint_pos_target[:, :6].cpu().tolist())
                for n, a in self.arms.items()}

    def _joint_status(self, targets):
        errors = {n: (a.data.joint_pos[:, :6]-targets[n][:, :6]).abs().amax(dim=1).cpu().numpy()
                  for n,a in self.arms.items()}
        speeds = {n: a.data.joint_vel.abs().amax(dim=1).cpu().numpy()
                  for n,a in self.arms.items()}
        return errors, speeds

    def approach_endpoints(self, targets):
        starts = {n:a.data.joint_pos.clone() for n,a in self.arms.items()}
        self.set_balls(np.tile([.1, -.8, .04], (self.n, 1)))
        for n in targets:
            targets[n][~self.preparation_enabled] = starts[n][~self.preparation_enabled]
        deltas = {n:targets[n]-starts[n] for n in targets}
        longest = max(float(d.abs().max()) for d in deltas.values())
        travel = max(120, int(math.ceil(1.5*longest/(.85*self.dt))))
        original = {}
        for n,a in self.arms.items():
            ids = [a.joint_names.index(f"joint{i+1}") for i in range(6)]
            original[n] = (ids, a.data.joint_effort_limits[:,ids].clone(),
                           a.data.joint_stiffness[:,ids].clone(), a.data.joint_damping[:,ids].clone())
            a.write_joint_effort_limit_to_sim(self.transit_limits.expand(self.n,6), ids)
            a.write_joint_stiffness_to_sim(6000., ids)
            a.write_joint_damping_to_sim(180., ids)
        self.transit_history = [dict(phase="start", step=self.steps, state=self._state(),
            original_gains={n:dict(effort=x[1].cpu().tolist(),stiffness=x[2].cpu().tolist(),damping=x[3].cpu().tolist()) for n,x in original.items()})]
        ready = np.zeros(self.n, dtype=bool)
        try:
            for step in range(1, travel+round(8/self.dt)+1):
                if step <= travel:
                    t=step/travel; t=t*t*(3-2*t)
                    for n,a in self.arms.items():
                        a.set_joint_position_target(starts[n]+t*deltas[n])
                self.tick()
                if step % 60 == 0:
                    errors,speeds=self._joint_status(targets)
                    ready=np.logical_and.reduce([errors[n] <= .001 for n in errors]+[speeds[n] <= .005 for n in speeds])
                    self.transit_history.append(dict(phase="travel" if step<travel else "tail", step=self.steps,
                        errors_rad={n:v.tolist() for n,v in errors.items()},ready=ready.tolist(), state=self._state()))
                    if step >= travel and np.all(ready[self.preparation_enabled]):
                        break
            # Restore smoothly BEFORE the original-gain stability/readback check.
            for ramp in range(1, round(1/self.dt)+1):
                t=ramp/round(1/self.dt)
                for n,a in self.arms.items():
                    ids,effort,stiffness,damping=original[n]
                    a.write_joint_effort_limit_to_sim((1-t)*self.transit_limits.expand(self.n,6)+t*effort,ids)
                    a.write_joint_stiffness_to_sim((1-t)*6000.+t*stiffness,ids)
                    a.write_joint_damping_to_sim((1-t)*180.+t*damping,ids)
                self.tick()
        finally:
            for n,a in self.arms.items():
                ids,effort,stiffness,damping=original[n]
                a.write_joint_effort_limit_to_sim(effort,ids)
                a.write_joint_stiffness_to_sim(stiffness,ids)
                a.write_joint_damping_to_sim(damping,ids)
        self.transit_ready_at_high_gain=ready.tolist()
        errors,speeds=self._joint_status(targets)
        ready=np.logical_and.reduce([errors[n] <= .001 for n in errors]+[speeds[n] <= .005 for n in speeds])
        self.drive_ticks=step
        self.transit_joint_error_rad=np.maximum.reduce(list(errors.values())).tolist()
        self.transit_ready_after_drive=ready.tolist()
        self.transit_history.append(dict(phase="restored_original_hold",step=self.steps,state=self._state(),
            restored_gains={n:dict(effort=a.data.joint_effort_limits[:,original[n][0]].cpu().tolist(),
                                  stiffness=a.data.joint_stiffness[:,original[n][0]].cpu().tolist(),
                                  damping=a.data.joint_damping[:,original[n][0]].cpu().tolist()) for n,a in self.arms.items()}))
        self.assert_buckets_fixed()

    def _measure(self, board_targets, drop_targets, iteration, records):
        prev,prev_normal=self.board_poses()
        prev_drop=self.arms["Arm10"].data.body_pos_w[:,self.flange_idx].cpu().numpy().copy()-self.origins
        start_step=self.steps
        for _ in range(60): self.tick()
        centers,normals=self.board_poses()
        drop=self.arms["Arm10"].data.body_pos_w[:,self.flange_idx].cpu().numpy().copy()-self.origins
        # Match the historical gate over every native DOF, including the
        # simulated prismatic fingers, rather than only the six FK axes.
        all_velocity={n:a.data.joint_vel.cpu().numpy().copy() for n,a in self.arms.items()}
        speed=np.maximum.reduce([np.abs(v).max(axis=1) for v in all_velocity.values()])
        state=self._state()
        metrics=[]
        for i in range(self.n):
            m=dict(pose_errors(board_targets[i],centers[i],normals[i]),
                iteration=iteration,measurement_start_step=start_step,physics_step=self.steps,joint_speed_rad_s=float(speed[i]),
                position_drift_mm=float(np.linalg.norm(centers[i]-prev[i])*1000),
                normal_drift_deg=float(np.rad2deg(np.arccos(np.clip(normals[i]@prev_normal[i],-1,1)))),
                dropper_position_error_mm=float(np.linalg.norm(drop[i]-drop_targets[i][:3,3])*1000),
                dropper_drift_mm=float(np.linalg.norm(drop[i]-prev_drop[i])*1000),
                previous_board_center_m=prev[i].tolist(),previous_board_normal=prev_normal[i].tolist(),
                previous_dropper_flange_m=prev_drop[i].tolist(),
                all_articulation_joint_velocity={n:v[i].tolist() for n,v in all_velocity.items()},
                articulation_joint_names={n:list(a.joint_names) for n,a in self.arms.items()},
                board_center_m=centers[i].tolist(),board_normal=normals[i].tolist(),
                dropper_flange_m=drop[i].tolist(),arms={n:{k:v[i] for k,v in s.items()} for n,s in state.items()})
            metrics.append(m);records[i].append(m)
        return metrics,centers,normals

    def _wait_stable(self, board_targets, drop_targets, iteration, records, eligible):
        # Steady Cartesian bias can be corrected only after motion is quiet;
        # drift alone extends the wait rather than changing the target.
        # The fixed-request audit found a slow board-base yaw convergence, often
        # understated by joint velocities. Give actual pose drift time to decay
        # under the ORIGINAL holding drives before considering bias feedback.
        maximum=40. if iteration==0 else 8.
        for window in range(round(maximum/.25)):
            metrics,centers,normals=self._measure(board_targets,drop_targets,iteration,records)
            if (window+1)*.25 >= 1.75 and all(release_ready(metrics[i]) or feedback_ready(metrics[i]) for i in np.flatnonzero(eligible)):
                break
        return metrics,centers,normals

    def prepare(self, actions):
        from continuous_feasibility import qualify_action, approximate_path_check
        from expanded_pose_menu import endpoint_targets
        from feasibility_contract import dynamic_cell
        from prepare_assets import matrix
        from robust_ik import solve
        self.spec["cells"]=[dynamic_cell(c,a) for c,a in zip(self.base_cells,actions)]
        self.path_reasons=[None]*self.n;self.path_checks=[None]*self.n
        self.qualifications=[];invalid=[None]*self.n
        # Keep the original open-finger targets at active sites. The approach
        # routine replaces every inactive target with its actual parked state.
        q={n:a.data.default_joint_pos.clone() for n,a in self.arms.items()}
        board_targets=[];drop_targets=[];releases=[]
        base=matrix([self.cfg["arms"]["base_spacing_m"],0,0],[0,0,np.pi])
        mount=matrix([0,0,self.cfg["board"]["tool_offset_m"]],[0,0,self.cfg["board"]["mount_yaw_rad"]])
        for i,action in enumerate(actions):
            endpoints=endpoint_targets(self.cfg,self.base_cells[i],action)
            board_targets.append(base@endpoints["Arm14"]@mount)
            drop_targets.append(endpoints["Arm10"])
            releases.append(np.asarray(endpoints["Arm10"][:3,3])-[0,0,.095])
            reason=None
            if not self.active_mask[i]:reason="inactive_split"
            elif self.trial_mode!="coverage" and getattr(self,"support_missing",[False]*self.n)[i]:reason="no_geometry_support"
            elif not self.home_readback["path_eligible"][i]:reason="common_home_path_proxy_rejected"
            if reason:
                self.qualifications.append(dict(accepted=False,reason=reason));invalid[i]=reason;continue
            actual={n:a.data.joint_pos[i,:6].cpu().numpy().copy() for n,a in self.arms.items()}
            seeds=getattr(self,"proposal_ik_seeds",None)
            seeds={n:seeds[i][n]["q"] for n in self.arms} if seeds is not None and seeds[i] is not None else actual
            if self.trial_mode=="coverage" and self.coverage_fits[i] is not None:
                seeds={n:self.coverage_fits[i][n]["q"] for n in self.arms}
            def fresh(target,links,seed,normal_only):
                name="Arm14" if normal_only else "Arm10"
                return equivalent_fit(target,solve(target,links,seed,normal_only),links,actual[name],normal_only)
            fit=qualify_action(self.cfg,self.base_cells[i],action,self.continuous_links,seeds,solver=fresh)
            self.qualifications.append(fit)
            if not fit["accepted"]:invalid[i]=fit["reason"];continue
            check=approximate_path_check(self.cfg,self.continuous_links,actual,{n:fit["ik"][n]["q"] for n in self.arms},cell=self.base_cells[i],samples=16)
            self.path_checks[i]=check
            if not check["accepted"]:
                self.path_reasons[i]=check["reasons"][0];invalid[i]=self.path_reasons[i];continue
            for n in self.arms:
                for j in range(6):q[n][i,self.arms[n].joint_names.index(f"joint{j+1}")]=fit["ik"][n]["q"][j]
        self.preparation_enabled=np.array([x is None for x in invalid])
        self.approach_endpoints(q)
        records=[[] for _ in range(self.n)]
        board_commands=np.array(board_targets).copy();drop_commands=np.array(drop_targets).copy()
        self.correction_history=[]
        for iteration in range(4):
            metrics,centers,normals=self._wait_stable(board_targets,drop_targets,iteration,records,np.array([x is None for x in invalid]))
            bad=[i for i in range(self.n) if invalid[i] is None and not release_ready(metrics[i])]
            if not bad:break
            if iteration==3:
                for i in bad:
                    lag=max(max(abs(p-t) for p,t in zip(s["position_rad"],s["target_rad"])) for s in metrics[i]["arms"].values())
                    invalid[i]=("setup_motion_not_quiet" if not quiet(metrics[i]) else
                                "setup_transit_incomplete" if lag>.02 else "endpoints_not_settled")
                break
            for i in bad:
                m=metrics[i]
                lag=max(max(abs(p-t) for p,t in zip(s["position_rad"],s["target_rad"])) for s in m["arms"].values())
                # Other cells' correction waits may finish this one's motion.
                # Keep it pending until the finite final window, without ever
                # correcting a moving or grossly unfinished endpoint.
                if not quiet(m) or lag>.02:continue
                # Never apply board feedback when only the dropper is offset.
                commands={}
                if m["position_error_mm"]>2 or m["normal_error_deg"]>.5:
                    board_commands[i]=correction(board_commands[i],board_targets[i],centers[i],normals[i])
                    commands["Arm14"]=np.linalg.inv(base)@board_commands[i]@np.linalg.inv(mount)
                if m["dropper_position_error_mm"]>2:
                    drop_commands[i][:3,3]+=drop_targets[i][:3,3]-np.asarray(m["dropper_flange_m"])
                    commands["Arm10"]=drop_commands[i]
                actual={n:a.data.joint_pos[i,:6].cpu().numpy().copy() for n,a in self.arms.items()}
                # The other arm keeps its commanded target, which can differ
                # slightly from the current measured pose. Verify that motion.
                solved={n:q[n][i,:6].cpu().numpy().copy() for n in actual}
                fits={}
                for n,target in commands.items():
                    self.solve_reference={n:actual[n]}
                    fit=self._ik(target,n,actual[n]);fits[n]=fit
                    if not fit["solved"]:invalid[i]="settling_correction_ik";break
                    solved[n]=fit["q"]
                if invalid[i] is None:
                    check=approximate_path_check(self.cfg,self.continuous_links,actual,solved,cell=self.base_cells[i],samples=16)
                    if not check["accepted"]:invalid[i]="settling_correction_path_proxy"
                    else:
                        for n in commands:
                            for j in range(6):q[n][i,self.arms[n].joint_names.index(f"joint{j+1}")]=solved[n][j]
                self.correction_history.append(dict(cell_id=self.base_cells[i]["id"],iteration=iteration,
                    arms=list(commands),fits=fits,invalid=invalid[i],measured=m))
            for n,a in self.arms.items():a.set_joint_position_target(q[n])
        self.release_centers,self.release_normals=centers.copy(),normals.copy()
        self.scoring_normals=normals.copy()
        self.settlement_records=records
        self.assert_buckets_fixed()
        # Preserve FeasibilityShots' explicit recomputed-distance guard,
        # including its boundary behavior beyond the parameter-space LOW.
        for i,cell in enumerate(self.spec["cells"]):
            if cell["distance_m"] < .28:
                invalid[i]="board_bucket_separation"
        return np.asarray(releases),normals,records,invalid
