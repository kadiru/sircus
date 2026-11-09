"""Explicit native scene construction and measured, finite startup diagnostics.

Importing this module starts nothing. The original film source stays unchanged:
only its imports/classes and scene-construction prefix are compiled after the
caller has explicitly started one AppLauncher. Its training/filming main loop
and top-level launcher are never executed by this adapter.
"""

import ast
import copy
import hashlib
import json
from pathlib import Path
import time
import types

import numpy as np

from candidate_continuation_journal import write_new_json
from candidate_continuation_launcher import STARTUP_HELPERS, DiagnosticLedger
from candidate_continuation_ledger import canonical_hash

SCHEMA = "sircus-candidate-continuation-startup-evidence-v1"
SOURCE = Path(__file__).resolve().parent


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scene_program(raw):
    """Return original definitions and the setup prefix, without executing any."""
    tree = ast.parse(raw)
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    first = next(i for i,n in enumerate(tree.body) if isinstance(n, ast.Import)
                 and any(a.name == "isaaclab.sim" for a in n.names))
    definitions = [copy.deepcopy(n) for n in tree.body[first:tree.body.index(main)]]
    require(all(isinstance(n,(ast.Import,ast.ImportFrom,ast.ClassDef,ast.FunctionDef)) for n in definitions),
            "Unreviewed top-level statement in original native definitions")
    setup = [];found_gate=False
    for node in main.body:
        setup.append(copy.deepcopy(node))
        # Include the original origins equality gate immediately after env construction.
        if isinstance(node, ast.If) and isinstance(node.test,ast.UnaryOp) \
                and isinstance(node.test.operand,ast.Call) \
                and ast.unparse(node.test.operand.func) == "np.array_equal":
            found_gate=True
            break
    require(found_gate and any(isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id == "env" for t in n.targets)
                for n in setup) and isinstance(setup[-1],ast.If), "Original scene setup boundary changed")
    setup.extend(ast.parse("return env, presentation_bundle, saved_camera, camera_binding, stage_settings(), preview_backdrop").body)
    function = ast.FunctionDef(name="_construct_original_scene",args=ast.arguments(posonlyargs=[],args=[],
        kwonlyargs=[],kw_defaults=[],defaults=[]),body=setup,decorator_list=[])
    return ast.fix_missing_locations(ast.Module(body=definitions+[function],type_ignores=[]))


def construct_scene(*, output, assets, source_run, camera, device="cuda:0"):
    """Production-only explicit call; caller owns the sole initialized app."""
    for name, sha in STARTUP_HELPERS.items():
        require(digest(SOURCE/name) == sha, "Original startup helper bytes changed: " + name)
    output = Path(output)
    require(not output.exists() and output.parent.is_dir(), "Native setup path must be untouched")
    output.mkdir()
    args = types.SimpleNamespace(assets=str(assets),output=output,device=device,
        appearance="silver",continuous_source=Path(source_run),continuous_camera_json=Path(camera),
        shot_benchmark_plan=Path("continuation-dispatch-only"),candidate_comparison_plan=None,
        dark_replay=None,expanded_pose_menu=None,rl_study_source=None,presentation_source=None,
        rl_study_camera_json=None,presentation_camera_json=None,training_camera_json=None,camera_json=None,
        camera_previews=False,recording_plan="fixed-station-v1",preview_only=False,
        diverse_settled_support=None)
    namespace = dict(__file__=str(SOURCE/"film_cylinder_farm.py"),__name__="_sircus_continuation_scene",
        args=args,continuous=True,presentation=False,rl_study=False,training_camera=False,
        copy=copy,json=json,Path=Path,time=time,__builtins__=__builtins__)
    import shutil
    namespace["shutil"] = shutil
    exec(compile(scene_program((SOURCE/"film_cylinder_farm.py").read_text()),
                 str(SOURCE/"film_cylinder_farm.py"),"exec"),namespace)
    env,bundle,saved,binding,stage,backdrop = namespace["_construct_original_scene"]()
    env.stage_metadata,env.saved_camera = stage,saved
    env.stage_metadata["camera_backdrop"] = backdrop
    if binding is not None: env.camera_binding = binding
    env.foreign_probe_index = 7
    return env,bundle


def _plain(value):
    if hasattr(value,"detach"): value = value.detach().cpu().numpy()
    array = np.asarray(value,dtype=float)
    require(np.isfinite(array).all(), "Nonfinite startup readback")
    return array.tolist()


class StartupRecorder:
    """Temporarily observe original set_balls/tick calls, restoring all methods.

    No ball placement, physics tick or retry is added. Raw initialized states and
    per-step contact forces permit independent schedule/cost/outcome checking.
    """
    def __init__(self, env, directory, *, publish=lambda records,parking:None,
                 clock=time.monotonic, cpu=time.process_time):
        self.env,self.directory,self.publish = env,Path(directory),publish
        self.clock,self.cpu = clock,cpu
        self.directory.mkdir()
        self.records,self.evidence,self.parking = [],[],[]
        self.current,self.kind = None,None
        self.found = np.zeros(35,dtype=bool)
        self.rays,self.contacts = 0,0

    def _finish(self):
        if self.current is None: return
        current,self.current = self.current,None
        current["end_step"] = self.env.steps
        current["wall_s"] = self.clock()-current.pop("wall_begin")
        current["process_cpu_s"] = self.cpu()-current.pop("cpu_begin")
        if not current["forces_N"]:
            name=f"parking-{len(self.parking):04d}.json"
            write_new_json(self.directory/name,current)
            self.parking.append(dict(file=name,sha256=digest(self.directory/name)))
            return
        index=len(self.records)
        forces=np.asarray(current["forces_N"],dtype=float)
        require(forces.ndim==3 and forces.shape[1:]==(35,6) and np.isfinite(forces).all()
                and (forces>=0).all(),"Invalid startup contact observation")
        steps=len(forces)
        require(current["end_step"]-current["start_step"]==steps,
                "Startup physics clock lost a sample")
        if self.kind=="rim":
            from candidate_continuation_launcher import RIM_ANGLES
            require(self.rays<len(RIM_ANGLES),"Repeated rim diagnostics")
            first=np.zeros((35,6))
            for row in forces:
                touched=(first.max(axis=1)<=.01)&(row.max(axis=1)>.01)
                first[touched]=row[touched]
            self.found |= (first[:,1]>.01)&(first[:,[0,2,3,4,5]].max(axis=1)<=.01)
            kind,passed,active="rim_ray",bool(self.found.all()),35
            extra=dict(azimuth_deg=RIM_ANGLES[self.rays]);self.rays+=1
        elif self.contacts<5:
            from candidate_continuation_launcher import CONTACT_KINDS
            kind=CONTACT_KINDS[self.contacts];self.contacts+=1
            target={"bucket":1,"rim":1,"table":2,"floor":3,"robot":4}[kind]
            peaks=forces.max(axis=0);passed=bool((peaks[:,target]>.01).all())
            if kind in ("table","floor"):
                passed = passed and bool((peaks[:,[0,1]]<=.01).all())
            active,extra=35,{}
        else:
            require(self.contacts==5,"Repeated isolation diagnostic")
            self.contacts+=1;kind="foreign_cell_collision_filter";active=1;extra={}
            passed=bool(forces[:,0].max()<=.01 and current["final_ball_z_m"]<-1)
        record=dict(kind=kind,placements=35,active_probes=active,physics_steps=steps,
            initialized_states_sha256=current["initialized_sha256"],passed=passed,
            wall_s=current["wall_s"],process_cpu_s=current["process_cpu_s"],**extra)
        name=f"cohort-{index:04d}.json"
        write_new_json(self.directory/name,current)
        self.evidence.append(dict(file=name,sha256=digest(self.directory/name)))
        self.records.append(record)
        validator=DiagnosticLedger();validator.observe(self.records,len(self.parking))
        self.publish(copy.deepcopy(self.records),len(self.parking))

    def run(self):
        env=self.env
        require(env.n==35 and all(name not in vars(env) for name in ("tick","set_balls")),
                "Startup instrumentation requires genuine inherited methods")
        original_set,original_tick=env.set_balls,env.tick
        def place(*args,**kwargs):
            self._finish()
            original_set(*args,**kwargs)
            initial=dict(step=env.steps,position_m=_plain(env.ball.data.root_pos_w),
                         velocity_m_s=_plain(env.ball.data.root_lin_vel_w))
            require(np.asarray(initial["position_m"]).shape==np.asarray(initial["velocity_m_s"]).shape==(35,3),
                    "Startup must record all initialized ball states")
            name=f"initialized-{len(self.records)+len(self.parking):04d}.json"
            write_new_json(self.directory/name,initial)
            self.current=dict(initialized_file=name,initialized_sha256=digest(self.directory/name),
                start_step=env.steps,forces_N=[],wall_begin=self.clock(),cpu_begin=self.cpu())
        def tick():
            original_tick()
            if self.current is not None:
                self.current["forces_N"].append(_plain(env.forces()))
                self.current["final_ball_z_m"]=float(_plain(env.ball.data.root_pos_w)[0][2])
        env.set_balls,env.tick=place,tick
        try:
            self.kind="rim";env.exposed_rim_probes();self._finish()
            env.probe_positions["table"]=[1.313,-.625,.25]
            self.kind="contact";probes=env.probes();self._finish()
            write_new_json(env.dest/"probes.json",probes)
            ledger=DiagnosticLedger();ledger.observe(self.records,len(self.parking))
            require(ledger.complete,"Incomplete original startup diagnostics")
            result=dict(schema=SCHEMA,status="completed",records=self.records,evidence=self.evidence,
                        parking=self.parking,summary=ledger.summary(),
                        origins_m=_plain(env.origins),bucket_xy_m=_plain(env.bucket_xy),
                        foreign_probe_index=7,
                        helper_sha256=copy.deepcopy(STARTUP_HELPERS))
            result["fingerprint"]=canonical_hash(result)
            write_new_json(self.directory/"receipt.json",result)
            return result
        except BaseException as error:
            # Preserve observed samples even if the original helper throws.
            if self.current is not None:
                saved=copy.deepcopy(self.current)
                saved.pop("wall_begin",None);saved.pop("cpu_begin",None)
                write_new_json(self.directory/"incomplete-current.json",saved)
            write_new_json(self.directory/"failure.json",dict(status="incomplete_no_retry",error=repr(error)))
            raise
        finally:
            del env.set_balls;del env.tick


def audit_startup(directory, *, scene, cells, origins):
    """Independently replay actual startup placement/clock/contact evidence."""
    from candidate_native_audit import verify_controls
    from candidate_continuation_launcher import RIM_ANGLES
    directory=Path(directory)
    def read(name):
        path=directory/name
        require(Path(name).name==name and path.is_file() and not path.is_symlink(),"Unsafe startup evidence")
        return json.loads(path.read_bytes(),parse_constant=lambda _:(_ for _ in ()).throw(ValueError("Nonfinite startup JSON")))
    receipt=read("receipt.json")
    require(receipt.get("schema")==SCHEMA and receipt.get("status")=="completed"
        and receipt.get("fingerprint")==canonical_hash({k:v for k,v in receipt.items() if k!="fingerprint"})
        and receipt.get("helper_sha256")==STARTUP_HELPERS,"Startup receipt/source identity changed")
    origins=np.asarray(origins,dtype=float);xy=np.asarray([c["bucket_xy_m"] for c in cells],dtype=float)
    require(origins.shape==(35,3) and xy.shape==(35,2)
        and receipt["origins_m"]==origins.tolist() and receipt["bucket_xy_m"]==xy.tolist()
        and receipt["foreign_probe_index"]==7,"Startup cell/origin association changed")
    ledger=DiagnosticLedger();ledger.observe(receipt["records"],len(receipt["parking"]))
    require(ledger.complete and ledger.summary()==receipt["summary"],"Startup schedule/cost summary changed")
    require(len(receipt["evidence"])==len(receipt["records"]),"Startup cohort coverage changed")
    expected={"receipt.json"};previous_end=None;found=np.zeros(35,dtype=bool);directions=np.zeros((35,2));rays=0
    bucket=scene["bucket"];top=bucket["assumed_interior_depth_m"]+bucket["assumed_bottom_thickness_m"]
    for i,(record,entry) in enumerate(zip(receipt["records"],receipt["evidence"])):
        require(entry["file"]==f"cohort-{i:04d}.json" and digest(directory/entry["file"])==entry["sha256"],"Startup raw cohort bytes changed")
        raw=read(entry["file"]);expected.add(entry["file"])
        initial=read(raw["initialized_file"])
        require(raw["initialized_sha256"]==record["initialized_states_sha256"]
            and digest(directory/raw["initialized_file"])==raw["initialized_sha256"],"Startup initialized-state bytes changed")
        expected.add(raw["initialized_file"])
        forces=np.asarray(raw["forces_N"],dtype=float)
        require(forces.shape==(record["physics_steps"],35,6) and np.isfinite(forces).all() and (forces>=0).all()
            and type(raw["start_step"]) is int and type(raw["end_step"]) is int
            and raw["end_step"]-raw["start_step"]==len(forces)
            and initial["step"]==raw["start_step"]
            and (previous_end is None or previous_end==raw["start_step"]),"Startup physics clock/sample coverage changed")
        previous_end=raw["end_step"]
        require(raw["wall_s"]==record["wall_s"] and raw["process_cpu_s"]==record["process_cpu_s"],"Startup cost source changed")
        xyz=np.zeros((35,3));kind=record["kind"]
        if kind=="rim_ray":
            theta=np.deg2rad(RIM_ANGLES[rays]);direction=np.array([np.cos(theta),np.sin(theta)]);rays+=1
            xyz[:,:2]=xy+direction*bucket["inner_top_diameter_m"]/2
            xyz[:,2]=top+scene["ball"]["radius_m"]+.012
            first=np.zeros((35,6))
            for f in forces:
                touched=(first.max(axis=1)<=.01)&(f.max(axis=1)>.01);first[touched]=f[touched]
            clear=(first[:,1]>.01)&(first[:,[0,2,3,4,5]].max(axis=1)<=.01)
            directions[clear&~found]=direction;found|=clear
            passed=bool(found.all())
        elif kind in ("bucket","rim"):
            xyz[:,:2]=xy;xyz[:,2]=top+.15
            if kind=="rim":
                xyz[:,:2]+=directions*bucket["inner_top_diameter_m"]/2
                xyz[:,2]=top+scene["ball"]["radius_m"]+.012
            passed=bool((forces.max(axis=0)[:,1]>.01).all())
        elif kind in ("table","floor","robot"):
            xyz[:]=dict(table=[1.313,-.625,.25],floor=[1.8,-.65,-.5],robot=[0,0,.75])[kind]
            peaks=forces.max(axis=0);target=dict(table=2,floor=3,robot=4)[kind]
            passed=bool((peaks[:,target]>.01).all())
            if kind in ("table","floor"):passed=passed and bool((peaks[:,[0,1]]<=.01).all())
        else:
            xyz[:]=[.1,-.8,.04];xyz+=origins
            xyz[0]=origins[7]+[*xy[7],top+.15]
            passed=bool(forces[:,0].max()<=.01 and raw["final_ball_z_m"]<-1)
        if kind!="foreign_cell_collision_filter":xyz+=origins
        position=np.asarray(initial["position_m"],dtype=float);velocity=np.asarray(initial["velocity_m_s"],dtype=float)
        require(position.shape==velocity.shape==(35,3) and np.isfinite(position).all() and np.isfinite(velocity).all()
            and np.allclose(position,xyz,atol=1e-5,rtol=0) and np.array_equal(velocity,np.zeros((35,3)))
            and record["passed"] is passed,"Startup placement/contact gates changed")
    for i,entry in enumerate(receipt["parking"]):
        require(entry["file"]==f"parking-{i:04d}.json" and digest(directory/entry["file"])==entry["sha256"],"Parking bytes changed")
        raw=read(entry["file"]);expected.add(entry["file"])
        initial=read(raw["initialized_file"]);expected.add(raw["initialized_file"])
        require(digest(directory/raw["initialized_file"])==raw["initialized_sha256"]
            and raw["forces_N"]==[] and raw["start_step"]==raw["end_step"]==initial["step"]
            and np.allclose(initial["position_m"],origins+[.1,-.8,.04],atol=1e-5,rtol=0)
            and np.array_equal(initial["velocity_m_s"],np.zeros((35,3))),"Parking reset was a hidden probe")
    require({p.name for p in directory.iterdir()}==expected,"Unlisted or partial startup evidence")
    control_hashes={n:digest(directory.parent/n) for n in ("probes.json","rim-rays.json")}
    verify_controls(directory.parent,scene,control_hashes)
    return dict(status="passed",cohorts=len(receipt["records"]),initialized_trajectories=ledger.summary()["initialized_trajectories"],
        physics_steps=ledger.summary()["physics_steps"],receipt_sha256=digest(directory/"receipt.json"),
        startup_control_sha256=control_hashes,placements_steps_contacts_recomputed=True,
        scope="Original finite diagnostic placements/clocks/contact gates and cost-source joins; separate from task requests.")
