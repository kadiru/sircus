"""One explicitly authorized application: startup, untouched TEST, CPU audit.

No simulation, policy or process construction occurs on import. Tests supply
synthetic components. Production uses a separately owned watchdog and never
constructs a trainer, resumes the old failed ledger or repeats a reservation.
"""

import copy
import math
from pathlib import Path
import resource
import shutil
import sys
import time

from candidate_continuation_journal import Journal, write_new_json, _sync_directory
from candidate_continuation_ledger import canonical_hash
from candidate_v4_evaluation import validate_prepared, protocol

SCHEMA="sircus-v4-test-application-v1"
RESULT_FIELDS={"request_id","released","decline_reason","captured","clean","source_sha256","return"}


def require(value,message):
    if not value:raise ValueError(message)


def usage():
    children=resource.getrusage(resource.RUSAGE_CHILDREN)
    divisor=1024**2 if sys.platform=="darwin" else 1024
    return dict(cpu_s=time.process_time()+children.ru_utime+children.ru_stime,
        lifetime_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/divisor,
        largest_child_lifetime_peak_rss_mib=children.ru_maxrss/divisor)


class PhaseCosts:
    def __init__(self,*,clock=time.monotonic,measure=usage,device=lambda reset:None):
        self.clock,self.measure,self.device=clock,measure,device;self.rows={};self.current=None
    def begin(self,name):
        require(name in protocol()["phases"] and self.current is None and name not in self.rows,"Repeated or overlapping evaluation phase")
        self.current=(name,self.clock(),self.measure());self.device(True)
    def finish(self):
        require(self.current is not None,"No active evaluation phase")
        name,began,before=self.current;after=self.measure()
        row=dict(wall_s=self.clock()-began,aggregate_cpu_s=after["cpu_s"]-before["cpu_s"],
            lifetime_peak_rss_mib=after["lifetime_peak_rss_mib"],largest_child_lifetime_peak_rss_mib=after["largest_child_lifetime_peak_rss_mib"],
            device=self.device(False))
        require(all(type(row[k]) in (int,float) and math.isfinite(row[k]) and row[k]>=0
            for k in ("wall_s","aggregate_cpu_s","lifetime_peak_rss_mib","largest_child_lifetime_peak_rss_mib")),"Invalid evaluation resource measurement")
        require(row["device"] is None or type(row["device"]) is dict and all(type(v) in (int,float) and math.isfinite(v) and v>=0
            for v in row["device"].values()),"Invalid evaluation device measurement")
        self.rows[name]=row;self.current=None
    def receipt(self):
        return dict(phases=copy.deepcopy(self.rows),scope="Non-overlapping new application phase wall and self plus waited/reaped-child CPU deltas. Host supervisor/watchdog costs are separate; active unreaped child CPU is not included yet. Audit child CPU overlaps its application audit phase. RSS is process-lifetime high-water and largest single child, not isolated or summed phase peaks. GPU occupancy includes PhysX/renderer; Torch allocation is not model-only memory. Inherited V4 TRAIN and failed-attempt costs are not added to these phases.")


def run_application(output,plan,components,publisher,*,approved=False,costs=None,clock=time.monotonic,free_bytes=None):
    """Execute supplied evaluation components exactly once; mocks default off."""
    plan=validate_prepared(plan);synthetic=components.synthetic
    require(type(synthetic) is bool and plan["synthetic"] is synthetic,"Application provenance differs from prepared plan")
    require(type(approved) is bool and (approved or synthetic),"Separate execution approval required before native application")
    require(not any(hasattr(components,key) for key in ("train","optimizer","learning_engine","imitation_step","start_ppo")),
        "Evaluation components cannot expose a training API")
    output=Path(output)
    require(output.is_absolute() and output.resolve()==output and output.is_dir() and not output.is_symlink(),
        "Supervisor-owned new output directory required")
    for name in ("evaluation-plan.json","journal","native","application-result.json","application-failure.json"):
        require(not (output/name).exists(),"Evaluation output already used; no retry or resume")
    frozen=dict(final_checkpoint_sha256=plan["final_checkpoint_sha256"],parent_checkpoint_sha256=plan["parent_checkpoint_sha256"],
        bindings=copy.deepcopy(plan["bindings"]),model_hashes=copy.deepcopy(plan["model_hashes"]))
    plan_hash=canonical_hash(plan)
    limits=protocol();began=last=phase_start=clock();phase="initialization"
    require(type(began) in (int,float) and math.isfinite(began),"Invalid initial evaluation clock")
    counts=dict(requested=0,terminal=0,released=0,rejected=0,captures=0,clean_captures=0,unresolved=0)
    diagnostics=[];parking=0;pending=None
    free_bytes=free_bytes or (lambda:shutil.disk_usage(output).free)
    costs=costs or PhaseCosts(clock=clock,device=components.device_snapshot)
    def guard():
        nonlocal last
        require(canonical_hash(plan)==plan_hash,"Frozen in-memory TEST plan changed")
        now=clock();require(type(now) in (int,float) and math.isfinite(now) and now>=last,"Evaluation monotonic clock invalid or moved backwards")
        last=now
        require(now-began<=limits["wall_limit_s"] and now-phase_start<=limits["phases"][phase],"Evaluation application or phase deadline reached")
        free=free_bytes();require(type(free) is int and free>=limits["min_free_bytes"],"Original 6 GiB storage floor reached")
        require(components.verify_frozen()==frozen,"Frozen model, checkpoint, source or input bytes changed")
    def publish(*,running=True,exit_code=None):
        publisher.publish(dict(phase=phase,diagnostics=copy.deepcopy(diagnostics),parking_resets=parking,
            evaluation_requests=counts["requested"],evaluation_terminal=counts["terminal"],evaluation_releases=counts["released"],
            train_requests=0,optimizer_steps=0,running=running,exit_code=exit_code,audit_cpu_only=phase=="audit"))
    def startup_progress(records,resets):
        nonlocal diagnostics,parking
        diagnostics,parking=copy.deepcopy(records),resets;guard();publish()
    def next_phase(name):
        nonlocal phase,phase_start
        guard();costs.finish();phase=name;phase_start=clock();costs.begin(name);guard();publish()
    def record(cohort,rows):
        nonlocal pending,counts
        require(pending is not None and type(rows) in (tuple,list) and 0<len(rows)<=len(pending),"Expected ordered terminal TEST prefix")
        checked=[]
        for choice,row in zip(pending,rows):
            require(type(row) is dict and set(row)==RESULT_FIELDS and row["request_id"]==choice["request_id"],"TEST outcome identity or order changed")
            require(all(type(row[k]) is bool for k in ("released","captured","clean"))
                and (not row["clean"] or row["captured"]) and (not row["captured"] or row["released"]),"Contradictory TEST outcome flags")
            require((row["released"] and row["decline_reason"] is None) or (not row["released"] and type(row["decline_reason"]) is str
                and row["decline_reason"].strip()),"Invalid TEST decline reason")
            value=row["return"]
            require(type(value) in (int,float) and math.isfinite(value) and ((not row["released"] and value==-1.)
                or (row["captured"] and value==(1.6 if row["clean"] else 1.5))
                or (row["released"] and not row["captured"] and -.8<=value<=0.)),"TEST reward rules changed")
            require(type(row["source_sha256"]) is str and len(row["source_sha256"])==64
                and all(c in "0123456789abcdef" for c in row["source_sha256"]),"Missing native TEST source digest")
            require(choice["enabled"] or (not row["released"] and row["decline_reason"]=="no_qualified_candidate_support"),"Empty TEST support was replaced or released")
            checked.append(copy.deepcopy(row))
        updated=copy.deepcopy(counts)
        for row in checked:
            updated["terminal"]+=1;updated["released"]+=int(row["released"]);updated["rejected"]+=int(not row["released"])
            updated["captures"]+=int(row["captured"]);updated["clean_captures"]+=int(row["clean"])
        updated["unresolved"]=updated["requested"]-updated["terminal"]
        journal.append("outcomes",dict(cohort=cohort["index"],results=checked,requested=updated["requested"],terminal=updated["terminal"]))
        counts=updated;pending=pending[len(checked):] or None;publish()
    # Frozen inputs are checked before creating any application evidence or App.
    guard();write_new_json(output/"evaluation-plan.json",plan)
    journal=Journal(output/"journal")
    journal.append("start",dict(plan_fingerprint=plan["fingerprint"],synthetic=synthetic,mock_only=synthetic,frozen=frozen))
    try:
        costs.begin(phase);publish()
        startup=components.initialize(startup_progress)
        diagnostics,parking=copy.deepcopy(startup["records"]),len(startup["parking"])
        guard();publish();next_phase("evaluation")
        for cohort in plan["cohorts"]:
            guard();require(pending is None and counts["requested"]+35<=210,"Pending TEST requests cannot be replaced")
            choices=copy.deepcopy(cohort["choices"])
            requested=counts["requested"]+35
            event=journal.append("reserve",dict(cohort=cohort["index"],original_cohort=cohort["original_cohort"],choices=choices,
                requested=requested,terminal=counts["terminal"],plan_fingerprint=plan["fingerprint"]))
            pending=choices;counts["requested"]=requested;counts["unresolved"]=requested-counts["terminal"];publish()
            reservation=dict(journal_head_sha256=event["fingerprint"],cohort=cohort["index"],original_cohort=cohort["original_cohort"],
                requested=requested,request_ids=[c["request_id"] for c in choices])
            try:rows=components.execute_reserved(copy.deepcopy(cohort),copy.deepcopy(choices),copy.deepcopy(reservation),journal_directory=journal.directory)
            except BaseException as error:
                prefix=getattr(error,"terminal_prefix",None)
                if prefix is not None:record(cohort,prefix)
                raise
            record(cohort,rows)
            require(pending is None,"Partial TEST cohort preserved; no replacement or retry")
            guard()
        require(counts["requested"]==counts["terminal"]==210 and counts["unresolved"]==0,"Incomplete TEST-only budget")
        components.close_native();guard()
        native_calls=0 if synthetic else 210
        journal.append("final",dict(plan_fingerprint=plan["fingerprint"],frozen=frozen,counts=copy.deepcopy(counts),
            synthetic=synthetic,mock_only=synthetic,training_requests=0,optimizer_steps=0,native_requests_executed=native_calls))
        summary=dict(schema="sircus-v4-test-evaluation-result-v1",status="synthetic_complete_audit_pending" if synthetic else "native_complete_audit_pending",plan_fingerprint=plan["fingerprint"],
            journal_head_sha256=journal.head,frozen=frozen,counts=copy.deepcopy(counts),synthetic=synthetic,mock_only=synthetic,
            training_requests=0,optimizer_steps=0,native_requests_executed=native_calls,
            original_assessment_completed=False,model_promoted=False,execution_authorized=False,release_authorized=False)
        next_phase("audit")
        summary["resources"]=costs.receipt();summary["fingerprint"]=canonical_hash(summary)
        write_new_json(output/"evaluation-result.json",summary)
        audit=components.audit(copy.deepcopy(summary));guard()
        require(type(audit) is dict and audit.get("status")== ("passed_synthetic" if synthetic else "passed")
            and audit.get("all_components_passed") is True and audit.get("synthetic") is synthetic
            and audit.get("plan_fingerprint")==plan["fingerprint"]
            and audit.get("journal_head_sha256")==journal.head
            and audit.get("fingerprint")==canonical_hash({k:v for k,v in audit.items() if k!="fingerprint"}),
            "Independent TEST model/startup/physics audit did not pass")
        costs.finish()
        result=dict(summary,schema=SCHEMA,status="synthetic_application_complete" if synthetic else "completed_audited",
            audit=audit,resources=costs.receipt(),native_application_starts=0 if synthetic else 1,
            original_assessment_completed=False,model_promoted=False,execution_authorized=False,release_authorized=False,
            scope="Separate 210-request untouched TEST evaluation of frozen V4; old assessment remains incomplete. Synthetic runs establish software behavior only.")
        result.pop("fingerprint",None);result["fingerprint"]=canonical_hash(result)
        write_new_json(output/"application-result.json",result);publish(running=False,exit_code=0)
        return result
    except BaseException as error:
        close_error=None
        try:components.close_native()
        except BaseException as problem:close_error=repr(problem)
        if costs.current is not None:
            try:costs.finish()
            except Exception:pass
        failure=dict(schema=SCHEMA,status="incomplete_no_retry",synthetic=synthetic,phase=phase,error=repr(error),
            close_error=close_error,plan_fingerprint=plan["fingerprint"],counts=copy.deepcopy(counts),resources=costs.receipt(),
            training_requests=0,optimizer_steps=0,original_assessment_completed=False,model_promoted=False,
            execution_authorized=False,release_authorized=False)
        journal.append("failure",copy.deepcopy(failure));write_new_json(output/"application-failure.json",failure)
        publish(running=False,exit_code=1)
        raise
    finally:
        components.close_native()
