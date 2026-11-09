"""One application: finite startup, bounded TRAIN, final-only assessment, audit.

Imports perform no inference, optimization, native construction or process work.
The production components are created only by the explicitly authorized entry
point. Unit tests supply fake components; the orchestration itself has no retry.
"""

import copy
import hashlib
import json
import math
from pathlib import Path
import resource
import time

from candidate_continuation_contract import IMITATION_MAX_STEPS
from candidate_continuation_journal import write_new_json
from candidate_continuation_launcher import PHASES
from candidate_continuation_ledger import canonical_hash

SCHEMA="sircus-candidate-continuation-application-v1"
MANIFEST_SCHEMA="sircus-candidate-continuation-production-inputs-v1"


def require(value,message):
    if not value:raise ValueError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    def pairs(items):
        value={}
        for key,item in items:
            require(key not in value,"Duplicate application JSON key");value[key]=item
        return value
    return json.loads(Path(path).read_bytes(),object_pairs_hook=pairs,
        parse_constant=lambda _:(_ for _ in ()).throw(ValueError("Nonfinite application JSON")))


def load_manifest(path,expected_sha256,*,container_root=Path("/workspace")):
    """Verify exact raw bytes; map host paths without rewriting the manifest."""
    path=Path(path)
    require(sha(path)==expected_sha256,"Externally bound production manifest changed")
    value=read(path)
    require(value.get("schema")==MANIFEST_SCHEMA and value.get("execution_authorized") is False
        and value.get("release_authorized") is False,"Manifest is identity, not execution permission")
    host=Path(value["workspace_host"])
    require(host.is_absolute() and str(host)==str(host.resolve()),"Noncanonical workspace root")
    def mapped(raw):
        original=Path(raw)
        require(original.is_absolute() and ".." not in original.parts and original.is_relative_to(host),
                "Manifest path is outside the declared workspace")
        target=Path(container_root)/original.relative_to(host)
        require(target.resolve().is_relative_to(Path(container_root).resolve())
            and not any(p.is_symlink() for p in (target,*target.parents) if p.is_relative_to(container_root)),
            "Linked or escaped production input path")
        return target
    require(type(value.get("files")) is dict and value["files"],"Missing bound production inputs")
    paths={}
    for role,entry in value["files"].items():
        require(type(role) is str and set(entry)=={"path","sha256"},"Malformed production input")
        p=mapped(entry["path"])
        require(p.is_file() and not p.is_symlink() and sha(p)==entry["sha256"],"Production input bytes changed: "+role)
        paths[role]=p
    require(all(k in paths for k in ("parent_checkpoint","fresh_split_manifest","train_bank")),
            "Missing mandatory production input role")
    source=mapped(value["source_host"])/"sircus/isaac"
    actual=canonical_hash({p.name:sha(p) for p in sorted(source.glob("*.py"))})
    require(actual==value.get("source_sha256"),"Production source snapshot changed")
    return value,paths,mapped


def host_usage():
    children=resource.getrusage(resource.RUSAGE_CHILDREN)
    return dict(cpu_s=time.process_time()+children.ru_utime+children.ru_stime,
        lifetime_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        largest_child_lifetime_peak_rss_mib=children.ru_maxrss/1024)


class PhaseCosts:
    def __init__(self,*,clock=time.monotonic,usage=host_usage,device=lambda reset:None):
        self.clock,self.usage,self.device=clock,usage,device
        self.rows={};self.current=None

    def begin(self,name):
        require(name in PHASES and name not in self.rows and self.current is None,"Repeated/overlapping phase cost")
        self.current=(name,self.clock(),self.usage());self.device(True)

    def finish(self):
        require(self.current is not None,"No active resource phase")
        name,begin,before=self.current;after=self.usage()
        value=dict(wall_s=self.clock()-begin,aggregate_cpu_s=after["cpu_s"]-before["cpu_s"],
            lifetime_peak_rss_mib=after["lifetime_peak_rss_mib"],
            largest_child_lifetime_peak_rss_mib=after["largest_child_lifetime_peak_rss_mib"],
            device=self.device(False))
        require(all(type(value[k]) in (int,float) and math.isfinite(value[k]) and value[k]>=0
                    for k in ("wall_s","aggregate_cpu_s","lifetime_peak_rss_mib","largest_child_lifetime_peak_rss_mib")),
                "Invalid phase resource measurement")
        if value["device"] is not None:
            require(type(value["device"]) is dict and all(type(v) in (int,float) and math.isfinite(v) and v>=0
                for v in value["device"].values()),"Invalid phase device measurement")
        self.rows[name]=value;self.current=None
        return value

    def receipt(self):
        return dict(phases=copy.deepcopy(self.rows),scope="Non-overlapping entry-process phase wall and self plus waited/reaped-child CPU deltas; active unreaped child CPU is not yet included. Host-supervisor/watchdog CPU is separate. RSS is lifetime high-water, child RSS is largest individual child, not a summed phase peak. Torch allocation peaks include persistent scene tensors; sampled whole-device occupancy includes renderer/PhysX and foreign overhead, not model-only memory. Null device means no device sample, never zero inferred.")


def run_pipeline(output,session,components,publisher,*,costs=None):
    """Execute supplied components once. Tests provide only fabricated components."""
    output=Path(output);costs=costs or PhaseCosts(device=components.device_snapshot)
    phase="initialization";diagnostics=[];parking=0
    counts=dict(train_requests=0,train_releases=0,assessment_requests=0,assessment_releases=0)
    def publish(*,running=True,exit_code=None):
        publisher.publish(dict(phase=phase,diagnostics=copy.deepcopy(diagnostics),parking_resets=parking,
            **counts,running=running,exit_code=exit_code,audit_cpu_only=phase=="audit"))
    def startup_progress(records,resets):
        nonlocal diagnostics,parking
        diagnostics,parking=copy.deepcopy(records),resets;publish()
    def update(ledger):
        summary=ledger.summary()
        for role in ("train","assessment"):
            counts[role+"_requests"]=summary[role]["requested"]
            counts[role+"_releases"]=summary[role]["released"]
    def next_phase(name):
        nonlocal phase
        costs.finish();phase=name;costs.begin(name);publish()
    try:
        costs.begin(phase);publish()
        runner,producer,startup=components.initialize(startup_progress)
        diagnostics,parking=copy.deepcopy(startup["records"]),len(startup["parking"])
        next_phase("imitation")
        for _ in range(IMITATION_MAX_STEPS):
            runner.imitation_step();publish()
        runner.start_ppo();next_phase("train")
        while runner.ledger.next_batch() is not None:
            choices=runner.reserve_train_batch();update(runner.ledger);publish()
            results=producer.execute_reserved(runner,choices,expected_journal_head=runner.journal.head)
            runner.record_results(results);update(runner.ledger);publish()
        trained=runner.finish();update(runner.ledger);publish()
        next_phase("assessment")
        assessment,assessor=components.begin_assessment(runner,trained)
        while assessment.ledger.next_batch() is not None:
            choices=assessment.reserve_batch();update(assessment.ledger);publish()
            results=assessor.execute_reserved(assessment,choices,expected_journal_head=assessment.journal.head)
            assessment.record_results(results);update(assessment.ledger);publish()
        assessed=assessment.finish();update(assessment.ledger);publish()
        # The independent reader runs in a child with CUDA hidden, after app close.
        components.close_native();next_phase("audit")
        audit=components.audit(trained,assessed)
        require(audit.get("status")=="passed" and audit.get("all_components_passed") is True,
                "Combined independent audit did not pass; no promotion")
        costs.finish()
        result=dict(schema=SCHEMA,status="completed_audited",session=session,
            synthetic=components.synthetic,counters=counts,final_checkpoint_sha256=trained["checkpoint_sha256"],
            assessment_result_sha256=sha(assessment.directory/"assessment-result.json"),audit=audit,resources=costs.receipt(),
            preparation=components.preparation_evidence() if hasattr(components,"preparation_evidence") else {},
            experience=dict(inherited_parent_requests=10000,inherited_parent_updates=73,
                prior_frozen_selector_comparison_requests=70,new_learning_counters=trained.get("counters"),
                scope="Extra-budget continuation of one seed; not equal-cost superiority or uncertainty over independent training seeds."),
            native_application_starts=1,model_promoted=False,execution_authorized=False,release_authorized=False)
        result["fingerprint"]=canonical_hash(result)
        write_new_json(output/"application-result.json",result)
        publish(running=False,exit_code=0)
        return result
    except BaseException as error:
        abort_error=None
        try:components.abort(error)
        except BaseException as aborted:abort_error=repr(aborted)
        if costs.current is not None:
            try:costs.finish()
            except Exception:pass
        write_new_json(output/"application-failure.json",dict(schema=SCHEMA,status="incomplete_no_retry",
            session=session,phase=phase,counters=counts,error=repr(error),abort_error=abort_error,
            resources=costs.receipt(),model_promoted=False))
        raise
    finally:
        components.close_native()
