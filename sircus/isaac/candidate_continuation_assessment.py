"""Final-only frozen three-method assessment; no optimizer or launch path.

The pure planner fixes all 420 actions before any assessment outcome. The
durable runner charges requests before exposing them and never resumes/retries.
Actual native dispatch is a separate explicit adapter, on the same application.
"""

import copy
import hashlib
import math
from pathlib import Path
import shutil
import time

import numpy as np

from candidate_continuation_journal import Journal, write_new_json, _sync_directory
from candidate_continuation_ledger import Ledger, Outcome, ASSESSMENT_COHORTS, canonical_hash, PARENT_CHECKPOINT_SHA256
from candidate_continuation_native import source_fingerprint
from candidate_continuation_policy import FrozenCandidateSupport, TIE_TOLERANCE
from continuous_feasibility import ACTION_NAMES, LOW, HIGH, decode_action, canonical_hash as geometry_hash

PLAN_SCHEMA = "sircus-candidate-continuation-assessment-plan-v1"
RESULT_SCHEMA = "sircus-candidate-continuation-assessment-result-v1"
RESULT_FIELDS = {"request_id", "released", "decline_reason", "source_sha256", "return", "captured", "clean"}
MIN_FREE_BYTES = 6*1024**3
ASSESSMENT_SECONDS = 1800


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _sha(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _gaussian(value):
    mean, scale = np.asarray(value["latent_mean"],float), np.asarray(value["latent_scale"],float)
    require(mean.shape == scale.shape == (5,) and np.isfinite(mean).all()
        and np.isfinite(scale).all() and (scale > 0).all(), "Invalid frozen assessment Gaussian")
    return mean, scale


def deterministic_index(support, gaussian):
    """Same fixed-latent Gaussian scores/tie law as categorical continuation."""
    require(type(support) is FrozenCandidateSupport, "Expected immutable assessment support")
    mean, scale = _gaussian(gaussian)
    latent = np.asarray(support.latents,dtype=float)
    scores = -.5*np.square((latent-mean)/scale).sum(axis=1)-np.log(scale).sum()-2.5*math.log(2*math.pi)
    require(np.isfinite(scores).all(), "Nonfinite assessment preference")
    tied = np.flatnonzero(scores >= scores.max()-TIE_TOLERANCE).tolist()
    index = support.parent_index if support.parent_index in tied else min(tied,key=lambda i:support.candidate_ids[i])
    return int(index), scores.tolist()


def predict_distributions(policy, cells):
    """One forward-only deterministic export, including inherited radius prior.

    Called only by an explicitly authorized future entry point; imports do no
    inference. Tests use fake snapshots instead of a real model.
    """
    from continuous_policy import torch, build_observations, physical_actions, weight_hash
    require(torch is not None, "Configured Torch required for frozen inference")
    before = weight_hash(policy)
    rng = torch.get_rng_state().clone()
    device = next(policy.parameters()).device
    with torch.no_grad():
        normal = policy.distribution(torch.tensor(build_observations(cells),device=device))
        values = physical_actions(normal.loc.tanh(),LOW,HIGH)
        result = {cell["id"]:dict(exact_action=decode_action(cell,row,"parent-"+cell["id"]),
            latent_mean=mean.detach().cpu().tolist(), latent_scale=scale.detach().cpu().tolist())
            for cell,row,mean,scale in zip(cells,values,normal.loc,normal.scale)}
    require(weight_hash(policy) == before and torch.equal(rng,torch.get_rng_state()),
            "Frozen assessment inference changed weights or CPU RNG")
    return result


def build_assessment_plan(train_ledger, *, final_checkpoint_sha256, cells_by_split, supports,
                          prepared_records, parent_distributions, continued_distributions,
                          model_hashes, bindings):
    """Freeze the final artifact and exact 12-cohort action schedule, no outcomes."""
    require(type(train_ledger) is Ledger and train_ledger.phase == "train"
        and train_ledger.pending is None and train_ledger.stopped_reason is None
        and train_ledger.next_batch() is None, "Assessment requires completed capped TRAIN history")
    require(_sha(final_checkpoint_sha256) and type(bindings) is dict
        and all(_sha(v) for v in bindings.values()) and bindings.get("source_sha256") == source_fingerprint(),
        "Assessment source/checkpoint bindings changed")
    require(set(model_hashes) == {"parent_weight_hash","continued_weight_hash"}
        and all(_sha(v) for v in model_hashes.values()), "Missing frozen method model identities")
    counts = dict(train=35,validation=35,test=70)
    require(set(cells_by_split) == set(counts), "Assessment split coverage changed")
    cells = {}
    for split,count in counts.items():
        rows = cells_by_split[split]
        expected = {"train":train_ledger.train_context_ids,"validation":train_ledger.validation_context_ids,
                    "test":train_ledger.test_context_ids}[split]
        require(len(rows) == count and tuple(c["id"] for c in rows) == expected
            and all(c.get("split") == split for c in rows), "Assessment split/context order changed")
        cells.update({c["id"]:c for c in rows})
    require(len(cells) == 140 and set(supports) == set(parent_distributions) == set(continued_distributions) == set(cells),
            "Assessment inputs must cover exactly 140 fixed positions")
    for key,support in supports.items():
        require(support is None or type(support) is FrozenCandidateSupport and support.context_id == key,
                "Assessment support/context mismatch")
        _gaussian(parent_distributions[key]); _gaussian(continued_distributions[key])
    assessment = train_ledger.begin_assessment(final_checkpoint_sha256)
    requests = assessment._requests("assessment",0,420)
    choices,cohorts = [],[]
    for cohort,(split,offset,method) in enumerate(ASSESSMENT_COHORTS):
        local = []
        for request in requests[cohort*35:(cohort+1)*35]:
            key = request.context_id; support = supports[key]
            parent = copy.deepcopy(parent_distributions[key]["exact_action"])
            require(all(type(parent.get(k)) in (int,float) and math.isfinite(parent[k]) for k in ACTION_NAMES),
                    "Missing original direct-parent action")
            vector = np.asarray([parent[k] for k in ACTION_NAMES])
            require((vector >= LOW).all() and (vector <= HIGH).all(), "Direct-parent action outside original domain")
            candidate_id,index,seed,scores = None,None,None,None
            enabled,decline,action = True,None,parent
            if method != "parent":
                if support is None:
                    enabled,decline = False,"no_qualified_candidate_support"
                else:
                    gaussian = parent_distributions[key] if method == "candidate_bank" else continued_distributions[key]
                    index,scores = deterministic_index(support,gaussian)
                    candidate_id,action = support.candidate_ids[index],support.action(index)
                    record = prepared_records[candidate_id]
                    decision=record["decision"]
                    require(record["cell_id"] == key and record["candidate_id"] == candidate_id
                        and record["action"] == action and record["accepted"] is True
                        and decision["accepted"] is True and decision["release_authorized"] is False
                        and decision.get("action") == action
                        and decision.get("binding",{}).get("context",{}).get("id") == key
                        and decision["qualification"].get("accepted") is True
                        and decision.get("path",{}).get("accepted") is True
                        and decision.get("fingerprint") == geometry_hash({k:v for k,v in decision.items() if k!="fingerprint"})
                        and set(decision["qualification"]["ik"]) == {"Arm10","Arm14"}
                        and all(fit["solved"] is True and len(fit["q"])==6 and np.isfinite(fit["q"]).all()
                                for fit in decision["qualification"]["ik"].values()),
                        "Assessment candidate lacks unchanged prepared qualification")
                    seed = None if index == support.parent_index else copy.deepcopy(record["decision"]["qualification"]["ik"])
            choice = dict(request_id=request.request_id,request_index=request.request_index,
                context_id=key,split=split,method=method,cohort=cohort,action=action,
                candidate_id=candidate_id,index=index,support_fingerprint=None if support is None else support.support_fingerprint,
                latent=None if index is None else list(support.latents[index]),scores=scores,
                enabled=enabled,decline_reason=decline,ik_seeds=seed)
            choices.append(choice);local.append(choice)
        cohorts.append(dict(index=cohort,split=split,offset=offset,method=method,
            request_ids=[c["request_id"] for c in local],choice_sha256=canonical_hash(local)))
    value = dict(schema=PLAN_SCHEMA,train_ledger_fingerprint=train_ledger.to_dict()["fingerprint"],
        initial_assessment_ledger_fingerprint=assessment.to_dict()["fingerprint"],
        final_checkpoint_sha256=final_checkpoint_sha256,parent_checkpoint_sha256=PARENT_CHECKPOINT_SHA256,
        model_hashes=copy.deepcopy(model_hashes),bindings=copy.deepcopy(bindings),
        cells_by_split=copy.deepcopy(cells_by_split),choices=choices,cohorts=cohorts,
        supports_sha256=canonical_hash({k:None if v is None else v.to_dict() for k,v in supports.items()}),
        prepared_records_sha256=canonical_hash(prepared_records),
        parent_distributions_sha256=canonical_hash(parent_distributions),
        continued_distributions_sha256=canonical_hash(continued_distributions),
        requests=420,optimizer_updates=0,held_out_outcomes_observed=0,
        execution_authorized=False,release_authorized=False)
    value["fingerprint"] = canonical_hash(value)
    return value


class AssessmentRunner:
    """New-only, final-frozen assessment ledger; no optimizer is accepted."""

    def __init__(self,directory,train_ledger,plan,*,parent_checkpoint_path,final_checkpoint_path,
                 state_hashes,synthetic,clock=time.monotonic,free_bytes=None):
        require(type(synthetic) is bool and type(train_ledger) is Ledger, "Explicit assessment provenance required")
        require(plan.get("schema") == PLAN_SCHEMA and plan.get("fingerprint") == canonical_hash({k:v for k,v in plan.items() if k != "fingerprint"})
            and plan.get("train_ledger_fingerprint") == train_ledger.to_dict()["fingerprint"]
            and plan.get("requests") == 420 and plan.get("optimizer_updates") == 0,
            "Assessment plan changed")
        self.plan = copy.deepcopy(plan);self.directory=Path(directory)
        self.parent_checkpoint_path,self.final_checkpoint_path=Path(parent_checkpoint_path),Path(final_checkpoint_path)
        self.state_hashes,self.clock=state_hashes,clock
        self.free_bytes=free_bytes or (lambda:shutil.disk_usage(self.directory.parent).free)
        self.synthetic=synthetic;self.failed=self.finished=False
        self.started=self.clock();self.last_clock=self.started;self.cpu_started=time.process_time()
        self.parent_bytes=digest(self.parent_checkpoint_path)
        require(synthetic or self.parent_bytes == PARENT_CHECKPOINT_SHA256,"Frozen parent checkpoint bytes changed")
        require(synthetic or train_ledger.bindings.complete,"Real assessment requires complete preparation/source bindings")
        require(synthetic or plan["model_hashes"]["parent_weight_hash"] ==
            "ea9a69ec063e3d71cda2b92f1da64939f644244d4b4dbd70b500be132632e549",
            "Real assessment did not use frozen seed-22 weights")
        self.ledger=train_ledger.begin_assessment(plan["final_checkpoint_sha256"])
        require(self.ledger.to_dict()["fingerprint"] == plan["initial_assessment_ledger_fingerprint"],"Initial assessment ledger changed")
        self._guard()
        self.directory.mkdir();_sync_directory(self.directory.parent)
        write_new_json(self.directory/"assessment-plan.json",self.plan)
        self.journal=Journal(self.directory/"journal")
        self.journal.append("start",dict(plan_fingerprint=plan["fingerprint"],bindings=plan["bindings"],
            synthetic=synthetic,ledger=self.ledger.to_dict(),model_hashes=plan["model_hashes"],
            parent_checkpoint_sha256=self.parent_bytes,final_checkpoint_sha256=plan["final_checkpoint_sha256"]))

    def _guard(self):
        require(not self.failed and not self.finished,"Assessment stopped; no resume/retry")
        now=self.clock()
        require(type(now) in (int,float) and math.isfinite(now) and now >= self.last_clock,
                "Assessment monotonic clock is invalid or moved backwards")
        self.last_clock=now;elapsed=now-self.started
        require(0 <= elapsed <= ASSESSMENT_SECONDS,
                "Assessment wall limit reached or clock invalid")
        free=self.free_bytes()
        require(type(free) is int and free >= MIN_FREE_BYTES,"Original 6 GiB storage floor reached")
        require(digest(self.parent_checkpoint_path)==self.parent_bytes
            and digest(self.final_checkpoint_path)==self.plan["final_checkpoint_sha256"]
            and self.state_hashes()==self.plan["model_hashes"],"Frozen assessment checkpoint/model changed")
        require(self.plan["bindings"]["source_sha256"]==source_fingerprint(),"Assessment source changed")

    def _fail(self,error):
        if self.failed or self.finished:return
        self.failed=True
        self.ledger=self.ledger.halt(type(error).__name__+": "+str(error))
        if hasattr(self,"journal"):
            self.journal.append("failure",dict(reason=str(error),ledger=self.ledger.to_dict()))

    def reserve_batch(self):
        try:
            self._guard();batch=self.ledger.next_batch()
            require(batch is not None,"Assessment already at its fixed request cap")
            ledger=self.ledger.reserve_batch(batch)
            start=batch.requests[0].request_index
            choices=copy.deepcopy(self.plan["choices"][start:start+len(batch.requests)])
            require([c["request_id"] for c in choices]==[r.request_id for r in batch.requests],"Assessment plan order changed")
            self.journal.append("reserve",dict(ledger=ledger.to_dict(),choices=choices,
                plan_fingerprint=self.plan["fingerprint"],model_hashes=self.plan["model_hashes"]))
            self.ledger=ledger
            return choices
        except BaseException as error:self._fail(error);raise

    def record_results(self,results):
        try:
            self._guard();pending=self.ledger.pending
            require(pending is not None and type(results) in (tuple,list) and results,"Missing assessment outcome prefix")
            remaining=pending.requests[len(pending.outcomes):]
            require(len(results)<=len(remaining),"Assessment outcome count exceeded reservation")
            outcomes=[]
            for row,request in zip(results,remaining):
                require(type(row) is dict and set(row)==RESULT_FIELDS and row["request_id"]==request.request_id,
                        "Assessment outcome identity/order changed")
                require(all(type(row[k]) is bool for k in ("released","captured","clean"))
                    and (not row["clean"] or row["captured"]) and (not row["captured"] or row["released"]),
                    "Assessment outcome flags contradict release")
                value=row["return"]
                require(type(value) in (int,float) and math.isfinite(value) and _sha(row["source_sha256"]),"Invalid assessment source/reward")
                require((not row["released"] and value == -1.) or (row["captured"] and value == (1.6 if row["clean"] else 1.5))
                    or (row["released"] and not row["captured"] and -.8 <= value <= 0.),"Assessment reward changed")
                choice=self.plan["choices"][request.request_index]
                require(choice["enabled"] or (not row["released"] and row["decline_reason"]==choice["decline_reason"]),
                        "Empty support was replaced or released")
                outcomes.append(Outcome(row["request_id"],row["released"],row["decline_reason"],row["source_sha256"]))
            ledger=self.ledger.record_outcomes(tuple(outcomes))
            self.journal.append("outcomes",dict(ledger=ledger.to_dict(),results=copy.deepcopy(list(results))))
            self.ledger=ledger;return self.ledger.summary()
        except BaseException as error:self._fail(error);raise

    def finish(self):
        try:
            self._guard()
            require(self.ledger.summary()["assessment_complete"] and self.ledger.pending is None,
                    "Incomplete assessment cannot be promoted")
            self.journal.append("final",dict(ledger=self.ledger.to_dict(),plan_fingerprint=self.plan["fingerprint"],
                model_hashes=self.plan["model_hashes"],final_checkpoint_sha256=self.plan["final_checkpoint_sha256"]))
            result=dict(schema=RESULT_SCHEMA,status="assessment_complete_audit_pending",synthetic=self.synthetic,
                plan_fingerprint=self.plan["fingerprint"],bindings=self.plan["bindings"],
                journal_head_sha256=self.journal.head,ledger=self.ledger.to_dict(),
                final_checkpoint_sha256=self.plan["final_checkpoint_sha256"],model_hashes=self.plan["model_hashes"],
                requests=420,optimizer_updates=0,wall_s=self.clock()-self.started,process_cpu_s=time.process_time()-self.cpu_started,
                execution_authorized=False,release_authorized=False,model_promoted=False)
            result["fingerprint"]=canonical_hash(result)
            write_new_json(self.directory/"assessment-result.json",result);self.finished=True
            return copy.deepcopy(result)
        except BaseException as error:self._fail(error);raise
