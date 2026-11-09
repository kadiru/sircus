"""Independent proposal/FK/path audit of an outcome-blind held-out bank.

This reader reconstructs the raw proposal union and every reached geometry gate.
It performs no IK optimization, training, ball simulation or native dispatch.
"""

from collections import Counter
import copy
import math

import numpy as np
from scipy.stats import qmc

from candidate_continuation_policy import FrozenCandidateSupport
from candidate_fresh_bank import (VERSION,QUALIFICATION_VERSION,SEED,RAW_PROPOSALS,
    source_hashes,validate_plan)
from candidate_fresh_splits import cells_from_splits
from continuous_feasibility import ACTION_NAMES,LOW,HIGH,IK_PROTOCOLS,canonical_hash,decode_action
from feasible_candidates import plain,signed,unsigned,by_cell,exact_action,values_of
from feasible_candidates_audit import audit_decision


def require(value,message):
    if not value:
        raise ValueError(message)


def audit_fresh_bank(plan,result,splits,scene,links,common_starts,*,parent_actions,
                     parent_latent_means,parent_latent_scales,bindings):
    """Reconstruct all aliases and qualify evidence without solving fresh IK."""
    value=validate_plan(plan)
    unsigned(result)
    cells=cells_from_splits(splits)
    require(value["cells"]==cells and value["split_manifest"]==splits
        and value["split_fingerprint"]==splits["fingerprint"],
            "Bank relabelled or replaced fresh coordinates")
    require(value["input_bindings"]==bindings,"External parent/source bindings changed")
    parent=by_cell(parent_actions,cells,"Audit frozen parent actions")
    means=by_cell(parent_latent_means,cells,"Audit frozen parent means")
    scales=by_cell(parent_latent_scales,cells,"Audit frozen parent scales")
    expected,distributions={},{}

    def add(cell,action,kind,index):
        vector=[float(action[k]) for k in ACTION_NAMES]
        identifier=canonical_hash(dict(cell_id=cell["id"],values=vector))
        if identifier not in expected:
            expected[identifier]=dict(candidate_id=identifier,cell_id=cell["id"],action=copy.deepcopy(action),values=vector,sources=[])
        expected[identifier]["sources"].append(dict(kind=kind,index=index,
            original_action_id=action["id"],physical_sha256=canonical_hash(vector)))

    for i,cell in enumerate(cells):
        name=cell["id"]
        mean,scale=np.asarray(means[name],float),np.asarray(scales[name],float)
        require(mean.shape==scale.shape==(5,) and np.isfinite(mean).all()
            and np.isfinite(scale).all() and (scale>0).all(),"Invalid audit parent distribution")
        action=exact_action(cell,parent[name],"parent")
        distributions[name]=dict(latent_mean=mean.tolist(),latent_scale=scale.tolist(),exact_action=action)
        add(cell,action,"parent",0)
        generator=np.random.default_rng(SEED+i)
        samples=generator.standard_normal((16,5))*scale+mean
        for index,latent in enumerate(samples):
            vector=LOW+(np.tanh(latent)+1)*(HIGH-LOW)/2
            add(cell,decode_action(cell,vector,f"nearby-{name}-{index:04d}"),"nearby",index)
        sobol=qmc.Sobol(d=5,scramble=True,seed=SEED+100000+i).random_base2(6)
        for index,unit in enumerate(sobol):
            add(cell,decode_action(cell,LOW+unit*(HIGH-LOW),f"global-{name}-{index:04d}"),"global",index)
    require(value["parent_distributions"]==distributions,"Frozen parent mean/scale/action changed")
    require(value["records"]==list(expected.values()),"Independent raw proposal reconstruction differs")
    require(sum(len(r["sources"]) for r in expected.values())==RAW_PROPOSALS,
            "Raw held-out proposal denominator changed")
    require(result.get("schema")==QUALIFICATION_VERSION
        and result.get("plan_fingerprint")==plan["fingerprint"]
        and result.get("source_sha256")==source_hashes()
        and result.get("scene_sha256")==canonical_hash(plain(scene))
        and result.get("common_starts")==plain(common_starts),"Qualification scene/start/source binding changed")
    require(type(result.get("workers")) is int and 1<=result["workers"]<=4
        and type(result.get("custom_test_solver")) is bool
        and all(type(result.get(name)) in (int,float) and math.isfinite(result[name])
            for name in ("wall_limit_s","cpu_limit_s"))
        and 0<result["wall_limit_s"]<=1800 and 0<result["cpu_limit_s"]<=7200,
        "Qualification limits changed")
    cost=result["cost"]
    require(all(type(cost.get(name)) is int for name in ("completed","total","admitted","workers"))
        and all(type(cost.get(name)) in (int,float) and math.isfinite(cost[name])
            for name in ("wall_s","aggregate_cpu_s","monitor_poll_s"))
        and cost["status"]=="qualified" and cost["completed"]==cost["total"]==len(expected)
        and cost["workers"]==result["workers"] and cost["monitor_poll_s"]==.05
        and 0<=cost["wall_s"]<=result["wall_limit_s"]
        and 0<=cost["aggregate_cpu_s"]<=result["cpu_limit_s"],"Incomplete or over-budget qualification")
    require(all(type(result.get(name)) is int for name in ("raw_proposals","unique_candidates","admitted"))
        and len(result["records"])==len(expected) and result["raw_proposals"]==RAW_PROPOSALS
        and result["unique_candidates"]==len(expected),"Qualification candidate coverage changed")
    by_cell_id={c["id"]:c for c in cells}
    accepted=[]
    for original,row in zip(expected.values(),result["records"]):
        require(all(row.get(k)==original[k] for k in ("candidate_id","cell_id","action","sources"))
            and type(row["preflight_calls"]) is int and row["preflight_calls"]==1,
            "Exact proposal association or one-preflight rule changed")
        cell=by_cell_id[row["cell_id"]]
        decision=row["decision"]
        require(decision["binding"]["ik_seeds_rad"]==plain(common_starts),"Fresh proposal used outcome-informed seed")
        require(result["custom_test_solver"] or decision["binding"]["solver_protocol"]==IK_PROTOCOLS["full_robust_v2"],
            "Original finite robust-IK protocol changed")
        geometry_admitted=audit_decision(decision,scene,cell,row["action"],links,common_starts)
        vector=np.asarray([row["action"][k] for k in ACTION_NAMES],float)
        parent_vector=np.asarray([distributions[cell["id"]]["exact_action"][k] for k in ACTION_NAMES],float)
        normalized=2*(vector-LOW)/(HIGH-LOW)-1
        encoding_ok=bool(np.array_equal(vector,parent_vector) or ((normalized>-1)&(normalized<1)).all())
        admit=geometry_admitted and encoding_ok
        reason=decision["reason"] if not geometry_admitted else (None if encoding_ok else "categorical_encoding_saturated")
        status="admitted" if admit else ("unresolved" if reason.startswith("endpoint_ik_unresolved_") else "rejected_candidate")
        require(type(row["geometry_accepted"]) is bool and type(row["encoding_eligible"]) is bool
            and type(row["accepted"]) is bool and row["geometry_accepted"]==geometry_admitted
            and row["encoding_eligible"]==encoding_ok and row["accepted"]==admit
            and row["reason"]==reason and row["status"]==status,"Admission or exclusion reason changed")
        require(type(row["cpu_s"]) in (int,float) and np.isfinite(row["cpu_s"]) and row["cpu_s"]>=0,
            "Invalid per-proposal CPU measurement")
        if admit:accepted.append(row)
    require(result["admitted"]==cost["admitted"]==len(accepted) and len(result["supports"])==105,
            "Admitted support accounting changed")
    require(cost["aggregate_cpu_s"]+1e-6>=sum(r["cpu_s"] for r in result["records"]),
            "Aggregate CPU does not include completed proposal work")
    for cell,sheet in zip(cells,result["supports"]):
        rows=sorted((r for r in accepted if r["cell_id"]==cell["id"]),key=lambda r:r["candidate_id"])
        declined=[r for r in result["records"] if r["cell_id"]==cell["id"] and not r["accepted"]]
        require(sheet["cell_id"]==cell["id"] and type(sheet["admitted"]) is int and sheet["admitted"]==len(rows)
            and sheet["empty"] is (not bool(rows))
            and sheet["reasons"]==dict(sorted(Counter(r["reason"] for r in declined).items())),
            "Empty context removed or per-context reasons changed")
        if not rows:
            require(sheet["support"] is None,"Empty support was fabricated")
            continue
        support=FrozenCandidateSupport.from_dict(sheet["support"])
        require(support.context_id==cell["id"]
            and support.candidate_ids==tuple(r["candidate_id"] for r in rows)
            and support.actions==tuple(r["action"] for r in rows),"Fixed categorical support changed")
        parent_value=distributions[cell["id"]]
        parent_index=None
        for index,row in enumerate(rows):
            vector=np.asarray([row["action"][k] for k in ACTION_NAMES])
            if np.array_equal(vector,values_of(parent_value["exact_action"])):
                parent_index=index
                require(support.latents[index]==tuple(parent_value["latent_mean"]),"Exact original parent latent changed")
            else:
                latent=np.arctanh(2*(vector-LOW)/(HIGH-LOW)-1)
                require(np.allclose(support.latents[index],latent,rtol=0,atol=1e-12),"Candidate latent encoding changed")
        require(support.parent_index==parent_index,"Exact parent identity changed")
    require(all(type(result.get(name)) is int and result[name]==0 for name in
        ("optimizer_updates","native_shots","outcome_files_read"))
        and result.get("release_authorized") is False,"Geometry evidence cannot train or release")
    return signed(dict(schema=VERSION+"-audit",status="passed",plan_fingerprint=plan["fingerprint"],
        qualification_fingerprint=result["fingerprint"],split_fingerprint=splits["fingerprint"],
        input_bindings=copy.deepcopy(bindings),raw_proposals=RAW_PROPOSALS,unique_candidates=len(expected),
        admitted=len(accepted),declined=len(expected)-len(accepted),
        supported_contexts=sum(s["admitted"]>0 for s in result["supports"]),
        empty_contexts=[s["cell_id"] for s in result["supports"] if s["empty"]],
        exclusion_reasons=dict(sorted(Counter(r["reason"] for r in result["records"] if not r["accepted"]).items())),
        one_preflight_per_unique_proposal=True,proposal_sources_reconstructed=True,
        synthetic_test_only=result["custom_test_solver"],
        native_shots=0,optimizer_updates=0,outcome_files_read=0,release_authorized=False,
        scope="Independent proposal reconstruction and recorded FK/path gate verification only; no IK search, physics/capture certificate or impossibility proof."))
