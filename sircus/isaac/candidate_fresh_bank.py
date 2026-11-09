"""Outcome-blind held-out geometry bank; no training, flight or native dispatch.

Exactly 81 raw proposals are retained at each of 105 fresh coordinates. Only
the frozen parent's CPU predictions and geometry gates are inputs. Historical
successes and outcomes are not accepted by this API.
"""

from collections import Counter
import copy
import hashlib
import math
import multiprocessing as mp
from pathlib import Path
import queue
import resource
import time

import numpy as np
from scipy.stats import qmc

from candidate_continuation_ledger import PARENT_CHECKPOINT_SHA256
from candidate_continuation_policy import freeze_support
from continuous_feasibility import ACTION_NAMES, HIGH, LOW, canonical_hash, decode_action
from feasible_candidates import by_cell, exact_action, plain, signed, unsigned, values_of
from joint_feasibility import preflight

VERSION = "sircus-candidate-fresh-heldout-bank-v1"
QUALIFICATION_VERSION = VERSION+"-qualification"
SEED = 20261127
KINDS = ("parent", "nearby", "global")
RAW_PROPOSALS = 8505
ARMS = ("Arm10", "Arm14")
HAS_LIVE_CHILD_CPU = Path("/proc/self/stat").is_file()
SCOPE = ("Geometry-assisted outcome-held-out finite support; endpoint/FK, provisional "
    "capsules and sampled path proxies only. No flight, outcome query, historical "
    "success lookup, policy update, native movement or release authorization.")


def require(value, message):
    if not value:
        raise ValueError(message)


def source_hashes():
    import continuous_feasibility, joint_feasibility, robust_ik, settled_transit
    modules = (Path(__file__), Path(continuous_feasibility.__file__),
        Path(joint_feasibility.__file__), Path(robust_ik.__file__), Path(settled_transit.__file__))
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in modules}


def _cells(splits):
    from candidate_fresh_splits import cells_from_splits
    return cells_from_splits(splits)


def _bindings(bindings):
    require(type(bindings) is dict and bindings.get("parent_checkpoint_sha256") == PARENT_CHECKPOINT_SHA256,
            "Exact frozen seed-22 checkpoint binding required")
    require(all(type(k) is str and type(v) is str and len(v) == 64
        and set(v) <= set("0123456789abcdef") for k,v in bindings.items()),
        "Expected SHA256-only external bindings")
    return copy.deepcopy(bindings)


def generate_fresh_candidates(splits, parent_actions, parent_latent_means,
                              parent_latent_scales, *, bindings):
    """Freeze all 8,505 source aliases; exact physical deduplication only."""
    cells = _cells(splits)
    parent = by_cell(parent_actions, cells, "Fresh parent actions")
    means = by_cell(parent_latent_means, cells, "Fresh parent means")
    scales = by_cell(parent_latent_scales, cells, "Fresh parent scales")
    records, positions, distributions = [], {}, {}

    def add(cell, action, kind, index):
        vector = values_of(action).tolist()
        identifier = canonical_hash(dict(cell_id=cell["id"], values=vector))
        if identifier not in positions:
            positions[identifier] = len(records)
            records.append(dict(candidate_id=identifier, cell_id=cell["id"],
                action=copy.deepcopy(action), values=vector, sources=[]))
        records[positions[identifier]]["sources"].append(dict(kind=kind, index=index,
            original_action_id=action["id"], physical_sha256=canonical_hash(vector)))

    for i, cell in enumerate(cells):
        name = cell["id"]
        mean, scale = np.asarray(means[name], float), np.asarray(scales[name], float)
        require(mean.shape == scale.shape == (5,) and np.isfinite(mean).all()
            and np.isfinite(scale).all() and (scale > 0).all(), "Invalid frozen Gaussian")
        action = exact_action(cell, parent[name], "parent-"+name)
        require(np.allclose(2*(values_of(action)-LOW)/(HIGH-LOW)-1, np.tanh(mean),
                rtol=0, atol=2e-7), "Parent action and immutable mean disagree")
        distributions[name] = dict(latent_mean=mean.tolist(), latent_scale=scale.tolist(), exact_action=action)
        add(cell, action, "parent", 0)
        latent = mean+scale*np.random.default_rng(SEED+i).standard_normal((16,5))
        for j,vector in enumerate(LOW+(np.tanh(latent)+1)*(HIGH-LOW)/2):
            add(cell, decode_action(cell, vector, f"nearby-{name}-{j:04d}"), "nearby", j)
        unit = qmc.Sobol(d=5, scramble=True, seed=SEED+100000+i).random_base2(6)
        for j,vector in enumerate(LOW+unit*(HIGH-LOW)):
            add(cell, decode_action(cell, vector, f"global-{name}-{j:04d}"), "global", j)
    return signed(dict(schema=VERSION, split_fingerprint=splits["fingerprint"],
        split_manifest=copy.deepcopy(splits), cells=cells,
        parent_distributions=distributions, input_bindings=_bindings(bindings),
        protocol=dict(seed=SEED, parent_per_cell=1, nearby_per_cell=16, global_per_cell=64,
            action_names=list(ACTION_NAMES), low=LOW.tolist(), high=HIGH.tolist(),
            deduplication="Exact cell/physical-five-vector identity; retain every source alias",
            parent_decode_check="2e-7 normalized tanh consistency; preserve exact supplied original action",
            qualification="One original preflight per unique proposal from canonical starts; no retries or action replacement"),
        source_sha256=source_hashes(), records=records, raw_proposals=RAW_PROPOSALS,
        source_requests={"parent":105,"nearby":1680,"global":6720}, unique_candidates=len(records),
        historical_success_records_read=0, outcome_files_read=0, optimizer_updates=0,
        native_shots=0, release_authorized=False, scope=SCOPE))


def validate_plan(plan):
    value = unsigned(plan)
    require(value.get("schema") == VERSION and value.get("source_sha256") == source_hashes(),
            "Fresh bank source/schema changed")
    require(type(value["raw_proposals"]) is int and type(value["unique_candidates"]) is int
            and all(type(n) is int for n in value["source_requests"].values())
            and value["raw_proposals"] == RAW_PROPOSALS and value["source_requests"] == {"parent":105,"nearby":1680,"global":6720},
            "Fresh proposal counts changed")
    require(len(value["cells"]) == 105 and len({c["id"] for c in value["cells"]}) == 105
            and [c["split"] for c in value["cells"]] == ["validation"]*35+["test"]*70,
            "Fresh bank context coverage changed")
    require(value["cells"] == _cells(value["split_manifest"])
        and value["split_fingerprint"] == value["split_manifest"]["fingerprint"],
        "Fresh coordinate manifest changed or cells were relabelled")
    require(all(type(value["protocol"][name]) is int for name in
            ("seed","parent_per_cell","nearby_per_cell","global_per_cell"))
            and value["protocol"]["seed"] == SEED and value["protocol"]["parent_per_cell"] == 1
            and value["protocol"]["nearby_per_cell"] == 16 and value["protocol"]["global_per_cell"] == 64
            and value["protocol"]["action_names"] == list(ACTION_NAMES)
            and value["protocol"]["low"] == LOW.tolist() and value["protocol"]["high"] == HIGH.tolist(),
            "Fresh bank proposal protocol changed")
    _bindings(value["input_bindings"])
    cells = {c["id"]:c for c in value["cells"]}
    require(set(value["parent_distributions"]) == set(cells), "Parent distribution membership changed")
    aliases, identifiers, counts = set(), set(), Counter()
    for row in value["records"]:
        require(row["cell_id"] in cells and row["candidate_id"] not in identifiers, "Duplicate or foreign candidate")
        action = exact_action(cells[row["cell_id"]], row["action"], row["candidate_id"])
        vector = values_of(action).tolist()
        require(row["values"] == vector and row["candidate_id"] == canonical_hash(dict(cell_id=row["cell_id"],values=vector)),
                "Fresh physical identity changed")
        require(type(row["sources"]) is list and row["sources"], "Missing raw proposal aliases")
        for alias in row["sources"]:
            key = row["cell_id"],alias["kind"],alias["index"]
            require(type(alias["index"]) is int and alias["kind"] in KINDS and key not in aliases
                and alias["physical_sha256"] == canonical_hash(vector), "Raw source alias changed")
            aliases.add(key); counts[alias["kind"]] += 1
        identifiers.add(row["candidate_id"])
    expected = {(c["id"],k,i) for c in value["cells"] for k,n in (("parent",1),("nearby",16),("global",64)) for i in range(n)}
    require(aliases == expected and dict(counts) == value["source_requests"]
        and len(identifiers) == value["unique_candidates"], "Raw source union is incomplete")
    require(all(type(value.get(name)) is int and value[name] == 0 for name in
        ("historical_success_records_read","outcome_files_read","optimizer_updates","native_shots"))
        and value.get("release_authorized") is False, "Fresh geometry bank cannot learn or release")
    return value


def encoding_eligible(row, distribution):
    vector = values_of(row["action"])
    if np.array_equal(vector, values_of(distribution["exact_action"])):
        return True
    normalized = 2*(vector-LOW)/(HIGH-LOW)-1
    return bool(((normalized > -1) & (normalized < 1)).all())


def _one(scene, cell, row, links, starts, distribution, solver=None, solver_protocol=None):
    began = time.process_time()
    kwargs = {} if solver is None else dict(solver=solver,solver_protocol=solver_protocol)
    decision = preflight(scene,cell,row["action"],links,starts,seeds=starts,**kwargs)
    eligible = encoding_eligible(row,distribution)
    accepted = bool(decision["accepted"] and eligible)
    reason = decision["reason"] if not decision["accepted"] else (None if eligible else "categorical_encoding_saturated")
    return dict(candidate_id=row["candidate_id"],cell_id=row["cell_id"],action=copy.deepcopy(row["action"]),
        sources=copy.deepcopy(row["sources"]),decision=decision,preflight_calls=1,
        geometry_accepted=decision["accepted"],encoding_eligible=eligible,accepted=accepted,
        status="admitted" if accepted else ("unresolved" if reason.startswith("endpoint_ik_unresolved_") else "rejected_candidate"),
        reason=reason,cpu_s=time.process_time()-began)


def _worker(inbox,outbox,scene,cells,links,starts,distributions):
    while True:
        item = inbox.get()
        if item is None:
            return
        index,row = item
        try:
            result = _one(scene,cells[row["cell_id"]],row,links,starts,distributions[row["cell_id"]])
            outbox.put((index,result,None))
        except BaseException as error:
            outbox.put((index,None,type(error).__name__+": "+str(error)))
            return


class PreparationLimitError(RuntimeError):
    def __init__(self,message,partial):
        super().__init__(message)
        self.partial = partial


def qualify_fresh_candidates(plan,scene,links,common_starts,*,workers=4,progress=None,
        solver=None,solver_protocol=None,wall_limit_s=1800.,cpu_limit_s=7200.):
    """Bound owned CPU workers; failed or partial work never becomes a bank.

    A custom solver is a single-process synthetic-test seam only. Real work uses
    owned spawn workers; terminating them cannot interrupt any foreign process.
    CPU totals include parent and all reaped children. Polling limits are checked
    every 50 ms; any measured overage fails rather than certifying completion.
    """
    value = validate_plan(plan)
    require(type(workers) is int and 1 <= workers <= 4, "Choose at most four CPU workers")
    require(all(type(n) in (int,float) and math.isfinite(n) for n in (wall_limit_s,cpu_limit_s))
        and 0 < wall_limit_s <= 1800 and 0 < cpu_limit_s <= 7200, "Preparation limits cannot increase")
    require(solver is None or (workers == 1 and solver_protocol is not None), "Synthetic solver requires one worker and explicit protocol")
    require(solver is not None or HAS_LIVE_CHILD_CPU,
        "Production preparation requires Linux live child CPU accounting")
    starts = plain(common_starts)
    require(type(starts) is dict and set(starts) == set(ARMS)
        and all(np.asarray(q).shape == (6,) and np.isfinite(q).all() for q in starts.values()), "Both canonical arm starts required")
    cells = {c["id"]:c for c in value["cells"]}
    rows = value["records"]
    began, parent_cpu = time.monotonic(),time.process_time()
    children_before = resource.getrusage(resource.RUSAGE_CHILDREN)
    processes, records, next_index = [],{},0
    partial = None
    context = mp.get_context("spawn")
    inbox,outbox = context.Queue(maxsize=2*workers),context.Queue()

    def cpu_now():
        # Linux /proc includes running children; RUSAGE_CHILDREN accounts for
        # reaped ones. Elsewhere completed per-job CPU is a conservative monitor
        # with an exact reaped-child total checked before publication.
        running = 0.
        for process in processes:
            try:
                fields = Path(f"/proc/{process.pid}/stat").read_text().rsplit(")",1)[1].split()
                import os
                running += (int(fields[11])+int(fields[12]))/os.sysconf("SC_CLK_TCK")
            except (OSError,ValueError,IndexError):
                pass
        reaped = resource.getrusage(resource.RUSAGE_CHILDREN)
        return time.process_time()-parent_cpu+running+reaped.ru_utime+reaped.ru_stime-children_before.ru_utime-children_before.ru_stime

    def snapshot(status):
        return dict(status=status,completed=len(records),total=len(rows),
            admitted=sum(r["accepted"] for r in records.values()),wall_s=time.monotonic()-began,
            aggregate_cpu_s=cpu_now(),workers=workers,monitor_poll_s=.05,
            cpu_scope="Parent process plus all running/reaped owned children; no GPU work")

    try:
        if solver is None:
            for _ in range(workers):
                p=context.Process(target=_worker,args=(inbox,outbox,scene,cells,links,starts,value["parent_distributions"]))
                p.start();processes.append(p)
            for index in range(min(len(rows),2*workers)):
                inbox.put((index,rows[index]));next_index += 1
        while len(records) < len(rows):
            require(time.monotonic()-began < wall_limit_s, "Preparation wall limit reached")
            require(cpu_now() < cpu_limit_s, "Preparation aggregate CPU limit reached")
            if solver is not None:
                index=len(records);row=rows[index]
                result=_one(scene,cells[row["cell_id"]],row,links,starts,value["parent_distributions"][row["cell_id"]],solver,solver_protocol)
            else:
                try:
                    index,result,error=outbox.get(timeout=.05)
                except queue.Empty:
                    require(all(p.is_alive() for p in processes),"Owned qualification worker exited without a terminal receipt")
                    continue
                require(error is None,"Qualification worker failed: "+str(error))
                if next_index < len(rows):
                    inbox.put((next_index,rows[next_index]));next_index += 1
            require(index not in records,"Duplicate geometry receipt")
            records[index]=result
            if progress and (len(records)%100 == 0 or len(records)==len(rows)):
                progress(snapshot("qualifying"))
    except BaseException as error:
        partial=snapshot("incomplete")
        partial["records"]=[records[i] for i in sorted(records)]
        raise PreparationLimitError(str(error),partial) from error
    finally:
        for p in processes:
            if p.is_alive():p.terminate()
        for p in processes:
            p.join(timeout=5)
            if p.is_alive():p.kill();p.join(timeout=5)
        # Killed workers cannot drain a full input pipe. Never leave an owned
        # feeder-thread join capable of defeating the preparation deadline.
        inbox.cancel_join_thread();outbox.cancel_join_thread()
        inbox.close();outbox.close()
        if partial is not None:
            partial.update(snapshot("incomplete"))
    ordered=[records[i] for i in range(len(rows))]
    supports=[]
    for cell in value["cells"]:
        selected=sorted((r for r in ordered if r["cell_id"]==cell["id"] and r["accepted"]),key=lambda r:r["candidate_id"])
        parent=value["parent_distributions"][cell["id"]]
        support=freeze_support(cell["id"],selected,parent["exact_action"],parent["latent_mean"]).to_dict() if selected else None
        declined=[r for r in ordered if r["cell_id"]==cell["id"] and not r["accepted"]]
        supports.append(dict(cell_id=cell["id"],admitted=len(selected),support=support,
            empty=not bool(selected),reasons=dict(sorted(Counter(r["reason"] for r in declined).items()))))
    cost=snapshot("qualified")
    if cost["wall_s"] > wall_limit_s or cost["aggregate_cpu_s"] > cpu_limit_s:
        raise PreparationLimitError("Measured preparation cost exceeded its cap; no bank is certified",
            dict(cost,status="incomplete",records=[records[i] for i in sorted(records)]))
    return signed(dict(schema=QUALIFICATION_VERSION,plan_fingerprint=plan["fingerprint"],
        source_sha256=source_hashes(),scene_sha256=canonical_hash(plain(scene)),common_starts=starts,
        records=ordered,unique_candidates=len(rows),raw_proposals=RAW_PROPOSALS,
        admitted=sum(r["accepted"] for r in ordered),supports=supports,cost=cost,
        workers=workers,custom_test_solver=solver is not None,wall_limit_s=wall_limit_s,cpu_limit_s=cpu_limit_s,
        optimizer_updates=0,native_shots=0,outcome_files_read=0,release_authorized=False,scope=SCOPE))
