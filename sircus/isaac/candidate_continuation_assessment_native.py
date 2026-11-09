"""Final assessment on the same initialized, independently guarded native app.

No app is started here. A future explicit entry point supplies the already
validated TRAIN producer and a final-frozen assessment runner. Every cohort
uses one canonical home and one unchanged native guard/scorer call, no search.
"""

import copy
from pathlib import Path
import time

import numpy as np

from candidate_continuation_assessment import require, digest
from candidate_continuation_journal import read_events, write_new_json, _sync_directory
from candidate_continuation_native import _sync_file, _raw_incomplete
from candidate_native_audit import verify_controls
from presentation_recording import save_physics
from settled_transit import protocol as setup_protocol, release_ready
from shot_benchmark_native import compressed_json
from shot_benchmark_reward import build_features, reward

PACKET_SCHEMA="sircus-candidate-continuation-assessment-native-trial-v1"


class GuardedAssessmentProducer:
    """One durable dispatch per frozen cohort; never a new app or retry."""

    def __init__(self,train_producer,plan):
        train_producer._validate_binding()
        self.train_producer=train_producer;self.env=train_producer.env
        self.binding=copy.deepcopy(train_producer.binding);self.plan=copy.deepcopy(plan)
        self.controls_directory=Path(self.env.dest)
        self.original_cells=copy.deepcopy(train_producer.original_cells)
        require(plan["bindings"]["source_sha256"]==self.binding["source_sha256"],"Assessment/native source differs")
        require(plan["bindings"].get("native_binding_sha256")==self.binding["fingerprint"],
                "Assessment must bind the same validated native application")
        self.claimed=set()

    def execute_reserved(self,runner,choices,*,expected_journal_head):
        """Dispatch already persisted exact choices, returning terminal rows only."""
        self.train_producer._validate_binding();runner._guard()
        require(runner.plan==self.plan and runner.ledger.phase=="assessment"
            and runner.ledger.pending is not None and not runner.ledger.pending.outcomes,
            "Expected a fresh final-only assessment reservation")
        pending=runner.ledger.pending
        require(len(choices)==len(pending.requests)==35 and runner.journal.head==expected_journal_head
            and expected_journal_head not in self.claimed,"Stale or repeated assessment dispatch")
        events=read_events(runner.journal.directory)
        require(events[-1]["kind"]=="reserve" and events[-1]["fingerprint"]==expected_journal_head
            and events[-1]["payload"]==dict(ledger=runner.ledger.to_dict(),choices=list(choices),
                plan_fingerprint=self.plan["fingerprint"],model_hashes=self.plan["model_hashes"]),
            "Assessment request was not the exact durable journal head")
        first=pending.requests[0].request_index;cohort=first//35
        require(list(choices)==self.plan["choices"][first:first+35],"Assessment choices changed after freeze")
        directory=Path(runner.directory).resolve()
        require(not directory.is_relative_to(self.train_producer.frozen_root),"Assessment cannot modify frozen experiment")
        native=directory/"native"
        if not native.exists():native.mkdir();_sync_directory(directory)
        stem=f"assessment-{cohort:06d}"
        target,physics_path=native/(stem+".json.gz"),native/(stem+".npz")
        require(not target.exists() and not physics_path.exists(),"Assessment evidence already exists")
        controls=verify_controls(self.controls_directory,self.env.cfg,self.binding["startup_control_sha256"])
        write_new_json(native/(stem+".dispatch.json"),dict(reservation_sha256=expected_journal_head,
            ledger_fingerprint=runner.ledger.to_dict()["fingerprint"],request_ids=[c["request_id"] for c in choices],
            assessment_plan_fingerprint=self.plan["fingerprint"],native_binding_fingerprint=self.binding["fingerprint"],
            status="claimed_before_native_call"))
        self.claimed.add(expected_journal_head)
        by_id={c["id"]:c for rows in self.plan["cells_by_split"].values() for c in rows}
        cells=[dict(copy.deepcopy(self.original_cells[i]),**copy.deepcopy(by_id[c["context_id"]])) for i,c in enumerate(choices)]
        actions=[copy.deepcopy(c["action"]) for c in choices]
        enabled=[c["enabled"] for c in choices];seeds=[copy.deepcopy(c["ik_seeds"]) for c in choices]
        env=self.env;began,cpu=time.monotonic(),time.process_time()
        try:
            env.dest=directory
            env.capture_trial=env.film_trial=False;env.phase=None
            env.base_cells=copy.deepcopy(cells);env.spec["cells"]=copy.deepcopy(cells)
            env.bucket_xy=np.asarray([c["bucket_xy_m"] for c in cells])
            env.home_readback=env.home()
            env.active_mask=np.asarray(enabled,dtype=bool);env.support_missing=[False]*35
            env.proposal_ik_seeds=copy.deepcopy(seeds)
            env.trial_mode,env.trial_batch=choices[0]["method"],cohort
            env.frames,env.physics_samples=[],[]
            env.capture_trial,env.film_trial,env.phase=True,False,"setup"
            env.trial_start_step=env.steps
            rows,raw=env.shot_batch(copy.deepcopy(actions),record_video=False,enabled=env.active_mask)
            env.capture_trial=env.film_trial=False;env.phase=None
            require(len(rows)==35,"Assessment native row coverage changed")
            physics,features=None,[None]*35
            if env.physics_samples:
                save_physics(physics_path,env.physics_samples);_sync_file(physics_path)
                physics=dict(file=physics_path.name,sha256=digest(physics_path),samples=len(env.physics_samples))
                position=np.asarray([s["position_m"] for s in env.physics_samples])-env.origins[None]
                velocity=np.asarray([s["velocity_m_s"] for s in env.physics_samples])
                forces=np.asarray([s["forces_N"] for s in env.physics_samples])
                for i,row in enumerate(rows):
                    if not row["invalid"]:
                        features[i]=build_features(env.cfg,cells[i]["bucket_xy_m"],position[:,i],velocity[:,i],forces[:,i],env.dt,env.scoring_normals[i])
            returns=[]
            for i,(row,choice) in enumerate(zip(rows,choices)):
                row["native_invalid"]=row["invalid"]
                if not choice["enabled"]:
                    # The unchanged ParallelShots boundary labels disabled rows
                    # after prepare(), while qualification retains inactive_split.
                    require(row["invalid"]=="no_selected_setting" and env.preflight_records[i] is None
                        and env.qualifications[i]==dict(accepted=False,reason="inactive_split")
                        and env.path_checks[i] is None and env.path_reasons[i] is None
                        and row["route_success"] is False and row["success"] is False
                        and row["route_quality"] == 0 and features[i] is None,
                        "Empty bank slot was executed or fabricated a guard decline")
                    row["invalid"]=choice["decline_reason"]
                if not row["invalid"]:
                    require(choice["enabled"] and env.preflight_records[i]["accepted"] is True
                        and env.qualifications[i]["accepted"] is True and env.path_checks[i]["accepted"] is True
                        and env.path_reasons[i] is None and release_ready(row["settled"]),
                        "Assessment release bypassed unchanged guard gates")
                require(type(row["route_success"]) is bool and type(row["route_quality"]) is int
                    and row["route_quality"] in ((1,2) if row["route_success"] else (0,))
                    and (not row["invalid"] or not row["route_success"]),"Contradictory assessment capture flags")
                row.update(split=choice["split"],qualification=env.qualifications[i],path_rejection=env.path_reasons[i],
                    path_proxy=env.path_checks[i],scoring_board_normal=env.scoring_normals[i].tolist())
                returns.append(reward(row,features[i]))
            require(physics is not None or all(r["invalid"] for r in rows),"Assessment release lacks physics evidence")
            packet=dict(schema=PACKET_SCHEMA,mode="assessment",batch=pending.batch_index,cohort=cohort,
                split=choices[0]["split"],method=choices[0]["method"],synthetic=runner.synthetic,
                reservation_sha256=expected_journal_head,assessment_plan_fingerprint=self.plan["fingerprint"],
                native_binding=self.binding,model_hashes=self.plan["model_hashes"],
                final_checkpoint_sha256=self.plan["final_checkpoint_sha256"],cells=cells,actions=actions,
                choices=copy.deepcopy(list(choices)),request_ids=[c["request_id"] for c in choices],enabled=enabled,
                decline_reasons=[c["decline_reason"] for c in choices],proposal_ik_seeds=seeds,
                rows=rows,rewards=returns,reward_features=features,home=env.home_readback,
                preflight_records=copy.deepcopy(env.preflight_records),release_step=env.release_step,score_end_step=env.steps,
                physics=physics,origins_m=env.origins.tolist(),setup_controller=setup_protocol(),startup_controls=controls,
                settling_history=raw["settling"],transit_history=env.transit_history,correction_history=env.correction_history,
                trajectory_m=raw["trajectory_m"],wall_s=time.monotonic()-began,process_cpu_s=time.process_time()-cpu,
                optimizer_updates=0,scope="Final frozen direct assessment; exact one request per method/position; no held-out search or retries.")
            receipt=compressed_json(target,packet);_sync_file(target)
            write_new_json(native/(stem+".receipt.json"),receipt)
            self.train_producer._validate_binding();runner._guard()
            return [dict(request_id=choice["request_id"],released=not bool(row["invalid"]),
                decline_reason=row["invalid"] or None,source_sha256=receipt["sha256"],captured=bool(row["route_success"]),
                clean=row["route_quality"]==2,**{"return":returns[i]}) for i,(choice,row) in enumerate(zip(choices,rows))]
        except BaseException as error:
            env.capture_trial=env.film_trial=False;env.phase=None
            partial=None
            if env.physics_samples and not physics_path.exists():
                try:save_physics(physics_path,env.physics_samples);_sync_file(physics_path)
                except Exception as telemetry_error:
                    path=native/(stem+".incomplete-physics.json")
                    write_new_json(path,dict(status="invalid_unscored_raw_evidence",error=str(telemetry_error),samples=_raw_incomplete(env.physics_samples)))
                    partial=dict(file=path.name,sha256=digest(path))
            write_new_json(native/(stem+".failure.json"),dict(status="incomplete_no_retry",reservation_sha256=expected_journal_head,
                error=type(error).__name__+": "+str(error),physics_sha256=digest(physics_path) if physics_path.exists() else None,partial_physics=partial))
            runner._fail(error)
            raise
