"""Corruption tests for independent IK/search evidence checks; no native rollout."""

import copy
import unittest

import numpy as np

from sircus.mjlab.expanded_actions import WIDE_BOUNDS
from sircus.mjlab.verify_stable import (Audit, EXPECTED_INDICES, STUDY_VERSION,
    STABLE_MODES, CONTROL_MODES, audit_records, audit_solver_arrays,
    audit_endpoint_caches, digest, statistics)


def outcome(success=True):
    return dict(invalid=None, board_bounce=True, route_success=success,
        route_quality=2 if success else 0, whole_ball_entry=True,
        max_retention_s=1. if success else .5, external_contact_steps=0,
        closest_opening_distance_m=.02, board_contact_episodes=1,
        single_bounce_success=success, native_release=True, geometry_eligible=True,
        minimum_mouth_clearance_mm=20., source_phase="targeted_development",
        values=[.6,-.1,.32,28.,.3], q10=[0.]*6,q14=[0.]*6,
        branch10=0,branch14=0,geometry_cache_file="endpoint-batch-0000.npz",geometry_cache_row=0,
        pose_gate=dict(board_position_error_mm=0.,dropper_error_mm=0.,board_normal_error_deg=0.),
        endpoint_collision_pairs=[],scalar_score_audit_passed=True)


class StableAuditTests(unittest.TestCase):
    def fixture(self):
        full=[dict(id=f"old-test-{i:02d}",original_test_index=i,bucket_xy_m=[.9+i*.001,-.5],role="targeted_development")
              for i in (8,9,10,12,19,21,23,26,35,39,41,48)]
        targets=[t for t in full if t["original_test_index"] in EXPECTED_INDICES]
        original=dict(templates=[dict(source_trial_id=f"base-{i}") for i in range(17)])
        original["sha256"]=digest(original)
        controls=[f"prior/control-{i}" for i in (10,19,21,26,39,48)]
        spec=dict(version=STUDY_VERSION,no_training=True,base_bank_sha256=original["sha256"],
            development_layouts=targets,original_development_layouts=full,
            confirmations_per_candidate=10,final_confirmations_per_candidate=20,
            minimum_confirmation_rate=.8,native_per_target=128,top_candidates=1,
            control_source_ids=controls,control_repeats=1,action_bounds=WIDE_BOUNDS)
        spec["fingerprint"]=digest(spec)
        rows=[]
        def add(target,phase,success=True,**extra):
            r=outcome(success);r.update(id=f"stable-shot-{len(rows):07d}",target_id=target["id"],bucket_xy_m=target["bucket_xy_m"],phase=phase,**extra);rows.append(r);return r
        for mode in CONTROL_MODES:
            for tid in (10,19,21,26,39,48):
                t=next(t for t in full if t["original_test_index"]==tid)
                add(t,"control_"+mode,control_source_id=f"prior/control-{tid}",repeat=0)
        source=add(targets[0],"search")
        first=[add(targets[0],"confirmation",i<8,confirmation_source_id=source["id"]) for i in range(10)]
        final=[add(targets[0],"final_confirmation",i<16,confirmation_source_id=source["id"]) for i in range(20)]
        value=np.asarray(source["values"]);xy=np.asarray(source["bucket_xy_m"])
        template=dict(source_trial_id=source["id"],source_target_id=source["target_id"],source_phase="targeted_development",
            source_record_sha256=digest(source),relative_board_xy_m=(value[:2]-xy).tolist(),height_m=value[2],tilt_deg=value[3],gap_m=value[4],
            first_confirmation=statistics(first),final_confirmation=statistics(final),
            nominal_source=dict(requested_values=source["values"],**{k:source[k] for k in ("bucket_xy_m","q10","q14","branch10","branch14","geometry_cache_file","geometry_cache_row")}))
        bank=dict(templates=copy.deepcopy(original["templates"])+[template],parent_bank_sha256=original["sha256"],stable_ik_contract_fingerprint=spec["fingerprint"])
        bank["sha256"]=digest(bank)
        report=dict(status="completed",phase="completed",no_training=True,contract_fingerprint=spec["fingerprint"],
            proposal_bank_sha256=bank["sha256"],records=len(rows),native_search_attempts=1,confirmation_attempts=10,final_confirmation_attempts=20,
            control_attempts=30,native_search_statistics=statistics([source]),confirmation_statistics=statistics(first),final_confirmation_statistics=statistics(final),
            additions=1,base_templates=17,confirmed_templates=18,
            cells=[dict(t,search=statistics([source] if t==targets[0] else []),observed_assisted_entries=0,promotion=template if t==targets[0] else None) for t in targets])
        return spec,report,rows,bank,original

    def checks(self,fixture):
        return {c["name"]:c["passed"] for c in audit_records(*fixture).checks}

    def test_legitimate_records_pass(self):
        checks=self.checks(self.fixture());self.assertTrue(all(checks.values()),checks)

    def test_shortened_final_denominator_cannot_be_hidden(self):
        f=self.fixture();f[2].pop();f[1]["records"]-=1;f[1]["final_confirmation_attempts"]-=1
        f[1]["final_confirmation_statistics"]=statistics([r for r in f[2] if r["phase"]=="final_confirmation"])
        self.assertFalse(self.checks(f)["final_confirmation_denominators"])

    def test_assisted_entry_is_not_a_direct_capture(self):
        f=self.fixture();next(r for r in f[2] if r["phase"]=="search")["external_contact_steps"]=1
        self.assertFalse(self.checks(f)["unchanged_direct_capture_invariants"])

    def test_invalid_endpoint_cannot_be_counted_as_native_release(self):
        f=self.fixture();f[2][0]["invalid"]="endpoint_ik_or_pose"
        self.assertFalse(self.checks(f)["native_release_accounting"])

    def test_original_prefix_and_nominal_source_are_protected(self):
        f=self.fixture();f[3]["templates"][0]["source_trial_id"]="changed"
        self.assertFalse(self.checks(f)["unchanged_base_template_prefix"])
        f=self.fixture();f[3]["templates"][-1]["nominal_source"]["q14"]=[.1]*6
        self.assertFalse(self.checks(f)["old-test-08_native_promotion_provenance"])

    def solver_fixture(self):
        n=3;xy=np.asarray([[.9+i*.01,-.5] for i in range(n)]);values=np.tile([.6,-.1,.3,25.,.2],(n,1))
        q10=np.arange(n*6).reshape(n,6)/100.;q14=-q10;order=np.arange(n)[::-1]
        arrays=dict(bucket=xy,values=values,known_q10=q10,known_q14=q14,reverse_order=order,seeded_preserved=np.ones(n,bool))
        queries=[dict(bucket_xy_m=x.tolist(),values=v.tolist(),q10=a.tolist(),q14=b.tolist(),known_valid=True) for x,v,a,b in zip(xy,values,q10,q14)]
        comparison=[]
        for mode in ("old",)+STABLE_MODES:
            data=dict(mask=np.ones(n,bool),reasons=np.zeros(n,int),q10=q10,q14=q14,
                branch10=np.zeros(n,int),branch14=np.zeros(n,int),query10_sha256=np.asarray([str(i)*64 for i in range(n)]),query14_sha256=np.asarray([str(i+3)*64 for i in range(n)]))
            for suffix,index in (("",np.arange(n)),("_reverse",order),("_partition",np.arange(n))):
                for key,value in data.items():arrays[mode+suffix+"_"+key]=value[index].copy()
            comparison.append(dict(method=mode,queries=n,accepted=n,order_mask_changes=0,batch_mask_changes=0,max_q_delta_rad=0.))
        claimed=dict(comparison=comparison,known_branch_preservation=dict(known_valid_queries=n,preserved=n,passed=True),order_batch_invariance=dict(passed=True))
        return arrays,queries,claimed

    def solver_checks(self,fixture):
        a=Audit();audit_solver_arrays(a,*fixture);return {c["name"]:c["passed"] for c in a.checks}

    def test_legitimate_solver_arrays_pass(self):
        c=self.solver_checks(self.solver_fixture());self.assertTrue(all(c.values()),c)

    def test_batch_joint_or_branch_change_is_detected(self):
        f=self.solver_fixture();f[0]["stable_free_reverse_q14"][0,0]+=.001
        self.assertFalse(self.solver_checks(f)["stable_free_reverse_invariance"])
        f=self.solver_fixture();f[0]["stable_free_partition_branch14"][0]=2
        self.assertFalse(self.solver_checks(f)["stable_free_partition_branch14"])

    def test_known_pair_must_be_preserved_exactly(self):
        f=self.solver_fixture();f[0]["stable_seeded_q10"][0,0]+=1e-10
        self.assertFalse(self.solver_checks(f)["known_valid_joint_pairs_preserved_exactly"])

    def cache_fixture(self):
        source=dict(id="prior/source",values=[.6,-.1,.3,25.,.2],bucket_xy_m=[.9,-.5],q10=[0.]*6,q14=[0.]*6)
        row=dict(id="stable-shot-0000001",phase="confirmation",confirmation_source_id=source["id"],
            geometry_cache_file="endpoint-batch-0000.npz",geometry_cache_row=0,
            values=[.6,-.1,.3,25.,.21],bucket_xy_m=source["bucket_xy_m"],q10=[.01]*6,q14=[.01]*6,
            geometry_eligible=True,native_release=True,invalid=None,branch10=0,branch14=0,
            known_hint_used=True,known_pair_preserved=False,known_request_match=False,preserved_exact_nominal=False,refined_continuation=True)
        cache=dict(bucket=np.asarray([row["bucket_xy_m"]]),values=np.asarray([row["values"]]),eligible=np.asarray([True]),
            **{k:np.asarray([row[k]]) for k in ("q10","q14","branch10","branch14","known_hint_used","known_pair_preserved","known_request_match","preserved_exact_nominal","refined_continuation")},
            known_q10=np.asarray([source["q10"]]),known_q14=np.asarray([source["q14"]]),known_values=np.asarray([source["values"]]),known_bucket_xy=np.asarray([source["bucket_xy_m"]]))
        return [row],{row["geometry_cache_file"]:cache},[source],{},dict(control_source_ids=[])

    def cache_checks(self,fixture):
        a=Audit();audit_endpoint_caches(a,*fixture);return {c["name"]:c["passed"] for c in a.checks}

    def test_jitter_requires_real_continuation_not_same_saved_joints(self):
        f=self.cache_fixture();c=self.cache_checks(f);self.assertTrue(all(c.values()),c)
        row=f[0][0];cache=next(iter(f[1].values()))
        for k in ("q10","q14"):row[k]=[0.]*6;cache[k][0]=0
        self.assertFalse(self.cache_checks(f)["nominal_preservation_and_jitter_not_absorbed"])

    def test_hint_metadata_cannot_claim_jitter_as_original_request(self):
        f=self.cache_fixture();next(iter(f[1].values()))["known_values"][0,-1]=.21
        self.assertFalse(self.cache_checks(f)["measured_continuation_hint_provenance"])


if __name__=="__main__":unittest.main()
