"""Synthetic evidence checks, not simulated robot or capture measurements."""

import copy
import json
import math
from pathlib import Path
import sys
import unittest

import numpy as np

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
import feasible_candidates_audit as audit
from continuous_feasibility import canonical_hash, decode_action, LOW, HIGH
from expanded_pose_menu import endpoint_targets
from feasible_candidates import generate_candidates
from joint_feasibility import preflight


def sign(value):
    value = copy.deepcopy(value)
    value.pop("fingerprint", None)
    value["fingerprint"] = canonical_hash(value)
    return value


class IndependentCandidateAuditTests(unittest.TestCase):
    def setUp(self):
        self.scene = json.loads((SOURCE / "scene.json").read_text())
        self.cell = dict(id="train-000", bucket_xy_m=[.27, -.10], split="train")
        self.action = decode_action(self.cell, [0.,.4,.35,.24,math.radians(29)], "kept-exact")
        self.starts = {n:[0.]*6 for n in ("Arm10","Arm14")}
        targets = endpoint_targets(self.scene,self.cell,self.action)
        self.links = {n:[(targets[n].copy() if i==0 else np.eye(4), np.array([0.,0.,1.]),
                          -2*math.pi,2*math.pi) for i in range(6)] for n in self.starts}

        def solve(*args):
            return dict(q=[0.]*6, solved=True,residual=0.,position_error_mm=0.,
                        orientation_error_deg=0., attempts=[dict(nfev=1,residual=0.)])
        self.decision = preflight(self.scene,self.cell,self.action,self.links,self.starts,
                                  solver=solve,solver_protocol="synthetic-test-witness")
        self.parents={self.cell["id"]:self.action}
        self.means={self.cell["id"]:np.arctanh(2*(np.array(audit.values(self.action))-LOW)/(HIGH-LOW)-1).tolist()}
        self.scales={self.cell["id"]:[.01]*5}
        ik={n:solve() for n in self.starts}
        self.successes=[dict(case_id="saved-1",cell=self.cell,action=self.action,ik=ik)]
        self.plan=generate_candidates([self.cell],self.parents,self.means,self.scales,self.successes,
                                      nearby_per_cell=1,global_per_cell=1)
        rows=[]
        for record in self.plan["records"]:
            decision=preflight(self.scene,self.cell,record["action"],self.links,self.starts,
                                solver=solve,solver_protocol="synthetic-test-witness")
            seed_source=(dict(kind="historical_witness",case_id=record["witnesses"][0]["case_id"])
                         if record["witnesses"] else dict(kind="common_start"))
            rows.append(dict(candidate_id=record["candidate_id"],cell_id=record["cell_id"],action=record["action"],
                             sources=record["sources"],accepted=decision["accepted"],reason=decision["reason"],
                             decision=decision,attempts=[dict(seed_source=seed_source,decision=decision)]))
        self.result=sign(dict(schema="sircus-train-feasible-candidates-v1-qualification",
                             plan_fingerprint=self.plan["fingerprint"],records=rows,
                             source_sha256=self.plan["source_sha256"],optimizer_updates=0,native_shots=0,
                             release_authorized=False))
        self.decision=rows[0]["decision"]

    def run_audit(self, **changes):
        args=dict(plan=self.plan,result=self.result,successes=self.successes,
                  scene=self.scene,links=self.links,common_starts={self.cell["id"]:self.starts},
                  parent_actions=self.parents,parent_latent_means=self.means,parent_latent_scales=self.scales)
        args.update(changes)
        return audit.audit_candidates(**args)

    def repair_plan_result(self,plan,result):
        plan=sign(plan)
        result=copy.deepcopy(result);result["plan_fingerprint"]=plan["fingerprint"]
        return plan,sign(result)

    def test_counts_raw_aliases_separately_from_exact_deduplicated_candidates(self):
        result=self.run_audit()
        self.assertEqual(result["raw_source_requests"],4)
        self.assertEqual(result["unique_candidates"],3)
        self.assertEqual(result["successful_train_survive"],1)
        self.assertEqual(result["successful_train_excluded"],0)
        self.assertEqual(result["independently_rechecked_attempts"],3)
        self.assertFalse(result["release_authorized"])
        self.assertEqual(result["new_native_shots"],0)

    def test_missing_success_alias_fails_even_when_remaining_candidate_is_feasible(self):
        plan=copy.deepcopy(self.plan);result=copy.deepcopy(self.result)
        plan["records"][0]["sources"].pop()
        result["records"][0]["sources"].pop()
        plan,result=self.repair_plan_result(plan,result)
        with self.assertRaisesRegex(ValueError,"source union|witness coverage"):
            self.run_audit(plan=plan,result=result)

    def test_exact_success_context_and_action_are_required(self):
        for key,value in (("gap_m",self.action["gap_m"]+1e-12),):
            changed=copy.deepcopy(self.successes);changed[0]["action"][key]=value
            with self.assertRaisesRegex(ValueError,"TRAIN action/context"):
                self.run_audit(successes=changed)
        changed=copy.deepcopy(self.successes);changed[0]["cell"]["split"]="test"
        with self.assertRaisesRegex(ValueError,"TRAIN action/context"):
            self.run_audit(successes=changed)

    def test_duplicate_or_unknown_candidate_cannot_replace_missing_coverage(self):
        result=copy.deepcopy(self.result);result["records"].append(copy.deepcopy(result["records"][0]))
        with self.assertRaisesRegex(ValueError,"coverage differs"):
            self.run_audit(result=sign(result))

    def test_qualification_cannot_change_selected_physical_action(self):
        result=copy.deepcopy(self.result);result["records"][0]["action"]["gap_m"]+=.001
        with self.assertRaisesRegex(ValueError,"changed candidate"):
            self.run_audit(result=sign(result))

    def test_forged_fingerprint_and_changed_source_identity_fail(self):
        plan=copy.deepcopy(self.plan);plan["records"][0]["sources"][0]["physical_sha256"]="0"*64
        with self.assertRaisesRegex(ValueError,"fingerprint"):
            self.run_audit(plan=plan)
        plan,result=self.repair_plan_result(plan,self.result)
        with self.assertRaisesRegex(ValueError,"physical identity"):
            self.run_audit(plan=plan,result=result)

    def test_state_change_invalidates_existing_admission(self):
        starts=copy.deepcopy(self.starts);starts["Arm10"][0]=.001
        with self.assertRaisesRegex(ValueError,"Stale"):
            self.run_audit(common_starts={self.cell["id"]:starts})

    def test_reached_fk_and_proxy_evidence_are_independently_recomputed(self):
        result=copy.deepcopy(self.result)
        decision=copy.deepcopy(self.decision)
        decision["qualification"]["ik"]["Arm10"]["q"][0]=.1
        decision=sign(decision)
        result["records"][0]["decision"]=decision
        result["records"][0]["attempts"][0]["decision"]=decision
        with self.assertRaises(ValueError):
            self.run_audit(result=sign(result))

    def test_no_extra_solver_attempts_are_allowed_after_admission(self):
        result=copy.deepcopy(self.result)
        result["records"][0]["attempts"].append(copy.deepcopy(result["records"][0]["attempts"][0]))
        with self.assertRaisesRegex(ValueError,"continued after admission|seed budget"):
            self.run_audit(result=sign(result))

    def test_altered_parent_inputs_and_sampler_seed_cannot_pass(self):
        means=copy.deepcopy(self.means);means[self.cell["id"]][0]+=.001
        with self.assertRaisesRegex(ValueError,"parent differs"):
            self.run_audit(parent_latent_means=means)
        plan=copy.deepcopy(self.plan);plan["protocol"]["random_seed"]+=1
        plan,result=self.repair_plan_result(plan,self.result)
        with self.assertRaisesRegex(ValueError,"numerical proposal"):
            self.run_audit(plan=plan,result=result)

    def test_historical_witness_cannot_be_swapped_under_same_action(self):
        plan=copy.deepcopy(self.plan);plan["records"][0]["witnesses"][0]["ik"]["Arm10"]["q"][0]+=.001
        plan,result=self.repair_plan_result(plan,self.result)
        with self.assertRaisesRegex(ValueError,"Historical witness changed"):
            self.run_audit(plan=plan,result=result)

    def test_total_summary_cannot_inflate_admitted_count(self):
        result=copy.deepcopy(self.result);result.update(requests=3,admitted=3)
        with self.assertRaisesRegex(ValueError,"total accounting"):
            self.run_audit(result=sign(result))

    def test_seed_source_and_numerical_seed_must_match_planned_witness(self):
        result=copy.deepcopy(self.result)
        result["records"][0]["attempts"][0]["seed_source"]={"kind":"common_start"}
        with self.assertRaisesRegex(ValueError,"seed association"):
            self.run_audit(result=sign(result))
        result=copy.deepcopy(self.result)
        decision=copy.deepcopy(self.decision)
        binding=copy.deepcopy(decision["binding"]);binding["ik_seeds_rad"]["Arm10"][0]=.01
        decision["binding"]=sign(binding);decision=sign(decision)
        result["records"][0]["decision"]=decision
        result["records"][0]["attempts"][0]["decision"]=decision
        with self.assertRaisesRegex(ValueError,"seed association"):
            self.run_audit(result=sign(result))


if __name__=="__main__":
    unittest.main()
