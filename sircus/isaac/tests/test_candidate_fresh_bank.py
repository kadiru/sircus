"""Synthetic held-out geometry-bank tests; no real IK, policy or ball shots."""

import copy
from pathlib import Path
import queue
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import candidate_fresh_bank as bank
import candidate_fresh_bank_audit as audit
from candidate_fresh_splits import build_history_receipt,make_fresh_splits
from shot_benchmark_contract import make_splits


class FreshBankTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        old=make_splits([[99.,99.]])
        history=build_history_receipt([[99.,99.]],{"synthetic-coordinate-file":"a"*64},scope="Synthetic only")
        cls.splits=make_fresh_splits(history,old)
        cls.cells=bank._cells(cls.splits)
        middle=(bank.LOW+(bank.HIGH-bank.LOW)/2).tolist()
        cls.parent={c["id"]:bank.decode_action(c,middle,"parent-"+c["id"]) for c in cls.cells}
        cls.means={c["id"]:[0.]*5 for c in cls.cells}
        cls.scales={c["id"]:[.2]*5 for c in cls.cells}
        cls.bindings=dict(parent_checkpoint_sha256=bank.PARENT_CHECKPOINT_SHA256,parent_prediction_sha256="b"*64)
        cls.plan=bank.generate_fresh_candidates(cls.splits,cls.parent,cls.means,cls.scales,bindings=cls.bindings)
        cls.starts={name:[0.]*6 for name in bank.ARMS}

    def test_exact_8505_union_and_deterministic_parent_gaussian_sobol(self):
        self.assertEqual(self.plan["raw_proposals"],8505)
        self.assertEqual(self.plan["source_requests"],{"parent":105,"nearby":1680,"global":6720})
        self.assertEqual(sum(len(r["sources"]) for r in self.plan["records"]),8505)
        repeated=bank.generate_fresh_candidates(self.splits,self.parent,self.means,self.scales,bindings=self.bindings)
        self.assertEqual(self.plan,repeated)
        rows=[r for r in self.plan["records"] if r["cell_id"]==self.cells[0]["id"]]
        samples=np.random.default_rng(20261127).standard_normal((16,5))*.2
        expected=bank.LOW+(np.tanh(samples)+1)*(bank.HIGH-bank.LOW)/2
        np.testing.assert_array_equal([r["values"] for r in rows[1:17]],expected)
        self.assertEqual(rows[0]["action"],self.parent[self.cells[0]["id"]])
        self.assertEqual(set(c["split"] for c in self.plan["cells"]),{"validation","test"})
        self.assertEqual(self.plan["outcome_files_read"],0)

    def test_exact_dedup_retains_every_raw_alias(self):
        # Vanishing Gaussian scale produces exact duplicate parent physical
        # actions; none of those 16 raw source aliases may disappear.
        tiny={c["id"]:[1e-300]*5 for c in self.cells}
        plan=bank.generate_fresh_candidates(self.splits,self.parent,self.means,tiny,bindings=self.bindings)
        self.assertEqual(plan["raw_proposals"],8505)
        self.assertLess(plan["unique_candidates"],8505)
        first=plan["records"][0]
        self.assertEqual(len(first["sources"]),17)
        self.assertEqual(first["sources"][0]["kind"],"parent")
        self.assertEqual([s["index"] for s in first["sources"][1:]],list(range(16)))
        bank.validate_plan(plan)

    def test_wrong_parent_or_unmatched_means_and_counts_fail(self):
        changed=dict(self.bindings,parent_checkpoint_sha256="f"*64)
        with self.assertRaisesRegex(ValueError,"checkpoint"):
            bank.generate_fresh_candidates(self.splits,self.parent,self.means,self.scales,bindings=changed)
        means=copy.deepcopy(self.means);means[self.cells[0]["id"]][0]=2.
        with self.assertRaisesRegex(ValueError,"immutable mean"):
            bank.generate_fresh_candidates(self.splits,self.parent,means,self.scales,bindings=self.bindings)
        changed=copy.deepcopy(self.plan);changed["raw_proposals"]-=1;changed=bank.signed({k:v for k,v in changed.items() if k!="fingerprint"})
        with self.assertRaisesRegex(ValueError,"counts"):
            bank.validate_plan(changed)
        with self.assertRaises(TypeError):
            bank.generate_fresh_candidates(self.splits,self.parent,self.means,self.scales,bindings=self.bindings,successes=[])
        for field in ("historical_success_records_read","outcome_files_read","optimizer_updates","native_shots"):
            for invalid in (False,0.):
                value=copy.deepcopy(self.plan);value[field]=invalid
                value=bank.signed({k:v for k,v in value.items() if k!="fingerprint"})
                with self.subTest(counter=field,value=invalid),self.assertRaises(ValueError):bank.validate_plan(value)
        for invalid in (False,0.):
            value=copy.deepcopy(self.plan);value["records"][0]["sources"][0]["index"]=invalid
            value=bank.signed({k:v for k,v in value.items() if k!="fingerprint"})
            with self.subTest(alias_index=invalid),self.assertRaisesRegex(ValueError,"alias"):bank.validate_plan(value)

    def test_saturated_nonparent_encoding_fails_without_projection(self):
        row=copy.deepcopy(self.plan["records"][0])
        row["action"]=bank.decode_action(self.cells[0],bank.LOW.tolist(),"boundary")
        self.assertFalse(bank.encoding_eligible(row,self.plan["parent_distributions"][row["cell_id"]]))
        self.assertEqual(bank.values_of(row["action"]).tolist(),bank.LOW.tolist())
        distribution=dict(self.plan["parent_distributions"][row["cell_id"]],exact_action=row["action"],latent_mean=[-30.]*5)
        self.assertTrue(bank.encoding_eligible(row,distribution))

    def fake_preflight(self,scene,cell,action,links,starts,**kwargs):
        accepted=cell["id"]==self.cells[0]["id"]
        return dict(accepted=accepted,reason=None if accepted else "outer_link_length_bound",
            binding=dict(ik_seeds_rad=copy.deepcopy(starts),solver_protocol=dict(name="synthetic")),
            action=copy.deepcopy(action))

    def qualify(self):
        with patch.object(bank,"preflight",side_effect=self.fake_preflight) as call:
            result=bank.qualify_fresh_candidates(self.plan,{}, {},self.starts,workers=1,
                solver=lambda *a:None,solver_protocol="synthetic")
        self.assertEqual(call.call_count,self.plan["unique_candidates"])
        return result

    def audit(self,result,**kwargs):
        arguments=dict(parent_actions=self.parent,parent_latent_means=self.means,
            parent_latent_scales=self.scales,bindings=self.bindings)
        arguments.update(kwargs)
        with patch.object(audit,"audit_decision",side_effect=lambda decision,*a:decision["accepted"]):
            return audit.audit_fresh_bank(self.plan,result,self.splits,{}, {},self.starts,**arguments)

    def test_single_preflight_every_proposal_and_all_105_contexts_kept(self):
        result=self.qualify()
        self.assertEqual(len(result["records"]),self.plan["unique_candidates"])
        self.assertEqual(len(result["supports"]),105)
        self.assertEqual(sum(s["empty"] for s in result["supports"]),104)
        self.assertTrue(all(s["support"] is None for s in result["supports"][1:]))
        self.assertEqual(result["common_starts"],self.starts)
        report=self.audit(result)
        self.assertEqual(report["status"],"passed")
        self.assertEqual(report["supported_contexts"],1)
        self.assertTrue(report["synthetic_test_only"])

    def test_independent_audit_reconstructs_raw_sources_and_rejects_tampering(self):
        result=self.qualify()
        changed=copy.deepcopy(self.means);changed[self.cells[0]["id"]][0]=.1
        with self.assertRaisesRegex(ValueError,"Frozen parent"):
            self.audit(result,parent_latent_means=changed)
        for mutate in (
            lambda r:r["records"][0].update(preflight_calls=2),
            lambda r:r["records"][0].update(preflight_calls=True),
            lambda r:r["records"][0].update(accepted=False),
            lambda r:r["supports"][1].update(empty=False),
            lambda r:r["records"][0]["decision"]["binding"]["ik_seeds_rad"]["Arm10"].__setitem__(0,.1),
            lambda r:r.update(optimizer_updates=False),
            lambda r:r.update(native_shots=False),
            lambda r:r.update(outcome_files_read=0.),
            lambda r:r["supports"][1].update(admitted=False),
            lambda r:r["cost"].update(wall_s=False),
            lambda r:r["cost"].update(aggregate_cpu_s=False),
            lambda r:r["cost"].update(workers=True),
        ):
            value=copy.deepcopy(result);mutate(value)
            value=bank.signed({k:v for k,v in value.items() if k!="fingerprint"})
            with self.assertRaises(ValueError):self.audit(value)

    def test_preparation_limits_preserve_partial_work_and_never_certify(self):
        with patch.object(bank,"preflight",side_effect=self.fake_preflight):
            with self.assertRaises(bank.PreparationLimitError) as caught:
                bank.qualify_fresh_candidates(self.plan,{}, {},self.starts,workers=1,
                    solver=lambda *a:None,solver_protocol="synthetic",wall_limit_s=1e-12)
        self.assertEqual(caught.exception.partial["status"],"incomplete")
        self.assertEqual(caught.exception.partial["completed"],0)
        with self.assertRaisesRegex(ValueError,"four"):
            bank.qualify_fresh_candidates(self.plan,{}, {},self.starts,workers=5)
        with self.assertRaisesRegex(ValueError,"cannot increase"):
            bank.qualify_fresh_candidates(self.plan,{}, {},self.starts,cpu_limit_s=7201)
        for invalid in (True,float("nan"),float("inf")):
            with self.subTest(limit=invalid),self.assertRaisesRegex(ValueError,"cannot increase"):
                bank.qualify_fresh_candidates(self.plan,{}, {},self.starts,cpu_limit_s=invalid)

    def test_deadline_terminates_only_owned_workers(self):
        owned=[]
        class Process:
            def __init__(self,**kwargs):self.pid=99999999;self.alive=False;self.terminated=False;owned.append(self)
            def start(self):self.alive=True
            def is_alive(self):return self.alive
            def terminate(self):self.alive=False;self.terminated=True
            def join(self,timeout):pass
            def kill(self):self.alive=False
        class Queue(queue.Queue):
            def close(self):pass
            def cancel_join_thread(self):pass
        class Context:
            def Queue(self,maxsize=0):return Queue(maxsize=maxsize)
            def Process(self,**kwargs):return Process(**kwargs)
        with patch.object(bank.mp,"get_context",return_value=Context()),patch.object(bank,"HAS_LIVE_CHILD_CPU",True):
            with self.assertRaises(bank.PreparationLimitError):
                bank.qualify_fresh_candidates(self.plan,{}, {},self.starts,workers=4,wall_limit_s=1e-12)
        self.assertEqual(len(owned),4)
        self.assertTrue(all(p.terminated for p in owned))


if __name__=="__main__":unittest.main()
