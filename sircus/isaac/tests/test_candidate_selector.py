"""Synthetic bank selection and lineage tests; no simulator or GPU."""

import copy
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
import candidate_selector as selector
from continuous_feasibility import canonical_hash, decode_action, HIGH, LOW
from feasible_candidates import generate_candidates, signed, values_of


def sign(value):
    value = copy.deepcopy(value)
    value.pop("fingerprint", None)
    return signed(value)


class SelectorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.published = self.root/"published.json"
        self.cells = [dict(id=f"train-{i:03d}", split="train", bucket_xy_m=[.1+i*.01, -.1]) for i in range(35)]
        vector = np.array([0., .4, .35, .24, math.radians(29)])
        self.parents = {c["id"]:decode_action(c, vector, "parent-"+c["id"]) for c in self.cells}
        self.means = {c["id"]:np.arctanh(2*(vector-LOW)/(HIGH-LOW)-1).tolist() for c in self.cells}
        self.scales = {c["id"]:[.2]*5 for c in self.cells}
        self.fits = {n:dict(solved=True,q=[0.]*6,residual=0.) for n in selector.ARMS}
        successes = [dict(case_id="old-"+c["id"], cell=c,
            action=decode_action(c, vector+[0,0,.01,0,0], "old-"+c["id"]), ik=self.fits) for c in self.cells]
        self.plan = generate_candidates(self.cells,self.parents,self.means,self.scales,successes,
                                        nearby_per_cell=0,global_per_cell=0)
        self.starts = {c["id"]:{n:[0.]*6 for n in selector.ARMS} for c in self.cells}
        rows = []
        for row in self.plan["records"]:
            decision = dict(action=copy.deepcopy(row["action"]),accepted=True,
                qualification=dict(accepted=True,ik=copy.deepcopy(self.fits)),path=dict(accepted=True),
                release_authorized=False,binding=dict(context=dict(id=row["cell_id"]),
                    measured_starts_rad=copy.deepcopy(self.starts[row["cell_id"]])))
            rows.append(dict(candidate_id=row["candidate_id"], cell_id=row["cell_id"],
                             action=copy.deepcopy(row["action"]),sources=copy.deepcopy(row["sources"]),
                             accepted=True,reason=None,decision=decision))
        self.qual = dict(plan_fingerprint=self.plan["fingerprint"],records=rows,admitted=len(rows),
                         commonstarts=self.starts,optimizer_updates=0,native_shots=0)
        self.seal()

    def seal(self):
        self.plan = sign(self.plan)
        self.qual["plan_fingerprint"] = self.plan["fingerprint"]
        self.qual["admitted"] = sum(r["accepted"] for r in self.qual["records"])
        self.qual = sign(self.qual)
        self.audit = signed(dict(status="passed",plan_fingerprint=self.plan["fingerprint"],
            qualification_fingerprint=self.qual["fingerprint"],admitted=self.qual["admitted"],
            optimizer_updates=0,new_native_shots=0))
        self.summary = signed(dict(schema="sircus-seed22-train-candidate-bank-v1",status="passed",seed=20261122,
            artifacts_sha256={name:selector.serialized_sha256(value) for name,value in (
                ("plan.json",self.plan),("qualification.json",self.qual),("audit.json",self.audit))},
            admitted=self.qual["admitted"],optimizer_updates=0,new_native_shots=0,held_out_shot_files_read=0,
            input_bindings=dict(original=dict(checkpoint_sha256=selector.PARENT_SHA256,
                experiment_sha256="a"*64,split_manifest_sha256="b"*64),assets_sha256=dict(Arm10="c"*64,Arm14="d"*64))))
        self.published.write_text(json.dumps(self.summary,indent=2)+"\n")

    def build(self):
        with patch.object(selector,"PUBLISHED_SUMMARY",self.published):
            return selector.build_comparison(self.plan,self.qual,self.audit,self.summary)

    def decline(self, row):
        row["accepted"]=False;row["reason"]="synthetic_unresolved"
        row["decision"]["accepted"]=False;row["decision"]["qualification"]["accepted"]=False
        row["decision"]["path"]=None

    def test_all_qualified_parent_actions_are_preserved_with_matching_native_ik_protocol(self):
        result=self.build()
        self.assertEqual(result["request_count"],70)
        self.assertEqual(result["cohort_order"],["parent","candidate_bank"])
        self.assertEqual(result["parent_preserved"],35)
        self.assertEqual(result["changed_bank_actions"],0)
        self.assertEqual(result["cohorts"][0]["actions"],result["cohorts"][1]["actions"])
        self.assertEqual(result["cohorts"][0]["ik_seeds"],[None]*35)
        self.assertEqual(result["cohorts"][1]["ik_seeds"],[None]*35)
        self.assertEqual([r["request_index"] for r in result["requests"]],list(range(70)))
        self.assertEqual([r["method"] for r in result["requests"]],["parent"]*35+["candidate_bank"]*35)
        self.assertFalse(result["release_authorized"])
        self.assertEqual(result["optimizer_updates"],0)

    def test_replacement_uses_only_qualified_candidate_and_its_prepared_seed(self):
        parent_row=next(r for r in self.qual["records"] if r["cell_id"]=="train-000" and any(s["kind"]=="parent" for s in r["sources"]))
        self.decline(parent_row);self.seal()
        result=self.build()
        self.assertEqual(result["parent_preserved"],34)
        self.assertEqual(result["changed_bank_actions"],1)
        self.assertNotEqual(values_of(result["cohorts"][0]["actions"][0]).tolist(),values_of(result["cohorts"][1]["actions"][0]).tolist())
        self.assertEqual(result["cohorts"][1]["ik_seeds"][0],self.fits)
        self.assertTrue(result["cohorts"][1]["native_enabled"][0])

    def test_missing_support_is_explicit_decline_and_stays_in_70_requests(self):
        for row in self.qual["records"]:
            if row["cell_id"]=="train-000":self.decline(row)
        self.seal();result=self.build()
        self.assertEqual(result["bank_predeclined"],1)
        self.assertFalse(result["cohorts"][1]["native_enabled"][0])
        self.assertEqual(result["cohorts"][1]["decline_reasons"][0],"no_qualified_candidate_support")
        self.assertEqual(len(result["requests"]),70)
        self.assertTrue(result["requests"][35]["predeclined"])
        self.assertEqual(result["requests"][35]["action"],result["requests"][0]["action"])

    def test_finite_probabilities_normalize_and_do_not_use_outcome_fields(self):
        result=self.build();selection=result["selections"][0]
        self.assertAlmostEqual(sum(c["categorical_probability"] for c in selection["candidates"]),1.)
        self.assertEqual(selection["reason"],"qualified_parent_preserved")
        before=[c["score"] for c in selection["candidates"]]
        for row in self.plan["records"]:row["reward"]=999;row["route_success"]=False
        for row in self.qual["records"]:row["reward"]=-999;row["route_success"]=True
        self.seal();after=[c["score"] for c in self.build()["selections"][0]["candidates"]]
        self.assertEqual(before,after)

    def test_nonparent_ties_use_lexical_candidate_identity(self):
        cell=self.cells[0];mean=np.zeros(5);scale=np.ones(5)
        parent=decode_action(cell,LOW+(np.tanh(mean)+1)*(HIGH-LOW)/2,"parent")
        a=decode_action(cell,LOW+(np.tanh([.5,0,0,0,0])+1)*(HIGH-LOW)/2,"a")
        b=decode_action(cell,LOW+(np.tanh([-.5,0,0,0,0])+1)*(HIGH-LOW)/2,"b")
        choices=[dict(candidate_id="z",action=a),dict(candidate_id="a",action=b)]
        p=selector.preference_probabilities(mean,scale,choices,parent)
        self.assertEqual(p["selected_index"],1)
        self.assertAlmostEqual(sum(p["probabilities"]),1.)
        self.assertFalse(p["stochastic_sampling"])

    def test_saturated_boundary_fails_closed_without_clipping(self):
        action=decode_action(self.cells[0],LOW,"boundary")
        with self.assertRaisesRegex(ValueError,"saturated latent boundary"):
            selector.preference_probabilities([0]*5,[1]*5,[dict(candidate_id="x",action=action)],self.parents["train-000"])

    def test_changed_state_action_or_heldout_membership_fails_closed(self):
        changed=copy.deepcopy(self.qual)
        self.qual["records"][0]["decision"]["binding"]["measured_starts_rad"]["Arm10"][0]=.01
        self.seal()
        with self.assertRaisesRegex(ValueError,"state/action binding"):self.build()
        self.qual=changed
        self.qual["records"][0]["action"]["gap_m"]+=.001;self.seal()
        with self.assertRaisesRegex(ValueError,"exact candidate"):self.build()
        self.qual=changed;self.plan["cells"][0]["split"]="test";self.seal()
        with self.assertRaisesRegex(ValueError,"TRAIN"):self.build()

    def test_hash_mismatch_and_unpublished_bank_cannot_be_selected(self):
        self.summary["admitted"]+=1
        with self.assertRaisesRegex(ValueError,"fingerprint"):self.build()
        self.seal();self.published.write_text("{}\n")
        with self.assertRaisesRegex(ValueError,"published candidate bank"):self.build()

    def test_loader_verifies_actual_artifact_bytes(self):
        for name,value in (("plan.json",self.plan),("qualification.json",self.qual),
                           ("audit.json",self.audit),("summary.json",self.summary)):
            (self.root/name).write_text(json.dumps(value,indent=2)+"\n")
        with patch.object(selector,"PUBLISHED_SUMMARY",self.published):
            values=selector.load_bank(self.root)
            self.assertEqual(values[0],self.plan)
            (self.root/"plan.json").write_text("{}\n")
            with self.assertRaisesRegex(ValueError,"SHA mismatch"):
                selector.load_bank(self.root)

    def test_comparison_request_mask_probability_and_parent_protocol_cannot_change(self):
        original=self.build()
        cases=[]
        result=copy.deepcopy(original);result["request_count"]=71;cases.append(result)
        result=copy.deepcopy(original);result["requests"][35]["action"]["gap_m"]+=.001;cases.append(result)
        result=copy.deepcopy(original);result["cohorts"][0]["ik_seeds"][0]=self.fits;cases.append(result)
        result=copy.deepcopy(original);result["cohorts"][1]["ik_seeds"][0]=self.fits;cases.append(result)
        result=copy.deepcopy(original);result["selections"][0]["candidates"][0]["categorical_probability"]+=.01;cases.append(result)
        result=copy.deepcopy(original);result["optimizer_updates"]=1;cases.append(result)
        result=copy.deepcopy(original);result["cells"][0]["split"]="validation";cases.append(result)
        for result in cases:
            with self.subTest(case=cases.index(result)),self.assertRaises(ValueError):
                selector.validate_comparison(sign(result))

    def test_native_binding_can_be_appended_and_requires_immutable_source_and_checkpoint(self):
        result=self.build()
        result["native_binding"]=dict(original_source=dict(commit="1"*40,geometry_fingerprint="2"*64),
            source_commit="3"*40,source_archive_sha256="4"*64,
            checkpoint_path="/workspace/frozen/checkpoint.pt",checkpoint_sha256=selector.PARENT_SHA256,
            geometry_fingerprint="2"*64,source_scene_sha256="5"*64,bank_preprocessing={},selection_wall_s=.1)
        selector.validate_comparison(sign(result))
        for key,value in (("checkpoint_sha256","0"*64),("selection_wall_s",-1),("source_commit","main")):
            changed=copy.deepcopy(result);changed["native_binding"][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                selector.validate_comparison(sign(changed))


if __name__=="__main__":unittest.main()
