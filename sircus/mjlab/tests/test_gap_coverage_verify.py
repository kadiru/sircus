import copy
from contextlib import ExitStack
import sys
from types import SimpleNamespace
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from sircus.mjlab import gap_coverage_verify as audit
from sircus.mjlab import calibrated_admission, consistent_flight_selector


def requests():
    rows = []
    for role in ("positive_control", "negative_control"):
        rows.extend(dict(id=f"pre-{role}-{i}", role=role, phase="controls_pre", method="control") for i in range(3))
    for method in audit.METHODS:
        rows.extend(dict(id=f"{method}-{i}", role="test", phase="test", method=method,
                         context_id=f"context-{i}", bucket_xy_m=[i*.02, -.2]) for i in range(50))
    for role in ("positive_control", "negative_control"):
        rows.extend(dict(id=f"post-{role}-{i}", role=role, phase="controls_post", method="control") for i in range(3))
    return rows


class GapFrozenProtocolChecks(unittest.TestCase):
    def test_seeded_layouts_exclude_old_test_and_all_ancestors(self):
        old = dict(excluded=[[.1, -.5], [.7, .2]], splits=dict(
            validation=[dict(bucket_xy_m=[.2, .1])], test=[dict(bucket_xy_m=[.8, -.3])]))
        result = audit.expected_splits(old)
        self.assertEqual(result, audit.expected_splits(old))
        self.assertEqual(len(result["splits"]["test"]), 50)
        xy = np.asarray([r["bucket_xy_m"] for r in result["splits"]["test"]])
        excluded = np.asarray(result["excluded"])
        self.assertEqual(len(excluded), 4)
        self.assertGreaterEqual(np.linalg.norm(xy[:, None]-excluded[None], axis=2).min(), .01)
        distance = np.linalg.norm(xy[:, None]-xy[None], axis=2)
        distance += np.eye(50)*100
        self.assertGreaterEqual(distance.min(), .01)
        self.assertTrue(((xy >= [.1, -.65]) & (xy <= [1.24, .25])).all())

    def test_complete_paired_budget_passes(self):
        audit.budget_checks(requests(), audit.AuditChecks())

    def test_same_length_replaced_context_is_rejected(self):
        rows = requests()
        next(r for r in rows if r.get("method") == "consistent")["bucket_xy_m"] = [1.2, .2]
        with self.assertRaises(audit.EvidenceError):
            audit.budget_checks(rows, audit.AuditChecks())

    def test_missing_failed_request_cannot_disappear_from_denominator(self):
        with self.assertRaises(audit.EvidenceError):
            audit.budget_checks(requests()[:-1], audit.AuditChecks())

    def test_repeated_identity_is_rejected(self):
        rows = requests(); rows[-1]["id"] = rows[-2]["id"]
        with self.assertRaises(audit.EvidenceError):
            audit.budget_checks(rows, audit.AuditChecks())

    def test_model_freeze_must_precede_acquisition(self):
        packet = dict(version=audit.STUDY_VERSION, contract_fingerprint="fp", requests=requests(),
                      frozen_before_any_native_outcome=True, native_planning_queries=0, frozen_unix_s=100.)
        audit.freeze_checks(packet, dict(acquired_unix_s=101.), dict(fingerprint="fp"), audit.AuditChecks())
        for bad in (dict(acquired_unix_s=99.), dict(acquired_unix_s=True), dict(acquired_unix_s=float("nan"))):
            with self.assertRaises(audit.EvidenceError):
                audit.freeze_checks(packet, bad, dict(fingerprint="fp"), audit.AuditChecks())

    def test_boolean_zero_is_not_integer_native_query_count(self):
        packet = dict(version=audit.STUDY_VERSION, contract_fingerprint="fp", requests=requests(),
                      frozen_before_any_native_outcome=True, native_planning_queries=False, frozen_unix_s=100.)
        with self.assertRaises(audit.EvidenceError):
            audit.freeze_checks(packet, dict(acquired_unix_s=101.), dict(fingerprint="fp"), audit.AuditChecks())


class GapSelectionEvidenceChecks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        self.items = [dict(id=f"candidate-{i:02d}", values=[.5, -.2, .3, 30., .2]) for i in range(16)]
        self.order = list(reversed(range(16)))

    def tearDown(self):
        self.tmp.cleanup()

    def selection(self, flags):
        records = []
        for offset, flag in enumerate(flags):
            name = f"proof-{offset}.json"; path = self.root/name
            path.write_text(json.dumps(dict(passed=flag, reason=None if flag else "unresolved")))
            index = self.order[offset]
            records.append(dict(candidate_index=index, candidate_id=self.items[index]["id"],
                                passed=flag, reason=None if flag else "unresolved", connection_file=name,
                                connection_sha256=audit.sha(path)))
        chosen = self.order[next((i for i, f in enumerate(flags) if f), 0)]
        return dict(checks=records, selected_index=chosen)

    def check(self, selection):
        with patch("sircus.mjlab.calibrated_admission.verify_connection", return_value=True):
            return audit.selection_check(selection, self.order, self.items, self.root, {}, {}, audit.AuditChecks(), "test")

    def test_exact_first_admitted_candidate(self):
        result = self.check(self.selection([False, True]))
        self.assertEqual(result, dict(index=14, admitted=True, geometry_checks=2))

    def test_all_four_rejections_preserve_original_top_one(self):
        result = self.check(self.selection([False]*4))
        self.assertEqual(result, dict(index=15, admitted=False, geometry_checks=4))

    def test_rejected_prefix_cannot_stop_early(self):
        with self.assertRaises(audit.EvidenceError):
            self.check(self.selection([False, False]))

    def test_admitted_candidate_cannot_be_skipped(self):
        with self.assertRaises(audit.EvidenceError):
            self.check(self.selection([True, False]))

    def test_geometry_proof_tampering_rejected(self):
        selection = self.selection([True])
        (self.root/selection["checks"][0]["connection_file"]).write_text('{"passed":false}')
        with self.assertRaises(audit.EvidenceError):
            self.check(selection)

    def test_wrong_candidate_identity_rejected(self):
        selection = self.selection([True]); selection["checks"][0]["candidate_index"] = 2
        with self.assertRaises(audit.EvidenceError):
            self.check(selection)

    def test_prediction_order_must_cover_all_sixteen(self):
        selection = self.selection([True]); self.order[-1] = self.order[0]
        with self.assertRaises(audit.EvidenceError):
            self.check(selection)


class IndependentCandidateChecks(unittest.TestCase):
    def setUp(self):
        from sircus.mjlab.tests.test_gap_coverage_candidates import original_bank
        from sircus.mjlab import gap_coverage_candidates
        self.original, self.anchor = original_bank()
        self.banks = dict(consistent=dict(candidates=copy.deepcopy(self.original)),
                         gap_coverage=dict(candidates=gap_coverage_candidates.candidates(self.original, self.anchor)))

    def check(self):
        return audit.candidate_banks_check(self.banks, self.original, self.anchor,
                                          audit.AuditChecks(), "fixture")

    def test_independent_reconstruction_matches_wrapper_without_calling_it(self):
        with patch("sircus.mjlab.gap_coverage_candidates.candidates", side_effect=AssertionError("Producer not allowed")):
            expected = self.check()
        self.assertEqual(expected["gap_coverage"][8]["values"][4], self.anchor["values"][4]-.12)
        self.assertEqual(expected["gap_coverage"][9]["values"][4], self.anchor["values"][4]+.12)
        for i in set(range(16))-{8,9}:
            self.assertEqual(expected["gap_coverage"][i],self.original[i])

    def test_unsupported_negative_endpoint_is_not_repaired(self):
        from sircus.mjlab.tests.test_gap_coverage_candidates import original_bank
        from sircus.mjlab import gap_coverage_candidates
        self.original, self.anchor = original_bank(.08)
        self.banks = dict(consistent=dict(candidates=copy.deepcopy(self.original)),
                         gap_coverage=dict(candidates=gap_coverage_candidates.candidates(self.original, self.anchor)))
        self.assertAlmostEqual(self.check()["gap_coverage"][8]["values"][4],-.04)
        self.banks["gap_coverage"]["candidates"][8]["values"][4]=0.
        with self.assertRaises(audit.EvidenceError):self.check()

    def test_untouched_candidate_and_board_pose_tampering_rejected(self):
        for index, coordinate in ((0,4),(3,0),(8,0),(9,2),(15,3)):
            with self.subTest(index=index,coordinate=coordinate):
                changed=copy.deepcopy(self.banks)
                changed["gap_coverage"]["candidates"][index]["values"][coordinate]+=.001
                with self.assertRaises(audit.EvidenceError):
                    audit.candidate_banks_check(changed,self.original,self.anchor,audit.AuditChecks(),"tamper")

    def test_original_bank_or_traceability_tampering_rejected(self):
        for kind in ("baseline", "source_id", "residual", "metadata", "budget", "extra_bank"):
            changed=copy.deepcopy(self.banks)
            if kind=="baseline":changed["consistent"]["candidates"][8]["values"][4]+=.001
            elif kind=="source_id":changed["gap_coverage"]["candidates"][8]["id"]="candidate-09"
            elif kind=="residual":changed["gap_coverage"]["candidates"][8]["normalized_residual"][4]=-.9
            elif kind=="metadata":changed["gap_coverage"]["candidates"][8]["gap_coverage"]["original_candidate"]["values"][0]+=.001
            elif kind=="budget":changed["gap_coverage"]["candidates"].append(copy.deepcopy(self.original[0]))
            else:changed["extra"]={}
            with self.subTest(kind=kind),self.assertRaises(audit.EvidenceError):
                audit.candidate_banks_check(changed,self.original,self.anchor,audit.AuditChecks(),"tamper")

    def test_no_inputs_mutated(self):
        before=copy.deepcopy((self.banks,self.original,self.anchor))
        self.check()
        self.assertEqual((self.banks,self.original,self.anchor),before)


class ActorComparisonRevisionTests(unittest.TestCase):
    def case19(self):
        # Actual case19 tilt scalars expose float32 rounding amplified by the
        # eight-degree scale; the other coordinates are generic test constants.
        anchor=dict(values=[.5,-.2,.3,27.534005060312847,.2])
        saved=np.array([0.,0.,0.,.35687845945358276,0.])
        independent=np.array([0.,0.,0.,.35687875747680664,0.],dtype=np.float32)
        residual=np.tanh(saved)
        values=np.array(anchor['values'])+residual*np.array([.04,.04,.05,8.,.12])
        return dict(latent_mean=saved.tolist(),normalized_residual=residual.tolist(),values=values.tolist()),independent,anchor

    def test_case19_float32_difference_is_a_unit_error_not_changed_action(self):
        recorded,mean,anchor=self.case19()
        old_action=np.array(anchor['values'])+np.tanh(mean.astype(float))*np.array([.04,.04,.05,8.,.12])
        delta=abs(recorded['values'][3]-old_action[3])
        self.assertAlmostEqual(delta,2.1045684910347973e-6,places=15)
        with self.assertRaises(audit.EvidenceError):
            audit.AuditChecks().numeric(recorded['values'],old_action,'old mixed-unit comparison',2e-6)
        audit.actor_proposal_checks(recorded,mean,anchor,audit.AuditChecks(),'case19')
        self.assertEqual(audit.VERSION,'sircus-gap-coverage-audit-v2')
        self.assertEqual(audit.ACTOR_COMPARISON_REVISION['independent_latent_atol'],2e-6)
        self.assertEqual(audit.ACTOR_COMPARISON_REVISION['normalized_action_atol'],2e-6)

    def test_even_small_action_corruption_fails_exact_saved_derivation(self):
        for index in range(5):
            recorded,mean,anchor=self.case19();recorded['values'][index]+=1e-8
            with self.subTest(index=index),self.assertRaises(audit.EvidenceError):
                audit.actor_proposal_checks(recorded,mean,anchor,audit.AuditChecks(),'tampered action')

    def test_small_residual_corruption_fails_tanh_identity(self):
        recorded,mean,anchor=self.case19();recorded['normalized_residual'][4]+=1e-8
        recorded['values']=(np.array(anchor['values'])+np.array(recorded['normalized_residual'])*np.array([.04,.04,.05,8.,.12])).tolist()
        with self.assertRaises(audit.EvidenceError):
            audit.actor_proposal_checks(recorded,mean,anchor,audit.AuditChecks(),'tampered residual')

    def test_coherent_changed_actor_still_fails_original_independent_bound(self):
        recorded,mean,anchor=self.case19();recorded['latent_mean'][4]+=3e-6
        recorded['normalized_residual']=np.tanh(recorded['latent_mean']).tolist()
        recorded['values']=(np.array(anchor['values'])+np.array(recorded['normalized_residual'])*np.array([.04,.04,.05,8.,.12])).tolist()
        with self.assertRaises(audit.EvidenceError):
            audit.actor_proposal_checks(recorded,mean,anchor,audit.AuditChecks(),'changed actor')


class ForecastLedgerChecks(unittest.TestCase):
    """Exercise the two-bank audit wiring without Torch, native setup or GPUs."""
    def setUp(self):
        from sircus.mjlab import gap_coverage_candidates
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        (self.root/"native").mkdir()
        self.context=dict(id="gap-coverage-test-00",split="test",bucket_xy_m=[.8,-.2])
        self.anchor=dict(id="anchor",values=[.5,-.2,.3,28.,.2],bucket_xy_m=[.75,-.2])
        self.train=[dict(id="train-1",role="train",native_release=True,route_success=True,
            invalid_physics=False,route_quality=2,learning_reward=1.,bucket_xy_m=[.75,-.2],
            values=self.anchor["values"],effective_source_hint=self.anchor)]
        original=audit.old.candidates(self.context,self.anchor,self.anchor["values"],self.anchor["values"],self.train)
        self.prediction=audit.old.decode_raw(np.zeros((5,9)).tolist())
        self.resolution=self.root/"native/resolution.json";self.resolution.write_text("{}")
        self.connection=self.root/"native/connection.json";self.connection.write_text('{"passed":true,"reason":null}')
        metadata=dict(source_hint=self.anchor,effective_source_hint=self.anchor,
            context_resolution_file="resolution.json",context_resolution_sha256=audit.sha(self.resolution),
            context_home_sha256="home",context_layout_eligible=True,context_anchor_admitted=True,context_hard_reject=False)
        observation=audit.calibrated.expected_observation(self.context["bucket_xy_m"],self.anchor).tolist()
        banks={m:dict(candidates=items,predictions=[copy.deepcopy(self.prediction) for _ in items],ranking=dict(order=list(range(16))))
            for m,items in (("consistent",original),("gap_coverage",gap_coverage_candidates.candidates(original,self.anchor)))}
        selection={m:dict(checks=[dict(candidate_index=0,candidate_id="candidate-00",passed=True,reason=None,
            connection_file="connection.json",connection_sha256=audit.sha(self.connection))],selected_index=0,
            selected_values=original[0]["values"],all_geometry_checks_rejected=False) for m in audit.METHODS}
        self.log=dict(context_id=self.context["id"],bucket_xy_m=self.context["bucket_xy_m"],anchor=self.anchor,
            context_metadata=metadata,observation=observation,native_queries=0,
            actor_values={m:dict(values=self.anchor["values"],normalized_residual=[0.]*5,latent_mean=[0.]*5) for m in ("bc","ppo")},
            banks=banks,selection=selection)
        rows=[]
        for method in audit.METHODS:
            rows.append(dict(id="gap-coverage-"+method+"-"+self.context["id"],role="test",method=method,
                context_id=self.context["id"],bucket_xy_m=self.context["bucket_xy_m"],**metadata,
                values=original[0]["values"],command_values=original[0]["values"],observation=observation,
                source_teacher_id=self.anchor["id"],candidate_index=0,candidate_id="candidate-00",ranking_sha256=audit.digest(self.log),
                geometry_checks=1,model_candidates=16,native_planning_queries=0,selection_or_learning=False,
                ball_flight_search_at_inference=False,perturbation_mixed_units=[0.]*5,
                frozen_artifact_sha256=audit.original.ARTIFACT_SHA["ensemble"],
                prediction_model_sha256=audit.original.ARTIFACT_SHA["ensemble"],model_prediction=self.prediction))
        self.packet=dict(requests=rows)
        self.spec=dict(splits=dict(test=[self.context]),scene={},native_reference=dict(allocated_options=dict(native=dict(gravity=[0,0,-9.81]))))
        for name,value in (("setup-home",{}),("prior",dict(records=[])),("calibration",dict(score_config={}))):
            (self.root/(name+".json")).write_text(json.dumps(value))

    def tearDown(self):self.tmp.cleanup()

    def check(self):
        (self.root/"rankings.json").write_text(json.dumps([self.log]))
        with ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules,{"torch":SimpleNamespace(load=lambda *a,**k:{})}))
            stack.enter_context(patch.object(audit.old.common,"load_checkpoint",return_value=dict(policy={})))
            stack.enter_context(patch.object(audit.old.common,"network_forward",return_value=np.zeros(5)))
            stack.enter_context(patch.object(audit.old,"ensemble_forward",side_effect=lambda checkpoint,x:[copy.deepcopy(self.prediction) for _ in x]))
            stack.enter_context(patch("sircus.mjlab.consistent_flight_selector.rank",return_value=dict(order=list(range(16)))))
            stack.enter_context(patch("sircus.mjlab.calibrated_admission.verify_resolution",return_value=True))
            stack.enter_context(patch("sircus.mjlab.calibrated_admission.verify_connection",return_value=True))
            return audit._forecasts(self.root,self.packet,self.spec,self.train,audit.AuditChecks())

    def test_complete_two_bank_ledger_passes_independent_wiring(self):
        result=self.check()
        self.assertEqual(set(result),{(self.context["id"],m) for m in audit.METHODS})
        self.assertTrue(all(r==dict(index=0,admitted=True,geometry_checks=1) for r in result.values()))

    def test_actor_prediction_cannot_change_without_weights(self):
        self.log["actor_values"]["bc"]["latent_mean"][4]=.01
        with self.assertRaises(audit.EvidenceError):self.check()

    def test_selected_request_must_match_its_frozen_bank(self):
        self.packet["requests"][1]["values"]=self.packet["requests"][1]["values"].copy()
        self.packet["requests"][1]["values"][4]+=.001
        with self.assertRaises(audit.EvidenceError):self.check()

    def test_native_query_and_prediction_tampering_rejected(self):
        self.packet["requests"][1]["native_planning_queries"]=1
        with self.assertRaises(audit.EvidenceError):self.check()
        self.packet["requests"][1]["native_planning_queries"]=0
        self.log["banks"]["gap_coverage"]["predictions"][9]["capture_probability"]+=.001
        with self.assertRaises(audit.EvidenceError):self.check()


class ProducerContractAgreement(unittest.TestCase):
    def test_producer_and_independent_split_agree(self):
        from sircus.mjlab import gap_coverage_study as producer
        reference=dict(excluded=[[.1,-.5],[.7,.2]],splits=dict(
            test=[dict(bucket_xy_m=[.3+.02*i,-.3]) for i in range(30)]))
        splits,excluded=producer.fresh_splits(reference)
        expected=audit.expected_splits(reference)
        self.assertEqual((splits,excluded),(expected["splits"],expected["excluded"]))
        self.assertEqual(len(splits["test"]),50)
        self.assertEqual(splits["test"][0]["id"],"gap-coverage-test-00")
        self.assertEqual(splits["test"][-1]["id"],"gap-coverage-test-49")

    def test_new_protocol_does_not_mutate_old_budget_or_guard(self):
        from sircus.mjlab import gap_coverage_study as producer
        from sircus.mjlab import consistent_flight_verify as old_guard
        protocol=producer.protocol()
        self.assertEqual((producer.VERSION,producer.SEED,producer.MAX_REQUESTS),
                         (audit.STUDY_VERSION,audit.SEED,audit.MAX_REQUESTS))
        self.assertEqual(protocol["methods"],list(audit.METHODS))
        self.assertEqual((protocol["test_layouts"],protocol["control_requests"],protocol["maximum_fresh_requests"]),(50,12,112))
        self.assertEqual((protocol["candidates"],protocol["maximum_geometry_checks"]),(16,4))
        self.assertEqual(protocol["changed_candidate_indices"],[8,9])
        self.assertEqual(protocol["gap_residual_endpoints"],[-1,1])
        self.assertEqual(protocol["unchanged_candidate_count"],14)
        self.assertEqual(old_guard.MAX_REQUESTS,132)
        self.assertEqual(old_guard.STUDY_VERSION,"sircus-consistent-flight-v1")
        self.assertEqual(old_guard.METHODS,("bc","ppo","ensemble","consistent"))


if __name__ == "__main__":
    unittest.main()
