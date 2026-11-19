"""Final assessment software checks using fabricated states/models/outcomes."""

import ast
import copy
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import candidate_continuation_assessment as subject
import candidate_continuation_assessment_native as native
from candidate_continuation_ledger import Ledger,Outcome,canonical_hash,ASSESSMENT_COHORTS
from candidate_continuation_policy import freeze_support
from continuous_feasibility import decode_action,LOW,HIGH,canonical_hash as geometry_hash
from board_route import BoardRoute
from entry_quality import EntryQuality
from rl_contract import ShotScore


def original_parallel_shot_batch():
    """Execute the original method body without importing Torch or an app."""
    path=Path(__file__).resolve().parents[1]/"parallel_runtime.py"
    module=ast.parse(path.read_text(),filename=str(path))
    cls=next(node for node in module.body if isinstance(node,ast.ClassDef) and node.name=="ParallelShots")
    method=next(node for node in cls.body if isinstance(node,ast.FunctionDef) and node.name=="shot_batch")
    namespace=dict(np=np,time=time,ShotScore=ShotScore,EntryQuality=EntryQuality,BoardRoute=BoardRoute)
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),"exec"),namespace)
    return namespace["shot_batch"]


def complete_train_ledger():
    """Accounting mock only; no model, optimizer, environment or shot exists."""
    ledger=Ledger.new(tuple(f"train-{i:03d}" for i in range(35)),
        tuple(f"validation-{i:03d}" for i in range(35)),tuple(f"test-{i:03d}" for i in range(70)))
    while (batch:=ledger.next_batch()) is not None:
        ledger=ledger.reserve_batch(batch)
        ledger=ledger.record_batch(tuple(Outcome(r.request_id,True,None,"0"*64) for r in batch.requests))
    return ledger


def assessment_inputs(ledger):
    cells_by_split={split:[dict(id=key,split=split,bucket_xy_m=[.6+i*.001,-.1]) for i,key in enumerate(ids)]
        for split,ids in (("train",ledger.train_context_ids),("validation",ledger.validation_context_ids),("test",ledger.test_context_ids))}
    supports,records,parent,continued={},{},{},{}
    empty={"validation-024","test-024","test-058"}
    for cell in sum(cells_by_split.values(),[]):
        key=cell["id"];values=((LOW+HIGH)/2).tolist()
        direct=decode_action(cell,values,"parent-"+key)
        alternate=list(values);alternate[1]+=.05
        rows=[dict(candidate_id=key+"-parent",action=direct),
              dict(candidate_id=key+"-other",action=decode_action(cell,alternate,"other-"+key))]
        support=freeze_support(key,rows,direct,[0.]*5)
        supports[key]=None if key in empty else support
        for row in rows:
            decision=dict(accepted=True,release_authorized=False,action=row["action"],binding=dict(context=dict(id=key)),
                qualification=dict(accepted=True,ik={n:dict(q=[0.]*6,solved=True) for n in ("Arm10","Arm14")}),
                path=dict(accepted=True))
            decision["fingerprint"]=geometry_hash(decision)
            records[row["candidate_id"]]=dict(row,cell_id=key,accepted=True,
                decision=decision)
        parent[key]=dict(exact_action=direct,latent_mean=[0.]*5,latent_scale=[1.]*5)
        continued[key]=dict(latent_mean=list(support.latents[1]),latent_scale=[1.]*5)
    return cells_by_split,supports,records,parent,continued


class AssessmentFixture:
    @classmethod
    def setUpClass(cls):
        cls.train=complete_train_ledger()
        cls.inputs=assessment_inputs(cls.train)

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.parent_path=self.root/"parent.pt";self.parent_path.write_bytes(b"synthetic parent")
        self.final_path=self.root/"final.pt";self.final_path.write_bytes(b"synthetic final")
        self.cells,self.supports,self.records,self.parent,self.continued=copy.deepcopy(self.inputs)
        self.hashes=dict(parent_weight_hash="c"*64,continued_weight_hash="d"*64)
        self.bindings=dict(source_sha256="a"*64,native_binding_sha256="b"*64)
        self.source=patch.object(subject,"source_fingerprint",return_value="a"*64)
        self.source.start();self.addCleanup(self.source.stop)
        self.plan=self.build_plan()
        self.clock=types.SimpleNamespace(value=0.)
        self.free=types.SimpleNamespace(value=10*1024**3)

    def build_plan(self,**changes):
        kwargs=dict(final_checkpoint_sha256=subject.digest(self.final_path),cells_by_split=self.cells,
            supports=self.supports,prepared_records=self.records,parent_distributions=self.parent,
            continued_distributions=self.continued,model_hashes=self.hashes,bindings=self.bindings)
        kwargs.update(changes)
        return subject.build_assessment_plan(self.train,**kwargs)

    def runner(self,**changes):
        kwargs=dict(parent_checkpoint_path=self.parent_path,final_checkpoint_path=self.final_path,
            state_hashes=lambda:copy.deepcopy(self.hashes),synthetic=True,clock=lambda:self.clock.value,
            free_bytes=lambda:self.free.value)
        kwargs.update(changes)
        return subject.AssessmentRunner(self.root/"assessment",self.train,self.plan,**kwargs)

    @staticmethod
    def results(choices):
        return [dict(request_id=c["request_id"],released=False,
            decline_reason=c["decline_reason"] or "synthetic_guard_decline",source_sha256="e"*64,
            captured=False,clean=False,**{"return":-1.}) for c in choices]


class AssessmentTests(AssessmentFixture,unittest.TestCase):
    def test_exact_all_three_methods_order_empty_support_denominators_and_seeds(self):
        plan=self.plan
        self.assertEqual(len(plan["choices"]),420)
        self.assertEqual([(c["split"],c["offset"],c["method"]) for c in plan["cohorts"]],list(ASSESSMENT_COHORTS))
        self.assertEqual(sum(not c["enabled"] for c in plan["choices"]),6)
        parent=[c for c in plan["choices"] if c["method"]=="parent"]
        bank=[c for c in plan["choices"] if c["method"]=="candidate_bank"]
        continued=[c for c in plan["choices"] if c["method"]=="continued"]
        self.assertEqual(len(parent),140);self.assertTrue(all(c["enabled"] and c["ik_seeds"] is None for c in parent))
        self.assertTrue(all(c["index"]==0 and c["ik_seeds"] is None for c in bank if c["enabled"]))
        self.assertTrue(all(c["index"]==1 and c["ik_seeds"] is not None for c in continued if c["enabled"]))
        self.assertEqual(len({(c["context_id"],c["method"]) for c in plan["choices"]}),420)

    def test_420_fixed_requests_complete_without_any_update_interface(self):
        runner=self.runner()
        for _ in range(12):
            choices=runner.reserve_batch();runner.record_results(self.results(choices))
        result=runner.finish()
        self.assertEqual(result["requests"],420);self.assertEqual(result["optimizer_updates"],0)
        self.assertEqual(result["status"],"assessment_complete_audit_pending")
        self.assertTrue(result["ledger"]["counters"]["assessment_complete"])
        self.assertFalse(result["model_promoted"])
        with self.assertRaises(ValueError):runner.reserve_batch()
        self.assertEqual(self.final_path.read_bytes(),b"synthetic final")

    def test_checkpoint_frozen_before_heldout_actions_and_changed_bytes_rejected(self):
        self.final_path.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError,"checkpoint/model changed"):self.runner()
        self.assertFalse((self.root/"assessment").exists())

    def test_training_must_be_finished_and_all_fresh_positions_fixed(self):
        incomplete=Ledger.new(self.train.train_context_ids,self.train.validation_context_ids,self.train.test_context_ids)
        with self.assertRaisesRegex(ValueError,"capped TRAIN"):subject.build_assessment_plan(incomplete,
            final_checkpoint_sha256=subject.digest(self.final_path),cells_by_split=self.cells,supports=self.supports,
            prepared_records=self.records,parent_distributions=self.parent,continued_distributions=self.continued,
            model_hashes=self.hashes,bindings=self.bindings)
        self.cells["test"].pop()
        with self.assertRaisesRegex(ValueError,"order changed"):self.build_plan()

    def test_ordered_outcome_prefix_preserves_charged_requests(self):
        runner=self.runner();choices=runner.reserve_batch();results=self.results(choices)
        runner.record_results(results[:7])
        self.assertEqual(runner.ledger.summary()["assessment"]["requested"],35)
        self.assertEqual(runner.ledger.summary()["assessment"]["unresolved"],28)
        runner.record_results(results[7:])
        self.assertIsNone(runner.ledger.pending)

    def test_no_retry_no_replacement_no_partial_promotion(self):
        runner=self.runner();runner.reserve_batch()
        with self.assertRaises(ValueError):runner.reserve_batch()
        self.assertTrue(runner.failed)
        self.assertEqual(runner.ledger.summary()["assessment"]["requested"],35)
        with self.assertRaises(ValueError):runner.finish()
        with self.assertRaises(FileExistsError):self.runner()

    def test_no_weight_update_or_source_change_after_freeze(self):
        runner=self.runner();choices=runner.reserve_batch()
        self.hashes["continued_weight_hash"]="f"*64
        with self.assertRaisesRegex(ValueError,"model changed"):runner.record_results(self.results(choices))
        self.assertEqual(runner.ledger.summary()["assessment"]["unresolved"],35)

    def test_storage_clock_and_assessment_wall_cap_stop_without_refund(self):
        runner=self.runner();runner.reserve_batch();self.clock.value=1800.001
        with self.assertRaisesRegex(ValueError,"wall limit"):runner.record_results(self.results(self.plan["choices"][:35]))
        self.assertEqual(runner.ledger.summary()["assessment"]["requested"],35)
        self.assertTrue(runner.failed)

    def test_result_order_duplicate_reward_or_empty_support_release_fails(self):
        runner=self.runner();choices=runner.reserve_batch();results=self.results(choices)
        results[0]["request_id"]=results[1]["request_id"]
        with self.assertRaisesRegex(ValueError,"order changed"):runner.record_results(results)

    def test_nonfinite_gaussian_and_unknown_candidate_are_not_filtered(self):
        self.continued["test-000"]["latent_mean"][0]=float("nan")
        with self.assertRaisesRegex(ValueError,"Gaussian"):self.build_plan()

    def test_parent_priority_then_lexical_tie_are_deterministic(self):
        support=self.supports["train-000"]
        index,_=subject.deterministic_index(support,self.parent["train-000"])
        self.assertEqual(index,support.parent_index)

    def test_empty_support_cannot_be_released_or_replaced(self):
        runner=self.runner()
        for _ in range(3):
            choices=runner.reserve_batch();runner.record_results(self.results(choices))
        choices=runner.reserve_batch();results=self.results(choices)
        empty=next(i for i,c in enumerate(choices) if not c["enabled"])
        results[empty].update(released=True,decline_reason=None,**{"return":-.8})
        with self.assertRaisesRegex(ValueError,"Empty support"):runner.record_results(results)
        self.assertEqual(runner.ledger.summary()["assessment"]["requested"],140)

    def test_prepared_fit_tampering_is_not_silently_accepted(self):
        self.records["train-000-other"]["decision"]["qualification"]["ik"]["Arm10"]["q"][0]=1.
        with self.assertRaisesRegex(ValueError,"prepared qualification"):self.build_plan()

    def test_monotonic_clock_and_storage_floor_checked_between_cohorts(self):
        runner=self.runner();self.clock.value=10.;choices=runner.reserve_batch()
        self.clock.value=9.
        with self.assertRaisesRegex(ValueError,"backwards"):runner.record_results(self.results(choices))


class FakeNativeEnv:
    """Original batch boundary with fake setup/physics and real CPU scorers."""
    shot_batch=original_parallel_shot_batch()

    def __init__(self,directory,cells):
        self.dest=directory;self.cfg=dict(simulation=dict(physics_dt_s=1/240),ball=dict(radius_m=.0375),
            bucket=dict(inner_top_diameter_m=.177,inner_bottom_diameter_m=.15,assumed_interior_depth_m=.175,
                assumed_bottom_thickness_m=.008,segments=64))
        self.base_cells=copy.deepcopy(cells);self.spec={};self.origins=np.zeros((35,3));self.dt=1/240
        self.calls=[];self.steps=0;self.physics_samples=[];self.fail=False;self.n=35
        self.placements=[];self.tick_calls=0
    def home(self):self.calls.append("home");return dict(synthetic=True)
    def prepare(self,actions):
        self.calls.append("shot")
        assert list((self.dest/"native").glob("*.dispatch.json")),"No durable pre-call claim"
        if self.fail:raise RuntimeError("synthetic interrupted cohort")
        enabled=self.active_mask
        self.preflight_records=[dict(accepted=False) if use else None for use in enabled]
        self.qualifications=[dict(accepted=False,reason="synthetic_guard_decline" if use else "inactive_split") for use in enabled]
        self.path_reasons=[None]*35;self.path_checks=[None]*35
        self.scoring_normals=np.tile([0.,0.,1.],(35,1));self.transit_history=[];self.correction_history=[];self.release_step=self.steps
        invalid=["synthetic_guard_decline" if use else "inactive_split" for use in enabled]
        return np.tile([.4,-.1,.5],(35,1)),self.scoring_normals,[[dict(synthetic=True)] for _ in range(35)],invalid
    def set_balls(self,xyz):self.placements.append(np.asarray(xyz).copy())
    def render(self):pass
    def assert_buckets_fixed(self):pass
    def tick(self):
        self.tick_calls+=1
        raise AssertionError("All synthetic setup declines must stay parked without physics ticks")


class NativeAssessmentAdapterTests(AssessmentFixture,unittest.TestCase):
    """Same input fixture, with a fake initialized app and raw evidence writer."""

    def producer(self):
        env=FakeNativeEnv(self.root,self.cells["train"])
        binding=dict(source_sha256="a"*64,fingerprint="b"*64,startup_control_sha256={})
        train=types.SimpleNamespace(env=env,binding=binding,original_cells=copy.deepcopy(self.cells["train"]),
            frozen_root=self.root/"frozen",_validate_binding=lambda:None)
        control=patch.object(native,"verify_controls",return_value=dict(synthetic=True));control.start();self.addCleanup(control.stop)
        return native.GuardedAssessmentProducer(train,self.plan)

    def test_native_adapter_calls_same_app_once_and_durable_claim_blocks_retry(self):
        runner=self.runner();producer=self.producer();choices=runner.reserve_batch()
        results=producer.execute_reserved(runner,choices,expected_journal_head=runner.journal.head)
        self.assertEqual(producer.env.calls,["home","shot"])
        self.assertEqual(len(results),35);self.assertTrue(all(not r["released"] for r in results))
        packet=json.loads(gzip.decompress((runner.directory/"native/assessment-000000.json.gz").read_bytes()))
        self.assertTrue(packet["synthetic"]);self.assertEqual(packet["method"],"parent")
        self.assertEqual(packet["model_hashes"],self.hashes)
        with self.assertRaises(ValueError):producer.execute_reserved(runner,choices,expected_journal_head=runner.journal.head)
        self.assertEqual(producer.env.calls,["home","shot"])

    def test_all_420_native_mock_requests_include_six_empty_support_declines(self):
        runner=self.runner();producer=self.producer();all_results=[]
        for _ in range(12):
            choices=runner.reserve_batch()
            results=producer.execute_reserved(runner,choices,expected_journal_head=runner.journal.head)
            all_results.extend(results);runner.record_results(results)
        runner.finish()
        self.assertEqual(len(all_results),420)
        self.assertEqual(sum(r["decline_reason"]=="no_qualified_candidate_support" for r in all_results),6)
        self.assertEqual(producer.env.calls,["home","shot"]*12)
        packet=json.loads(gzip.decompress((runner.directory/"native/assessment-000003.json.gz").read_bytes()))
        self.assertEqual(packet["rows"][24]["native_invalid"],"no_selected_setting")
        self.assertEqual(packet["rows"][24]["invalid"],"no_qualified_candidate_support")
        self.assertEqual(packet["enabled"].count(False),1)
        self.assertEqual(packet["rows"][24]["qualification"],dict(accepted=False,reason="inactive_split"))
        self.assertIsNone(packet["preflight_records"][24]);self.assertIsNone(packet["rows"][24]["path_proxy"])
        self.assertIsNone(packet["rows"][24]["path_rejection"]);self.assertIsNone(packet["reward_features"][24])
        self.assertFalse(packet["rows"][24]["success"]);self.assertFalse(packet["rows"][24]["route_success"])
        self.assertEqual(packet["rewards"][24],-1.);self.assertEqual(packet["trajectory_m"][24],[])
        self.assertIsNone(packet["physics"]);self.assertEqual(packet["release_step"],packet["score_end_step"])
        self.assertEqual(producer.env.tick_calls,0);self.assertEqual(len(producer.env.placements),12)
        self.assertTrue(all(np.array_equal(xyz,np.tile([.1,-.8,.04],(35,1))) for xyz in producer.env.placements))
        self.assertEqual(runner.ledger.summary()["assessment"]["requested"],420)
        self.assertEqual(runner.ledger.summary()["assessment"]["released"],0)

    def test_original_native_boundary_cannot_launder_executed_or_altered_empty_slots(self):
        original_root=self.root
        changes=[("wrong raw boundary label",lambda env,rows,i:rows[i].update(invalid="inactive_split")),
            ("executed row",lambda env,rows,i:rows[i].update(invalid=None)),
            ("fabricated preflight",lambda env,rows,i:env.preflight_records.__setitem__(i,dict(accepted=False))),
            ("accepted qualification",lambda env,rows,i:env.qualifications.__setitem__(i,dict(accepted=True))),
            ("altered inactive qualification",lambda env,rows,i:env.qualifications.__setitem__(i,dict(accepted=False,reason="other"))),
            ("path check",lambda env,rows,i:env.path_checks.__setitem__(i,dict(accepted=True))),
            ("path decline",lambda env,rows,i:env.path_reasons.__setitem__(i,"other")),
            ("success",lambda env,rows,i:rows[i].update(success=True)),
            ("capture",lambda env,rows,i:rows[i].update(route_success=True)),
            ("clean",lambda env,rows,i:rows[i].update(route_quality=2))]
        for index,(label,change) in enumerate(changes):
            with self.subTest(change=label):
                self.root=original_root/f"changed-{index}";self.root.mkdir()
                runner=self.runner();producer=self.producer()
                for _ in range(3):
                    earlier=runner.reserve_batch();runner.record_results(self.results(earlier))
                choices=runner.reserve_batch();empty=next(i for i,c in enumerate(choices) if not c["enabled"])
                boundary=producer.env.shot_batch
                def changed(actions,*,record_video,enabled):
                    rows,raw=boundary(actions,record_video=record_video,enabled=enabled)
                    change(producer.env,rows,empty)
                    return rows,raw
                producer.env.shot_batch=changed
                with self.assertRaisesRegex(ValueError,"Empty bank slot"):
                    producer.execute_reserved(runner,choices,expected_journal_head=runner.journal.head)
                self.assertTrue(runner.failed);self.assertEqual(producer.env.tick_calls,0)
                self.assertEqual(runner.ledger.summary()["assessment"]["requested"],140)
                self.assertEqual(runner.ledger.summary()["assessment"]["unresolved"],35)
                self.assertTrue((runner.directory/"native/assessment-000003.dispatch.json").is_file())
                self.assertTrue((runner.directory/"native/assessment-000003.failure.json").is_file())
                with self.assertRaises(ValueError):producer.execute_reserved(runner,choices,expected_journal_head=runner.journal.head)
        self.root=original_root

    def test_interrupted_native_cohort_is_preserved_and_not_retried(self):
        runner=self.runner();producer=self.producer();choices=runner.reserve_batch();producer.env.fail=True
        with self.assertRaisesRegex(RuntimeError,"interrupted"):producer.execute_reserved(runner,choices,expected_journal_head=runner.journal.head)
        self.assertTrue(runner.failed)
        self.assertTrue((runner.directory/"native/assessment-000000.failure.json").exists())
        self.assertEqual(runner.ledger.summary()["assessment"]["requested"],35)
        with self.assertRaises(ValueError):producer.execute_reserved(runner,choices,expected_journal_head=runner.journal.head)
        self.assertEqual(producer.env.calls,["home","shot"])


if __name__=="__main__":unittest.main()
