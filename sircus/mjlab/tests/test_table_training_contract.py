"""TRAIN label/action binding, fresh layout inventory and finite v3 protocol."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from sircus.mjlab import table_training_contract as C
from sircus.mjlab.benchmark_contract import observation


def scored(identity="train-1", **updates):
    row = dict(id=identity, context_id="context-1", role="train", native_release=True,
        invalid=None, invalid_physics=False, route_success=True, route_quality=2,
        board_bounce=True, whole_ball_entry=True, success=True, max_retention_s=1.,
        closest_opening_distance_m=.01, forbidden_contact_steps=0, table_contact_steps=0,
        single_board_bounce_success=True, direct_capture=True, table_assisted_capture=False,
        scoring_version=C.SCORING_VERSION, route_version=C.ROUTE_VERSION, reward_version=C.REWARD_VERSION)
    row.update(updates)
    return row


def pair():
    anchor = dict(id="geometry-1", bucket_xy_m=[.3,-.4], values=[.55,-.2,.3,28.,.35])
    raw = scored(scoring_version="sircus-calibrated-native-capture-v2", route_success=False,
        route_quality=0, method="ppo", seed=20261411, train_index=0,
        bucket_xy_m=[.3,-.4], values=list(anchor["values"]), perturbation_mixed_units=[0.]*5,
        normalized_residual=[0.]*5, effective_source_hint=anchor,
        observation=observation([.3,-.4],anchor).tolist(), contact_protocol={"substeps":64},
        native_cache_sha256="a"*64, native_cache_row=3, calibration_fingerprint="b"*64,
        scalar_score_audit_passed=True)
    new = scored(table_contact_steps=5,direct_capture=False,table_assisted_capture=True)
    old = {k:raw[k] for k in ("route_success","success","invalid_physics","board_bounce","whole_ball_entry")}
    derived = dict(source_study="calibrated-benchmark-v2", phase="train", id=raw["id"],
                   status="scored", original=copy.deepcopy(raw), old_score=old, new_score=new)
    return raw,derived,{"context-1":[.3,-.4]}


class DatasetBindingTests(unittest.TestCase):
    def test_audited_table_credit_preserves_action_and_source_identity(self):
        raw, derived, train = pair(); before=copy.deepcopy(raw)
        row=C.join_training_row(raw,derived,"calibrated-benchmark-v2",train)
        self.assertEqual(raw,before)
        self.assertTrue(row["newly_credited_capture"])
        self.assertFalse(row["original_route_success"])
        self.assertEqual(row["source_record_sha256"],C.digest(raw))
        self.assertEqual(row["values"],raw["values"])
        self.assertEqual(row["native_cache_sha256"],raw["native_cache_sha256"])
        self.assertEqual(row["native_cache_row"],3)
        self.assertAlmostEqual(row["learning_reward"],C.reward(row))

    def test_reject_heldout_wrong_coordinate_noise_and_action_reassociation(self):
        mutations = [lambda r,d:r.update(role="test"),lambda r,d:r.update(bucket_xy_m=[.2,.1]),
            lambda r,d:r.update(perturbation_mixed_units=[.001]*5),
            lambda r,d:d.update(phase="test"),lambda r,d:d.update(source_study="policy64-v1"),
            lambda r,d:d["original"].update(values=[.5]*5),
            lambda r,d:d["original"].update(native_cache_row=4),
            lambda r,d:d["original"].update(native_cache_sha256="c"*64),
            lambda r,d:r.update(contact_protocol={"substeps":128}),
            lambda r,d:r.update(observation=[0.]*9),lambda r,d:r.update(normalized_residual=[.1]*5),
            lambda r,d:d["old_score"].update(route_success=True)]
        for index,mutate in enumerate(mutations):
            raw,derived,train=pair();mutate(raw,derived)
            with self.subTest(index=index),self.assertRaises(ValueError):
                C.join_training_row(raw,derived,"calibrated-benchmark-v2",train)

    def test_nonrelease_keeps_negative_reward_without_fabricated_trace(self):
        raw,derived,train=pair()
        raw.update(native_release=False,invalid="finite_ik_unresolved")
        derived.update(status="rejected",original=copy.deepcopy(raw))
        row=C.join_training_row(raw,derived,"calibrated-benchmark-v2",train)
        self.assertEqual(row["learning_reward"],-1.)
        self.assertFalse(C.captured(row))
        self.assertFalse(row["newly_credited_capture"])

    def test_one_teacher_per_context_prefers_clean_reward_then_stable_id(self):
        rows=[scored("rim",route_quality=1,closest_opening_distance_m=0.),
              scored("z",closest_opening_distance_m=.2),scored("a",closest_opening_distance_m=.2),
              scored("table",context_id="context-2",direct_capture=False,table_assisted_capture=True)]
        selected=C.select_teachers(rows)
        self.assertEqual([r["id"] for r in selected],["a","table"])
        self.assertEqual(selected,C.select_teachers(list(reversed(rows))))
        with self.assertRaisesRegex(ValueError,"Only TRAIN"):
            C.select_teachers([scored(role="validation")])

    def test_statistics_partitions_all_requests_and_keeps_table_routes(self):
        rows=[scored(),scored("table",direct_capture=False,table_assisted_capture=True),
              scored("rejected",native_release=False,invalid="ik"),
              scored("badphysics",invalid_physics=True,route_success=False),
              scored("robot",route_success=False,forbidden_contact_steps=3)]
        counts=C.statistics(rows)
        self.assertEqual((counts["attempts"],counts["native_releases"],counts["setup_rejections"]),(5,4,1))
        self.assertEqual((counts["captures"],counts["native_misses"]),(2,2))
        self.assertEqual((counts["direct_captures"],counts["table_assisted_captures"]),(1,1))
        self.assertEqual(counts["forbidden_contact_shots"],1)


class SplitTests(unittest.TestCase):
    def test_budget_and_schedule_are_finite_balanced_and_reproducible(self):
        p=C.protocol();self.assertEqual(p["maximum_fresh_requests"],766)
        self.assertEqual(sum(p[k] for k in ("train_requests","validation_requests","test_requests","control_requests")),766)
        self.assertEqual(p["updates"]*p["batch_size"],512)
        self.assertEqual(p["milestones"],[0,256,512])
        self.assertEqual(p["test_methods"],["old_bc","old_ppo","bc","ppo"])
        self.assertEqual(len(C.schedule()),512);self.assertEqual(C.schedule(),C.schedule())
        self.assertEqual(set(np.bincount(C.schedule()).tolist()),{14,15})

    def test_coordinates_include_all_roles_and_outcomes_from_diagnostics(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);output=root/"output";repo=root/"repo"
            (repo/"baselines"/"published").mkdir(parents=True);(output/"diagnostic").mkdir(parents=True)
            (repo/"baselines"/"published"/"custom.json").write_text(json.dumps({"nested":{"bucket_xy_m":[.4,.2]}}))
            path=output/"diagnostic"/"records.jsonl"
            path.write_text('\n'.join(json.dumps({"bucket_xy_m":p,"route_success":success,"role":role})
                for p,success,role in (([.2,.3],False,"test"),([.5,.6],True,"train"))))
            (output/"diagnostic"/"exclusions.json").write_text(json.dumps([[.7,.8],[.9,1.]]))
            first=C.collect_exclusions(output,repo)
            # Top-level exclusions also need to remain in the split inventory.
            self.assertEqual(first["coordinates"],[[.2,.3],[.4,.2],[.5,.6],[.7,.8],[.9,1.]])
            self.assertEqual(len(first["files_sha256"]),3)
            path.write_text(path.read_text().replace('false','true'))
            second=C.collect_exclusions(output,repo)
            self.assertEqual(first["coordinates"],second["coordinates"])
            self.assertNotEqual(first["files_sha256"],second["files_sha256"])

    def test_fresh_splits_keep_training_and_five_mm_separation(self):
        repo=Path(__file__).resolve().parents[3]
        source=C.read(repo/"baselines/mjlab-calibrated-benchmark-20261104/splits.json")
        excluded=[[.2,.1],[.5,.2]]
        pools,before=C.make_splits(source,excluded)
        self.assertEqual([len(pools[k]) for k in ("train","validation","test")],[35,12,50])
        self.assertEqual((pools,before),C.make_splits(source,excluded))
        fresh=np.asarray([r["bucket_xy_m"] for role in ("validation","test") for r in pools[role]])
        self.assertGreaterEqual(np.linalg.norm(fresh[:,None]-np.asarray(before)[None,:],axis=-1).min(),.005)
        pairwise=np.linalg.norm(fresh[:,None]-fresh[None,:],axis=-1);np.fill_diagonal(pairwise,np.inf)
        self.assertGreaterEqual(pairwise.min(),.005)


if __name__ == "__main__":
    unittest.main()
