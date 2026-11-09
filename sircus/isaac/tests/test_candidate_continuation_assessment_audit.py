"""Fabricated assessment evidence only; no actual model, app or shots."""

import copy
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import candidate_continuation_assessment as dispatcher
import candidate_continuation_assessment_audit as subject
import candidate_continuation_native as native
from candidate_continuation_ledger import Ledger, Outcome, SourceBindings, canonical_hash
from test_candidate_continuation_assessment import assessment_inputs
from test_candidate_continuation_native_guards import synthetic_guard_packet
from candidate_continuation_policy import freeze_support
from continuous_feasibility import decode_action, LOW, HIGH


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def history(bindings):
    ledger = Ledger.new(tuple(f"train-{i:03d}" for i in range(35)),
        tuple(f"validation-{i:03d}" for i in range(35)),tuple(f"test-{i:03d}" for i in range(70)),bindings=bindings)
    while (batch := ledger.next_batch()) is not None:
        ledger = ledger.reserve_batch(batch)
        ledger = ledger.record_batch(tuple(Outcome(r.request_id,True,None,"0"*64) for r in batch.requests))
    return ledger


class AssessmentAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        empty = Ledger.new(tuple(f"train-{i:03d}" for i in range(35)),
            tuple(f"validation-{i:03d}" for i in range(35)),tuple(f"test-{i:03d}" for i in range(70)))
        cls.inputs = assessment_inputs(empty)
        support_sha = canonical_hash([cls.inputs[1][key].to_dict() for key in empty.train_context_ids])
        cls.train = history(SourceBindings(train_support_sha256=support_sha))

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.directory = self.root/"assessment"
        self.cells,self.supports,self.records,self.parent,self.continued = copy.deepcopy(self.inputs)
        self.scene = json.loads((Path(__file__).resolve().parents[1]/"scene.json").read_bytes())
        self.parent_path,self.final_path = self.root/"parent.pt",self.root/"final.pt"
        self.parent_path.write_bytes(b"synthetic parent");self.final_path.write_bytes(b"synthetic final")
        self.camera,self.poses = self.root/"camera.json",self.root/"poses.json"
        self.camera.write_text("{}")
        self.poses.write_text(json.dumps(dict(dropper_joint_rad=[0.]*6,board_joint_rad=[0.]*6)))
        xml = '<robot>'+''.join(f'<joint name="joint{i}" type="revolute"><origin xyz="0 0 0" rpy="0 0 0"/><axis xyz="0 0 1"/><limit lower="-6.28" upper="6.28"/></joint>' for i in range(1,7))+'</robot>'
        self.assets = {n:self.root/(n+".urdf") for n in ("Arm10","Arm14")}
        for path in self.assets.values():path.write_text(xml)
        for name in ("probes.json","rim-rays.json"):(self.root/name).write_text("{}")
        self.hashes = dict(parent_weight_hash="c"*64,continued_weight_hash="d"*64)
        self.controls = dict(mock_startup=True)
        self.patches = [patch.object(native,"source_fingerprint",return_value="a"*64),
            patch.object(dispatcher,"source_fingerprint",return_value="a"*64),
            patch.object(subject,"source_fingerprint",return_value="a"*64),
            patch.object(subject,"verify_controls",return_value=self.controls),
            patch.object(subject,"verify_assessment_packet",side_effect=self.fake_packet)]
        for p in self.patches:p.start();self.addCleanup(p.stop)
        train_records = {k:v for k,v in self.records.items() if v["cell_id"] in self.train.train_context_ids}
        self.native_binding = native.make_binding(scene=self.scene,geometry_fingerprint="b"*64,
            camera_sha256=digest(self.camera),assets_sha256={n:digest(p) for n,p in self.assets.items()},
            poses_sha256=digest(self.poses),cells=self.cells["train"],canonical_home_rad={"Arm10":[0.]*6,"Arm14":[0.]*6},
            supports=[self.supports[k] for k in self.train.train_context_ids],prepared_records=train_records,
            startup_control_sha256={n:digest(self.root/n) for n in ("probes.json","rim-rays.json")})
        self.references = dict(parent=list(self.train.train_context_ids[:13]),candidate_bank=list(self.train.train_context_ids[:16]))
        self.bindings = dict(source_sha256="a"*64,native_binding_sha256=self.native_binding["fingerprint"],
            historical_capture_ids_sha256=canonical_hash(self.references))
        self.plan = dispatcher.build_assessment_plan(self.train,final_checkpoint_sha256=digest(self.final_path),
            cells_by_split=self.cells,supports=self.supports,prepared_records=self.records,
            parent_distributions=self.parent,continued_distributions=self.continued,model_hashes=self.hashes,bindings=self.bindings)
        self.runner = dispatcher.AssessmentRunner(self.directory,self.train,self.plan,
            parent_checkpoint_path=self.parent_path,final_checkpoint_path=self.final_path,state_hashes=lambda:self.hashes,
            synthetic=True,clock=lambda:0.,free_bytes=lambda:10*1024**3)
        self.build_evidence()

    @staticmethod
    def fake_packet(directory,packet,**kwargs):
        choices = kwargs["choices"]
        rows = [dict(request_id=c["request_id"],released=False,captured=False,clean=False,
            decline_reason=c["decline_reason"] or "synthetic_guard_decline",**{"return":-1.}) for c in choices]
        return dict(guards=dict(mock_guard=True),scores=dict(terminal_results=rows,rejections=35,captures=0))

    def write_packet(self,cohort,packet):
        stem = f"assessment-{cohort:06d}"
        raw = json.dumps(packet,allow_nan=False,separators=(",",":")).encode()
        compressed = gzip.compress(raw,mtime=0)
        path = self.directory/"native"/(stem+".json.gz");path.write_bytes(compressed)
        receipt = dict(file=path.name,sha256=digest(path),bytes=len(compressed),uncompressed_sha256=hashlib.sha256(raw).hexdigest())
        (self.directory/"native"/(stem+".receipt.json")).write_text(json.dumps(receipt))
        return receipt["sha256"]

    def build_evidence(self):
        folder = self.directory/"native";folder.mkdir()
        by_id = {c["id"]:c for cells in self.cells.values() for c in cells}
        for cohort in range(12):
            choices = self.runner.reserve_batch();pending = self.runner.ledger.pending
            cells = [dict(copy.deepcopy(self.cells["train"][i]),**copy.deepcopy(by_id[c["context_id"]])) for i,c in enumerate(choices)]
            packet = dict(schema=subject.PACKET_SCHEMA,mode="assessment",batch=pending.batch_index,cohort=cohort,
                split=choices[0]["split"],method=choices[0]["method"],synthetic=True,
                reservation_sha256=self.runner.journal.head,assessment_plan_fingerprint=self.plan["fingerprint"],
                native_binding=self.native_binding,model_hashes=self.hashes,final_checkpoint_sha256=digest(self.final_path),
                cells=cells,choices=choices,request_ids=[c["request_id"] for c in choices],
                setup_controller=subject.setup_protocol(),startup_controls=self.controls,physics=None,
                optimizer_updates=0,wall_s=0.,process_cpu_s=0.)
            packet_sha = self.write_packet(cohort,packet)
            claim = dict(reservation_sha256=self.runner.journal.head,ledger_fingerprint=self.runner.ledger.to_dict()["fingerprint"],
                request_ids=packet["request_ids"],assessment_plan_fingerprint=self.plan["fingerprint"],
                native_binding_fingerprint=self.native_binding["fingerprint"],status="claimed_before_native_call")
            (folder/f"assessment-{cohort:06d}.dispatch.json").write_text(json.dumps(claim))
            rows = self.fake_packet(None,packet,choices=choices)["scores"]["terminal_results"]
            results = [dict(r,source_sha256=packet_sha) for r in rows]
            if cohort == 0:
                self.runner.record_results(results[:7]);self.runner.record_results(results[7:])
            else:self.runner.record_results(results)
        self.result = self.runner.finish()
        self.head = self.runner.journal.head
        self.result_sha = digest(self.directory/"assessment-result.json")

    def audit(self,**changes):
        args = dict(expected_head_sha256=self.head,expected_bindings=self.bindings,train_ledger=self.train,
            expected_native_binding=self.native_binding,scene=self.scene,cells_by_split=self.cells,supports=self.supports,
            prepared_records=self.records,parent_distributions=self.parent,continued_distributions=self.continued,
            model_hashes=self.hashes,assets_paths=self.assets,poses_path=self.poses,camera_path=self.camera,
            parent_checkpoint_path=self.parent_path,final_checkpoint_path=self.final_path,startup_directory=self.root,
            expected_result_sha256=self.result_sha,historical_capture_ids=self.references)
        args.update(changes)
        return subject.audit_assessment(self.directory,**args)

    def test_all420_requests_and_empty_support_failures_survive_independent_join(self):
        before = {p:digest(p) for p in self.directory.rglob("*") if p.is_file()}
        result = self.audit()
        self.assertEqual((result["requests"],result["rejections"],result["releases"]),(420,420,0))
        self.assertEqual(len(result["batches"]),12)
        self.assertFalse(result["model_promoted"]);self.assertTrue(result["synthetic"])
        for method in subject.METHODS:self.assertEqual(result["comparison"]["methods"][method]["test"]["requested"],70)
        self.assertEqual(result["comparison"]["methods"]["continued"]["test"]["rejection_reasons"]["no_qualified_candidate_support"],2)
        self.assertEqual(before,{p:digest(p) for p in before})

    def test_rehashed_plan_or_wrong_frozen_readout_cannot_replace_actions(self):
        path = self.directory/"assessment-plan.json";saved = json.loads(path.read_bytes())
        saved["choices"][0]["action"]["height_m"] += .01
        saved.pop("fingerprint");saved["fingerprint"] = canonical_hash(saved);path.write_text(json.dumps(saved))
        with self.assertRaisesRegex(ValueError,"independently recomputed"):self.audit()

    def test_readout_action_argmax_is_recomputed_not_trusted(self):
        changed = copy.deepcopy(self.continued);changed["test-000"]["latent_mean"] = [9.]*5
        with self.assertRaisesRegex(ValueError,"independently recomputed"):self.audit(continued_distributions=changed)

    def test_final_checkpoint_bytes_frozen_before_assessment_are_mandatory(self):
        self.final_path.write_bytes(b"changed model")
        with self.assertRaisesRegex(ValueError,"Final frozen"):self.audit()

    def test_corrupt_packet_bytes_missing_or_duplicate_evidence_cannot_pass(self):
        path = self.directory/"native/assessment-000000.json.gz";raw = path.read_bytes();path.write_bytes(raw+b"x")
        with self.assertRaisesRegex(ValueError,"byte receipt"):self.audit()
        path.write_bytes(raw)
        (self.directory/"native/retry.json").write_text("{}")
        with self.assertRaisesRegex(ValueError,"Unlisted"):self.audit()

    def test_rehashed_wrong_method_or_native_binding_is_rejected(self):
        path = self.directory/"native/assessment-000000.json.gz"
        saved = json.loads(gzip.decompress(path.read_bytes()));saved["method"] = "continued"
        self.write_packet(0,saved)
        with self.assertRaisesRegex(ValueError,"packet method/checkpoint/source/cohort"):self.audit()

    def test_rehashed_terminal_reward_or_capture_cannot_pass(self):
        paths = sorted((self.directory/"journal").glob("*.json"))
        row = json.loads(paths[-2].read_bytes());row["payload"]["results"][0]["captured"] = True
        row.pop("fingerprint");row["fingerprint"] = canonical_hash(row);paths[-2].write_text(json.dumps(row))
        final = json.loads(paths[-1].read_bytes());final["previous_sha256"] = row["fingerprint"]
        final.pop("fingerprint");final["fingerprint"] = canonical_hash(final);paths[-1].write_text(json.dumps(final))
        self.head = final["fingerprint"]
        with self.assertRaisesRegex(ValueError,"result identity"):self.audit(expected_result_sha256=None)

    def test_result_byte_binding_and_reference_witness_binding_are_required(self):
        with self.assertRaisesRegex(ValueError,"result bytes"):self.audit(expected_result_sha256="0"*64)
        refs = copy.deepcopy(self.references);refs["parent"][0] = "train-034"
        with self.assertRaisesRegex(ValueError,"witness reference"):self.audit(historical_capture_ids=refs)

    def test_linked_journal_or_external_wrong_head_never_certifies(self):
        with self.assertRaisesRegex(ValueError,"externally bound"):self.audit(expected_head_sha256="0"*64)
        path = self.directory/"journal/event-000000.json";outside=self.root/"event.json";outside.write_bytes(path.read_bytes())
        path.unlink();path.symlink_to(outside)
        with self.assertRaisesRegex(ValueError,"linked assessment"):self.audit()


class AssessmentNumericsTests(unittest.TestCase):
    def test_independent_preference_keeps_fixed_latents_and_deterministic_ties(self):
        cell=dict(id="example",split="test",bucket_xy_m=[.6,-.1])
        values=((LOW+HIGH)/2).tolist()
        parent=decode_action(cell,values,"parent")
        alt=list(values);alt[0]=.1
        rows=[dict(candidate_id="z",action=parent),dict(candidate_id="a",action=decode_action(cell,alt,"alternate"))]
        support=freeze_support(cell["id"],rows,parent,[0.]*5)
        mean=((np.asarray(support.latents[0])+np.asarray(support.latents[1]))/2).tolist()
        result=subject.independent_preference(dict(latent_mean=mean,latent_scale=[1.]*5),support)
        self.assertEqual(result["index"],0)  # Qualified exact parent wins a numerical tie.
        result=subject.independent_preference(dict(latent_mean=list(support.latents[1]),latent_scale=[1.]*5),support)
        self.assertEqual(result["index"],1)  # The old parent's encoding never follows the changed mean.
        negative=list(values);negative[0]=-.1
        other=freeze_support(cell["id"],[dict(candidate_id="z",action=decode_action(cell,negative,"negative")),rows[1]],parent,[0.]*5)
        self.assertIsNone(other.parent_index)
        self.assertEqual(subject.independent_preference(dict(latent_mean=[0.]*5,latent_scale=[1.]*5),other)["index"],1)
        with self.assertRaisesRegex(ValueError,"frozen Gaussian"):
            subject.independent_preference(dict(latent_mean=[0.]*5,latent_scale=[0.]*5),support)

    def test_original_guard_and_score_wrappers_on_same_heldout_empty_support_packet(self):
        packet,scene,cells,links,seeds = synthetic_guard_packet()
        for cell,row in zip(cells,packet["rows"]):cell["split"] = row["split"] = "test"
        choices = [dict(request_id=f"assessment-{i:06d}",context_id=c["id"],split="test",method="continued",
            action=a,enabled=True,decline_reason=None,ik_seeds=None) for i,(c,a) in enumerate(zip(cells,packet["actions"]))]
        packet.update(schema=subject.PACKET_SCHEMA,mode="assessment",optimizer_updates=0,cells=cells,choices=choices,
            request_ids=[c["request_id"] for c in choices],decline_reasons=[None]*35,
            rewards=[-1.]*35,reward_features=[None]*35,physics=None,origins_m=np.zeros((35,3)).tolist())
        for row in packet["rows"]:row["native_invalid"] = row["invalid"]
        choices[4].update(enabled=False,decline_reason="no_qualified_candidate_support")
        packet["enabled"][4] = False;packet["decline_reasons"][4] = "no_qualified_candidate_support"
        packet["preflight_records"][4] = None
        packet["rows"][4].update(invalid="no_qualified_candidate_support",native_invalid="no_selected_setting",
            qualification=dict(accepted=False,reason="inactive_split"),path_proxy=None,path_rejection=None)
        before = copy.deepcopy(packet)
        with tempfile.TemporaryDirectory() as folder, patch('robust_ik.solve',side_effect=AssertionError('No solver call')), \
                patch.object(subject,"verify_guards",wraps=subject.verify_guards) as guards, \
                patch.object(subject,"verify_score",wraps=subject.verify_score) as scores:
            result = subject.verify_assessment_packet(folder,packet,scene=scene,cells=cells,links=links,choices=choices)
        self.assertEqual(result["scores"]["requests"],35);self.assertEqual(result["scores"]["rejections"],35)
        self.assertEqual(result["scores"]["terminal_results"][4]["decline_reason"],"no_qualified_candidate_support")
        self.assertEqual(packet,before)
        self.assertEqual(guards.call_args.args[0]["rows"][4]["invalid"],"inactive_split")
        self.assertEqual(scores.call_args.args[1]["rows"][4]["invalid"],"no_qualified_candidate_support")
        self.assertEqual(packet["rows"][4]["native_invalid"],"no_selected_setting")
        changes=[("wrong raw mirror",lambda p:p["rows"][4].update(native_invalid="inactive_split")),
            ("semantic relabel",lambda p:p["rows"][4].update(invalid="other")),
            ("fabricated preflight",lambda p:p["preflight_records"].__setitem__(4,dict(accepted=False))),
            ("accepted qualification",lambda p:p["rows"][4].update(qualification=dict(accepted=True))),
            ("path check",lambda p:p["rows"][4].update(path_proxy=dict(accepted=True))),
            ("path decline",lambda p:p["rows"][4].update(path_rejection="other")),
            ("success",lambda p:p["rows"][4].update(success=True)),
            ("capture",lambda p:p["rows"][4].update(route_success=True)),
            ("quality",lambda p:p["rows"][4].update(route_quality=2)),
            ("reward",lambda p:p["rewards"].__setitem__(4,0.)),
            ("reward features",lambda p:p["reward_features"].__setitem__(4,{}))]
        for label,change in changes:
            altered=copy.deepcopy(before);change(altered);saved=copy.deepcopy(altered)
            with self.subTest(change=label),self.assertRaisesRegex(ValueError,"Empty-support"):
                subject.verify_assessment_packet(Path("/unused"),altered,scene=scene,cells=cells,links=links,choices=choices)
            self.assertEqual(altered,saved)
        altered=copy.deepcopy(before);altered["enabled"][4]=True
        with self.assertRaisesRegex(ValueError,"slot/decline/seed mapping"):
            subject.verify_assessment_packet(Path("/unused"),altered,scene=scene,cells=cells,links=links,choices=choices)
        altered=copy.deepcopy(before);altered["rows"][0]["native_invalid"]="no_selected_setting"
        with self.assertRaisesRegex(ValueError,"enabled request"):
            subject.verify_assessment_packet(Path("/unused"),altered,scene=scene,cells=cells,links=links,choices=choices)

    def test_position_paired_bootstrap_and_train_history_are_descriptive(self):
        ids={s:[dict(id=f"{s}-{i:03d}",split=s,bucket_xy_m=[.5+i*.001,-.1]) for i in range(n)]
             for s,n in subject.SPLIT_COUNTS.items()}
        outcomes=[]
        for split,cells in ids.items():
            for method in subject.METHODS:
                for i,cell in enumerate(cells):
                    captured = i < (30 if method == "continued" else 20 if method == "candidate_bank" else 25)
                    released = i < len(cells)-3
                    outcomes.append(dict(split=split,method=method,context_id=cell["id"],released=released,
                        captured=captured,clean=captured and i%2 == 0,decline_reason=None if released else "unresolved_ik"))
        references=dict(parent=[c["id"] for c in ids["train"][:13]],candidate_bank=[c["id"] for c in ids["train"][:16]])
        report=subject.comparison_report(outcomes,ids,historical_capture_ids=references)
        paired=report["test_pairs"]["continued_minus_candidate_bank"]["captured"]
        self.assertEqual((paired["gains"],paired["losses"],paired["ties"]),(10,0,60))
        self.assertAlmostEqual(paired["difference_rate"],10/70)
        self.assertGreater(paired["paired_bootstrap_95_percent_interval"][0],0)
        self.assertEqual(report["bootstrap"]["replicates"],10000)
        self.assertEqual(report["bootstrap"]["seed"],20261128)
        self.assertFalse(report["model_promoted"])
        self.assertEqual(report["methods"]["continued"]["test"]["requested"],70)
        self.assertEqual(report["train_retention"]["historical_witnesses"]["parent"]["continued_retained"],13)
        with self.assertRaisesRegex(ValueError,"exactly 420"):subject.comparison_report(outcomes[:-1],ids)

    def test_bootstrap_does_not_change_global_rng_and_declines_are_not_omitted(self):
        ids={s:[dict(id=f"{s}-{i:03d}",split=s,bucket_xy_m=[.5,-.1]) for i in range(n)] for s,n in subject.SPLIT_COUNTS.items()}
        rows=[dict(split=s,method=m,context_id=c["id"],released=False,captured=False,clean=False,
            decline_reason="no_qualified_candidate_support") for s,cells in ids.items() for m in subject.METHODS for c in cells]
        original=np.random.get_state();self.addCleanup(np.random.set_state,original)
        np.random.seed(111);before=np.random.get_state()
        report=subject.comparison_report(rows,ids)
        after=np.random.get_state()
        self.assertTrue(np.array_equal(before[1],after[1]))
        self.assertEqual(report["test_pairs"]["continued_minus_parent"]["captured"]["paired_bootstrap_95_percent_interval"],[0.,0.])
        self.assertEqual(report["methods"]["candidate_bank"]["test"]["rejected"],70)


if __name__ == "__main__":unittest.main()
