"""Mock numeric verifiers to isolate durable native evidence/ledger joining.

The companion score and guard suites exercise those numeric verifiers using
synthetic traces. These fixtures are bookkeeping mocks, never native results.
"""

import copy
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import candidate_continuation_native_audit as subject
from candidate_continuation_journal import Journal
from candidate_continuation_ledger import Ledger, Outcome, SourceBindings, canonical_hash
from candidate_continuation_native import make_binding, prepare_packet, PACKET_SCHEMA
from candidate_continuation_policy import freeze_support
from continuous_feasibility import decode_action, LOW, HIGH, canonical_hash as geometry_hash
from test_candidate_continuation_assessment import original_parallel_shot_batch
from test_candidate_continuation_native_guards import synthetic_guard_packet


def original_boundary_padding_packet(paid):
    """Original batch body, fake setup/parking, real CPU scoring and guards."""
    packet,scene,cells,links,seeds=synthetic_guard_packet()
    enabled=[i<paid for i in range(35)]
    qualifications=[copy.deepcopy(row["qualification"]) for row in packet["rows"]]
    for i in range(paid,35):
        cells[i].update(id=f"inactive-0041-{i:02d}",split="inactive")
        qualifications[i]=dict(accepted=False,reason="inactive_split")
        packet["preflight_records"][i]=None
    releases=np.asarray([row["release_ball_center_m"] for row in packet["rows"]])
    normals=np.asarray([row["scoring_board_normal"] for row in packet["rows"]])
    placements=[]
    def no_tick():raise AssertionError("All synthetic guard declines must remain parked")
    env=types.SimpleNamespace(n=35,dt=1/240,cfg=scene,spec=dict(cells=cells),
        prepare=lambda actions:(releases,normals,packet["settling_history"],
            [q["reason"] for q in qualifications]),
        set_balls=lambda xyz:placements.append(np.asarray(xyz).copy()),
        render=lambda:None,assert_buckets_fixed=lambda:None,tick=no_tick)
    rows,raw=original_parallel_shot_batch()(env,packet["actions"],record_video=False,enabled=enabled)
    for i,row in enumerate(rows):
        row.update(split=cells[i]["split"],qualification=qualifications[i],path_rejection=None,
            path_proxy=copy.deepcopy(packet["rows"][i]["path_proxy"]) if enabled[i] else None,
            scoring_board_normal=normals[i].tolist())
    identifiers=[f"request-{i:04d}" for i in range(paid)]
    packet.update(schema=PACKET_SCHEMA,mode="train",cells=cells,rows=rows,enabled=enabled,
        request_ids=identifiers,choices=[dict(request_id=identifiers[i],context_id=cells[i]["id"],action=packet["actions"][i])
            if i<paid else None for i in range(35)],
        rewards=[-1. if use else None for use in enabled],reward_features=[None]*35,
        trajectory_m=raw["trajectory_m"],origins_m=np.zeros((35,3)).tolist(),physics=None)
    return packet,scene,cells,links,seeds,placements


class PaddingGuardViewTests(unittest.TestCase):
    def test_original_34_and_1_paid_boundaries_keep_35_padding_rows_unpaid_and_raw(self):
        total_padding=0
        for paid in (34,1):
            packet,scene,cells,links,seeds,placements=original_boundary_padding_packet(paid)
            original=copy.deepcopy(packet)
            # This is the original failing reader call, with the unchanged
            # generic guard and the actual original batch boundary labels.
            with patch('robust_ik.solve',side_effect=AssertionError('No solver call')):
                with self.assertRaisesRegex(ValueError,"first cause changed"):
                    subject.verify_guards(packet,scene,cells,links,seeds)
                guarded,proof=subject._padding_guard_view(packet)
                subject.verify_guards(guarded,scene,cells,links,seeds)
                scores=subject.verify_score(Path('/unused'),packet,scene)
            self.assertEqual(scores["requests"],paid);self.assertEqual(scores["rejections"],paid)
            self.assertEqual(scores["releases"],0);self.assertEqual(len(scores["terminal_results"]),paid)
            self.assertEqual(proof["normalized_slots"],list(range(paid,35)));total_padding+=proof["unpaid_slots"]
            self.assertEqual(packet,original);self.assertEqual(guarded["rows"][:paid],packet["rows"][:paid])
            self.assertEqual(len(placements),1);self.assertTrue(np.array_equal(placements[0],np.tile([.1,-.8,.04],(35,1))))
            for i in range(paid,35):
                self.assertEqual(packet["rows"][i]["invalid"],"no_selected_setting")
                self.assertEqual(guarded["rows"][i]["invalid"],"inactive_split")
                self.assertIsNone(packet["choices"][i]);self.assertIsNone(packet["proposal_ik_seeds"][i])
                self.assertIsNone(packet["preflight_records"][i]);self.assertIsNone(packet["rewards"][i])
                self.assertIsNone(packet["reward_features"][i]);self.assertEqual(packet["trajectory_m"][i],[])
        self.assertEqual(total_padding,35)

    def test_padding_adapter_rejects_any_executed_or_relabelled_unpaid_evidence(self):
        packet,*_=original_boundary_padding_packet(34)
        changes=[("raw label",lambda p:p["rows"][34].update(invalid="inactive_split")),
            ("executed",lambda p:p["rows"][34].update(invalid=None)),
            ("request",lambda p:p["rows"][34].update(request_id="unpaid-injected")),
            ("choice",lambda p:p["choices"].__setitem__(34,{})),
            ("IK",lambda p:p["proposal_ik_seeds"].__setitem__(34,{})),
            ("preflight",lambda p:p["preflight_records"].__setitem__(34,{})),
            ("qualification",lambda p:p["rows"][34].update(qualification=dict(accepted=True,reason="inactive_split"))),
            ("qualification boolean",lambda p:p["rows"][34].update(qualification=dict(accepted=0,reason="inactive_split"))),
            ("path",lambda p:p["rows"][34].update(path_proxy={})),
            ("path reason",lambda p:p["rows"][34].update(path_rejection="other")),
            ("reward",lambda p:p["rewards"].__setitem__(34,-1.)),
            ("features",lambda p:p["reward_features"].__setitem__(34,{})),
            ("success",lambda p:p["rows"][34].update(success=True)),
            ("capture",lambda p:p["rows"][34].update(route_success=True)),
            ("quality",lambda p:p["rows"][34].update(route_quality=1)),
            ("quality boolean",lambda p:p["rows"][34].update(route_quality=False)),
            ("trajectory",lambda p:p["trajectory_m"].__setitem__(34,[[.1,-.8,.04]])),
            ("split",lambda p:p["rows"][34].update(split="train"))]
        for label,change in changes:
            altered=copy.deepcopy(packet);change(altered);before=copy.deepcopy(altered)
            with self.subTest(change=label),self.assertRaisesRegex(ValueError,"Unpaid native padding"):
                subject._padding_guard_view(altered)
            self.assertEqual(altered,before)

    def test_padding_cannot_be_omitted_enabled_or_inserted_into_paid_request_prefix(self):
        packet,*_=original_boundary_padding_packet(34)
        for field in ("cells","actions","choices","enabled","rows","proposal_ik_seeds", "preflight_records","rewards","reward_features","trajectory_m"):
            changed=copy.deepcopy(packet);changed[field].pop()
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,"35-slot evidence"):
                subject._padding_guard_view(changed)
        for change in (lambda p:p["enabled"].__setitem__(34,True),lambda p:p["request_ids"].append("extra")):
            changed=copy.deepcopy(packet);change(changed)
            with self.assertRaisesRegex(ValueError,"paid prefix"):subject._padding_guard_view(changed)

    def test_active_first_cause_mismatch_is_never_normalized(self):
        packet,scene,cells,links,seeds,_=original_boundary_padding_packet(34)
        packet["rows"][0]["invalid"]="fabricated_active_reason"
        guarded,_=subject._padding_guard_view(packet)
        self.assertEqual(guarded["rows"][0],packet["rows"][0])
        with patch('robust_ik.solve',side_effect=AssertionError('No solver call')):
            with self.assertRaisesRegex(ValueError,"first cause changed"):
                subject.verify_guards(guarded,scene,cells,links,seeds)


class NativeEvidenceJoinTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name)/"run"
        if hasattr(type(self), "fixture"):
            cached = type(self).fixture
            shutil.copytree(cached["path"], self.run)
            for key in ("cells", "scene", "supports", "records", "binding", "bindings", "controls"):
                setattr(self, key, copy.deepcopy(cached[key]))
            self.assets, self.poses, self.camera = cached["assets"], cached["poses"], cached["camera"]
            self.journal = types.SimpleNamespace(head=cached["head"])
            self.start_mocks()
            return
        self.run.mkdir(); (self.run/"native").mkdir()
        xml = '<robot>'+''.join(f'<joint name="joint{i}" type="revolute"><origin xyz="0 0 0" rpy="0 0 0"/><axis xyz="0 0 1"/><limit lower="-6.28" upper="6.28" effort="1"/></joint>' for i in range(1,7))+'</robot>'
        self.assets = {n:Path(self.temp.name)/(n+".urdf") for n in ("Arm10", "Arm14")}
        for path in self.assets.values(): path.write_text(xml)
        self.poses = Path(self.temp.name)/"poses.json"
        self.poses.write_text(json.dumps(dict(dropper_joint_rad=[0.]*6, board_joint_rad=[0.]*6)))
        self.camera = Path(self.temp.name)/"camera.json"; self.camera.write_text("{}")
        digest = lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
        self.cells = [dict(id=f"train-{i:03d}", split="train", bucket_xy_m=[.6, -.1]) for i in range(35)]
        self.scene = dict(synthetic=True)
        self.supports, self.records = [], {}
        for cell in self.cells:
            action = decode_action(cell, ((LOW+HIGH)/2).tolist(), cell["id"]+"-action")
            row = dict(candidate_id=cell["id"]+"-candidate", action=action)
            self.supports.append(freeze_support(cell["id"], [row], action, [0.]*5))
            decision = dict(accepted=True, release_authorized=False, qualification=dict(ik={}))
            decision["fingerprint"] = geometry_hash(decision)
            self.records[row["candidate_id"]] = dict(row, cell_id=cell["id"], accepted=True, decision=decision)
        control_hashes = {}
        for name in ("probes.json", "rim-rays.json"):
            (self.run/name).write_text("{}")
            control_hashes[name] = hashlib.sha256(b"{}").hexdigest()
        self.binding = make_binding(scene=self.scene, geometry_fingerprint="a"*64,
            camera_sha256=digest(self.camera), assets_sha256={n:digest(p) for n,p in self.assets.items()}, poses_sha256=digest(self.poses),
            cells=self.cells, canonical_home_rad={"Arm10":[0.]*6,"Arm14":[0.]*6},
            supports=self.supports, prepared_records=self.records, startup_control_sha256=control_hashes)
        self.bindings = {k:self.binding[k] for k in ("source_sha256", "support_sha256")}
        self.controls = dict(mock_startup=True)
        self.start_mocks()
        self.build_history()
        cache_temp = tempfile.TemporaryDirectory()
        type(self).addClassCleanup(cache_temp.cleanup)
        cache = Path(cache_temp.name)
        shutil.copytree(self.run, cache/"run")
        assets = {}
        for name, path in self.assets.items():
            assets[name] = cache/path.name; shutil.copyfile(path, assets[name])
        poses, camera = cache/"poses.json", cache/"camera.json"
        shutil.copyfile(self.poses, poses); shutil.copyfile(self.camera, camera)
        type(self).fixture = dict(path=cache/"run", assets=assets, poses=poses, camera=camera,
            head=self.journal.head, **{k:copy.deepcopy(getattr(self,k)) for k in
                ("cells", "scene", "supports", "records", "binding", "bindings", "controls")})

    def start_mocks(self):
        self.guard = patch.object(subject, "verify_guards", return_value=dict(mock_guard=True))
        self.score = patch.object(subject, "verify_score", side_effect=self.score_packet)
        self.control = patch.object(subject, "verify_controls", return_value=self.controls)
        self.guards = self.guard.start(); self.scorer = self.score.start(); self.control.start()
        self.addCleanup(self.guard.stop); self.addCleanup(self.score.stop); self.addCleanup(self.control.stop)

    def build_history(self,*,padded=False):
        self.journal = Journal(self.run/"journal")
        ledger = Ledger.new(tuple(c["id"] for c in self.cells),
            bindings=SourceBindings(train_support_sha256=self.binding["support_sha256"]))
        self.journal.append("start", dict(synthetic=True, ledger=ledger.to_dict(), bindings=self.bindings))
        lookup = {s.context_id:s for s in self.supports}
        while (batch := ledger.next_batch()) is not None:
            ledger = ledger.reserve_batch(batch)
            choices = []
            for request in batch.requests:
                s = lookup[request.context_id]
                choices.append(dict(request_id=request.request_id, context_id=request.context_id,
                    candidate_id=s.candidate_ids[0], index=0, support_fingerprint=s.support_fingerprint,
                    action=s.action(0), latent=list(s.latents[0]), old_log_probability=0., old_value=0.))
            reservation = self.journal.append("reserve", dict(ledger=ledger.to_dict(),
                choices=choices, policy_weight_hash="f"*64))
            mapped = prepare_packet(choices, ledger, self.cells, self.supports, self.records, self.cells)
            stem = f"train-{batch.batch_index:06d}"
            packet = dict(schema=PACKET_SCHEMA, mode="train", batch=batch.batch_index, synthetic=True,
                native_binding=self.binding, reservation_sha256=reservation["fingerprint"],
                policy_weight_hash="f"*64, setup_controller=self.binding["setup_controller"], startup_controls=self.controls,
                cells=mapped["cells"], actions=mapped["actions"], choices=mapped["selected"],
                enabled=mapped["enabled"], request_ids=mapped["request_ids"], proposal_ik_seeds=mapped["ik_seeds"],
                physics=None, wall_s=1., process_cpu_s=.2)
            # These are bookkeeping-only outcomes; numeric guard/score calls
            # stay mocked in this class. The companion class below executes
            # the original batch method and unchanged numerical verifiers.
            released=[False]*len(batch.requests)
            if padded:
                released=[not (batch.batch_index==0 or batch.batch_index==1 and i<34
                    or batch.batch_index==41 and i==0) for i in range(len(batch.requests))]
            packet.update(rows=[dict(action,cell_id=cell["id"],split=cell["split"],
                invalid=(None if released[i] else "mock_decline") if use else "no_selected_setting",
                qualification=dict(accepted=False,reason="mock_decline" if use else "inactive_split"),
                path_proxy=None,path_rejection=None,success=False,route_success=False,route_quality=0)
                for i,(action,cell,use) in enumerate(zip(packet["actions"],packet["cells"],packet["enabled"]))],
                preflight_records=[dict(synthetic=True) if use else None for use in packet["enabled"]],
                rewards=[(-.8 if released[i] else -1.) if use else None for i,use in enumerate(packet["enabled"])],
                reward_features=[None]*35,trajectory_m=[[] for _ in range(35)])
            packet_sha = self.write_packet(stem, packet)
            claim = dict(reservation_sha256=reservation["fingerprint"], ledger_fingerprint=ledger.to_dict()["fingerprint"],
                request_ids=mapped["request_ids"], native_binding_fingerprint=self.binding["fingerprint"],
                status="claimed_before_native_call")
            (self.run/"native"/(stem+".dispatch.json")).write_text(json.dumps(claim))
            results = [dict(row, source_sha256=packet_sha) for row in self.score_packet(None, packet, None)["terminal_results"]]
            ledger = ledger.record_outcomes(tuple(Outcome(r["request_id"],r["released"],r["decline_reason"],packet_sha) for r in results))
            self.journal.append("outcomes", dict(ledger=ledger.to_dict(), results=results))
        self.journal.append("final", dict(ledger=ledger.to_dict()))

    @staticmethod
    def score_packet(directory, packet, scene):
        return dict(terminal_results=[dict(request_id=k,released=packet["rows"][i]["invalid"] is None,
            decline_reason=packet["rows"][i]["invalid"],captured=False,clean=False,
            **{"return":packet["rewards"][i]}) for i,k in enumerate(packet["request_ids"])])

    def write_packet(self, stem, packet):
        raw = json.dumps(packet, allow_nan=False, separators=(",", ":")).encode()
        packed = gzip.compress(raw, mtime=0)
        path = self.run/"native"/(stem+".json.gz")
        path.write_bytes(packed)
        sha = hashlib.sha256(packed).hexdigest()
        receipt = dict(file=path.name, sha256=sha, bytes=len(packed), uncompressed_sha256=hashlib.sha256(raw).hexdigest())
        (self.run/"native"/(stem+".receipt.json")).write_text(json.dumps(receipt))
        return sha

    def change_packet(self, mutate):
        path = self.run/"native/train-000000.json.gz"
        packet = json.loads(gzip.decompress(path.read_bytes()))
        mutate(packet)
        self.write_packet("train-000000", packet)

    def audit(self, **changes):
        args = dict(expected_head_sha256=self.journal.head, expected_bindings=self.bindings,
            expected_native_binding=self.binding, scene=self.scene, cells=self.cells,
            supports=self.supports, prepared_records=self.records, assets_paths=self.assets,
            poses_path=self.poses, camera_path=self.camera)
        args.update(changes)
        return subject.audit_native_train(self.run, **args)

    def test_every_request_at_cap_is_joined_and_scope_stays_limited(self):
        before = {p:hashlib.sha256(p.read_bytes()).hexdigest() for p in self.run.rglob("*") if p.is_file()}
        result = self.audit()
        self.assertEqual((result["requests"],result["rejections"],result["releases"]),(1750,1750,0))
        self.assertEqual(len(result["batches"]),50)
        self.assertEqual(self.guards.call_count,50); self.assertEqual(self.scorer.call_count,50)
        self.assertTrue(result["learning_checkpoint_audit_required"])
        self.assertFalse(result["assessment_audit_implemented"])
        self.assertTrue(result["synthetic"]); self.assertFalse(result["execution_authorized"])
        self.assertAlmostEqual(result["native_batch_costs"]["wall_s"],50.)
        self.assertEqual(before,{p:hashlib.sha256(p.read_bytes()).hexdigest() for p in before})

    def padded_history(self):
        original=self.run;self.run=Path(self.temp.name)/"padded-run";self.run.mkdir();(self.run/"native").mkdir()
        for name in ("probes.json","rim-rays.json"):shutil.copyfile(original/name,self.run/name)
        self.build_history(padded=True)

    def test_34_then_1_paid_slot_history_preserves_1470_requests_and_35_unpaid_padding_slots(self):
        self.padded_history()
        before={p:subject._digest(p) for p in self.run.rglob("*") if p.is_file()}
        result=self.audit()
        self.assertEqual((result["requests"],result["releases"],result["rejections"]),(1470,1400,70))
        self.assertEqual(len(result["batches"]),43)
        self.assertEqual([b["padding_guard_view"]["paid_requests"] for b in result["batches"][-2:]],[34,1])
        self.assertEqual([b["padding_guard_view"]["normalized_slots"] for b in result["batches"][-2:]],[[34],list(range(1,35))])
        self.assertEqual(result["padding_guard_view"]["normalized_unpaid_slots"],35)
        self.assertEqual(result["auditor_source_sha256"],subject._digest(Path(subject.__file__)))
        self.assertEqual(before,{p:subject._digest(p) for p in before})
        for guard_call,score_call in zip(self.guards.call_args_list,self.scorer.call_args_list):
            guarded,raw=guard_call.args[0],score_call.args[1];paid=len(raw["request_ids"])
            self.assertEqual(guarded["rows"][:paid],raw["rows"][:paid])
            for i in range(paid,35):
                self.assertEqual(guarded["rows"][i]["invalid"],"inactive_split")
                self.assertEqual(raw["rows"][i]["invalid"],"no_selected_setting")

    def test_rehashed_padding_label_is_rejected_before_numeric_verifiers_for_that_packet(self):
        self.padded_history();path=self.run/"native/train-000041.json.gz"
        packet=json.loads(gzip.decompress(path.read_bytes()));packet["rows"][34]["invalid"]="inactive_split"
        self.write_packet("train-000041",packet)
        with self.assertRaisesRegex(ValueError,"Unpaid native padding"):self.audit()
        self.assertEqual(self.guards.call_count,41);self.assertEqual(self.scorer.call_count,41)

    def test_physics_bytes_are_bound_before_any_guard_view_adapter(self):
        self.change_packet(lambda packet:packet.update(physics=dict(file="train-000000.npz",sha256="0"*64)))
        (self.run/"native/train-000000.npz").write_bytes(b"synthetic changed physics")
        with patch.object(subject,"_padding_guard_view",wraps=subject._padding_guard_view) as adapter:
            with self.assertRaisesRegex(ValueError,"physics bytes changed"):self.audit()
        adapter.assert_not_called();self.guards.assert_not_called();self.scorer.assert_not_called()

    def test_external_journal_head_cannot_be_replaced(self):
        with self.assertRaisesRegex(ValueError,"externally bound"):
            self.audit(expected_head_sha256="a"*64)

    def test_rehashed_action_change_still_fails_support_join(self):
        self.change_packet(lambda p:p["actions"][0].update(height_m=.4))
        with self.assertRaisesRegex(ValueError,"reserved support"): self.audit()

    def test_rehashed_dispatch_or_source_or_padding_change_is_rejected(self):
        original = (self.run/"native/train-000000.json.gz").read_bytes()
        mutations = (lambda p:p.update(reservation_sha256="0"*64),
            lambda p:p.update(synthetic=False), lambda p:p.update(batch=True),
            lambda p:p["enabled"].__setitem__(34,False), lambda p:p["native_binding"].update(source_sha256="0"*64),
            lambda p:p["proposal_ik_seeds"].__setitem__(0,{}),
            lambda p:p["cells"][0].update(split="test"))
        for mutate in mutations:
            (self.run/"native/train-000000.json.gz").write_bytes(original)
            self.change_packet(mutate)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError): self.audit()

    def test_missing_receipt_or_unlisted_retry_is_not_hidden(self):
        path = self.run/"native/train-000049.receipt.json"
        raw = path.read_bytes(); path.unlink()
        with self.assertRaisesRegex(ValueError,"Missing"): self.audit()
        path.write_bytes(raw)
        (self.run/"native/retry.json").write_text("{}")
        with self.assertRaisesRegex(ValueError,"Unlisted"): self.audit()

    def test_packet_byte_tamper_does_not_get_to_numeric_verifiers(self):
        path = self.run/"native/train-000000.json.gz"; path.write_bytes(path.read_bytes()+b"x")
        with self.assertRaisesRegex(ValueError,"byte receipt"): self.audit()
        self.guards.assert_not_called(); self.scorer.assert_not_called()

    def test_fabricated_terminal_label_is_rejected_even_after_rechaining(self):
        paths = sorted((self.run/"journal").glob("*.json"))
        row = json.loads(paths[-2].read_bytes()); row["payload"]["results"][0]["captured"] = True
        row.pop("fingerprint"); row["fingerprint"] = canonical_hash(row)
        paths[-2].write_text(json.dumps(row))
        final = json.loads(paths[-1].read_bytes()); final["previous_sha256"] = row["fingerprint"]
        final.pop("fingerprint"); final["fingerprint"] = canonical_hash(final)
        paths[-1].write_text(json.dumps(final)); self.journal.head = final["fingerprint"]
        with self.assertRaisesRegex(ValueError,"independent native audit"): self.audit()

    def test_incomplete_history_never_passes(self):
        path = sorted((self.run/"journal").glob("*.json"))[-1]
        path.unlink()
        self.journal.head = json.loads(sorted((self.run/"journal").glob("*.json"))[-1].read_bytes())["fingerprint"]
        with self.assertRaisesRegex(ValueError,"Incomplete"): self.audit()

    def test_external_binding_must_match_current_source(self):
        binding = copy.deepcopy(self.binding); binding["guard_source_sha256"]["joint_feasibility.py"] = "0"*64
        with self.assertRaisesRegex(ValueError,"identities changed"): self.audit(expected_native_binding=binding)

    def test_raw_asset_and_pose_bytes_are_checked_before_guard_reconstruction(self):
        path = self.assets["Arm10"]
        raw = path.read_bytes(); path.write_bytes(raw+b" ")
        with self.assertRaisesRegex(ValueError,"asset, pose or camera bytes"): self.audit()
        path.write_bytes(raw)
        raw = self.poses.read_bytes(); self.poses.write_text("{}")
        with self.assertRaisesRegex(ValueError,"asset, pose or camera bytes"): self.audit()
        self.poses.write_bytes(raw)

    def test_linked_journal_evidence_cannot_be_certified(self):
        path = self.run/"journal/event-000000.json"
        outside = Path(self.temp.name)/"event.json"; outside.write_bytes(path.read_bytes())
        path.unlink(); path.symlink_to(outside)
        with self.assertRaisesRegex(ValueError,"linked journal"): self.audit()

    def test_strict_json_and_plain_paths(self):
        for raw in (b'{"a":1,"a":2}',b'{"a":NaN}'):
            with self.assertRaises(ValueError): subject._json(raw)
        with self.assertRaises(ValueError): subject._file(self.run, "../outside")
        link = self.run/"linked"; link.symlink_to(self.run/"probes.json")
        with self.assertRaises(ValueError): subject._file(self.run, "linked")

    def test_nonfinite_or_negative_cost_is_not_published(self):
        self.change_packet(lambda p:p.update(wall_s=-1))
        with self.assertRaisesRegex(ValueError,"process cost"): self.audit()


if __name__ == "__main__":
    unittest.main()
