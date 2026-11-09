"""Production wiring through fake files/models/subprocesses only."""

import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock,patch

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import candidate_continuation_application as application
import candidate_continuation_production as subject
from candidate_continuation_ledger import SourceBindings,canonical_hash
from candidate_continuation_policy import freeze_support
from continuous_feasibility import LOW,HIGH,decode_action


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.host=Path("/mock-workspace")
        source=self.root/"snapshot/sircus/isaac";source.mkdir(parents=True)
        (source/"test.py").write_text("# synthetic snapshot\n")
        self.source=source
        self.files={}
        for role in ("parent_checkpoint","fresh_split_manifest","train_bank"):
            path=self.root/(role+".json");path.write_text("{}")
            self.files[role]=dict(path=str(self.host/path.relative_to(self.root)),sha256=application.sha(path))
        self.manifest=dict(schema=application.MANIFEST_SCHEMA,workspace_host=str(self.host),
            source_host=str(self.host/"snapshot"),files=self.files,
            source_sha256=canonical_hash({"test.py":application.sha(source/"test.py")}),
            execution_authorized=False,release_authorized=False)
        self.path=self.root/"manifest.json"
        self.save()

    def save(self):
        self.path.write_text(json.dumps(self.manifest));self.sha=application.sha(self.path)

    def load(self,**changes):
        args=dict(path=self.path,expected_sha256=self.sha,container_root=self.root);args.update(changes)
        return application.load_manifest(**args)

    def test_host_paths_map_without_changing_raw_manifest_bytes(self):
        before=self.path.read_bytes();value,paths,mapped=self.load()
        self.assertEqual(value,self.manifest);self.assertEqual(self.path.read_bytes(),before)
        self.assertEqual(paths["parent_checkpoint"],self.root/"parent_checkpoint.json")
        self.assertEqual(mapped(self.host/"snapshot"),self.root/"snapshot")

    def test_external_sha_source_and_input_tampering_fail_closed(self):
        with self.assertRaisesRegex(ValueError,"manifest changed"):self.load(expected_sha256="0"*64)
        (self.source/"test.py").write_text("# changed\n")
        with self.assertRaisesRegex(ValueError,"source snapshot"):self.load()
        (self.source/"test.py").write_text("# synthetic snapshot\n")
        (self.root/"train_bank.json").write_text("[]")
        with self.assertRaisesRegex(ValueError,"input bytes"):self.load()

    def test_path_escape_symlink_and_authorization_flags_are_rejected(self):
        for name in ("/elsewhere/input.json","/mock-workspace/../elsewhere.json"):
            self.manifest["files"]["train_bank"]["path"]=name;self.save()
            with self.assertRaises(ValueError):self.load()
        self.manifest["files"]["train_bank"]["path"]="/mock-workspace/linked.json"
        (self.root/"linked.json").symlink_to(self.root/"train_bank.json");self.save()
        with self.assertRaises(ValueError):self.load()
        self.manifest["execution_authorized"]=True;self.save()
        with self.assertRaisesRegex(ValueError,"not execution permission"):self.load()

    def test_duplicate_json_and_missing_mandatory_role_rejected(self):
        self.path.write_text('{"a":1,"a":2}');self.sha=application.sha(self.path)
        with self.assertRaisesRegex(ValueError,"Duplicate"):self.load()
        self.manifest["files"].pop("train_bank");self.save()
        with self.assertRaisesRegex(ValueError,"mandatory"):self.load()


class ComponentTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.components=subject.ProductionComponents(self.root/"manifest.json","a"*64,self.root,"b"*32)

    def test_construction_and_preinit_device_snapshot_are_inert(self):
        self.assertIsNone(self.components.app);self.assertIsNone(self.components.device_snapshot(False))
        self.components.close_native();self.assertFalse(self.components.closed)

    def test_input_validation_failure_precedes_app_or_engine_imports(self):
        with patch.object(subject,"production_inputs",side_effect=ValueError("mock input changed")):
            with self.assertRaisesRegex(ValueError,"mock input changed"):self.components.initialize(Mock())
        self.assertIsNone(self.components.app);self.assertIsNone(self.components.runner)

    def test_native_close_is_idempotent_and_abort_never_retries(self):
        self.components.app=Mock()
        self.components.close_native();self.components.close_native()
        self.components.app.close.assert_called_once_with(skip_cleanup=True)
        self.components.runner=types.SimpleNamespace(finished=False,failed=False,_fail=Mock())
        self.components.assessment=types.SimpleNamespace(finished=True,failed=False,_fail=Mock())
        error=RuntimeError("synthetic stop");self.components.abort(error)
        self.components.runner._fail.assert_called_once_with(error)
        self.components.assessment._fail.assert_not_called()

    def test_cpu_audit_subprocess_is_fixed_bounded_hidden_and_never_executed(self):
        for name in ("learning","assessment","native-setup/startup"):(self.root/name).mkdir(parents=True,exist_ok=True)
        for name in ("learning/result.json","assessment/assessment-result.json","native-binding.json","native-setup/startup/receipt.json"):
            (self.root/name).write_text("{}")
        (self.root/"combined-audit.json").write_text('{"status":"passed","all_components_passed":true}')
        with patch.object(subject.subprocess,"run") as run,patch.object(subject,"source_fingerprint",return_value="c"*64):
            result=self.components.audit({},{});
        self.assertEqual(result["status"],"passed");self.assertEqual(run.call_count,1)
        args,kwargs=run.call_args
        self.assertIn("--audit-only",args[0]);self.assertNotIn("--execute-approved",args[0])
        self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"],"")
        self.assertEqual(kwargs["timeout"],600);self.assertTrue(kwargs["check"])
        request=application.read(self.root/"audit-request.json")
        self.assertEqual(request["session"],"b"*32)
        index=args[0].index("--expected-audit-request-sha256")
        self.assertEqual(args[0][index+1],application.sha(self.root/"audit-request.json"))
        with patch.object(subject.subprocess,"run") as retry:
            with self.assertRaises(FileExistsError):self.components.audit({}, {})
            retry.assert_not_called()

    def test_missing_or_mismatched_dependencies_never_install_anything(self):
        import continuous_policy
        with patch.object(continuous_policy,"torch",None):
            with self.assertRaisesRegex(ValueError,"Torch runtime"):subject.verify_dependencies({})
        fake=types.SimpleNamespace(__version__="mock-torch")
        with patch.object(continuous_policy,"torch",fake):
            with self.assertRaisesRegex(ValueError,"versions differ"):subject.verify_dependencies({})

    def test_frozen_scene_uses_its_original_readable_json_hash(self):
        scene=json.loads(Path(subject.__file__).with_name("scene.json").read_text())
        identity=subject._frozen_identity()
        # Compact ledger hashes and the historical scene hash are intentionally distinct.
        self.assertNotEqual(canonical_hash(scene),identity["scene_sha256"])
        manifest=dict(files={name:dict(sha256=value) for name,value in dict(
            parent_checkpoint=subject._parent_sha(),camera=identity["camera_sha256"],poses=identity["poses_sha256"],
            dropper_urdf=identity["assets_sha256"]["Arm10"],board_urdf=identity["assets_sha256"]["Arm14"]).items()})
        subject.verify_original_identity(manifest,scene)
        scene["ball"]["radius_m"]+=.001
        with self.assertRaisesRegex(ValueError,"Original frozen"):subject.verify_original_identity(manifest,scene)


class FrozenReadoutTests(unittest.TestCase):
    def setUp(self):
        self.cell=dict(id="train-000",split="train",bucket_xy_m=[.5,-.1])
        self.action=decode_action(self.cell,((LOW+HIGH)/2).tolist(),"parent-train-000")
        self.support=freeze_support("train-000",[dict(candidate_id="one",action=self.action)],self.action,[0.]*5)
        self.readout={"train-000":dict(exact_action=self.action,latent_mean=[0.]*5,latent_scale=[1.]*5)}
        self.inputs=dict(paths=dict(parent_checkpoint=Path("/mock/parent.pt")),bindings={},
            parent_distributions=copy.deepcopy(self.readout),supports={"train-000":self.support},
            cells_by_split=dict(train=[self.cell],validation=[],test=[]),historical_capture_ids={},
            source_bindings=SourceBindings(**{k:"d"*64 for k in asdict(SourceBindings()) if k!="parent_checkpoint_sha256"}))

    def context(self,parent=None):
        import candidate_continuation_assessment as assessment
        import candidate_continuation_checkpoint as checkpoints
        import continuous_policy
        with patch.object(subject,"_parent",return_value="mock parent"),\
             patch.object(checkpoints,"load_continuation",return_value=("mock final",{})),\
             patch.object(assessment,"predict_distributions",side_effect=[parent or self.readout,self.readout]),\
             patch.object(continuous_policy,"weight_hash",side_effect=["a"*64,"b"*64]),\
             patch.object(subject,"source_fingerprint",return_value="e"*64):
            return subject.assessment_context(self.inputs,None,{"checkpoint_sha256":"f"*64},Path("/mock/final.pt"),{"fingerprint":"c"*64})

    def test_independent_frozen_models_and_all_ledger_bindings_are_carried(self):
        value=self.context()
        self.assertEqual(value["model_hashes"],dict(parent_weight_hash="a"*64,continued_weight_hash="b"*64))
        for key,item in asdict(self.inputs["source_bindings"]).items():self.assertEqual(value["bindings"][key],item)
        self.assertEqual(value["parent_distributions"],self.readout)

    def test_parent_mean_and_physical_action_tampering_are_detected(self):
        changed=copy.deepcopy(self.readout);changed["train-000"]["latent_mean"][0]=.1
        with self.assertRaisesRegex(ValueError,"distribution changed"):self.context(changed)
        changed=copy.deepcopy(self.readout);changed["train-000"]["exact_action"]["height_m"]+=.01
        with self.assertRaisesRegex(ValueError,"direct-parent action"):self.context(changed)


if __name__=="__main__":unittest.main()
