import ast
import copy
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import continuous_pilot as pilot
from continuous_feasibility import canonical_hash
from expanded_farm import make_spec


class PilotContractTests(unittest.TestCase):
    def bundle(self):
        scene = json.loads(Path(pilot.__file__).with_name("scene.json").read_text())
        return dict(spec=make_spec(scene, 10), source={"camera_sha256": "a"*64})

    def test_separate_signed_split_and_budget(self):
        bundle = self.bundle()
        original = copy.deepcopy(bundle)
        study = pilot.make_study(bundle)
        self.assertEqual(bundle, original)
        self.assertEqual([sum(c["split"] == s for c in study["cells"])
                          for s in ("train", "validation", "test")], [25,5,5])
        self.assertEqual(study["training_batches"],64)
        self.assertEqual(study["fingerprint"], canonical_hash({k:v for k,v in study.items() if k != "fingerprint"}))
        self.assertNotEqual(study["fingerprint"], bundle["spec"]["fingerprint"])

    def test_cli_guard(self):
        self.assertFalse(pilot.validate_options(None,None,64,appearance="silver"))
        self.assertTrue(pilot.validate_options("old","camera",4,appearance="silver"))
        for source,camera,batches,appearance,incompatible in [(None,"camera",64,"silver",False),
            ("old",None,64,"silver",False),("old","camera",65,"silver",False),
            ("old","camera",64,"dark",False),("old","camera",64,"silver",True)]:
            with self.assertRaises(ValueError):
                pilot.validate_options(source,camera,batches,appearance=appearance,incompatible=incompatible)

    def test_live_transit_never_resets_joints(self):
        tree=ast.parse(Path(pilot.__file__).read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name == "ContinuousShots")
        drive=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name == "approach_endpoints")
        attrs=[n.func.attr for n in ast.walk(drive) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
        for forbidden in ("write_joint_state_to_sim","reset"):
            self.assertNotIn(forbidden,attrs)
        self.assertIn("set_joint_position_target",attrs)
        for call in ast.walk(cls):
            if isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute) and call.func.attr == "approach_endpoints":
                self.assertIsInstance(call.func.value,ast.Name)
                self.assertEqual(call.func.value.id,"self")

    def test_full_time_recording_stride(self):
        text=Path(pilot.__file__).read_text()
        self.assertIn('(self.steps-self.trial_start_step) % 8 == 0',text)
        self.assertNotIn('MOVE_FRAMES',text)
        self.assertIn('film=True',text)


if __name__ == "__main__":
    unittest.main()
