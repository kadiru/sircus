"""Study identity and common episode setup checks without a simulator."""

import copy
import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
from action_training import ActionTrainingShots
from presentation_recording import load_source
from rl_film_study import (ASSESSMENT_STAGES, FILMED_BATCHES, RLStudyShots,
                           canonical_hash, make_study, validate_options, wilson)

BASELINE = SOURCE.parents[1]/"baselines/isaac-expanded-farm-20261102"


class StudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = load_source(BASELINE, BASELINE/"selected-camera.json")

    def test_separate_protocol_preserves_original_geometry_and_camera(self):
        study = make_study(self.bundle)
        unsigned = dict(study)
        self.assertEqual(unsigned.pop("fingerprint"), canonical_hash(unsigned))
        self.assertEqual(study["geometry_fingerprint"], self.bundle["spec"]["fingerprint"])
        self.assertEqual(self.bundle["spec"]["training_batches"], 10)
        self.assertEqual(study["training_batches"]*35, 4480)
        self.assertEqual(study["assessment_batches"]*len(ASSESSMENT_STAGES)*35, 1120)
        self.assertEqual(study["training_batches"]//study["batches_per_update"], 64)
        self.assertEqual(study["filmed_batches"], list(range(8))+list(range(60,68))+list(range(120,128)))
        self.assertEqual(study["camera_sha256"], self.bundle["source"]["camera_sha256"])

    def test_study_binds_actual_menu_and_hyperparameters(self):
        study = make_study(self.bundle)
        changed = copy.deepcopy(self.bundle)
        changed["source"]["menu_sha256"] = "a"*64
        self.assertNotEqual(make_study(changed)["fingerprint"], study["fingerprint"])
        changed = copy.deepcopy(self.bundle)
        changed["menus"][0].pop()
        with self.assertRaisesRegex(ValueError,"24-option"):
            make_study(changed)

    def test_filming_schedule_is_prespecified_and_cannot_use_outcomes(self):
        changed = copy.deepcopy(self.bundle)
        changed["report"]["captures"] = 0
        changed["report"]["batches"] = []
        self.assertEqual(make_study(changed), make_study(self.bundle))
        self.assertEqual(len(FILMED_BATCHES), 24)
        self.assertEqual(FILMED_BATCHES, sorted(set(FILMED_BATCHES)))

    def test_common_initialization_follows_optional_visual_drive(self):
        calls = []
        fake = types.SimpleNamespace(ParallelShots=types.SimpleNamespace(
            approach_endpoints=lambda env,q:calls.append(("canonical",q))))
        env = types.SimpleNamespace(film_trial=False)
        targets = {"Arm10": "original-target", "Arm14":"original-board-target"}
        with patch.dict(sys.modules,{"parallel_runtime":fake}), patch.object(
            ActionTrainingShots,"approach_endpoints",side_effect=lambda e,q:calls.append(("driven",q))):
            RLStudyShots.approach_endpoints(env,targets)
            self.assertEqual(calls,[("canonical",targets)])
            calls.clear();env.film_trial=True
            RLStudyShots.approach_endpoints(env,targets)
            self.assertEqual(calls,[("driven",targets),("canonical",targets)])

    def test_mode_options_cannot_silently_mix_other_recordings(self):
        self.assertFalse(validate_options(None,None,appearance="dark"))
        self.assertTrue(validate_options(Path("source"),Path("camera"),appearance="silver"))
        for args in ((None,Path("camera"),"silver",False),
                     (Path("source"),None,"silver",False),
                     (Path("source"),Path("camera"),"dark",False),
                     (Path("source"),Path("camera"),"silver",True)):
            with self.assertRaises(ValueError):
                validate_options(args[0],args[1],appearance=args[2],incompatible=args[3])

    def test_uncertainty_includes_all_attempts_and_is_not_a_point_claim(self):
        lo,hi=wilson(70,280)
        self.assertLess(lo,.25);self.assertGreater(hi,.25)
        self.assertEqual(wilson(0,0),[0,1])
        self.assertLess(wilson(0,280)[1],.02)
        self.assertGreater(wilson(280,280)[0],.98)

    def test_entrypoint_retains_historical_and_presentation_routes(self):
        code=(SOURCE/"film_cylinder_farm.py").read_text()
        self.assertIn('p.add_argument("--rl-study-source"',code)
        self.assertIn("run_study(env, presentation_bundle)",code)
        self.assertIn("run_presentation(env, presentation_bundle, args.presentation_batches)",code)
        self.assertIn("run_training(env, spec, args.geometry_menu_source)",code)


if __name__ == "__main__":
    unittest.main()
