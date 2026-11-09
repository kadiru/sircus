"""Synthetic input lineage tests; no simulator, checkpoint inference or optimizer."""

import ast
import copy
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import candidate_continuation_inputs as inputs
import candidate_continuation_readiness as readiness
from candidate_continuation_policy import freeze_support
from continuous_feasibility import ACTION_NAMES, canonical_hash, decode_action


def sign(value):
    result = copy.deepcopy(value)
    result["fingerprint"] = canonical_hash(result)
    return result


class TeacherJoinTests(unittest.TestCase):
    def setUp(self):
        self.cells = [dict(id="train-000", split="train", bucket_xy_m=[.4, -.2]),
                      dict(id="train-001", split="train", bucket_xy_m=[.5, -.2])]
        self.supports, self.records, self.successes = [], [], []
        for i, cell in enumerate(self.cells):
            parent = decode_action(cell, [0., .4, .35, .24, .5], "parent")
            action = decode_action(cell, [0., .4, .36, .24, .5], "success-"+str(i))
            key = "trials/train-00000.json.gz#"+str(i)
            identifier = canonical_hash(dict(cell_id=cell["id"], values=[action[n] for n in ACTION_NAMES]))
            record = dict(candidate_id=identifier, cell_id=cell["id"], action=action,
                          sources=[dict(kind="historical_train_success", case_id=key)])
            self.records.append(record)
            self.supports.append(freeze_support(cell["id"], [record], parent, [0.]*5))
            self.successes.append(dict(success_key=key, cell=cell, action=action, clean=i == 0))
        self.evidence = sign(dict(cells=self.cells, successes=self.successes))

    def join(self):
        return inputs.teacher_choices(self.evidence, self.cells, self.supports, self.records)

    def test_exact_success_join_has_no_average_or_invented_labels(self):
        result = self.join()
        self.assertEqual([r["index"] for r in result], [0, 0])
        self.assertEqual([r["clean"] for r in result], [True, False])
        self.assertEqual([r["candidate_id"] for r in result], [r["candidate_id"] for r in self.records])

    def test_changed_action_or_unqualified_success_fails(self):
        changed = copy.deepcopy(self.successes)
        changed[0]["action"]["height_m"] += .01
        self.evidence = sign(dict(cells=self.cells, successes=changed))
        with self.assertRaisesRegex(ValueError, "action changed"):
            self.join()
        self.evidence = sign(dict(cells=self.cells, successes=self.successes))
        other = decode_action(self.cells[0], [0., .4, .37, .24, .5], "not-success")
        self.supports[0] = freeze_support(self.cells[0]["id"], [dict(candidate_id="other", action=other)], other, [0.]*5)
        with self.assertRaisesRegex(ValueError, "excluded"):
            self.join()

    def test_duplicate_and_unlisted_demonstration_receipts_fail(self):
        self.evidence = sign(dict(cells=self.cells, successes=self.successes+[self.successes[0]]))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.join()
        self.evidence = sign(dict(cells=self.cells, successes=self.successes[:1]))
        with self.assertRaisesRegex(ValueError, "membership differs"):
            self.join()

    def test_validation_or_changed_context_cannot_be_teacher_data(self):
        for field, value in (("split", "validation"), ("bucket_xy_m", [.41, -.2])):
            evidence = copy.deepcopy(self.evidence)
            evidence.pop("fingerprint")
            evidence["successes"][0]["cell"][field] = value
            self.evidence = sign(evidence)
            with self.assertRaisesRegex(ValueError, "TRAIN|changed"):
                self.join()
            self.evidence = sign(dict(cells=self.cells, successes=self.successes))

    def test_evidence_fingerprint_is_required_before_join(self):
        self.evidence["successes"][0]["clean"] = False
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            self.join()

    def test_inverse_roundoff_check_preserves_the_signed_stored_values(self):
        support = self.supports[0]
        raw = support.to_dict()
        raw["latents"][0][0] = math.nextafter(raw["latents"][0][0], math.inf)
        raw["support_fingerprint"] = canonical_hash({k:v for k,v in raw.items() if k != "support_fingerprint"})
        before = copy.deepcopy(raw)
        parent = dict(exact_action=decode_action(self.cells[0], [0., .4, .35, .24, .5], "parent"),
                      latent_mean=[0.]*5)
        checked = inputs.checked_support(self.cells[0]["id"], parent, raw)
        self.assertEqual(checked.to_dict(), before)
        self.assertEqual(raw, before)
        raw["latents"][0][2] += 1e-5
        raw["support_fingerprint"] = canonical_hash({k:v for k,v in raw.items() if k != "support_fingerprint"})
        with self.assertRaisesRegex(ValueError, "encoding changed"):
            inputs.checked_support(self.cells[0]["id"], parent, raw)

    def test_original_parent_latent_remains_exact_even_below_inverse_tolerance(self):
        parent = decode_action(self.cells[0], [0., .4, .35, .24, .5], "parent")
        support = freeze_support(self.cells[0]["id"], [dict(candidate_id="parent", action=parent)], parent, [0.]*5)
        raw = support.to_dict()
        raw["latents"][0][0] = math.nextafter(0., math.inf)
        raw["support_fingerprint"] = canonical_hash({k:v for k,v in raw.items() if k != "support_fingerprint"})
        with self.assertRaisesRegex(ValueError, "original parent latent"):
            inputs.checked_support(self.cells[0]["id"], dict(exact_action=parent, latent_mean=[0.]*5), raw)

    def test_readiness_checks_cpu_and_external_export_hash_before_loading_a_model(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"not-an-export.json"
            path.write_text('{}')
            with patch.object(readiness, "torch", object()), patch.object(readiness, "load_checkpoint") as load:
                with patch.dict(readiness.os.environ, {"CUDA_VISIBLE_DEVICES":"0"}):
                    with self.assertRaisesRegex(ValueError, "CPU-only"):
                        readiness.check_readiness(path, path, expected_inputs_sha256="0"*64)
                with patch.dict(readiness.os.environ, {"CUDA_VISIBLE_DEVICES":""}):
                    with self.assertRaisesRegex(ValueError, "export bytes changed"):
                        readiness.check_readiness(path, path, expected_inputs_sha256="0"*64)
                load.assert_not_called()

    def test_adapter_commands_contain_no_optimizer_or_native_launch(self):
        root = Path(inputs.__file__).parent
        for name in inputs.MODULES[:5]:
            source = (root/name).read_text()
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Attribute):
                        self.assertNotIn(node.func.attr, ("step", "backward", "Popen", "execv"), name)
                    elif isinstance(node.func, ast.Name):
                        self.assertNotIn(node.func.id, ("AppLauncher", "Adam", "SGD", "run_benchmark"), name)
            self.assertNotIn("import isaac", source)


if __name__ == "__main__":
    unittest.main()
