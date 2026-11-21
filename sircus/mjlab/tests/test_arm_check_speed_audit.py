"""CPU controls for outer comparison identities and complete attempt accounting."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory
import unittest


HAS_DEPS = bool(importlib.util.find_spec("numpy") and importlib.util.find_spec("torch"))


@unittest.skipUnless(HAS_DEPS, "Admission audit imports require NumPy and CPU Torch")
class ArmCheckSpeedAuditTests(unittest.TestCase):
    def setUp(self):
        from sircus.mjlab import arm_check_speed_audit as audit
        self.audit = audit

    def fixture(self):
        from sircus.mjlab.planning_budget_plan import _wrap_choice
        layouts = [dict(id=f"case-{index}", context_id=f"case-{index}", split="test",
                        bucket_xy_m=[.4 + index * .1, -.3]) for index in range(2)]
        wrappers, slots, observations = [], [], {}
        for index, layout in enumerate(layouts):
            for variant in self.audit.METHODS:
                released = not (index == 1 and variant == "baseline")
                inner_id = layout["id"] + "-warm_early"
                request = dict(id=inner_id, context_id=layout["id"], method="warm_early",
                               bucket_xy_m=layout["bucket_xy_m"], intended_route="direct") if released else None
                choice = dict(id=inner_id, context_id=layout["id"], method="warm_early",
                              bucket_xy_m=layout["bucket_xy_m"], intended_route="direct" if released else None,
                              request=request)
                site = dict(layout=layout, method="warm_early", choice=choice,
                            request_wall_s=12. if variant == "baseline" else 8.,
                            decision_latency_s=10. if variant == "baseline" else 6.)
                wrappers.append(dict(variant=variant, site=site))
                slot = _wrap_choice(choice, variant)
                slots.append(slot)
                if released:
                    observations[slot["id"]] = dict(
                        invalid_physics=False, route_success=True, shot_route="direct",
                        experimental_route=dict(robot_assisted_capture=False))
        return layouts, slots, observations, wrappers

    def test_no_setup_stays_in_denominator_and_latency_includes_overhead(self):
        result = self.audit.summarize(*self.fixture())
        self.assertEqual(result["methods"]["baseline"]["captures"], 1)
        self.assertEqual(result["methods"]["baseline"]["capture_rate"], .5)
        self.assertEqual(result["methods"]["baseline"]["no_candidate"], 1)
        self.assertEqual(result["methods"]["optimized"]["capture_rate"], 1.)
        self.assertEqual(result["paired"], dict(both=1, gained=1, lost=0, neither=0))
        self.assertEqual(result["latency"]["baseline"]["mean_s"], 12.)
        self.assertEqual(result["decision_latency"]["baseline"]["mean_s"], 10.)

    def test_summary_matches_producer_on_frozen_fixture(self):
        from sircus.mjlab.arm_check_speed_study import summarize
        fixture = self.fixture()
        self.assertEqual(self.audit.summarize(*fixture), summarize(*fixture))

    def test_outer_slot_cannot_change_inner_choice(self):
        fixture = self.fixture()
        fixture[1][0]["request"]["bucket_xy_m"][0] += .01
        with self.assertRaisesRegex(ValueError, "immutable inner choice"):
            self.audit.summarize(*fixture)

    def test_extra_observation_or_duplicate_wrapper_is_rejected(self):
        fixture = self.fixture()
        fixture[2]["unscheduled-drop"] = deepcopy(next(iter(fixture[2].values())))
        with self.assertRaises(ValueError):
            self.audit.summarize(*fixture)
        fixture = self.fixture()
        fixture[3].append(deepcopy(fixture[3][0]))
        with self.assertRaisesRegex(ValueError, "One ordered original plan"):
            self.audit.summarize(*fixture)

    def test_robot_assisted_capture_remains_separate(self):
        fixture = self.fixture()
        measurement = fixture[2]["case-1-optimized"]
        measurement.update(route_success=False, shot_route=None,
                           experimental_route=dict(robot_assisted_capture=True))
        result = self.audit.summarize(*fixture)
        self.assertEqual(result["methods"]["optimized"]["captures"], 1)
        self.assertEqual(result["methods"]["optimized"]["outcomes"]["arm_assisted_capture"], 1)
        self.assertEqual(result["paired"]["neither"], 1)

    def slot_files(self, root):
        layouts, slots, _, wrappers = self.fixture()
        for name in ("plans.json", "layouts.json", "neural-checkpoint.json", "contract.json"):
            (root / name).write_text("{}")
        bindings = {p.name: self.audit.sha(p) for p in root.glob("*.json")}
        (root / "choices.json").write_text(json.dumps(dict(slots=slots, bindings=bindings)))
        native_bindings = dict(bindings, **{"choices.json": self.audit.sha(root / "choices.json")})
        expected = {variant: [slot["request"] for slot in slots
                              if slot["method"] == variant and slot["request"] is not None]
                    for variant in self.audit.METHODS}
        (root / "native-started.json").write_text(json.dumps(dict(
            bindings=native_bindings, requests_sha256=self.audit.digest(expected))))
        manifest = {p.name: self.audit.sha(p) for p in root.glob("*.json")}
        return layouts, wrappers, manifest

    def test_release_receipts_bind_outer_identity_without_rewriting_inner(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            layouts, wrappers, manifest = self.slot_files(root)
            slots, requests = self.audit.verify_slots(root, layouts, wrappers, manifest, self.audit.Checks())
            self.assertEqual(len(slots), 4)
            self.assertEqual(len(requests["test_baseline"]), 1)
            self.assertEqual(len(requests["test_optimized"]), 2)
            self.assertTrue(all(item["site"]["choice"]["method"] == "warm_early" for item in wrappers))
            receipt = self.audit.read(root / "native-started.json")
            receipt["requests_sha256"] = "0" * 64
            (root / "native-started.json").write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError, "frozen before test drops"):
                self.audit.verify_slots(root, layouts, wrappers, manifest, self.audit.Checks())

    def worker_fixture(self):
        layouts = [dict(id=f"case-{i}") for i in range(3)]
        order = [item["id"] for item in layouts]
        random.Random(9).shuffle(order)
        spec = dict(methods=list(self.audit.METHODS), processing_seed=9, processing_order=order)
        wrappers = []
        for variant_index, variant in enumerate(self.audit.METHODS):
            for index, identity in enumerate(order):
                generation, job = ((0, 0), (0, 1), (1, 0))[index]
                reused = index == 1
                pid = 100 + 10 * variant_index + generation
                site = dict(
                    layout=dict(id=identity), method="warm_early", worker_pool_id=variant,
                    worker_generation=generation, worker_job_index=job, worker_pid=pid,
                    worker_reused=reused, prior_worker_status=(None, "completed", "deadline")[index],
                    status=("completed", "deadline", "completed")[index], startup_s=0. if reused else 1.,
                    job_id=f"{variant}-{generation}-{job}",
                    readiness_receipt=None if reused else dict(kind="ready", pid=pid))
                wrappers.append(dict(variant=variant, site=site))
        return wrappers, layouts, spec

    def test_worker_reuse_is_separate_by_variant_despite_same_inner_method(self):
        fixture = self.worker_fixture()
        self.audit.verify_workers(*fixture, self.audit.Checks())
        bad = next(item["site"] for item in fixture[0] if item["site"]["prior_worker_status"] == "deadline")
        bad.update(worker_reused=True, readiness_receipt=None)
        with self.assertRaisesRegex(ValueError, "reuse only after clean"):
            self.audit.verify_workers(*fixture, self.audit.Checks())


if __name__ == "__main__":
    unittest.main()
