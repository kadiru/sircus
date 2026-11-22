"""CPU-only evidence tamper controls for the bounded reference-stall fallback."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

HAS_DEPS = bool(importlib.util.find_spec("numpy") and importlib.util.find_spec("torch"))


@unittest.skipUnless(HAS_DEPS, "Admission audit imports require NumPy and CPU Torch")
class SharedVerifierParameterTests(unittest.TestCase):
    def setUp(self):
        from sircus.mjlab import direct_path_first_audit as shared
        from sircus.mjlab import stalled_reference_audit as audit
        from sircus.mjlab.tests.test_direct_path_first_audit import DirectPathFirstAuditTests
        self.shared, self.audit = shared, audit
        self.old = DirectPathFirstAuditTests()
        self.old.setUp()

    def fixture(self):
        """Rename only the outer lane in a fresh fixture; immutable inner names stay."""
        layouts, slots, observations, wrappers = self.old.fixture()
        for wrapper in wrappers:
            if wrapper["variant"] == "direct_first":
                wrapper["variant"] = "hybrid"
        from sircus.mjlab.planning_budget_plan import _wrap_choice
        slots = [_wrap_choice(wrapper["site"]["choice"], wrapper["variant"]) for wrapper in wrappers]
        observations = {key.replace("direct_first", "hybrid"): value for key, value in observations.items()}
        return layouts, slots, observations, wrappers

    def test_explicit_variants_keep_no_setup_in_denominator_without_global_mutation(self):
        before = self.shared.METHODS
        result = self.audit.summarize(*self.fixture())
        self.assertEqual(result["methods"]["baseline"]["capture_rate"], .5)
        self.assertEqual(result["methods"]["hybrid"]["capture_rate"], 1.)
        self.assertEqual(result["methods"]["baseline"]["no_candidate"], 1)
        self.assertEqual(self.shared.METHODS, before)
        self.assertEqual(self.shared.summarize(*self.old.fixture())["methods"]["direct_first"]["captures"], 2)

    def test_shared_slot_bindings_accept_explicit_outer_variants_only(self):
        layouts, slots, _, wrappers = self.fixture()
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("plans.json", "layouts.json", "neural-checkpoint.json", "contract.json"):
                (root / name).write_text("{}")
            bindings = {p.name: self.shared.sha(p) for p in root.glob("*.json")}
            (root / "choices.json").write_text(json.dumps(dict(slots=slots, bindings=bindings)))
            bindings = dict(bindings, **{"choices.json": self.shared.sha(root / "choices.json")})
            expected = {variant: [slot["request"] for slot in slots
                                  if slot["method"] == variant and slot["request"] is not None]
                        for variant in self.audit.METHODS}
            (root / "native-started.json").write_text(json.dumps(dict(
                bindings=bindings, requests_sha256=self.shared.digest(expected))))
            manifest = {p.name: self.shared.sha(p) for p in root.glob("*.json")}
            _, requests = self.shared.verify_slots(root, layouts, wrappers, manifest, self.shared.Checks(),
                                                   variants=self.audit.METHODS)
            self.assertEqual(set(requests), {"test_baseline", "test_hybrid"})
            with self.assertRaises(ValueError):
                self.shared.verify_slots(root, layouts, wrappers, manifest, self.shared.Checks())
            self.assertTrue(all(wrapper["site"]["method"] == "warm_early" for wrapper in wrappers))

    def test_shared_worker_lifecycle_keeps_independent_pools_with_explicit_variants(self):
        wrappers, layouts, spec = self.old.worker_fixture()
        spec["methods"] = list(self.audit.METHODS)
        for wrapper in wrappers:
            if wrapper["variant"] == "direct_first":
                wrapper["variant"] = "hybrid"
        self.shared.verify_workers(wrappers, layouts, spec, self.shared.Checks(), variants=self.audit.METHODS)
        before = deepcopy(self.shared.METHODS)
        with self.assertRaises(ValueError):
            self.shared.verify_workers(wrappers, layouts, spec, self.shared.Checks())
        self.assertEqual(self.shared.METHODS, before)

    def test_new_summary_rejects_changed_choice_or_extra_observation(self):
        fixture = self.fixture()
        fixture[1][0]["request"]["bucket_xy_m"][0] += .01
        with self.assertRaisesRegex(ValueError, "immutable inner choice"):
            self.audit.summarize(*fixture)
        fixture = self.fixture()
        fixture[2]["unexpected-drop"] = deepcopy(next(iter(fixture[2].values())))
        with self.assertRaises(ValueError):
            self.audit.summarize(*fixture)



@unittest.skipUnless(HAS_DEPS, "Admission audit imports require NumPy and CPU Torch")
class SwitchReceiptTests(unittest.TestCase):
    def fixture(self, directory, mode="fallback"):
        from types import SimpleNamespace
        from sircus.mjlab import stalled_reference_audit as audit
        from sircus.mjlab import stalled_reference_plan as core
        from sircus.mjlab.calibrated_admission import VERSION as REFERENCE_VERSION
        from sircus.mjlab.direct_path_first import VERSION as DIRECT_VERSION
        clock = SimpleNamespace(value=0.)
        timer = core.ReferencePathClock(clock=lambda: clock.value)
        profile = SimpleNamespace(job_id="pool-0-0", started=0.)
        layout = dict(id="case-0", context_id="case-0", bucket_xy_m=[.6, -.3])
        timer.begin(0., lambda event: None)
        receipt = core.QueryReceipt(directory, layout, profile, 30., timer)
        receipt.emit("query_started")
        menu = [dict(id="proposal-0", values=[.6, -.3, .2, 28., .3])]
        clock.value = .1
        receipt.selection_prefix = dict(menu_sha256=core._digest(menu), menu_count=1,
            ranked_proposal_ids=["proposal-0"], ranked_proposal_ids_sha256=core._digest(["proposal-0"]))
        receipt.emit("selection_prefix_frozen")
        clock.value = 1.
        timer.active = dict(index=0, start_s=1.)
        receipt.emit("reference_path_started")
        switched = mode == "fallback"
        if mode != "timeout":
            clock.value = 6.01 if switched else 2.
            timer.intervals = [dict(index=0, start_s=1., end_s=clock.value,
                duration_s=clock.value-1., completed=not switched, interrupted=switched)]
            timer.path_s = clock.value-1.
            timer.active = None
            timer.interrupted = switched
            receipt.emit("reference_path_interrupted" if switched else "reference_path_finished")
            if switched:
                clock.value = 6.02
                receipt.fallback = dict(reason="reference_path_budget_exhausted", trigger_elapsed_s=6.02,
                    reference_path_s=5.01, threshold_s=5., remaining_budget_s=23.98,
                    published_before_switch=0, reference_search_complete=False)
                receipt.emit("fallback_triggered")
                clock.value = 6.2
                receipt.mode = "direct_first"
                receipt.direct_initialized, receipt.initialization_s = True, .18
                receipt.emit("fallback_started")
            receipt.published = 1
            clock.value = 10.
            receipt.emit("query_completed")
        resolved = "direct-contexts/a.json" if switched else "resolutions/a.json"
        request = None if mode == "timeout" else dict(context_resolution_file=resolved)
        if request:
            path = directory / resolved
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(dict(version=DIRECT_VERSION if switched else REFERENCE_VERSION)))
        site = dict(layout=layout, job_id=profile.job_id, choice=dict(request=request),
                    status="deadline" if mode == "timeout" else "completed",
                    accepted_snapshot_sha256=None if mode == "timeout" else "snapshot-sha",
                    wall_planning_s=30.1 if mode == "timeout" else 10.1,
                    decision_latency_s=30. if mode == "timeout" else 10.1,
                    profile=dict(blocks=[dict(stage="arm_setup_checks", start_s=.5, end_s=30.)]),
                    plan=dict(neural=dict(admitted=[] if request is None else [request])),
                    accepted_receipts=[] if mode == "timeout" else [dict(elapsed_s=9.)])
        site["stalled_reference"] = core.read_receipt(directory, site)
        snapshots = [] if mode == "timeout" else [dict(stalled_reference_snapshot=dict(
            version=core.VERSION, job_id=profile.job_id, engine_mode=receipt.mode,
            selection_prefix=receipt.selection_prefix, fallback=receipt.fallback))]
        manifest = {path.relative_to(directory.parent).as_posix(): audit.sha(path)
                    for path in directory.rglob("*.json")}
        return dict(site=site, snapshots=snapshots, directory=directory, manifest=manifest,
                    checks=audit.Checks(), spec=dict(planning_budget_s=30., reference_path_s=5.),
                    variant="hybrid", expected_menu=menu, expected_order=menu)

    def rewrite_chain(self, fixture):
        """Rebind altered fixture bytes so semantic checks, not just hashes, fail."""
        from sircus.mjlab import stalled_reference_audit as audit
        receipt, previous = fixture["site"]["stalled_reference"], None
        for event, binding in zip(receipt["events"], receipt["receipt_files"]):
            event["previous_event_sha256"] = previous
            path = fixture["directory"] / binding["file"]
            path.write_text(json.dumps(event, sort_keys=True, separators=(",", ":")))
            binding["sha256"] = previous = audit.sha(path)
            fixture["manifest"][path.relative_to(fixture["directory"].parent).as_posix()] = previous
        receipt.update(receipt["events"][-1])

    def test_reference_fallback_and_no_candidate_timeout_have_auditable_receipts(self):
        from sircus.mjlab.stalled_reference_audit import verify_switch
        for mode in ("reference", "fallback", "timeout"):
            with self.subTest(mode=mode), TemporaryDirectory() as temporary:
                verify_switch(**self.fixture(Path(temporary), mode))

    def test_changed_event_bytes_or_omitted_interval_are_rejected(self):
        from sircus.mjlab.stalled_reference_audit import verify_switch
        with TemporaryDirectory() as temporary:
            fixture = self.fixture(Path(temporary))
            event_path = next((Path(temporary) / "stalled-reference-receipts").rglob("event-00003.json"))
            event_path.write_text("{}")
            with self.assertRaisesRegex(ValueError, "immutable event bytes"):
                verify_switch(**fixture)
        with TemporaryDirectory() as temporary:
            fixture = self.fixture(Path(temporary))
            fixture["site"]["stalled_reference"]["events"][3]["reference_path_intervals"] = []
            self.rewrite_chain(fixture)
            with self.assertRaises(ValueError):
                verify_switch(**fixture)

    def test_reference_trigger_cannot_claim_exhaustive_failure_or_extra_budget(self):
        from sircus.mjlab.stalled_reference_audit import verify_switch
        for field, value in (("reference_search_complete", True), ("remaining_budget_s", 30.),
                             ("published_before_switch", 1), ("reference_path_s", 4.)):
            with self.subTest(field=field), TemporaryDirectory() as temporary:
                fixture = self.fixture(Path(temporary))
                for event in fixture["site"]["stalled_reference"]["events"]:
                    if event["fallback"] is not None:
                        event["fallback"][field] = value
                self.rewrite_chain(fixture)
                with self.assertRaises(ValueError):
                    verify_switch(**fixture)

    def test_menu_cannot_change_after_switch_and_snapshot_cannot_predate_it(self):
        from sircus.mjlab.stalled_reference_audit import verify_switch
        with TemporaryDirectory() as temporary:
            fixture = self.fixture(Path(temporary))
            fixture["expected_menu"][0]["values"][4] += .01
            with self.assertRaisesRegex(ValueError, "independently rebuilt menu"):
                verify_switch(**fixture)
        with TemporaryDirectory() as temporary:
            fixture = self.fixture(Path(temporary))
            fixture["site"]["accepted_receipts"][0]["elapsed_s"] = 4.
            with self.assertRaisesRegex(ValueError, "before the actual stall"):
                verify_switch(**fixture)

    def test_path_clock_cannot_count_menu_or_other_phase_time(self):
        from sircus.mjlab.stalled_reference_audit import verify_switch
        with TemporaryDirectory() as temporary:
            fixture = self.fixture(Path(temporary))
            fixture["site"]["profile"]["blocks"][0]["stage"] = "menu"
            with self.assertRaisesRegex(ValueError, "excludes other planning phases"):
                verify_switch(**fixture)

    def test_selected_native_adapter_cannot_change_after_plan(self):
        from sircus.mjlab.stalled_reference_audit import verify_switch
        with TemporaryDirectory() as temporary:
            fixture = self.fixture(Path(temporary))
            fixture["site"]["stalled_reference"]["selected_adapter"] = "reference"
            with self.assertRaisesRegex(ValueError, "native adapter follows"):
                verify_switch(**fixture)


if __name__ == "__main__":
    unittest.main()
