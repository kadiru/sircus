"""Spatial novelty, paired accounting and outcome-blind preparation checks."""
import copy
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

try:
    from sircus.mjlab import stalled_reference_study as study
    from sircus.mjlab.corrected_contract import grid_layouts
    from sircus.mjlab.planning_budget_plan import _wrap_choice
except ModuleNotFoundError as error:
    # The legacy runner imported by this study requires Torch. Lightweight CI
    # deliberately omits it; do not conceal a different missing dependency.
    if error.name != "torch":
        raise
    study = None


def provenance():
    grid = {row["id"]: row["bucket_xy_m"] for row in grid_layouts()}
    return dict(excluded_coordinates_m=[*grid.values(), [.9712010092480674, -.2853881822068356]],
        grid=grid, bounds=[[.1, 1.24], [-.65, .25]],
        geometry=dict(fixed_arm_bounds={
            "Arm10": dict(center_xy_m=[0., 0.], radius_m=.09),
            "Arm14": dict(center_xy_m=[1.15, 0.], radius_m=.09)},
            bucket_radius_m=.075, tabletop_xy_bounds_m=[[-.2, -.9], [1.5, .5]]))


def choice(layout, admitted=True):
    request = dict(id=layout["id"]+"-proposal-b0", context_id=layout["id"],
        bucket_xy_m=layout["bucket_xy_m"], intended_route="direct", method="warm_early",
        values=[.5, -.2, .3, 28., .35], native_prepared=True,
        prepared_connection_file="connections/a.json", prepared_connection_sha256="a"*64,
        context_resolution_file="resolutions/b.json", context_resolution_sha256="b"*64,
        selected_q10=[0.]*6, selected_q14=[.1]*6,
        board_heading_offset_deg=0., board_twist_deg=0., heading_key="fixture")
    return dict(id=layout["id"]+"-warm_early", context_id=layout["id"], method="warm_early",
        bucket_xy_m=layout["bucket_xy_m"], stratum=layout.get("stratum", "fixture"),
        intended_route="direct" if admitted else None, request=request if admitted else None)


def outcome(label):
    return dict(invalid_physics=False, route_success=label in {"direct", "table"},
        shot_route=label if label in {"direct", "table"} else None,
        experimental_route=dict(robot_assisted_capture=label == "arm_assisted_capture"))


@unittest.skipIf(study is None, "Pinned study imports require Torch; covered on Ubuntu CPU")
class StudyTests(unittest.TestCase):
    def test_promotion_requires_capture_timing_and_known_target_gates(self):
        comparison = dict(methods={v:dict(captures=16) for v in study.METHODS},
                          latency=dict(baseline=dict(mean_s=10.), hybrid=dict(mean_s=10.5)))
        diagnostic = dict(methods=dict(hybrid=dict(outcome="table")))
        # Use the evaluator's versioned success labels rather than display labels.
        from sircus.mjlab.intergrid_route_selector_plan import STANDARD_CAPTURES
        diagnostic["methods"]["hybrid"]["outcome"] = next(iter(STANDARD_CAPTURES))
        self.assertTrue(study.promotion_decision(comparison, diagnostic)["eligible"])
        for field, value in (("captures", 15), ("mean_s", 11.01)):
            changed = copy.deepcopy(comparison)
            changed["methods" if field == "captures" else "latency"]["hybrid"][field] = value
            self.assertFalse(study.promotion_decision(changed, diagnostic)["eligible"])
        diagnostic["methods"]["hybrid"]["outcome"] = "flight_miss"
        self.assertFalse(study.promotion_decision(comparison, diagnostic)["eligible"])

    def test_mixed_native_dispatch_preserves_order_and_unique_packets(self):
        pair = {v:SimpleNamespace(adapter=SimpleNamespace(packet=0), mode=v)
                for v in ("reference", "direct_first")}
        requests = [dict(id=str(i), context_resolution_file=f"{folder}/{i}.json")
                    for i, folder in enumerate(("resolutions", "direct-contexts", "direct-contexts", "resolutions"))]
        calls = []
        def execute(engine, rows, phase, directory):
            calls.append((engine.mode, engine.adapter.packet, [r["id"] for r in rows]))
            engine.adapter.packet += 1
            return rows
        self.assertEqual(study.execute_mixed(pair, requests, "test", "native-hybrid", execute), requests)
        self.assertEqual(calls, [("reference", 0, ["0"]), ("direct_first", 1, ["1", "2"]), ("reference", 2, ["3"])])
        pair["direct_first"] = None
        with self.assertRaises(ValueError):
            study.execute_mixed(pair, requests[1:2], "test", "native-baseline", execute)

    def test_fresh20_are_reproducible_spatially_disjoint_and_stratified(self):
        inputs = provenance()
        before = copy.deepcopy(inputs)
        rows = study.make_layouts(inputs)
        self.assertEqual(inputs, before)
        self.assertEqual(rows, study.make_layouts(inputs))
        self.assertEqual(len(rows), 20)
        self.assertEqual(len({row["id"] for row in rows}), 20)
        seen = list(inputs["excluded_coordinates_m"])
        for row in rows:
            self.assertEqual(row["context_id"], row["id"])
            self.assertEqual(row["split"], "test")
            self.assertEqual(row["stratum"], row["region"])
            self.assertGreaterEqual(min(math.dist(row["bucket_xy_m"], xy) for xy in seen), .005)
            seen.append(row["bucket_xy_m"])
        self.assertEqual(sum("fixed base" in row["region"] for row in rows), 4)
        self.assertEqual(sum("grid edge" in row["region"] for row in rows), 8)
        self.assertNotEqual(rows, study.make_layouts(inputs, seed=study.SEED+2))

    def test_summary_counts_no_setup_and_flight_miss_in_full_denominator(self):
        layouts = [dict(id=f"site-{i}", bucket_xy_m=[.3+i*.1, -.3]) for i in range(4)]
        labels = [("direct", "table"), (None, "direct"), ("direct", "flight_miss"),
                  ("flight_miss", None)]
        slots, records, wrappers = [], {}, []
        for layout, pair in zip(layouts, labels):
            for variant, label in zip(study.METHODS, pair):
                slot = _wrap_choice(choice(layout, admitted=label is not None), variant)
                slots.append(slot)
                if label is not None:
                    records[slot["id"]] = outcome(label)
                wrappers.append(dict(variant=variant, site=dict(request_wall_s=3., decision_latency_s=2.)))
        comparison = study.summarize(layouts, slots, records, wrappers)
        self.assertEqual(comparison["paired"], dict(both=1, gained=1, lost=1, neither=1))
        for method in study.METHODS:
            row = comparison["methods"][method]
            self.assertEqual((row["positions"], row["admitted"], row["no_candidate"], row["captures"]), (4, 3, 1, 2))
            self.assertEqual(row["capture_rate"], .5)
            self.assertEqual(row["setup_rate"], .75)
            self.assertEqual(row["flight_misses"], 1)
        with self.assertRaises(ValueError):
            study.summarize(layouts, slots, dict(records, unexpected=outcome("direct")), wrappers)

    def test_arm_assisted_capture_does_not_silently_change_primary_metric(self):
        layouts = [dict(id="site", bucket_xy_m=[.7, -.2])]
        slots = [_wrap_choice(choice(layouts[0]), method) for method in study.METHODS]
        records = {slot["id"]: outcome("arm_assisted_capture") for slot in slots}
        wrappers = [dict(variant=v, site=dict(request_wall_s=1., decision_latency_s=1.)) for v in study.METHODS]
        result = study.summarize(layouts, slots, records, wrappers)
        self.assertEqual(result["paired"]["neither"], 1)
        for counts in result["methods"].values():
            self.assertEqual(counts["captures"], 0)
            self.assertEqual(counts["outcomes"]["arm_assisted_capture"], 1)

    def test_plan_lanes_receive_identical_inputs_and_layout_ids(self):
        created, calls, saved = [], [], []
        def baseline_initializer():
            pass
        class Planner:
            def __init__(self, method, libs, model, history, config, neural, **kwargs):
                self.method, self.arguments, self.kwargs = method, (libs, model, history, config, neural), kwargs
                self.worker = SimpleNamespace(initializer=baseline_initializer)
                self.variant = kwargs["directory"].name.removeprefix("native-")
                created.append(self)
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def plan(self, layout):
                calls.append((self.variant, copy.deepcopy(layout), self.worker.initializer))
                return dict(status="completed", choice=choice(layout))
        layouts = [dict(id="site-0", bucket_xy_m=[.6, -.3]), dict(id="site-1", bucket_xy_m=[.8, -.4])]
        args = SimpleNamespace(output=Path("/fixture/study"), assets=Path("/fixture/assets"))
        frozen = ({"scene": "fixture"}, {"frozen_library": []}, {"model": []},
                  {"training_history": []}, {"network": []}, {"prior": []}, {"rule": []})
        with patch.object(study, "planner_inputs", return_value=frozen), \
             patch.object(study, "read", return_value={"progress": {}}) as reads, \
             patch.object(study, "publish"), patch.object(study, "require_storage"), \
             patch.object(study, "atomic", side_effect=lambda path, data: saved.append((path, data))), \
             patch("sircus.mjlab.fast_planning_plan.ReusablePlanner", Planner), \
             patch("sircus.mjlab.stalled_reference_plan.StalledReferencePlanner", Planner):
            result = study.plan_batch(args, layouts, args.output)
        self.assertEqual(len(result), 4)
        self.assertEqual([p.method for p in created], ["warm_early", "warm_early"])
        self.assertEqual(created[0].arguments, created[1].arguments)
        self.assertEqual(created[0].kwargs["budget_s"], created[1].kwargs["budget_s"])
        for layout in layouts:
            pair = [call for call in calls if call[1]["id"] == layout["id"]]
            self.assertEqual(len(pair), 2)
            self.assertEqual(pair[0][1], pair[1][1])
        initializers = {variant: initializer for variant, _, initializer in calls}
        self.assertIs(initializers["baseline"], study.initialize_baseline)
        self.assertIs(initializers["hybrid"], baseline_initializer)
        self.assertEqual(reads.call_args_list[0].args, (args.output/"report.json",))
        self.assertEqual(reads.call_count, 1)  # No current native results enter selection.

    def test_preparation_freezes_both_choices_before_any_native_observations(self):
        layouts = [dict(id="site", bucket_xy_m=[.6, -.3])]
        original_choice = choice(layouts[0])
        sites = {("site", v): dict(choice=copy.deepcopy(original_choice)) for v in study.METHODS}
        saved = {}
        args = SimpleNamespace(output=Path("/fixture/study"))
        contract = dict(processing_order=["site"], frozen_inputs={"layouts.json": "hash"})
        def read(path):
            if path.name == "layouts.json":
                return {"layouts": layouts}
            if path.name == "report.json":
                return {"progress": {}}
            raise AssertionError("Preparation attempted to read outcomes: " + str(path))
        with patch.object(study, "frozen_check", return_value=contract), \
             patch.object(study, "plan_batch", return_value=sites), patch.object(study, "read", side_effect=read), \
             patch.object(study, "sha", return_value="hash"), patch.object(study, "publish"), \
             patch.object(study, "atomic", side_effect=lambda path, data: saved.__setitem__(path.name, data)):
            study.prepare(args)
        slots = saved["choices.json"]["slots"]
        self.assertEqual(len(slots), 2)
        self.assertEqual(len({s["request"]["id"] for s in slots}), 2)
        for slot in slots:
            for key, value in original_choice["request"].items():
                if key not in {"id", "method"}:
                    self.assertEqual(slot["request"][key], value)
        self.assertEqual(sites["site", "baseline"]["choice"], original_choice)
        self.assertEqual(set(saved), {"plans.json", "choices.json"})


class DependencyGuardTests(unittest.TestCase):
    def test_collection_without_torch_skips_only_study_cases(self):
        code = """
import builtins, runpy, sys
original_import = builtins.__import__
def without_torch(name, *args, **kwargs):
    if name == 'torch' or name.startswith('torch.'):
        raise ModuleNotFoundError('Torch intentionally absent in test', name='torch')
    return original_import(name, *args, **kwargs)
builtins.__import__ = without_torch
scope = runpy.run_path(sys.argv[1], run_name='dependency_probe')
assert scope['study'] is None
assert scope['StudyTests'].__unittest_skip__ is True
assert not getattr(scope['DependencyGuardTests'], '__unittest_skip__', False)
"""
        result = subprocess.run([sys.executable, "-c", code, str(Path(__file__).resolve())],
            capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
