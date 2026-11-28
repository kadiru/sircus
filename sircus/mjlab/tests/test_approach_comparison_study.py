"""Frozen-model protocol, fair order, exact releases and fail-closed preflight."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from sircus.mjlab import approach_comparison_study as study
from sircus.mjlab import approach_comparison_audit as audit
from sircus.mjlab.benchmark_verify import digest, read, sha
from sircus.mjlab.sweep_worker import atomic


class ApproachComparisonStudyTests(unittest.TestCase):
    def test_balanced_order_is_seeded_and_independent_of_outcomes(self):
        layouts=[dict(id=f"p{i}", bucket_xy_m=[i/100, 0]) for i in range(50)]
        order, methods=study.schedule(layouts)
        self.assertEqual(set(order),{r["id"] for r in layouts})
        self.assertNotEqual(order,[r["id"] for r in layouts])
        self.assertEqual(sum(v[0]=="current" for v in methods.values()),25)
        self.assertTrue(all(set(v)==set(study.METHODS) for v in methods.values()))
        self.assertEqual((order,methods),study.schedule([dict(r,success=False) for r in layouts]))

    def test_release_schedule_keeps_declines_out_of_native_but_not_slots(self):
        layouts=[dict(id=f"p{i}") for i in range(4)]
        order, methods=study.schedule(layouts)
        slots=[dict(context_id=l["id"], method=m,
                    request=None if l["id"]=="p1" and m=="expanded" else dict(id=l["id"]+m))
               for l in layouts for m in study.METHODS]
        planned=study.planned_requests(dict(processing_order=order,method_order=methods),slots)
        self.assertEqual([r["id"] for r in planned["current"]],[identity+"current" for identity in order])
        self.assertEqual([r["id"] for r in planned["expanded"]],[identity+"expanded" for identity in order if identity!="p1"])
        self.assertEqual(len(slots),8)

    def test_accounting_includes_setup_declines_and_reports_nearest_rank_p90(self):
        layouts=[dict(id=f"p{i}",bucket_xy_m=[i/100,0]) for i in range(50)]
        slots=[]; rows=[]; plans=[]
        for i,layout in enumerate(layouts):
            for method in study.METHODS:
                identity=layout["id"]+method
                admitted=not(i==0 and method=="expanded")
                slots.append(dict(id=identity,context_id=layout["id"],method=method,request={} if admitted else None))
                plans.append(dict(variant=method,site=dict(decision_latency_s=float(i+1),request_wall_s=float(i+2))))
                if admitted:
                    rows.append(dict(id=identity,adapter_directory="native-"+method,board_first_scalar_audit_passed=True,
                                     board_first_score=dict(route_success=(i%2==0),invalid_physics=False)))
        result=study.summarize(layouts,slots,rows,plans)
        self.assertEqual(result["methods"]["expanded"]["captures"],24)
        self.assertEqual(result["methods"]["expanded"]["capture_rate"],24/50)
        self.assertEqual(result["methods"]["expanded"]["setup_rejections"],1)
        self.assertEqual(result["methods"]["expanded"]["p90_planning_s"],45.)
        self.assertEqual(result["methods"]["expanded"]["p90_request_s"],46.)
        self.assertEqual(result["paired"],dict(both=24,current_only=1,expanded_only=0,neither=25))
        with self.assertRaises(ValueError):study.summarize(layouts,slots,rows+rows[:1],plans)
        with self.assertRaises(ValueError):study.summarize(layouts,slots,rows[1:],plans)

    def parent_fixture(self, root):
        source=root/"parent"; source.mkdir()
        neural=root/"neural"; neural.mkdir()
        library=root/"library"; library.mkdir()
        names=(*study.COPIED,"current-checkpoint.json","expanded-checkpoint.json","fit.json",
               "train-dataset.jsonl","validation-dataset.jsonl","candidate-sources.json",
               "library-summary.json","library-manifest.json","all-positions.json","layouts.json")
        for i,name in enumerate(names):atomic(source/name,dict(marker=i))
        for name in study.COPIED:(neural/name).write_bytes((source/name).read_bytes())
        (neural/"neural-checkpoint.json").write_bytes((source/"current-checkpoint.json").read_bytes())
        for name in names:
            if name not in (*study.COPIED,"current-checkpoint.json","expanded-checkpoint.json","fit.json","layouts.json"):
                (library/name).write_bytes((source/name).read_bytes())
        atomic(source/"evidence-manifest.json",{name:sha(source/name) for name in names})
        atomic(source/"report.json",dict(status="completed",evidence_manifest_sha256=sha(source/"evidence-manifest.json")))
        proof=root/"parent-audit.json"
        atomic(proof,dict(status="passed",report_sha256=sha(source/"report.json"),evidence_manifest_sha256=sha(source/"evidence-manifest.json")))
        return SimpleNamespace(expanded_study=source, expanded_audit=proof, neural_study=neural, library=library)

    def test_audited_parent_requires_exact_library_and_model_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            a=self.parent_fixture(Path(folder))
            result=study.expanded_parent_binding(a)
            self.assertIn("expanded-checkpoint.json",result["inputs"])
            atomic(a.expanded_study/"expanded-checkpoint.json",dict(changed=True))
            with self.assertRaisesRegex(ValueError,"input changed"):
                study.expanded_parent_binding(a)
        with tempfile.TemporaryDirectory() as folder:
            a=self.parent_fixture(Path(folder))
            atomic(a.neural_study/"neural-checkpoint.json",dict(changed=True))
            with self.assertRaisesRegex(ValueError,"current frozen model"):
                study.expanded_parent_binding(a)

    def test_current_lane_binds_archived_scene_ranking_and_deadlines(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            a=self.parent_fixture(root)
            a.current_planner_study=a.expanded_study
            a.current_planner_audit=a.expanded_audit
            (a.current_planner_study/"neural-checkpoint.json").write_bytes((a.neural_study/"neural-checkpoint.json").read_bytes())
            base=dict(scene=dict(gravity=-9.81),identity=dict(name="native64"),flight_ranking=dict(rule=dict(coefficient=.65)),
                      planning_budget_s=30.,reference_path_s=5.,early_score_m=.005,physics_substeps=64)
            atomic(a.neural_study/"contract.json",base)
            def save_contract(value):
                contract=deepcopy(value);contract["fingerprint"]=digest(contract)
                atomic(a.current_planner_study/"contract.json",contract)
                manifest=read(a.current_planner_study/"evidence-manifest.json")
                for name in ("contract.json","neural-checkpoint.json"):
                    manifest[name]=sha(a.current_planner_study/name)
                atomic(a.current_planner_study/"evidence-manifest.json",manifest)
                atomic(a.current_planner_study/"report.json",dict(status="completed",evidence_manifest_sha256=sha(a.current_planner_study/"evidence-manifest.json")))
                atomic(a.current_planner_audit,dict(status="passed",report_sha256=sha(a.current_planner_study/"report.json"),
                      evidence_manifest_sha256=sha(a.current_planner_study/"evidence-manifest.json")))
            save_contract(base)
            self.assertEqual(study.current_planner_binding(a)["configuration"]["flight_ranking"],base["flight_ranking"])
            for key,value in (("scene",{}),("identity",{}),("flight_ranking",dict(rule=dict(coefficient=.66))),
                              ("planning_budget_s",31.),("reference_path_s",6.),("early_score_m",.006),("physics_substeps",128)):
                changed=deepcopy(base);changed[key]=value;save_contract(changed)
                with self.subTest(key=key):
                    with self.assertRaises(ValueError):study.current_planner_binding(a)

    def test_freeze_copies_models_and_never_fits(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); a=self.parent_fixture(root)
            a.parent_study=root/"controls";a.parent_study.mkdir()
            atomic(a.parent_study/"setup-home.json",dict(home=True))
            a.output=root/"new"
            a.assets=root/"assets"
            layouts=dict(layouts=[dict(id=f"p{i}") for i in range(50)],provenance={})
            spec=dict(identity={},scene={})
            with patch.object(study,"cpu_only"),patch.object(study,"source_inputs",return_value=(spec,{}, {},dict(flight_ranking={}),{})), \
                 patch.object(study,"library_inputs"),patch.object(study,"fresh_layouts",return_value=layouts), \
                 patch.object(study,"current_planner_binding",return_value={}), \
                 patch("sircus.mjlab.corrected_grid.source_manifest",return_value={}), \
                 patch("sircus.mjlab.expanded_selector_model.fit",side_effect=AssertionError("No fitting allowed")):
                study.freeze(a)
            contract=read(a.output/"contract.json")
            self.assertFalse(contract["training_performed"])
            self.assertEqual(contract["seed"],2026112717)
            self.assertEqual(contract["max_native_releases"],118)
            for method in study.METHODS:
                self.assertEqual(sha(a.output/(method+"-checkpoint.json")),sha(a.expanded_study/(method+"-checkpoint.json")))
            with patch.object(study,"cpu_only"),patch.object(study,"source_inputs",return_value=(spec,{}, {},{},{})),patch.object(study,"library_inputs"):
                with self.assertRaisesRegex(ValueError,"existing study"):study.freeze(a)

    def preflight_fixture(self,root):
        output=root/"study";output.mkdir()
        atomic(output/"contract.json",dict(methods=list(study.METHODS)))
        for method in study.METHODS:atomic(output/(method+"-checkpoint.json"),dict(method=method))
        bindings={p.name:sha(p) for p in output.iterdir()}
        atomic(output/"choices.json",dict(bindings=bindings,slots=[]))
        expected=dict(bindings,**{"choices.json":sha(output/"choices.json")})
        receipt=root/"preflight.json"
        atomic(receipt,dict(version=audit.VERSION,status="passed",stage="preflight",
            contract_sha256=sha(output/"contract.json"),choices_sha256=sha(output/"choices.json"),
            preparation_manifest_sha256=digest(expected),
            checkpoint_sha256={m:sha(output/(m+"-checkpoint.json")) for m in study.METHODS}))
        return SimpleNamespace(output=output,preflight_output=receipt)

    def test_native_requires_exact_independent_preflight(self):
        with tempfile.TemporaryDirectory() as folder:
            a=self.preflight_fixture(Path(folder))
            audit.require_preflight(a)
            atomic(a.output/"expanded-checkpoint.json",dict(changed=True))
            with self.assertRaises(ValueError):audit.require_preflight(a)
        with tempfile.TemporaryDirectory() as folder:
            a=self.preflight_fixture(Path(folder))
            proof=read(a.preflight_output);proof["status"]="failed";atomic(a.preflight_output,proof)
            with self.assertRaises(ValueError):audit.require_preflight(a)
        with tempfile.TemporaryDirectory() as folder:
            a=self.preflight_fixture(Path(folder))
            inside=a.output/"preflight.json";inside.write_bytes(a.preflight_output.read_bytes());a.preflight_output=inside
            with self.assertRaises(ValueError):audit.require_preflight(a)

    def test_queue_expiration_retry_is_distinct_from_partial_native_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            a=SimpleNamespace(output=Path(folder))
            atomic(a.output/"report.json",dict(version=study.VERSION,status="running",phase="waiting_for_4090",gpu_wait_s=30.))
            study.record_failure(a,"run",True,TimeoutError("busy"))
            report=read(a.output/"report.json")
            self.assertEqual((report["status"],report["phase"]),("running","waiting_for_4090"))
            self.assertTrue(report["queue_expired"])
            atomic(a.output/"native-started.json",{})
            study.record_failure(a,"run",True,TimeoutError("native deadline"))
            self.assertEqual(read(a.output/"report.json")["phase"],"run_failed")

    def test_history_parser_covers_failed_packets_and_bare_splits(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            path=root/"rows.jsonl"
            path.write_text(json.dumps(dict(bucket_xy_m=[.8,-.4],success=False))+"\n"+json.dumps(dict(request=dict(bucket_xy_m=[.9,-.3])))+"\n")
            self.assertEqual(audit.history_coordinates(path),{(.8,-.4),(.9,-.3)})
            atomic(root/"splits.json",dict(train=[[.1,.2]],future_splits=dict(test=[[.3,.4]])))
            self.assertEqual(audit.history_coordinates(root/"splits.json"),{(.1,.2),(.3,.4)})

    def test_partial_preparation_is_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)
            atomic(output/"contract.json",{})
            atomic(output/"layouts.json",{})
            study.start_preparation(output,dict(frozen_inputs={}))
            original=(output/"preparation-started.json").read_bytes()
            with self.assertRaises(ValueError):study.start_preparation(output,dict(frozen_inputs={}))
            self.assertEqual((output/"preparation-started.json").read_bytes(),original)


if __name__ == "__main__":unittest.main()
