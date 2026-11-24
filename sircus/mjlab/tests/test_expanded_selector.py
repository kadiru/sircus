"""Deduplicated targets, leakage boundaries and one-drop accounting."""
from copy import deepcopy
import importlib.util
import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch

from sircus.mjlab import expanded_selector_model as model
from sircus.mjlab.expanded_selector_study import summarize, METHODS, GPU_CAP_S
from sircus.mjlab.neural_shot_model import _weights


def record(identity, xy, capture=.5, split="train"):
    return dict(dedup_key=identity, split=split, setup_probability=1., capture_probability=capture,
        valid_native_count=2 if capture is not None else 0, repeats_count=2,
        request=dict(id=identity, bucket_xy_m=list(xy), values=[.55,-.2,.32,28.,.35],
                     intended_route="direct",board_heading_offset_deg=0.,board_twist_deg=0.))


class ExpandedSelectorTests(unittest.TestCase):
    def test_preparation_cannot_silently_repeat_partial_work(self):
        from sircus.mjlab.expanded_selector_study import start_preparation
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/"contract.json").write_text("{}")
            (root/"layouts.json").write_text("[]")
            (root/"native-current").mkdir()
            with self.assertRaises(ValueError):start_preparation(root,dict(frozen_inputs={}))
            (root/"native-current").rmdir()
            receipt=start_preparation(root,dict(frozen_inputs={}))
            self.assertEqual(json.loads((root/"preparation-started.json").read_text()),receipt)
            with self.assertRaises(ValueError):start_preparation(root,dict(frozen_inputs={}))

    def test_native_initial_column_mismatch_is_rejected(self):
        import numpy as np
        from sircus.mjlab.expanded_selector_audit import verify_initial_trace
        from sircus.mjlab.benchmark_verify import Checks
        row=dict(id="shot",native_initial_ball_center_m=[.5,0.,.7],release_ball_center_m=[.5,0.,.7])
        arrays=dict(position_m=np.array([[[.5,0.,.7]]]),velocity_m_s=np.zeros((1,1,3)))
        verify_initial_trace(row,arrays,0,Checks())
        arrays["position_m"][0,0,0]=.51
        with self.assertRaises(Exception):verify_initial_trace(row,arrays,0,Checks())
        arrays["position_m"][0,0,0]=.5
        arrays["velocity_m_s"][0,0,0]=.1
        with self.assertRaises(Exception):verify_initial_trace(row,arrays,0,Checks())

    def test_fractional_repeats_are_one_target_not_extra_rows(self):
        rows = [record("a",(.6,-.4),.75),record("b",(.6,-.4),0.),record("c",(.7,-.4),1.)]
        encoded = model.encoded(rows,"train")
        self.assertEqual(encoded[0]["labels"],[1.,.75])
        self.assertEqual(_weights(encoded,1),[.25,.25,.5])
        rows[0]["repeats_count"] = 200
        self.assertEqual(model.encoded(rows,"train"),encoded)

    def test_unknown_flight_has_no_capture_weight(self):
        row=record("a",(.6,-.4),None)
        row["setup_probability"]=0.
        encoded=model.encoded([row],"train")
        self.assertEqual(encoded[0]["masks"],[1.,0.])
        self.assertEqual(_weights(encoded,1),[0.])

    def test_bad_targets_and_duplicate_settings_fail(self):
        row=record("a",(.6,-.4))
        with self.assertRaises(ValueError):model.encoded([row,deepcopy(row)],"train")
        for target in (-.1,1.1,float('nan'),True):
            bad=deepcopy(row);bad["capture_probability"]=target
            with self.assertRaises(ValueError):model.encoded([bad],"train")
        with self.assertRaises(ValueError):model.encoded([dict(row,split="test")],"train")
        with self.assertRaises(ValueError):model.encoded([dict(row,valid_native_count=0)],"train")

    def test_native_query_joints_do_not_feed_features(self):
        row=record("a",(.6,-.4))
        before=model.encoded([row],"train")
        row["request"].update(q10=[1]*6,q14=[2]*6,native_success=True)
        self.assertEqual(model.encoded([row],"train"),before)

    def test_menu_preserves_original_pre_ik_hints_and_route_provenance(self):
        try:
            from sircus.mjlab.expanded_selector_plan import expanded_menu
            from sircus.mjlab import neural_shot_plan
        except ModuleNotFoundError as exc:
            self.skipTest(str(exc))
        source = dict(id="recipe", dedup_key="recipe-key", split="train", values=[.55,-.2,.32,28.,.35],
            bucket_xy_m=[.8,-.2], q10=[9.]*6,q14=[8.]*6, seed_q10=[1.]*6,seed_q14=[2.]*6,
            intended_route="direct", observed_routes=["robot"], board_heading_offset_deg=7.,board_twist_deg=135.,
            capture_probability=1.)
        layout=dict(id="fresh",bucket_xy_m=[.85,-.2],split="test")
        with patch.object(neural_shot_plan,"menu",return_value=[]):
            menu=expanded_menu(layout,dict(baseline_libraries={},expanded_sources=[source]),{})
        self.assertEqual(len(menu),3)
        self.assertEqual([round(r["values"][0],4) for r in menu],[.6,.55,.575])
        for proposal in menu:
            self.assertEqual(proposal["seed_q10"],[1.]*6)
            self.assertEqual(proposal["source_observed_route"],["robot"])
            self.assertEqual(proposal["board_heading_offset_deg"],7.)
            self.assertNotIn("q10",proposal)
        source["split"]="validation"
        with patch.object(neural_shot_plan,"menu",return_value=[]):
            with self.assertRaises(ValueError):expanded_menu(layout,dict(baseline_libraries={},expanded_sources=[source]),{})

    @unittest.skipUnless(importlib.util.find_spec("torch"),"Torch is only required in pinned training runtime")
    def test_training_freezes_position_splits(self):
        train=[record("a",(.6,-.4),1.),record("b",(.6,-.4),0.)]
        val=[record("c",(.7,-.4),1.,"validation"),record("d",(.7,-.4),0.,"validation")]
        fit=model.fit(train,val,epochs=(1,2))
        self.assertEqual(fit["checkpoint"]["training_rows"],2)
        val[0]["request"]["bucket_xy_m"]=[.6,-.4]
        with self.assertRaises(ValueError):model.fit(train,val,epochs=(1,2))

    def test_one_drop_accounting_includes_setup_refusals(self):
        layouts=[dict(id="a",bucket_xy_m=[.6,-.4]),dict(id="b",bucket_xy_m=[.7,-.4])]
        slots=[];rows=[];plans=[]
        for layout in layouts:
            for method in METHODS:
                identity=layout["id"]+"-"+method
                admitted=not(layout["id"]=="b" and method=="current")
                slots.append(dict(id=identity,context_id=layout["id"],method=method,request={} if admitted else None))
                plans.append(dict(variant=method,site=dict(decision_latency_s=10.,request_wall_s=11.)))
                if admitted:rows.append(dict(id=identity,adapter_directory="native-"+method,board_first_scalar_audit_passed=True,
                    board_first_score=dict(route_success=layout["id"]=="a",invalid_physics=False)))
        result=summarize(layouts,slots,rows,plans)
        self.assertEqual(result["methods"]["current"]["captures"],1)
        self.assertEqual(result["methods"]["current"]["setup_rejections"],1)
        self.assertEqual(result["methods"]["expanded"]["flight_misses"],1)
        self.assertEqual(result["methods"]["expanded"]["mean_request_s"],11.)
        self.assertEqual(result["paired"],dict(both=1,current_only=0,expanded_only=0,neither=1))
        with self.assertRaises(ValueError):summarize(layouts,slots,rows+rows[:1],plans)
        self.assertIs(type(GPU_CAP_S),int)


if __name__ == "__main__":unittest.main()
