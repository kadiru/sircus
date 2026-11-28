"""Frozen cohort, proof-source packet routing and independent release accounting."""
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from sircus.mjlab import preserved_shot_study as study
from sircus.mjlab import preserved_shot_plan as plan
from sircus.mjlab.sweep_worker import atomic
from sircus.mjlab.tests.test_preserved_shot_plan import fixture


def experiment(count=50,repeats=1,mode="reference"):
    layouts,plans,selections,slots=[],[],[],[]
    for i in range(count):
        current,expanded,timing=fixture(f"p{i}",expanded=i%2==0,source=mode)
        selection=plan.select_pair(current,expanded,timing)
        layouts.append(current["layout"])
        plans.extend(dict(variant=m,site=s) for m,s in (("current",current),("expanded",expanded)))
        selections.append(selection)
        slots.extend(plan.slots_for_pair(current,expanded,selection,repeats))
    order,methods=study.schedule(layouts)
    contract=dict(processing_order=order,method_order=methods,attempts_per_method_per_position=repeats)
    return contract,layouts,plans,selections,slots


class PreservedShotStudyTests(unittest.TestCase):
    def test_packets_use_proof_source_not_result_method(self):
        contract,_,_,_,slots=experiment()
        packets=study.native_schedule(contract,slots)
        lookup={s["id"]:s for s in slots}
        self.assertEqual(sum(len(p["request_ids"]) for p in packets),100)
        self.assertEqual(len({i for p in packets for i in p["request_ids"]}),100)
        self.assertLessEqual(len(packets),4)
        for packet in packets:
            self.assertLessEqual(len(packet["request_ids"]),32)
            for identity in packet["request_ids"]:
                slot=lookup[identity]
                self.assertEqual(packet["source_method"],slot["source_method"])
                self.assertEqual(packet["method"],slot["method"])
        self.assertEqual({p["source_method"] for p in packets if p["method"]=="protected"},{"current","expanded"})

    def test_development_repeat_order_and_no_replanning(self):
        contract,_,_,_,slots=experiment(9,3,"direct_first")
        packets=study.native_schedule(contract,slots)
        self.assertEqual(sum(len(p["request_ids"]) for p in packets),54)
        self.assertTrue(all(p["adapter"]=="direct_first" for p in packets))
        planned=study.planned_requests(contract,slots)
        self.assertEqual({m:len(rows) for m,rows in planned.items()},{"current":27,"protected":27})
        self.assertEqual([r["id"] for r in planned["current"]],
            [f"{i}-current-r{r:02d}" for i in contract["processing_order"] for r in range(3)])

    def test_duplicate_missing_and_mismatched_slots_fail_closed(self):
        contract,_,_,_,slots=experiment(2)
        for broken in (slots[:-1],slots+[slots[0]]):
            with self.assertRaises(ValueError):study.native_schedule(contract,broken)
        broken=deepcopy(slots);broken[0]["request"]["method"]="protected"
        with self.assertRaises(ValueError):study.native_schedule(contract,broken)

    def test_declines_keep_denominator_and_native_lanes_use_source(self):
        contract,layouts,plans,selections,slots=experiment(2)
        slots[0]["request"]=None
        rows=[]
        for slot in slots:
            if slot["request"] is not None:
                rows.append(dict(id=slot["id"],adapter_directory="native-"+slot["source_method"],
                    board_first_score=dict(route_success=slot["method"]=="protected",invalid_physics=False)))
        result=study.summarize(layouts,slots,rows,plans,selections)
        self.assertEqual(result["methods"]["current"]["setup_rejections"],1)
        self.assertEqual(result["methods"]["current"]["flight_misses"],1)
        self.assertEqual(result["methods"]["protected"]["capture_rate"],1.)
        self.assertEqual(result["methods"]["current"]["mean_request_s"],11.)
        self.assertEqual(result["methods"]["protected"]["mean_request_s"],12.)
        self.assertEqual(result["protected_fallback_positions"],1)
        self.assertFalse(result["paired_timing"]["measured_cpu_consumption"])
        with self.assertRaises(ValueError):study.summarize(layouts,slots,rows+rows[:1],plans,selections)

    def test_known_layout_ids_are_not_changed_by_development_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            layouts=[dict(id=f"original-{i}",context_id=f"original-{i}",split="test",bucket_xy_m=[i/10,0.]) for i in range(12)]
            slots=[dict(context_id=l["id"],method="expanded",request=None if i<9 else {}) for i,l in enumerate(layouts)]
            atomic(root/"layouts.json",dict(layouts=layouts))
            atomic(root/"choices.json",dict(slots=slots))
            result=study.development_layouts(SimpleNamespace(comparison_study=root))
            self.assertEqual(result["layouts"],layouts[:9])
            slots[9]["request"]=None
            atomic(root/"choices.json",dict(slots=slots))
            with self.assertRaises(ValueError):study.development_layouts(SimpleNamespace(comparison_study=root))

    def test_schedule_is_frozen_independent_of_outcomes(self):
        layouts=[dict(id=f"p{i}") for i in range(50)]
        first=study.schedule(layouts)
        second=study.schedule([dict(r,success=True) for r in layouts])
        self.assertEqual(first,second)
        self.assertEqual(sum(v[0]=="current" for v in first[1].values()),25)


if __name__=="__main__":unittest.main()
