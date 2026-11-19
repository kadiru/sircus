import unittest

import numpy as np

from sircus.mjlab.corrected_contract import (PHYSICS,grid_layouts,split_coordinates,
    geometry_hint,diverse_selection,qualification,reject_row,validate_packages,PINNED_PACKAGES)


class ContractTests(unittest.TestCase):
    def test_35_unique_grid_coordinates_and_full_extents(self):
        rows=grid_layouts();xy=np.asarray([r["bucket_xy_m"] for r in rows])
        self.assertEqual(len(rows),35);self.assertEqual(len({r["id"] for r in rows}),35)
        self.assertEqual(len(set(map(tuple,xy))),35)
        np.testing.assert_allclose(xy.min(0),[.1,-.65]);np.testing.assert_allclose(xy.max(0),[1.24,.25])
        self.assertEqual({r["role"] for r in rows},{"development_then_train"})

    def test_future_splits_frozen_disjoint_and_separated(self):
        excluded=[r["bucket_xy_m"] for r in grid_layouts()]+[[.5,-.5]]
        a=split_coordinates(excluded);self.assertEqual(a,split_coordinates(excluded))
        self.assertEqual([len(a[k]) for k in ("train","validation","test")],[35,30,100])
        all_points=np.asarray(excluded+a["validation"]+a["test"])
        dist=np.linalg.norm(all_points[:,None]-all_points[None,:],axis=2)
        np.fill_diagonal(dist,np.inf);self.assertGreaterEqual(float(dist.min()),.005)

    def test_geometry_hint_discards_prior_success_and_reward(self):
        row=dict(id="source",values=[.5,-.2,.4,30,.3],q10=[0]*6,q14=[1]*6,
            bucket_xy_m=[.2,-.6],success=True,route_success=True,reward=100,route_quality=2)
        hint=geometry_hint(row,"history")
        for key in ("success","route_success","route_quality","reward"):self.assertNotIn(key,hint)
        self.assertEqual(hint["source_id"],"source")
        self.assertIsNone(geometry_hint(dict(row,q10=[float("nan")]*6),"history"))

    def test_selection_uses_eligible_geometry_only(self):
        values=np.asarray([[.5,-.2,.3,30,.2],[.9,.2,.5,45,.5],[.3,-.5,.12,8,.05],[.51,-.2,.3,30,.2]])
        eligible=np.asarray([1,0,1,1],bool)
        a=diverse_selection(values,eligible,[dict(success=True)]*4,2,123)
        b=diverse_selection(values,eligible,[dict(success=False)]*4,2,123)
        self.assertEqual(a,b);self.assertNotIn(1,a);self.assertEqual(len(a),2)
        self.assertIn(2,a)

    def test_rejections_stay_in_confirmation_denominator(self):
        nominal=[dict(native_release=True,route_success=True)]*3
        fresh=[dict(native_release=True,route_success=True)]*18+[dict(native_release=False,route_success=False)]*2
        self.assertTrue(qualification(nominal,fresh,dict(passed=True)))
        self.assertFalse(qualification(nominal,fresh[:18],dict(passed=True)))
        self.assertFalse(qualification(nominal,fresh,dict(passed=False)))
        self.assertFalse(qualification(nominal,fresh[:17]+[dict(native_release=False,route_success=False)]*3,dict(passed=True)))

    def test_reject_never_creates_release_or_capture(self):
        request=dict(id="r",cell_id="cell",candidate_id="c",values=[.5,-.2,.4,30,.3],bucket_xy_m=[.2,-.6])
        row=reject_row(request,"strict_endpoint_rejected",phase="search")
        self.assertFalse(row["native_release"]);self.assertFalse(row["route_success"])
        self.assertEqual(row["values"],request["values"])

    def test_physics_binds_reference64_and_unchanged_score_clock(self):
        self.assertEqual(PHYSICS["substeps"],64);self.assertEqual(PHYSICS["gravity_mode"],"passive_only")
        self.assertEqual(PHYSICS["board_mode"],"articulated")
        self.assertEqual(PHYSICS["native_dt_s"],1/15360);self.assertEqual(PHYSICS["scoring_dt_s"],1/240)

    def test_dependency_mismatch_fails_before_native(self):
        valid=dict(PINNED_PACKAGES,torch="2.7.0+cu126")
        self.assertEqual(validate_packages(valid),valid)
        with self.assertRaises(RuntimeError):validate_packages(dict(valid,mujoco="3.12.0"))
        with self.assertRaises(RuntimeError):validate_packages({})


if __name__=="__main__":unittest.main()
