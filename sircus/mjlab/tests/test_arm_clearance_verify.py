import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from sircus.mjlab import arm_clearance_verify as audit
def requests():
    rows = []
    for role in ("positive_control", "negative_control"):
        rows.extend(dict(id=f"pre-{role}-{i}", role=role, phase="controls_pre", method="control") for i in range(3))
    for method in audit.METHODS:
        rows.extend(dict(id=f"{method}-{i}", role="test", phase="test", method=method,
                         context_id=f"context-{i}", bucket_xy_m=[i*.02, -.2]) for i in range(50))
    for role in ("positive_control", "negative_control"):
        rows.extend(dict(id=f"post-{role}-{i}", role=role, phase="controls_post", method="control") for i in range(3))
    return rows


class ClearanceFrozenProtocolChecks(unittest.TestCase):
    def test_seeded_layouts_exclude_old_test_and_all_ancestors(self):
        old = dict(excluded=[[.1, -.5], [.7, .2]], splits=dict(
            validation=[dict(bucket_xy_m=[.2, .1])], test=[dict(bucket_xy_m=[.8, -.3])]))
        result = audit.expected_splits(old)
        self.assertEqual(result, audit.expected_splits(old))
        self.assertEqual(len(result["splits"]["test"]), 50)
        xy = np.asarray([r["bucket_xy_m"] for r in result["splits"]["test"]])
        excluded = np.asarray(result["excluded"])
        self.assertEqual(len(excluded), 4)
        self.assertGreaterEqual(np.linalg.norm(xy[:, None]-excluded[None], axis=2).min(), .01)
        distance = np.linalg.norm(xy[:, None]-xy[None], axis=2)
        distance += np.eye(50)*100
        self.assertGreaterEqual(distance.min(), .01)
        self.assertTrue(((xy >= [.1, -.65]) & (xy <= [1.24, .25])).all())

    def test_complete_paired_budget_passes(self):
        audit.budget_checks(requests(), audit.AuditChecks())

    def test_same_length_replaced_context_is_rejected(self):
        rows = requests()
        next(r for r in rows if r.get("method") == "consistent")["bucket_xy_m"] = [1.2, .2]
        with self.assertRaises(audit.EvidenceError):
            audit.budget_checks(rows, audit.AuditChecks())

    def test_missing_failed_request_cannot_disappear_from_denominator(self):
        with self.assertRaises(audit.EvidenceError):
            audit.budget_checks(requests()[:-1], audit.AuditChecks())

    def test_repeated_identity_is_rejected(self):
        rows = requests(); rows[-1]["id"] = rows[-2]["id"]
        with self.assertRaises(audit.EvidenceError):
            audit.budget_checks(rows, audit.AuditChecks())

    def test_model_freeze_must_precede_acquisition(self):
        packet = dict(version=audit.STUDY_VERSION, contract_fingerprint="fp", requests=requests(),
                      frozen_before_any_native_outcome=True, native_planning_queries=0, frozen_unix_s=100.)
        audit.freeze_checks(packet, dict(acquired_unix_s=101.), dict(fingerprint="fp"), audit.AuditChecks())
        for bad in (dict(acquired_unix_s=99.), dict(acquired_unix_s=True), dict(acquired_unix_s=float("nan"))):
            with self.assertRaises(audit.EvidenceError):
                audit.freeze_checks(packet, bad, dict(fingerprint="fp"), audit.AuditChecks())

    def test_boolean_zero_is_not_integer_native_query_count(self):
        packet = dict(version=audit.STUDY_VERSION, contract_fingerprint="fp", requests=requests(),
                      frozen_before_any_native_outcome=True, native_planning_queries=False, frozen_unix_s=100.)
        with self.assertRaises(audit.EvidenceError):
            audit.freeze_checks(packet, dict(acquired_unix_s=101.), dict(fingerprint="fp"), audit.AuditChecks())



class ClearanceSelectionChecks(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.items=[dict(id=f'c{i}',values=[.5,-.2,.3,30.,.2]) for i in range(16)]
        self.order=list(range(16))
    def selection(self,endpoint,clear=None):
        records=[];first_geometry=None;chosen=None
        for i,flag in enumerate(endpoint):
            proof=dict(passed=flag,reason=None if flag else 'setup_reject',q10=[0.]*6,q14=[0.]*6)
            path=self.root/f'proof{i}.json';path.write_text(json.dumps(proof))
            r=dict(candidate_index=i,candidate_id=f'c{i}',passed=flag,reason=proof['reason'],
                   connection_file=path.name,connection_sha256=audit.sha(path))
            if flag:
                if first_geometry is None:first_geometry=i
                if clear is not None:r['clearance']=dict(passed=clear[i],minimum_clearance_lower_bound_m=.003 if clear[i] else -.01)
                if clear is None or clear[i]:chosen=i
            records.append(r)
        abstain=clear is not None and chosen is None and first_geometry is not None
        return dict(checks=records,selected_index=chosen if chosen is not None else first_geometry if abstain else 0,
                    all_geometry_checks_rejected=first_geometry is None,policy_abstain=abstain)
    def check(self,selection,clear=None):
        class Gate:
            def check(self,q10,q14,bucket,values,diagnostic,gravity=None):
                flag=clear[diagnostic];return dict(passed=flag,minimum_clearance_lower_bound_m=.003 if flag else -.01)
        with patch('sircus.mjlab.calibrated_admission.verify_connection',return_value=True):
            return audit.selection_check(selection,self.order,self.items,self.root,{}, {},audit.AuditChecks(),'fixture',
                       method='arm_clearance' if clear is not None else 'consistent',gate=Gate(),
                       diagnostics=list(range(16)),bucket=[.6,.1],gravity=[0,0,-9.81])
    def test_original_first_admitted_and_no_guard(self):
        self.assertEqual(self.check(self.selection([False,True])),dict(index=1,admitted=True,geometry_checks=2,policy_abstain=False,geometry_admitted=True))
    def test_guard_skips_blocked_endpoint_then_passes(self):
        clear=[False,True];result=self.check(self.selection([True,True],clear),clear)
        self.assertEqual((result['index'],result['admitted'],result['policy_abstain']),(1,True,False))
    def test_four_guard_failures_abstain_without_unsafe_fallback(self):
        clear=[False]*4;result=self.check(self.selection([True]*4,clear),clear)
        self.assertTrue(result['policy_abstain']);self.assertFalse(result['admitted']);self.assertEqual(result['index'],0)
    def test_four_endpoint_failures_are_not_policy_abstention(self):
        result=self.check(self.selection([False]*4,[False]*4),[False]*4)
        self.assertFalse(result['policy_abstain']);self.assertFalse(result['geometry_admitted'])
    def test_cannot_stop_after_failing_one_guard(self):
        with self.assertRaises(audit.EvidenceError):self.check(self.selection([True],[False]),[False])
    def test_cannot_skip_first_guard_success(self):
        with self.assertRaises(audit.EvidenceError):self.check(self.selection([True,True],[True,True]),[True,True])
    def test_tampered_guard_or_missing_abstention_refused(self):
        clear=[False]*4;sel=self.selection([True]*4,clear);sel['checks'][0]['clearance']['passed']=True
        with self.assertRaises(audit.EvidenceError):self.check(sel,clear)
        sel=self.selection([True]*4,clear);sel['policy_abstain']=False
        with self.assertRaises(audit.EvidenceError):self.check(sel,clear)
    def test_identical_candidate_bank_required(self):
        original=[dict(id=f'c{i}',values=[.5,-.2,.3,30.,.2]) for i in range(16)]
        banks={m:dict(candidates=copy.deepcopy(original)) for m in audit.METHODS}
        audit.candidate_banks_check(banks,original,{},audit.AuditChecks(),'fixture')
        banks['arm_clearance']['candidates'][8]['values'][4]+=.06
        with self.assertRaises(audit.EvidenceError):audit.candidate_banks_check(banks,original,{},audit.AuditChecks(),'changed')

class ClearanceBoundArithmeticChecks(unittest.TestCase):
    def test_scalar_certificate_does_not_call_production_path_math(self):
        from sircus.mjlab import arm_clearance_selector as selector
        planes=np.asarray([[1,0,0,-.1],[-1,0,0,-.1],[0,1,0,-.1],
                           [0,-1,0,-.1],[0,0,1,-.1],[0,0,-1,-.1]],dtype=float)
        class Gate:
            radius=.04
            def colliders(self,*args):return [dict(name='box',planes=planes)]
        # x remains1m; all other axes inside the box, so plane bound is exactly.9m.
        h=.003; n=2; speed=.01; bound=.9-.04-speed*h/(2*n)-1e-10
        proof=dict(passed=True,minimum_clearance_lower_bound_m=bound,
            closest=dict(geom='box',time_s=0.,center_m=[1.,0.,0.],surface_clearance_lower_bound_m=bound),
            horizon_s=h,samples=3,actual_spacing_s=h/n,maximum_speed_m_s=speed,
            between_sample_allowance_m=speed*h/(2*n),required_margin_m=.002,collision_geometry_count=1)
        members=[dict(member=i,rim_band_start_s=h,origin_m=[1.,0.,0.],velocity_world_m_s=[0.,speed,0.]) for i in range(5)]
        saved=dict(passed=True,minimum_clearance_lower_bound_m=bound,members=[dict(proof,member=i) for i in range(5)])
        with patch.object(selector,'path_bound',side_effect=AssertionError('must be independent')):
            audit.independent_clearance_arithmetic(Gate(),[],[],[],dict(members=members),[0.,0.,0.],saved,audit.AuditChecks(),'fixture')
            for field,value in [('samples',4),('minimum_clearance_lower_bound_m',bound+.01),('passed',False)]:
                bad=copy.deepcopy(saved);bad['members'][0][field]=value
                with self.assertRaises(audit.EvidenceError):
                    audit.independent_clearance_arithmetic(Gate(),[],[],[],dict(members=members),[0.,0.,0.],bad,audit.AuditChecks(),'tampered')
            bad=copy.deepcopy(saved);bad['members'][0]['closest']['center_m'][0]=2.
            with self.assertRaises(audit.EvidenceError):
                audit.independent_clearance_arithmetic(Gate(),[],[],[],dict(members=members),[0.,0.,0.],bad,audit.AuditChecks(),'bad witness')


class ClearanceContactEvidenceChecks(unittest.TestCase):
    def fixture(self,root):
        (root/'native').mkdir()
        force=np.zeros((6,1,6));force[2,0,4]=.02;force[4,0,4]=.01;force[5,0,4]=.03
        cache=root/'native'/'trace.npz'
        np.savez(cache,record_ids=np.asarray(['native']),position_m=np.zeros((7,1,3)),
                 velocity_m_s=np.zeros((7,1,3)),forces_N=force,time_s=np.arange(7)*.001)
        rows=[dict(id='native',native_release=True,native_cache_file=cache.name,native_cache_row=0,
                   native_cache_sha256=audit.sha(cache),bucket_xy_m=[.7,-.2],board_normal=[0,0,1],
                   invalid_physics=False,physics_steps=4,route_success=False,route_quality=0),
              dict(id='reject',native_release=False,invalid_physics=None)]
        positions=[dict(record_id=row['id'],context_id='c',method=method,native_release=row['native_release'],
                        invalid_physics=row['invalid_physics'],clearance_rejected=method=='arm_clearance',
                        retention_s=0. if method=='consistent' else None,whole_ball_entry=False if method=='consistent' else None)
                   for row,method in zip(rows,audit.METHODS)]
        selections={method:dict(selected_index=0,checks=[dict(candidate_index=0,**(dict(clearance=dict(
            passed=False,minimum_clearance_lower_bound_m=-.01)) if method=='arm_clearance' else {}))]) for method in audit.METHODS}
        (root/'rankings.json').write_text(json.dumps([dict(context_id='c',selection=selections)]))
        metric=dict(available=True,start_index=1,start_time_s=.001,end_time_s=.002,
                    cutoff_reason='contact',cutoff_contact_channels=['robot'])
        return rows,positions,metric
    def test_raw_contacts_and_abstentions_remain_distinct_and_preserve_raw_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rows,positions,metric=self.fixture(root)
            with patch('sircus.mjlab.xy650_verify.first_flight',return_value=metric):
                enriched,summary=audit.contact_results(root,rows,positions,dict(scene={}),{},audit.AuditChecks())
            for original,derived in zip(positions,enriched):
                for key,value in original.items():self.assertEqual(derived[key],value)
            self.assertEqual(enriched[0]['contact_observation_status'],'observed')
            self.assertTrue(enriched[0]['robot_contact_before_terminal'])
            self.assertTrue(enriched[0]['robot_contact_after_terminal'])
            self.assertTrue(enriched[0]['robot_contact_post_rebound'])
            self.assertEqual(enriched[0]['first_post_bounce_contact_channels'],['robot'])
            self.assertEqual(enriched[0]['first_contact_delay_ms'],1.)
            self.assertIsNone(enriched[1]['robot_contact_before_terminal'])
            self.assertIsNone(enriched[1]['retention_s'])
            self.assertFalse(enriched[1]['invalid'])
            self.assertEqual(summary[0]['force_threshold_N'],.01)
            self.assertEqual(summary[0]['robot_contact_trials_before_terminal'],1)
            self.assertEqual(summary[1]['observed_native'],0)
            self.assertEqual(summary[1]['native_releases'],0)
            self.assertEqual(summary[1]['clearance_rejections'],1)
    def test_wrong_cache_identity_or_hash_refuses_contact_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rows,positions,metric=self.fixture(root)
            for key,value in [('id','wrong'),('native_cache_sha256','tampered')]:
                bad=copy.deepcopy(rows);bad[0][key]=value
                with self.assertRaises(audit.EvidenceError):
                    audit.contact_results(root,bad,positions,dict(scene={}),{},audit.AuditChecks())


if __name__=='__main__':unittest.main()
