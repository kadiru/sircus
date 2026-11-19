"""CPU target/admission mocks; no native dynamics, robot or GPU is used."""

import copy
import unittest

import numpy as np

from sircus.mjlab.impact650_geometry import ImpactGeometry,impact_targets
from sircus.mjlab.learning_contract import digest
from sircus.mjlab.learning_ik import shot_targets
from sircus.mjlab.tests.test_clean650_geometry import Solver,Gate,scene,seed


def config():
    result=scene();result['board']['size_m']=[.2,.3,.005];return result


def request(ident='impact',offset=(.02,.01)):
    r=seed();r.update(id=ident,role='discovery',phase='coarse',impact_offset_m=list(offset));return r


class ImpactGeometryTests(unittest.TestCase):
    def instance(self,*,mode='accept',valid=True,clearance=.006):
        ik=Solver(valid);gate=Gate(mode,clearance)
        return ImpactGeometry(config(),'unused-assets',ik=ik,gate=gate),ik,gate

    def test_surface_basis_gap_release_and_real_dropper_target(self):
        cfg=config();values=np.array([[.5,-.3,.38,30.,.41]]);bucket=np.array([[1.15,-.3]]);offset=np.array([[.04,.02]])
        old=shot_targets(cfg,bucket,values[:,:3],values[:,3],values[:,4]);out=impact_targets(cfg,bucket,values,offset)
        expected=np.array([.5+.04*np.cos(np.pi/6),-.28,.36])
        np.testing.assert_allclose(out['impact_point_world_m'][0],expected,atol=1e-15)
        np.testing.assert_allclose(out['release_center_m'][0],expected+[0,0,.4475])
        np.testing.assert_allclose(out['dropper_flange_world'][0,:3,3],out['release_center_m'][0]+[0,0,.095])
        np.testing.assert_array_equal(out['board_world'],old['board_world'])
        np.testing.assert_array_equal(out['board_flange_local'],old['board_flange_local'])
        np.testing.assert_allclose(np.cross(out['impact_forward_world'],out['impact_lateral_world']),out['board_normal_world'],atol=1e-15)
        np.testing.assert_allclose((out['impact_forward_world']*out['board_normal_world']).sum(1),0,atol=1e-15)
        self.assertAlmostEqual(out['impact_to_bucket_horizontal_m'][0],np.linalg.norm(bucket[0]-expected[:2]))
        self.assertAlmostEqual(out['release_center_height_m'][0],.8075)

    def test_zero_offset_reproduces_every_historical_target_exactly(self):
        cfg=config();r=request(offset=(0,0));values=np.array([r['values']]);bucket=np.array([r['bucket_xy_m']])
        old=shot_targets(cfg,bucket,values[:,:3],values[:,3],values[:,4])
        new=impact_targets(cfg,bucket,values,np.zeros((1,2)))
        for key in old:np.testing.assert_array_equal(new[key],old[key],err_msg=key)

    def test_zero_controls_and_frozen_verification_preserve_exact_branches(self):
        g,ik,gate=self.instance();zero=request('zero',(0,0));control=request('control',(0,0));control['role']='positive_control'
        verify=request('verify');verify.update(freeze_joints=True,q10=[3.]*6,q14=[4.]*6)
        rows=g.check_many([zero,control,verify])
        self.assertFalse(ik.calls);self.assertEqual(len(gate.calls),3)
        for row,source in zip(rows,[zero,control,verify]):
            self.assertTrue(row['eligible']);self.assertEqual(row['q10'],source['q10']);self.assertEqual(row['q14'],source['q14'])
            self.assertIn('release_center_m',row)

    def test_nonzero_offset_solves_only_dropper_and_never_changes_board(self):
        g,ik,gate=self.instance();r=request();before=copy.deepcopy(r);out=g.check_many([r])[0]
        self.assertNotEqual(out['q10'],r['q10']);self.assertEqual(out['q14'],r['q14']);self.assertEqual(r,before)
        self.assertEqual(ik.calls[0]['starts'],16);self.assertEqual(ik.calls[0]['iterations'],180)
        self.assertFalse(ik.calls[0]['preserve_known'].any())
        np.testing.assert_array_equal(gate.calls[0]['q14'],r['q14'])
        np.testing.assert_array_equal(ik.calls[0]['targets'][0],gate.calls[0]['targets']['dropper_flange_world'])
        self.assertEqual(out['geometry_sha256'],digest(g.proofs[0]))

    def test_shared_math_never_reuses_pair_clearance_and_hint_is_in_cache_key(self):
        g,ik,gate=self.instance(mode='second_reject');a=request('accepted');b=request('rejected');b['q14']=[4.]*6
        out=g.check_many([a,b]);self.assertEqual([r['eligible'] for r in out],[True,False])
        self.assertEqual(len(ik.calls),1);self.assertEqual(len(ik.calls[0]['targets']),1)
        self.assertEqual(len(g.proofs[1]['attempts']),3);self.assertEqual(out[1]['q14'],b['q14'])
        c=request('new-hint');c['q10']=[2.]*6;g.check_many([c])
        self.assertEqual(len(ik.calls),2)

    def test_independent_board_failure_stops_futile_dropper_branch_checks(self):
        g,ik,gate=self.instance(mode='independent14');r=request();r['q14']=[0.]*6
        self.assertFalse(g.check_many([r])[0]['eligible']);self.assertEqual(len(gate.calls),1)

    def test_nominal_board_footprint_bounds_bucket_and_control_offset_are_checked(self):
        g,ik,gate=self.instance();outside=request('outside',(.2,0));bounds=request('bounds');bounds['values'][2]=.119
        table=request('table');table['bucket_xy_m']=[1.395001,-.3]
        close=request('close');close['bucket_xy_m']=[.595,-.27]
        control=request('control');control['role']='negative_control'
        rows=g.check_many([outside,bounds,table,close,control])
        self.assertFalse(any(r['eligible'] for r in rows));self.assertFalse(ik.calls);self.assertFalse(gate.calls)
        self.assertEqual([p['reason'] for p in g.proofs],['impact_point_outside_board','action_bounds_or_invalid','bucket_table_footprint','board_bucket_distance','control_impact_offset'])

    def test_failed_ik_is_retained_and_no_spherical_prior_can_veto_a_query(self):
        g,ik,gate=self.instance(valid=False);r=request();out=g.check_many([r])[0]
        self.assertFalse(out['eligible']);self.assertEqual(out['id'],r['id']);self.assertEqual(out['q14'],r['q14'])
        self.assertEqual(g.proofs[0]['reason'],'ik_unresolved');self.assertFalse(gate.calls)
        g,ik,gate=self.instance();r=request();r['values'][2:]=[.58,24.,.6]
        self.assertTrue(g.check_many([r])[0]['eligible']);self.assertTrue(ik.calls)

    def test_false_success_below5mm_and_invalid_arrays_fail_closed(self):
        g,_,_=self.instance(clearance=.0049)
        with self.assertRaises(RuntimeError):g.check_many([request()])
        cfg=config();r=request()
        for offsets in ([[float('nan'),0]],[[0,0,0]]):
            with self.assertRaises(ValueError):impact_targets(cfg,[r['bucket_xy_m']],[r['values']],offsets)


if __name__=='__main__':unittest.main()
