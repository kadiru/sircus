"""CPU convex-collider certificates, including cases where AABBs mislead."""
import itertools
from types import SimpleNamespace
import unittest
import numpy as np
from sircus.mjlab import fixed35_six_geometry as G


def box(center=(0,0,0), half=(1,1,1), rotation=None):
    rotation=np.eye(3) if rotation is None else np.asarray(rotation)
    points=np.asarray(list(itertools.product((-1.,1.),repeat=3)))*half
    return G.polyhedron(points@rotation.T+center)


class GeometryTests(unittest.TestCase):
    def test_true_overlap_has_independent_common_interior(self):
        a,b=box(),box((1.5,0,0))
        proof=G.overlap_certificate(a,b)
        self.assertTrue(proof['verified_overlap'])
        self.assertAlmostEqual(proof['common_interior_radius_m'],.25-G.TOL)
        self.assertGreaterEqual(min(G.interior_slack(s,proof['witness_world_m']) for s in (a,b)),.25-1e-10)
        self.assertEqual(G.separation_bound(a,b)['lower_bound_m'],0.)

    def test_touch_is_not_positive_overlap_or_clearance(self):
        a,b=box(),box((2,0,0))
        self.assertFalse(G.overlap_certificate(a,b)['verified_overlap'])
        self.assertEqual(G.separation_bound(a,b)['lower_bound_m'],0.)

    def test_overlap_of_aabbs_does_not_imply_mesh_overlap(self):
        c=np.sqrt(.5);rot=np.array([[c,-c,0],[c,c,0],[0,0,1.]])
        a=box(half=(1,.005,.005),rotation=rot)
        b=box(np.array([-c,c,0])*.03,half=(1,.005,.005),rotation=rot)
        self.assertTrue(np.all(np.minimum(a['points'].max(0),b['points'].max(0)) > np.maximum(a['points'].min(0),b['points'].min(0))))
        self.assertFalse(G.overlap_certificate(a,b)['verified_overlap'])
        self.assertAlmostEqual(G.separation_bound(a,b)['lower_bound_m'],.02,places=10)

    def test_cylinder_not_replaced_by_enclosing_box(self):
        a=G.cylinder([0,0,0],np.eye(3),1.,1.)
        b=box((.95,.95,0),(.04,.04,.1))
        self.assertFalse(G.overlap_certificate(a,b)['verified_overlap'])
        self.assertGreater(G.separation_bound(a,b)['lower_bound_m'],.28)
        self.assertAlmostEqual(G.support(a,np.array([[1,1,0]]))[0],np.sqrt(2))

    def test_cylinder_interior_witness_verified_against_exact_shape(self):
        a=G.cylinder([0,0,0],np.eye(3),.1,.004)
        b=box((.09,0,0),(.02,.02,.01))
        p=G.overlap_certificate(a,b)
        self.assertTrue(p['verified_overlap'])
        self.assertGreater(p['common_interior_radius_m'],.0039)
        self.assertGreaterEqual(G.interior_slack(a,p['witness_world_m']),p['common_interior_radius_m'])
        self.assertLess(a['inscribed_radial_error_m'],.00006)

    def test_rotated_translated_cylinder_support_and_certificate(self):
        rot=np.array([[0.,0,1],[0,1,0],[-1,0,0]])
        a=G.cylinder([2,3,4],rot,.2,.1)
        self.assertAlmostEqual(G.support(a,np.array([[1,0,0]]))[0],2.1)
        p=G.overlap_certificate(a,box((2,3,4),(.01,.01,.01)))
        self.assertTrue(p['verified_overlap'])
        self.assertAlmostEqual(p['common_interior_radius_m'],.01-G.TOL)

    def test_compiled_mesh_uses_geom_transform_not_raw_vertices(self):
        v=np.asarray(list(itertools.product((-1.,1.),repeat=3)))
        m=SimpleNamespace(geom_type=[7],geom_dataid=[0],mesh_vertadr=[0],mesh_vertnum=[8],mesh_vert=v)
        d=SimpleNamespace(geom_xmat=np.eye(3).reshape(1,9),geom_xpos=np.array([[5.,4.,3.]]))
        shape=G.compiled_shape(m,d,0)
        np.testing.assert_allclose(shape['points'],v+[5,4,3])
        self.assertAlmostEqual(G.interior_slack(shape,[5,4,3]),1.)

    def test_bad_native_witness_cannot_be_reported_valid(self):
        class MJ:
            @staticmethod
            def mj_geomDistance(m,d,i,j,limit,witness):
                witness[:]=[0,0,0,20,0,0]
                return -.01
        r=G.signed_query(MJ,None,None,0,1,box(),box())
        self.assertFalse(r['validated_signed_query'])

    def test_unsupported_geometries_and_nonrigid_cylinder_fail_closed(self):
        with self.assertRaises(ValueError):G.cylinder([0,0,0],np.eye(3)*2,1,1)
        m=SimpleNamespace(geom_type=[2]);d=SimpleNamespace(geom_xmat=np.eye(3).reshape(1,9))
        with self.assertRaises(ValueError):G.compiled_shape(m,d,0)

    def test_fixed_base_ancestry_includes_all_parents(self):
        m=SimpleNamespace(body_dofnum=[0,0,0,1,0],body_parentid=[0,0,1,2,3])
        self.assertEqual(G.fixed_ancestry(m,2),(True,[2,1]))
        self.assertFalse(G.fixed_ancestry(m,4)[0])


if __name__=='__main__':unittest.main()
