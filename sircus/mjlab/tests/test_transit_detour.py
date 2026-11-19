import unittest
import numpy as np
from sircus.mjlab.transit_detour import rrt_detour
from sircus.mjlab.transit_audit import joint_samples


class DetourTests(unittest.TestCase):
    def test_clear_direct_path_remains_direct(self):
        path,result=rrt_detour(np.zeros(6),np.ones(6)*.1,lambda a,b:{"passed":True},np.ones(6)*-2,np.ones(6)*2)
        self.assertTrue(result["passed"]);self.assertTrue(result["direct"]);self.assertEqual(len(path),2)

    def test_impenetrable_barrier_fails_with_finite_budget(self):
        path,result=rrt_detour(np.zeros(6),np.ones(6),lambda a,b:{"passed":False},np.ones(6)*-2,np.ones(6)*2,max_nodes=10)
        self.assertIsNone(path);self.assertFalse(result["passed"]);self.assertEqual(result["edge_checks"],10)

    def test_seeded_detour_avoids_blocked_straight_path(self):
        def edge(a,b):
            q=joint_samples(np.r_[a,np.zeros(6)],np.r_[b,np.zeros(6)])[:,:6]
            return {"passed":not bool(((q[:,0]>.04)&(q[:,0]<.08)&(np.abs(q[:,1])<.025)).any())}
        goal=np.zeros(6);goal[0]=.12
        args=(np.zeros(6),goal,edge,np.ones(6)*-1,np.ones(6))
        first,r=rrt_detour(*args,seed=7)
        again,_=rrt_detour(*args,seed=7)
        self.assertTrue(r["passed"]);self.assertFalse(r["direct"])
        np.testing.assert_array_equal(first,again)
        self.assertTrue(all(edge(a,b)["passed"] for a,b in zip(first,first[1:])))


if __name__=="__main__":unittest.main()
