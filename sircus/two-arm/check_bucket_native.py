"""Geometry and mount tests; no robot connection or physical actuation."""
import copy
import json
import math
from pathlib import Path
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from bucket_geometry import clearance_boxes, WALL_SECTORS
from two_arm_kinematics import board_target, board_mount_rotation
ROOT=Path(__file__).resolve().parent

def specimen():
    c=json.loads((ROOT/'scene.json').read_text())
    c['bin']={'shape':'tapered_bucket','center_xy_mm':[680,245],
              'inner_top_diameter_mm':177,'outer_top_diameter_mm':200,
              'inner_bottom_diameter_mm':150,'outer_bottom_diameter_mm':158,
              'floor_top_mm':27,'interior_depth_mm':175,'bottom_thickness_mm':8,
              'reported_height_mm':175,'height_interpretation':'unresolved'}
    return c

class BucketNativeTests(unittest.TestCase):
    def test_outer_frustum_and_inner_wall_are_enclosed(self):
        c=specimen();boxes=clearance_boxes(c);walls=boxes[1:-1]
        self.assertEqual(len(walls),WALL_SECTORS)
        for i, wall in enumerate(walls):
            rotation=np.array(wall['rotation_matrix']);center=np.array(wall['center_mm']);half=np.array(wall['size_mm'])/2
            self.assertTrue(np.allclose(rotation.T@rotation,np.eye(3)))
            self.assertAlmostEqual(np.linalg.det(rotation),1)
            theta=i*2*math.pi/WALL_SECTORS
            for z in np.linspace(0,175,9):
                inner=75+(88.5-75)*z/175;outer=79+(100-79)*(z+8)/183
                for radius in (inner,outer):
                    for delta in np.linspace(-math.pi/WALL_SECTORS,math.pi/WALL_SECTORS,9):
                        p=np.array([680+radius*math.cos(theta+delta),245+radius*math.sin(theta+delta),27+z])
                        local=rotation.T@(p-center)
                        self.assertTrue(np.all(np.abs(local)<=half+1e-8),(i,z,delta,local,half))
    def test_no_lid_and_open_center(self):
        for wall in clearance_boxes(specimen())[1:]:
            rotation=np.array(wall.get('rotation_matrix',np.eye(3)))
            point=np.array([680,245,202])-wall['center_mm']
            self.assertFalse(np.all(np.abs(rotation.T@point)<=np.array(wall['size_mm'])/2))
    def test_clearance_uses_outer_diameters(self):
        c=specimen();a=clearance_boxes(c)
        c['bin']['outer_top_diameter_mm']=220
        b=clearance_boxes(c)
        self.assertNotEqual(a[1]['center_mm'],b[1]['center_mm'])
        self.assertGreater(b[1]['size_mm'][1],a[1]['size_mm'][1])
    def test_mount_90_preserves_plane_and_tool_pose(self):
        c=specimen();c['board']['rpy_deg']=[-30,0,0];c['robot14']['board_mount_yaw_deg']=0
        oldpoint,oldcup=board_target(c);rboard=Rotation.from_euler('xyz',[-30,0,0],degrees=True).as_matrix()
        rotated=rboard@Rotation.from_euler('z',90,degrees=True).as_matrix()
        c['board']['rpy_deg']=Rotation.from_matrix(rotated).as_euler('xyz',degrees=True).tolist()
        c['robot14']['board_mount_yaw_deg']=90
        point,cup=board_target(c)
        self.assertTrue(np.allclose(point,oldpoint))
        self.assertTrue(np.allclose(cup,oldcup))
        self.assertTrue(np.allclose(cup@board_mount_rotation(c),rotated))
        self.assertTrue(np.allclose(rotated[:,2],rboard[:,2]))
    def test_offset_reconstruction_with_mount(self):
        c=specimen();c['robot14']['board_mount_yaw_deg']=90;c['board']['support_offset_mm']=[13,-17,0]
        point,cup=board_target(c);board=cup@board_mount_rotation(c)
        local=-np.array(c['board']['support_offset_mm'])+[0,0,c['board']['size_mm'][2]/2]
        self.assertTrue(np.allclose(point+board@local,c['board']['center_mm']))

if __name__=='__main__':unittest.main()
