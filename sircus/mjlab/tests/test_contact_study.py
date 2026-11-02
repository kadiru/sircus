import unittest
import xml.etree.ElementTree as ET

import numpy as np

from sircus.mjlab.contact_model import board_xml
from sircus.mjlab.contact_study import flight_metrics, shot_plan


class ContactTests(unittest.TestCase):
    def test_proxy_preserves_geometry_and_names(self):
        xml='<mujoco><worldbody><body name="Arm14/link6"><body name="Arm14/impact_board" pos="0 0 .1"><geom name="board/geom" size=".1 .15 .0025" solref=".004 .15"/></body></body></worldbody></mujoco>'
        self.assertEqual(board_xml(xml,"articulated"),xml)
        root=ET.fromstring(board_xml(xml,"kinematic"))
        board=root.find("worldbody/body[@name='Arm14/impact_board']")
        self.assertEqual(board.get("mocap"),"true")
        self.assertEqual(board.find("geom").attrib,ET.fromstring(xml).find(".//geom").attrib)
        self.assertIsNone(root.find(".//body[@name='Arm14/link6']/body"))

    def test_proxy_refuses_jointed_board(self):
        with self.assertRaises(ValueError):board_xml('<mujoco><worldbody><body name="Arm14/impact_board"><joint/></body></worldbody></mujoco>',"kinematic")

    def test_plan_keeps_nominal_and_original_identical(self):
        source=dict(values=[.47,-.28,.4,36,.3],q10=[0]*6,q14=[1]*6)
        old=dict(id="old",values=[.471,-.28,.399,36.1,.301],q10=[.1]*6,q14=[1.1]*6,route_success=False)
        first=shot_plan(source,[old],fresh=4,nominal=3);second=shot_plan(source,[old],fresh=4,nominal=3)
        self.assertEqual(first,second);self.assertEqual(len(first),8)
        self.assertEqual(first[0]["values"],source["values"])
        self.assertEqual(first[3]["values"],old["values"])
        self.assertFalse(first[3]["historical_success"])
        self.assertNotIn("historical_success",first[4])

    def test_first_contact_outgoing_velocity_uses_contact_end(self):
        f=np.zeros((10,1,6));f[2:4,0,0]=20
        v=np.zeros((10,1,3));v[4,0,2]=2
        arrays=dict(forces_N=f,velocity_m_s=v,position_m=np.zeros_like(v))
        r=flight_metrics(arrays,dict(board_normal=[0,0,1]),0)
        self.assertEqual(r["first_board_contact_s"],3/240)
        self.assertEqual(r["outgoing_normal_speed_m_s"],2)

    def test_missing_contact_stays_missing(self):
        arrays=dict(forces_N=np.zeros((10,1,6)),velocity_m_s=np.zeros((10,1,3)),position_m=np.zeros((10,1,3)))
        r=flight_metrics(arrays,dict(board_normal=[0,0,1]),0)
        self.assertIsNone(r["outgoing_velocity_m_s"])


if __name__=="__main__":unittest.main()
