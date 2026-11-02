"""Measured-pose composition and read-only loopback serving contracts."""

import http.client
import json
import sys
import tempfile
import threading
import unittest
import xml.etree.ElementTree as ET
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from camera_scout_export import pose, visual_transform
from camera_scout_server import handler


class CameraScoutTests(unittest.TestCase):
    def test_native_wxyz_pose_and_fixed_visual_keep_measured_world_transform(self):
        q = [np.sqrt(0.5), 0, 0, np.sqrt(0.5)]
        measured = pose([4, 5, 6], q)
        root = ET.fromstring(
            '<robot><joint type="fixed"><parent link="body"/><child link="tool"/><origin xyz="1 0 0" rpy="0 0 0"/></joint></robot>'
        )
        result = visual_transform("tool", {"body": measured}, root)
        np.testing.assert_allclose(result[:3, 3], [4, 6, 6], atol=1e-12)
        # Column-major output must place the world translation in THREE's slots.
        np.testing.assert_allclose(result.T.flatten()[12:15], [4, 6, 6], atol=1e-12)
        np.testing.assert_array_equal(measured[:3, 3], [4, 5, 6])

    def test_missing_moving_body_is_rejected_instead_of_guessing_pose(self):
        root = ET.fromstring(
            '<robot><joint type="revolute"><parent link="body"/><child link="tool"/><origin xyz="1 0 0"/></joint></robot>'
        )
        with self.assertRaisesRegex(ValueError, "No measured transform"):
            visual_transform("tool", {"body": np.eye(4)}, root)

    def test_static_routes_do_not_expose_parent_files_or_command_apis(self):
        with tempfile.TemporaryDirectory() as temporary:
            data = Path(temporary)
            (data / "scene.json").write_text(json.dumps(dict(schema="fixture")))
            server = ThreadingHTTPServer(("127.0.0.1", 0), handler(data))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:

                def request(path, method="GET"):
                    connection = http.client.HTTPConnection(
                        "127.0.0.1", server.server_port, timeout=3
                    )
                    connection.request(method, path)
                    response = connection.getresponse()
                    status, raw = response.status, response.read()
                    connection.close()
                    return status, raw

                status, raw = request("/scene.json")
                self.assertEqual(status, 200)
                self.assertEqual(json.loads(raw), dict(schema="fixture"))
                self.assertEqual(request("/vendor/OrbitControls.js")[0], 200)
                for path in (
                    "/../input/dropper.urdf",
                    "/meshes/",
                    "/api/physical/status",
                    "/api/physical/action",
                ):
                    self.assertEqual(request(path)[0], 404)
                self.assertEqual(request("/api/physical/action", "POST")[0], 501)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=3)


if __name__ == "__main__":
    unittest.main()
