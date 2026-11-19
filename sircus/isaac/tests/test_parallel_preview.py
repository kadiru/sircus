import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path


class ParallelPreviewTests(unittest.TestCase):
    def test_allowlisted_media_and_no_control_routes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "output"
            out.mkdir()
            run = out / "parallel-world-20261101T000000Z-n32"
            run.mkdir()
            (run / "overview.jpg").write_bytes(b"fixture-image")
            (root / "outside.txt").write_text("must stay outside")
            (run / "close.jpg").symlink_to(root / "outside.txt")
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            command = [
                sys.executable,
                str(Path(__file__).resolve().parents[1] / "preview.py"),
                "--output",
                str(out),
                "--port",
                str(port),
            ]
            with subprocess.Popen(
                command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            ) as proc:
                try:
                    base = f"http://127.0.0.1:{port}"
                    for _ in range(100):
                        try:
                            with urllib.request.urlopen(
                                base + "/parallel-world.html", timeout=0.5
                            ) as r:
                                self.assertIn(b"One world. Many workcells.", r.read())
                            break
                        except urllib.error.URLError:
                            time.sleep(0.02)
                    else:
                        self.fail("Preview did not start")
                    with urllib.request.urlopen(
                        base + "/parallel-media/" + run.name + "/overview.jpg?frame=1"
                    ) as r:
                        self.assertEqual(r.read(), b"fixture-image")
                        self.assertEqual(r.headers["Content-Type"], "image/jpeg")
                    for path in [
                        "/parallel-media/" + run.name + "/close.jpg",
                        "/parallel-media/" + run.name + "/trials.jsonl",
                        "/parallel-media/../outside.txt",
                        "/api/move",
                    ]:
                        with (
                            self.subTest(path=path),
                            self.assertRaises(urllib.error.HTTPError) as e,
                        ):
                            urllib.request.urlopen(base + path)
                        self.assertEqual(e.exception.code, 404)
                    with self.assertRaises(urllib.error.HTTPError) as e:
                        urllib.request.urlopen(
                            urllib.request.Request(base + "/api/move", data=b"{}")
                        )
                    self.assertEqual(e.exception.code, 501)
                finally:
                    proc.terminate()
                    proc.wait(timeout=3)
