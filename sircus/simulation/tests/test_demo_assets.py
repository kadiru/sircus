"""The local demo reads only public synthetic media and original samples."""
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import server


class DemoAssets(unittest.TestCase):
    def request(self, path):
        handler = server.Handler.__new__(server.Handler)
        handler.path = path
        handler.send = Mock()
        handler.do_GET()
        return handler.send

    def test_only_whitelisted_original_samples_are_read(self):
        for name in server.DEMO_AUDIO_FILES:
            with self.subTest(name=name):
                sent = self.request('/demo-audio/' + name)
                self.assertEqual(sent.call_args.kwargs['content_type'], 'audio/wav')
                self.assertTrue(sent.call_args.args[0].startswith(b'RIFF'))

    def test_reject_other_files_and_traversal(self):
        for name in ('map.json', '../map.json', '../../.private/credentials.json',
                     '%2e%2e%2fmap.json', 'drum_roll.wav/../map.json', ''):
            with self.subTest(name=name):
                self.assertEqual(self.request('/demo-audio/' + name).call_args.args[1], 404)

    def test_sample_route_never_calls_backends(self):
        with patch.multiple(server, AUDIO=Mock(), PHYSICAL=Mock(), BRIDGE=Mock()):
            self.request('/demo-audio/boing.wav')
            self.assertEqual(server.AUDIO.mock_calls, [])
            self.assertEqual(server.PHYSICAL.mock_calls, [])
            self.assertEqual(server.BRIDGE.mock_calls, [])


if __name__ == '__main__':
    unittest.main()
