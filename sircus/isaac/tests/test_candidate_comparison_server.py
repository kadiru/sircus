"""Pure mocked HTTP handler tests; no socket or service is started."""

import io
from pathlib import Path
import sys
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE))
import candidate_comparison_server as server


class CandidateComparisonServerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'report.json').write_text('{"status":"queued"}')

    def request(self, path, head=False):
        cls = server.handler(self.root)
        handler = object.__new__(cls)
        handler.path = path
        handler.wfile = io.BytesIO()
        statuses, headers = [], {}
        handler.send_response = statuses.append
        handler.send_header = lambda key, value: headers.update({key: value})
        handler.end_headers = lambda: None
        handler.send_error = statuses.append
        handler.content(head)
        return statuses, headers, handler.wfile.getvalue()

    def test_explicit_report_only_and_no_output_directory_browsing(self):
        self.assertEqual(server.published_file(self.root, '/report.json?tick=1'), self.root.resolve() / 'report.json')
        for path in ('/qualification.json', '/trials/shot.json', '/../report.json', '/%2e%2e/report.json', '/api/move', '/report.json/child'):
            with self.subTest(path=path):
                self.assertIsNone(server.published_file(self.root, path))
                self.assertEqual(self.request(path)[0], [404])

    def test_report_symlink_is_never_served(self):
        report = self.root / 'report.json'
        report.unlink()
        outside = self.root / 'outside.json'
        outside.write_text('{"private":true}')
        report.symlink_to(outside)
        self.assertIsNone(server.published_file(self.root, '/report.json'))
        self.assertEqual(self.request('/report.json')[0], [404])

    def test_get_head_content_type_no_cache_and_script_restrictions(self):
        status, headers, body = self.request('/report.json')
        self.assertEqual(status, [200])
        self.assertEqual(body, b'{"status":"queued"}')
        self.assertEqual(headers['Content-Type'], 'application/json; charset=utf-8')
        self.assertEqual(headers['Cache-Control'], 'no-store')
        self.assertEqual(headers['X-Content-Type-Options'], 'nosniff')
        self.assertIn("connect-src 'self'", headers['Content-Security-Policy'])
        self.assertEqual(int(headers['Content-Length']), len(body))
        self.assertEqual(self.request('/report.json', head=True)[2], b'')

    def test_static_page_and_module_are_allowlisted(self):
        for path, snippet in (('/', b'Frozen weights.'), ('/candidate_comparison_panel.js', b'export function resultsAudited')):
            status, headers, body = self.request(path)
            self.assertEqual(status, [200])
            self.assertIn(snippet, body)
        cls = server.handler(self.root)
        for method in ('do_POST', 'do_PUT', 'do_DELETE', 'do_PATCH'):
            self.assertFalse(hasattr(cls, method))

    def test_no_runtime_or_control_imports(self):
        source = Path(server.__file__).read_text()
        for forbidden in ('import torch', 'import isaac', 'import subprocess', 'docker', 'Viam'):
            self.assertNotIn(forbidden, source)
        self.assertIn('("127.0.0.1", args.port)', source)


if __name__ == '__main__':
    unittest.main()
