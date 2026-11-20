"""Mock HTTP requests only; no port, simulator or service is started."""

import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import location_comparison_server as server
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_location_comparison_recovery_publication import recovery_presentation


class LocationComparisonServerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'media').mkdir()
        # Invented byte fixture, explicitly not a playable native trial.
        self.video = self.root / 'media' / 'batch-00000.mp4'
        self.video.write_bytes(b'invented-video-byte-fixture')
        def metrics(count):
            return dict(requests=count, captures=0, clean=0, rejected=0, misses=count)
        self.report = dict(schema=server.SCHEMA, status='completed_audited', synthetic=False,
            model_promoted=False,
            audit=dict(status='passed', complete=True, synthetic=False),
            videos=[dict(media=dict(url='/media/batch-00000.mp4', verified=True, duration_s=1.,
                sha256=hashlib.sha256(self.video.read_bytes()).hexdigest()))],
            train={m: metrics(8960) for m in ('original_usable', 'broader')},
            final_tests={m: dict(test_broad=metrics(70), test_local=metrics(25))
                         for m in ('parent', 'original_usable', 'broader')})
        self.save_report()

    def save_report(self):
        (self.root / 'report.json').write_text(json.dumps(self.report))

    def request(self, route, *, head=False, byte_range=None):
        viewer = object.__new__(server.handler(self.root))
        viewer.path = route
        viewer.headers = {} if byte_range is None else {'Range': byte_range}
        viewer.wfile = io.BytesIO()
        status, headers = [], {}
        viewer.send_response = status.append
        viewer.send_error = status.append
        viewer.send_header = lambda key, value: headers.update({key: value})
        viewer.end_headers = lambda: None
        viewer.content(head)
        return status, headers, viewer.wfile.getvalue()

    def test_only_report_and_exact_verified_media_are_served(self):
        self.assertEqual(self.request('/report.json?fresh=1')[0], [200])
        self.assertEqual(self.request('/api/location-comparison')[2], self.request('/report.json')[2])
        self.assertEqual(self.request('/media/batch-00000.mp4')[2], self.video.read_bytes())
        for path in ('/native/frames/00000.jpg', '/checkpoints/final.pt', '/comparison-trace.json',
                     '/media/unlisted.mp4', '/media', '/api/move', '/../report.json',
                     '/media/%2e%2e/report.json', '/media/..%2f../report.json'):
            with self.subTest(path=path):
                self.assertEqual(self.request(path)[0], [404])

    def test_recovery_serves_pinned_media_only_with_visible_failure_evidence(self):
        self.report = recovery_presentation(self.report)
        self.save_report()
        self.assertEqual(self.request('/media/batch-00000.mp4')[0], [200])
        good = copy.deepcopy(self.report)
        for key in ('native_exit_sha256', 'audit_exit_sha256', 'inventory_sha256', 'retirement_proof_sha256'):
            self.report = copy.deepcopy(good); self.report['audit'].pop(key); self.save_report()
            self.assertEqual(self.request('/media/batch-00000.mp4')[0], [404])
        self.report = copy.deepcopy(good)
        self.report['costs']['shutdown_failure']['clean_shutdown'] = True; self.save_report()
        self.assertEqual(self.request('/media/batch-00000.mp4')[0], [404])

    def test_pending_failed_synthetic_and_incomplete_reports_expose_no_video(self):
        original = copy.deepcopy(self.report)
        for change in ({'status': 'awaiting_audit'}, {'status': 'failed'}, {'synthetic': True},
                       {'audit': {'status': 'passed', 'complete': False, 'synthetic': False}},
                       {'audit': {'status': 'passed', 'complete': True, 'synthetic': True}},
                       {'schema': 'unknown'}, {'model_promoted': True}, {'train': {}}, {'final_tests': {}}):
            self.report = {**copy.deepcopy(original), **change}
            self.save_report()
            self.assertEqual(self.request('/report.json')[0], [200])
            self.assertEqual(self.request('/media/batch-00000.mp4')[0], [404])

    def test_missing_verification_changed_bytes_and_duplicate_routes_fail_closed(self):
        for mutation in ('unverified', 'bad_hash', 'changed_bytes', 'duplicate'):
            with self.subTest(mutation=mutation):
                original = copy.deepcopy(self.report)
                original_bytes = self.video.read_bytes()
                if mutation == 'unverified':
                    self.report['videos'][0]['media']['verified'] = False
                elif mutation == 'bad_hash':
                    self.report['videos'][0]['media']['sha256'] = 'bad'
                elif mutation == 'changed_bytes':
                    self.video.write_bytes(b'changed after publication')
                else:
                    self.report['videos'].append(copy.deepcopy(self.report['videos'][0]))
                self.save_report()
                self.assertEqual(self.request('/media/batch-00000.mp4')[0], [404])
                self.report = original
                self.video.write_bytes(original_bytes)
                self.save_report()

    def test_media_symlink_and_parent_symlink_even_inside_root_are_refused(self):
        content = self.video.read_bytes()
        self.video.unlink()
        target = self.root / 'private.mp4'
        target.write_bytes(content)
        self.video.symlink_to(target)
        self.assertEqual(self.request('/media/batch-00000.mp4')[0], [404])
        self.video.unlink()
        self.video.write_bytes(content)
        (self.root / 'media').rename(self.root / 'elsewhere')
        (self.root / 'media').symlink_to(self.root / 'elsewhere', target_is_directory=True)
        self.assertEqual(self.request('/media/batch-00000.mp4')[0], [404])

    def test_report_symlink_and_malformed_report_are_refused(self):
        path = self.root / 'report.json'
        path.write_text('{not valid json')
        self.assertEqual(self.request('/media/batch-00000.mp4')[0], [404])
        path.unlink()
        target = self.root / 'private.json'
        target.write_text(json.dumps(self.report))
        path.symlink_to(target)
        self.assertEqual(self.request('/report.json')[0], [404])

    def test_output_root_symlink_and_incomplete_counts_are_refused(self):
        alias = self.root / 'linked-output'
        alias.symlink_to(self.root, target_is_directory=True)
        self.assertIsNone(server.published_file(alias, '/report.json'))
        self.report['final_tests']['broader']['test_local']['requests'] = 24
        self.save_report()
        self.assertEqual(self.request('/media/batch-00000.mp4')[0], [404])

    def test_byte_ranges_head_and_browser_security_headers(self):
        status, headers, body = self.request('/media/batch-00000.mp4', byte_range='bytes=2-6')
        self.assertEqual(status, [206])
        self.assertEqual(body, self.video.read_bytes()[2:7])
        self.assertEqual(headers['Content-Length'], '5')
        self.assertEqual(headers['Content-Range'], f'bytes 2-6/{self.video.stat().st_size}')
        self.assertEqual(headers['Content-Type'], 'video/mp4')
        self.assertEqual(headers['Cache-Control'], 'no-store')
        self.assertEqual(headers['X-Content-Type-Options'], 'nosniff')
        self.assertIn("media-src 'self'", headers['Content-Security-Policy'])
        self.assertEqual(self.request('/media/batch-00000.mp4', head=True)[2], b'')
        self.assertEqual(self.request('/media/batch-00000.mp4', byte_range='bytes=-5')[2], self.video.read_bytes()[-5:])
        for value in ('bytes=999-', 'bytes=0-1,3-4', 'bytes=-0', 'bytes=10-1'):
            self.assertEqual(self.request('/media/batch-00000.mp4', byte_range=value)[0], [416])

    def test_static_allowlist_and_no_control_verbs(self):
        for route in server.STATIC:
            self.assertEqual(self.request(route)[0], [200])
        cls = server.handler(self.root)
        for method in ('do_POST', 'do_PUT', 'do_DELETE', 'do_PATCH'):
            self.assertFalse(hasattr(cls, method))
        source = Path(server.__file__).read_text()
        for forbidden in ('import torch', 'import isaac', 'import subprocess', 'docker', 'Viam'):
            self.assertNotIn(forbidden, source)
        self.assertIn("('127.0.0.1', args.port)", source)


if __name__ == '__main__':
    unittest.main()
