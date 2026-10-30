"""Acceptance-runner contracts: independent output, truthful evidence and files."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from sircus.physical_trials import validate_deterministic as validation


class ValidationRunnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_repeated_run_keeps_unknown_outcomes_and_checksummed_observations(self):
        with tempfile.TemporaryDirectory() as directory, patch.multiple(
                validation.module, LOAD_WAIT_SECONDS=.01, DROP_SETTLE_SECONDS=.01,
                SETTLE_SECONDS=.01, SAMPLE_SECONDS=.01, POST_RELEASE_SECONDS=.01):
            root = Path(directory) / 'run'
            report = await validation.validate(root, 2)
            self.assertEqual(report['status'], 'passed')
            self.assertEqual(report['duplicate_presses_rejected'], 2)
            self.assertIsNone(report['capture_rate'])
            self.assertEqual([row['ball_id'] for row in report['cycles']], ['ball-pink', 'ball-purple'])
            self.assertTrue(all(row['outcome'] is None for row in report['cycles']))
            self.assertEqual([row['loading_open_skipped'] for row in report['cycles']], [False, True])
            manifest = json.loads((root / 'manifest.json').read_text())['files']
            self.assertIn('report.json', manifest)
            for name, checksum in manifest.items():
                self.assertEqual(hashlib.sha256((root / name).read_bytes()).hexdigest(), checksum)

    async def test_existing_run_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / 'keep.txt'
            marker.write_text('prior evidence')
            with self.assertRaises(FileExistsError):
                await validation.validate(directory, 1)
            self.assertEqual(marker.read_text(), 'prior evidence')
            self.assertFalse((Path(directory) / 'report.json').exists())

    async def test_failed_grip_is_reported_as_failure_not_endurance_success(self):
        class NoBall(validation.MockAdapter):
            async def close_gripper(self):
                await super().close_gripper()
                return {'grabbed': False}

        with tempfile.TemporaryDirectory() as directory, patch.object(validation, 'MockAdapter', NoBall), patch.multiple(
                validation.module, LOAD_WAIT_SECONDS=.01, DROP_SETTLE_SECONDS=.01,
                SETTLE_SECONDS=.01, SAMPLE_SECONDS=.01, POST_RELEASE_SECONDS=.01):
            root = Path(directory) / 'run'
            with self.assertRaises(AssertionError):
                await validation.validate(root, 1)
            report = json.loads((root / 'report.json').read_text())
            self.assertEqual(report['status'], 'failed')
            self.assertEqual(report['completed_cycles'], 0)
            self.assertIsNone(report['capture_rate'])

    async def test_invalid_run_size_does_not_create_a_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'run'
            for value in (0, -1, True, 1001):
                with self.assertRaises(ValueError):
                    await validation.validate(root, value)
            self.assertFalse(root.exists())
