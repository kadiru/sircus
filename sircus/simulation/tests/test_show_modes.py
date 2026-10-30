"""Show mode may combine numerical tools with mocks, never physical execution."""
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import server

class ShowModeIsolation(unittest.TestCase):
    def test_numerical_routes_allow_only_explicit_show_mock(self):
        handler=server.Handler.__new__(server.Handler)
        for show,offline,mode,wanted in [(True,True,'dry_run',True),(False,True,'dry_run',False),
                (True,True,'hardware',False),(True,False,'dry_run',False),(True,True,None,False)]:
            with self.subTest(show=show,offline=offline,mode=mode):
                with patch.multiple(server, SHOW_ENABLED=show,OFFLINE=offline,
                     PHYSICAL=types.SimpleNamespace(mode=mode),OPERATIONS=Mock(),TRAINING_JOBS=Mock()):
                    self.assertEqual(handler.workflow_ready(),wanted)
    def test_hardware_cannot_be_mislabelled_as_mock_by_request(self):
        handler=server.Handler.__new__(server.Handler)
        with patch.multiple(server,SHOW_ENABLED=True,OFFLINE=True,
                PHYSICAL=types.SimpleNamespace(mode='hardware'),OPERATIONS=Mock(),TRAINING_JOBS=Mock()):
            self.assertFalse(handler.workflow_ready())

if __name__=='__main__':unittest.main()
