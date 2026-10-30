"""No SDK, sockets or subprocesses: exercise only launcher decisions."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from sircus.physical_trials import demo_setup
from sircus.physical_trials import run_hardware_demo as launcher


class HardwareLauncherTests(unittest.TestCase):
    def setUp(self):
        self.report = {'prepared': True, 'robot_prepared': True, 'audio_prepared': True}
        self.profile = {'python':'/fake/python', 'port':8765}
        self.argv = ['/fake/python', '/fake/server.py', '--physical-mode', 'hardware']
        self.patches = [patch.object(launcher, 'inspect_profile', return_value=self.report),
                        patch.object(launcher, 'inspect_loaded_profile', return_value=self.report),
                        patch.object(launcher, 'load_profile', return_value=self.profile),
                        patch.object(launcher, 'launch_argv', return_value=self.argv),
                        patch.object(launcher, 'load_audio_environment', return_value={'SIRCUS_AUDIO_URL':'http://127.0.0.1:8766'}),
                        patch.object(launcher.os, 'execvpe')]
        self.check, self.check_loaded, self.profile_load, self.command, self.audio, self.execute = [p.start() for p in self.patches]
        for p in self.patches: self.addCleanup(p.stop)

    def run_cli(self, *args):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            result = launcher.main(['--profile','/fake/profile.json',*args])
        return result, output.getvalue()

    def test_default_checks_without_execution_or_loading_secrets(self):
        code, output = self.run_cli()
        self.assertEqual(code, 0); self.execute.assert_not_called(); self.audio.assert_not_called()
        self.profile_load.assert_not_called();self.check_loaded.assert_not_called()
        self.assertIn('robot_prepared', output)

    def test_explicit_launch_uses_validated_profile_and_replaces_inherited_audio(self):
        with patch.dict(launcher.os.environ, {'SIRCUS_AUDIO_TOKEN':'inherited-secret', 'SIRCUS_AUDIO_URL':'old-address', 'SIRCUS_AUDIO_OTHER':'old'}, clear=False):
            code, output = self.run_cli('--launch')
        self.assertEqual(code, 0)
        executable, argv, env = self.execute.call_args.args
        self.assertEqual(executable, '/fake/python'); self.assertEqual(argv, self.argv)
        self.assertEqual(env['SIRCUS_AUDIO_URL'],'http://127.0.0.1:8766')
        self.assertNotIn('SIRCUS_AUDIO_TOKEN',env);self.assertNotIn('SIRCUS_AUDIO_OTHER',env)
        self.assertNotIn('secret',output);self.assertNotIn('8766',output)
        self.profile_load.assert_called_once()
        self.check_loaded.assert_called_once_with(self.profile)
        self.command.assert_called_once_with(self.profile,launcher.ROOT)
        self.check.assert_not_called()

    def test_profile_edited_during_inspection_cannot_change_launch_paths(self):
        profile = {'schema':demo_setup.SCHEMA,'machine':'sfhack10','python':'/reviewed/python',
                   'credentials_path':'/reviewed/credentials.json','data_root':'/reviewed/hardware',
                   'show_data_root':'/reviewed/show','port':8765,'audio_mode':'disabled','audio_env_file':None}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'profile.json'
            path.write_text(json.dumps(profile))
            def inspect(snapshot):
                self.assertEqual(snapshot,profile)
                path.write_text(json.dumps({**profile,'python':'/changed/python','data_root':'/changed/hardware'}))
                return self.report
            with patch.object(launcher,'load_profile',wraps=demo_setup.load_profile) as load, \
                 patch.object(launcher,'launch_argv',wraps=demo_setup.launch_argv), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.check_loaded.side_effect=inspect
                self.audio.return_value={}
                self.assertEqual(launcher.main(['--profile',str(path),'--launch']),0)
                load.assert_called_once_with(path)
        executable,argv,_=self.execute.call_args.args
        self.assertEqual(executable,'/reviewed/python')
        self.assertIn('/reviewed/hardware',argv)
        self.assertNotIn('/changed/hardware',argv)

    def test_invalid_profile_never_inspects_or_launches(self):
        self.profile_load.side_effect=launcher.SetupError('invalid')
        code,_=self.run_cli('--launch')
        self.assertEqual(code,2);self.check_loaded.assert_not_called();self.execute.assert_not_called()

    def test_robot_failure_blocks_deployment_before_secret_loading(self):
        self.report.update(prepared=False,robot_prepared=False)
        code, _ = self.run_cli('--launch')
        self.assertEqual(code, 2);self.execute.assert_not_called();self.audio.assert_not_called()

    def test_audio_failure_does_not_gate_robot_launch_or_leak_inherited_credentials(self):
        self.report.update(prepared=False,audio_prepared=False)
        with patch.dict(launcher.os.environ, {'SIRCUS_AUDIO_TOKEN':'inherited-secret'}, clear=False):
            code, _ = self.run_cli('--launch')
        self.assertEqual(code,0);self.audio.assert_not_called()
        self.assertFalse(any(k.startswith('SIRCUS_AUDIO_') for k in self.execute.call_args.args[2]))

    def test_audio_change_fails_silent_and_exec_error_is_redacted(self):
        self.audio.side_effect = launcher.SetupError('bad configuration')
        self.execute.side_effect = OSError('sensitive detail')
        code, output = self.run_cli('--launch')
        self.assertEqual(code,2);self.assertNotIn('sensitive',output)
        self.assertFalse(any(k.startswith('SIRCUS_AUDIO_') for k in self.execute.call_args.args[2]))


if __name__ == '__main__':unittest.main()
