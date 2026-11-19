"""Filesystem-only hardware profile checks; temporary fake files, no devices."""
import copy
import hashlib
import json
import os
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch

from sircus.physical_trials import demo_setup as setup


class HardwareDemoSetupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        venv = self.root / 'project-venv'
        python = venv / 'bin/python';python.parent.mkdir(parents=True)
        python.write_text('#!/bin/false\n');python.chmod(0o755)
        (venv/'pyvenv.cfg').write_text('version = 3.12.14\n')
        site = venv/'lib/python3.12/site-packages';site.mkdir(parents=True)
        for name, version in [('viam-sdk','0.85.1'),('numpy','2.5.4'),('scipy','1.18.1')]:
            folder = site / (name.replace('-','_')+'-'+version+'.dist-info');folder.mkdir()
            (folder/'METADATA').write_text(f'Name: {name}\nVersion: {version}\n')
        self.credentials = self.root/'credentials.json'
        self.write_json(self.credentials, {'address': setup.ADDRESS, 'api_key_id':'fake-key-id', 'api_key':'fake-not-a-secret'}, private=True)
        self.data = self.root/'data/hardware';self.data.mkdir(parents=True)
        pose = dict(x=150.,y=-600.,z=200.,o_x=0.,o_y=0.,o_z=-1.,theta=-85.)
        self.calibration = {'frame':'world','loading_pose':pose,'drop_pose':dict(pose,x=600.,y=-150.,z=600.),
                            'board_joints':[1.,2.,3.,4.,5.,6.],'confirmed':False}
        self.write_json(self.data/'calibration.json',self.calibration)
        self.profile = {'schema':setup.SCHEMA,'machine':'sfhack10','python':str(python),
                        'credentials_path':str(self.credentials),'data_root':str(self.data),
                        'show_data_root':str(self.root/'show'),'port':8765,'audio_mode':'disabled','audio_env_file':None}
        self.path = self.root/'profile.json';self.write_json(self.path,self.profile)

    def write_json(self,path,value,private=False):
        Path(path).write_text(json.dumps(value));Path(path).chmod(0o600 if private else 0o644)

    def inspect(self,profile=None):
        self.write_json(self.path,profile or self.profile)
        return setup.inspect_profile(self.path)

    def test_saved_fixed_targets_prepare_without_height_calibration_or_live_claim(self):
        report=self.inspect()
        self.assertTrue(report['prepared']);self.assertTrue(report['robot_prepared'])
        self.assertFalse(report['live_hardware_verified']);self.assertFalse(report['sound_goal_commissioned'])
        self.assertFalse(report['checks']['calibration']['variable_height_ready'])
        self.assertFalse(report['checks']['calibration']['height_limits_required_for_fixed_cycle'])
        self.assertEqual(report['checks']['audio']['status'],'disabled')
        argv=report['launch_argv']
        self.assertEqual(argv[argv.index('--physical-mode')+1],'hardware')
        self.assertEqual(argv[argv.index('--physical-data-root')+1],str(self.data))
        self.assertIn('--offline',argv);self.assertIn('--show',argv)
        self.assertNotIn('fake-not-a-secret',json.dumps(report))

    def test_inspection_never_connects_executes_or_changes_host_records(self):
        trialdir=self.data/'trials/retained';trialdir.mkdir(parents=True)
        self.write_json(trialdir/'trial.json',{'mode':'hardware','machine':'sfhack10','saved':False,'id':'retained'})
        before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in self.root.rglob('*') if p.is_file()}
        with patch.object(socket,'create_connection',side_effect=AssertionError('No network')), \
             patch('subprocess.Popen',side_effect=AssertionError('No process')):
            report=setup.inspect_profile(self.path)
        after={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in self.root.rglob('*') if p.is_file()}
        self.assertEqual(before,after)
        self.assertEqual(report['checks']['calibration']['pending_journals'],1)

    def test_credentials_wrong_target_permissions_and_symlink_rejected_without_leak(self):
        for content,mode in [({'address':'wrong','api_key_id':'secret-id','api_key':'secret-token'},0o600),
                             ({'address':setup.ADDRESS,'api_key_id':'secret-id','api_key':'secret-token'},0o644)]:
            with self.subTest(mode=mode):
                self.write_json(self.credentials,content);self.credentials.chmod(mode)
                report=self.inspect();self.assertFalse(report['robot_prepared'])
                self.assertNotIn('secret-token',json.dumps(report));self.assertNotIn('secret-id',json.dumps(report))
        self.credentials.chmod(0o600)
        symlink=self.root/'linked.json';symlink.symlink_to(self.credentials)
        report=self.inspect(dict(self.profile,credentials_path=str(symlink)))
        self.assertFalse(report['robot_prepared'])

    def test_foreign_or_mock_calibration_and_data_are_rejected(self):
        for mutation in [{'mode':'dry_run'},{'machine':'sircus-sim'},{'board_joints':[0.]},
                         {'frame':'camera'}, {'loading_pose':{'x':0}}]:
            with self.subTest(mutation=mutation):
                self.write_json(self.data/'calibration.json',{**self.calibration,**mutation})
                self.assertFalse(self.inspect()['robot_prepared'])
        self.write_json(self.data/'calibration.json',self.calibration)
        trialdir=self.data/'trials/mock';trialdir.mkdir(parents=True)
        self.write_json(trialdir/'trial.json',{'mode':'dry_run','machine':'sfhack10','saved':True})
        self.assertFalse(self.inspect()['robot_prepared'])
        self.assertFalse(self.inspect(dict(self.profile,data_root=str(self.root/'dry_run')))['robot_prepared'])

    def test_operator_configured_poses_cannot_bypass_explicit_world_frame(self):
        for frame in ('camera', None):
            with self.subTest(frame=frame):
                calibration = dict(self.calibration, pose_source='operator_configured_tcp_targets')
                if frame is None:
                    calibration.pop('frame')
                else:
                    calibration['frame'] = frame
                self.write_json(self.data/'calibration.json', calibration)
                report = self.inspect()
                self.assertFalse(report['robot_prepared'])
                self.assertEqual(report['checks']['calibration']['status'], 'missing_or_invalid')
        self.write_json(self.data/'calibration.json',
                        dict(self.calibration, pose_source='operator_configured_tcp_targets'))
        self.assertTrue(self.inspect()['robot_prepared'])

    def test_show_state_cannot_alias_or_nest_hardware_records(self):
        alias = self.root/'data-alias';alias.symlink_to(self.data.parent, target_is_directory=True)
        for path in (self.data, self.data/'show', self.data.parent,
                     self.data/'nested/..', alias/'hardware', alias/'hardware/show'):
            with self.subTest(path=path):
                report = self.inspect(dict(self.profile, show_data_root=str(path)))
                self.assertFalse(report['robot_prepared'])
                self.assertEqual(report['checks']['show_data']['status'], 'missing_or_invalid')
        self.assertTrue(self.inspect(dict(self.profile, show_data_root=str(self.data.parent/'show')))['robot_prepared'])

    def test_node_audio_configuration_is_private_mode_scoped_and_redacted(self):
        env=self.root/'audio.json';value={'SIRCUS_AUDIO_URL':'http://192.168.1.75:8766','SIRCUS_AUDIO_TOKEN':'fake-audio-token'}
        self.write_json(env,value,private=True)
        mini=dict(self.profile,audio_mode='mini',audio_env_file=str(env))
        with patch.object(setup.shutil,'which',return_value='/mock/node'):
            report=self.inspect(mini)
        self.assertTrue(report['robot_prepared']);self.assertTrue(report['audio_prepared'])
        self.assertFalse(report['checks']['audio']['connection_verified'])
        self.assertNotIn(value['SIRCUS_AUDIO_URL'],json.dumps(report));self.assertNotIn(value['SIRCUS_AUDIO_TOKEN'],json.dumps(report))
        self.assertEqual(setup.load_audio_environment(mini),value)
        with self.assertRaises(setup.SetupError):setup.load_audio_environment(dict(mini,audio_mode='local'))
        for url in ['http://8.8.8.8:8766','https://192.168.1.75:8766','http://user:pass@192.168.1.75:8766','http://192.168.1.75:8766/?token=fake']:
            self.write_json(env,dict(value,SIRCUS_AUDIO_URL=url),private=True)
            report=self.inspect(mini);self.assertFalse(report['audio_prepared']);self.assertTrue(report['robot_prepared'])
            self.assertNotIn(url,json.dumps(report));self.assertNotIn('fake-audio-token',json.dumps(report))

    def test_local_audio_allows_loopback_and_never_requires_sound_for_robot_readiness(self):
        env=self.root/'audio.json';self.write_json(env,{'SIRCUS_AUDIO_URL':'http://127.0.0.1:8766'},private=True)
        local=dict(self.profile,audio_mode='local',audio_env_file=str(env))
        with patch.object(setup.shutil,'which',return_value=None):report=self.inspect(local)
        self.assertTrue(report['robot_prepared']);self.assertFalse(report['audio_prepared'])
        self.assertEqual(report['checks']['audio']['status'],'node_missing')
        self.assertIn('launch_argv',report)

    def test_project_environment_version_is_checked_without_importing_sdk(self):
        p=Path(self.profile['python']).parent.parent/'pyvenv.cfg';p.write_text('version = 3.13.0\n')
        report=self.inspect();self.assertFalse(report['robot_prepared'])
        p.write_text('version = 3.12.14\n')
        metadata=next(p.parent.glob('lib/python*/site-packages/scipy-*.dist-info/METADATA'))
        metadata.write_text('Name: scipy\nVersion: 0.0.0\n')
        self.assertFalse(self.inspect()['robot_prepared'])

    def test_profile_rejects_implicit_paths_wrong_machine_and_unknown_fields(self):
        for change in [{'credentials_path':'relative.json'},{'machine':'sfhack14'},
                       {'port':True},{'audio_mode':'public'}, {'SIRCUS_AUDIO_TOKEN':'should-not-be-here'}]:
            with self.subTest(change=change):
                report=self.inspect({**self.profile,**change});self.assertFalse(report['prepared'])
                self.assertNotIn('should-not-be-here',json.dumps(report))


if __name__=='__main__':unittest.main()
