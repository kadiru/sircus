"""Ensure native controller stays on its two-fake-arm machine and pinned geometry."""
import copy
import unittest
from unittest.mock import patch
from cloud_guard import binding,expected_config,validate_cached_config,ROOT
from scipy.spatial.transform import Rotation
class GuardChecks(unittest.TestCase):
 def fixture(self):
  c=expected_config();c['cloud']={'id':binding()['part_id']};return c
 def test_exact(self):validate_cached_config(self.fixture(),binding())
 def test_serialized_frames(self):
  c=self.fixture()
  for r in c['components']:
   if 'frame' not in r:continue
   f=r['frame'];yaw=f['orientation']['value']['th'];q=Rotation.from_euler('z',yaw,degrees=True).as_quat()
   f['orientation']={'type':'quaternion','value':dict(zip(('X','Y','Z','W'),q))}
   f['translation']={k.upper():v for k,v in f['translation'].items()};f['id']=''
   r.update(type=r['api'].rsplit(':',1)[-1],namespace='rdk',depends_on=[],log_level='info')
  validate_cached_config(c,binding())
 def test_reject_changes(self):
  changes=[lambda c:c['cloud'].update(id='other'),lambda c:c.update(remotes=[{'name':'physical'}]),lambda c:c['components'][0].update(model='viam:ufactory:xarm6'),lambda c:c['components'][1]['frame']['translation'].update(x=1400),lambda c:c['components'][3]['frame']['translation'].update(z=101),lambda c:c['network'].update(bind_address='0.0.0.0:8082'),lambda c:c['modules'][0].update(executable_path='/tmp/other'),lambda c:c['services'][1]['attributes'].update(additional_sync_paths=['/tmp']),lambda c:c.update(processes=[{'id':'extra'}])]
  for change in changes:
   c=self.fixture();change(c)
   with self.assertRaises(RuntimeError):validate_cached_config(c,binding())
 def test_changed_geometry(self):
  original=type(ROOT).read_bytes
  with patch.object(type(ROOT),'read_bytes',lambda p:b'{}' if p.name=='suction-proxy.json' else original(p)):
   with self.assertRaises(RuntimeError):validate_cached_config(self.fixture(),binding())
if __name__=='__main__':unittest.main()
