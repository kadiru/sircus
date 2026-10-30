import unittest,sys,types
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from bridge import Handler
class Policy(unittest.TestCase):
 def h(self,ip,headers):
  h=object.__new__(Handler);h.client_address=(ip,123);h.server=types.SimpleNamespace(server_port=8766,server_address=('192.168.1.50',8766),token='example-token-not-a-real-secret');h.headers=headers;return h.authorized()
 def test_lan_requires_token(self):
  self.assertFalse(self.h('192.168.1.60',{'Host':'192.168.1.50:8766'}))
  self.assertTrue(self.h('192.168.1.60',{'Host':'192.168.1.50:8766','Authorization':'Bearer example-token-not-a-real-secret'}))
 def test_remote_wrong_token(self):self.assertFalse(self.h('192.168.1.60',{'Host':'192.168.1.50:8766','Authorization':'Bearer wrong'}))
 def test_local_and_host_origin(self):
  self.assertTrue(self.h('127.0.0.1',{'Host':'127.0.0.1:8766'}))
  self.assertFalse(self.h('127.0.0.1',{'Host':'evil.invalid:8766'}))
  self.assertFalse(self.h('127.0.0.1',{'Host':'127.0.0.1:8766','Origin':'http://evil.invalid'}))
if __name__=='__main__':unittest.main(verbosity=2)
