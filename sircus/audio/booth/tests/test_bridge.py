import json,sys,unittest,wave,struct,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from bridge import Controller,ROOT,MAP
class Backend:
 def __init__(self):self.commands=[]
 def send(self,**kw):self.commands.append(kw)
class Tests(unittest.TestCase):
 def setUp(self):
  self.b=Backend();self.t=10.;self.c=Controller(self.b,clock=lambda:self.t);self.c.ready=True
 def on(self):self.c.enable(True)
 def midi(self,v=90,n=0,status=144):return self.c.midi({'input':'mock','status':status,'data':[n,v]},'mock')
 def sounds(self):return [m['sound'] for m in self.b.commands if m['cmd']=='play']
 def test_starts_off(self):self.midi();self.assertEqual(self.sounds(),[]);self.assertFalse(self.c.status()['fx_enabled'])
 def test_free_rapid_and_variation(self):
  self.on()
  for i in range(8):self.midi();self.t+=.002
  self.assertEqual(len(self.sounds()),8);self.assertEqual(len(set(self.sounds())),4)
 def test_trial_first_is_heuristic_only(self):
  self.c.settings('mode',{'mode':'trial'});self.on();self.assertFalse(self.midi()['accepted'])
  self.c.event({'type':'release','trial_id':'x'});self.midi();self.midi();self.assertEqual(self.sounds(),['release','boing','bone_low']);self.assertNotIn('capture',self.sounds())
 def test_trial_expiration(self):
  self.c.settings('mode',{'mode':'trial'});self.on();self.c.event({'type':'release'});self.t+=13;self.assertFalse(self.midi()['accepted'])
 def test_velocity_zero_cc_wrong_note_channel(self):
  self.on();self.midi(v=0);self.midi(n=1);self.midi(status=145);self.midi(status=176);self.midi(status=128);self.assertEqual(self.sounds(),[])
 def test_optional_dedupe_and_real_retrigger(self):
  self.c.settings('mapping',{'exact_duplicate_ms':10});self.on();self.midi();self.t+=.002;self.midi();self.assertEqual(len(self.sounds()),1)
  self.midi(status=128,v=0);self.midi();self.assertEqual(len(self.sounds()),2)
  self.midi(v=91);self.assertEqual(len(self.sounds()),3)
 def test_dedupe_bound_and_finite(self):
  for d in ({'exact_duplicate_ms':100},{'velocity_gamma':float('nan')},{'channel':0},{'impact_note':True}):
   with self.assertRaises(ValueError):self.c.settings('mapping',d)
 def test_event_id_and_release_idempotency(self):
  self.on();e={'type':'release','trial_id':'x','event_id':'e'};self.c.event(e);self.midi();self.assertFalse(self.c.event(e)['accepted']);self.assertFalse(self.c.event({**e,'event_id':'f'})['accepted']);self.midi();self.assertEqual(self.sounds().count('release'),1)
 def test_explicit_terminal_ends_trial(self):
  self.on();self.c.settings('mode',{'mode':'trial'});self.c.event({'type':'release','trial_id':'x'});self.c.event({'type':'capture','trial_id':'old'});self.assertNotIn('capture',self.sounds());self.c.event({'type':'capture','trial_id':'x'});self.assertFalse(self.midi()['accepted'])
 def test_disable_panic_cancels_transport_and_arming(self):
  self.on();self.c.event({'type':'transport_start'});self.c.enable(False);self.assertFalse(self.c.transport);self.assertEqual(self.b.commands[-1],{'cmd':'panic'});self.assertFalse(self.c.status()['armed']);self.midi();self.assertEqual(self.sounds(),['transport'])
 def test_release_stops_whoosh(self):
  self.on();self.c.event({'type':'transport_start'});self.c.event({'type':'release'});self.assertFalse(self.c.transport);self.assertIn({'cmd':'stop_transport'},self.b.commands)
 def test_disconnect_reconnect_requires_reenable(self):
  self.on();self.c.native({'type':'connection','connected':False});self.assertFalse(self.c.enabled);self.c.native({'type':'connection','connected':True});self.assertFalse(self.c.enabled)
 def test_input_identity_filter(self):
  self.on();self.c.selected=12;self.c.connected=True;self.c.midi({'input':13,'status':144,'data':[0,90]});self.assertEqual(self.sounds(),[]);self.c.midi({'input':12,'status':144,'data':[0,90]});self.assertEqual(len(self.sounds()),1)
 def test_native_error_silences_voices(self):
  self.on();self.c.event({'type':'transport_start'});self.c.native({'type':'error','message':'injected MIDI error'});self.assertFalse(self.c.enabled);self.assertFalse(self.c.transport);self.assertEqual(self.b.commands[-1],{'cmd':'panic'})
 def test_velocity_scales_monotonically(self):
  self.on();self.midi(v=20);self.midi(v=100);v=[r['velocity'] for r in self.b.commands if r['cmd']=='play'];self.assertLess(v[0],v[1])
 def test_replay_validation(self):
  for rows in ([{'t':0,'status':144,'data':[0,128]}],[{'t':float('inf')}],[{'t':1,'status':144,'data':[0,50]},{'t':0,'status':144,'data':[0,50]}]):
   with self.assertRaises(ValueError):self.c.replay(rows)
 def test_logic_requires_manual_gate(self):
  with self.assertRaises(ValueError):self.c.settings('route',{'route':'logic'})
  self.assertEqual(self.c.route,'native')
 def test_assets_conservative_and_midi(self):
  for m in MAP:
   with wave.open(str(ROOT/'assets'/m['file'])) as w:
    self.assertEqual((w.getnchannels(),w.getsampwidth(),w.getframerate()),(1,2,48000));raw=w.readframes(w.getnframes());samples=struct.unpack('<'+'h'*(len(raw)//2),raw);self.assertLessEqual(max(abs(x) for x in samples),6554);self.assertGreater(max(abs(x) for x in samples),1000)
  self.assertEqual((ROOT/'assets/audition.mid').read_bytes()[:4],b'MThd')
class MidiFile(unittest.TestCase):
 def test_complete_audition_mapping_and_gate(self):
  b=(ROOT/'assets/audition.mid').read_bytes();self.assertEqual(struct.unpack('>IHHH',b[4:14]),(6,0,1,480));self.assertEqual(b[14:18],b'MTrk');length=struct.unpack('>I',b[18:22])[0];self.assertEqual(length,len(b)-22)
  i=22;notes=[];offs=[];gate=False;ended=False
  def variable():
   nonlocal i
   value=0
   while True:
    n=b[i];i+=1;value=(value<<7)|(n&127)
    if n<128:return value
  while i<len(b):
   variable();kind=b[i];i+=1
   if kind==255:
    meta=b[i];i+=1;size=variable();i+=size
    if meta==47:ended=True
   else:
    x,y=b[i:i+2];i+=2
    if kind==191 and x==11:gate=y>0
    elif kind==159:self.assertTrue(gate);notes.append(x)
    elif kind==143:offs.append(x)
  self.assertEqual(notes,[m['note'] for m in MAP]);self.assertEqual(offs,notes);self.assertFalse(gate);self.assertTrue(ended)
if __name__=='__main__':unittest.main(verbosity=2)
