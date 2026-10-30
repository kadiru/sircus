import copy,json,os,sys,tempfile,threading,time,types,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from bridge import Controller,Native,SOUNDS
from profiles import Profiles,EVENTS
from sircus.audio.contract import AudioEvent

class Backend:
 def __init__(self):self.commands=[]
 def send(self,**kw):self.commands.append(kw)

class ProfileTests(unittest.TestCase):
 def setUp(self):self.p=Profiles(SOUNDS)
 def test_any_sound_can_be_assigned_to_each_event(self):
  for event in EVENTS:
   for sound in SOUNDS:
    value=copy.deepcopy(self.p.assignments);value[event]['sounds']=[sound]
    self.p.apply(value);self.assertEqual(self.p.choose(event)[0],sound)
 def test_empty_and_zero_gain_silent(self):
  value=copy.deepcopy(self.p.assignments);value['impact']['sounds']=[];value['pickup']['gain']=0
  self.p.apply(value);self.assertIsNone(self.p.choose('impact')[0]);self.assertIsNone(self.p.choose('pickup')[0])
 def test_cycle_fixed_random_no_repeat(self):
  value=copy.deepcopy(self.p.assignments);value['impact']['sounds']=['boing','release'];value['impact']['selection']='cycle';self.p.apply(value)
  self.assertEqual([self.p.choose('impact')[0] for _ in range(4)],['boing','release']*2)
  value['impact']['selection']='fixed';self.p.apply(value);self.assertEqual([self.p.choose('impact')[0] for _ in range(3)],['boing']*3)
  value['impact']['selection']='random';self.p.apply(value);draws=[self.p.choose('impact')[0] for _ in range(30)];self.assertTrue(all(a!=b for a,b in zip(draws,draws[1:])))
 def test_invalid_profile_preserves_previous(self):
  original=copy.deepcopy(self.p.assignments)
  for key,value in [('sounds',['missing']),('gain',float('nan')),('pitch',13),('max_seconds',21),('loop',True),('avoid_repeat',1)]:
   invalid=copy.deepcopy(original);invalid['impact'][key]=value
   with self.assertRaises(ValueError):self.p.apply(invalid)
   self.assertEqual(self.p.assignments,original)
 def test_save_restart_roundtrip_and_failed_save_does_not_overwrite(self):
  with tempfile.TemporaryDirectory() as folder:
   path=Path(folder)/'profiles.json';p=Profiles(SOUNDS,path);p.load('Cartoon Circus');p.save('My circus',p.assignments)
   before=path.read_bytes();restored=Profiles(SOUNDS,path);self.assertEqual(restored.active,'My circus');self.assertEqual(restored.assignments,p.assignments)
   invalid=copy.deepcopy(p.assignments);invalid['carry']['max_seconds']=999
   with self.assertRaises(ValueError):p.save('My circus',invalid)
   self.assertEqual(path.read_bytes(),before)
   with self.assertRaises(ValueError):p.save('Ghost Booth',p.assignments)
   with patch('profiles.os.replace',side_effect=OSError('disk error')):
    with self.assertRaises(OSError):p.save('Different',p.assignments)
   self.assertEqual(path.read_bytes(),before);self.assertNotIn('Different',p.saved)
 def test_corrupt_store_is_preserved_with_visible_fallback(self):
  with tempfile.TemporaryDirectory() as folder:
   path=Path(folder)/'profiles.json';raw='{"schema":"sircus.sound-profiles.v1","profiles":[],"active":"x"}';path.write_text(raw)
   p=Profiles(SOUNDS,path);self.assertTrue(p.error);self.assertEqual(p.active,'Ghost Booth');self.assertEqual(path.read_text(),raw)

class EventTests(unittest.TestCase):
 def setUp(self):
  self.b=Backend();self.c=Controller(self.b);self.c.ready=True;self.c.enable(True)
 def ev(self,kind,**kw):
  value=dict(event_id=kind,trial_id='t1',source='simulation',event_type=kind,contact_object='unknown',timestamp=0,time_basis='simulation_elapsed_s',clock_id='test',provenance='Synthetic test',evidence='model_prediction');value.update(kw);return AudioEvent(**value)
 def played(self):return [c for c in self.b.commands if c['cmd']=='play']
 def test_event_lifecycle_uses_profile_and_stops_loops(self):
  for kind in ['pickup','carry_start','release','flight_start']:
   self.assertTrue(self.c.audio_event(self.ev(kind))['accepted'])
  self.assertTrue(self.c.transport)
  self.c.audio_event(self.ev('impact',contact_object='board'))
  self.assertFalse(self.c.transport);self.assertEqual([x['sound'] for x in self.played()],['eerie_bell','transport','release','transport','boing'])
  self.c.audio_event(self.ev('outcome',outcome='captured'));self.assertEqual(self.played()[-1]['sound'],'capture')
  self.assertFalse(self.c.audio_event(self.ev('impact',event_id='late'))['accepted'])
 def test_unknown_and_unconfirmed_outcome_silent(self):
  self.c.audio_event(self.ev('outcome'));self.assertEqual(self.played(),[])
  c=Controller(self.b);c.ready=True;c.enable(True)
  c.audio_event(self.ev('outcome',source='operator',time_basis='unix_s',evidence='operator_label',outcome='missed'))
  self.assertEqual(self.played(),[])
 def test_operator_action_cannot_confirm_outcome(self):
  with self.assertRaises(ValueError):self.ev('outcome',source='operator',time_basis='unix_s',evidence='operator_action',outcome='captured',confirmed=True)
 def test_retired_release_cannot_replace_current_trial(self):
  self.c.audio_event(self.ev('release'));self.c.audio_event(self.ev('pickup',trial_id='t2'))
  r=self.c.audio_event(self.ev('release',event_id='late-release'))
  self.assertFalse(r['accepted']);self.assertEqual(self.c.trial,'t2')
 def test_muted_event_never_replayed_on_unmute(self):
  self.c.enable(False);self.c.audio_event(self.ev('pickup'));self.c.enable(True)
  self.assertFalse(self.c.audio_event(self.ev('pickup'))['accepted']);self.assertEqual(self.played(),[])
 def test_domain_separation_and_return_to_live(self):
  self.c.audio_event(self.ev('pickup'));self.assertFalse(self.c.impact(85,'hardware')['accepted'])
  self.assertFalse(self.c.audio_event(self.ev('release',source='operator',time_basis='unix_s',evidence='operator_action'))['accepted'])
  self.c.settings('mode',{'mode':'free'});self.assertTrue(self.c.impact(85,'hardware')['accepted'])
 def test_loop_gain_pitch_and_profile_change_cancel(self):
  p=copy.deepcopy(self.c.profiles.assignments);p['flight'].update(sounds=['cartoon_spring'],gain=.5,pitch=-7,loop=True,max_seconds=2.5)
  self.c.edit_profile({'action':'apply','assignments':p});self.c.audio_event(self.ev('flight_start'))
  command=self.played()[-1];self.assertEqual((command['sound'],command['velocity'],command['pitch'],command['loop'],command['max_seconds']),('cartoon_spring',42,-7,True,2.5))
  self.c.edit_profile({'action':'load','name':'Minimal'});self.assertFalse(self.c.transport);self.assertEqual(self.b.commands[-1],{'cmd':'stop_all'})
 def test_cancelling_replay_prevents_delayed_events(self):
  pending=[]
  class Thread:
   def __init__(self,target,**kw):pending.append(target)
   def start(self):pass
  for stop in [lambda:self.c.stop_playback(),lambda:self.c.settings('mode',{'mode':'free'}),lambda:self.c.enable(False)]:
   self.c.enable(True)
   with patch('bridge.threading.Thread',Thread):self.c.replay_events([self.ev('pickup').to_dict()])
   stop();pending.pop()();self.assertEqual(self.played(),[])
 def test_recorded_midi_does_not_accept_hardware_hits(self):
  with patch('bridge.threading.Thread'):
   self.c.replay([{'t':0,'status':144,'data':[0,85]}])
  self.assertFalse(self.c.impact(85,'hardware')['accepted']);self.assertTrue(self.c.impact(85,'replay')['accepted'])
 def test_stop_retires_late_starts_but_does_not_cancel_other_trial(self):
  self.c.audio_event(self.ev('carry_start'))
  self.c.stop('old-trial');self.assertTrue(self.c.transport)
  self.c.stop('t1');self.assertFalse(self.c.transport)
  self.assertFalse(self.c.audio_event(self.ev('release'))['accepted'])
  self.assertFalse(self.c.audio_event(self.ev('pickup',trial_id='old-trial'))['accepted'])
 def test_recorded_physical_events_block_live_input(self):
  self.c.audio_event(self.ev('pickup',trial_id='previous-live',source='operator',time_basis='unix_s',evidence='operator_action'))
  self.b.commands.clear()
  event=self.ev('impact',source='sensor',evidence='midi_gesture',time_basis='host_monotonic_s')
  pending=[]
  class Thread:
   def __init__(self,target,**kw):pending.append(target)
   def start(self):pass
  with patch('bridge.threading.Thread',Thread):self.c.replay_events([event.to_dict()])
  self.assertFalse(self.c.impact(90,'hardware')['accepted'])
  pending[0]();self.assertEqual(len(self.played()),1)
  self.c.settings('mode',{'mode':'free'});self.assertTrue(self.c.impact(90,'hardware')['accepted'])
  self.assertFalse(self.c.audio_event(self.ev('release',trial_id='previous-live'))['accepted'])
 def test_max_event_identifier_and_invalid_replay(self):
  self.assertTrue(self.c.audio_event(self.ev('pickup',event_id='e'*256))['accepted'])
  for value in [None,[],{},[{}]]:
   with self.assertRaises(ValueError):self.c.replay_events(value)
 def test_full_native_pipe_fails_without_blocking(self):
  read,write=os.pipe();os.set_blocking(write,False)
  try:
   try:
    while True:os.write(write,b'x'*4096)
   except BlockingIOError:pass
   native=object.__new__(Native);native.lock=threading.Lock();native.p=types.SimpleNamespace(poll=lambda:None,stdin=types.SimpleNamespace(fileno=lambda:write))
   before=time.monotonic()
   with self.assertRaises(BlockingIOError):native.send(cmd='play',sound='boing')
   self.assertLess(time.monotonic()-before,.1)
  finally:os.close(read);os.close(write)

if __name__=='__main__':unittest.main()
