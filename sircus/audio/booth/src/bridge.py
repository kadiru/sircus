#!/usr/bin/env python3
"""Independent local Sircus audio bridge. Python stdlib + Apple CoreMIDI/AVAudioEngine."""
import argparse,collections,hashlib,hmac,http.server,ipaddress,json,math,os,secrets,signal,subprocess,threading,time,urllib.parse,sys,uuid
from pathlib import Path
from profiles import Profiles, EVENTS
sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
from sircus.audio.contract import AudioEvent
from sircus.audio.replay import schedule as event_schedule
ROOT=Path(__file__).resolve().parents[1]
MAP=json.loads((ROOT/'assets/map.json').read_text())
SOUNDS={m['name']:m for m in MAP}

def finite(v,lo,hi):
    if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not lo<=v<=hi:raise ValueError(f'number must be {lo}..{hi}')
    return v

def integer(v,lo,hi):
    finite(v,lo,hi)
    if int(v)!=v:raise ValueError('integer required')
    return int(v)

class Native:
    def __init__(self,callback):
        self.lock=threading.Lock();self.callback=callback
        self.p=subprocess.Popen([str(ROOT/'runtime/sircus-native'),str(ROOT)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=open(ROOT/'runtime/native.stderr.log','a'),text=True,bufsize=1)
        os.set_blocking(self.p.stdin.fileno(),False)
        threading.Thread(target=self.read,daemon=True).start()
        threading.Thread(target=self.heartbeat,daemon=True).start()
    def heartbeat(self):
        while self.p.poll() is None:
            try:self.send(cmd="heartbeat")
            except Exception:return
            time.sleep(.5)
    def read(self):
        for line in self.p.stdout:
            try:self.callback(json.loads(line))
            except Exception as e:self.callback({'type':'error','message':str(e)})
        self.callback({'type':'native_exit','message':'Audio/MIDI helper stopped. FX disabled; restart launcher.'})
    def send(self,**msg):
        with self.lock:
            if self.p.poll() is not None:raise RuntimeError('Native audio helper is not running')
            raw=(json.dumps(msg)+'\n').encode()
            if os.write(self.p.stdin.fileno(),raw)!=len(raw):raise RuntimeError('Audio pipe full; cue dropped')
    def close(self):
        try:self.send(cmd='panic');self.p.stdin.close();self.p.wait(timeout=3)
        except Exception:self.p.terminate()

class Controller:
    def __init__(self,backend=None,clock=time.monotonic,config=None,profile_path=None):
        self.clock=clock;self.lock=threading.RLock();self.backend=backend
        self.config=config or json.loads((ROOT/'config.json').read_text())
        self.enabled=False;self.volume=finite(self.config['volume'],0,1);self.mode=self.config['mode'];self.route=self.config['route']
        self.mapping=dict(self.config['mapping']);self.inputs=[];self.selected=None;self.connected=False
        self.ready=False;self.last_midi=0;self.midi_status=[];self.meter=0;self.current_peak=0;self.block_ns=0;self.last_meter=0;self.last_error=None;self.transport=False
        self.trial=None;self.armed_until=0;self.impact_count=0;self.variant=0;self.sequence=0;self.monitor=collections.deque(maxlen=600)
        self.seen=collections.OrderedDict();self.released=collections.OrderedDict();self.duplicate_last={};self.record_file=None;self.record_start=0
        self.replay_generation=0;self.logic_confirmed=False
        self.profiles=Profiles(SOUNDS,profile_path);self.loop_kind=None;self.domain=None;self.terminal=False;self.board_seen=False
        self.retired_trials=set();self.playback_source=None
    def log(self,kind,**kw):
        self.sequence+=1;r={'seq':self.sequence,'time':time.time(),'kind':kind,**kw};self.monitor.append(r);return r
    def send(self,**kw):
        if self.backend:self.backend.send(**kw)
    def status(self):
        with self.lock:
            return {'fx_enabled':self.enabled,'volume':self.volume,'mode':self.mode,'route':self.route,'audio_ready':self.ready,
                'audio_meter_peak':self.meter,'audio_current_peak':self.current_peak,'audio_block_time_ns':self.block_ns,'meter_age_s':None if not self.last_meter else round(self.clock()-self.last_meter,2),
                'last_midi_age_s':None if not self.last_midi else round(self.clock()-self.last_midi,2),'midi_status':self.midi_status,'inputs':self.inputs,'selected_input':self.selected,'connected':self.connected,
                'transport':self.transport,'trial_id':self.trial,'armed':self.clock()<self.armed_until,'impact_count':self.impact_count,
                'mapping':self.mapping,'last_error':self.last_error,'monitor_seq':self.sequence,'recording':self.record_file is not None,
                'virtual_output':'Sircus FX','output_channel':16,'logic_commissioning_acknowledged':self.logic_confirmed,'logic_route_verified':False,
                'profile':self.profiles.state(),'loop_event':self.loop_kind,'domain':self.domain,'playback_source':self.playback_source,
                'events':[{'key':k,'title':v[0],'description':v[1]} for k,v in EVENTS.items()]}
    def disable(self,reason='fx_off'):
        self.enabled=False;self.transport=False;self.loop_kind=None;self.armed_until=0;self.replay_generation+=1;self.playback_source=None
        self.send(cmd='panic');self.log('mute',reason=reason)
    def enable(self,value):
        if not isinstance(value,bool):raise ValueError('enabled must be boolean')
        with self.lock:
            if not value:self.disable();return
            if self.route=='native' and not self.ready:raise ValueError('Native audio unavailable; see status/native.stderr.log')
            if self.route=='logic' and not self.logic_confirmed:raise ValueError('Acknowledge manual Logic commissioning before enabling that route; native route is verified')
            self.enabled=True;self.send(cmd='volume',value=self.volume);self.send(cmd='enable',value=True);self.log('fx',enabled=True)
    def play(self,sound,velocity=85,loop=False,pitch=0,max_seconds=8):
        if sound not in SOUNDS:raise ValueError('unknown sound')
        velocity=integer(velocity,1,127)
        if not self.enabled:return False
        if self.route=='logic' and (pitch or loop):raise ValueError('Use native playback for event pitch and loops; Logic sampler route needs separate commissioning')
        self.send(cmd='play',sound=sound,velocity=velocity,loop=loop,pitch=pitch,max_seconds=max_seconds)
        self.log('sound',sound=sound,velocity=velocity,trial_id=self.trial,loop=loop,pitch=pitch)
        return True
    def stop_transport(self):
        self.transport=False;self.loop_kind=None;self.send(cmd='stop_transport')
    def cue(self,key,velocity=85):
        sound,row=self.profiles.choose(key)
        if sound is None:return None
        v=round(velocity*row['gain'])
        if v<1:return None
        looping=row['loop']
        self.play(sound,v,looping,row['pitch'],row['max_seconds'])
        if looping:self.transport=True;self.loop_kind=key
        return sound
    def preview(self,d):
        with self.lock:
            key=d.get('event')
            if key not in EVENTS:raise ValueError('unknown preview event')
            if not self.enabled:return {'accepted':False,'reason':'fx_disabled'}
            self.stop_playback()
            sound=self.cue(key)
            self.log('preview',event=key,source='manual_sound_test',sound=sound)
            return {'accepted':True,'sound':sound}
    def edit_profile(self,d):
        with self.lock:
            action=d.get('action')
            if action=='apply':self.profiles.apply(d.get('assignments'))
            elif action=='load':self.profiles.load(d.get('name'))
            elif action=='save':self.profiles.save(d.get('name'),d.get('assignments'))
            else:raise ValueError('unknown profile action')
            self.stop_playback()
            return self.profiles.state()
    def stop_playback(self):
        with self.lock:
            self.replay_generation+=1;self.playback_source=None;self.stop_transport();self.send(cmd='stop_all');self.armed_until=0
    def stop(self,trial=None):
        with self.lock:
            if trial is not None and (not isinstance(trial,str) or not 1<=len(trial)<=256):raise ValueError('invalid trial_id')
            target=trial or self.trial
            if target and target not in self.retired_trials and len(self.retired_trials)>=4096:
                self.disable('trial_capacity');raise ValueError('trial capacity reached; restart audio service')
            if target:self.retired_trials.add(target)
            if trial is None or self.trial in (None,trial):self.stop_playback();self.terminal=True
            return {'accepted':True}
    def unique(self,event):
        eid=event.get('event_id')
        if eid is None:return True
        if not isinstance(eid,str) or not 1<=len(eid)<=272:raise ValueError('invalid event_id')
        key=(event.get('trial_id'),eid)
        if key in self.seen:return False
        if len(self.seen)>=4096:return False
        self.seen[key]=self.clock()
        return True
    def event(self,e):
        if e.get('schema') is not None:return self.audio_event(AudioEvent.from_dict(e))
        kind=e.get('type')
        if kind in ('test','audition'):
            with self.lock:
                if not self.enabled:return {'accepted':False,'reason':'fx_disabled'}
                if kind=='test':
                    sound=e.get('sound');velocity=integer(e.get('velocity',85),1,127)
                    if sound not in SOUNDS:raise ValueError('unknown sound')
                    self.stop_playback();self.play(sound,velocity)
                else:self.start_audition()
                return {'accepted':True}
        aliases={'transport_start':'carry_start','transport_stop':'carry_stop','arrive':'carry_stop',
                 'capture':'outcome','miss':'outcome'}
        event_type=aliases.get(kind,kind)
        outcome={'capture':'captured','miss':'missed'}.get(kind,'unknown')
        velocity=integer(e.get('velocity',85),1,127)
        source=e.get('source','operator')
        # Legacy manual actions are explicit operator requests, not sensor observations.
        if source not in ('operator','simulation'):raise ValueError('sensor evidence requires a versioned AudioEvent')
        ev=AudioEvent(event_id=e.get('event_id') or uuid.uuid4().hex,trial_id=e.get('trial_id') or 'manual-session',
                      source=source,event_type=event_type,contact_object='unknown',timestamp=0 if source=='simulation' else time.time(),
                      time_basis='simulation_elapsed_s' if source=='simulation' else 'unix_s',clock_id='ghost-booth-manual',
                      evidence='model_prediction' if source=='simulation' else 'operator_label' if event_type=='outcome' else 'operator_action',
                      provenance='Ghost Booth explicit operator event' if source=='operator' else 'explicit synthetic Ghost Booth event',
                      outcome=outcome,confirmed=source=='operator' and outcome!='unknown',intensity=velocity/127)
        return self.audio_event(ev)
    def audio_event(self,ev,from_replay=False):
        with self.lock:
            kind=ev.event_type;trial=ev.trial_id;domain='simulation' if ev.source=='simulation' else 'physical'
            if self.playback_source and not from_replay and kind!='stop':return {'accepted':False,'reason':'replay_active'}
            velocity=max(1,min(127,round((ev.intensity if ev.intensity is not None else 85/127)*127)))
            start=kind in ('pickup','carry_start','release')
            if trial in self.retired_trials:return {'accepted':False,'reason':'stale_trial'}
            if self.trial and trial!=self.trial and not start:return {'accepted':False,'reason':'stale_trial'}
            if self.trial==trial and self.domain and self.domain!=domain:return {'accepted':False,'reason':'source_domain_mismatch'}
            if kind=='release' and trial in self.released:return {'accepted':False,'reason':'trial_already_released'}
            if not self.unique({'event_id':ev.source+':'+ev.event_id,'trial_id':trial}):return {'accepted':False,'reason':'duplicate_or_capacity'}
            if not self.enabled:return {'accepted':False,'reason':'fx_disabled'}
            if self.trial!=trial:
                if self.trial:
                    if len(self.retired_trials)>=4096:return {'accepted':False,'reason':'trial_capacity'}
                    self.retired_trials.add(self.trial)
                self.stop_transport();self.trial=trial;self.domain=domain;self.terminal=False;self.board_seen=False;self.impact_count=0
            if kind=='stop':
                self.stop_transport();self.armed_until=0;self.terminal=True;self.replay_generation+=1;self.send(cmd='stop_all')
            elif self.terminal:return {'accepted':False,'reason':'trial_ended'}
            elif kind=='pickup':self.cue('pickup',velocity)
            elif kind in ('carry_start','flight_start'):
                self.stop_transport();self.cue('carry' if kind=='carry_start' else 'flight',velocity)
            elif kind in ('carry_stop','flight_stop'):self.stop_transport()
            elif kind=='release':
                if trial:self.released[trial]=True
                self.stop_transport();self.impact_count=0;self.armed_until=self.clock()+self.mapping['trial_window_s'];self.cue('release',velocity)
            elif kind=='outcome':
                self.stop_transport();self.armed_until=0;self.terminal=True
                if ev.outcome!='unknown' and (ev.source=='simulation' or ev.confirmed):self.cue('capture' if ev.outcome=='captured' else 'miss',velocity)
            elif kind=='impact':
                first=ev.contact_object=='board' and not self.board_seen
                if ev.contact_object=='board':self.board_seen=True
                result=self.impact(velocity,ev.source,first,domain=='simulation',from_replay)
                if not result['accepted']:return result
            self.log('event',event=kind,event_id=ev.event_id,trial_id=trial,source=ev.source,evidence=ev.evidence,
                     time_basis=ev.time_basis,timestamp=ev.timestamp,contact_object=ev.contact_object,outcome=ev.outcome,provenance=ev.provenance)
            return {'accepted':True}
    def impact(self,velocity,source='midi',first_board=False,simulation=False,replaying=False):
        if not self.enabled:return {'accepted':False,'reason':'fx_disabled'}
        if self.playback_source and source in ('hardware','mock','midi'):return {'accepted':False,'reason':'replay_active'}
        if not simulation and self.domain=='simulation':return {'accepted':False,'reason':'simulation_active'}
        if self.domain=='recorded_midi' and source!='replay':return {'accepted':False,'reason':'recorded_replay_active'}
        if not (simulation or replaying) and source!='replay' and self.mode=='trial' and self.clock()>=self.armed_until:return {'accepted':False,'reason':'trial_not_armed'}
        self.stop_transport()
        first=first_board or (self.mode=='trial' and self.impact_count==0)
        sound=self.cue('first_impact' if first else 'impact',velocity)
        self.variant+=1;self.impact_count+=1
        self.log('impact',source=source,ordinal=self.impact_count,heuristic='first_after_release' if first and not first_board else None,trial_id=self.trial)
        return {'accepted':True,'sound':sound}
    def midi(self,m,source='hardware'):
        with self.lock:
            status=integer(m.get('status'),128,239);data=m.get('data')
            if not isinstance(data,list) or len(data)!=(1 if status&240 in (192,208) else 2):raise ValueError('invalid MIDI data length')
            data=[integer(v,0,127) for v in data]
            port=m.get('input');kind=status&240;channel=(status&15)+1
            self.last_midi=self.clock() if source=='hardware' else self.last_midi
            self.log('midi',source=source,input=port,status=status,channel=channel,data=data)
            if self.record_file and source=='hardware':
                self.record_file.write(json.dumps({'t':self.clock()-self.record_start,'status':status,'data':data,'input':port})+'\n');self.record_file.flush()
            if source=='hardware' and (not self.connected or port!=self.selected):return
            if kind==128 or (kind==144 and data[1]==0):
                self.duplicate_last={k:v for k,v in self.duplicate_last.items() if not (k[0]==port and k[1]&15==status&15 and k[2][0]==data[0])}
                return
            if kind!=144:return # CCs and note-offs are monitored, never forwarded
            if channel!=self.mapping['channel'] or data[0]!=self.mapping['impact_note'] or data[1]<self.mapping['min_velocity']:return
            key=(port,status,tuple(data));now=self.clock();window=self.mapping['exact_duplicate_ms']/1000
            prev=self.duplicate_last.get(key);self.duplicate_last[key]=now
            if prev is not None and window>0 and 0<=now-prev<window:
                self.log('suppressed',reason='identical_message_window',window_ms=window*1000);return
            if len(self.duplicate_last)>512:self.duplicate_last={key:now}
            velocity=max(1,min(127,round(127*(data[1]/127)**self.mapping['velocity_gamma'])))
            return self.impact(velocity,source)
    def native(self,m):
        with self.lock:
            kind=m.get('type')
            if kind=='midi':self.midi(m);return
            if kind=='ready':self.ready=m['audio_ready'];self.midi_status=m['midi_status']
            elif kind=='inputs':self.inputs=m['inputs']
            elif kind=='meter':
                self.meter=m['peak'];self.current_peak=m.get('current_peak',0);self.block_ns=m.get('block_time_ns',0);self.last_meter=self.clock();self.ready=m['audio_ready']
                if self.enabled and self.route=='native' and not self.ready:self.disable('audio_state_lost')
                return
            elif kind=='connection':self.connected=m['connected'];self.disable('input_connection_changed')
            elif kind=='watchdog':self.disable('audio_watchdog')
            elif kind=='loop_stopped':self.transport=False;self.loop_kind=None
            elif kind in ('error','native_exit'):
                self.last_error=m.get('message');self.enabled=False;self.transport=False;self.loop_kind=None;self.armed_until=0
                self.replay_generation+=1
                if kind=='native_exit':self.ready=False;self.connected=False
                else:
                    try:self.send(cmd='panic')
                    except Exception:pass
            self.log('native',message=m)
    def settings(self,kind,d):
        with self.lock:
            if kind=='fx':self.enable(d.get('enabled'))
            elif kind=='panic':self.disable('audio_panic')
            elif kind=='volume':
                self.volume=finite(d.get('volume'),0,1);self.send(cmd='volume',value=self.volume)
            elif kind=='mode':
                if d.get('mode') not in ['free','trial']:raise ValueError('mode must be free or trial')
                self.stop_playback();self.mode=d['mode'];self.impact_count=0
                if self.trial:self.retired_trials.add(self.trial)
                self.trial=None;self.domain=None;self.terminal=False
            elif kind=='input':
                value=d.get('id')
                if value is not None and not any(r['id']==value and r['online'] for r in self.inputs):raise ValueError('choose an online listed hardware MIDI input')
                self.disable('input_selection');self.selected=value;self.connected=False;self.send(cmd='input',value=value)
            elif kind=='mapping':
                m=dict(self.mapping)
                for key,value in d.items():
                    if key not in m:raise ValueError('unknown mapping field')
                    if key in ['impact_note','channel','min_velocity']:value=integer(value,1 if key!='impact_note' else 0,16 if key=='channel' else 127)
                    else:value=finite(value,*{'exact_duplicate_ms':(0,20),'trial_window_s':(.1,120),'velocity_gamma':(.1,3)}[key])
                    m[key]=value
                self.mapping=m;self.duplicate_last={};self.log('mapping',mapping=m)
            elif kind=='route':
                if d.get('route') not in ['native','logic']:raise ValueError('route must be native or logic')
                if d['route']=='logic' and d.get('commissioning_acknowledged') is not True:raise ValueError('Logic route requires explicit commissioning_acknowledged:true for supervised manual routing/mute tests')
                self.disable('route_change');self.route=d['route'];self.logic_confirmed=d.get('commissioning_acknowledged') is True;self.send(cmd='route',value=self.route)
            elif kind=='record':
                if self.record_file:self.record_file.close();self.record_file=None
                if d.get('enabled') is True:
                    p=ROOT/'runtime'/('midi-'+time.strftime('%Y%m%d-%H%M%S')+'-'+secrets.token_hex(2)+'.jsonl')
                    self.record_file=p.open('x');self.record_start=self.clock();self.log('record',path=p.name)
            else:raise ValueError('unknown setting')
            return self.status()
    def replay(self,rows):
        if not isinstance(rows,list) or len(rows)>10000:raise ValueError('replay must be at most 10000 rows')
        last=0
        for row in rows:
            if not isinstance(row,dict):raise ValueError('replay rows must be objects')
            t=finite(row.get('t'),0,600)
            if t<last:raise ValueError('replay times must be monotonic')
            last=t
            st=integer(row.get('status'),128,239);data=row.get('data')
            if not isinstance(data,list) or len(data)!=(1 if st&240 in (192,208) else 2):raise ValueError('invalid replay MIDI')
            for n in data:integer(n,0,127)
        with self.lock:
            if not self.enabled:raise ValueError('enable FX before replay')
            if self.trial:self.retired_trials.add(self.trial)
            self.stop_playback();self.trial=None;self.domain='recorded_midi';self.playback_source='midi';g=self.replay_generation
        def run():
            start=self.clock()
            for row in rows:
                while self.clock()-start<row['t']:
                    if self.replay_generation!=g:return
                    time.sleep(max(0,min(.005,row['t']-(self.clock()-start))))
                with self.lock:
                    if self.replay_generation!=g or not self.enabled:return
                    self.midi(row,'replay')
        threading.Thread(target=run,daemon=True).start()
        return {'accepted':True,'rows':len(rows),'source':'recorded_or_synthetic_replay'}
    def replay_events(self,rows):
        if not isinstance(rows,list) or not 1<=len(rows)<=4096:raise ValueError('replay needs 1–4096 events')
        events=[AudioEvent.from_dict(row) for row in rows]
        scheduled=event_schedule(events)
        with self.lock:
            if not self.enabled:raise ValueError('enable FX before replay')
            if self.trial and self.trial!=events[0].trial_id:self.retired_trials.add(self.trial)
            self.stop_playback();self.playback_source='events';g=self.replay_generation
            self.trial=events[0].trial_id;self.domain='simulation' if events[0].source=='simulation' else 'physical'
            self.terminal=False;self.board_seen=False;self.impact_count=0;self.armed_until=0
            # Explicit replay may repeat this trace, without reopening other retired trials.
            self.retired_trials.discard(self.trial);self.released.pop(self.trial,None)
            self.seen=collections.OrderedDict((k,v) for k,v in self.seen.items() if k[0]!=self.trial)
        def run():
            start=self.clock()
            for delay,event in scheduled:
                while self.clock()-start<delay:
                    if self.replay_generation!=g:return
                    time.sleep(.005)
                with self.lock:
                    if self.replay_generation!=g or not self.enabled:return
                    self.audio_event(event,from_replay=True)
            with self.lock:
                if self.replay_generation==g:self.stop_transport()
        threading.Thread(target=run,daemon=True).start()
        return {'accepted':True,'rows':len(scheduled),'domain':self.domain}
    def start_audition(self):
        self.stop_playback();g=self.replay_generation
        def run():
            for m in MAP:
                with self.lock:
                    if g!=self.replay_generation or not self.enabled:return
                    self.play(m['name'],85)
                end=self.clock()+m['seconds']+.15
                while self.clock()<end:
                    if g!=self.replay_generation:return
                    time.sleep(.01)
        threading.Thread(target=run,daemon=True).start()

class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def reply(self,data,status=200):
        b=json.dumps(data).encode();self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(b)));self.end_headers();self.wfile.write(b)
    def authorized(self):
        # Localhost panel is same-origin; no browser CORS permission and no form POSTs.
        host=self.headers.get('Host','')
        allowed={f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}',f'{self.server.server_address[0]}:{self.server.server_port}'}
        if host not in allowed:return False
        origin=self.headers.get('Origin')
        if origin and origin!='http://'+host:return False
        local=ipaddress.ip_address(self.client_address[0]).is_loopback
        if local:return True
        auth=self.headers.get('Authorization','')
        return hmac.compare_digest(auth,'Bearer '+self.server.token)
    def do_GET(self):
        if not self.authorized():return self.reply({'error':'unauthorized'},401)
        path=urllib.parse.urlparse(self.path)
        if path.path=='/api/status':return self.reply(self.server.c.status())
        if path.path=='/api/monitor':
            try:after=int(urllib.parse.parse_qs(path.query).get('after',['0'])[0])
            except ValueError:return self.reply({'error':'invalid cursor'},400)
            with self.server.c.lock:return self.reply({'events':[r for r in self.server.c.monitor if r['seq']>after],'seq':self.server.c.sequence})
        if path.path=='/api/sounds':return self.reply(MAP)
        files={'/':('web/index.html','text/html'),'/app.js':('web/app.js','text/javascript'),'/style.css':('web/style.css','text/css')}
        for m in MAP:files['/assets/'+m['file']]=('assets/'+m['file'],'audio/wav')
        if path.path not in files:return self.reply({'error':'not found'},404)
        f,mime=files[path.path];b=(ROOT/f).read_bytes();self.send_response(200);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(b)));self.send_header('X-Content-Type-Options','nosniff');self.send_header('Content-Security-Policy',"default-src 'self'; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'");self.end_headers();self.wfile.write(b)
    def do_POST(self):
        if not self.authorized():return self.reply({'error':'unauthorized'},401)
        if self.headers.get('Content-Type','').split(';')[0]!='application/json':return self.reply({'error':'application/json required'},415)
        try:
            n=int(self.headers.get('Content-Length','0'))
            if not 0<n<=1024*1024:raise ValueError('body must be 1..1048576 bytes')
            d=json.loads(self.rfile.read(n))
            if not isinstance(d,dict):raise ValueError('object required')
            c=self.server.c;path=urllib.parse.urlparse(self.path).path
            if path=='/api/event':result=c.event(d)
            elif path=='/api/profile':result=c.edit_profile(d)
            elif path=='/api/preview':result=c.preview(d)
            elif path=='/api/replay-events':result=c.replay_events(d.get('events'))
            elif path=='/api/stop':
                result=c.stop(d.get('trial_id'))
            elif path=='/api/mock-midi':result=c.midi(d,'mock') or {'accepted':False,'reason':'not_mapped'}
            elif path=='/api/replay':result=c.replay(d.get('events'))
            elif path.startswith('/api/'):result=c.settings(path[5:],d)
            else:raise ValueError('unknown endpoint')
            self.reply(result)
        except (ValueError,TypeError,KeyError) as e:self.reply({'error':str(e)},400)
        except Exception as e:
            try:self.server.c.disable('backend_error')
            except Exception:pass
            self.reply({'error':str(e)},503)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--host',default='127.0.0.1');ap.add_argument('--port',type=int,default=8766);ap.add_argument('--allow-private-lan',action='store_true');ap.add_argument('--mock-backend',action='store_true');args=ap.parse_args()
    address=ipaddress.ip_address(args.host)
    if not address.is_loopback and not (args.allow_private_lan and address.version==4 and (address in ipaddress.ip_network('10.0.0.0/8') or address in ipaddress.ip_network('172.16.0.0/12') or address in ipaddress.ip_network('192.168.0.0/16'))):ap.error('bind to localhost or an explicit RFC1918 interface with --allow-private-lan; no wildcard/public binding')
    (ROOT/'runtime').mkdir(exist_ok=True)
    token_path=ROOT/'runtime/lan-token'
    token=os.environ.get('SIRCUS_TOKEN')
    if not token:
        if not token_path.exists():
            fd=os.open(token_path,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
            with os.fdopen(fd,'w') as f:f.write(secrets.token_urlsafe(32)+'\n')
        token=token_path.read_text().strip()
    if len(token)<24:ap.error('shared token must have at least 24 characters')
    c=Controller(profile_path=ROOT/'runtime/profiles.json');server=http.server.ThreadingHTTPServer((args.host,args.port),Handler);server.c=c;server.token=token
    if args.mock_backend:
        class Mock:
            def send(self,**kw):c.log('mock_backend',command=kw)
            def close(self):pass
        c.backend=Mock();c.ready=True;c.log('warning',message='Mock backend: NO audio/MIDI output')
    else:c.backend=Native(c.native)
    print(f'Sircus sound panel: http://{args.host}:{args.port} — FX OFF',flush=True)
    print(f'LAN token stored at {token_path}; never print it in logs or put it in browser URLs.',flush=True)
    def stop(*_):
        try:c.disable('shutdown');c.backend.close()
        finally:os._exit(0)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:server.serve_forever(poll_interval=.2)
    finally:c.backend.close();server.server_close()
if __name__=='__main__':main()
