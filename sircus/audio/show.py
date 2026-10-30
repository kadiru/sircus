"""Four-ball audio routing and bounded synthetic playback; never robot control.

Ball slots are operator configuration, not detected colors. MIDI endpoint/channel/
note bindings are explicit and identify a ball only, never what it struck.
"""
from dataclasses import replace
import copy
import json
import math
from pathlib import Path
import re
import threading
import time
import uuid
from .contract import AudioEvent

SCHEMA = 'sircus.show-audio.v1'
BALL_IDS = ('ball-pink','ball-purple','ball-green','ball-gray')
CONTACTS = ('surface','bucket','collision','unknown','carry')
CATALOG = tuple(row['name'] for row in json.loads((Path(__file__).parent/'booth/assets/map.json').read_text()))


def default_config():
    colors=('#ef78ba','#a47de3','#67b875','#9da3aa')
    cues=(('cartoon_boing','cartoon_plop','pinball','bone_low'),
          ('boing','bone_high','cartoon_spring','bone_high'),
          ('cartoon_spring','eerie_bell','pinball','cartoon_squeak'),
          ('cartoon_squeak','cartoon_plop','cartoon_boing','pinball'))
    return {'schema':SCHEMA,'impact_window_s':30.,'balls':[{'ball_id':bid,'label':bid[5:].title(),
        'color':color,'midi_binding':None,'sounds':dict(zip(CONTACTS,(*sounds,'drum_roll')))}
        for bid,color,sounds in zip(BALL_IDS,colors,cues)]}


def validate_config(value):
    if not isinstance(value,dict) or set(value)!={'schema','balls','impact_window_s'} or value['schema']!=SCHEMA:
        raise ValueError('Versioned four-ball audio configuration required')
    window=value['impact_window_s']
    if isinstance(window,bool) or not isinstance(window,(int,float)) or not math.isfinite(window) or not 1<=window<=30:
        raise ValueError('Impact window must be 1..30 seconds')
    if not isinstance(value['balls'],list) or len(value['balls'])!=4:
        raise ValueError('Exactly four ball slots required')
    ids=set();bindings=set()
    for ball in value['balls']:
        if not isinstance(ball,dict) or set(ball)!={'ball_id','label','color','midi_binding','sounds'}:
            raise ValueError('Invalid ball configuration fields')
        if ball['ball_id'] not in BALL_IDS or ball['ball_id'] in ids:raise ValueError('Four unique stable ball IDs required')
        ids.add(ball['ball_id'])
        if not isinstance(ball['label'],str) or not 1<=len(ball['label'].strip())<=40:raise ValueError('Ball label must be 1..40 characters')
        if not isinstance(ball['color'],str) or not re.fullmatch(r'#[0-9a-fA-F]{6}',ball['color']):raise ValueError('Use #RRGGBB colors')
        if not isinstance(ball['sounds'],dict) or set(ball['sounds'])!=set(CONTACTS) or any(v not in CATALOG for v in ball['sounds'].values()):
            raise ValueError('Sound assignments must use the catalog for all five cue types')
        binding=ball['midi_binding']
        if binding is not None:
            if not isinstance(binding,dict) or set(binding)!={'input_id','channel','note'}:raise ValueError('Explicit input/channel/note mapping required')
            for key,lo,hi in [('input_id',-2**31,2**31-1),('channel',1,16),('note',0,127)]:
                if type(binding[key]) is not int or not lo<=binding[key]<=hi:raise ValueError('Invalid MIDI binding '+key)
            if binding['input_id']==0:raise ValueError('MIDI endpoint identity must be nonzero')
            identity=tuple(binding[k] for k in ('input_id','channel','note'))
            if identity in bindings:raise ValueError('A MIDI identity cannot refer to two balls')
            bindings.add(identity)
    return copy.deepcopy(value)


class BallRouter:
    def __init__(self, config=None, *, clock=time.monotonic):
        self.clock=clock;self.config=validate_config(config or default_config())
        self.trials={};self.retired=set();self.latest={}

    def configure(self,value):
        checked=validate_config(value)
        retired=list(self.trials)
        self.retired.update(retired);self.trials.clear();self.latest.clear();self.config=checked
        return retired

    def begin(self,trial_id,ball_id,source):
        if ball_id not in BALL_IDS:raise ValueError('Unknown ball ID')
        if trial_id in self.retired:raise ValueError('Retired ball trial')
        if trial_id in self.trials:
            row=self.trials[trial_id]
            if row['ball_id']!=ball_id or row['source_domain']!=source:raise ValueError('Trial identity/domain cannot change')
            return row
        if len(self.trials)>=256 or len(self.retired)>=4096:raise ValueError('Ball trial capacity reached; reset through a fresh process')
        old=self.latest.get(ball_id)
        if old is not None and old!=trial_id:self.cancel(old)
        ball=next(b for b in self.config['balls'] if b['ball_id']==ball_id)
        row={'ball_id':ball_id,'source_domain':source,'ball':copy.deepcopy(ball),'released_at':None,'last_clocks':{}}
        self.trials[trial_id]=row;self.latest[ball_id]=trial_id
        return row

    def cancel(self,trial_id):
        self.retired.add(trial_id)
        row=self.trials.pop(trial_id,None)
        if row and self.latest.get(row['ball_id'])==trial_id:self.latest.pop(row['ball_id'],None)

    def route(self,event):
        if event.ball_id is None:return event
        domain='simulation' if event.source=='simulation' else 'physical'
        row=self.trials.get(event.trial_id)
        if row is None:
            if event.event_type not in ('pickup','carry_start','release') and event.evidence!='operator_action':
                raise ValueError('Ball event has no explicit trial association')
            row=self.begin(event.trial_id,event.ball_id,domain)
        if row['ball_id']!=event.ball_id or row['source_domain']!=domain:raise ValueError('Ball event identity/domain mismatch')
        previous=row['last_clocks'].get(event.clock_id)
        if previous is not None and event.timestamp<previous:raise ValueError('Delayed event from an older source timestamp')
        row['last_clocks'][event.clock_id]=event.timestamp
        if event.event_type=='release':row['released_at']=self.clock()
        if event.event_type=='stop':self.cancel(event.trial_id);return event
        key=None
        if event.event_type=='carry_start':key='carry'
        elif event.event_type=='impact':
            obj=event.contact_object
            key='surface' if obj in ('board','surface') else 'bucket' if obj=='bucket' or obj.startswith('bucket-') else 'collision' if obj in ('ball','other_ball') and event.other_ball_id else 'unknown'
        return replace(event,sound_id=row['ball']['sounds'][key]) if key else event

    def midi_identity(self,input_id,channel,note):
        for ball in self.config['balls']:
            if ball['midi_binding']=={'input_id':input_id,'channel':channel,'note':note}:
                trial=self.latest.get(ball['ball_id']);row=self.trials.get(trial)
                if row and row['released_at'] is not None and 0<=self.clock()-row['released_at']<=self.config['impact_window_s']:
                    return trial,ball['ball_id']
        return None


class ShowAudio:
    """Root dashboard owns this object; methods are audio-only and never await I/O."""
    def __init__(self,observer,*,config_path=None,clock=time.monotonic):
        self.observer=observer;self.clock=clock;self.lock=threading.RLock();self.config_lock=threading.Lock();self.config_path=Path(config_path) if config_path else None
        config=json.loads(self.config_path.read_text()) if self.config_path and self.config_path.is_file() else default_config()
        self.router=BallRouter(config,clock=clock);self._generation=0;self._cancel=threading.Event();self._replay_trial=None;self._thread=None;self._schedule_pending=False
        self._last_error=None
        observer.show=self
        if hasattr(observer,'configure_show_remote'):observer.configure_show_remote(self.router.config)

    def status(self):
        with self.lock:
            return {'schema':SCHEMA,'config':copy.deepcopy(self.router.config),'catalog':list(CATALOG),
                    'replay_running':bool(self._schedule_pending and self._thread and self._thread.is_alive() and not self._cancel.is_set()),'replay_trial_id':self._replay_trial,
                    'error':self._last_error,'capabilities':{'physical_color_detection':False,'contact_detection':False,
                    'native_midi_inputs':'explicit multi-endpoint bindings supported in source; physical BLE commissioning outstanding',
                    'multiball_event_routing':True,'bindings_default':'unbound; operator must verify endpoint identity',
                    'carry_sound':'percussive loop through carry and 2-second settling; bounded to 20s',
                    'outcomes':'unknown stays silent; ODD MIDI never proves capture/miss'},
                    'active_ball_trials':[{'trial_id':k,'ball_id':v['ball_id'],'released':v['released_at'] is not None} for k,v in self.router.trials.items()]}

    def configure(self,config):
        checked=validate_config(config)
        # Serialize operator configuration, but never put disk I/O under the
        # lifecycle lock. Robot hooks use try-locks and may drop optional cues.
        with self.config_lock:
            self.cancel_replay()
            if self.config_path:
                self.config_path.parent.mkdir(parents=True,exist_ok=True)
                temp=self.config_path.with_suffix('.tmp');temp.write_text(json.dumps(checked,indent=2)+'\n');temp.replace(self.config_path)
            with self.lock:
                for trial in self.router.configure(checked):self.observer.stop(trial)
                self.observer.configure_show_remote(checked)
        return self.status()

    def decorate(self,value):
        event=AudioEvent.from_dict(value)
        if not self.lock.acquire(blocking=False):return None
        try:return self.router.route(event).to_dict()
        finally:self.lock.release()

    def cancel_trial(self,trial_id):
        # AudioObserver's retired set is authoritative even when this cosmetic
        # router snapshot cannot be updated immediately. No physical Stop waits.
        if not self.lock.acquire(blocking=False):return False
        try:self.router.cancel(trial_id);return True
        finally:self.lock.release()

    def begin_ball_trial(self,trial_id,ball_id,source='physical'):
        with self.lock:self.router.begin(trial_id,ball_id,source)

    def emit_event(self,event):
        self.observer.emit(event.to_dict() if isinstance(event,AudioEvent) else event)

    def _finish_cancel_locked(self,cancel,trial):
        # A prior playback's finally must never clear or stop its replacement.
        if self._cancel is not cancel or self._replay_trial!=trial:return
        self._generation+=1;self._replay_trial=None;self._schedule_pending=False
        if trial:self.observer.stop(trial)

    def cancel_replay(self,*,blocking=True):
        # Transport/error paths may be called from optional robot hooks. Setting
        # the event is enough to stop scheduling; its owner completes retirement.
        cancel=self._cancel;cancel.set()
        if not self.lock.acquire(blocking=blocking):return None
        try:
            if self._cancel is cancel:self._finish_cancel_locked(cancel,self._replay_trial)
        finally:self.lock.release()
        return self.status() if blocking else None

    def start_replay(self,result,trial_id,ball_id,*,carry_seconds=1.,settle_seconds=2.):
        from .sources import versioned_simulation_events
        if ball_id not in BALL_IDS:raise ValueError('Unknown ball ID')
        if type(carry_seconds) not in (int,float) or not math.isfinite(carry_seconds) or not 0<=carry_seconds<=15 or settle_seconds!=2:
            raise ValueError('Use carry 0..15s and the explicit two-second settling segment')
        contacts=versioned_simulation_events(result,trial_id)
        if len(contacts)>4096 or any(e.timestamp>60 for e in contacts):raise ValueError('Replay exceeds bounded audio schedule')
        self.cancel_replay()
        readiness=self.observer.playback_readiness()
        if not readiness['ready']:
            self._last_error='Audio replay not scheduled: enable a compatible audio service first'
            return self.status()
        self._last_error=None
        source_trial_id=trial_id
        playback_trial_id='show-playback-'+uuid.uuid4().hex
        with self.lock:
            # Mute/error must either invalidate this permit or observe this newly
            # assigned Event. A stale readiness snapshot can never revive cues.
            with self.observer.playback_creation(readiness['epoch']) as ready:
                if not ready:
                    self._last_error='Audio replay not scheduled: readiness changed during creation'
                    return self.status()
                self._generation+=1;generation=self._generation;self._cancel=threading.Event();cancel=self._cancel;self._replay_trial=playback_trial_id;self._schedule_pending=True
            self.router.begin(playback_trial_id,ball_id,'simulation')
            release_at=carry_seconds+settle_seconds
            def lifecycle(kind,t,clock):
                return AudioEvent(uuid.uuid4().hex,playback_trial_id,'simulation',kind,'unknown',t,'simulation_elapsed_s',clock,
                    'Synthetic show choreography; not robot movement, physical sensing or measured carry timing','model_prediction',ball_id=ball_id,source_trial_id=source_trial_id)
            schedule=[(0.,lifecycle('carry_start',0.,trial_id+'/choreography')),
                      (release_at,lifecycle('carry_stop',release_at,trial_id+'/choreography')),
                      (release_at,lifecycle('release',0.,trial_id))]
            schedule += [(release_at+e.timestamp,replace(e,ball_id=ball_id,trial_id=playback_trial_id,source_trial_id=source_trial_id,source_event_id=e.event_id)) for e in contacts if e.event_type!='outcome' or e.outcome!='unknown']
            def run():
                start=self.clock()
                try:
                    for delay,event in schedule:
                        if cancel.wait(max(0.,start+delay-self.clock())):return
                        with self.lock:
                            if generation!=self._generation or cancel.is_set():return
                            self.emit_event(event)
                    with self.lock:
                        if self._cancel is cancel:self._schedule_pending=False
                    # Keep one cancellation owner for completed playback tails.
                    # No cues are pending here and status reports replay idle.
                    cancel.wait()
                except Exception:
                    cancel.set()
                    with self.lock:
                        if self._cancel is cancel:self._last_error='Optional audio replay failed'
                finally:
                    with self.lock:
                        if cancel.is_set():self._finish_cancel_locked(cancel,playback_trial_id)
            self._thread=threading.Thread(target=run,daemon=True,name='sircus-show-audio');self._thread.start()
        return self.status()

    def close(self):self.cancel_replay()
