"""Validated sound assignments and host-local, atomic profile persistence."""
import copy
import json
import math
import os
from pathlib import Path
import random
import tempfile

EVENTS = {
    'pickup': ('Picked up', 'A short pickup cue'),
    'carry': ('Carried by arm', 'Stops on arrival, release or cancellation'),
    'release': ('Released / dropped', 'The moment of release'),
    'flight': ('Travelling through air', 'Stops on impact, outcome or cancellation'),
    'first_impact': ('First impact after release', 'A sound heuristic, not board detection'),
    'impact': ('Bounce / impact', 'Each subsequent impact, or every free-play tap'),
    'capture': ('Captured', 'Requires an explicit outcome'),
    'miss': ('Missed', 'Requires an explicit outcome'),
}


def assignment(sounds, selection='fixed', gain=.85, loop=False):
    return dict(sounds=sounds, selection=selection, gain=gain, loop=loop,
                avoid_repeat=True, pitch=0, max_seconds=8)


def builtins():
    ghost = dict(pickup=assignment(['eerie_bell']), carry=assignment(['transport'],loop=True,gain=.55),
                 release=assignment(['release']), flight=assignment(['transport'],loop=True,gain=.45),
                 first_impact=assignment(['boing']), impact=assignment(['bone_low','bone_high','eerie_bell','pinball'],'cycle'),
                 capture=assignment(['capture']), miss=assignment(['miss']))
    cartoon = copy.deepcopy(ghost)
    cartoon.update(pickup=assignment(['cartoon_plop']),release=assignment(['cartoon_squeak']),
                   flight=assignment(['release'],loop=True,gain=.45),first_impact=assignment(['cartoon_boing']),
                   impact=assignment(['cartoon_squeak','cartoon_plop','cartoon_spring'],'random'),
                   capture=assignment(['cartoon_celebration']),miss=assignment(['cartoon_wong']))
    minimal = {k: assignment([]) for k in EVENTS}
    minimal.update(impact=assignment(['bone_low'],gain=.5),first_impact=assignment(['bone_low'],gain=.5),
                   capture=assignment(['eerie_bell'],gain=.5),miss=assignment(['cartoon_plop'],gain=.5))
    return {'Ghost Booth':ghost, 'Cartoon Circus':cartoon, 'Minimal':minimal}


def validate(value, sounds):
    if not isinstance(value,dict) or set(value)!=set(EVENTS):
        raise ValueError('profile must contain all eight event assignments')
    for key, row in value.items():
        if not isinstance(row,dict) or set(row)!=set(assignment([])):
            raise ValueError('invalid assignment fields')
        pool=row['sounds']
        if not isinstance(pool,list) or len(pool)>64 or any(not isinstance(s,str) or s not in sounds for s in pool):
            raise ValueError('choose sounds from the catalog')
        if len(set(pool))!=len(pool):raise ValueError('sound choices must be unique')
        if row['selection'] not in ('fixed','cycle','random'):raise ValueError('invalid selection mode')
        for name,lo,hi in [('gain',0,1),('pitch',-12,12),('max_seconds',.2,20)]:
            n=row[name]
            if isinstance(n,bool) or not isinstance(n,(int,float)) or not math.isfinite(n) or not lo<=n<=hi:
                raise ValueError('invalid '+name)
        for name in ('loop','avoid_repeat'):
            if type(row[name]) is not bool:raise ValueError(name+' must be boolean')
        if row['loop'] and key not in ('carry','flight'):raise ValueError('only carry and flight can loop')
    return copy.deepcopy(value)


class Profiles:
    def __init__(self,sounds,path=None):
        self.sounds=sounds;self.path=Path(path) if path else None
        self.presets={name:validate(p,sounds) for name,p in builtins().items()}
        self.saved={};self.active='Ghost Booth';self.assignments=copy.deepcopy(self.presets[self.active])
        self.counters={};self.previous={};self.error=None
        if self.path and self.path.exists():
            try:
                if self.path.stat().st_size>256000:raise ValueError('profile file too large')
                data=json.loads(self.path.read_text())
                if not isinstance(data,dict) or data['schema']!='sircus.sound-profiles.v1' or not isinstance(data['profiles'],dict) or len(data['profiles'])>32:raise ValueError('invalid profile store')
                for name,p in data['profiles'].items():
                    self.name(name);self.saved[name]=validate(p,sounds)
                self.load(data['active'])
            except (ValueError,KeyError,TypeError,OSError) as exc:
                self.saved={};self.active='Ghost Booth';self.assignments=copy.deepcopy(self.presets[self.active])
                self.error='Saved profiles could not be loaded: '+str(exc)

    def name(self,name):
        if not isinstance(name,str) or not name.strip() or len(name)>64 or name in self.presets:
            raise ValueError('use a new profile name (1–64 characters); built-in profiles are read-only')

    def state(self):
        return dict(active=self.active, names=list(self.presets)+list(self.saved),
                    builtins=list(self.presets), assignments=copy.deepcopy(self.assignments), error=self.error)

    def apply(self,value):
        self.assignments=validate(value,self.sounds);self.counters.clear();self.previous.clear()

    def load(self,name):
        if name not in self.presets and name not in self.saved:raise ValueError('unknown profile')
        self.apply(self.presets[name] if name in self.presets else self.saved[name]);self.active=name

    def save(self,name,value):
        self.name(name);checked=validate(value,self.sounds)
        if name not in self.saved and len(self.saved)>=32:raise ValueError('at most 32 saved profiles')
        saved={**self.saved,name:checked}
        if self.path:
            self.path.parent.mkdir(parents=True,exist_ok=True)
            payload=json.dumps(dict(schema='sircus.sound-profiles.v1',active=name,profiles=saved),allow_nan=False)
            fd,tmp=tempfile.mkstemp(dir=self.path.parent,prefix='.profiles-')
            try:
                with os.fdopen(fd,'w') as f:f.write(payload);f.flush();os.fsync(f.fileno())
                os.replace(tmp,self.path)
            finally:
                if os.path.exists(tmp):os.unlink(tmp)
        self.saved=saved;self.active=name;self.error=None;self.apply(checked)

    def choose(self,event):
        row=self.assignments[event];pool=list(row['sounds'])
        if not pool or row['gain']==0:return None,row
        if row['selection']=='random':
            if row['avoid_repeat'] and len(pool)>1:pool=[s for s in pool if s!=self.previous.get(event)]
            sound=random.choice(pool)
        elif row['selection']=='cycle':
            index=self.counters.get(event,0);sound=pool[index%len(pool)];self.counters[event]=index+1
        else:sound=pool[0]
        self.previous[event]=sound
        return sound,row
