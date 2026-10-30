"""Render the original shared MIDI gestures as small native-sampler assets."""
import json
import math
from pathlib import Path
import sys
import struct
import wave

sys.path.insert(0,str(Path(__file__).resolve().parents[4]))
from sircus.audio.score import CUES, score
from generate_pack import A, R, save, mix, vlq

TITLES={'boing':'Cartoon boing','squeak':'Rubber squeak','plop':'Soft plop','spring':'Tiny spring',
        'celebration':'Major-key celebration','wong':'Three comic wongs'}

def render(cue):
    rows=score(cue,gain=1);out=[];index=0;pitch=60;gate=False;bend=0;phase=0;level=0
    for i in range(int((rows[-1][0]+.08)*R)):
        t=i/R
        while index<len(rows) and rows[index][0]<=t:
            _,p=rows[index];index+=1
            if p[0]&0xf0==0x90:pitch=p[1];gate=True
            elif p[0]&0xf0==0x80:gate=False
            elif p[0]&0xf0==0xe0:bend=((p[1]+(p[2]<<7))-8192)/8192*12
        level+=(float(gate)-level)/(R*.006)
        phase+=2*math.pi*440*2**((pitch-69+bend)/12)/R
        out.append(level*(math.sin(phase)+.16*math.sin(phase*2)))
    return out

def main():
    mapping=[m for m in json.loads((A/'map.json').read_text()) if not m['name'].startswith('cartoon_')]
    for i,cue in enumerate(CUES):
        name='cartoon_'+cue;raw=render(cue);save(name,raw)
        mapping.append(dict(name=name,title=TITLES[cue],note=48+i,channel=16,file=name+'.wav',
                            seconds=len(raw)/R,peak_dbfs=20*math.log10(.2),family='Cartoon'))
    (A/'map.json').write_text(json.dumps(mapping,indent=2)+'\n')
    parts=[];events=[];cursor=0
    for m in mapping:
        with wave.open(str(A/m['file'])) as w:raw=w.readframes(w.getnframes())
        arr=[n/32768 for n in struct.unpack('<'+'h'*(len(raw)//2),raw)]
        parts.append((int(cursor*R),arr))
        events.extend([(int(cursor*960),bytes([0x9f,m['note'],85])),
                       (int((cursor+m['seconds'])*960),bytes([0x8f,m['note'],0]))])
        cursor+=m['seconds']+.25
    save('audition',mix(parts))
    track=b'\x00\xff\x51\x03\x07\xa1\x20\x00\xbf\x40\x00\x00\xbf\x0b\x55';last=0
    for tick,msg in sorted(events):track+=vlq(tick-last)+msg;last=tick
    track+=b'\x00\xbf\x0b\x00\x00\xbf\x78\x00\x00\xff\x2f\x00'
    (A/'audition.mid').write_bytes(b'MThd'+struct.pack('>IHHH',6,0,1,480)+b'MTrk'+struct.pack('>I',len(track))+track)
    print('Rendered six original MIDI gestures for the shared native sound catalog')

if __name__=='__main__':main()
