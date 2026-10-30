"""Reproducible original snare-like drum roll; writes only its new asset/catalog row.

No playback, downloads or external packages. Existing sound assets are untouched.
"""
import io
import json
import math
from pathlib import Path
import random
import struct
import wave

RATE=48000
SECONDS=1
HITS=8
PEAK=.16
SEED=20261030
ROOT=Path(__file__).resolve().parents[1]


def samples():
    rng=random.Random(SEED);values=[]
    period=RATE//HITS
    for i in range(RATE*SECONDS):
        local=i%period;t=local/RATE
        # Eight alternating accented snare strokes; silent finite support at each
        # boundary makes the one-second loop sample-continuous without a click.
        if t>=.11:values.append(0.);continue
        attack=min(1.,t/.0015);tail=min(1.,(.11-t)/.012)
        noise=rng.uniform(-1.,1.)
        body=math.sin(2*math.pi*185*t)*math.exp(-30*t)
        ring=.22*math.sin(2*math.pi*327*t)*math.exp(-48*t)
        snare=.8*noise*math.exp(-43*t)
        accent=1. if (i//period)%2==0 else .82
        values.append(accent*attack*tail*(body+ring+snare))
    scale=PEAK/max(abs(v) for v in values)
    return [round(v*scale*32767) for v in values]


def wav_bytes():
    output=io.BytesIO();values=samples()
    with wave.open(output,'wb') as stream:
        stream.setparams((1,2,RATE,len(values),'NONE','not compressed'))
        stream.writeframes(struct.pack('<'+'h'*len(values),*values))
    return output.getvalue()


def main():
    assets=ROOT/'assets';assets.mkdir(exist_ok=True)
    (assets/'drum_roll.wav').write_bytes(wav_bytes())
    rows=json.loads((assets/'map.json').read_text())
    rows=[r for r in rows if r['name']!='drum_roll']
    rows.append({'name':'drum_roll','title':'Circus snare roll','note':54,'channel':16,
                 'file':'drum_roll.wav','seconds':SECONDS,'peak_dbfs':20*math.log10(PEAK),
                 'family':'Percussion','loop_seam':'zero-valued boundary; eight original synthesized strokes'})
    (assets/'map.json').write_text(json.dumps(rows,indent=2)+'\n')
    print('Generated drum_roll.wav only and updated its catalog row; no audio played')


if __name__=='__main__':main()
