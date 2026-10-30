"""Original additive/FM/noise synthesis. No assets, services or Python packages required."""
import math, random, wave, struct, json
from pathlib import Path
R=48000
ROOT=Path(__file__).resolve().parents[1]
A=ROOT/'assets'
TAU=2*math.pi

def tone(dur, f, decay=3, fm=0, noise=0, seed=1):
    rng=random.Random(seed);phase=0;low=0;out=[]
    n=int(dur*R)
    for i in range(n):
        t=i/R;u=i/max(1,n-1);hz=f(t,u) if callable(f) else f
        phase+=TAU*hz/R;low=.94*low+.06*rng.uniform(-1,1)
        attack=min(1,t/.006);tail=min(1,(dur-t)/.025)
        env=attack*tail*math.exp(-decay*t)
        out.append(env*(math.sin(phase+fm*math.sin(phase*2.01))+.22*math.sin(phase*2)+noise*low))
    return out

def energy_hum():
    """Four-second periodic oscillator loop: electrical buzz, no noise source.

    All carriers and modulators complete integer cycles at the loop boundary.
    Harmonics keep the low hum audible on small speakers without a hiss layer.
    """
    out=[]
    for i in range(4*R):
        t=i/R
        phase=TAU*150*t+.18*math.sin(TAU*.75*t)+.025*math.sin(TAU*5.25*t)
        second=TAU*151.5*t+.10*math.sin(TAU*.5*t)
        body=sum(a*math.sin(h*phase) for h,a in [(1,.70),(2,.26),(3,.16),(4,.08),(6,.035)])
        body+=.18*math.sin(second)+.06*math.sin(2*second)
        out.append((.90+.06*math.sin(TAU*1.5*t))*body)
    return out

def energy_sweep():
    """Short tonal blade-motion gesture, with eased pitch and amplitude."""
    dur=.9;out=[]
    for i in range(int(dur*R)):
        t=i/R;u=t/dur
        # Integral of 140 + 420*sin(pi*t/dur)^2 Hz: rise, then fall.
        phase=TAU*(140*t+420*(t/2-dur*math.sin(TAU*u)/(4*math.pi)))
        body=math.sin(phase)+.28*math.sin(2*phase)+.13*math.sin(3*phase)+.045*math.sin(5*phase)
        out.append(math.sin(math.pi*u)**1.4*body)
    return out

def mix(parts):
    out=[0.0]*max(start+len(p) for start,p in parts)
    for start,p in parts:
        for i,v in enumerate(p):out[start+i]+=v
    return out

def save(name,arr,peak=.20):
    scale=peak/max(max(abs(v) for v in arr),1e-9)
    arr=[v*scale for v in arr]
    with wave.open(str(A/(name+'.wav')),'wb') as w:
        w.setparams((1,2,R,len(arr),'NONE','not compressed'))
        w.writeframes(struct.pack('<'+'h'*len(arr),*(round(v*32767) for v in arr)))
    return arr

def vlq(n):
    out=[n&127];n>>=7
    while n:out.insert(0,(n&127)|128);n>>=7
    return bytes(out)

def main():
    A.mkdir(exist_ok=True)
    # Impact sounds intentionally built first.
    sounds=[
     ('boing',36,tone(.95,lambda t,u:155+240*math.exp(-t*10)+65*math.sin(TAU*11*t)*math.exp(-t*5),2.8,.45)),
     ('bone_low',37,tone(.19,480,24,2.8,.4)),
     ('bone_high',38,tone(.14,740,32,2.4,.3,2)),
     ('eerie_bell',39,tone(.75,760,5,1.6)),
     ('pinball',40,mix([(0,tone(.42,1200,10,1.2)),(int(.035*R),tone(.3,1600,14,.4))])),
     ('transport',41,energy_hum()),
     ('release',42,tone(.65,lambda t,u:1250*2**(-2.4*u),1.1,.08)),
     ('capture',43,mix([(int(i*.11*R),tone(.85,f,4,.5)) for i,f in enumerate([523.25,622.25,783.99,1046.5])])),
     ('miss',44,tone(.9,lambda t,u:520*2**(-2*u)*(1+.05*math.sin(TAU*6*t)),1.7,.4)),
     ('energy_sweep',45,energy_sweep()),
    ]
    data={};mapping=[]
    for name,note,raw in sounds:
        peak=.16 if name in ('transport','energy_sweep') else .20
        data[name]=save(name,raw,peak)
        row={'name':name,'note':note,'channel':16,'file':name+'.wav','seconds':len(raw)/R,'peak_dbfs':20*math.log10(peak)}
        if name in ('transport','energy_sweep'):
            row.update(title='Energy hum' if name=='transport' else 'Energy sweep',family='Sci-fi')
        mapping.append(row)
    (A/'map.json').write_text(json.dumps(mapping,indent=2)+'\n')
    # Short sequential audition, all impacts first, not a performance recording.
    parts=[];events=[];cursor=0
    for m in mapping:
        arr=data[m['name']];parts.append((int(cursor*R),arr))
        events += [(int(cursor*960),bytes([0x9f,m['note'],85])),(int((cursor+m['seconds'])*960),bytes([0x8f,m['note'],0]))]
        cursor+=m['seconds']+.25
    save('audition',mix(parts),.20)
    track=b'\x00\xff\x51\x03\x07\xa1\x20\x00\xbf\x40\x00\x00\xbf\x0b\x55';last=0
    for tick,msg in sorted(events):track+=vlq(tick-last)+msg;last=tick
    track+=b'\x00\xbf\x0b\x00\x00\xbf\x78\x00\x00\xff\x2f\x00'
    (A/'audition.mid').write_bytes(b'MThd'+struct.pack('>IHHH',6,0,1,480)+b'MTrk'+struct.pack('>I',len(track))+track)
    (A/'LICENSE.txt').write_text('Original synthesized Sircus sounds created for this project. No third-party recordings, purchased plugins, cloud generation, or sound libraries. You may use, modify and redistribute these generated assets.\n\n'
        'Eight haunted sounds and two original tonal sci-fi gestures: src/generate_pack.py.\n'
        'Six cartoon sounds: src/render_scores.py, rendered from the original sircus/audio/score.py MIDI gestures with a simple oscillator and smoothed amplitude envelope.\n'
        'Regenerate in that order to rebuild the combined catalog and audition files.\n')
    print(json.dumps({'sounds':len(mapping),'sample_rate':R,'audition_seconds':cursor,'max_sample_peak':.20}))
if __name__=='__main__':main()
