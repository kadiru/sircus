"""Offline evidence renderer. Requires Pillow and imageio-ffmpeg in a media-only venv.

Uses saved numerical frames and static endpoint boxes; does not import a trainer,
change a checkpoint, interpolate robot motion, or contact hardware/services.
"""
from __future__ import annotations
import argparse, bisect, hashlib, json, math, subprocess
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg

W,H,FPS=1280,720,15
CREAM='#f6f4eb'; INK='#173d42'; SOFT='#647b78'; TEAL='#16848a'; CORAL='#da6d4e'
FONT_PATH=next((p for p in ['/System/Library/Fonts/Helvetica.ttc','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'] if Path(p).exists()),None)
FONTS={}
def font(size):
    if size not in FONTS:FONTS[size]=ImageFont.truetype(FONT_PATH,size) if FONT_PATH else ImageFont.load_default(size=size)
    return FONTS[size]
def text(d,xy,value,size=18,fill=INK):d.text(xy,str(value),font=font(size),fill=fill)
def wrapped(d,xy,value,width,size=16,fill=SOFT,line=23):
    x,y=xy; current=''
    for word in str(value).split():
        candidate=(current+' '+word).strip()
        if current and d.textlength(candidate,font=font(size))>width:text(d,(x,y),current,size,fill);y+=line;current=word
        else:current=candidate
    if current:text(d,(x,y),current,size,fill);y+=line
    return y

def scope(row):
    flag=row.get('scope',{}).get('in_distribution')
    return 'OUTSIDE TRAINED LAYOUT RANGE' if flag is False else 'Within trained layout range' if flag is True else 'Layout-range status unreported'
def mmul(m,p):return [sum(m[i][j]*p[j] for j in range(3)) for i in range(3)]
def rotation(rpy):
    a,b,c=[math.radians(x) for x in rpy];sa,ca=math.sin(a),math.cos(a);sb,cb=math.sin(b),math.cos(b);sc,cc=math.sin(c),math.cos(c)
    return [[cc*cb,cc*sb*sa-sc*ca,cc*sb*ca+sc*sa],[sc*cb,sc*sb*sa+cc*ca,sc*sb*ca-cc*sa],[-sb,cb*sa,cb*ca]]
def box(center,size,matrix=None):
    matrix=matrix or [[1,0,0],[0,1,0],[0,0,1]]
    return [[center[i]+p[i] for i in range(3)] for p in [mmul(matrix,[size[0]*x/2,size[1]*y/2,size[2]*z/2]) for x,y,z in [(-1,-1,-1),(-1,-1,1),(-1,1,-1),(-1,1,1),(1,-1,-1),(1,-1,1),(1,1,-1),(1,1,1)]]]
FACES=[[0,4,6,2],[1,3,7,5],[0,1,5,4],[2,6,7,3],[0,2,3,1],[4,5,7,6]]
def box_faces(points,color):return [([points[i] for i in face],color) for face in FACES]

def scene_mesh(row,frame):
    scene=row['scene'];table=scene['table'];polys=box_faces(box(table['center_mm'],table['size_mm']), (175,194,177,255))
    for key,color in [('arm10_release_boxes',(217,225,207,255)),('arm14_boxes',(102,172,157,255))]:
        for item in row.get(key,[]):polys+=box_faces(box(item['center_mm'],item['size_mm'],item.get('rotation_matrix') or item.get('rotation') or rotation(item.get('rpy_deg',[0,0,0]))),color)
    b=scene['board'];center=(frame or {}).get('board_center_mm',b['center_mm']);m=(frame or {}).get('board_rotation') or rotation(b['rpy_deg'])
    polys+=box_faces(box(center,b['size_mm'],m),(65,159,148,255))
    bucket=scene['bin'];cx,cy=bucket['center_xy_mm'];z=bucket['floor_top_mm'];top=z+bucket['interior_depth_mm'];rings=[]
    for radius,height in [(bucket['outer_bottom_diameter_mm']/2,z-bucket['bottom_thickness_mm']),(bucket['outer_top_diameter_mm']/2,top),(bucket['inner_top_diameter_mm']/2,top),(bucket['inner_bottom_diameter_mm']/2,z)]:
        rings.append([[cx+radius*math.cos(i*math.tau/36),cy+radius*math.sin(i*math.tau/36),height] for i in range(36)])
    for i in range(36):
        j=(i+1)%36
        polys.append(([rings[0][i],rings[0][j],rings[1][j],rings[1][i]],(233,185,78,110)))
        polys.append(([rings[1][i],rings[1][j],rings[2][j],rings[2][i]],(249,211,132,210)))
    return polys,rings

class Projection:
    def __init__(self,row):
        polys,_=scene_mesh(row,None);points=[p for face,_ in polys for p in face]+[[150,-600,200],[0,0,0],[0,0,300]]
        # Fit all recorded ball positions too; exits remain visible rather than cropped.
        points += [f['ball_xyz_mm'] for f in row['result'].get('frames',[])]
        lo=[min(p[i] for p in points) for i in range(3)];hi=[max(p[i] for p in points) for i in range(3)]
        self.target=[(lo[i]+hi[i])/2 for i in range(3)];self.back=[0,-.8,.6];self.up=[0,.6,.8]
        self.scale=550/2/math.tan(math.radians(38)/2);aspect=890/550;tan=math.tan(math.radians(38)/2)*.85;self.distance=500
        for x in [lo[0],hi[0]]:
            for y in [lo[1],hi[1]]:
                for z in [lo[2],hi[2]]:
                    d=[v-t for v,t in zip([x,y,z],self.target)];depth=sum(v*q for v,q in zip(d,self.back));vertical=sum(v*q for v,q in zip(d,self.up))
                    self.distance=max(self.distance,depth+abs(d[0])/(tan*aspect),depth+abs(vertical)/tan)
    def xyz(self,p):
        d=[v-t for v,t in zip(p,self.target)];depth=self.distance-sum(v*q for v,q in zip(d,self.back));vertical=sum(v*q for v,q in zip(d,self.up))
        return (35+445+d[0]*self.scale/depth,95+275-vertical*self.scale/depth,depth)


def chart(d,report,rect=(50,175,850,540)):
    left,top,right,bottom=rect;rows=report['learning_curve'];ns=sorted(set(r['train_scenes'] for r in rows));lo,hi=math.log2(ns[0]),math.log2(ns[-1]);x=lambda n:left+60+(math.log2(n)-lo)/max(1,hi-lo)*(right-left-95);y=lambda f:bottom-35-f*(bottom-top-60)
    d.rounded_rectangle(rect,10,fill='#fffef7',outline='#dcded1')
    for tick in [0,.25,.5,.75,1]:
        d.line([(left+52,y(tick)),(right-25,y(tick))],fill='#dfe3d5');text(d,(left+10,y(tick)-8),f'{tick:.0%}',13,SOFT)
    for n in ns:text(d,(x(n)-10,bottom-25),n,14,SOFT)
    colors=['#16848a','#d17252','#7a89ab']
    for seed,color in zip(sorted(set(r['seed'] for r in rows)),colors):
        selected=sorted([r for r in rows if r['seed']==seed],key=lambda r:r['train_scenes']);points=[(x(r['train_scenes']),y(r['captures']/r['episodes'])) for r in selected];d.line(points,fill=color,width=3)
        for r,(px,py) in zip(selected,points):
            if r.get('ci95'):d.line([(px,y(r['ci95'][0])),(px,y(r['ci95'][1]))],fill=color,width=1)
            d.ellipse((px-5,py-5,px+5,py+5),fill=color,outline=CREAM)
    for i,(seed,color) in enumerate(zip(sorted(set(r['seed'] for r in rows)),colors)):text(d,(left+75+i*215,bottom+15),f'Seed {seed}',15,color)

def poster(report):
    im=Image.new('RGB',(W,H),CREAM);d=ImageDraw.Draw(im);text(d,(40,26),'SIRCUS / SIMPLE CIRCUS',22);text(d,(1015,30),'SIMULATION EVIDENCE',13,TEAL)
    text(d,(40,83),'Learning the angle. Finding the height.',38);text(d,(42,135),'Actual frozen-policy test results across 48 held-out layouts; 95% intervals shown.',17,SOFT)
    chart(d,report);summary=report['summary'];selected=[r for r in report['learning_curve'] if r['seed']==summary['default_seed']]
    text(d,(902,185),'PREDECLARED SEED',14,TEAL);text(d,(902,211),summary['default_seed'],23)
    for i,r in enumerate(selected):text(d,(902,255+i*32),f"N {r['train_scenes']:>3}:  {r['captures']} / {r['episodes']}",22)
    y=411
    for label,key in [('Fixed action','fixed_baseline'),('Bounded search','limited_search_oracle')]:
        r=summary[key];text(d,(902,y),f"{label}: {r['captures']} / {r['episodes']}",18);y+=32
    wrapped(d,(902,y+10),'Search is a separate per-scene comparison. Frozen inference does not run an optimizer.',325,16)
    text(d,(42,610),'Training-layout count (log scale). Different seeds vary; improvement is not monotonic.',17,SOFT)
    text(d,(42,648),'Provisional geometry and contact dynamics. Static arm endpoints; no native paths or physical validation.',16,SOFT)
    text(d,(42,678),'Board bounce + whole-ball entry + one continuous second inside the bucket.',15,TEAL)
    return im

def clip_frame(report,row,t,projection):
    frames=row['result'].get('frames',[]);times=[f['t'] for f in frames];index=max(0,bisect.bisect_right(times,max(0,t)) - 1) if frames else None;frame=frames[index] if frames else None
    im=Image.new('RGB',(W,H),CREAM);d=ImageDraw.Draw(im,'RGBA');paired=row.get('paired_layout');label=f"SAME PREDECLARED TEST LAYOUT / N {row['train_scenes']}" if paired else 'SIRCUS / NUMERICAL REPLAY'
    text(d,(32,23),label,21);text(d,(32,59),scope(row),14,CORAL if row.get('scope',{}).get('in_distribution') is False else SOFT)
    d.rounded_rectangle((25,90,935,650),14,fill='#102f35');polys,rings=scene_mesh(row,frame)
    polys.sort(key=lambda item:sum(projection.xyz(p)[2] for p in item[0])/len(item[0]),reverse=True)
    for points,color in polys:d.polygon([projection.xyz(p)[:2] for p in points],fill=color,outline=(30,75,77,90))
    for ring in rings[1:3]:d.line([projection.xyz(p)[:2] for p in ring+[ring[0]]],fill=(255,217,141,230),width=2)
    for end,name,color in [([260,0,0],'+X','#f08d72'),([0,260,0],'+Y','#9dd59b'),([0,0,260],'+Z','#92bde3')]:
        d.line([projection.xyz([0,0,0])[:2],projection.xyz(end)[:2]],fill=color,width=2);text(d,projection.xyz(end)[:2],name,13,color)
    load=projection.xyz([150,-600,200]);d.ellipse((load[0]-4,load[1]-4,load[0]+4,load[1]+4),outline='#e9ce83',width=2);text(d,(load[0]-55,load[1]+8),'Historical loading TCP',12,'#e9ce83')
    if frames:
        points=[projection.xyz(f['ball_xyz_mm'])[:2] for f in frames[:index+1]]
        if len(points)>1:d.line(points,fill=(255,186,121,200),width=2)
        ball=projection.xyz(frame['ball_xyz_mm']);radius=max(4,projection.scale*row['scene']['ball']['diameter_mm']/2/ball[2]);d.ellipse((ball[0]-radius,ball[1]-radius,ball[0]+radius,ball[1]+radius),fill='#f08053',outline='#f9b689',width=2)
    else:wrapped(d,(230,345),'Endpoint IK failed. No numerical trajectory was executed.',500,22,'#f4c2ac')
    text(d,(42,624),'Front view from -Y | +Z up | actual numerical ball trace',13,'#9fbfba')
    y=100;y=wrapped(d,(966,y),row['label'],284,21,INK,27)+12
    text(d,(966,y),row['result']['outcome'].replace('_',' ').upper(),20,TEAL if row['result']['success'] else CORAL);y+=35
    text(d,(966,y),f"Recorded t = {frame['t']:.2f} s" if frame else 'No valid trajectory',15,SOFT);y+=44
    for key,label in [('roll_deg','Roll'),('pitch_deg','Pitch'),('u_mm','U'),('v_mm','V'),('gap_mm','Gap')]:
        text(d,(966,y),f"{label}: {row['action'][key]:.1f} {'deg' if key.endswith('deg') else 'mm'}",18);y+=29
    role='Forced U/V contact intervention' if row.get('role')=='forced_contact_intervention' else 'Frozen checkpoint prediction'
    y=wrapped(d,(966,y+13),role,280,16,TEAL)+8
    y=wrapped(d,(966,y),'Arm geometry shows static endpoint poses. No robot path is generated.',280,15,SOFT,21)+9
    wrapped(d,(966,y),'Provisional board/mount dynamics. Synthetic results only.',280,15,SOFT,21)
    text(d,(32,675),'Checkpoint '+row['checkpoint_id'][:24]+' | '+row['profile_id'],13,SOFT)
    text(d,(32,697),'Loading marker is screenshot TCP, not a measured ball center. Bucket 175 mm height remains unresolved.',12,SOFT)
    return im

def select_clips(report):
    rows=report['replays'];result=[];seen=set()
    def add(row):
        if row and row['id'] not in seen:result.append(row);seen.add(row['id'])
    for row in sorted([r for r in rows if r.get('paired_layout')],key=lambda r:r['train_scenes']):add(row)
    plain=[r for r in rows if not r.get('paired_layout') and r.get('role')=='frozen_policy']
    add(next((r for r in plain if r['result']['success']),None))
    distance=[r for r in plain if 'distance' in r['id']]
    if distance:
        distance.sort(key=lambda r:math.dist(r['scene']['board']['center_mm'][:2],r['scene']['bin']['center_xy_mm']));add(distance[0]);add(distance[-1])
    interventions=[r for r in rows if r.get('role')=='forced_contact_intervention']
    for row in interventions:add(row)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    report=json.loads(args.report.read_text());assert report['data_origin']=='synthetic' and report['physically_verified'] is False
    rows=select_clips(report);assert rows and all(r['role'] in ['frozen_policy','forced_contact_intervention'] for r in rows)
    args.output.parent.mkdir(parents=True,exist_ok=True);cover=poster(report);poster_path=args.output.with_suffix('.png');cover.save(poster_path)
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe();log=args.output.with_suffix('.ffmpeg.log');command=[ffmpeg,'-y','-f','rawvideo','-pixel_format','rgb24','-video_size',f'{W}x{H}','-framerate',str(FPS),'-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','22','-pix_fmt','yuv420p','-movflags','+faststart',str(args.output)]
    total=0
    with log.open('wb') as stderr:
        process=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=stderr)
        try:
            for _ in range(10*FPS):process.stdin.write(cover.tobytes());total+=1
            for n,row in enumerate(rows):
                projection=Projection(row);print(f"Rendering {n+1}/{len(rows)} {row['id']} outcome={row['result']['outcome']}",flush=True)
                for i in range(int(6.4*FPS)):
                    frame=clip_frame(report,row,i/FPS-.5,projection);process.stdin.write(frame.tobytes());total+=1
                    if i==int(3*FPS):frame.save(args.output.with_name(args.output.stem+f'-sample-{n+1:02}.png'))
            for _ in range(4*FPS):process.stdin.write(cover.tobytes());total+=1
        finally:process.stdin.close()
        if process.wait()!=0:raise RuntimeError('ffmpeg failed; see '+str(log))
    evidence={'schema':'sircus-offline-evidence-video-v1','report_sha256':hashlib.sha256(args.report.read_bytes()).hexdigest(),'duration_seconds':total/FPS,'frames':total,'fps':FPS,'width':W,'height':H,'clips':[{'id':r['id'],'checkpoint_id':r['checkpoint_id'],'outcome':r['result']['outcome'],'scope':r.get('scope'),'train_scenes':r.get('train_scenes'),'role':r['role']} for r in rows],'source':'saved numerical frames; frame selection by recorded timestamp; no interpolated robot motion','native_paths_checked':False,'physically_verified':False}
    args.output.with_suffix('.json').write_text(json.dumps(evidence,indent=2)+'\n');print(json.dumps({'video':str(args.output),'poster':str(poster_path),'duration':total/FPS,'bytes':args.output.stat().st_size}),flush=True)
if __name__=='__main__':main()
