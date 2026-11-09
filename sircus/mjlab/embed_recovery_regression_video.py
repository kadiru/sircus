"""One-shot additive publication of a byte-bound, software-rendered saved movie.

Run after the progress publisher has finished. The existing HTTP allowlist is
unchanged: media is inlined in this panel only, never in compact CSS snapshots.
The ordinary snapshot publisher preserves this media block on later refreshes.
"""
from __future__ import annotations
import argparse
import base64
import hashlib
import json
from pathlib import Path

from .prepare_recovery_regression import (START, END, VIDEO_START, VIDEO_END,
    checked_diagnosis, read, require, sha, atomic_bytes)

VERSION='sircus-recovery-regression-recorded-video-v1'
MAX_VIDEO_BYTES=32*1024*1024


def verified_video(video,manifest,diagnosis):
    video,manifest=Path(video).resolve(),Path(manifest).resolve()
    source=checked_diagnosis(diagnosis);raw=read(manifest)
    require(video.is_file() and video.suffix=='.mp4' and 0<video.stat().st_size<=MAX_VIDEO_BYTES,'Bounded MP4 file required')
    require(raw.get('version')==VERSION and raw.get('status')=='completed','Completed regression movie required')
    require(raw.get('native_steps_added')==0 and raw.get('held_arm_poses') is True
        and raw.get('sampled_ball_centers_only') is True and raw.get('interpolation') is False
        and raw.get('historical_labels_unchanged') is True and raw.get('native_dt_s')==1/15360,
        'Saved native64 centers with held joints and no new dynamics required')
    require(raw.get('backend',{}).get('software_only') is True and raw.get('backend',{}).get('cuda_visible_devices')=='','CPU software-rendered movie required')
    require(raw.get('output_sha256')==sha(video) and raw.get('video_bytes')==video.stat().st_size,'Exact movie bytes required')
    require(all(raw.get('source',{}).get(k)==source['source'][k] for k in ('report_sha256','audit_sha256','contract_fingerprint')),'Movie and diagnosis source bindings differ')
    lookup={r['record_id']:r for r in source['flight_misses']}
    pairs=[]
    for loss in source['losses']:
        if loss['recovery_bc']['native_release'] is True:
            for method in ('parent_bc','recovery_bc'):
                row=loss[method];lookup[row['record_id']]=row;pairs.append(row['record_id'])
    clips=raw.get('clips',[])
    require(len(pairs)==4 and len(clips)==5 and [c.get('record_id') for c in clips[:4]]==pairs,'Both exact native-flight parent/recovery pairs required')
    for clip in clips:
        row=lookup.get(clip.get('record_id'));require(row is not None,'Movie request absent from bound diagnosis')
        require(clip.get('record_sha256')==row['record_sha256'] and clip.get('capture')==row['capture']
            and clip.get('values')==row['values'] and clip.get('bucket_xy_m')==row['bucket_xy_m']
            and clip.get('native_cache_sha256')==row.get('trace',{}).get('native_cache_sha256'),'Movie clip and original native record differ')
    last=lookup[clips[-1]['record_id']]
    require(last['capture'] is False and last.get('forbidden_contact_steps',0)>0,'Observed representative forbidden-contact miss required')
    require(isinstance(raw.get('duration_s'),(int,float)) and 0<raw['duration_s']<=120,'Bounded two-minute movie required')
    return raw


def embed_video(original,video_bytes):
    require(original.count(START)==original.count(END)==1,'Exactly one regression panel required')
    require(original.count(VIDEO_START)==original.count(VIDEO_END) and original.count(VIDEO_START)<=1,'Ambiguous video markers')
    lo,hi=original.index(START),original.index(END)+len(END);require(hi>lo,'Reversed panel markers')
    panel=original[lo:hi]
    payload=base64.b64encode(video_bytes).decode('ascii')
    block=(VIDEO_START+'\n<figure style="margin:16px 0"><video controls playsinline preload="none" '
        'aria-label="Recorded native 3D replay of two parent recovery pairs and a representative miss" '
        'style="width:100%;max-width:1280px;border:1px solid #354558;border-radius:8px">'
        '<source type="video/mp4" src="data:video/mp4;base64,'+payload+'"></video>'
        '<figcaption class="rr-small">Actual saved native ball centers and held arm joints: both paired flight regressions, '
        'then a representative forbidden-contact miss. Normal speed and 0.25× close-ups. No new physics, interpolated '
        'ball motion or invented arm transit. Inspected historical TEST positions are now development evidence.</figcaption></figure>\n'+VIDEO_END)
    if VIDEO_START in original:
        a,b=original.index(VIDEO_START),original.index(VIDEO_END)+len(VIDEO_END)
        require(lo<a<b<hi,'Video must remain within its own panel')
        return original[:a]+block+original[b:]
    anchor='<h3>Recorded examples</h3>'
    require(panel.count(anchor)==1,'Recorded examples section required')
    return original[:lo]+panel.replace(anchor,anchor+block)+original[hi:]


def publish(target,video,manifest,diagnosis,expected_html_sha256):
    target=Path(target).resolve()
    require(target not in {Path(p).resolve() for p in (video,manifest,diagnosis)},'Keep publication separate from evidence')
    raw=verified_video(video,manifest,diagnosis)
    before=target.read_bytes();require(sha(target)==expected_html_sha256,'HTML changed; preserving existing work')
    movie=Path(video).read_bytes();require(hashlib.sha256(movie).hexdigest()==raw['output_sha256'],'Movie changed before publication')
    after=embed_video(before.decode(),movie).encode();atomic_bytes(target,after,before)
    return dict(status='published',version=VERSION,before_html_sha256=expected_html_sha256,
        html_sha256=sha(target),video_sha256=raw['output_sha256'],manifest_sha256=sha(manifest),clips=5,
        duration_s=raw['duration_s'],css_unchanged=True,services_restarted=0)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('target-html','video','manifest','diagnosis'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--expected-html-sha256',required=True);a=p.parse_args()
    print(json.dumps(publish(a.target_html,a.video,a.manifest,a.diagnosis,a.expected_html_sha256)))

if __name__=='__main__':main()
