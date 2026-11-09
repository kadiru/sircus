"""Add a verified recorded-trajectory movie to the running read-only interface.

The existing server has no movie route. Inline media is generated into runtime
HTML only; neither the MP4 nor its base64 data belongs in source control. This
publisher adds no simulation, service restart or change to the study results.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import math
from pathlib import Path
import re

from .prepare_table_height_tilt import compact as study_compact, END as STUDY_END
from .prepare_table_training import atomic_bytes
from .prepare_table_training_diagnosis import report_path, sha, take

VERSION = 'sircus-height-tilt-recorded-video-v1'
KEY = 'table_height_tilt_replay'
START = '<!--HEIGHT_TILT_REPLAY_START-->'
END = '<!--HEIGHT_TILT_REPLAY_END-->'
TITLE = 'Recorded simulation replay • held arm endpoints • no new physics'
MAX_VIDEO_BYTES = 32 * 1024 * 1024


def require(value, message):
    if not value:
        raise ValueError(message)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def valid_hash(value):
    return isinstance(value, str) and re.fullmatch('[a-f0-9]{64}', value) is not None


def event_list(value):
    if isinstance(value, dict):
        value = [dict(label=key, time_s=time) for key, time in value.items() if time is not None]
    require(isinstance(value, list), 'Clip contact events must be a list or mapping')
    result = []
    for item in value:
        require(isinstance(item, dict), 'Recorded contact events must be objects')
        label, time = item.get('label'), item.get('time_s')
        require(isinstance(label, str) and bool(label) and finite(time) and time >= 0,
                'Finite recorded contact time and label required')
        result.append(dict(label=label, time_s=time))
    return sorted(result, key=lambda event: event['time_s'])


def compact(manifest, movie, study, audit=None):
    """Bind runtime media to a completed audited study; exclude raw local paths."""
    manifest, movie = Path(manifest), Path(movie)
    raw = json.loads(manifest.read_text())
    require(raw.get('version') == VERSION and raw.get('status') == 'completed',
            'Completed recorded height/tilt video manifest required')
    require(raw.get('native_steps_added') == 0 and raw.get('held_arm_poses') is True
            and raw.get('sampled_ball_centers_only') is True,
            'Only held-endpoint recorded ball-center playback is supported')
    evidence = study_compact(study, audit_path=audit)
    require(evidence.get('grid_protocol', {}).get('substeps') == 64,
            'This replay panel requires the fixed 64-substep study')
    proof = evidence['audit']
    require(proof.get('status') == 'passed' and proof.get('full_checks_verified') is True,
            'Completed independently audited height/tilt study required')
    source = dict(report_sha256=proof['report_sha256'], audit_sha256=proof['proof_sha256'],
                  contract_fingerprint=evidence['contract_fingerprint'])
    require(all(raw.get('source', {}).get(key) == value for key, value in source.items()),
            'Video source differs from the audited study')
    size = movie.stat().st_size
    require(0 < size <= MAX_VIDEO_BYTES and raw.get('video_bytes') == size,
            'Video byte count differs or exceeds the inline media limit')
    require(valid_hash(raw.get('output_sha256')) and sha(movie) == raw['output_sha256'],
            'Movie checksum differs from its manifest')
    require(movie.read_bytes()[4:8] == b'ftyp', 'An MP4 media container is required')
    duration, fps = raw.get('duration_s'), raw.get('fps')
    require(finite(duration) and 0 < duration <= 180 and finite(fps) and 0 < fps <= 120,
            'Finite movie duration and frame rate required')
    frames = raw.get('frames')
    count = len(frames) if isinstance(frames, list) else raw.get('frame_count', frames)
    require(isinstance(count, int) and not isinstance(count, bool) and count > 0
            and abs(count / fps - duration) < 1 / fps,
            'Frame count and movie duration disagree')
    require(valid_hash(raw.get('frames_sha256')), 'Recorded frame-map checksum required')
    if isinstance(frames, list):
        digest = hashlib.sha256(json.dumps(frames, sort_keys=True, allow_nan=False).encode()).hexdigest()
        require(digest == raw['frames_sha256'], 'Recorded frame-map checksum differs')
    clips = raw.get('clips')
    require(isinstance(clips, list) and 1 <= len(clips) <= 12, 'One to twelve recorded clips required')
    result = take(raw, ('version', 'status', 'output_sha256', 'video_bytes', 'duration_s',
                        'fps', 'width', 'height', 'frames_sha256', 'native_steps_added',
                        'held_arm_poses', 'sampled_ball_centers_only'))
    result.update(source=source, frame_count=count, manifest_sha256=sha(manifest), clips=[])
    previous_end = 0
    for clip in clips:
        record, start, end, values = (clip.get(key) for key in ('record_id', 'start_s', 'end_s', 'values'))
        require(isinstance(record, str) and record and isinstance(clip.get('label'), str),
                'Recorded shot identifier and clip label required')
        require(finite(start) and finite(end) and 0 <= start < end <= duration
                and start >= previous_end, 'Clip timeline must be ordered and inside the movie')
        require(isinstance(values, list) and len(values) >= 5 and all(finite(v) for v in values),
                'Finite recorded shot settings required')
        require(isinstance(clip.get('capture'), bool), 'Explicit recorded capture outcome required')
        previous_end = end
        clean = take(clip, ('record_id', 'record_sha256', 'native_cache_sha256', 'context_id', 'label',
                           'start_s', 'end_s', 'values', 'capture', 'whole_ball_entry', 'source_interval_s'))
        clean['events'] = event_list(clip.get('events', []))
        result['clips'].append(clean)
    json.dumps(result, allow_nan=False)
    return result


def embed_html(original, snapshot, movie_bytes):
    require(original.count(START) == original.count(END) and original.count(START) <= 1,
            'Ambiguous replay markers')
    require(hashlib.sha256(movie_bytes).hexdigest() == snapshot['output_sha256'],
            'Movie changed after verification')
    escape = html.escape
    rows, chapters = [], []
    for clip in snapshot['clips']:
        start = clip['start_s']
        chapters.append(f'<button type="button" data-replay-start="{start}">{escape(clip["label"])}</button>')
        values = clip['values']
        events = ' → '.join(f'{e["label"]} {e["time_s"]:.3f}s' for e in clip['events']) or 'No event timing recorded'
        outcome = 'Retained capture' if clip['capture'] else 'Miss'
        rows.append('<tr>' + ''.join(f'<td>{escape(str(value))}</td>' for value in
                    (clip['label'], f'{values[2]*1000:.1f} mm', f'{values[3]:.1f}°',
                     f'{values[4]*1000:.1f} mm', outcome, events, clip['record_id'])) + '</tr>')
    source = snapshot['source']
    encoded = base64.b64encode(movie_bytes).decode('ascii')
    style = """#heightTiltReplay{max-width:1480px;margin:36px auto;padding:30px;box-sizing:border-box;background:#101f2a;border:1px solid #334f61;border-radius:18px;color:#dce9ef;font:15px/1.55 system-ui,sans-serif;scroll-margin-top:20px}#heightTiltReplay h2{font-size:26px;line-height:1.25}#heightTiltReplay video{display:block;width:100%;max-height:80vh;background:#080d12;border-radius:10px}#heightTiltReplay .htr-meta{font-size:13px;color:#b8cbd7;overflow-wrap:anywhere}#heightTiltReplay .htr-controls{display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin:16px 0}#heightTiltReplay button,#heightTiltReplay select{background:#1b303e;color:#f0f7fa;border:1px solid #5b7c91;border-radius:6px;padding:9px;cursor:pointer}#heightTiltReplay a{color:#a5e7dc}#heightTiltReplay table{border-collapse:collapse;font-size:12px;width:100%}#heightTiltReplay td,#heightTiltReplay th{text-align:left;padding:10px;border-bottom:1px solid #304754;vertical-align:top}#heightTiltReplay .htr-table{overflow:auto}#heightTiltReplay details{margin-top:20px}@media(max-width:700px){#heightTiltReplay{margin:20px 8px;padding:18px}#heightTiltReplay h2{font-size:22px}}"""
    script = """(function(){const panel=document.getElementById('heightTiltReplay'),video=panel.querySelector('video');for(const button of panel.querySelectorAll('[data-replay-start]'))button.addEventListener('click',()=>{video.currentTime=Number(button.dataset.replayStart);video.play().catch(()=>{});});panel.querySelector('select').addEventListener('change',event=>{video.playbackRate=Number(event.target.value);});panel.querySelector('[data-replay-download]').href=video.querySelector('source').src;})();"""
    data = json.dumps(snapshot, allow_nan=False).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    block = f'''{START}
<style>{style}</style>
<section id="heightTiltReplay" aria-label="Recorded height and tilt experiment video">
<p class="htr-meta">SAVED MJLAB / MUJOCO TRAJECTORIES · FIXED 64 SUBSTEPS</p>
<h2>{TITLE}</h2>
<p>Watch the saved drops, including slow-motion close-ups. These are recorded ball paths from the completed study. The arms remain at their tested endpoint poses; this video does not show arm transit or a new rollout.</p>
<video controls playsinline preload="metadata" aria-label="Recorded drops with slow motion"><source src="data:video/mp4;base64,{encoded}" type="video/mp4"></video>
<div class="htr-controls" aria-label="Jump to recorded shot">{''.join(chapters)}</div>
<div class="htr-controls"><label>Video speed <select aria-label="Video playback speed"><option value="0.25">¼×</option><option value="0.5">½×</option><option value="1" selected>1× (includes recorded slow motion)</option></select></label><a data-replay-download download="sircus-height-tilt-replay.mp4">Download MP4</a></div>
<p class="htr-meta">Event times below are seconds after release in the original simulation, not video time. The positive control is an established reference shot, not a new success from the height/tilt grid. This small selection illustrates failure routes; the full study results remain above.</p>
<div class="htr-table"><table><thead><tr><th>Clip</th><th>Board height</th><th>Tilt</th><th>Drop gap</th><th>Recorded outcome</th><th>Contact timeline</th><th>Evidence ID</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<details><summary>Video and study provenance</summary><p class="htr-meta">Movie SHA256 {snapshot['output_sha256']}<br>Manifest SHA256 {snapshot['manifest_sha256']}<br>Frame map SHA256 {snapshot['frames_sha256']}<br>Study report SHA256 {source['report_sha256']}<br>Study audit SHA256 {source['audit_sha256']}<br>Contract {source['contract_fingerprint']}<br>{snapshot['frame_count']} frames · {snapshot['fps']} fps · {snapshot['duration_s']} seconds · zero new physics steps.</p></details>
</section>
<script type="application/json" id="height-tilt-replay-snapshot">{data}</script>
<script>{script}</script>
{END}'''
    if START in original:
        begin, end = original.index(START), original.index(END) + len(END)
        require(end > begin, 'Replay markers out of order')
        return original[:begin] + block + original[end:]
    require(original.count(STUDY_END) == 1, 'Existing height/tilt study panel required')
    at = original.index(STUDY_END) + len(STUDY_END)
    return original[:at] + '\n' + block + original[at:]


def install(target, snapshot, movie, expected_sha256):
    target = Path(target)
    original = target.read_bytes()
    require(hashlib.sha256(original).hexdigest() == expected_sha256,
            'Target HTML changed; existing work preserved')
    updated = embed_html(original.decode(), snapshot, Path(movie).read_bytes()).encode()
    atomic_bytes(target, updated, original)


def publish_display(target, snapshot):
    path = Path(target)
    for _ in range(3):
        original = path.read_bytes()
        value = json.loads(original)
        value[KEY] = snapshot
        try:
            atomic_bytes(path, (json.dumps(value, indent=2, allow_nan=False) + '\n').encode(), original)
            return
        except RuntimeError:
            continue
    raise RuntimeError('Concurrent display writer preserved')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'movie', 'study', 'audit', 'target-html', 'snapshot', 'display'):
        parser.add_argument('--' + name, type=Path, required=name in ('manifest', 'movie', 'study'))
    parser.add_argument('--expected-html-sha256')
    args = parser.parse_args()
    immutable_root = report_path(args.study).parent.resolve()
    inputs = {args.manifest.resolve(), args.movie.resolve()}
    for target in (args.target_html, args.snapshot, args.display):
        require(not target or (not target.resolve().is_relative_to(immutable_root) and target.resolve() not in inputs),
                'Publication must stay outside immutable study and media evidence')
    snapshot = compact(args.manifest, args.movie, args.study, args.audit)
    if args.target_html:
        install(args.target_html, snapshot, args.movie, args.expected_html_sha256)
    if args.snapshot:
        atomic_bytes(args.snapshot, (json.dumps(snapshot, indent=2, allow_nan=False) + '\n').encode())
    if args.display:
        publish_display(args.display, snapshot)
    print(json.dumps(dict(status=snapshot['status'], duration_s=snapshot['duration_s'],
                          clips=len(snapshot['clips']), output_sha256=snapshot['output_sha256'])))


if __name__ == '__main__':
    main()
