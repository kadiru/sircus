"""Add audited bouncer-clearance results and optional recorded video to the UI.

No simulation, inference or service control happens here. Runtime HTML embeds
media because the existing server has no movie route; binaries stay out of Git.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
from pathlib import Path
import re

from .prepare_table_height_tilt import COUNTS, BINDINGS, compact_cell
from .prepare_table_training import atomic_bytes
from .prepare_table_training_diagnosis import take, proof, report_path, default_audit, sha
from .prepare_height_tilt_replay import event_list, finite, valid_hash, MAX_VIDEO_BYTES

VERSION = 'sircus-bouncer-clearance-study-v1'
AUDIT = 'sircus-bouncer-clearance-audit-v1'
VIDEO = 'sircus-bouncer-clearance-recorded-video-v1'
KEY = 'bouncer_clearance_study'
START = '<!--BOUNCER_CLEARANCE_START-->'
END = '<!--BOUNCER_CLEARANCE_END-->'


def require(value, message):
    if not value:
        raise ValueError(message)


def compact(study, audit_path=None):
    path = report_path(study)
    raw = json.loads(path.read_text())
    require(raw.get('version') == VERSION, 'Explicit bouncer-clearance study required')
    out = take(raw, ('version', 'status', 'phase', 'contract_fingerprint', 'description',
                    'interpretation', 'cpu_elapsed_s', 'native_elapsed_s', 'native_requested',
                    'requested_configurations', 'error', 'training', 'noise',
                    'posthoc_development', 'admission_matrix_sha256'))
    out['parent_bindings'] = take(raw.get('parent_bindings'), BINDINGS)
    out['grid_protocol'] = take(raw.get('grid_protocol'), (
        'contexts', 'offsets_m', 'x_offsets_m', 'y_offsets_m', 'xy_offsets_m',
        'fixed_parameters', 'changed_parameters', 'candidates_per_case', 'total_candidates',
        'substeps', 'nominal_repeats', 'selection_order', 'selection_requires_capture',
        'selection_requires_zero_robot_contact_before_scoring_terminal', 'selection_requires_board_bounce',
        'source_record_ids', 'heading', 'alternative_same_pose_joint_branches'))
    require(out['grid_protocol'].get('substeps') == 64, 'Fixed64 study required')
    out['budget'] = take(raw.get('budget'), ('maximum_requested', 'candidate_proposals',
        'original_requests', 'control_requests', 'maximum_selected_repeat_requests',
        'cpu_admission_limit_s', 'native_acquired_limit_s', 'actual_requested',
        'native_discovery_requests', 'selected_repeat_requests'))
    out['admission_summary'] = take(raw.get('admission_summary'), ('proposals', 'completed', 'admitted', 'declined', 'unattempted'))
    out['discovery_summary'] = take(raw.get('discovery_summary'), ('proposals', 'admitted') + COUNTS)
    out['counts'] = take(raw.get('counts'), COUNTS)
    out['cases'] = []
    for case in raw.get('cases', []):
        item = take(case, ('context_id', 'source_record_id', 'source_record_sha256', 'bucket_xy_m', 'baseline_values'))
        item['grid'] = [compact_cell(cell) for cell in case.get('grid', [])]
        selected = case.get('selection')
        item['selection'] = None if selected is None else take(selected, (
            'candidate_id', 'grid_index', 'case_grid_index', 'values', 'row_sha256',
            'connection_sha256', 'route_quality', 'learning_reward', 'capture', 'diagnostic_robot_contact_steps'))
        item['repeats'] = {key: take(value, COUNTS) for key, value in case.get('repeats', {}).items()
                           if key in ('baseline', 'selected')}
        out['cases'].append(item)
    out['repeats'] = {key: take(value, COUNTS) for key, value in raw.get('repeats', {}).items()
                      if key in ('baseline', 'selected')}
    out['control_gates'] = {key: dict(take(value, ('status', 'direct_captures',
        'table_assisted_captures', 'negative_captures')), counts=take(value.get('counts'), COUNTS))
        for key, value in raw.get('control_gates', {}).items() if key in ('pre', 'post')}
    audit_path = Path(audit_path) if audit_path else default_audit(path)
    out['audit'] = proof(path, audit_path, AUDIT, raw)
    out['contact_diagnostics'] = []
    if out['audit']['status'] == 'passed':
        audited = json.loads(audit_path.read_text())
        if audited.get('native_checks_verified') is not True:
            out['audit'].update(status='unverified', full_checks_verified=False)
        else:
            out['contact_diagnostics'] = [take(row, (
                'record_id', 'physics_steps', 'scoring_terminal_s', 'capture', 'robot_contact_steps',
                'robot_contact_steps_before_first_bucket_or_table',
                'robot_contact_steps_from_first_bucket_or_table', 'first_bucket_or_table_contact_s',
                'first_board_contact_s', 'first_bucket_contact_s', 'first_table_contact_s',
                'first_floor_contact_s', 'first_robot_contact_s', 'first_unattributed_contact_s'))
                for row in audited.get('contact_diagnostics', [])]
    json.dumps(out, allow_nan=False)
    return out


def movie_summary(manifest, movie, study_snapshot):
    manifest, movie = Path(manifest), Path(movie)
    raw = json.loads(manifest.read_text())
    require(raw.get('version') == VIDEO and raw.get('status') == 'completed', 'Completed bouncer replay required')
    require(raw.get('native_steps_added') == 0 and raw.get('held_arm_poses') is True
            and raw.get('sampled_ball_centers_only') is True, 'Recorded held-endpoint replay required')
    audit = study_snapshot['audit']
    require(audit.get('status') == 'passed' and audit.get('full_checks_verified') is True, 'Verified study required for replay')
    source = dict(report_sha256=audit['report_sha256'], audit_sha256=audit['proof_sha256'],
                  contract_fingerprint=study_snapshot['contract_fingerprint'])
    require(all(raw.get('source', {}).get(k) == v for k, v in source.items()), 'Video belongs to a different study')
    data = movie.read_bytes()
    require(0 < len(data) <= MAX_VIDEO_BYTES and len(data) == raw.get('video_bytes')
            and valid_hash(raw.get('output_sha256')) and hashlib.sha256(data).hexdigest() == raw['output_sha256'],
            'Movie bytes/checksum differ')
    require(data[4:8] == b'ftyp', 'MP4 required')
    frames, fps, duration = raw.get('frames'), raw.get('fps'), raw.get('duration_s')
    require(isinstance(frames, list) and frames and finite(fps) and 0 < fps <= 60
            and finite(duration) and 0 < duration <= 120 and abs(len(frames)/fps-duration) < 1/fps,
            'Finite recorded movie timeline required')
    require(hashlib.sha256(json.dumps(frames, sort_keys=True, allow_nan=False).encode()).hexdigest()
            == raw.get('frames_sha256'), 'Frame-map checksum differs')
    clips = raw.get('clips')
    require(isinstance(clips, list) and 1 <= len(clips) <= 5, 'One to five evidence clips required')
    out = take(raw, ('version', 'status', 'output_sha256', 'video_bytes', 'width', 'height',
        'fps', 'duration_s', 'frames_sha256', 'native_steps_added', 'held_arm_poses', 'sampled_ball_centers_only'))
    out.update(source=source, manifest_sha256=sha(manifest), frame_count=len(frames), clips=[])
    end = 0
    for clip in clips:
        require(isinstance(clip.get('record_id'), str) and isinstance(clip.get('label'), str), 'Clip identity required')
        require(finite(clip.get('start_s')) and finite(clip.get('end_s'))
                and end <= clip['start_s'] < clip['end_s'] <= duration, 'Ordered clip timeline required')
        require(isinstance(clip.get('capture'), bool), 'Explicit capture outcome required')
        require(isinstance(clip.get('values'), list) and len(clip['values']) == 5
                and all(finite(v) for v in clip['values']), 'Recorded shot values required')
        clean = take(clip, ('record_id', 'context_id', 'label', 'start_s', 'end_s', 'values',
            'bucket_xy_m', 'capture', 'whole_ball_entry', 'record_sha256', 'native_cache_sha256',
            'source_interval_s', 'slow_source_interval_s'))
        clean['events'] = event_list(clip.get('events', {}))
        out['clips'].append(clean)
        end = clip['end_s']
    return out


def number(value, scale=1, places=0):
    return f'{value*scale:.{places}f}' if finite(value) else '—'


def table(headers, rows):
    esc = html.escape
    return '<div class="bc-table"><table><thead><tr>' + ''.join('<th>'+esc(h)+'</th>' for h in headers) + '</tr></thead><tbody>' + ''.join(
        '<tr>'+''.join('<td>'+esc(str(v))+'</td>' for v in row)+'</tr>' for row in rows) + '</tbody></table></div>'


def render_panel(snapshot, movie_bytes=None):
    esc = html.escape
    proof = snapshot['audit']
    verified = snapshot.get('status') == 'completed' and proof.get('status') == 'passed'
    status = 'Completed · independent native audit passed.' if verified else 'Provisional · '+str(snapshot.get('phase', snapshot.get('status', 'pending')))+'; final audit pending.'
    admission = snapshot['admission_summary']; discovery = snapshot['discovery_summary']
    stats = table(['Proposals', 'Admitted', 'Declined', 'Native discovery drops', 'Robot-contact drops', 'Discovery captures'], [[
        admission.get('proposals', '—'), admission.get('admitted', '—'), admission.get('declined', '—'),
        discovery.get('native_releases', '—'), discovery.get('robot_contact_shots', '—'), discovery.get('captures', '—')]])
    matched = []; case_html = []
    for case in snapshot['cases']:
        for variant, label in (('baseline', 'Original pose'), ('selected', 'Selected clear candidate')):
            r = case['repeats'].get(variant, {})
            matched.append([case['context_id'], label, str(r.get('captures', 0))+'/'+str(r.get('attempts', 0)),
                r.get('native_releases', 0), r.get('setup_rejections', 0),
                str(r.get('robot_contact_shots', 0))+'/'+str(r.get('native_releases', 0)), r.get('native_misses', 0)])
        bucket = case.get('bucket_xy_m', [])
        base = case.get('baseline_values', [])
        rows = []
        for cell in case['grid']:
            v = cell['values']; a = cell['admission']; d = cell['discovery']; selected = ' · selected' if cell.get('selected') else ''
            outcome = ('Capture' if d.get('captured') else 'Miss') if d and d.get('native_release') else 'Not flown'
            if not d and a.get('passed') is False: outcome = 'Setup declined'
            rows.append([str(cell['case_grid_index'])+selected,
                f'{number(v[0],1000,1)} / {number(v[1],1000,1)}',
                f'{number(v[0]-base[0],1000,0)} / {number(v[1]-base[1],1000,0)}' if len(base)==5 else '—',
                f'{number(v[2],1000,1)} / {number(v[3],1,1)} / {number(v[4],1000,1)}',
                'Passed' if a.get('passed') is True else str(a.get('reason') or a.get('status', 'Pending')),
                outcome, 'Yes' if d and d.get('robot_contact_steps', 0)>0 else ('No' if d and d.get('native_release') else '—')])
        selected = case.get('selection')
        selection = 'No clear candidate qualified for repeats.' if selected is None else 'Frozen selected candidate: '+str(selected['candidate_id'])+'. Selection is not a capture claim; use the repeated outcomes above.'
        case_html.append('<details><summary>'+esc(case['context_id'])+' · all proposals</summary><p>Fixed bucket X/Y: '+esc(' / '.join(number(x,1000,1) for x in bucket))+' mm.</p><p>'+esc(selection)+'</p>'+table(
            ['Candidate','Board X / Y mm','Shift X / Y mm','H mm / tilt ° / gap mm','Admission','Native outcome','Robot contact'],rows)+'</details>')
    controls = []
    for name, gate in snapshot['control_gates'].items():
        controls.append([name, gate.get('status','pending'),gate.get('direct_captures','—'),gate.get('table_assisted_captures','—'),gate.get('negative_captures','—')])
    contacts = ''
    if snapshot.get('contact_diagnostics'):
        contacts = '<details><summary>Measured contact timing · every native drop</summary><p>Robot contacts before the first bucket/table contact are listed separately from later contacts. These measured events are not a prediction, and neither group by itself establishes capture. Force samples count above 0.01 N up to the scoring terminal.</p>' + table(
            ['Record', 'Capture', 'First board / robot / bucket / table (s)', 'Robot-contact ticks before / after first bucket or table', 'Scored through (s)'],
            [[d['record_id'], 'Yes' if d['capture'] else 'No',
              ' / '.join(number(d.get('first_'+k+'_contact_s'), 1, 3) for k in ('board','robot','bucket','table')),
              str(d['robot_contact_steps_before_first_bucket_or_table'])+' / '+str(d['robot_contact_steps_from_first_bucket_or_table']),
              number(d['scoring_terminal_s'],1,3)] for d in snapshot['contact_diagnostics']]) + '</details>'
    video = ''
    if snapshot.get('video'):
        m = snapshot['video']
        require(movie_bytes is not None and hashlib.sha256(movie_bytes).hexdigest() == m['output_sha256'], 'Verified movie bytes required')
        encoded = base64.b64encode(movie_bytes).decode('ascii')
        chapters = ''.join(f'<button type="button" data-bc-time="{c["start_s"]}">{esc(c["label"])}</button>' for c in m['clips'])
        video = f'''<h3>Recorded before-and-after drops</h3><p>Saved native ball samples and held arm poses. No new physics is run for this video. Normal-speed views show the full recorded interval; slow views may show a shorter impact excerpt. Controls are references, not newly discovered successes.</p><video controls playsinline preload="metadata" aria-label="Bouncer clearance recorded comparison"><source src="data:video/mp4;base64,{encoded}" type="video/mp4"></video><div class="bc-controls">{chapters}</div><div class="bc-controls"><label>Playback speed <select aria-label="Bouncer replay speed"><option value="1">1× (includes slow motion)</option><option value="0.5">½×</option><option value="0.25">¼×</option></select></label><a data-bc-download download="sircus-bouncer-clearance.mp4">Download MP4</a></div>'''
        video += table(['Recorded example', 'Outcome', 'Events (seconds after release)'], [[c['label'],'Capture' if c['capture'] else 'Miss',' → '.join(f'{e["label"]} {e["time_s"]:.3f}s' for e in c['events'])] for c in m['clips']])
    return f'''<section id="bouncerClearanceStudy" aria-label="Bouncer clearance study"><p class="bc-meta">BOUNCER CLEARANCE · FIXED 64 SUBSTEPS · DEVELOPMENT SEARCH</p><h2>Keep the rebound path clear</h2><p class="bc-status">{esc(status)}</p><p>Failed trials are expected learning data. This targeted exploration preserves them, tries alternate board positions with unchanged physics and scoring, and compares contact reduction separately from bucket capture. No policy is trained or promoted here.</p>{stats}<p>Bucket position, board height, tilt and drop gap stay fixed per case. Board X/Y shifts by −120, −60, 0, +60 or +120 mm; the board still faces the bucket. Setup admission checks do not guarantee a clear ball flight.</p><h3>Matched nominal repeats</h3>{table(['Case','Variant','Captures / requests','Releases','Setup declines','Robot contact / releases','Released misses'],matched)}<p>Selection requires an observed board bounce and no robot contact through the complete scoring interval. A clear candidate can still miss the bucket. Discovery selects the candidate; fresh repeats test that frozen choice. Three inspected layouts and exact repeats do not establish generalization or noise robustness.</p>{video}<h3>Every proposed configuration remains visible</h3>{''.join(case_html)}{contacts}<h3>Bracketing controls</h3>{table(['Gate','Status','Direct captures','Table-assisted captures','Negative-control captures'],controls)}<details><summary>Evidence and limits</summary><p class="bc-meta">Report SHA256 {esc(str(proof.get('report_sha256','pending')))}<br>Audit SHA256 {esc(str(proof.get('proof_sha256','pending')))}<br>Contract {esc(str(snapshot.get('contract_fingerprint','pending')))}<br>Independent checks: {esc(str(proof.get('checks_count','pending')))}</p><p>Simulation endpoints are initialized and held. This is not a learned arm-motion or physical-robot demonstration. Table-assisted and rim-assisted captures remain valid after one continuous second fully inside. The original results and policies remain unchanged.</p></details></section>'''


CSS = '''#bouncerClearanceStudy{max-width:1480px;margin:36px auto;padding:30px;background:#101f2a;border:1px solid #375663;border-radius:18px;box-sizing:border-box;color:#e0edf2;font:15px/1.55 system-ui,sans-serif;scroll-margin-top:20px}#bouncerClearanceStudy h2{font-size:28px;line-height:1.2}#bouncerClearanceStudy h3{margin-top:28px}#bouncerClearanceStudy .bc-meta{font-size:12px;color:#adc7d3;overflow-wrap:anywhere}#bouncerClearanceStudy .bc-status{color:#8ee0c4;font-weight:600}#bouncerClearanceStudy table{border-collapse:collapse;width:100%;font-size:13px}#bouncerClearanceStudy th,#bouncerClearanceStudy td{text-align:left;padding:10px;border-bottom:1px solid #304a59;vertical-align:top}#bouncerClearanceStudy .bc-table{overflow:auto}#bouncerClearanceStudy details{margin:16px 0}#bouncerClearanceStudy summary{cursor:pointer}#bouncerClearanceStudy video{display:block;width:100%;max-height:80vh;background:#080d12;border-radius:10px}#bouncerClearanceStudy .bc-controls{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin:14px 0}#bouncerClearanceStudy button,#bouncerClearanceStudy select{background:#1b303e;color:#f0f7fa;border:1px solid #5b7c91;border-radius:6px;padding:9px;cursor:pointer}#bouncerClearanceStudy a{color:#a5e7dc}@media(max-width:700px){#bouncerClearanceStudy{margin:20px 8px;padding:18px}#bouncerClearanceStudy h2{font-size:23px}}'''
JS = '''(function(){const panel=document.getElementById('bouncerClearanceStudy'),video=panel.querySelector('video');if(!video)return;for(const b of panel.querySelectorAll('[data-bc-time]'))b.addEventListener('click',()=>{video.currentTime=Number(b.dataset.bcTime);video.play().catch(()=>{});});panel.querySelector('select').addEventListener('change',e=>{video.playbackRate=Number(e.target.value);});panel.querySelector('[data-bc-download]').href=video.querySelector('source').src;})();'''


def embed_html(original, snapshot, movie_bytes=None):
    require(original.count(START)==original.count(END) and original.count(START)<=1, 'Ambiguous study markers')
    data=json.dumps(snapshot,allow_nan=False).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    block=START+'\n<style>'+CSS+'</style>\n'+render_panel(snapshot,movie_bytes)+'\n<script type="application/json" id="bouncer-clearance-snapshot">'+data+'</script>\n<script>'+JS+'</script>\n'+END
    if START in original:
        begin,end=original.index(START),original.index(END)+len(END)
        require(end>begin,'Study markers out of order')
        return original[:begin]+block+original[end:]
    anchors=list(re.finditer(r'</body\s*>',original,re.I))
    require(bool(anchors),'Existing document body required')
    at=anchors[-1].start()
    return original[:at]+block+'\n'+original[at:]


def install(target,snapshot,expected_sha256,movie=None):
    path=Path(target);old=path.read_bytes()
    require(hashlib.sha256(old).hexdigest()==expected_sha256,'Target HTML changed; preserving existing work')
    atomic_bytes(path,embed_html(old.decode(),snapshot,None if movie is None else Path(movie).read_bytes()).encode(),old)


def publish_display(target,snapshot):
    path=Path(target)
    for _ in range(3):
        original=path.read_bytes();value=json.loads(original);value[KEY]=snapshot
        try:
            atomic_bytes(path,(json.dumps(value,indent=2,allow_nan=False)+'\n').encode(),original);return
        except RuntimeError:continue
    raise RuntimeError('Concurrent display writer preserved')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('study','audit','movie','manifest','target-html','display','snapshot'):
        p.add_argument('--'+name,type=Path,required=name=='study')
    p.add_argument('--expected-html-sha256');a=p.parse_args()
    root=report_path(a.study).parent.resolve()
    inputs={x.resolve() for x in (a.movie,a.manifest) if x}
    for target in (a.target_html,a.display,a.snapshot):
        require(not target or (not target.resolve().is_relative_to(root) and target.resolve() not in inputs),'Publication must stay outside source evidence')
    require(bool(a.movie)==bool(a.manifest),'Movie and manifest must be supplied together')
    snapshot=compact(a.study,a.audit)
    if a.movie:snapshot['video']=movie_summary(a.manifest,a.movie,snapshot)
    if a.target_html:install(a.target_html,snapshot,a.expected_html_sha256,a.movie)
    if a.display:publish_display(a.display,snapshot)
    if a.snapshot:atomic_bytes(a.snapshot,(json.dumps(snapshot,indent=2,allow_nan=False)+'\n').encode())
    print(json.dumps(dict(status=snapshot['status'],audit=snapshot['audit']['status'],video=bool(a.movie))))


if __name__=='__main__':main()
