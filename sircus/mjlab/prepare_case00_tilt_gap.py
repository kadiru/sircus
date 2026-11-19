"""Add audited case00-tilt-gap results and optional recorded video to the UI.

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

VERSION = 'sircus-case00-tilt-gap-study-v1'
AUDIT = 'sircus-case00-tilt-gap-audit-v1'
VIDEO = 'sircus-case00-tilt-gap-recorded-video-v1'
KEY = 'case00_tilt_gap_study'
START = '<!--CASE00_TILT_GAP_START-->'
END = '<!--CASE00_TILT_GAP_END-->'


def require(value, message):
    if not value:
        raise ValueError(message)


def compact(study, audit_path=None):
    path = report_path(study)
    raw = json.loads(path.read_text())
    require(raw.get('version') == VERSION, 'Explicit case00-tilt-gap study required')
    out = take(raw, ('version', 'status', 'phase', 'contract_fingerprint', 'description',
                    'interpretation', 'cpu_elapsed_s', 'native_elapsed_s', 'native_requested',
                    'requested_configurations', 'error', 'training', 'noise',
                    'posthoc_development', 'admission_matrix_sha256'))
    out['parent_bindings'] = take(raw.get('parent_bindings'), BINDINGS)
    out['grid_protocol'] = take(raw.get('grid_protocol'), (
        'contexts', 'tilts_deg', 'gap_offsets_m',
        'fixed_parameters', 'changed_parameters', 'candidates_per_case', 'total_candidates',
        'substeps', 'nominal_repeats', 'selection_rule', 'selection_requires_capture',
        'selection_maximum', 'selection_frozen_before_repeats',
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
        for cell, original in zip(item['grid'], case.get('grid', [])):
            if cell['discovery'] is not None:
                cell['discovery']['invalid'] = original['discovery'].get('invalid')
        item['selection'] = [take(selected, (
            'candidate_id', 'grid_index', 'case_grid_index', 'values', 'row_sha256',
            'connection_sha256', 'route_quality', 'learning_reward', 'capture', 'diagnostic_robot_contact_steps'))
            for selected in case.get('selection', [])]
        item['candidate_repeats'] = [dict(take(row, (
            'candidate_id', 'grid_index', 'values', 'discovery_record_sha256')),
            summary=take(row.get('summary'), COUNTS)) for row in case.get('candidate_repeats', [])]
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
    require(raw.get('version') == VIDEO and raw.get('status') == 'completed', 'Completed Case00 replay required')
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


def cell_status(cell):
    """Never turn unexecuted or invalid physics into an observed flight miss."""
    admission, discovery = cell['admission'], cell['discovery']
    if discovery is None:
        return ('D', 'Setup declined', 'declined') if admission.get('passed') is False else ('P', 'Pending native drop', 'pending')
    if discovery.get('invalid_physics') or discovery.get('invalid'):
        return 'X', 'Invalid native trial', 'invalid'
    if not discovery.get('native_release'):
        return 'D', 'Not released / setup declined', 'declined'
    if discovery.get('captured'):
        return 'C', 'Capture', 'capture'
    if discovery.get('robot_contact_steps', 0) > 0:
        return 'R', 'Miss with robot contact', 'contact'
    return 'M', 'Miss without robot contact', 'miss'


def fraction(summary):
    return str(summary.get('captures', 0))+'/'+str(summary.get('attempts', 0))


def matrix(case, protocol):
    esc = html.escape
    base = case['baseline_values']
    tilts, offsets = protocol['tilts_deg'], protocol['gap_offsets_m']
    cells = case['grid']
    repeats = {r['candidate_id']: r['summary'] for r in case['candidate_repeats']}
    content = '<div class="bc-table"><table class="ctg-matrix"><caption>Discovery outcomes: tilt rows × drop-gap columns. Each cell is one proposed configuration.</caption><thead><tr><th>Tilt ↓ / gap →</th>'
    content += ''.join('<th>'+number((base[4]+offset),1000,1)+' mm<br><span class="bc-meta">'+f'{offset*1000:+.0f}'+' mm</span></th>' for offset in offsets)+'</tr></thead><tbody>'
    used = set()
    for tilt in tilts:
        content += '<tr><th>'+number(tilt,1,0)+'°</th>'
        for offset in offsets:
            matching = [cell for cell in cells if abs(cell['values'][3]-tilt)<1e-9 and abs(cell['values'][4]-(base[4]+offset))<1e-9]
            require(len(matching)<=1, 'Duplicate tilt/gap matrix cell')
            if not matching:
                content += '<td class="ctg-pending">P</td>'
                continue
            cell = matching[0]
            used.add(cell['candidate_id'])
            code, label, style = cell_status(cell)
            identifier = cell['grid_index']
            require(type(identifier) is int and identifier>=0, 'Integer grid index required')
            reason = cell['admission'].get('reason') or ''
            tooltip = f"Candidate {identifier}: {label}; tilt {tilt:g}°; gap {(base[4]+offset)*1000:.3f} mm. {reason}"
            fresh = repeats.get(cell['candidate_id'])
            detail = '<small>fresh '+fraction(fresh)+'</small>' if fresh is not None else ''
            content += f'<td class="ctg-{style}" title="{esc(tooltip,quote=True)}"><a href="#ctg-cell-{identifier}" aria-label="{esc(tooltip,quote=True)}">{code}{detail}</a></td>'
        content += '</tr>'
    require(len(used)==len(cells), 'Every proposal must appear in the tilt/gap matrix')
    return content+'</tbody></table></div><p class="bc-meta">C capture · R miss with robot contact · M miss without robot contact · D setup declined (no flight) · X invalid trial · P pending. “fresh” shows separate repeat captures/requests, never added to the discovery result. Click any cell for its exact settings.</p>'


def render_panel(snapshot, movie_bytes=None):
    esc = html.escape
    audit = snapshot['audit']
    verified = snapshot.get('status')=='completed' and audit.get('status')=='passed'
    status = 'Completed · independent native audit passed.' if verified else 'Provisional · '+str(snapshot.get('phase',snapshot.get('status','pending')))+'; final audit pending.'
    admission, discovery = snapshot['admission_summary'], snapshot['discovery_summary']
    stats = table(['Proposals','Admitted','Setup declined','Discovery releases','Discovery captures','Discovery robot contacts'], [[
        admission.get('proposals','—'),admission.get('admitted','—'),admission.get('declined','—'),
        discovery.get('native_releases','—'),discovery.get('captures','—'),discovery.get('robot_contact_shots','—')]])
    cases = []
    for case in snapshot['cases']:
        base = case['baseline_values']; bucket = case['bucket_xy_m']
        frozen = table(['Board center X / Y / Z (mm)','Bucket center X / Y (mm)','Starting tilt / gap'], [[
            ' / '.join(number(v,1000,3) for v in base[:3]),' / '.join(number(v,1000,3) for v in bucket),
            number(base[3],1,1)+'° / '+number(base[4],1000,3)+' mm']])
        baseline = case['repeats'].get('baseline',{})
        repeats = [['Unchanged starting setting',number(base[3],1,1),number(base[4],1000,3),'Reference',fraction(baseline),
                    baseline.get('native_releases',0),baseline.get('setup_rejections',0),baseline.get('robot_contact_shots',0)]]
        for row in case['candidate_repeats']:
            r, values = row['summary'], row['values']
            repeats.append([row['candidate_id'],number(values[3],1,1),number(values[4],1000,3),'1/1 discovery',fraction(r),
                r.get('native_releases',0),r.get('setup_rejections',0),r.get('robot_contact_shots',0)])
        selection_text = (f"{len(case['selection'])} discovery captures selected; every qualifying setting receives three fresh nominal repeats."
            if case['selection'] else ('No discovery captures qualified for fresh repeats.' if verified else 'Discovery and repeat selection are pending.'))
        details = '<div class="bc-table"><table><thead><tr>'+''.join('<th>'+h+'</th>' for h in ['Candidate','Tilt °','Drop gap mm','Gap change mm','Admission','Discovery outcome','Robot-contact ticks','Direct / table-assisted capture'])+'</tr></thead><tbody>'
        for cell in case['grid']:
            d = cell['discovery'] or {}; a = cell['admission']; values = cell['values']
            entries = [cell['candidate_id'],number(values[3],1,1),number(values[4],1000,3),number(values[4]-base[4],1000,1),
                'Passed' if a.get('passed') is True else str(a.get('reason') or a.get('status','Pending')),
                cell_status(cell)[1],d.get('robot_contact_steps','—'),
                ('Yes' if d.get('direct_capture') else 'No')+' / '+('Yes' if d.get('table_assisted_capture') else 'No') if d else '—']
            details += '<tr id="ctg-cell-'+str(cell['grid_index'])+'">'+''.join('<td>'+esc(str(v))+'</td>' for v in entries)+'</tr>'
        details += '</tbody></table></div>'
        cases.append(f'<h3>{esc(case["context_id"])} · fixed setup</h3>'+frozen+matrix(case,snapshot['grid_protocol'])+
            '<h3>Fresh nominal repeats, separated from discovery</h3><p>'+esc(selection_text)+'</p>'+table(
                ['Setting','Tilt °','Gap mm','Discovery','Fresh captures / requests','Fresh releases','Fresh setup declines','Fresh robot-contact drops'],repeats)+
            '<details><summary>All 70 proposals and exact settings</summary>'+details+'</details>')
    contacts = ''
    if snapshot.get('contact_diagnostics'):
        contacts = '<details><summary>Measured contact timing · every native drop</summary><p>Robot contacts before the first bucket/table contact are separated from later contacts. Events use recorded force above 0.01 N through the scoring terminal.</p>'+table(
            ['Record','Capture','First board / robot / bucket / table (s)','Robot-contact ticks before / after bucket or table','Scored through (s)'],
            [[d['record_id'],'Yes' if d['capture'] else 'No',
              ' / '.join(number(d.get('first_'+k+'_contact_s'),1,3) for k in ('board','robot','bucket','table')),
              str(d['robot_contact_steps_before_first_bucket_or_table'])+' / '+str(d['robot_contact_steps_from_first_bucket_or_table']),
              number(d['scoring_terminal_s'],1,3)] for d in snapshot['contact_diagnostics']])+'</details>'
    controls = [[name,g.get('status','pending'),g.get('direct_captures','—'),g.get('table_assisted_captures','—'),g.get('negative_captures','—')]
                for name,g in snapshot['control_gates'].items()]
    video = ''
    if snapshot.get('video'):
        media = snapshot['video']
        require(movie_bytes is not None and hashlib.sha256(movie_bytes).hexdigest()==media['output_sha256'],'Verified movie bytes required')
        encoded = base64.b64encode(movie_bytes).decode('ascii')
        chapters = ''.join(f'<button type="button" data-bc-time="{c["start_s"]}">{esc(c["label"])}</button>' for c in media['clips'])
        video = f'<h3>Actual recorded drops</h3><p>Saved native ball samples and held arm poses, with no new physics or invented motion. The full scored interval plays at normal speed, followed by a slow impact view. Selected examples do not replace the complete matrix above.</p><video controls playsinline preload="metadata" aria-label="Case00 tilt and gap recorded comparison"><source src="data:video/mp4;base64,{encoded}" type="video/mp4"></video><div class="bc-controls">{chapters}</div><div class="bc-controls"><label>Playback speed <select aria-label="Case00 replay speed"><option value="1">1× (includes slow motion)</option><option value="0.5">½×</option><option value="0.25">¼×</option></select></label><a data-bc-download download="sircus-case00-tilt-gap.mp4">Download MP4</a></div>'
        video += table(['Recorded example','Outcome','Events (seconds after release)'],[[c['label'],'Capture' if c['capture'] else 'Miss',
            ' → '.join(f'{e["label"]} {e["time_s"]:.3f}s' for e in c['events'])] for c in media['clips']])
    return f'''<section id="case00TiltGapStudy" aria-label="Case00 tilt and gap study"><p class="bc-meta">CASE 00 · FIXED 64 SUBSTEPS · TARGETED EXPLORATION</p><h2>From a clear rebound to a bucket capture</h2><p class="bc-status">{esc(status)}</p><p>Seventy tilt/drop-gap proposals explore one saved contact-free setup. Board XYZ and bucket XY stay fixed; tilt and release gap vary. Failed trials remain learning data. This is posthoc search, with no PPO training or policy promotion.</p>{stats}{''.join(cases)}<p>All discovery captures receive fresh repeats. Three successes in three unchanged trials would demonstrate nominal repetition only, not guaranteed reliability, noise robustness or performance at other bucket positions. Admission is not proof of a clear flight; robot contact and capture remain separate outcomes.</p>{video}{contacts}<h3>Unchanged bracketing controls</h3>{table(['Gate','Status','Direct captures','Table-assisted captures','Negative-control captures'],controls)}<details><summary>Evidence and limits</summary><p class="bc-meta">Report SHA256 {esc(str(audit.get('report_sha256','pending')))}<br>Audit SHA256 {esc(str(audit.get('proof_sha256','pending')))}<br>Contract {esc(str(snapshot.get('contract_fingerprint','pending')))}<br>Independent checks: {esc(str(audit.get('checks_count','pending')))}</p><p>Arms are initialized at held endpoints; connecting motions and pickup are not trained here. Board → table → bucket and rim-assisted captures remain valid after one continuous second fully inside, with unchanged v3 scoring. Historical trials retain their original labels.</p></details></section>'''

CSS = '''#case00TiltGapStudy{max-width:1480px;margin:36px auto;padding:30px;background:#101f2a;border:1px solid #375663;border-radius:18px;box-sizing:border-box;color:#e0edf2;font:15px/1.55 system-ui,sans-serif;scroll-margin-top:20px}#case00TiltGapStudy h2{font-size:28px;line-height:1.2}#case00TiltGapStudy h3{margin-top:28px}#case00TiltGapStudy .bc-meta{font-size:12px;color:#adc7d3;overflow-wrap:anywhere}#case00TiltGapStudy .bc-status{color:#8ee0c4;font-weight:600}#case00TiltGapStudy table{border-collapse:collapse;width:100%;font-size:13px}#case00TiltGapStudy th,#case00TiltGapStudy td{text-align:left;padding:10px;border-bottom:1px solid #304a59;vertical-align:top}#case00TiltGapStudy .bc-table{overflow:auto}#case00TiltGapStudy details{margin:16px 0}#case00TiltGapStudy summary{cursor:pointer}#case00TiltGapStudy video{display:block;width:100%;max-height:80vh;background:#080d12;border-radius:10px}#case00TiltGapStudy .bc-controls{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin:14px 0}#case00TiltGapStudy button,#case00TiltGapStudy select{background:#1b303e;color:#f0f7fa;border:1px solid #5b7c91;border-radius:6px;padding:9px;cursor:pointer}#case00TiltGapStudy a{color:#a5e7dc}@media(max-width:700px){#case00TiltGapStudy{margin:20px 8px;padding:18px}#case00TiltGapStudy h2{font-size:23px}}'''
CSS += """#case00TiltGapStudy .ctg-matrix caption{text-align:left;margin:12px 0}#case00TiltGapStudy .ctg-matrix td{text-align:center;min-width:65px}#case00TiltGapStudy .ctg-matrix td a{display:block;color:inherit;font-weight:750;text-decoration:none}#case00TiltGapStudy .ctg-matrix small{display:block;font-size:10px;font-weight:400}#case00TiltGapStudy .ctg-capture{background:#1e5b47}#case00TiltGapStudy .ctg-contact{background:#70403c}#case00TiltGapStudy .ctg-miss{background:#374e67}#case00TiltGapStudy .ctg-declined{background:#343940}#case00TiltGapStudy .ctg-invalid{background:#754d14}#case00TiltGapStudy .ctg-pending{background:#15232c}#case00TiltGapStudy tr:target{outline:2px solid #8fe4ce;scroll-margin-top:30px}"""

JS = '''(function(){const panel=document.getElementById('case00TiltGapStudy');for(const link of panel.querySelectorAll('.ctg-matrix a'))link.addEventListener('click',()=>{const target=document.getElementById(link.getAttribute('href').slice(1));if(target){const details=target.closest('details');if(details)details.open=true;}});const video=panel.querySelector('video');if(!video)return;for(const b of panel.querySelectorAll('[data-bc-time]'))b.addEventListener('click',()=>{video.currentTime=Number(b.dataset.bcTime);video.play().catch(()=>{});});panel.querySelector('select').addEventListener('change',e=>{video.playbackRate=Number(e.target.value);});panel.querySelector('[data-bc-download]').href=video.querySelector('source').src;})();'''


def embed_html(original, snapshot, movie_bytes=None):
    require(original.count(START)==original.count(END) and original.count(START)<=1, 'Ambiguous study markers')
    data=json.dumps(snapshot,allow_nan=False).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    block=START+'\n<style>'+CSS+'</style>\n'+render_panel(snapshot,movie_bytes)+'\n<script type="application/json" id="case00-tilt-gap-snapshot">'+data+'</script>\n<script>'+JS+'</script>\n'+END
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
