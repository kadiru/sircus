"""Publish derived three-miss UI data and inline saved-trace movies.

Reads preserved evidence. Does not execute physics, modify native results or
restart services. Only the supplied derived display and one owned HTML asset
placeholder may be changed, after all inputs and media have been validated.
"""
from __future__ import annotations
import argparse
import base64
import hashlib
import json
import math
from pathlib import Path
import re

VERSION = 'sircus-hybrid-miss-correction-v1'
CASES = tuple('stalled-reference-test-' + x for x in ('03', '09', '19'))
ASSET_PATTERN = re.compile(r'(<script\b(?=[^>]*\bid=["\']hybridMissVideoAssets["\'])[^>]*>)(.*?)(</script>)', re.S)
MAX_VIDEO_BYTES = 10 * 1024 * 1024


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def settings(request):
    if request is None:
        return None
    values = request['values']
    return dict(board_xy_m=values[:2], board_height_m=values[2], tilt_deg=values[3],
                drop_gap_m=values[4], q10=request['selected_q10'], q14=request['selected_q14'])


def normalized_result(raw, *, declined, study_completed, repeats):
    if raw is None:
        return None
    fields = ('requested', 'completed', 'released', 'captures', 'direct', 'table', 'robot_contacts', 'outcomes')
    result = {k: raw[k] for k in fields}
    require(result['requested'] == repeats, 'Repeat denominator changed')
    if declined:
        require(result['released'] == result['captures'] == 0 and not result['outcomes'],
                'Unadmitted correction has native outcomes')
        if study_completed:
            require(raw.get('setup_rejected') == repeats, 'Missing explicit setup-decline accounting')
            result.update(completed=repeats, outcomes=['setup_rejected'] * repeats)
    return result


def diagnosis_text(row):
    events = row['observed_events']
    robot = next((e for e in events if e['channel'] == 'robot'), None)
    pairs = row.get('reconstructed_first_robot_contacts', [])
    link = next((name for pair in pairs for name in pair['geometries'] if name.startswith('Arm14/')), 'board arm')
    if robot is None:
        return 'No robot contact is recorded in the saved flight; inspect the contact timeline.'
    sentence = f"After the board rebound, the ball hit {link} at {robot['start_s']:.3f} s. It never entered the bucket completely."
    conditional = row['conditional_no_robot_flight']
    route = row['original_prediction'].get('predicted_route') or 'direct'
    margin = conditional.get(route, {}).get('conservative_rim_band_clearance_m')
    if margin is not None:
        sentence += (f" Ignoring robot contact, a simplified {route} trajectory has {margin*1000:+.1f} mm rim clearance"
                     ' (a prediction, not another simulated drop).')
    return sentence


def flatten_controls(raw, completed):
    blocks = [raw.get(name) for name in ('controls_pre', 'controls_post')]
    available = [row for row in blocks if row is not None]
    status = 'passed' if len(available) == 2 and all(row.get('status') == 'passed' for row in available) else 'pending'
    if any(row.get('status') == 'failed' for row in available): status = 'failed'
    return dict(status=status, completed=9 * len(available), planned=18)


def compose(study, diagnosis_path, audit_path=None):
    study = Path(study)
    plan, report, diagnosis = read(study/'preparation.json'), read(study/'report.json'), read(diagnosis_path)
    require(plan['version'] == report['version'] == VERSION, 'Unsupported correction study')
    require(diagnosis['version'] == 'sircus-hybrid-miss-diagnosis-v1' and diagnosis['status'] == 'completed', 'Incomplete original diagnosis')
    require(report['physics_substeps'] == 64, 'Expected native64 study')
    require([c['id'] for c in plan['cases']] == list(CASES), 'Expected exactly the three diagnosed misses')
    drows = {row['id'].removesuffix('-hybrid'): row for row in diagnosis['cases']}
    require(set(drows) == set(CASES), 'Diagnosis targets do not match')
    audit = read(audit_path) if audit_path and Path(audit_path).exists() else {}
    audit_passed = audit.get('status') == 'passed'
    if audit_passed:
        require(audit.get('report_sha256') == sha(study/'report.json'), 'Audit does not bind current report')
        require(audit.get('plan_sha256') == sha(study/'preparation.json'), 'Audit does not bind frozen preparation')
    completed = report['status'] == 'completed'
    results = {row['id']: row for row in report.get('cases', [])}
    section = dict(version=VERSION, status=report['status'], phase=report['phase'],
        physics_substeps=64, audit_status='passed' if audit_passed else 'pending',
        progress=dict(report['progress']), controls=flatten_controls(report.get('controls', {}), completed),
        correction=dict(parameter='joint_configuration', label='Change only how the board arm bends',
            description='Keep bucket, full board pose, release point and dropper joints fixed. Choose an alternative board-arm joint configuration before new drops, using predicted rebound clearance.'),
        cases=[], next=('Review the matched repeat results before changing the live planner. No retraining or automatic promotion.'
                       if completed else 'Saved original videos are available. The correction test will run only when the shared 4090 is available.'),
        provenance=dict(preparation_sha256=sha(study/'preparation.json'), report_sha256=sha(study/'report.json'),
                        diagnosis_sha256=sha(diagnosis_path), audit_sha256=sha(audit_path) if audit else None))
    if report.get('error'): section['error'] = str(report['error'])
    for source in plan['cases']:
        key = source['id']; d = drows[key]; original, corrected = source['original'], source['corrected']
        require(original['values'] == d['values'] and original['bucket_xy_m'] == d['bucket_xy_m'], 'Original geometry differs from diagnosed trace')
        if corrected:
            for field in ('values', 'bucket_xy_m', 'selected_q10', 'board_heading_offset_deg', 'board_twist_deg'):
                require(original[field] == corrected[field], 'More than board-arm joints changed')
        group = results.get(key, {}).get('results', {})
        item = dict(id=key, bucket_xy_m=original['bucket_xy_m'],
            diagnosis=dict(summary=diagnosis_text(d), evidence_sha256=d['trace']['sha256'],
                events=[dict(label=e['channel']+' contact', time_s=e['start_s']) for e in d['observed_events']]),
            settings=dict(original=settings(original), corrected=settings(corrected)),
            modification=('Same Cartesian shot and dropper joints; different board-arm joint configuration.' if corrected
                          else 'No alternative board-arm configuration passed the unchanged checks within this bounded preparation. No corrected ball release.'),
            results={method: normalized_result(group.get(method), declined=source[method] is None,
                study_completed=completed, repeats=plan['repeats']) for method in ('original', 'corrected')}, videos={})
        if item['settings']['corrected']:
            proximal_change = max(abs(math.remainder(a-b, 2*math.pi)) for a,b in
                                  zip(original['selected_q14'][:3], corrected['selected_q14'][:3]))
            if proximal_change < 1e-4:
                item['modification'] += (' The accepted alternative changes the wrist configuration; joints 1–3 differ by less than 0.0001 rad.'
                                         ' This is not expected to move the obstructing middle link out of the ball path.')
            route = source['route_for_ranking']
            before = source['original_prediction'].get(route, {}).get('arm_clearance_m')
            after = source['predictions'][corrected['id']].get(route, {}).get('arm_clearance_m')
            if before is not None and after is not None:
                item['settings']['corrected']['description'] = f'Predicted {route}-route arm clearance: {before*1000:+.2f} → {after*1000:+.2f} mm. Not a measured capture.'
        section['cases'].append(item)
    if completed:
        section['progress'] = dict(completed=sum((c['results'][m] or {}).get('completed',0) for c in section['cases'] for m in ('original','corrected')),
                                   planned=6*plan['repeats'])
    return section, plan, report, diagnosis


def media_assets(section, plan, report, diagnosis, media_dir):
    """Only verified local MP4s are embedded; no network URLs or base64 in report."""
    assets = {}; originals = {r['id'].removesuffix('-hybrid'): r for r in diagnosis['cases']}
    groups = {row['id']: row for row in report.get('cases', [])}
    for case in section['cases']:
        for variant in ('original', 'corrected'):
            stem = f"case{case['id'][-2:]}-{variant}"; movie = Path(media_dir)/(stem+'.mp4'); manifest = movie.with_suffix('.json')
            if not movie.exists() and not manifest.exists(): continue
            require(movie.exists() and manifest.exists(), 'Incomplete video pair: '+stem)
            receipt = read(manifest)
            require(receipt['status'] == 'completed' and receipt['native_steps_added'] == 0 and receipt['held_arm_poses'] is True,
                    'Video is not a saved held-endpoint replay')
            require(receipt['output_sha256'] == sha(movie), 'Video bytes changed: '+stem)
            require(movie.stat().st_size <= MAX_VIDEO_BYTES, 'Movie exceeds inline size limit')
            require(receipt['context_id'] == case['id'], 'Video belongs to another bucket case')
            if variant == 'original':
                expected = originals[case['id']]['trace']['sha256']
                require(receipt['record_id'] == originals[case['id']]['id'], 'Wrong original recording')
            else:
                rows = groups.get(case['id'], {}).get('results', {}).get('corrected', {}).get('rows', [])
                matched = [r for r in rows if r['id'] == receipt['record_id']]
                require(len(matched) == 1, 'Corrected video is not from the correction study')
                expected = matched[0]['native_cache_sha256']
            require(receipt['trace_sha256'] == expected, 'Video native trace differs')
            payload = 'data:video/mp4;base64,' + base64.b64encode(movie.read_bytes()).decode('ascii')
            assets[stem] = dict(url=payload, trace_sha256=expected)
            case['videos'][variant] = dict(asset_id=stem, trace_sha256=expected, label=receipt['label'],
                                          video_sha256=receipt['output_sha256'], manifest_sha256=sha(manifest))
    return assets


def embed_html(html, assets):
    require(len(ASSET_PATTERN.findall(html)) == 1, 'Expected exactly one owned video placeholder; install the updated HTML first')
    payload = json.dumps(assets, separators=(',', ':'), allow_nan=False).replace('<', '\\u003c')
    return ASSET_PATTERN.sub(lambda m: m[1]+payload+m[3], html, count=1)


def atomic_text(path, text):
    target = Path(path); temp = target.with_name(target.name+'.hybrid-publish.tmp')
    temp.write_text(text); temp.replace(target)


def publish(a):
    section, plan, report, diagnosis = compose(a.study, a.diagnosis, a.audit)
    assets = media_assets(section, plan, report, diagnosis, a.media_dir) if a.media_dir else {}
    display = read(a.display_report) if a.display_report.exists() else {}
    display['hybrid_miss_correction'] = section
    index = a.runtime_index.read_text() if a.runtime_index else None
    updated_html = embed_html(index, assets) if index is not None else None
    protected = [a.study/'report.json', a.study/'preparation.json', a.diagnosis, a.audit]
    require(all(p is None or a.display_report.resolve() != p.resolve() for p in protected), 'Display output must be separate from evidence')
    # Compute everything before modifying either derived target. Native evidence stays untouched.
    if updated_html is not None: atomic_text(a.runtime_index, updated_html)
    atomic_text(a.display_report, json.dumps(display, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status=section['status'], audit_status=section['audit_status'], cases=len(section['cases']), videos=len(assets),
                          display=str(a.display_report), runtime_index=str(a.runtime_index) if a.runtime_index else None)))
    return section


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('study','diagnosis','display-report'): p.add_argument('--'+name,type=Path,required=True)
    for name in ('audit','media-dir','runtime-index'): p.add_argument('--'+name,type=Path)
    return publish(p.parse_args())


if __name__ == '__main__': main()
