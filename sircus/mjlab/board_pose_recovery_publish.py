"""Publish audited board-pose search results and verified recorded-flight media.

Only derived display files change. This module never runs physics or changes a
planner. Historical results and every original native trace remain untouched.
"""
from __future__ import annotations
import argparse
import base64
import json
from pathlib import Path
import re

from .hybrid_miss_publish import read, sha, require, atomic_text, MAX_VIDEO_BYTES

VERSION = 'sircus-board-pose-recovery-v1'
CASES = tuple('stalled-reference-test-' + x for x in ('03', '09', '19'))
ASSET_PATTERN = re.compile(r'(<script\b(?=[^>]*\bid=["\']boardPoseVideoAssets["\'])[^>]*>)(.*?)(</script>)', re.S)


def settings(request):
    if request is None: return None
    v = request['values']
    return dict(board_xy_m=v[:2], board_height_m=v[2], tilt_deg=v[3], drop_gap_m=v[4],
        heading_offset_deg=request.get('board_heading_offset_deg', 0), twist_deg=request.get('board_twist_deg', 0),
        q10=request['selected_q10'], q14=request['selected_q14'])


def compact_result(raw):
    if raw is None: return None
    keys = ('requested', 'completed', 'released', 'captures', 'robot_contacts', 'outcomes')
    row = {k: raw[k] for k in keys}
    row['direct'] = raw['outcomes'].count('direct')
    row['table'] = raw['outcomes'].count('table')
    row['arm_assisted'] = raw['outcomes'].count('arm_assisted_capture')
    return row


def selected_request(case, results):
    candidates = {r['id']: r for r in case.get('candidates', [])}
    best = results.get('best_candidate')
    if isinstance(best, str): return candidates[best]
    if isinstance(best, dict):
        request = best.get('request', best)
        require(request['id'] in candidates and candidates[request['id']] == request,
                'Selected shot is not a frozen candidate')
        return request
    return None


def compose(study, audit_path=None):
    study = Path(study)
    plan, report = read(study/'preparation.json'), read(study/'report.json')
    require(plan['version'] == report['version'] == VERSION, 'Unsupported board-pose study')
    require(report['physics_substeps'] == 64, 'Expected native64 study')
    require([c['id'] for c in plan['cases']] == list(CASES), 'Expected the three diagnosed targets')
    audit = read(audit_path) if audit_path and Path(audit_path).exists() else {}
    verified = audit.get('status') == 'passed'
    if verified:
        require(audit.get('report_sha256') == sha(study/'report.json'), 'Audit does not bind report')
        require(audit.get('plan_sha256') == sha(study/'preparation.json'), 'Audit does not bind preparation')
        require(audit.get('evidence_manifest_sha256') == sha(study/'evidence-manifest.json'), 'Audit does not bind evidence')
    groups = {r['id']: r for r in report.get('cases', [])}
    controls = report.get('controls', {})
    blocks = [controls.get(k) for k in ('controls_pre', 'controls_post')]
    available = [b for b in blocks if b is not None]
    control_status = ('passed' if len(available) == 2 and all(b.get('status') == 'passed' for b in available)
                      else 'failed' if any(b.get('status') == 'failed' for b in available) else 'pending')
    section = dict(version=VERSION, status=report['status'], phase=report['phase'], physics_substeps=64,
        audit_status='passed' if verified else 'pending', progress=dict(report.get('progress', {})),
        controls=dict(status=control_status, completed=9*len(available), planned=18), cases=[],
        next='Review clear, repeated captures before any planner update. This search does not retrain a model or change the existing 17/20 planner result.',
        provenance=dict(preparation_sha256=sha(study/'preparation.json'), report_sha256=sha(study/'report.json'),
                        audit_sha256=sha(audit_path) if audit else None))
    if report.get('error'): section['error'] = str(report['error'])
    for case in plan['cases']:
        found = groups.get(case['id'], {}); groups_here = found.get('results', {})
        selected = selected_request(case, found)
        confirmed = found.get('confirmed_settings', [])
        section['cases'].append(dict(id=case['id'], bucket_xy_m=case['original']['bucket_xy_m'],
            preparation=dict(proposed=len(case.get('proposals', [])), checked=len(case.get('records', [])),
                             admitted=len(case.get('admitted', [])), selected=len(case.get('candidates', []))),
            settings=dict(original=settings(case['original']), selected=settings(selected)),
            selected_id=selected['id'] if selected else None,
            confirmed_settings=[dict(candidate_id=r['candidate_id'], repeat_captures=r['repeat_captures'],
                repeat_completed=r['repeat_completed'], settings=settings(r['request'])) for r in confirmed],
            results={k: compact_result(groups_here.get(k)) for k in ('original', 'search', 'repeat')}, videos={}))
    return section, plan, report


def media_plan(section, report):
    """Deterministic media selection, separate from search/capture accounting."""
    groups = {r['id']: r for r in report.get('cases', [])}; tasks = []
    for case in section['cases']:
        results = groups.get(case['id'], {}).get('results', {})
        original = results.get('original', {}).get('rows', [])
        selected = case['selected_id']
        chosen = [r for r in results.get('repeat', {}).get('rows', []) if r.get('candidate_id') == selected]
        if not chosen:
            chosen = [r for r in results.get('search', {}).get('rows', []) if r.get('candidate_id', r['id']) == selected]
        for variant, rows in (('original', original), ('selected', chosen)):
            rows = [r for r in rows if r.get('native_release') is True and r.get('invalid_physics') is False]
            if not rows: continue
            row = sorted(rows, key=lambda r: (r.get('repeat', 0), r['id']))[0]
            label = ('Original missed setting' if variant == 'original' else 'Selected board-pose search setting')
            tasks.append(dict(stem=f"case{case['id'][-2:]}-{variant}", variant=variant, case_id=case['id'],
                record_id=row['id'], trace_sha256=row['native_cache_sha256'],
                label=f"Case {case['id'][-2:]} / {label}",
                caption='Recorded native64 ball samples; arms held at endpoints. Development search, not a new policy score.'))
    return tasks


def media_assets(section, report, media_dir):
    assets = {}; cases = {r['id']: r for r in section['cases']}
    for task in media_plan(section, report):
        movie = Path(media_dir)/(task['stem']+'.mp4'); manifest = movie.with_suffix('.json')
        if not movie.exists() and not manifest.exists(): continue
        require(movie.exists() and manifest.exists(), 'Incomplete video pair: '+task['stem'])
        receipt = read(manifest)
        require(receipt['status'] == 'completed' and receipt['native_steps_added'] == 0 and receipt['held_arm_poses'] is True,
                'Video is not a saved held-endpoint replay')
        require(receipt['output_sha256'] == sha(movie), 'Video bytes changed')
        require(movie.stat().st_size <= MAX_VIDEO_BYTES, 'Video exceeds inline limit')
        require(receipt['record_id'] == task['record_id'] and receipt['context_id'] == task['case_id'], 'Wrong recorded shot')
        require(receipt['trace_sha256'] == task['trace_sha256'], 'Wrong native trace')
        assets[task['stem']] = dict(url='data:video/mp4;base64,'+base64.b64encode(movie.read_bytes()).decode('ascii'),
                                    trace_sha256=task['trace_sha256'])
        cases[task['case_id']]['videos'][task['variant']] = dict(asset_id=task['stem'], trace_sha256=task['trace_sha256'],
            label=receipt['label'], video_sha256=receipt['output_sha256'], manifest_sha256=sha(manifest))
    return assets


def embed_html(html, assets):
    require(len(ASSET_PATTERN.findall(html)) == 1, 'Expected exactly one owned board-pose video placeholder')
    payload = json.dumps(assets, separators=(',', ':'), allow_nan=False).replace('<', '\\u003c')
    return ASSET_PATTERN.sub(lambda m: m[1]+payload+m[3], html, count=1)


def publish(a):
    section, plan, report = compose(a.study, a.audit)
    tasks = media_plan(section, report)
    assets = media_assets(section, report, a.media_dir) if a.media_dir else {}
    protected = [a.study/'report.json', a.study/'preparation.json', a.study/'evidence-manifest.json', a.audit]
    outputs = [p for p in (a.display_report, a.runtime_index, a.video_plan) if p is not None]
    require(len({p.resolve() for p in outputs}) == len(outputs), 'Derived outputs must be distinct')
    require(all(p is None or target.resolve() != p.resolve() for p in protected for target in outputs), 'Outputs must be separate from evidence')
    display = read(a.display_report) if a.display_report.exists() else {}
    display['board_pose_recovery'] = section
    html = embed_html(a.runtime_index.read_text(), assets) if a.runtime_index else None
    if html is not None: atomic_text(a.runtime_index, html)
    atomic_text(a.display_report, json.dumps(display, indent=2, allow_nan=False)+'\n')
    if a.video_plan: atomic_text(a.video_plan, json.dumps(tasks, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status=section['status'], audit_status=section['audit_status'], cases=len(section['cases']), videos=len(assets))))
    return section


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('study', 'display-report'): p.add_argument('--'+name, type=Path, required=True)
    for name in ('audit', 'media-dir', 'runtime-index', 'video-plan'): p.add_argument('--'+name, type=Path)
    return publish(p.parse_args())


if __name__ == '__main__': main()
