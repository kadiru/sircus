"""Publish the audited fresh-position approach comparison, without promotion.

A pending snapshot contains progress only, never provisional capture rates.
All existing display panels and media remain unchanged.
"""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
from statistics import mean, median

from .benchmark_verify import contained, digest, read, sha
from .expanded_selector_publish import outcome

VERSION = 'sircus-approach-comparison-display-v1'
STUDY_VERSION = 'sircus-approach-comparison-v1'
AUDIT_VERSION = 'sircus-approach-comparison-independent-audit-v1'
METHODS = ('current', 'expanded')
DISPLAY_METHOD = {'current':'current', 'expanded':'approach'}
PROTOCOL = ('Current interactive planner versus expanded candidate search plus full bucket-approach ranking. '
            'The proposed method uses the previously frozen expanded neural selector/library plus a bounded local tilt/drop-gap menu and approach ranking. '
            'Both frozen methods use the same decision deadline, arm setup/path checks, 64-substep physics '
            'and one attempted drop per target. Both include the same nested-IK adapter bug fix; the live service remains unchanged. '
            'Native commands are frozen before execution in batches of up to 32 independent worlds. '
            'This compares complete methods, not the bucket check alone. '
            'The current planner may penalize predicted later robot contacts while the proposed ranking does not; actual v4 outcome scoring permits those later contacts for both methods. '
            'No fitting or test-drop feedback changes selections. The forecast does not simulate arbitrary '
            'multi-bounce contact response; native board-first capture scoring remains unchanged.')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def percent90(values):
    """Prespecified nearest-rank 90th percentile (45th ordered value of 50)."""
    values=sorted(values)
    return values[max(0,math.ceil(.9*len(values))-1)]


def statistics(values):
    require(values and all(finite(v) and v>=0 for v in values), 'Finite nonnegative planning durations required')
    return dict(mean=mean(values), median=median(values), p90=percent90(values))


def contract_for(study):
    contract=read(study/'contract.json')
    require(contract.get('version')==STUDY_VERSION and contract.get('training_performed') is False,
            'Frozen comparison must declare no model fitting')
    require(contract.get('fingerprint') == digest({k:v for k,v in contract.items() if k!='fingerprint'}), 'Frozen contract changed')
    require((contract.get('methods'),contract.get('test_positions'),contract.get('physics_substeps'),
             contract.get('attempts_per_method_per_position'),contract.get('scoring_version')) ==
            (list(METHODS),50,64,1,'sircus-board-first-native-capture-v4'), 'Expected fifty paired native64 attempts with unchanged scoring')
    require(finite(contract.get('planning_budget_s')) and contract['planning_budget_s']>0, 'Positive planning deadline required')
    return contract


def compose(study, audit_path):
    study,audit_path=Path(study),Path(audit_path)
    report,audit=read(study/'report.json'),read(audit_path)
    require((report.get('version'),report.get('status'),report.get('phase'),audit.get('version'),audit.get('status')) ==
            (STUDY_VERSION,'completed','completed',AUDIT_VERSION,'passed'), 'Completed study and matching independent audit required')
    require(report.get('native_completed') is True and audit.get('frozen_before_test') is True
            and type(audit.get('checks_count')) is int and audit['checks_count']>0, 'Complete frozen native evidence must be audited')
    bindings={'report_sha256':'report.json','evidence_manifest_sha256':'evidence-manifest.json',
              'contract_sha256':'contract.json','choices_sha256':'choices.json','native_started_sha256':'native-started.json'}
    for key,name in bindings.items():
        require(audit.get(key)==sha(study/name), 'Audit no longer binds '+name)
    contract=contract_for(study)
    manifest,choices,started=read(study/'evidence-manifest.json'),read(study/'choices.json'),read(study/'native-started.json')
    for name,expected in manifest.items():
        require(sha(contained(study,name))==expected, 'Changed evidence '+name)
    for name in ('contract.json','layouts.json','plans.json','choices.json','rows.jsonl','native-started.json'):
        require(manifest.get(name)==sha(study/name),'Unbound comparison evidence '+name)
    for name,expected in contract['frozen_inputs'].items():
        require(sha(contained(study,name))==expected==choices['bindings'].get(name),'Frozen input changed '+name)
    expected=dict(choices['bindings'],**{'choices.json':sha(study/'choices.json')})
    require(started.get('bindings')==expected,'Choices were not bound before native execution')
    for name,expected_hash in expected.items():
        require(sha(contained(study,name))==expected_hash,'Native-start evidence changed '+name)
    layouts=read(study/'layouts.json')['layouts'];slots=choices['slots'];plans=read(study/'plans.json')
    require(len(layouts)==50 and len({r['id'] for r in layouts})==50
            and len({tuple(r['bucket_xy_m']) for r in layouts})==50
            and all(len(r['bucket_xy_m'])==2 and all(finite(v) for v in r['bucket_xy_m']) for r in layouts), 'Fifty distinct finite targets required')
    keyed={(s['context_id'],s['method']):s for s in slots}
    wanted={(r['id'],method) for r in layouts for method in METHODS}
    require(len(slots)==len(keyed)==len({s['id'] for s in slots})==100 and set(keyed)==wanted, 'One slot for every method and target required')
    timed={(p['site']['layout']['id'],p['variant']):p['site'] for p in plans}
    require(len(plans)==len(timed)==100 and set(timed)==wanted,'One measured planning request per target and method required')
    rows=[json.loads(line) for line in (study/'rows.jsonl').read_text().splitlines() if line.strip()]
    native={r['id']:r for r in rows if r['adapter_directory']!='native-controls'}
    require(len({r['id'] for r in rows})==len(rows) and len(rows)-len(native)==18
            and set(native)=={s['id'] for s in slots if s['request'] is not None},'Exactly one native drop per admitted slot plus eighteen controls required')
    for phase in ('controls_pre','controls_post'):
        gate=report.get('controls',{}).get(phase,{})
        require(gate.get('status')=='passed' and gate.get('board_first_passed') is True,'Both matched control gates must pass')
    require(report.get('comparison')==audit.get('comparison'),'Report and independent comparison differ')
    results=[];paired=dict(both=0,current_only=0,approach_only=0,neither=0)
    for layout in layouts:
        row=dict(id=layout['id'],bucket_xy_m=layout['bucket_xy_m'])
        for method in METHODS:
            slot=keyed[layout['id'],method];site=timed[layout['id'],method]
            require(slot['bucket_xy_m']==layout['bucket_xy_m'],'Attempt target changed')
            decision,request=site['decision_latency_s'],site['request_wall_s']
            require(finite(decision) and finite(request) and 0<=decision<=request,'Invalid measured request/decision time')
            row[DISPLAY_METHOD[method]]=dict(outcome=outcome(slot,native),decision_s=decision,request_s=request)
        old,new=row['current']['outcome']=='capture',row['approach']['outcome']=='capture'
        paired['both' if old and new else 'current_only' if old else 'approach_only' if new else 'neither']+=1
        results.append(row)
    expected_pair=dict(paired);expected_pair['expanded_only']=expected_pair.pop('approach_only')
    require(report['comparison']['paired']==expected_pair,'Paired gains/losses differ from recorded attempts')
    methods=[]
    for method in METHODS:
        key=DISPLAY_METHOD[method];column=[r[key] for r in results]
        counts={name:sum(r['outcome']==value for r in column) for name,value in
                (('captures','capture'),('setup_rejections','setup refused'),('released_misses','flight miss'),('invalid_physics','invalid physics'))}
        measured=report['comparison']['methods'][method]
        require(measured['positions']==50 and measured['releases']==50-counts['setup_rejections'], 'Release accounting differs')
        for key2,raw in (('captures','captures'),('setup_rejections','setup_rejections'),('released_misses','flight_misses'),('invalid_physics','invalid_physics')):
            require(measured[raw]==counts[key2],'Outcome accounting differs: '+raw)
        decision,request=statistics([r['decision_s'] for r in column]),statistics([r['request_s'] for r in column])
        for group,suffix in ((decision,'planning_s'),(request,'request_s')):
            for label,value in group.items():
                require(finite(measured.get(label+'_'+suffix)) and math.isclose(measured[label+'_'+suffix],value,rel_tol=1e-10,abs_tol=1e-10),'Measured planning statistic differs: '+label+'_'+suffix)
        checkpoint=sha(study/(method+'-checkpoint.json'))
        require(audit.get('checkpoint_sha256',{}).get(method)==checkpoint,'Frozen checkpoint identity changed')
        methods.append(dict(id=key,attempts=50,releases=measured['releases'],**counts,decision_s=decision,request_s=request,checkpoint_sha256=checkpoint))
    native_releases=18+sum(m['releases'] for m in methods)
    require(len(rows)==native_releases,'Total release count differs')
    wait,lease=report.get('gpu_wait_s'),report.get('gpu_lease_s')
    require(finite(wait) and wait>=0 and finite(lease) and lease>0,'GPU wait and execution time must be separate recorded durations')
    old,new=(m['captures'] for m in methods)
    return dict(version=VERSION,status='completed',phase='completed',audit_status='passed',positions=50,
        physics_substeps=64,attempts_per_position=1,frozen_before_test=True,planning_budget_s=contract['planning_budget_s'],
        methods=methods,results=results,paired=paired,controls_passed=18,native_releases=native_releases,
        gpu_wait_s=wait,gpu_lease_s=lease,checks_count=audit['checks_count'],protocol=PROTOCOL,
        conclusion=(f'Current planner {old}/50; expanded search plus full-approach check {new}/50 ({new-old:+d}). '
                    'This batch compares the complete methods; it does not isolate the check from the expanded search. The live planner remains unchanged.'),
        provenance=dict(report_sha256=sha(study/'report.json'),audit_sha256=sha(audit_path),
                        contract_sha256=sha(study/'contract.json'),manifest_sha256=sha(study/'evidence-manifest.json')))


def pending(study):
    study=Path(study);report=read(study/'report.json');contract=contract_for(study)
    require(report.get('version')==STUDY_VERSION,'Matching approach-comparison job required')
    require(report.get('status') in ('running','failed','completed'),'Known job status required')
    require(finite(report.get('updated_at_unix')),'A timestamped status snapshot is required')
    phase=report.get('phase','preparing');wait=report.get('gpu_wait',{})
    waiting=phase=='waiting_for_4090'
    if waiting:
        require(wait.get('status')=='waiting_for_4090' and finite(wait.get('wait_s')) and wait['wait_s']>=0
                and report.get('native_completed') is not True,'A genuine recorded GPU-wait event is required')
        require(finite(report.get('gpu_wait_s',wait['wait_s'])) and report.get('gpu_wait_s',wait['wait_s'])>=wait['wait_s'],
                'Cumulative GPU wait must cover the latest queue interval')
    prepared=report.get('progress',{}).get('prepared')
    text=(f'Prepared {prepared}/100 planning requests. ' if type(prepared) is int and 0<=prepared<=100 else '')
    text+='Final capture rates remain unavailable until native trials and the independent audit finish.'
    return dict(version=VERSION,status='failed' if report['status']=='failed' else 'pending',
        phase='awaiting_audit' if report['status']=='completed' else phase,audit_status='pending',
        queue_expired=report.get('queue_expired') is True,
        positions=50,physics_substeps=64,planning_budget_s=contract['planning_budget_s'],
        observed_at_unix=report['updated_at_unix'],progress_text=text,
        gpu_wait_s=report.get('gpu_wait_s',wait.get('wait_s')) if waiting else None,
        provenance=dict(report_sha256=sha(study/'report.json'),contract_sha256=sha(study/'contract.json')))


def publish(study,audit,display,*,pending_only=False):
    study,display=Path(study),Path(display);audit=Path(audit) if audit else None
    temporary=display.with_suffix(display.suffix+'.approach.tmp')
    require(not any(p.resolve().is_relative_to(study.resolve()) or (audit and p.resolve()==audit.resolve()) for p in (display,temporary)), 'Display must remain separate from immutable study/audit evidence')
    require(pending_only != bool(audit),'Supply an audit for final results or choose pending-only progress')
    section=pending(study) if pending_only else compose(study,audit)
    data=read(display) if display.exists() else {}
    require(not(pending_only and data.get('approach_comparison',{}).get('status')=='completed'), 'Pending progress cannot overwrite published completed results')
    data['approach_comparison']=section
    temporary.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');os.replace(temporary,display)
    return dict(status=section['status'],phase=section['phase'],positions=50)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True);p.add_argument('--audit',type=Path)
    p.add_argument('--display-report',type=Path,required=True);p.add_argument('--pending',action='store_true')
    a=p.parse_args();print(json.dumps(publish(a.study,a.audit,a.display_report,pending_only=a.pending)))


if __name__=='__main__':main()
