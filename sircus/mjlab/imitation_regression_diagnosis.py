"""Read-only native-evidence diagnosis of six corrected-imitation regressions.

Inspected historical TEST positions become development evidence. This builder
rescores saved trajectories on CPU; it does not run physics or fit a model.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
from .benchmark_verify import contained, digest, read, sha
from .replay650_verify import AuditChecks
from .table_training_verify import captured, check_native_score
from . import table_assisted_score_oracle as oracle
from .recovery_regression_diagnosis import compact, trace_summary, setup_evidence, contact_events, classify, PARAMETERS, SCOPE

VERSION='sircus-imitation-regression-diagnosis-v1'
AUDIT_VERSION='sircus-imitation-regression-diagnosis-audit-v1'
SOURCE=dict(report_sha256='1fde3f7de5b5c2912a133c858105388a704a80ab0e2cb39e8d79c0c7a189bf14',
 audit_sha256='ea4992828178d9e62ba41c63579ecc3876cb5bffaacf6b590c828d45876a7f63',
 rows_sha256='f9d48dd328e203b76d7f6be41c01e436ea938701821e902bbbd7945df7248843',
 manifest_sha256='e65451629a36e3c486c3c7dbedd5f2de67de97fac46f6d4631eb3c38c68f3957',
 contract_fingerprint='297b367fa2702d688084bc2e74dd30a83c6cf4312ba078938def5f9b223adce4')
LOSS_IDS=['corrected-imitation-test-'+s for s in ('r0-c0-n0','r0-c1-n0','r0-c6-n1','r1-c1-n0','r1-c2-n0','r3-c0-n1')]


def verify_inputs(study,audit,checks):
    study,audit=Path(study).resolve(),Path(audit).resolve()
    for key,path in (('report_sha256',study/'report.json'),('audit_sha256',audit),('rows_sha256',study/'rows.jsonl'),('manifest_sha256',study/'evidence-manifest.json')):
        checks.equal(sha(path),SOURCE[key],'Exact completed source/'+key)
    report,spec,proof,manifest=map(read,(study/'report.json',study/'contract.json',audit,study/'evidence-manifest.json'))
    checks.equal((report['version'],report['status'],report['phase']),('sircus-corrected-imitation-v1','completed','completed'),'Completed corrected imitation source')
    checks.equal((proof['version'],proof['status'],proof['full_checks_verified'],proof['native_checks_verified'],proof['failures']),('sircus-corrected-imitation-audit-v1','passed',True,True,[]),'Full native audit passed')
    checks.equal((proof['checks_count'],len(proof['checks'])),(55633,55633),'All original checks retained')
    checks.require(all(c.get('passed') is True for c in proof['checks']),'Every original check passed')
    checks.equal((proof['report_sha256'],proof['evidence_manifest_sha256']),(SOURCE['report_sha256'],SOURCE['manifest_sha256']),'Audit binds source bytes')
    unsigned=dict(spec);fp=unsigned.pop('fingerprint')
    checks.equal((digest(unsigned),fp,report['contract_fingerprint'],proof['contract_fingerprint']),(SOURCE['contract_fingerprint'],)*4,'Exact frozen contract')
    checks.equal(report['evidence_sha256'],manifest,'Source report binds manifest');checks.equal(spec['protocol']['substeps'],64,'Native64 unchanged')
    consumed={}
    def bound(name):
        path=contained(study,name);actual=sha(path);checks.equal(actual,manifest[name],'Authenticated source/'+name);consumed[name]=actual;return path
    bound('contract.json');rows=[json.loads(x) for x in bound('rows.jsonl').read_text().splitlines()]
    checks.equal((len(rows),len({r['id'] for r in rows})),(158,158),'Complete158 original requests')
    tests=[r for r in rows if r['phase']=='test'];checks.equal(Counter(r['method'] for r in tests),Counter(current_bc=70,corrected_bc=70),'Both complete70-position comparisons')
    by={(r['method'],r['context_id']):r for r in tests};checks.equal(len(by),140,'Unique paired outcomes')
    own=sorted([r for r in tests if r['method']=='corrected_bc'],key=lambda r:r['context_id'])
    losses=[r['context_id'] for r in own if not captured(r) and captured(by['current_bc',r['context_id']])]
    gains=[r['context_id'] for r in own if captured(r) and not captured(by['current_bc',r['context_id']])]
    checks.equal(losses,LOSS_IDS,'All six original regressions');checks.equal((sum(captured(r) for r in own),sum(captured(r) for r in tests if r['method']=='current_bc'),len(gains)),(38,40,4),'Historical results retained')
    calibration=read(bound('calibration.json'))['score_config']
    return study,spec,proof,calibration,rows,by,bound,consumed


def build(study,audit,output):
    output,study=Path(output).resolve(),Path(study).resolve()
    if output.exists() or output.is_relative_to(study):raise ValueError('Fresh diagnosis directory outside immutable source required')
    checks=AuditChecks();study,spec,proof,calibration,rows,by,bound,consumed=verify_inputs(study,audit,checks)
    selected=[by[method,context] for context in LOSS_IDS for method in ('current_bc','corrected_bc')];items={};groups={}
    for row in selected:
        connection=read(bound('native/'+row['connection_file']));resolution=read(bound('native/'+row['context_resolution_file']))
        checks.equal((consumed['native/'+row['connection_file']],consumed['native/'+row['context_resolution_file']]),(row['connection_sha256'],row['context_resolution_sha256']),row['id']+'/exact setup proofs')
        checks.equal((connection['bucket_xy_m'],connection['values'],connection['resolution_sha256']),(row['bucket_xy_m'],row['values'],row['context_resolution_sha256']),row['id']+'/same command and scene')
        checks.equal((connection['action_substituted'],connection['passed']),(False,row['native_release']),row['id']+'/no substituted action')
        if row['native_release']:
            checks.equal((connection['q10'],connection['q14']),(row['q10'],row['q14']),row['id']+'/exact branch')
            groups.setdefault('native/'+row['native_cache_file'],[]).append(row)
        items[row['id']]=compact(row,connection,resolution)
    for name,members in sorted(groups.items()):
        with np.load(bound(name),allow_pickle=False) as archive:
            ids=archive['record_ids'].tolist();arrays={k:archive[k] for k in archive.files if k!='record_ids'}
        for row in members:
            column=row['native_cache_row'];checks.equal((str(ids[column]),consumed[name]),(row['id'],row['native_cache_sha256']),row['id']+'/exact recorded trajectory')
            trace={k:v[:,column] if v.ndim>=2 and k!='time_s' else v for k,v in arrays.items()}
            score=oracle.score_trace(spec['scene'],row['bucket_xy_m'],row['board_normal'],calibration,trace);check_native_score(row,score,checks)
            items[row['id']].update(trace_summary(row,trace,spec['scene'],calibration))
    losses=[]
    for ident in LOSS_IDS:
        current,corrected=(items[by[m,ident]['id']] for m in ('current_bc','corrected_bc'))
        delta=(np.asarray(corrected['values'])-current['values']).tolist()
        checks.equal(current['bucket_xy_m'],corrected['bucket_xy_m'],ident+'/paired bucket')
        checks.equal(by['current_bc',ident]['effective_source_hint'],by['corrected_bc',ident]['effective_source_hint'],ident+'/common geometry anchor')
        losses.append(dict(context_id=ident,bucket_xy_m=corrected['bucket_xy_m'],current_bc=current,corrected_bc=corrected,
            delta_values=delta,delta_release_height_m=delta[2]+delta[4],same_anchor=True))
    rejects=[p['corrected_bc'] for p in losses if not p['corrected_bc']['native_release']];misses=[p['corrected_bc'] for p in losses if p['corrected_bc']['native_release']]
    checks.equal((len(rejects),len(misses)),(3,3),'Three finite IK declines and three released route failures')
    checks.equal(Counter(r['setup_proof']['group'] for r in rejects),Counter(finite_ik_dropper=1,finite_ik_board=2),'Recorded per-arm IK failures')
    for name,expected in list(consumed.items()):checks.equal(sha(contained(study,name)),expected,'Source bytes unchanged/'+name)
    checks.equal(sha(audit),SOURCE['audit_sha256'],'Source audit unchanged')
    source=dict(SOURCE,full_checks_verified=True,native_checks_verified=True,checks_count=proof['checks_count'],
        producer_commit='fecbe4d61bd92c912e0b4d80b1fdd7be3a2bcb48',diagnosis_source_sha256=sha(__file__),consumed_files_sha256=consumed,
        helper_source_sha256={name:sha(Path(__file__).with_name(name)) for name in ('recovery_regression_diagnosis.py','table_assisted_score_oracle.py','table_training_verify.py','xy650_verify.py')})
    recommendation=dict(mode='matched_current_action_midpoint',interpolation_alpha=.5,parameter_indices=list(range(5)),context_ids=LOSS_IDS,
        settings=[dict(context_id=p['context_id'],current_record_sha256=p['current_bc']['record_sha256'],corrected_record_sha256=p['corrected_bc']['record_sha256'],
            original_values=p['corrected_bc']['values'],blend_values=(np.asarray(p['corrected_bc']['values'])+.5*(np.asarray(p['current_bc']['values'])-p['corrected_bc']['values'])).tolist()) for p in losses],
        rule='Move each corrected action exactly halfway toward its matched successful current action in all five task coordinates.',
        rationale='Distinct finite IK, robot-contact and rim-flight regressions followed small joint action changes. A single conservative midpoint tests reduced action drift without selecting different corrections per outcome.',
        coupled_geometry='All five coordinates change jointly, including target-facing yaw and derived release Z; this does not isolate the causal effect of any one coordinate.',
        posthoc_development=True,new_training=False,additional_coordinate_search=False)
    result=dict(version=VERSION,status='completed',scope=SCOPE,source=source,parameter_order=PARAMETERS,
        summary=dict(requested=70,current_bc_captures=40,corrected_bc_captures=38,gains=4,losses=6,setup_rejections=3,flight_misses=3,
            failure_groups=dict(Counter(p['corrected_bc']['group'] for p in losses)),forbidden_contact_misses=sum(bool(r['forbidden_contact_steps']) for r in misses)),
        scene=spec['scene'],losses=losses,setup_rejections=rejects,flight_misses=misses,recommended_correction=recommendation,
        limitations=[SCOPE,'Six selected regressions are posthoc development cases, not a fresh policy score.',
            'Midpoint changes five coupled coordinates; successful correction would not identify a unique causal parameter.',
            'Finite IK refusal is not proof of unreachable geometry; recorded robot contacts are not named links without further collision reconstruction.',
            'One retained ball had a prior robot contact and remains a route failure. Table and rim assistance remain allowed.'])
    result['diagnosis_fingerprint']=digest(result);output.mkdir(parents=True);path=output/'diagnosis.json';path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    derived=dict(version=AUDIT_VERSION,status='passed',full_checks_verified=True,report_sha256=sha(path),diagnosis_fingerprint=result['diagnosis_fingerprint'],source=source,
        checks_count=checks.count,checks=checks.entries,failures=[],scope='Source binding, six paired losses and independent dense native64 rescoring; native validity inherited from full55633-check proof.')
    (output/'audit.json').write_text(json.dumps(derived,indent=2,allow_nan=False)+'\n');return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('study','audit','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();result=build(a.study,a.audit,a.output);print(json.dumps(dict(summary=result['summary'],diagnosis_fingerprint=result['diagnosis_fingerprint'])))

if __name__=='__main__':main()
