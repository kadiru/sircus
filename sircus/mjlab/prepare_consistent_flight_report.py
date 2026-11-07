"""Additive read-only publication; fresh native evidence remains gated by its audit."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

KEY='consistent_flight_study'
VERSION='sircus-consistent-flight-study-v1'
METHODS=['bc','ppo','ensemble','consistent']
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def compact(study,audit_path=None):
    study=Path(study);raw=read(study/'report.json');spec=read(study/'contract.json')
    protocol=raw.get('protocol',{})
    exact=dict(methods=METHODS,test_layouts=30,maximum_fresh_requests=132,substeps=64,new_train_requests=0)
    if raw.get('version')!='sircus-consistent-flight-v1' or any(protocol.get(k)!=v for k,v in exact.items()):
        raise ValueError('Exact frozen30/132/fixed64/no-fit study required')
    from .consistent_flight_study import digest
    unsigned=dict(spec);fp=unsigned.pop('fingerprint',None)
    if digest(unsigned)!=fp or raw.get('contract_fingerprint')!=fp or raw['protocol']!=spec['protocol']:
        raise ValueError('Report contract changed')
    if sha(study/'development.json')!=raw['development_sha256']:raise ValueError('Development evidence changed')
    development=read(study/'development.json')
    if development.get('new_native_requests')!=0 or development.get('new_training_requests')!=0:
        raise ValueError('Original development scope changed')
    cases=[]
    for row in development['cases']:
        proxy=row['proxy'];available=proxy['complete_rim_band_members']
        cases.append({k:copy.deepcopy(row[k]) for k in ('id','values','bucket_xy_m','prediction_capture_score','observed_capture','observed_native_release','note')})
        cases[-1].update(proxy_aperture_margin_m=proxy['worst_clearance_m'],
            proxy_status=f'{available}/5 members have a complete descending rim passage; {proxy["nominal_clear_aperture_members"]}/5 predict clear aperture')
    cases.sort(key=lambda r:(-r['prediction_capture_score'],r['id']))
    high=[r for r in cases if r['prediction_capture_score']>=.99]
    highnegative=[r for r in high if r['proxy_aperture_margin_m'] is not None and r['proxy_aperture_margin_m']<0]
    summary=[dict(metric='Old recorded TEST requests now used for development',value=len(cases),scope='Original outcomes; no new native shots'),
        dict(metric='Changed model-only top candidate',value=f'{development["changed_top_candidates"]}/{development["contexts"]}',scope='Original VAL/TEST candidate banks; changed candidate capture unknown'),
        dict(metric='Old requests with model score ≥99%',value=len(high),scope='Uncalibrated scores'),
        dict(metric='Those with negative conservative aperture margin',value=len(highnegative),scope='May still capture after rim contact; no hard rejection'),
        dict(metric='New model fitting',value=0,scope='All three saved checkpoints unchanged')]
    counts=raw.get('counts',{})
    value=dict(version=VERSION,status=raw['status'],scope=raw['scope'],
        protocol=dict(test_positions=30,maximum_requests=132,substeps=64,methods=METHODS,new_training=False),
        progress=dict(requested=counts.get('requested'),planned_requests=132,native_releases=counts.get('native_releases')),
        development=dict(cases=cases,summary=summary),comparisons=[dict(method=r['method'],**r['counts']) for r in raw.get('comparisons',[])],
        report_sha256=sha(study/'report.json'),audit=dict(passed=None),
        provenance=dict(source_report_sha256=raw['source_report_sha256'],selector_sha256=raw['selector_sha256'],
            ensemble_checkpoint_sha256=raw['artifacts']['ensemble']['sha256'],plan_sha256=raw.get('prediction_plan_sha256')),
        findings=['The new rank combines frozen velocity predictions with requested board geometry, gravity and finite ball size.',
            'Rim-assisted capture remains possible with a negative aperture margin. Native entry and one-second retention still decide capture.',
            'Existing setup/path collision checks remain. Post-bounce robot collision prediction is not added in this ablation.'],
        next_steps=['Wait for the exclusive 4090 slot, then evaluate the frozen132 requests and independent dense-evidence audit.',
                    'Compare all requested shots, including setup rejects. No extra fitting or automatic policy activation.'])
    if raw['status']=='queued' and raw.get('queue'):
        value['queue']=dict(gpu='RTX 4090',status=raw['phase'],reason='Existing 4090 workload or shared lock is active',wait_s=raw['queue'].get('wait_s'))
    elif raw['status'] in ('prepared','preparing'):
        value['queue']=dict(gpu='RTX 4090',status='CPU preparation',reason='GPU allocation waits until all decisions are frozen')
    elif raw['status']=='running':
        value['queue']=dict(gpu='RTX 4090',status='Acquired exclusive slot',reason='Executing frozen requests; results provisional')
        value['next_steps']=['Complete the frozen comparison and independent audit before interpreting differences.']
    elif raw['status']=='completed':
        value['queue']=dict(gpu='RTX 4090',status='Native execution completed',reason='GPU slot released; CPU verification is separate')
        value['next_steps']=['Complete independent verification, then review fresh paired results.']
    if raw.get('error'):value['findings'].append('Worker stopped; retained evidence requires review.')
    if audit_path and Path(audit_path).is_file():
        proof=read(audit_path);checks=proof.get('checks')
        if proof.get('status')=='failed':value['audit']=dict(passed=False)
        elif (proof.get('version')=='sircus-consistent-flight-audit-v1' and proof.get('status')=='passed'
            and proof.get('failures')==[] and isinstance(checks,list) and len(checks)>0
            and type(proof.get('checks_count')) is int and proof['checks_count']==len(checks)
            and all(isinstance(c,dict) and c.get('passed') is True for c in checks)
            and proof.get('report_sha256')==value['report_sha256']
            and proof.get('evidence_manifest_sha256')==sha(study/'evidence-manifest.json')
            and proof.get('contract_fingerprint')==fp and proof.get('requested_trials')==132
            and proof.get('training_requests')==0 and proof.get('comparisons')==raw['comparisons']
            and proof.get('producer_source_sha256')==spec['source_sha256']):
            value.update(status='verified',audit=dict(passed=True,report_sha256=value['report_sha256'],checks_count=len(checks)))
            value['next_steps']=['Review fresh paired results and concrete misses before any further training.']
        else:raise ValueError('Incomplete or mismatched native audit; no verified publication')
    return value


def prepare(base,study,output,*,audit_path=None):
    base,study,output=map(Path,(base,study,output))
    if output.resolve()==base.resolve() or output.resolve().is_relative_to(study.resolve()):raise ValueError('Separate display outside evidence required')
    result=read(base)
    if not isinstance(result,dict):raise ValueError('Existing object display required')
    result[KEY]=compact(study,audit_path)
    tmp=output.with_suffix(output.suffix+'.tmp');tmp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');tmp.replace(output)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('base','study','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--audit',type=Path)
    a=p.parse_args();prepare(a.base,a.study,a.output,audit_path=a.audit)

if __name__=='__main__':main()
