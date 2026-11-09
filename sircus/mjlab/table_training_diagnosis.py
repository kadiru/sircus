"""CPU-only, byte-bound diagnosis of completed table-training-v1 shots.

Original TEST becomes development evidence; no new shots, scores, model fits or
GPU imports. Contacts and trajectories are observed, not causal explanations.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
from .xy650_verify import first_flight

VERSION='sircus-table-training-diagnosis-v1'
AUDIT_VERSION='sircus-table-training-diagnosis-audit-v1'
SOURCE=dict(report_sha256='4605bcb801351ae91ad16b71a033ceb3866027db621f9d84227f23b1ef257e95',
 audit_sha256='6e50d87ddb7b2d4b37debbc2a5ac96507ff40bc236223277121acd385b8ec91b',
 rows_sha256='ddb1ecd67b71951fe5d1152a496c1bbf5b01e4174b38a7f7b8af7b813f4817e3',
 contract_fingerprint='20f198caec28ff0402e8f7d2f7d2d1de65434b294f964a3ea35cde36bc3a1114')
NAMES=('board','bucket','table','floor','robot','unattributed')
TRACE_CASES=('00','07','08','16','18','26','39')
COMPARATORS=('02','31')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def read(path):return json.loads(Path(path).read_text())

def setup_proof(study,row,evidence,check):
    """Extract only authenticated exact-action setup evidence, never flight labels."""
    native=(Path(study)/'native').resolve()
    loaded={}
    for name,filekey,hashkey in [('connection','connection_file','connection_sha256'),('resolution','context_resolution_file','context_resolution_sha256')]:
        relative=row[filekey];path=(native/relative).resolve()
        check(path.is_relative_to(native),row['id']+'/'+name+' contained')
        actual=sha(path)
        check(actual==row[hashkey]==evidence.get('native/'+relative),row['id']+'/'+name+' exact source proof')
        loaded[name]=read(path)
    connection,resolution=loaded['connection'],loaded['resolution']
    check(connection['resolution_file']==row['context_resolution_file'] and connection['resolution_sha256']==row['context_resolution_sha256'],row['id']+'/connection binds resolution')
    check(connection['bucket_xy_m']==resolution['bucket_xy_m']==row['bucket_xy_m'] and connection['values']==row['values'],row['id']+'/same exact action and bucket')
    check(connection['passed'] is False and connection['action_substituted'] is False and not row['native_release'],row['id']+'/no substituted action or native release')
    geometry=resolution['home_geometry'];attempt=connection.get('endpoint_attempts') or {}
    reason=connection['reason']
    explanation=('Bounded exact-action IK search found no valid board-arm branch; an anchor was admitted, but this requested residual was not. This does not prove the pose unreachable.'
                 if reason=='finite_ik_unresolved' else
                 'The bucket/fixed-base conservative clearance lower bound did not meet 5 mm. A zero lower bound or an inconsistent distance query is not evidence of penetration.')
    return dict(connection_sha256=row['connection_sha256'],resolution_sha256=row['context_resolution_sha256'],reason=reason,
      connection_policy=connection.get('connection_policy'),action_substituted=connection['action_substituted'],layout_eligible=resolution['layout_eligible'],
      anchor_admitted=resolution['anchor_admitted'],requested_values=connection['values'],anchor_values=connection.get('anchor_values'),
      valid_ik_branches=attempt.get('valid_ik_branches'),required_clearance_m=.005,
      home_geometry=dict(passed=geometry['passed'],reason=geometry['reason'],minimum_clearance_m=geometry['minimum_clearance_m'],
        closest_pair=geometry['closest_pair'],inconsistent_distance_queries=geometry['inconsistent_distance_queries'],
        unresolved_pair_count=len(geometry['unresolved_clearance_pairs']),fixed_base_blocker_count=len(geometry['fixed_base_blockers'])),interpretation=explanation)

def event_summary(force,times,stop):
    """First observed channel intervals, clipped at the unchanged score terminal."""
    result=[]
    for column,name in enumerate(NAMES):
        indices=np.flatnonzero(force[:stop,column]>.01)
        if len(indices):result.append(dict(label=name+' contact',category=name,time_s=float(times[indices[0]])))
    return sorted(result,key=lambda e:(e['time_s'],e['category']))

def classify(item):
    if not item['native_release']:return 'setup_'+str(item.get('reason','unknown'))
    if item['capture']:return 'capture_with_table_contact' if item['table_contact'] else 'direct_capture'
    first=item.get('first_post_board_contact')
    return (first['category']+'_first_miss') if first else 'unknown_first_flight_miss'

def choose_correction_cases(cases):
    """Frozen posthoc strata: first two shallow far failures per contact class."""
    eligible=[c for c in cases if c['native_release'] and not c['capture']
              and c['values'][3]<22 and c['board_bucket_distance_m']>=.45]
    selected=[]
    for group in ('table','bucket','robot'):
        own=sorted((c for c in eligible if (c.get('first_post_board_contact') or {}).get('category')==group),key=lambda c:c['context_id'])
        selected.extend(own[:2])
    return [c['context_id'] for c in selected]

def analyze(row,arrays,scene,calibration,*,include_trace=False):
    item=dict(context_id=row['context_id'],record_id=row['id'],record_sha256=digest(row),
      bucket_xy_m=row['bucket_xy_m'],values=row['values'],native_release=bool(row['native_release']),
      capture=bool(row.get('route_success')),board_center_m=row.get('board_center_m'),board_normal=row.get('board_normal'),reason=row.get('geometry_reason') or row.get('invalid'),
      table_contact=bool(row.get('table_contact_steps',0)),forbidden_contact=bool(row.get('forbidden_contact_steps',0)),
      whole_ball_entry=bool(row.get('whole_ball_entry')),max_retention_s=float(row.get('max_retention_s',0)),
      board_bucket_distance_m=float(np.linalg.norm(np.asarray(row['bucket_xy_m'])-row['values'][:2])),
      events=[],closest_approach_m=row.get('closest_opening_distance_m'),first_flight_forward_error_m=None,
      first_flight_lateral_error_m=None,first_post_board_contact=None,scope='Observed original TEST shot; now posthoc development evidence')
    if not row['native_release']:
        item['group']=classify(item);return item
    stop=int(row['physics_steps']); t=arrays['time_s'];p=arrays['position_m'];v=arrays['velocity_m_s'];f=arrays['forces_N']
    item['events']=event_summary(f,t,stop)
    for key,label in [('whole_ball_entry_time_s','whole-ball entry'),('capture_time_s','one-second capture')]:
        if row.get(key) is not None:item['events'].append(dict(label=label,category=key,time_s=float(row[key])))
    item['events'].sort(key=lambda e:e['time_s'])
    metric=first_flight(scene,row['bucket_xy_m'],row['board_normal'],calibration,arrays,valid=not row['invalid_physics'])
    item['first_flight']=metric
    if metric['available']:
        start,end=metric['start_index'],metric['end_index'];di=np.asarray(row['bucket_xy_m'])-row['values'][:2];di/=np.linalg.norm(di);lat=np.array([-di[1],di[0]])
        item['observed_rebound_velocity_m_s']=v[start].astype(float).tolist()
        item['observed_rebound_position_m']=p[start].astype(float).tolist()
        item['first_flight_end_position_m']=p[end].astype(float).tolist()
        if metric['cutoff_contact_channels']:
            item['first_post_board_contact']=dict(label=' + '.join(metric['cutoff_contact_channels'])+' contact',category='+'.join(metric['cutoff_contact_channels']),time_s=metric['end_time_s'])
        if metric['first_crossing'] is not None:
            xy=np.asarray(metric['first_crossing']['xy_m'])-row['bucket_xy_m']
            item['first_flight_forward_error_m']=float(xy@di);item['first_flight_lateral_error_m']=float(xy@lat)
        # Observe first free state after a table-force episode. Do not extrapolate
        # through that contact or infer that the table caused later capture.
        hits=np.flatnonzero(f[:stop,2]>.01)
        if len(hits):
            i=int(hits[0]);j=i
            while j<stop and f[j,2]>.01:j+=1
            item['first_table_contact']=dict(time_s=float(t[i]),position_m=p[i].astype(float).tolist(),incoming_velocity_m_s=v[i].astype(float).tolist(),
              episode_end_time_s=float(t[j]),outgoing_velocity_m_s=v[j].astype(float).tolist())
    if include_trace:
        indices=set(np.linspace(0,stop,min(stop+1,241),dtype=int).tolist())
        for event in item['events']:
            ix=int(np.searchsorted(t,event['time_s']));indices.update([max(0,ix-1),min(stop,ix),min(stop,ix+1)])
        indices=sorted(indices)
        item['trace']=dict(record_id=row['id'],record_sha256=digest(row),native_cache_sha256=row['native_cache_sha256'],
          time_s=t[indices].astype(float).tolist(),position_m=p[indices].astype(float).tolist(),
          scope='Downsampled actual saved native states through original scoring terminal; event-adjacent samples retained')
    item['group']=classify(item)
    if not item['capture']:
        contact=item.get('first_post_board_contact')
        item['reason']='No whole-ball entry; first post-board free flight ends at '+(contact['label'] if contact else 'an unobserved cutoff')+'. Later contacts are separate observations.'
    else:
        item['reason']='Whole-ball entry and one continuous second retained; '+('table contact observed before terminal.' if item['table_contact'] else 'no table contact before terminal.')
    return item

def build(study,audit,output):
    study=Path(study).resolve();output=Path(output).resolve();audit=Path(audit).resolve()
    if output.exists() or output.is_relative_to(study):raise ValueError('New derived output outside frozen study required')
    checks=[]
    def check(ok,label):
        checks.append(dict(passed=bool(ok),label=label))
        if not ok:raise ValueError(label)
    for key,path in [('report_sha256',study/'report.json'),('audit_sha256',audit),('rows_sha256',study/'rows.jsonl')]:check(sha(path)==SOURCE[key],'Exact source '+key)
    proof=read(audit);report=read(study/'report.json');spec=read(study/'contract.json')
    check(read(study/'evidence-manifest.json')==report['evidence_sha256'],'Source manifest equals exact report evidence identities')
    check(report['status']==report['phase']=='completed','Completed source')
    check(proof['status']=='passed' and proof['full_checks_verified'] is True and not proof['failures']
          and len(proof['checks'])==proof['checks_count'] and all(x['passed'] is True for x in proof['checks']),'Full parent audit passed')
    check(proof['report_sha256']==SOURCE['report_sha256'] and proof['contract_fingerprint']==spec['fingerprint']==SOURCE['contract_fingerprint'],'Parent audit and contract identity')
    check(spec['protocol']['substeps']==64,'Fixed64 source')
    for name in ('contract.json','calibration.json','rows.jsonl'):
        check(sha(study/name)==report['evidence_sha256'].get(name),'Report-bound consumed artifact/'+name)
    rows=[json.loads(x) for x in (study/'rows.jsonl').read_text().splitlines() if x]
    own=[r for r in rows if r.get('phase')=='test' and r.get('method')=='ppo']
    check(len(own)==len({r['context_id'] for r in own})==50,'All50 selected PPO positions')
    check(sum(bool(r.get('route_success')) for r in own)==15,'Original15 captures')
    check(sum(not r['native_release'] for r in own)==4,'Original4 setup declines')
    check(all(not r.get('invalid_physics') for r in own),'Zero TEST invalid physics')
    cal=read(study/'calibration.json')['score_config'];items=[];groups={}
    for row in own:
        if row['native_release']:groups.setdefault(row['native_cache_file'],[]).append(row)
        else:
            item=analyze(row,None,spec['scene'],cal)
            item['setup_proof']=setup_proof(study,row,report['evidence_sha256'],check);items.append(item)
    for name,group in groups.items():
        path=study/'native'/name;check(path.resolve().is_relative_to(study/'native'),'Contained archive')
        actual=sha(path);check(actual==report['evidence_sha256'].get('native/'+name),'Report-bound dense archive/'+name)
        with np.load(path,allow_pickle=False) as z:
            data={k:z[k] for k in ('position_m','velocity_m_s','forces_N','time_s','dt_s','contact_distance_m')};ids=z['record_ids'].tolist()
        for row in group:
            col=row['native_cache_row'];check(actual==row['native_cache_sha256'] and ids[col]==row['id'],row['id']+'/exact dense archive column')
            arrays={k:a[:,col] if a.ndim>=2 and k!='time_s' else a for k,a in data.items()}
            item=analyze(row,arrays,spec['scene'],cal,include_trace=row['context_id'][-2:] in TRACE_CASES+COMPARATORS)
            check(item['first_flight']['available'],row['id']+'/observed board rebound')
            item['native_cache_sha256']=actual;items.append(item)
    items.sort(key=lambda x:x['context_id']);failures=[x for x in items if not x['capture']];misses=[x for x in failures if x['native_release']]
    check(len(failures)==35 and len(misses)==31,'35 failures /31 native misses preserved')
    check(sum(x['forbidden_contact'] for x in misses)==13,'13 forbidden-contact misses')
    check(all(not x['whole_ball_entry'] for x in misses),'All31 misses failed whole-ball entry')
    counts=Counter(x['group'] for x in failures)
    descriptions={'table_first_miss':'First board rebound meets table; table contact is permitted. Later flight/entry still failed.',
      'bucket_first_miss':'First board rebound meets bucket before capture; contact alone does not prove a rim hit or entry.',
      'robot_first_miss':'First board rebound hits robot; later table/bucket motion is already deflected.',
      'setup_finite_ik_unresolved':'No admitted branch found within finite exact-action search; not proof of unreachable pose.',
      'setup_fixed_base_clearance_rejected':'Bucket/fixed-base clearance could not meet the unchanged conservative margin.'}
    source=dict(SOURCE,diagnosis_source_sha256=sha(__file__),parent_audit_checks=proof['checks_count'],producer_commit='b11e8b9772268d79a14a5bcb0b8d9761789da379',
      evidence_manifest_sha256=sha(study/'evidence-manifest.json'),calibration_sha256=sha(study/'calibration.json'),contract_sha256=sha(study/'contract.json'))
    diagnosis=dict(version=VERSION,status='completed',contract_fingerprint=SOURCE['contract_fingerprint'],source=source,
      summary=dict(requested=50,captures=15,flight_misses=31,setup_rejections=4,forbidden_contact_misses=13,no_forbidden_contact_misses=18),
      scene=dict(board_size_m=spec['scene']['board']['size_m'],bucket_base_z_m=0.,bucket_height_m=spec['scene']['bucket']['assumed_bottom_thickness_m']+spec['scene']['bucket']['assumed_interior_depth_m'],bucket_inner_radius_m=spec['scene']['bucket']['inner_top_diameter_m']/2,bucket_outer_radius_m=spec['scene']['bucket']['outer_top_diameter_m']/2,ball_radius_m=spec['scene']['ball']['radius_m'],geometry_basis='Configured simulator geometry; measured board dimensions, provisional ball and bucket depth; not a reconstructed camera image'),
      groups=[dict(label=k,count=v,description=descriptions.get(k,k)) for k,v in sorted(counts.items())],cases=failures,
      comparators=[x for x in items if x['context_id'][-2:] in COMPARATORS],correction_case_ids=choose_correction_cases(items),
      limitations=['Former TEST positions are posthoc development cases, not a fresh evaluation.',
        'First-flight crossing stops before first subsequent contact. It does not describe table-assisted later flight.',
        'Force channel presence does not identify causal failure or prove an individual bucket/rim contact.',
        'Saved endpoint shots do not validate driven arm transit; no new training or native trials in this diagnosis.'])
    diagnosis['recommended_correction']=dict(rule='Select first two sorted failures per first post-board contact class (table,bucket,robot), original tilt<22deg and board-bucket distance>=450mm; change only tilt to32deg',classes=['table','bucket','robot'],empty_eligible_classes=['bucket'],case_ids=diagnosis['correction_case_ids'],matched_repeats=3,controls=18,total_requests=42,posthoc_development=True,training=False,assumption='Hypothesis from the shallow far failure family versus successful ~32deg table-contact family, not causal proof or a learned policy improvement')
    check(diagnosis['correction_case_ids']==['table-training-test-00','table-training-test-01','table-training-test-18','table-training-test-26'],'Four cases under frozen first-two-per-class rule')
    output.mkdir(parents=True);rp=output/'diagnosis.json';rp.write_text(json.dumps(diagnosis,indent=2,allow_nan=False)+'\n')
    auditout=dict(version=AUDIT_VERSION,status='passed',full_checks_verified=True,report_sha256=sha(rp),contract_fingerprint=SOURCE['contract_fingerprint'],
      source=source,checks=checks,checks_count=len(checks),failures=[],scope='CPU diagnosis identity/accounting/trace checks; native capture validity inherited from exact complete parent audit')
    (output/'audit.json').write_text(json.dumps(auditout,indent=2)+'\n')
    return diagnosis

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--audit',required=True);p.add_argument('--output',required=True);a=p.parse_args();d=build(a.study,a.audit,a.output);print(json.dumps({k:d[k] for k in ('summary','groups','correction_case_ids')},indent=2))
