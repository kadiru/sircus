"""Finite recipe recovery on the original fixed 35-layout development grid."""
from __future__ import annotations
import copy
import itertools
from pathlib import Path
import numpy as np
from .benchmark_verify import digest, read, sha
from .corrected_imitation_contract import authenticate_study, require
from .fixed35_imitation_contract import fixed_grid, ROTATION
from .table_correction_native import META
from .table_training_contract import captured, clean, statistics
from .table_assisted_score import reward

VERSION = 'sircus-fixed35-recovery-v1'
AUDIT_VERSION = 'sircus-fixed35-recovery-audit-v1'
INPUT_KEYS = ('parent-study', 'parent-audit', 'reference', 'assets')
METHODS = ('preserved_current', 'recovery_search', 'recovery_repeat')
LOW = np.asarray([.30, -.50, .12, 8., .05])
HIGH = np.asarray([1.04, .35, .60, 60., .65])
PARAMETERS = ('board_x_m', 'board_y_m', 'board_height_m', 'tilt_deg', 'drop_gap_m')
SCOPE = ('Bounded posthoc recipe search on nine current-policy failures from the historical fixed 7x5 grid. '
         'Two fixed-base clearance blockers remain explicitly blocked; 24 successful current recipes are replayed without changing their joints. '
         'Bucket-facing yaw, native64/v3, whole-ball one-second capture including table/rim assistance. '
         'Three fresh nominal repeats confirm a discovered recipe. No policy fitting, PPO, new positions, bucket perturbations, '
         'noise, 128-step qualification, learned transit, hardware or policy promotion.')


def protocol():
    return dict(substeps=64, methods=list(METHODS), grid_positions=35, parent_captures=24,
        development_failures=11, fixed_base_skips=2, search_layouts=9, proposals_per_layout=80,
        search_requests=720, repeat_per_selected=3, candidates_per_layout=3, repeat_requests_max=81,
        preserved_requests=24, control_requests=18, maximum_requests=843,
        noise=False, bucket_perturbations=False, training=False, actor_steps=0, critic_steps=0,
        worker_limit_s=7200, search_limit_s=5400, queue_limit_s=900, global_limit_s=9900,
        audit_reserve_s=1800, storage_reserve_bytes=8*1024**3,
        rotation=ROTATION, bounds=dict(low=LOW.tolist(), high=HIGH.tolist()),
        evaluation_scope='historical fixed-grid recipe development; not held out and not a new policy score',
        search_order='80 frozen proposals per case, round-robin one per active case; no flight-dependent numeric changes',
        confirmation='Immediately repeat each captured discovery three times; stop a case after 3/3 or three failed confirmation groups',
        timing='Stop starting search rounds after 5400 acquired seconds; finish selected repeats and final controls within 7200 seconds',
        preservation='All24 current captured recipes replay once with authenticated unchanged action and joint branch',
        scope=SCOPE)


def load_inputs(a, root):
    public = read(Path(root)/'baselines/mjlab-fixed35-imitation-20261109/summary.json')
    source = authenticate_study(a.parent_study, a.parent_audit,
        dict(report_sha256=public['report_sha256'], proof_sha256=public['audit']['proof_sha256']), checks_count=40692)
    rows = source['rows']; require(len(rows)==88, 'Complete original88 requests required')
    current = sorted([r for r in rows if r.get('phase')=='test' and r.get('method')=='current_bc'], key=lambda r:r['context_id'])
    refit = sorted([r for r in rows if r.get('phase')=='test' and r.get('method')=='fixed35_bc'], key=lambda r:r['context_id'])
    grid = fixed_grid(); expected = {r['id']:r['bucket_xy_m'] for r in grid}
    require(len(current)==len(refit)==35 and all({r['context_id']:r['bucket_xy_m'] for r in group}==expected for group in (current,refit)), 'Exact original35 grid required')
    preserved = [r for r in current if captured(r)]
    failures = [r for r in current if not captured(r)]
    blocked = [r for r in failures if r.get('invalid')=='fixed_base_clearance_rejected']
    cases = [r for r in failures if r not in blocked]
    require((len(preserved),len(failures),len(blocked),len(cases))==(24,11,2,9), 'Published24/11/2/9 cohort differs')
    require({r['context_id'] for r in blocked}=={'fixed35-dev-grid-03-00','fixed35-dev-grid-03-06'}, 'Only the two fixed-base overlaps may skip search')
    for r in current+refit:
        for prefix in ('context_resolution','connection'):
            path=source['bound']('native-yaw/'+r[prefix+'_file'])
            require(sha(path)==r[prefix+'_sha256'], 'Bound native-yaw proof differs')
        connection=read(source['bound']('native-yaw/'+r['connection_file']))
        require(connection.get('release_behavior',{}).get('version')=='sircus-continuous-bucket-facing-release-v1', 'Original fixed-yaw proof required')
        if r in blocked:
            resolution=read(source['bound']('native-yaw/'+r['context_resolution_file']))
            require(resolution.get('hard_reject') is True and connection['reason']=='fixed_base_clearance_rejected', 'A finite IK failure is not a fixed-base proof')
        if r in failures and r.get('native_release'):
            require(sha(source['bound']('native-yaw/'+r['native_cache_file']))==r['native_cache_sha256'], 'Historical diagnostic trace differs')
    out={}
    for key,name in (('prior','prior.json'),('old_prior','old-prior.json'),('controls','controls.json'),
                     ('control_replays','control-replays.json'),('calibration','calibration.json'),('home','setup-home.json')):
        out[key]=read(source['bound'](name))
    require(len(out['prior']['records'])==19 and len(out['old_prior']['records'])==17, 'Unchanged19/17 geometry priors required')
    for name in ('verification.json','native-yaw/effective-options.json','native-yaw/native-compiled-geometry.json',
                 'control-parent/native/effective-options.json','control-parent/native/native-compiled-geometry.json'):
        source['bound'](name)
    for r in out['control_replays']:
        for prefix in ('context_resolution','connection'):
            require(sha(source['bound']('control-parent/native/'+r[prefix+'_file']))==r[prefix+'_sha256'], 'Original formal control proof differs')
    artifacts=copy.deepcopy(source['report']['artifacts'])
    for v in artifacts.values():require(sha(source['bound'](v['path']))==v['sha256'], 'Preserved model checkpoint differs')
    out.update(parent=source['spec'],parent_report=source['report'],rows=rows,current=current,refit=refit,
        preserved=preserved,failures=failures,blocked=blocked,cases=cases,grid=grid,artifacts=artifacts,
        bindings=dict(parent=source['bindings'],current_row_sha256={r['id']:digest(r) for r in current},
            scope='Original current24/35 and refit23/35 stay unchanged; all recovery cases are development'),
        diagnoses=diagnose_failures(Path(a.parent_study), failures))
    return out


def diagnose_failures(directory, failures):
    """Small historical summaries; contact order is evidence, not inferred causal blame."""
    result=[]; cache={}
    for r in failures:
        item=dict(context_id=r['context_id'],bucket_xy_m=r['bucket_xy_m'],values=r['values'],
            parent_row_id=r['id'],parent_row_sha256=digest(r),native_release=r['native_release'],invalid=r.get('invalid'))
        if r['native_release']:
            path=directory/'native-yaw'/r['native_cache_file']
            if str(path) not in cache:
                with np.load(path,allow_pickle=False) as z:cache[str(path)]={k:z[k] for k in ('position_m','forces_N','time_s')}
            z=cache[str(path)];idx=r['native_cache_row'];p=z['position_m'][:,idx];f=z['forces_N'][:,idx];t=z['time_s']
            contacts={}
            for col,name in enumerate(('board','bucket','table','floor','robot','unattributed')):
                hits=np.flatnonzero(f[:,col]>.01);contacts[name]=float(t[hits[0]+1]) if len(hits) else None
            j=int(np.argmin(np.linalg.norm(p[:,:2]-r['bucket_xy_m'],axis=1)))
            robot_first=contacts['robot'] is not None and (contacts['bucket'] is None or contacts['robot']<contacts['bucket'])
            kind=('robot_before_bucket' if robot_first else 'bucket_contact_without_entry' if contacts['bucket'] is not None else 'no_bucket_contact')
            item.update(category=kind,first_contacts_s=contacts,contact_force_threshold_N=.01,
                contact_scope='Full saved horizon, including later contacts after scoring termination; contact order does not alone establish causality',
                closest_xy=dict(time_s=float(t[j]),distance_m=float(np.linalg.norm(p[j,:2]-r['bucket_xy_m'])),ball_center_z_m=float(p[j,2])),
                whole_ball_entry=bool(r.get('whole_ball_entry')),max_retention_s=r.get('max_retention_s'),
                rim_band_contact=bool(r.get('rim_band_contact')),native_cache_sha256=r['native_cache_sha256'])
        else:item['category']='fixed_base_blocked' if r.get('invalid')=='fixed_base_clearance_rejected' else 'finite_setup_unresolved'
        result.append(item)
    return result


def request(source, *, identity, phase, method, values=None, **extra):
    values=copy.deepcopy(source['values'] if values is None else values)
    return dict({k:copy.deepcopy(source[k]) for k in META}, id=identity,phase=phase,role='development',method=method,
        context_id=source['context_id'],bucket_xy_m=copy.deepcopy(source['bucket_xy_m']),values=values,command_values=copy.deepcopy(values),
        parent_row_id=source['id'],parent_row_sha256=digest(source),substeps=64,
        original_role=source.get('role'),development_reclassified=True,selection_or_learning=False,
        perturbation_mixed_units=[0.]*5,**extra)


def proposals(source, refit, successes):
    """Literal80-plan: historical alternatives, local axes, coupled grid, XY and broad height support."""
    parent=np.asarray(source['values'],float);items=[]
    def add(v,kind,lineage=()):
        raw=np.asarray(v,float);require(raw.shape==(5,) and np.isfinite(raw).all(),'Finite five-coordinate proposal required')
        val=np.clip(raw,LOW,HIGH)
        items.append(dict(values=val.tolist(),raw_values=raw.tolist(),clipped=bool(np.any(raw!=val)),
            kind=kind,parent_source_ids=list(lineage) or [source['id']]))
    add(parent,'original_failed_current')
    alternate=[r for r in refit if r['context_id']==source['context_id']]
    require(len(alternate)==1,'Unique historical refit alternative required')
    add(alternate[0]['values'],'exact_historical_refit_capture' if captured(alternate[0]) else 'exact_historical_refit_action',[alternate[0]['id']])
    nearest=sorted(successes,key=lambda r:(float(np.linalg.norm(np.asarray(r['bucket_xy_m'])-source['bucket_xy_m'])),r['id']))[:4]
    require(len(nearest)==4,'Four historical successful neighbors required')
    for r in nearest:add(r['values'],'neighbor_success_action_at_fixed_bucket',[r['id']])
    for index,magnitudes,kind in ((4,(.02,.04,.06,.10,.16,.24),'gap_axis'),(3,(2.,4.,6.,10.,14.,18.),'tilt_axis'),(2,(.02,.05,.08,.12,.18),'height_axis')):
        for magnitude in magnitudes:
            for sign in (-1,1):
                v=parent.copy();v[index]+=sign*magnitude;add(v,kind)
    for tilt,gap in itertools.product((-8.,-4.,4.,8.),(-.08,-.04,.04,.08)):
        v=parent.copy();v[3]+=tilt;v[4]+=gap;add(v,'coupled_tilt_gap')
    shifts=[(x,y) for x in (-.06,0.,.06) for y in (-.06,0.,.06) if x or y]+[(-.12,0.),(.12,0.),(0.,-.12),(0.,.12)]
    for x,y in shifts:
        v=parent.copy();v[:2]+=[x,y];add(v,'bounce_plane_xy')
    for h,t,g in itertools.product((.20,.32,.44),(20.,32.),(.16,.32)):
        v=parent.copy();v[2:]=[h,t,g];add(v,'coarse_height_tilt_gap')
    require(len(items)==80,'Exactly80 frozen candidates per case required')
    for i,item in enumerate(items):item['proposal_index']=i
    return items


def make_plan(cases, refit, successes):
    pools={r['context_id']:proposals(r,refit,successes) for r in cases};plan=[]
    for index in range(80):
        for source in cases:
            proposal=pools[source['context_id']][index]
            plan.append(request(source,identity=f'fixed35-recovery-search-p{index:02d}-{source["context_id"]}',
                phase='search',method='recovery_search',values=proposal['values'],proposal_index=index,derivation=proposal,
                exact_parent_action=proposal['values']==source['values']))
    require(len(plan)==720 and len({r['id'] for r in plan})==720,'Exactly720 unique frozen slots required')
    return plan


def preserved_plan(successes):
    return [request(r,identity='fixed35-recovery-preserved-'+r['context_id'],phase='preserved',method='preserved_current',
                    exact_parent_action=True) for r in successes]


def repeat_plan(discovery, confirmation_index):
    require(captured(discovery) and 0<=confirmation_index<3,'Only captured discoveries within three confirmation groups')
    rows=[]
    for repeat in range(3):
        r=request(discovery,identity=f'fixed35-recovery-repeat-c{confirmation_index}-r{repeat}-{discovery["context_id"]}',
            phase='repeat',method='recovery_repeat',confirmation_index=confirmation_index,repeat=repeat,
            selected_from_id=discovery['id'],selected_from_sha256=digest(discovery),proposal_index=discovery['proposal_index'],
            selected_connection_sha256=discovery['connection_sha256'],selected_q10=discovery['q10'],selected_q14=discovery['q14'])
        # Original parent lineage is retained; the selected discovery is a separate binding.
        r.update(parent_row_id=discovery['parent_row_id'],parent_row_sha256=discovery['parent_row_sha256'])
        rows.append(r)
    return rows


def confirmation_groups(rows, context_id):
    groups=[]
    for i in range(3):
        group=[r for r in rows if r.get('phase')=='repeat' and r['context_id']==context_id and r['confirmation_index']==i]
        if not group:continue
        first=group[0]
        groups.append(dict(confirmation_index=i,selected_from_id=first['selected_from_id'],selected_from_sha256=first['selected_from_sha256'],
            values=first['values'],bucket_xy_m=first['bucket_xy_m'],requests=len(group),captures=sum(captured(r) for r in group),
            record_ids=[r['id'] for r in group],status='confirmed' if len(group)==3 and all(captured(r) for r in group) else 'failed' if len(group)==3 else 'incomplete'))
    return groups


def case_results(rows, cases, blocked, *, finished=False, budget_truncated=False):
    results=[]
    for parent in sorted(cases+blocked,key=lambda r:r['context_id']):
        search=[r for r in rows if r.get('phase')=='search' and r['context_id']==parent['context_id']]
        groups=confirmation_groups(rows,parent['context_id']);confirmed=next((g for g in groups if g['status']=='confirmed'),None)
        status=('fixed_base_blocked' if parent in blocked else 'confirmed' if confirmed else 'confirmation_limit' if len(groups)==3 else
                'search_exhausted' if len(search)==80 else 'budget_truncated' if finished and budget_truncated else 'in_progress')
        recipe=next((r for r in search if confirmed and r['id']==confirmed['selected_from_id']),None)
        results.append(dict(context_id=parent['context_id'],bucket_xy_m=parent['bucket_xy_m'],parent_row_id=parent['id'],parent_row_sha256=digest(parent),
            status=status,search_requests=len(search),search_captures=sum(captured(r) for r in search),search_counts=statistics(search),
            confirmation_groups=groups,confirmed_recipe=copy.deepcopy(recipe)))
    return results
