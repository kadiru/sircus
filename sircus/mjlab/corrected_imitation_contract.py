"""Five verified development recipes, fixed actor refit and fresh paired TEST."""
from __future__ import annotations
import copy
from pathlib import Path
import numpy as np
from .benchmark_contract import RESIDUAL_SCALE, observation, action_from_residual
from .benchmark_verify import contained, digest, read, sha
from .table_training_contract import captured, clean, statistics, collect_exclusions
from .table_assisted_score import reward
from .table_correction_native import META

VERSION='sircus-corrected-imitation-v1'
AUDIT_VERSION='sircus-corrected-imitation-audit-v1'
METHODS=('corrected_bc','current_bc')
TEST_SEED=2026110941
INPUT_KEYS=('parent-study','parent-audit','height-study','height-audit','gap-study','gap-audit','ik-study','ik-audit','history-root','reference','assets')
CORRECTIONS=(('height','recovery-test-r3-c1-n0','corrected'),
    ('gap','recovery-test-r0-c5-n1','corrected'),('gap','recovery-test-r0-c6-n1','corrected'),
    ('ik','recovery-test-r1-c1-n1','parent_xy'),('ik','recovery-test-r2-c4-n0','parent_xy'))
SCOPE=('Actor-only imitation refit using unchanged92 prior teachers plus five audited corrected development recipes, '
       'one recipe per context, not repeated labels. Current recovery BC and corrected BC are frozen before140 predictions '
       'on70 new stratified bucket positions at native64/v3. Same19 geometry prior; original17 controls. '
       'No PPO, flight search at inference, noise,128, learned transit/perception or policy activation.')


def require(condition,message):
    if not condition:raise ValueError(message)


def protocol():
    return dict(substeps=64,methods=list(METHODS),test_seed=TEST_SEED,test_layouts=70,test_requests=140,
        control_requests=18,maximum_requests=158,base_teacher_count=92,correction_teacher_count=5,teacher_count=97,
        actor_steps=800,critic_steps=0,actor_lr=.001,noise=False,residual_scale=RESIDUAL_SCALE.tolist(),
        worker_limit_s=3600,queue_limit_s=900,global_limit_s=6300,audit_reserve_s=1800,
        storage_reserve_bytes=8*1024**3,minimum_split_distance_m=.005,
        corrections=[list(x) for x in CORRECTIONS],
        fitting='Current recovery BC actor-only800 full-batch Adam steps at.001 on92 preserved plus5 corrected teachers; critic/logstd unchanged',
        selection='Exactly one prespecified repeat0 recipe per corrected context, all three existing repeats captured; no new validation or checkpoint selection',
        test='Both weights and all140 predictions frozen before first TEST outcome; one request per policy/layout, no retries or learning',
        control_prior='Exact original17 control joints/proofs; both model comparisons share unchanged19',scope=SCOPE)


def authenticate_study(directory,audit,public,*,report_key='report_sha256',audit_key='proof_sha256',checks_count):
    directory,audit=Path(directory),Path(audit)
    expected_report=public[report_key];expected_audit=public[audit_key]
    require(sha(directory/'report.json')==expected_report and sha(audit)==expected_audit,'Published study/audit bytes differ')
    report,spec,proof,manifest=map(read,(directory/'report.json',directory/'contract.json',audit,directory/'evidence-manifest.json'))
    unsigned=dict(spec);fp=unsigned.pop('fingerprint')
    require(digest(unsigned)==fp==report['contract_fingerprint']==proof['contract_fingerprint'],'Frozen contract fingerprint differs')
    require(report['status']=='completed' and proof['status']=='passed' and proof['full_checks_verified'] is True
        and proof['native_checks_verified'] is True and proof['failures']==[]
        and proof['checks_count']==len(proof['checks'])==checks_count and all(x.get('passed') is True for x in proof['checks'])
        and proof['report_sha256']==expected_report and report['evidence_sha256']==manifest
        and proof['evidence_manifest_sha256']==sha(directory/'evidence-manifest.json'),'Complete bound native audit required')
    consumed={}
    def bound(name):
        path=contained(directory,name);actual=sha(path)
        require(actual==manifest[name],'Consumed evidence changed: '+name);consumed[name]=actual;return path
    bound('contract.json');rows=[__import__('json').loads(x) for x in bound('rows.jsonl').read_text().splitlines()]
    return dict(directory=directory,report=report,spec=spec,rows=rows,manifest=manifest,bound=bound,consumed=consumed,
        bindings=dict(report_sha256=expected_report,audit_sha256=expected_audit,contract_fingerprint=fp,
            evidence_manifest_sha256=sha(directory/'evidence-manifest.json'),checks_count=checks_count,files_sha256=consumed))


def bind_native(study,row,namespace=None,*,with_trace=True):
    namespace=namespace or row.get('adapter_directory','native')
    require(namespace in ('native','native-controls','control-parent/native'),'Known source native namespace required')
    for field in ('context_resolution','connection'):
        require(sha(study['bound'](namespace+'/'+row[field+'_file']))==row[field+'_sha256'],'Native setup proof differs')
    if with_trace and row.get('native_release'):
        require(sha(study['bound'](namespace+'/'+row['native_cache_file']))==row['native_cache_sha256'],'Native trajectory binding differs')


def corrected_teacher(rows,context,variant,source):
    group=sorted([r for r in rows if r.get('phase')=='diagnostic' and r.get('context_id')==context and r.get('variant')==variant],key=lambda r:r['repeat'])
    require(len(group)==3 and [r['repeat'] for r in group]==[0,1,2] and all(captured(r) for r in group),'Three matching captured nominal repeats required')
    first=group[0]
    for r in group:
        require(all(r[k]==first[k] for k in ('values','bucket_xy_m','q10','q14','connection_sha256','context_resolution_sha256')),'Correction repeats differ in action/admitted branch')
    require(source['context_id']==context and source['method']=='recovery_bc' and source['phase']=='test','Original recovery TEST row required')
    require(first['source_record_id']==source['id'] and first['source_record_sha256']==digest(source),'Correction source binding differs')
    require(all(first[k]==source[k] for k in META) and first['bucket_xy_m']==source['bucket_xy_m'],'Original anchor/observation metadata changed')
    obs=observation(source['bucket_xy_m'],source['effective_source_hint']).tolist()
    require(obs==source['observation'],'Original observation is not exact geometry features')
    residual=(np.asarray(first['values'],float)-np.asarray(source['effective_source_hint']['values'],float))/RESIDUAL_SCALE
    require(np.isfinite(residual).all() and (np.abs(residual)<1).all(),'Corrected teacher outside original actor residual support')
    require(np.allclose(action_from_residual(source['effective_source_hint']['values'],residual),first['values'],rtol=0,atol=1e-12),'Corrected teacher action reconstruction differs')
    teacher=copy.deepcopy(first)
    teacher.update(id='corrected-teacher-'+context,role='train',original_role=source['role'],source_correction_role=first['role'],development_reclassified=True,
        normalized_residual=residual.tolist(),observation=obs,learning_reward=reward(first),
        source_correction_record_id=first['id'],source_correction_record_sha256=digest(first),
        correction_repeat_ids=[r['id'] for r in group],correction_repeat_sha256={r['id']:digest(r) for r in group})
    return teacher,group


def load_inputs(a,root):
    root=Path(root);public=read(root/'baselines/mjlab-coverage-recovery-20261109/summary.json')
    parent=authenticate_study(a.parent_study,a.parent_audit,dict(report_sha256=public['report_sha256'],proof_sha256=public['audit']['proof_sha256']),checks_count=283251)
    require(len(parent['rows'])==970,'Full970-row current study required')
    out={}
    for key,name in (('prior','prior.json'),('old_prior','old-prior.json'),('controls','controls.json'),('control_replays','control-replays.json'),
        ('calibration','calibration.json'),('home','setup-home.json'),('teachers','bootstrap/teachers.json')):
        out[key]=read(parent['bound'](name))
    parent['bound']('verification.json')
    require(len(out['prior']['records'])==19 and len(out['old_prior']['records'])==17 and len(out['teachers'])==92,'Current19/17 priors and92teachers required')
    require(len({r['context_id'] for r in out['teachers']})==92 and all(r['role']=='train' and captured(r) for r in out['teachers']),'Preserved unique successful92TRAIN teachers required')
    artifact=copy.deepcopy(parent['report']['artifacts']['recovery_bc']);path=parent['bound'](artifact['path'])
    require(sha(path)==artifact['sha256'] and artifact['samples']==92,'Current recovery checkpoint differs')
    artifact['source_path']=str(path)
    for r in out['control_replays']:bind_native(parent,r,'control-parent/native',with_trace=False)
    for name in ('control-parent/native/effective-options.json','control-parent/native/native-compiled-geometry.json',
        'native/effective-options.json','native/native-compiled-geometry.json'):parent['bound'](name)
    studies={}
    for name,base,count in (('height','recovery-regression',22266),('gap','regression-drop-gap',16072),('ik','regression-ik',25329)):
        summary=read(root/f'baselines/mjlab-{base}-20261109/summary.json')
        identity=dict(report_sha256=summary['provenance']['report_sha256'],proof_sha256=summary['provenance']['independent_audit_sha256']) if name=='height' else dict(report_sha256=summary['report_sha256'],proof_sha256=summary['audit']['proof_sha256'])
        studies[name]=authenticate_study(getattr(a,name+'_study'),getattr(a,name+'_audit'),identity,checks_count=count)
        require(studies[name]['spec']['identity']==parent['spec']['identity'],'Correction scene identity differs')
    corrected=[];lineage=[];sources=[]
    for kind,context,variant in CORRECTIONS:
        candidates=[r for r in parent['rows'] if r.get('phase')=='test' and r.get('method')=='recovery_bc' and r['context_id']==context]
        require(len(candidates)==1,'Unique original recovery context required');source=candidates[0];sources.append(source);bind_native(parent,source)
        teacher,group=corrected_teacher(studies[kind]['rows'],context,variant,source)
        for r in group:bind_native(studies[kind],r)
        corrected.append(teacher);lineage.append(dict(context_id=context,study=kind,variant=variant,repeat=0,
            teacher_id=teacher['id'],teacher_sha256=digest(teacher),source_record_id=source['id'],source_record_sha256=digest(source),
            correction_record_id=group[0]['id'],correction_record_sha256=digest(group[0]),
            confirmation_record_ids=[r['id'] for r in group],confirmation_record_sha256={r['id']:digest(r) for r in group}))
    teachers=sorted(copy.deepcopy(out['teachers'])+corrected,key=lambda r:r['context_id'])
    require(len(teachers)==97 and len({r['id'] for r in teachers})==97 and len({r['context_id'] for r in teachers})==97,'Exactly97unique teachers; no repeated labels')
    out.update(parent=parent['spec'],parent_report=parent['report'],rows=parent['rows'],sources=sources,
        artifacts={'current_bc':artifact},corrected_teachers=corrected,all_teachers=teachers,teacher_lineage=lineage,
        bindings=dict(parent=parent['bindings'],corrections={k:s['bindings'] for k,s in studies.items()},teacher_lineage=lineage,
            old_teacher_file_sha256=sha(parent['bound']('bootstrap/teachers.json')),
            old_teacher_sha256={r['id']:digest(r) for r in out['teachers']},
            original_test_reclassified='Five corrected inspected contexts are development; original37/70 unchanged'))
    return out


def test_split(excluded):
    points=np.asarray(sorted(set(tuple(map(float,p)) for p in excluded)),float)
    require(points.ndim==2 and points.shape[1]==2 and len(points) and np.isfinite(points).all(),'Finite historical coordinates required')
    rng=np.random.default_rng(TEST_SEED);low=np.asarray([.1,-.65]);size=(np.asarray([1.24,.25])-low)/[7,5];result=[]
    for iy in range(5):
        for ix in range(7):
            for n in range(2):
                for _ in range(10000):
                    xy=low+size*([ix,iy]+rng.uniform(.05,.95,2))
                    if np.linalg.norm(points-xy,axis=1).min()>=.005:break
                else:raise ValueError('Disjoint cell sampling exhausted')
                result.append(dict(id=f'corrected-imitation-test-r{iy}-c{ix}-n{n}',split='test',bucket_xy_m=xy.tolist(),grid_cell=[ix,iy]))
                points=np.vstack((points,xy))
    return result
