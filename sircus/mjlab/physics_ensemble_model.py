"""Small bootstrapped state-based forward model, not a video WAM or new PPO."""
import copy
import json
from pathlib import Path
import numpy as np
import torch
from .physics_ensemble_contract import MODEL_VERSION, SEEDS, contained, digest, sha, captured

INPUTS=('board_x_m','board_y_m','board_height_m','tilt_deg','gap_m','bucket_x_m','bucket_y_m')
TARGETS=('native_execution','board_bounce','whole_ball_entry','capture','retention_fraction',
         'log_first_flight_opening_distance','rebound_forward_m_s','rebound_lateral_m_s','rebound_vertical_m_s')
CENTER=np.asarray([.67,-.2,.35,30.,.3,.67,-.2])
SCALE=np.asarray([.57,.45,.3,30.,.3,.57,.45])
RESIDUAL_SCALE=np.asarray([.04,.04,.05,8.,.12])

def inputs(bucket,values):
    x=np.r_[values,bucket].astype(float)
    if x.shape!=(7,) or not np.isfinite(x).all():raise ValueError('Finite7-dimensional geometry required')
    return ((x-CENTER)/SCALE).tolist()

def target(row,trace,scene,calibration):
    from .xy650_metrics import first_flight
    native=row.get('native_release') is True
    valid=native and not row.get('invalid') and row.get('invalid_physics') is False
    y=np.zeros(9);mask=np.zeros(9);mask[0]=1.;y[0]=float(native)
    metric=dict(available=False,reason='unavailable_or_invalid_trace')
    if valid:
        y[1:5]=[bool(row['board_bounce']),bool(row['whole_ball_entry']),captured(row),min(1.,float(row['max_retention_s']))]
        mask[1:5]=1.
        metric=first_flight(scene,row['bucket_xy_m'],row['board_normal'],calibration,trace,valid=True)
        if metric['available']:
            i=metric['start_index'];velocity=np.asarray(trace['velocity_m_s'][i],float)
            direction=np.asarray(row['bucket_xy_m'])-np.asarray(row['values'][:2]);direction/=np.linalg.norm(direction)
            y[5]=np.log1p(float(metric['closest_opening_distance_m'])/.1)
            y[6:9]=[velocity[:2]@direction,velocity[:2]@np.array([-direction[1],direction[0]]),velocity[2]]
            mask[5:9]=1.
    if not np.isfinite(y).all():raise ValueError('Finite measured targets required')
    return y.tolist(),mask.tolist(),metric

def build_dataset(rows,native_dir,scene,calibration,*,require_train=True):
    """Only authenticated TRAIN gets fitted; evaluation may use the same metric with an explicit flag."""
    if require_train and any(r.get('role')!='train' for r in rows):raise ValueError('TRAIN only may fit forward model')
    if not rows or len({r['id'] for r in rows})!=len(rows):raise ValueError('Nonempty unique rows required')
    grouped={};traces={};native_dir=Path(native_dir)
    for r in rows:
        if r.get('native_release'):grouped.setdefault(r['native_cache_file'],[]).append(r)
    for name,group in grouped.items():
        path=contained(native_dir,name);expected=sha(path)
        with np.load(path,allow_pickle=False) as z:
            names=('position_m','velocity_m_s','forces_N','time_s','dt_s','contact_distance_m')
            arrays={k:z[k] for k in names if k in z.files};ids=z['record_ids'].tolist()
            for r in group:
                i=r['native_cache_row']
                if expected!=r['native_cache_sha256'] or str(ids[i])!=r['id']:raise ValueError('Bound TRAIN native cache differs')
                trace={k:v[:,i] if v.ndim>=2 and k!='time_s' else v for k,v in arrays.items()}
                y,mask,metric=target(r,trace,scene,calibration)
                traces[r['id']]=(y,mask,metric)
    result=[]
    for r in rows:
        y,mask,metric=traces.get(r['id'],target(r,None,scene,calibration) if not r.get('native_release') else (None,None,None))
        if y is None:raise ValueError('Native target trace missing')
        result.append(dict(id=r['id'],role=r['role'],context_id=r['context_id'],bucket_xy_m=r['bucket_xy_m'],values=r['values'],
            x=inputs(r['bucket_xy_m'],r['values']),y=y,mask=mask,record_sha256=digest(r),
            native_cache_sha256=r.get('native_cache_sha256'),native_cache_row=r.get('native_cache_row'),
            native_cache_file=r.get('native_cache_file'),first_flight=metric,
            physics_invalid=bool(r.get('invalid_physics')),native_release=r.get('native_release') is True))
    return result

def network():
    return torch.nn.Sequential(torch.nn.Linear(7,64),torch.nn.SiLU(),torch.nn.Linear(64,64),torch.nn.SiLU(),torch.nn.Linear(64,9))

def loss_tensor(output,y,mask):
    binary=torch.nn.functional.binary_cross_entropy_with_logits(output[:,:4],y[:,:4],reduction='none')
    reg=output[:,4:].clone();reg[:,0]=reg[:,0].sigmoid()
    losses=torch.cat((binary,(reg-y[:,4:])**2),dim=1)
    return ((losses*mask).sum(0)/mask.sum(0).clamp_min(1)).sum()

def fit(dataset,output,contract_fingerprint):
    if not dataset or any(r.get('role')!='train' for r in dataset):raise ValueError('TRAIN only')
    x,y,mask=(torch.tensor(np.asarray([r[k] for r in dataset]),dtype=torch.float32) for k in ('x','y','mask'))
    if x.shape!=(len(dataset),7) or y.shape!=(len(dataset),9) or mask.shape!=y.shape or not all(torch.isfinite(t).all() for t in (x,y,mask)):
        raise ValueError('Finite masked forward-model dataset required')
    if not bool(((mask==0)|(mask==1)).all()):raise ValueError('Binary observed-target masks required')
    output=Path(output);output.mkdir(parents=True,exist_ok=True);path=output/'ensemble.pt'
    if path.exists():raise ValueError('New model checkpoint required')
    torch.set_num_threads(2);states=[];initial=[];optimizers=[];bootstraps=[];curves=[];weight_ids=[]
    for member,seed in enumerate(SEEDS):
        torch.manual_seed(seed);net=network().cpu();initial.append(copy.deepcopy(net.state_dict()))
        idx=np.random.default_rng(seed).integers(0,len(dataset),len(dataset)).tolist();bootstraps.append(idx)
        xb,yb,mb=x[idx],y[idx],mask[idx];opt=torch.optim.Adam(net.parameters(),lr=.001)
        for step in range(601):
            loss=loss_tensor(net(xb),yb,mb)
            if step%100==0:
                with torch.no_grad():full=float(loss_tensor(net(x),y,mask))
                curves.append(dict(member=member,step=step,train_loss=float(loss.detach()),full_train_loss=full))
            if step==600:break
            opt.zero_grad();loss.backward();opt.step()
        states.append(copy.deepcopy(net.state_dict()));optimizers.append(opt.state_dict())
    state=dict(version=MODEL_VERSION,contract_fingerprint=contract_fingerprint,fit_device='cpu',fit_steps=600,lr=.001,
        member_seeds=list(SEEDS),dataset_sha256=digest(dataset),input_features=list(INPUTS),target_features=list(TARGETS),
        center=CENTER.tolist(),scale=SCALE.tolist(),members=states,initial_members=initial,optimizers=optimizers,
        bootstrap_indices=bootstraps,curves=curves)
    torch.save(state,path)
    artifact=dict(path='ensemble.pt',sha256=sha(path),version=MODEL_VERSION,dataset_sha256=state['dataset_sha256'],
        ensemble_members=5,fit_device='cpu',fit_steps=600,curves=curves,input_features=list(INPUTS),target_features=list(TARGETS))
    (output/'fit.json').write_text(json.dumps(artifact,indent=2,allow_nan=False)+'\n')
    return artifact

class Predictor:
    def __init__(self,state):
        if state['version']!=MODEL_VERSION or state['center']!=CENTER.tolist() or state['scale']!=SCALE.tolist() or len(state['members'])!=5:
            raise ValueError('Exact model contract required')
        self.state=state;self.members=[]
        for weights in state['members']:
            net=network();net.load_state_dict(weights);net.eval();self.members.append(net)

    def predict_inputs(self,x):
        xx=torch.tensor(np.asarray(x),dtype=torch.float32)
        if xx.ndim!=2 or xx.shape[1]!=7 or not torch.isfinite(xx).all():raise ValueError('Finite normalized input matrix required')
        with torch.no_grad():
            outputs=torch.stack([n(xx) for n in self.members]).numpy().astype(float)
        predictions=[]
        for i in range(len(x)):
            o=outputs[:,i,:];prob=1/(1+np.exp(-np.clip(o[:,:5],-50,50)))
            distance=.1*np.expm1(np.clip(o[:,5],0.,6.))
            product=prob[:,0]*prob[:,3]
            predictions.append(dict(admission_probability=float(prob[:,0].mean()),board_bounce_probability=float(prob[:,1].mean()),
                whole_entry_probability=float(prob[:,2].mean()),capture_probability=float(product.mean()),
                conditional_capture_probability=float(prob[:,3].mean()),uncertainty=float(product.std(ddof=0)),
                retention_fraction=float(prob[:,4].mean()),aperture_distance_m=float(distance.mean()),
                rebound_velocity_m_s=o[:,6:9].mean(0).tolist(),rebound_basis='board_to_bucket_forward_lateral_vertical',
                member_raw_output=o.tolist(),member_capture_probability=product.tolist(),
                ranking_score=float(product.mean()-.25*product.std(ddof=0)-.05*np.minimum(distance,2.).mean())))
        return predictions

def load(path):
    return Predictor(torch.load(path,map_location='cpu',weights_only=True))

def candidates(context,anchor,bc_values,ppo_values,train_rows):
    if not train_rows or any(r.get('role')!='train' for r in train_rows):raise ValueError('Candidate teachers must be TRAIN only')
    av=np.asarray(anchor['values'],float);bc=(np.asarray(bc_values)-av)/RESIDUAL_SCALE
    pp=(np.asarray(ppo_values)-av)/RESIDUAL_SCALE
    eligible=[r for r in train_rows if captured(r)]
    if not eligible:raise ValueError('At least one TRAIN capture required for declared candidate family')
    nearest=min(eligible,key=lambda r:(float(np.linalg.norm(np.asarray(r['bucket_xy_m'])-context['bucket_xy_m'])),-int(r.get('route_quality')==2),-r['learning_reward'],r['id']))
    nearest_residual=(np.asarray(nearest['values'])-np.asarray(nearest['effective_source_hint']['values']))/RESIDUAL_SCALE
    residuals=[bc,pp,np.zeros(5),nearest_residual];kinds=['saved_bc','saved_ppo','shared_geometry_anchor','nearest_current_train_capture']
    # Eight independent fixed axis changes avoid averaging incompatible capture families.
    for j,delta in ((2,-.3),(2,.3),(3,-.35),(3,.35),(4,-.3),(4,.3),(0,-.3),(0,.3)):
        v=bc.copy();v[j]+=delta;residuals.append(v);kinds.append(f'bc_axis_{j}_{delta}')
    seed=int(digest(dict(context_id=context['id'],seed=20261621))[:16],16);rng=np.random.default_rng(seed)
    for _ in range(4):residuals.append(bc+rng.uniform(-.35,.35,5));kinds.append('seeded_bc_joint_neighborhood')
    result=[]
    for i,(residual,kind) in enumerate(zip(residuals,kinds)):
        residual=np.clip(residual,-1.,1.);values=av+residual*RESIDUAL_SCALE
        result.append(dict(id=f'candidate-{i:02d}',values=values.tolist(),normalized_residual=residual.tolist(),source_kind=kind,
            source_training_ids=[nearest['id']] if i==3 else []))
    return result

def rank(predictor,context,anchor,items):
    if len(items)!=16:raise ValueError('Literal16 model-only proposals required')
    predictions=predictor.predict_inputs([inputs(context['bucket_xy_m'],r['values']) for r in items])
    order=sorted(range(16),key=lambda i:(-predictions[i]['ranking_score'],items[i]['id']))
    return dict(order=order,predictions=predictions,candidate_count=16,ball_flight_search_at_inference=False)

def prediction_metrics(predictor,dataset,split):
    if not dataset:return []
    pred=predictor.predict_inputs([r['x'] for r in dataset]);y=np.asarray([r['y'] for r in dataset]);mask=np.asarray([r['mask'] for r in dataset],bool)
    fields=('admission_probability','board_bounce_probability','whole_entry_probability','conditional_capture_probability','retention_fraction','aperture_distance_m')
    result=[]
    for j,name in enumerate(TARGETS):
        use=mask[:,j];n=int(use.sum())
        if not n:result.append(dict(split=split,target=name,support=0));continue
        actual=y[use,j]
        predicted=np.asarray([p[fields[j]] if j<6 else p['rebound_velocity_m_s'][j-6] for p in pred])[use]
        if j==5:actual=.1*np.expm1(actual)
        d=predicted-actual;row=dict(split=split,target=name,support=n,mae=float(np.abs(d).mean()),rmse=float(np.sqrt((d*d).mean())),
            unit='m' if j==5 else 'm/s' if j>=6 else 'fraction')
        if j<4:row.update(brier=float((d*d).mean()),accuracy=float(((predicted>=.5)==(actual>=.5)).mean()))
        result.append(row)
    use=np.asarray([not r.get('physics_invalid') for r in dataset],bool)
    if use.any():
        actual=y[:,0]*y[:,3];predicted=np.asarray([p['capture_probability'] for p in pred]);d=(predicted-actual)[use]
        result.append(dict(split=split,target='requested_capture_uncalibrated_score',support=int(use.sum()),
            brier=float((d*d).mean()),mae=float(np.abs(d).mean()),rmse=float(np.sqrt((d*d).mean())),unit='fraction'))
    return result
