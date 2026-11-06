"""Off-center native releases through the unchanged calibrated rollout/scorers.

A scoped endpoint provider supplies the real release center and corresponding
dropper flange target. The existing rollout places the ball at that center and
checks actual native FK against that target; no center-only spawn is retained.
"""

from contextlib import contextmanager
import json

import numpy as np

from . import impact650_contract as protocol
from .bank_geometry import FixedEndpointIK
from .corrected_contract import sha
from .impact650_geometry import BASIS_VERSION, GAP_DEFINITION, impact_targets
from .sweep_worker import atomic
from .xy650_native import NativeExecutor as CalibratedExecutor


VERSION = "sircus-impact650-native-release-provider-v1"


class ImpactFixedEndpointIK(FixedEndpointIK):
    """Exact admitted branches, with off-center targets consumed by native reset."""
    def __init__(self,scene,bucket,values,offsets,q10,q14,valid,*,device='cpu'):
        super().__init__(scene,bucket,values,q10,q14,valid,device=device)
        self.offsets=np.asarray(offsets,dtype=float).copy()
        if self.offsets.shape!=(len(self.bucket),2) or not np.isfinite(self.offsets).all():
            raise ValueError('Finite per-request impact offsets required')
        self.offsets.setflags(write=False)

    def solve(self,bucket_xy,board_xyz,tilt_deg,gap_m,**unused):
        result=super().solve(bucket_xy,board_xyz,tilt_deg,gap_m,**unused)
        n=len(bucket_xy)
        offsets=np.concatenate((self.offsets,np.repeat(self.offsets[-1:],n-len(self.offsets),axis=0)))
        values=np.column_stack((board_xyz,np.broadcast_to(tilt_deg,(n,)),np.broadcast_to(gap_m,(n,))))
        result.update(targets=impact_targets(self.scene,bucket_xy,values,offsets),version=VERSION)
        return result


@contextmanager
def fixed_impact_endpoints(environment,bucket,values,offsets,q10,q14,valid):
    previous=environment.ik
    environment.ik=ImpactFixedEndpointIK(environment.scene,bucket,values,offsets,q10,q14,valid,device=environment.device)
    try:yield environment
    finally:environment.ik=previous


class NativeExecutor(CalibratedExecutor):
    """Same engine/options/physics; each actual release carries its offset query."""
    def execute(self,requests):
        from .learning_contract import constraints
        from .xy650_metrics import first_flight
        for step in dict.fromkeys(r['substeps'] for r in requests):
            selected=[r for r in requests if r['substeps']==step]
            for offset in range(0,len(selected),self.batch):
                chunk=selected[offset:offset+self.batch];active=[r for r in chunk if r['eligible']]
                packet=len(self.batches);measured=[];filename=None;nominal=None
                native=self.output/'native'/str(step)
                if active:
                    env=self.environment(step)
                    bucket,values,impacts=(np.asarray([r[k] for r in active],float) for k in ('bucket_xy_m','values','impact_offset_m'))
                    q10,q14=(np.asarray([r[k] for r in active],float) for k in ('q10','q14'))
                    nominal=impact_targets(self.scene,bucket,values,impacts)
                    with fixed_impact_endpoints(env,bucket,values,impacts,q10,q14,np.ones(len(active),bool)):
                        measured,timing=env.run(bucket,values,constraints(len(active)),record=True)
                    if len(measured)!=len(active):raise RuntimeError('Native result count differs from admitted impact requests')
                    arrays=env.last_arrays;filename=f'native-{packet:04d}.npz'
                    np.savez_compressed(native/filename,**arrays,record_ids=np.asarray([r['id'] for r in active]));cache_hash=sha(native/filename)
                lookup={r['id']:(i,m) for i,(r,m) in enumerate(zip(active,measured))}
                for request in chunk:
                    row=dict(request,invalid='strict_endpoint_rejected',native_release=False,route_success=False,route_quality=0,
                        whole_ball_entry=False,board_bounce=False,max_retention_s=0.,invalid_physics=False,scalar_score_audit_passed=False)
                    if request['id'] in lookup:
                        i,m=lookup[request['id']]
                        if any(m[k]!=request[k] for k in ('values','bucket_xy_m','q10','q14')):
                            raise RuntimeError('Native frozen impact query changed')
                        expected=nominal['release_center_m'][i]
                        if not np.allclose(m['release_ball_center_m'],expected,rtol=0,atol=1e-12):
                            raise RuntimeError('Native release metadata used a center-only target')
                        error=float(np.linalg.norm(arrays['position_m'][0,i]-expected))
                        if not np.isfinite(error) or error>1e-6:
                            raise RuntimeError('Actual native ball spawn differs from off-center release')
                        if not np.allclose(m['native_initial_ball_center_m'],arrays['position_m'][0,i],rtol=0,atol=0):
                            raise RuntimeError('Native initial ball observation disagrees with dense trace')
                        if not 0<=float(m['pose_gate']['dropper_error_mm'])<2.:
                            raise RuntimeError('Actual dropper FK does not match shifted release target')
                        row.update(m);row.pop('trajectory_m',None)
                        row.update(impact_basis_version=BASIS_VERSION,impact_gap_definition=GAP_DEFINITION,
                            impact_point_world_m=nominal['impact_point_world_m'][i].tolist(),release_center_m=expected.tolist(),
                            release_center_height_m=float(expected[2]),impact_to_bucket_horizontal_m=float(nominal['impact_to_bucket_horizontal_m'][i]),
                            native_release_target_error_m=error,native_cache_file=filename,native_cache_row=i,native_cache_sha256=cache_hash,native_directory='native/'+str(step))
                        self.positions[row['id']]=arrays['position_m'][:,i].copy()
                        trace={k:(v[:,i] if v.ndim>=2 and k!='time_s' else v) for k,v in arrays.items()}
                        row['first_flight']=first_flight(self.scene,row['bucket_xy_m'],row['board_normal'],self.calibration['score_config'],trace,
                            valid=not row.get('invalid') and row.get('invalid_physics') is False)
                        row['execution_fingerprint']=protocol.execution_fingerprint(row)
                    if not row['native_release']:row.update(first_flight=None,execution_fingerprint=None)
                    self.rows.append(row)
                    with (self.output/'records.jsonl').open('a') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
                self.batches.append(dict(packet=packet,substeps=step,phase=chunk[0]['phase'],record_ids=[r['id'] for r in chunk],
                    native_record_ids=[r['id'] for r in active],native_directory='native/'+str(step),cache_file=filename))
                atomic(self.output/'batches.json',self.batches);self.publish(chunk[0]['phase']+'-'+str(step))
