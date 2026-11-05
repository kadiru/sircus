"""Dense calibrated execution only; exact admitted held endpoints are preserved."""
from pathlib import Path
import json

import numpy as np

from . import clean650_contract as protocol
from .sweep_worker import atomic
from .corrected_contract import sha


class NativeExecutor:
    def __init__(self,scene,assets,calibration,output,batch,publish):
        self.scene,self.assets,self.calibration,self.output,self.batch,self.publish=scene,assets,calibration,Path(output),batch,publish
        self.rows=[];self.positions={};self.batches=[];self.environments={};self.options={}

    def environment(self,step):
        if step not in self.environments:
            from .calibrated_env import CalibratedEnvironment,verified_native_options
            from .expanded_actions import WIDE_BOUNDS
            env=CalibratedEnvironment(self.scene,self.assets,self.calibration['score_config'],n=self.batch,bounds=WIDE_BOUNDS,substeps=step)
            self.environments[step]=env;native=self.output/'native'/str(step);native.mkdir(parents=True,exist_ok=False)
            atomic(native/'native-compiled-geometry.json',env.compiled_geometry)
            self.options[str(step)]=verified_native_options(env.sim,step)
            atomic(native/'effective-options.json',self.options[str(step)]);atomic(self.output/'effective-options.json',self.options)
        return self.environments[step]

    def execute(self,requests):
        from .bank_geometry import fixed_endpoints
        from .learning_contract import constraints
        for step in dict.fromkeys(r['substeps'] for r in requests):
            selected=[r for r in requests if r['substeps']==step]
            for offset in range(0,len(selected),self.batch):
                chunk=selected[offset:offset+self.batch];active=[r for r in chunk if r['eligible']]
                packet=len(self.batches);measured=[];filename=None
                native=self.output/'native'/str(step)
                if active:
                    env=self.environment(step);bucket=np.asarray([r['bucket_xy_m'] for r in active]);values=np.asarray([r['values'] for r in active])
                    q10,q14=(np.asarray([r[k] for r in active]) for k in ('q10','q14'))
                    with fixed_endpoints(env,bucket,values,q10,q14,np.ones(len(active),bool)):
                        measured,timing=env.run(bucket,values,constraints(len(active)),record=True)
                    if len(measured)!=len(active):raise RuntimeError('Native result count differs from exact admitted request count')
                    arrays=env.last_arrays;filename=f'native-{packet:04d}.npz'
                    np.savez_compressed(native/filename,**arrays,record_ids=np.asarray([r['id'] for r in active]));cache_hash=sha(native/filename)
                lookup={r['id']:(i,m) for i,(r,m) in enumerate(zip(active,measured))}
                for request in chunk:
                    row=dict(request,invalid='strict_endpoint_rejected',native_release=False,route_success=False,route_quality=0,
                        whole_ball_entry=False,board_bounce=False,max_retention_s=0.,invalid_physics=False,scalar_score_audit_passed=False)
                    if request['id'] in lookup:
                        i,m=lookup[request['id']]
                        if any(m[k]!=request[k] for k in ('values','bucket_xy_m','q10','q14')):raise RuntimeError('Native exact frozen query changed')
                        row.update(m);row.pop('trajectory_m',None)
                        row.update(native_cache_file=filename,native_cache_row=i,native_cache_sha256=cache_hash,native_directory='native/'+str(step))
                        self.positions[row['id']]=arrays['position_m'][:,i].copy()
                    self.rows.append(row)
                    with (self.output/'records.jsonl').open('a') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
                self.batches.append(dict(packet=packet,substeps=step,phase=chunk[0]['phase'],record_ids=[r['id'] for r in chunk],
                    native_record_ids=[r['id'] for r in active],native_directory='native/'+str(step),cache_file=filename))
                atomic(self.output/'batches.json',self.batches);self.publish(chunk[0]['phase']+'-'+str(step))
