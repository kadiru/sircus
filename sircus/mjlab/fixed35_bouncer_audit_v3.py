"""Explicit CPU audit correction for near-zero acos certificate roundoff.

V1 native evidence and V1/V2 audit failures stay immutable. V3 retains the V2
historical metadata adapter and separately reproduces every saved endpoint
certificate from its bound target matrix. The independent target construction
still uses the original tolerance. Only a derived near-zero normal-angle scalar
may differ by two microdegrees; all other certificate fields, acceptance reasons,
5mm clearances, joint values and score checks remain exact and unchanged.
"""
from __future__ import annotations
import argparse
import copy
import json
import math
from pathlib import Path
from unittest.mock import patch
import numpy as np
from . import fixed35_bouncer_audit_v2 as v2
from . import fixed35_bouncer_verify as original
from .fixed35_bouncer_verify import previous,fixed,yaw,audit_candidate_identity
from .benchmark_verify import EvidenceError,contained,digest,read,sha

VERSION='sircus-fixed35-bouncer-audit-v3'
ANGLE_ROUNDOFF_TOLERANCE_DEG=2e-6
NEAR_ZERO_ANGLE_LIMIT_DEG=1e-4


def compare_certificates(gate,candidate,row,targets,independent,endpoint,checks):
    label=row['id'];saved=endpoint['fresh_endpoint']
    for key in ('dropper_flange_world','board_flange_local','release_center_m'):
        checks.numeric(endpoint['targets'][key],targets[key],label+'/v3 same independently constructed target/'+key,1e-12)
    bound={key:np.asarray(value) for key,value in endpoint['targets'].items()}
    reproduced=gate.check(candidate['q10'],candidate['q14'],row['bucket_xy_m'],row['values'],bound)
    checks.equal(reproduced,saved,label+'/v3 entire saved-target certificate reproduced exactly')
    first,second=copy.deepcopy(independent),copy.deepcopy(saved)
    a,b=first.pop('board_normal_error_deg'),second.pop('board_normal_error_deg')
    checks.equal(first,second,label+'/v3 every other independent endpoint field remains exact')
    if a!=b:
        checks.require(np.isfinite([a,b]).all() and 0<=a<=NEAR_ZERO_ANGLE_LIMIT_DEG and 0<=b<=NEAR_ZERO_ANGLE_LIMIT_DEG,
            label+'/v3 numeric exception restricted to near-zero normal angle')
        checks.numeric(a,b,label+'/v3 derived acos roundoff only',ANGLE_ROUNDOFF_TOLERANCE_DEG)
    return dict(record_id=label,independent_normal_error_deg=a,saved_normal_error_deg=b,
        absolute_difference_deg=abs(a-b),tolerance_deg=ANGLE_ROUNDOFF_TOLERANCE_DEG,
        saved_target_certificate_exact=True,other_independent_fields_exact=True,
        minimum_clearance_lower_bound_m=independent.get('minimum_clearance_lower_bound_m'),
        reason=independent['reason'])


def corrected_geometry(study,rows,scene,assets,data,checks):
    from .learning_ik import read_chain
    from .corrected_geometry import StrictEndpointGate
    from .admission import native_pose_error,base14
    old=[r for r in rows if r['phase'] not in ('search','repeat')]
    previous.audit_geometry(study,old,scene,assets,checks)
    chains=(read_chain(assets/'dropper.urdf'),read_chain(assets/'board_arm.urdf'))
    gate=StrictEndpointGate(scene,assets,.005);seen=set();measured=[];numerics=[]
    for row in rows:
        if row['phase'] not in ('search','repeat'):continue
        label=row['id'];namespace=study/'native-yaw'
        checks.equal(row['adapter_directory'],'native-yaw',label+'/oriented native namespace')
        path=contained(namespace,row['connection_file']);checks.equal(sha(path),row['connection_sha256'],label+'/immutable saved candidate proof')
        connection=read(path);targets,_=fixed.audit_behavior(connection,scene,checks,label)
        candidate,endpoint=audit_candidate_identity(namespace,row,connection,data,checks)
        if row['connection_sha256'] not in seen:
            for field in ('dropper_flange_world','board_flange_local','release_center_m'):
                checks.numeric(endpoint['targets'][field],targets[field],label+'/independent exact target/'+field,1e-12)
            checked=gate.check(candidate['q10'],candidate['q14'],row['bucket_xy_m'],row['values'],targets)
            numerics.append(compare_certificates(gate,candidate,row,targets,checked,endpoint,checks))
            q=np.r_[candidate['q10'],candidate['q14']]
            # Native model FK corroborates the independent URDF endpoint reconstruction.
            errors=[native_pose_error(gate.strict,role,q,target) for role,target in (
                (10,targets['dropper_flange_world']),(14,base14(scene)@targets['board_flange_local']))]
            for got,wanted in zip(endpoint['native_pose_errors'],errors):
                for field,value in wanted.items():checks.numeric(got[field],value,label+'/native FK recheck/'+field,1e-10)
            accepted=not checked['reason'] and all(e['position_m']<1e-4 and .25*e['rotation_rad']<1e-4 for e in errors)
            checks.equal(endpoint['passed'],accepted,label+'/exact full-pose and strict clearance acceptance')
            checks.equal(endpoint['reason'],None if accepted else 'saved_candidate_endpoint_recheck_rejected',label+'/honest endpoint rejection')
            if accepted:
                checks.require(checked['minimum_clearance_lower_bound_m']>=.005-1e-12,label+'/original5mm endpoint clearance')
                for chain,field,target in zip(chains,('q10','q14'),(targets['dropper_flange_world'],targets['board_flange_local'])):
                    pose=yaw.forward_chain(chain,candidate[field])
                    checks.require(np.linalg.norm(pose[:3,3]-target[:3,3])<1e-4,label+'/saved full FK position/'+field)
                    checks.require(yaw.rotation_error(pose[:3,:3],target[:3,:3])<4e-4,label+'/saved full FK orientation/'+field)
            seen.add(row['connection_sha256'])
        evidence=row['yaw_evidence']
        checks.equal(row['release_behavior_version'],'sircus-continuous-bucket-facing-release-v1',label+'/fixed bucket-facing behavior')
        checks.numeric(evidence['requested_yaw_rad'],targets['yaw_rad'],label+'/display requested yaw',1e-12)
        checks.numeric(evidence['requested_rotation_matrix'],targets['dropper_flange_world'][:3,:3],label+'/display requested rotation',1e-12)
        if not row.get('native_release'):
            checks.require(not row.get('native_initial_q10') and not row.get('native_initial_q14') and not row.get('native_cache_file'),label+'/declined setup has no fabricated flight')
            continue
        checks.require(fixed.target_domain(row['values'],row['bucket_xy_m']),label+'/unchanged numeric bounds')
        poses=[]
        for chain,field,target in zip(chains,('q10','q14'),(targets['dropper_flange_world'],targets['board_flange_local'])):
            checks.equal(row['native_initial_'+field],np.asarray(candidate[field],np.float32).astype(float).tolist(),label+'/actual saved float32 joint reset/'+field)
            pose=yaw.forward_chain(chain,row['native_initial_'+field]);poses.append(pose)
            checks.require(np.linalg.norm(pose[:3,3]-target[:3,3])<1e-4,label+'/actual endpoint position/'+field)
            checks.require(yaw.rotation_error(pose[:3,:3],target[:3,:3])<4e-4,label+'/actual full endpoint orientation/'+field)
        checks.numeric(row['release_ball_center_m'],targets['release_center_m'],label+'/literal release center',1e-12)
        checks.numeric(row['board_center_m'],row['values'][:3],label+'/settled board center',1e-4)
        checks.numeric(row['board_normal'],targets['board_normal'],label+'/settled board normal',4e-4)
        drop,board=poses
        checks.equal(evidence['basis'],'model FK of observed native initial float32 joints; held endpoint',label+'/measured pose provenance')
        displayed=dict(measured_initial_rotation_matrix=drop[:3,:3],measured_initial_yaw_rad=math.atan2(drop[1,0],drop[0,0]),
            rotation_error_rad=yaw.rotation_error(drop[:3,:3],targets['dropper_flange_world'][:3,:3]),
            position_error_m=np.linalg.norm(drop[:3,3]-targets['dropper_flange_world'][:3,3]),
            board_rotation_error_rad=yaw.rotation_error(board[:3,:3],targets['board_flange_local'][:3,:3]),
            board_position_error_m=np.linalg.norm(board[:3,3]-targets['board_flange_local'][:3,3]),
            measured_initial_flange_position_m=drop[:3,3],requested_release_center_m=targets['release_center_m'])
        for key,wanted in displayed.items():checks.numeric(evidence[key],wanted,label+'/independent displayed pose/'+key,1e-9)
        measured.append(dict(record_id=label,yaw_rad=targets['yaw_rad']))
    return dict(new_proofs=len(seen),measured_yaw_releases=len(measured),measurements=measured,certificate_numerics=numerics)



def audit(*args,original_v2_failure,**kwargs):
    failure_path=Path(original_v2_failure);failure=read(failure_path)
    if (failure.get('version')!=v2.VERSION or failure.get('status')!='failed'
            or not any('independent complete5mm endpoint recheck' in str(f) for f in failure.get('failures',[]))):
        raise EvidenceError('The separately retained v2 exact-dictionary roundoff failure is required')
    with patch.object(v2,'ORIGINAL_GEOMETRY_AUDIT',corrected_geometry):
        result=v2.audit(*args,**kwargs)
    historical=result['audit_correction']
    result.update(version=VERSION,auditor_sha256=sha(Path(__file__)),
        audit_correction=dict(version=VERSION,kind='Explicit audit-only metadata and near-zero angular-roundoff adapters',
            historical_metadata_adapter=historical,original_v2_failure=failure,
            original_v2_failure_sha256=sha(failure_path),adapter_v2_sha256=sha(Path(v2.__file__)),
            adapter_sha256=sha(Path(__file__)),angle_roundoff_tolerance_deg=ANGLE_ROUNDOFF_TOLERANCE_DEG,
            near_zero_angle_limit_deg=NEAR_ZERO_ANGLE_LIMIT_DEG,
            certificate_numerics=result['yaw_geometry']['certificate_numerics'],
            producer_source_unchanged=True,native_rows_unchanged=True,report_unchanged=True,
            physics_rerun=False,scoring_changed=False,admission_checks_relaxed=False))
    result['limitations'].append('V2 exact certificate equality failed solely on near-zero acos roundoff. V3 reproduces complete saved certificates exactly and tolerates at most2 microdegrees only in independently derived near-zero normal-angle diagnostics; clearance/acceptance gates are unchanged.')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('study','source','assets','parent-study','parent-audit','original-study','original-audit','recovery-study','recovery-audit','reference','original-failure','original-v2-failure','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists() or args.output.resolve().is_relative_to(args.study.resolve()):
        raise ValueError('Fresh independent proof outside immutable experiment required')
    try:
        kwargs=vars(args).copy();kwargs.pop('output');result=audit(**kwargs)
    except Exception as error:
        args.output.write_text(json.dumps(dict(version=VERSION,status='failed',auditor_sha256=sha(Path(__file__)),
            original_audit_failure_sha256=sha(args.original_failure),original_v2_failure_sha256=sha(args.original_v2_failure),
            failures=[type(error).__name__+': '+str(error)]),indent=2,allow_nan=False)+'\n')
        raise
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('status','checks_count','requested_trials','native_releases','demonstrated_recipe_coverage')}))


if __name__=='__main__':main()
