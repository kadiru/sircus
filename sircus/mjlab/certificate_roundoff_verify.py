"""Exact certificate replay with bounded floating-point angular diagnostics.

This is an audit-only comparison. It never changes IK, collision/clearance
acceptance, path checks, physics, scores, targets or stored observations.
"""
from __future__ import annotations
import copy
import math
import numpy as np

VERSION='sircus-endpoint-certificate-roundoff-audit-v1'
ANGLE_TOLERANCE_DEG=2e-6
COSINE_TOLERANCE=8*np.finfo(np.float64).eps
TARGET_TOLERANCE=1e-12
TARGET_FIELDS=('board_world','board_flange_local','dropper_flange_world','release_center_m','board_normal_world')


def compare_certificates(gate,candidate,row,targets,independent,endpoint,checks):
    """Reproduce saved evidence exactly before comparing an independent target.

    Acos amplifies machine roundoff near parallel normals. Away from that case,
    the cosine guard makes the permitted angular difference much smaller than
    the two-microdegree ceiling. Classification and every nondiagnostic field
    must remain exactly equal, including complete strict-clearance evidence.
    """
    label=row['id'];saved=endpoint['fresh_endpoint']
    checks.equal(set(endpoint['targets']),set(TARGET_FIELDS),label+'/roundoff all complete target fields')
    for key in TARGET_FIELDS:
        checks.numeric(endpoint['targets'][key],targets[key],label+'/roundoff independent target/'+key,TARGET_TOLERANCE)
    bound={key:np.asarray(value) for key,value in endpoint['targets'].items()}
    reproduced=gate.check(candidate['q10'],candidate['q14'],row['bucket_xy_m'],row['values'],bound)
    checks.equal(reproduced,saved,label+'/roundoff complete saved-target certificate exactly reproduced')
    first,second=copy.deepcopy(independent),copy.deepcopy(saved)
    a,b=first.pop('board_normal_error_deg'),second.pop('board_normal_error_deg')
    checks.equal(first,second,label+'/roundoff every nondiagnostic field and classification exact')
    cosine_difference=0.
    if a!=b:
        checks.require(np.isfinite([a,b]).all() and 0<=a<=180 and 0<=b<=180,label+'/roundoff finite actual normal angles')
        cosine_difference=abs(math.cos(math.radians(a))-math.cos(math.radians(b)))
        checks.numeric(cosine_difference,0.,label+'/roundoff cosine differs only at floating-point scale',COSINE_TOLERANCE)
        checks.numeric(a,b,label+'/roundoff derived angle within two microdegrees',ANGLE_TOLERANCE_DEG)
    return dict(record_id=label,independent_normal_error_deg=a,saved_normal_error_deg=b,absolute_difference_deg=abs(a-b),
        cosine_difference=cosine_difference,angle_tolerance_deg=ANGLE_TOLERANCE_DEG,cosine_tolerance=COSINE_TOLERANCE,
        target_tolerance=TARGET_TOLERANCE,saved_target_certificate_exact=True,other_independent_fields_exact=True,
        minimum_clearance_lower_bound_m=independent.get('minimum_clearance_lower_bound_m'),reason=independent['reason'])
