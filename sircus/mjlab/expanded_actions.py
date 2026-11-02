"""Explicit action-range versions; never relax native IK/contact/scoring gates."""

import numpy as np
from .learning_contract import LOW, HIGH, validate_locks

VERSION = "sircus-expanded-shot-actions-v1"
OLD_BOUNDS = dict(low=LOW.tolist(), high=HIGH.tolist(), version="original-shot-bounds-v1")
WIDE_BOUNDS = dict(low=[.30,-.50,.12,8.,.05], high=[1.04,.35,.60,60.,.65], version=VERSION)


def checked_bounds(bounds=None):
    bounds = OLD_BOUNDS if bounds is None else bounds
    lo,hi=np.asarray(bounds["low"],dtype=float),np.asarray(bounds["high"],dtype=float)
    if lo.shape!=(5,) or hi.shape!=(5,) or not np.isfinite([lo,hi]).all() or not (hi>lo).all():
        raise ValueError("Expected five finite ordered action bounds")
    if lo[2]<=0 or lo[3]<0 or hi[3]>=90 or lo[4]<=0:
        raise ValueError("Board above table, positive gap and tilt in [0,90) required")
    return lo,hi


def project(values,locks,radius=.0375,bounds=None):
    lo,hi=checked_bounds(bounds)
    a=np.asarray(values,dtype=float).copy()
    if a.ndim!=2 or a.shape[1]!=5:
        raise ValueError("Expected shot actions [N,5]")
    validate_locks(locks,len(a))
    b,d=np.asarray(locks["board"]),np.asarray(locks["dropper"])
    bm,dm=np.isfinite(b).all(1),np.isfinite(d).all(1)
    a[bm,:3]=b[bm]
    a[dm&~bm,:2]=d[dm&~bm,:2]
    a[dm,4]=d[dm,2]-a[dm,2]-radius
    conflict=bm&dm&(np.linalg.norm(b[:,:2]-d[:,:2],axis=1)>1e-9)
    valid=np.isfinite(a).all(1)&~conflict
    valid&=(a>=lo-1e-9).all(1)&(a<=hi+1e-9).all(1)
    return a,valid
