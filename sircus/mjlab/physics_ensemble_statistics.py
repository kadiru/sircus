"""Post-hoc descriptive statistics; no model fit, simulation or proof claims."""
import itertools
import math

import numpy as np

VERSION='sircus-physics-ensemble-descriptive-statistics-v1'
METHODS=('bc','ppo','ensemble')
SEED=20261651
RESAMPLES=10000
SCOPE=('Post-hoc descriptive intervals from original audited outcomes; single policy '
       'study/one nominal shot per layout, not seed/noise reliability')


def _wilson(successes,total):
    z=1.959963984540054;p=successes/total;den=1+z*z/total
    center=(p+z*z/(2*total))/den
    half=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/den
    return [0. if successes==0 else max(0.,center-half),1. if successes==total else min(1.,center+half)]


def _mcnemar(a_only,b_only):
    discordant=a_only+b_only
    if not discordant:return 1.
    return min(1.,2*sum(math.comb(discordant,k) for k in range(min(a_only,b_only)+1))/(2**discordant))


def describe(test_positions):
    """Require the full paired180-slot cohort; retain every invalid/reject."""
    if not isinstance(test_positions,list) or len(test_positions)!=180:
        raise ValueError('Exactly180 paired TEST requests required')
    groups={method:{} for method in METHODS};coordinates={};records=set()
    for row in test_positions:
        if not isinstance(row,dict) or row.get('method') not in groups:
            raise ValueError('Only the three declared TEST methods are allowed')
        method=row['method'];context=row.get('context_id')
        if not isinstance(context,str) or not context or context in groups[method]:
            raise ValueError('Unique nonempty context per method required')
        try:xy=np.asarray(row.get('bucket_xy_m'),dtype=float)
        except (TypeError,ValueError) as error:raise ValueError('Finite paired bucket coordinates required') from error
        if xy.shape!=(2,) or not np.isfinite(xy).all():raise ValueError('Finite paired bucket coordinates required')
        xy=tuple(xy.tolist())
        if context in coordinates and coordinates[context]!=xy:raise ValueError('Paired bucket coordinates differ')
        coordinates[context]=xy
        if 'record_id' in row:
            identity=row['record_id']
            if not isinstance(identity,str) or not identity or identity in records:raise ValueError('Duplicate or invalid observed record ID')
            records.add(identity)
        actual=row.get('actual')
        flags=('native_release','capture','clean','invalid_physics')
        if not isinstance(actual,dict) or any(type(actual.get(k)) is not bool for k in flags):
            raise ValueError('Explicit observed nested booleans required')
        if 'invalid' not in actual or actual['invalid'] is not None and not isinstance(actual['invalid'],str):
            raise ValueError('Explicit observed invalid reason or None required')
        native,capture,clean,bad=(actual[k] for k in flags);invalid=bool(actual['invalid'])
        if capture and (not native or invalid or bad) or clean and not capture:
            raise ValueError('Observed capture/clean flags contradict release validity')
        if not native and (not invalid or bad) or native and invalid!=bad:
            raise ValueError('Rejects and native physics failures must remain distinguishable')
        groups[method][context]=actual
    contexts=sorted(groups[METHODS[0]])
    if len(contexts)!=60 or any(set(group)!=set(contexts) for group in groups.values()):
        raise ValueError('Exactly60 shared TEST layouts for all methods required')
    if len(set(coordinates.values()))!=60:raise ValueError('Distinct bucket positions required for all60 layouts')
    matrix=np.asarray([[groups[m][c]['capture'] for m in METHODS] for c in contexts],dtype=float)
    indices=np.random.default_rng(SEED).integers(0,60,size=(RESAMPLES,60))
    bootstrap=matrix[indices].mean(axis=1)
    comparison=[]
    for j,method in enumerate(METHODS):
        rows=list(groups[method].values());native=sum(r['native_release'] for r in rows)
        captures=int(matrix[:,j].sum());invalid=sum(r['native_release'] and (bool(r['invalid']) or r['invalid_physics']) for r in rows)
        comparison.append(dict(method=method,counts=dict(requested=60,native_releases=native,captures=captures,
            clean=sum(r['clean'] for r in rows),rejected=60-native,native_misses=native-captures-invalid,invalid=invalid),
            capture_ci95=_wilson(captures,60),ci95_method='Wilson score95%; denominator60 requested layouts'))
    paired=[]
    for ia,ib in itertools.combinations(range(3),2):
        a,b=matrix[:,ia].astype(bool),matrix[:,ib].astype(bool)
        a_only=int((a&~b).sum());b_only=int((b&~a).sum())
        paired.append(dict(a=METHODS[ia],b=METHODS[ib],a_only=a_only,b_only=b_only,
            shared=int((a&b).sum()),neither=int((~a&~b).sum()),delta_rate=float((a.astype(float)-b).mean()),
            ci95=np.percentile(bootstrap[:,ia]-bootstrap[:,ib],[2.5,97.5]).tolist(),
            interval_method='Paired-layout percentile95%;10000 shared resamples; seed20261651; delta=a-b',
            exact_mcnemar_p=_mcnemar(a_only,b_only)))
    return dict(version=VERSION,scope=SCOPE,test_comparison=comparison,paired_differences=paired,same_layouts=60)
