"""Bounded board-pose fallback inside the existing thirty-second planner.

The hybrid planner runs unchanged until it finishes, or until twelve seconds
have elapsed after publishing a shot with poor predicted arm clearance. The
remaining decision time may search translated/reoriented board poses. Every
choice retains the existing full endpoint and sampled path admission. Native
flight labels never enter this CPU selector; its approximation is not a promise
of capture. Immutable snapshots preserve the best checked prefix if time ends.
"""
from __future__ import annotations

import copy
import math
from types import MethodType
import time

from .fast_planning_plan import ReusablePlanner, CachedRanker, early_eligible
from .planning_budget_plan import _wrap_choice

VERSION = "sircus-board-pose-fallback-v1"
RULE = dict(
    version=VERSION, primary_prefix_s=12., minimum_fallback_s=.25,
    arm_clearance_trigger_m=.005, radial_offsets_m=[-.15,-.10,-.05,0.,.05,.10,.15],
    tangent_offsets_m=[-.15,-.10,-.05,0.,.05,.10,.15],
    heading_offsets_deg=[-16.,-8.,-4.,0.,4.,8.,16.], twist_offsets_deg=[0.,180.],
    tuned_proposals=128, height_delta_m=[-.12,.08], tilt_deg=[12.,44.], gap_m=[.08,.55],
    heading_range_deg=[-30.,30.], full_checks=16, multistart_retry_limit=2,
    selection="Strictly higher finite unchanged flight score; preserve the checked prefix on ties or missing evidence",
    deadline="One absolute decision deadline; initialization, screening, IK, ranking and publication consume it",
    evidence="No native flight labels; only privileged geometry, checked joint poses and fixed flight predictor",
)


def arm_margin(choice):
    prediction = choice.get("prediction") or {}
    route = prediction.get("predicted_route")
    value = (prediction.get(route) or {}).get("arm_clearance_m") if route in ("direct", "table") else None
    return float(value) if type(value) in (int,float) and math.isfinite(value) else None


def needs_board_fallback(choice):
    """Only a checked source shot can anchor this geometry search."""
    margin = arm_margin(choice)
    return choice.get("request") is not None and (margin is None or margin < RULE["arm_clearance_trigger_m"])


def better_prediction(candidate, incumbent):
    """Do not replace a checked prefix for a tie, NaN, or absent score."""
    new = (candidate or {}).get("score_m")
    old = (incumbent or {}).get("score_m")
    if type(new) not in (float,int) or not math.isfinite(new):
        return False
    return type(old) not in (float,int) or not math.isfinite(old) or new > old


def _halton(index, base):
    value, denominator = 0., 1.
    while index:
        denominator *= base
        index, digit = divmod(index,base)
        value += digit/denominator
    return value


def fallback_menu(layout, source, low, high):
    """Same relative menu at every target; no hardcoded recovered recipes."""
    values = source["values"]
    bucket = layout["bucket_xy_m"]
    dx,dy = bucket[0]-values[0],bucket[1]-values[1]
    distance = math.hypot(dx,dy)
    if not math.isfinite(distance) or distance <= .15:
        raise ValueError("A finite checked board-to-bucket displacement is required")
    forward=(dx/distance,dy/distance); side=(-forward[1],forward[0])
    original_heading=float(source.get("board_heading_offset_deg",0.))
    original_twist=float(source.get("board_twist_deg",0.))
    rows=[]
    def key(v,h,t):return tuple(round(float(x),10) for x in (*v,h,t))
    seen={key(values,original_heading,original_twist)}
    def add(v,heading,twist,group):
        twist=math.remainder(twist,360.)
        heading=max(-30.,min(30.,heading))
        if (not all(math.isfinite(x) for x in (*v,heading,twist))
                or any(x<lo-1e-9 or x>hi+1e-9 for x,lo,hi in zip(v,low,high,strict=True))
                or math.hypot(bucket[0]-v[0],bucket[1]-v[1])<=.15):return
        identity=key(v,heading,twist)
        if identity in seen:return
        seen.add(identity)
        rows.append(dict(id=f"{layout['id']}-board-fallback-{len(rows):05d}",
            context_id=layout['id'],bucket_xy_m=list(bucket),values=list(v),
            board_heading_offset_deg=heading,board_twist_deg=twist,
            phase=layout['split'],role=layout['split'],method='board_pose_fallback',
            intended_route=source['intended_route'],group=group,
            seed_q10=list(source['selected_q10']),seed_q14=list(source['selected_q14']),
            source_request_id=source['id']))
    for radial in RULE['radial_offsets_m']:
        for tangent in RULE['tangent_offsets_m']:
            for heading in RULE['heading_offsets_deg']:
                for twist in RULE['twist_offsets_deg']:
                    v=list(values)
                    v[:2]=[v[i]+radial*forward[i]+tangent*side[i] for i in range(2)]
                    add(v,original_heading+heading,original_twist+twist,'position_heading')
    for index in range(1,RULE['tuned_proposals']+1):
        u=[_halton(index,b) for b in (2,3,5,7,11,13,17)]
        v=list(values)
        v[:2]=[v[i]+(.30*u[0]-.15)*forward[i]+(.30*u[1]-.15)*side[i] for i in range(2)]
        v[2]+= -.12+.20*u[2];v[3]=12.+32.*u[3];v[4]=.08+.47*u[4]
        add(v,60.*u[5]-30.,180. if u[6]>=.5 else 0.,'position_height_tilt_gap')
    return rows


class PrefixTimeExpired(RuntimeError):
    """Cooperative handoff; not a failed or exhaustive arm/path certificate."""


class PrefixGuard:
    """Checks the soft boundary only after a risky checked snapshot exists."""
    def __init__(self,started,*,clock=time.perf_counter):
        self.started,self.clock=started,clock
        self.choice=None;self.active=True;self.installed=[]

    def expired(self):
        return self.active and self.choice is not None and needs_board_fallback(self.choice) and self.clock()-self.started>=RULE['primary_prefix_s']

    def check(self):
        if self.expired():raise PrefixTimeExpired('Risky checked prefix reached its twelve-second boundary')

    def install(self,engine):
        gate=engine.adapter.resolver.solver.gate
        if any(old is gate for old,_ in self.installed):return
        original=gate.check;guard=self
        def check(_self,*args,**kwargs):
            guard.check();result=original(*args,**kwargs);guard.check();return result
        gate.check=MethodType(check,gate);self.installed.append((gate,original))

    def close(self):
        self.active=False
        for gate,original in reversed(self.installed):gate.check=original
        self.installed.clear()


def initialize_worker(*args):
    from .stalled_reference_plan import initialize_worker as initialize
    initialize(*args)


def incremental_board_fallback_plan(layout,method,*common_args,publish,deadline,profile):
    from . import parallel_capture_grid_workers as workers
    from . import stalled_reference_plan as hybrid
    from .board_pose_recovery_study import nominal_prediction,wrist_screen,diversified
    from .clearance_expansion_plan import _ranker,needs_expansion
    from .benchmark_verify import digest
    if method!='warm_early':raise ValueError('Board fallback requires the warm_early selector')
    state=dict(version=VERSION,protocol=copy.deepcopy(RULE),triggered=False,trigger=None,
        baseline_choice=None,bounds=None,proposals=[],nominal_predictions={},wrist_screen={},check_order=[],
        records=[],admitted=[],predictions={},selected_fallback_candidate_id=None,completed=False)
    latest=None;guard=PrefixGuard(profile.started)
    guard.install(hybrid._reference_engine)
    if hybrid._direct_engine is not None:guard.install(hybrid._direct_engine)
    def prefix_publish(snapshot):
        nonlocal latest
        frozen=dict(snapshot,board_fallback=copy.deepcopy(state))
        accepted=publish(frozen)
        if accepted:
            latest=copy.deepcopy(snapshot)
            guard.choice=copy.deepcopy(snapshot['choice'])
            guard.install(workers._engine)
            # Return to the hybrid publisher first so its durable publication
            # count includes this snapshot. The next exact gate check can
            # cooperatively end the prefix without erasing that publication.
        return accepted
    interrupted=False
    try:
        reason=hybrid.incremental_stalled_reference_plan(layout,method,*common_args,
            publish=prefix_publish,deadline=deadline,profile=profile)
    except PrefixTimeExpired:
        interrupted=True;reason='completed'
    finally:
        guard.close()
    if latest is None or not needs_board_fallback(latest['choice']):return reason
    if deadline-time.perf_counter()<RULE['minimum_fallback_s']:return reason
    base=copy.deepcopy(latest['choice']);state['baseline_choice']=base
    state['triggered']=True
    state['trigger']=dict(reason='risky_prefix_soft_boundary' if interrupted else 'risky_prefix_finished',
        elapsed_s=time.perf_counter()-profile.started,arm_clearance_m=arm_margin(base),
        baseline_choice_sha256=digest(base),selection_uses_flight_labels=False)
    best=copy.deepcopy(base)
    def snapshot(stage):
        value=copy.deepcopy(latest)
        value.update(active_stage=stage,choice=copy.deepcopy(best),board_fallback=copy.deepcopy(state))
        value['completed_proposal_ids'] += [r['proposal_id'] for r in state['records']]
        return value
    if not publish(snapshot('board_fallback_triggered')):return 'completed'
    engine=workers._engine;ranker_config=common_args[3]
    state['bounds']=dict(low=list(map(float,engine.adapter.resolver.solver.low)),
                        high=list(map(float,engine.adapter.resolver.solver.high)))
    with profile.stage('physics_ranking'):
        ranker=CachedRanker(_ranker(ranker_config))
    with profile.stage('menu'):
        state['proposals']=fallback_menu(layout,base['request'],state['bounds']['low'],state['bounds']['high'])
    eligible=[]
    with profile.stage('physics_ranking'):
        for row in state['proposals']:
            if time.perf_counter()>=deadline:return 'completed'
            screen=wrist_screen(ranker_config['scene'],row,engine.adapter.resolver.models)
            state['wrist_screen'][row['id']]=screen
            if screen['passed']:
                state['nominal_predictions'][row['id']]=nominal_prediction(ranker.ranker,ranker_config['scene'],row)
                eligible.append(row)
    shortlist=diversified(eligible,state['nominal_predictions'],RULE['full_checks'])
    state['check_order']=[r['id'] for r in shortlist]
    if not publish(snapshot('board_fallback_menu')):return 'completed'
    rejected=[]
    def prepare(proposal,multistart=False):
        nonlocal best
        if time.perf_counter()>=deadline:return False
        with profile.stage('arm_setup_checks'):
            added,records=engine.prepare([proposal],one_per_setting=not multistart)
        if engine.adapter.env is not None:
            raise RuntimeError('CPU board-pose search allocated a native environment')
        if time.perf_counter()>=deadline:return False
        with profile.stage('physics_ranking'):
            predictions={r['id']:ranker.predict(r) for r in added}
            for request in added:
                prediction=predictions[request['id']]
                if better_prediction(prediction,best.get('prediction')):
                    best=copy.deepcopy(base)
                    best.update(prediction=prediction,candidate_pool='board_pose_fallback',
                        intended_route=request['intended_route'],
                        available_routes=sorted(set(base.get('available_routes',[]))|{request['intended_route']}),
                        request=copy.deepcopy(request))
                    best=_wrap_choice(best,method)
                    best['predicted_flight_risk']=needs_expansion(best)
                    state['selected_fallback_candidate_id']=request['id']
        if time.perf_counter()>=deadline:return False
        state['records'].extend(records);state['admitted'].extend(added);state['predictions'].update(predictions)
        early=early_eligible(best)
        if early:state['completed']=True
        if not publish(snapshot('early_good_shot' if early else 'board_fallback_check')):return False
        if early:return 'early_good_shot'
        if not added and not multistart:rejected.append(proposal)
        return True
    for row in shortlist:
        status=prepare(row)
        if status=='early_good_shot':return status
        if not status:return 'completed'
    for original in rejected[:RULE['multistart_retry_limit']]:
        proposal=dict(copy.deepcopy(original),id=original['id']+'-multistart',retry_of=original['id'],ik_search='multistart16')
        state['proposals'].append(proposal);state['check_order'].append(proposal['id'])
        status=prepare(proposal,True)
        if status=='early_good_shot':return status
        if not status:return 'completed'
    state['completed']=True
    publish(snapshot('finished'))
    return 'completed'


class BoardFallbackPlanner(ReusablePlanner):
    """Same reusable planner API and absolute deadline as the live hybrid."""
    def __init__(self,method,*args,**kwargs):
        if method!='warm_early':raise ValueError('Board fallback requires warm_early')
        super().__init__(method,*args,**kwargs)
        if self.worker.budget_s<=RULE['primary_prefix_s']:
            self.worker.close();raise ValueError('Total decision budget must exceed the primary prefix boundary')
        self.worker.function=incremental_board_fallback_plan
        self.worker.initializer=initialize_worker

    def plan(self,layout):
        from .stalled_reference_plan import read_receipt
        started=time.perf_counter();result=super().plan(layout)
        result['stalled_reference']=read_receipt(self.directory,result)
        result['request_wall_s']=time.perf_counter()-started
        return result
