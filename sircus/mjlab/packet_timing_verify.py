"""Immutable packet audit with an explicitly authenticated study-specific clock cap.

All request identity/order/hash/phase checks from coverage_training_verify are
retained. The caller must independently authenticate the effective worker cap.
This helper changes no native requests, timestamps, outcomes or admission rules.
"""
from __future__ import annotations
import math
from . import benchmark_verify as common
from .benchmark_verify import contained,read,sha

VERSION='sircus-authenticated-packet-worker-clock-audit-v1'


def audit_request_packets(directory,rows,checks,*,effective_worker_limit_s,reported_elapsed_s):
    checks.require(all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v)
                       for v in (effective_worker_limit_s,reported_elapsed_s)),
                   'Finite independently authenticated worker/report clock')
    checks.require(0<=reported_elapsed_s<effective_worker_limit_s,'Reported completed worker remains within authenticated cap')
    packets=[];seen=set();current=None;group=[]
    for row in rows:
        name=row['request_packet']
        if name!=current:
            if group:packets.append((current,group))
            checks.require(name not in seen,'Every request packet occurs once in chronological sequence')
            seen.add(name);current=name;group=[]
        group.append(row)
    if group:packets.append((current,group))
    timings=common.lines(directory/'timing.jsonl')
    checks.equal(len(timings),len(packets),'Every frozen batch has one execution timing')
    end=0.
    for index,((name,group),timing) in enumerate(zip(packets,timings)):
        checks.equal(name,f'requests/batch-{index:05d}.json','Ordered immutable request batches')
        path=contained(directory,name);packet=read(path)
        checks.require(packet['pre_outcome'] is True and 0<len(group)<=32,'Bounded pre-outcome request batch')
        checks.equal(packet['phase'],group[0]['phase'],'Batch phase binding')
        checks.equal(len(packet['rows']),len(group),'Every issued request accounted, including native declines')
        for request,row in zip(packet['rows'],group):
            checks.equal(row['request_packet_sha256'],sha(path),row['id']+'/pre-outcome request hash')
            for key,value in request.items():checks.equal(row.get(key),value,row['id']+'/request field unchanged/'+key)
        checks.equal(timing['phase'],packet['phase'],'Timing phase matches request batch')
        checks.equal(timing['request_packet'],name,'Timing binds actual request batch')
        checks.equal(timing['request_packet_sha256'],sha(path),'Timing binds exact request bytes')
        checks.require(all(isinstance(timing[k],(int,float)) and not isinstance(timing[k],bool) and math.isfinite(timing[k])
                           for k in ('acquired_start_s','acquired_end_s')),'Finite native batch clock values')
        checks.require(end<=timing['acquired_start_s']<=timing['acquired_end_s']<effective_worker_limit_s,
                       'Monotone native batch timings within authenticated study worker budget')
        checks.require(timing['acquired_end_s']<=reported_elapsed_s,'Native batch ended before completed worker report')
        end=timing['acquired_end_s']
    checks.equal({str(p.relative_to(directory)) for p in (directory/'requests').glob('*.json')},seen,'No missing or extra request packet')
    return dict(batches=len(packets),last_native_end_s=end)
