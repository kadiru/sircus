import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {runInNewContext} from 'node:vm';
import {parallelBoardSummary, physicsPlanners} from '../web/interactive-physics.mjs';

const app = readFileSync(new URL('../web/app.js', import.meta.url), 'utf8');
const source = `function showParallelBoardStudy() {${app.split('function showParallelBoardStudy() {')[1].split('function showGuardedBoardFallback()')[0]}`;
class Element {
  constructor() { this.children=[]; this.textContent=''; this.hidden=false; }
  append(...items) { this.children.push(...items); }
  replaceChildren(...items) { this.children=items; }
  get text() { return this.textContent + this.children.map(child => child.text).join(' '); }
}
function render(report) {
  const nodes=new Map();
  const $=id=>{ if(!nodes.has(id)) nodes.set(id,new Element()); return nodes.get(id); };
  runInNewContext(`${source}\nshowParallelBoardStudy();`,{report,parallelBoardSummary,$,document:{createElement:()=>new Element()}});
  return $;
}
function fixture() {
  const captures=new Set(['direct','table','unclassified_capture']);
  const positions=Array.from({length:6},(_,i)=>{
    const labels={baseline:i===0?['flight_miss','flight_miss','flight_miss']:i===1?[]:['direct','direct','direct'],
      parallel:i===1?['flight_miss','flight_miss','flight_miss']:i===2?['table','table','table']:
        i===3?['arm_assisted_capture','direct','flight_miss']:['direct','direct','direct']};
    const methods=Object.fromEntries(Object.entries(labels).map(([key,outcomes])=>[key,{requested:3,
      checked_shot:outcomes.length===3,released:outcomes.length,captures:outcomes.filter(v=>captures.has(v)).length,
      outcomes,request_wall_s:12+i-(key==='parallel'?2:0),decision_latency_s:11+i-(key==='parallel'?2:0)}]));
    return {id:`parallel-dev-${i}`,source_id:`previous-study-${i}`,bucket_xy_m:[.9+i*.05,-.1+i*.03],
      role:i===5?'known_regression':'previous_miss',methods,
      parallel_lane:{triggered:i<2,changed_choice:i===0,checked_candidates:i<2?8:0,admitted_candidates:i<2?3:0}};
  });
  const methods={},latency={};
  for(const id of ['baseline','parallel']) {
    const rows=positions.map(row=>row.methods[id]),times=rows.map(row=>row.request_wall_s),sum=times.reduce((a,b)=>a+b,0);
    methods[id]={positions:6,requested:18,checked_positions:rows.filter(row=>row.checked_shot).length,
      releases:rows.reduce((n,row)=>n+row.released,0),captures:rows.reduce((n,row)=>n+row.captures,0),
      reliable_positions:rows.filter(row=>row.captures===3).length};
    latency[id]={positions:6,mean_s:sum/6,median_s:(times[2]+times[3])/2,p90_s:times[5],max_s:times[5],aggregate_s:sum};
  }
  const control={status:'passed',direct_captures:3,table_assisted_captures:3,negative_captures:0};
  return {parallel_board_study:{version:'sircus-parallel-board-study-v1',scope:'development',status:'completed',phase:'completed',
    audit_status:'passed',physics_substeps:64,budgets:{positions:6,repeats:3,planning_s:30},
    progress:{prepared:12,planned:12,completed_releases:33,planned_releases:33},parallel:{activated:2,changed_choice:1},
    controls:{controls_pre:{...control},controls_post:{...control}},comparison:{positions,methods,latency},
    evidence_manifest_sha256:'a'.repeat(64),published_provenance:{report_sha256:'b'.repeat(64),audit_sha256:'c'.repeat(64),evidence_manifest_sha256:'a'.repeat(64)},
    next:'Keep the current planner; review selected development targets first.'}};
}

test('six selected targets show matched exact-command repeats, planning requests, and lane activity',()=>{
  const report=fixture(),before=structuredClone(report),view=parallelBoardSummary(report),$=render(report);
  assert.equal(view.complete,true); assert.equal(view.promotion_eligible,false); assert.equal(view.positions.length,6);
  assert.match($('parallelBoardPositions').text,/0 \/ 3 3 \/ 3 12.00 s 10.00 s/);
  assert.match($('parallelBoardPositions').text,/0 \/ 3 · no checked shot/);
  assert.match($('parallelBoardControl').text,/3 \/ 3 3 \/ 3 17.00 s 15.00 s/);
  assert.match($('parallelBoardTotals').text,/Captures across requested repeats 12 \/ 18 13 \/ 18/);
  assert.match($('parallelBoardTotals').text,/Actual drops 15 18/);
  assert.match($('parallelBoardTotals').text,/Mean full planning request 14.50 s 12.50 s/);
  assert.match($('parallelBoardTotals').text,/Arm-assisted retentions · separate 0 1/);
  assert.match($('parallelBoardActivity').text,/activated at 2 of six targets and changed the selected shot at 1/);
  assert.match($('parallelBoardActivity').text,/1 of the five recovery targets captured on all three parallel repeats/);
  assert.match($('parallelBoardActivity').text,/one planning request, not three repeated planning runs/);
  assert.equal($('parallelBoardLaneDetails').hidden,false);
  assert.deepEqual(report,before);
});

test('complete results require passed controls, audit provenance, native64 and exact development accounting',()=>{
  const mutations=[
    s=>s.status='running',s=>s.phase='auditing',s=>s.audit_status='pending',s=>s.scope='fresh',s=>s.physics_substeps=128,
    s=>s.budgets.positions=20,s=>s.budgets.repeats=2,s=>s.budgets.planning_s=60,
    s=>s.controls.controls_post.status='failed',s=>s.controls.controls_pre.negative_captures=1,
    s=>delete s.published_provenance,s=>s.published_provenance.audit_sha256='passed',
    s=>s.published_provenance.evidence_manifest_sha256='d'.repeat(64),
    s=>s.progress.completed_releases=32,s=>s.progress.prepared=11,s=>s.progress.planned=13,
    s=>s.comparison.positions[1].id=s.comparison.positions[0].id,
    s=>s.comparison.positions[1].source_id=s.comparison.positions[0].source_id,
    s=>s.comparison.positions[1].bucket_xy_m=s.comparison.positions[0].bucket_xy_m,
    s=>s.comparison.positions[0].role='known_regression',
    s=>s.comparison.positions[0].methods.parallel.outcomes[0]='predicted_capture',
    s=>s.comparison.positions[0].methods.parallel.outcomes[0]='setup_rejected',
    s=>s.comparison.positions[0].methods.parallel.requested=4,
    s=>s.comparison.positions[0].methods.parallel.checked_shot=false,
    s=>s.comparison.positions[0].methods.parallel.released=2,
    s=>s.comparison.positions[0].methods.parallel.captures=2,
    s=>s.comparison.positions[0].methods.parallel.request_wall_s=Infinity,
    s=>s.comparison.positions[0].methods.parallel.decision_latency_s=31,
    s=>s.comparison.positions[0].parallel_lane.admitted_candidates=9,
    s=>s.comparison.positions[0].parallel_lane.triggered=false,
    s=>s.parallel.activated=3,s=>s.parallel.changed_choice=2,
    s=>s.comparison.methods.parallel.captures=14,s=>s.comparison.methods.parallel.releases=17,
    s=>s.comparison.methods.parallel.reliable_positions=6,s=>s.comparison.methods.parallel.checked_positions=5,
    s=>s.comparison.latency.parallel.mean_s=12,s=>s.comparison.latency.parallel.p90_s=14,
    s=>s.comparison.latency.parallel.positions=18,
  ];
  for(const mutate of mutations) {
    const report=fixture();mutate(report.parallel_board_study); const view=parallelBoardSummary(report),$=render(report);
    assert.equal(view.complete,false,String(mutate));assert.deepEqual(view.positions,[],String(mutate));
    for(const id of ['parallelBoardPositions','parallelBoardControl','parallelBoardTotals','parallelBoardLanes']) assert.equal($(id).children.length,0,String(mutate));
    assert.equal($('parallelBoardActivity').text,'');assert.equal($('parallelBoardLaneDetails').hidden,true);
  }
});

test('pending results keep unknown counts as dashes and never borrow an earlier study',()=>{
  const report=fixture();report.guarded_board_fallback_study=report.parallel_board_study;
  report.parallel_board_study={version:'sircus-parallel-board-study-v1',scope:'development',status:'running',phase:'waiting_for_4090',progress:{prepared:12,planned:12}};
  const $=render(report);assert.match($('parallelBoardProgress').text,/Development drops: — \/ —/);
  assert.match($('parallelBoardStatus').text,/waiting for 4090 · final results pending/);
  assert.equal($('parallelBoardTotals').children.length,0);
  assert.equal(parallelBoardSummary({guarded_board_fallback_study:{}}),null);
  assert.equal(render({})('parallelBoardStudy').hidden,true);
});

test('a completely declined method preserves requested denominator with no fictitious drops',()=>{
  const report=fixture(),s=report.parallel_board_study;
  for(const row of s.comparison.positions) {
    Object.assign(row.methods.parallel,{checked_shot:false,released:0,captures:0,outcomes:[]});
    row.parallel_lane.changed_choice=false;
  }
  Object.assign(s.comparison.methods.parallel,{checked_positions:0,releases:0,captures:0,reliable_positions:0});
  s.parallel.changed_choice=0;s.progress.completed_releases=s.progress.planned_releases=15;
  assert.equal(parallelBoardSummary(report).complete,true);
  const $=render(report);assert.match($('parallelBoardTotals').text,/Captures across requested repeats 12 \/ 18 0 \/ 18/);
  assert.match($('parallelBoardTotals').text,/Actual drops 15 0/);
});

test('top panel preserves earlier panels/assets and cannot change live planner choices or run drops',()=>{
  const html=readFileSync(new URL('../web/index.html',import.meta.url),'utf8');
  for(const id of ['parallelBoardStudy','guardedBoardFallback','boardFallbackStudy','boardPoseRecovery','hybridMissCorrection','stalledReference']) {
    assert.equal((html.match(new RegExp(`id="${id}"`,'g'))||[]).length,1);
  }
  assert.ok(html.indexOf('id="parallelBoardStudy"')<html.indexOf('id="guardedBoardFallback"'));
  for(const id of ['boardPoseVideoAssets','hybridMissVideoAssets']) assert.match(html,new RegExp(`id="${id}"`));
  assert.match(html,/not new evaluation positions/);assert.match(html,/full 30-second decision allowance/);
  assert.match(html,/at least 5 mm of ball-to-arm clearance/);assert.match(html,/improve the finite predicted flight score/);
  assert.match(app,/parallelBoardSummary = module.parallelBoardSummary/);
  assert.equal((app.match(/    showParallelBoardStudy\(\);/g)||[]).length,2);
  assert.equal(physicsPlanners({planners:[{id:'hybrid'},{id:'fast'},{id:'parallel'}],default_planner:'hybrid'}).choices.length,2);
  assert.doesNotMatch(source,/innerHTML|fetch\(|physics\.drop|physics\.plan/);
});
