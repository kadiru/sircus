import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {runInNewContext} from 'node:vm';
import {guardedBoardFallbackSummary, physicsPlanners} from '../web/interactive-physics.mjs';

const app = readFileSync(new URL('../web/app.js', import.meta.url), 'utf8');
const source = `function showGuardedBoardFallback() {${app.split('function showGuardedBoardFallback() {')[1].split('function showBoardFallbackStudy()')[0]}`;
class Element {
  constructor() { this.children=[]; this.textContent=''; this.hidden=false; }
  append(...items) { this.children.push(...items); }
  replaceChildren(...items) { this.children=items; }
  get text() { return this.textContent + this.children.map(child => child.text).join(' '); }
}
function render(report) {
  const nodes = new Map();
  const $ = id => { if (!nodes.has(id)) nodes.set(id, new Element()); return nodes.get(id); };
  runInNewContext(`${source}\nshowGuardedBoardFallback();`,
    {report, guardedBoardFallbackSummary, $, document:{createElement:()=>new Element()}});
  return $;
}
function fixture() {
  const positions = Array.from({length:20}, (_,i) => ({id:`guarded-fresh-${i}`, bucket_xy_m:[.1+i*.025,-.25],
    outcomes:{baseline:i===0?'setup_rejected':'direct', guarded:i===1?'flight_miss':i===2?'table':'direct'}}));
  const timing = {mean_s:15,median_s:12,p90_s:30,max_s:31};
  const control = {status:'passed',direct_captures:3,table_assisted_captures:3,negative_captures:0};
  return {guarded_board_fallback_study:{version:'sircus-guarded-board-fallback-study-v1',status:'completed',phase:'completed',
    audit_status:'passed',physics_substeps:64,budgets:{positions:20,planning_s:30},
    controls:{controls_pre:{...control},controls_post:{...control}},
    progress:{prepared:40,planned:40,completed_releases:39,planned_releases:39},fallback:{activated:4,changed_choice:2},
    comparison:{methods:{baseline:{positions:20,admitted:19,captures:19,flight_misses:0,no_candidate:1},
      guarded:{positions:20,admitted:20,captures:19,flight_misses:1,no_candidate:0}},
      latency:{baseline:timing,guarded:{...timing,mean_s:14}}, paired:{both:18,gained:1,lost:1,neither:0},positions},
    diagnostic:{bucket_xy_m:[.948252196166334,-.09781574379730379],methods:{
      baseline:{requested:3,released:3,captures:3,outcomes:['direct','direct','direct'],checked_shot:true},
      guarded:{requested:3,released:3,captures:2,outcomes:['table','table','flight_miss'],checked_shot:true}}},
    next:'Keep the current planner until review.'}};
}

test('guarded result keeps twenty fresh denominators and separates three-repeat known-target diagnosis', () => {
  const report=fixture(), before=structuredClone(report), view=guardedBoardFallbackSummary(report), $=render(report);
  assert.equal(view.complete,true); assert.equal(view.total,20); assert.equal(view.promotion_eligible,false);
  assert.match($('guardedBoardFallbackTable').text,/First-drop captures 19 \/ 20 19 \/ 20/);
  assert.match($('guardedBoardFallbackTable').text,/Table-assisted captures 0 1/);
  assert.match($('guardedBoardFallbackTable').text,/Checked shots released 19 20/);
  assert.match($('guardedBoardFallbackTable').text,/No checked shot 1 0/);
  assert.match($('guardedBoardFallbackTable').text,/Mean full planning request 15.0 s 14.0 s/);
  assert.match($('guardedBoardFallbackPair').text,/gained 1 captures and lost 1/);
  assert.match($('guardedBoardFallbackUse').text,/activated at 4 positions and changed the selected shot at 2/);
  assert.match($('guardedBoardFallbackDiagnostic').text,/Captures across three requested repeats 3 \/ 3 2 \/ 3/);
  assert.match($('guardedBoardFallbackDiagnosticNote').text,/excluded from the 20 fresh-position results/);
  assert.match($('guardedBoardFallbackPositions').text,/100.0, -250.0 No checked shot Direct capture/);
  assert.equal($('guardedBoardFallbackDetails').hidden,false);
  assert.deepEqual(report,before);
});

test('incomplete, unaudited, uncalibrated or inconsistent accounting never exposes final scores', () => {
  const mutations = [
    s=>s.status='running', s=>s.phase='preparing', s=>s.audit_status='pending', s=>s.physics_substeps=128,
    s=>s.budgets.positions=50, s=>s.budgets.planning_s=31,
    s=>s.controls.controls_post.status='failed', s=>s.controls.controls_pre.negative_captures=1,
    s=>delete s.controls.controls_post, s=>s.progress.completed_releases=38,
    s=>s.progress.prepared=39, s=>s.comparison.paired.gained=2,
    s=>s.comparison.methods.guarded.captures=20,
    s=>s.comparison.positions[1].id=s.comparison.positions[0].id,
    s=>s.comparison.positions[1].bucket_xy_m=s.comparison.positions[0].bucket_xy_m,
    s=>s.comparison.positions[0].outcomes.guarded='predicted_capture',
    s=>s.comparison.latency.guarded.mean_s=NaN, s=>s.comparison.latency.guarded.p90_s=9,
    s=>s.fallback.activated=21, s=>s.fallback.changed_choice=5,
    s=>delete s.diagnostic, s=>s.diagnostic.bucket_xy_m=s.comparison.positions[0].bucket_xy_m,
    s=>s.diagnostic.methods.guarded.requested=4,
    s=>s.diagnostic.methods.guarded.released=2, s=>s.diagnostic.methods.guarded.captures=3,
    s=>s.diagnostic.methods.guarded.outcomes[0]='setup_rejected',
    s=>s.diagnostic.methods.guarded.checked_shot=false,
  ];
  for (const mutate of mutations) {
    const report=fixture(); mutate(report.guarded_board_fallback_study);
    const view=guardedBoardFallbackSummary(report), $=render(report);
    assert.equal(view.complete,false, String(mutate));
    for (const id of ['guardedBoardFallbackTable','guardedBoardFallbackPositions','guardedBoardFallbackDiagnostic']) {
      assert.equal($(id).children.length,0,String(mutate));
    }
    assert.equal($('guardedBoardFallbackPair').text,'');
    assert.equal($('guardedBoardFallbackDetails').hidden,true);
  }
});

test('a diagnostic setup rejection stays zero of three requested repeats without changing fresh results', () => {
  const report=fixture();
  report.guarded_board_fallback_study.diagnostic.methods.guarded={requested:3,released:0,captures:0,outcomes:[],checked_shot:false};
  const view=guardedBoardFallbackSummary(report), $=render(report);
  assert.equal(view.complete,true); assert.equal(view.methods[1].captures,19);
  assert.match($('guardedBoardFallbackDiagnostic').text,/Captures across three requested repeats 3 \/ 3 0 \/ 3/);
  assert.match($('guardedBoardFallbackDiagnostic').text,/Actual drops 3 0/);
});

test('arm-assisted diagnostic retention is explicit and does not become a qualifying capture', () => {
  const report=fixture(), row=report.guarded_board_fallback_study.diagnostic.methods.guarded;
  row.outcomes=['direct','arm_assisted_capture','flight_miss']; row.captures=1;
  const view=guardedBoardFallbackSummary(report), $=render(report);
  assert.equal(view.complete,true);
  assert.match($('guardedBoardFallbackDiagnostic').text,/Arm-assisted retentions · separate 0 1/);
  assert.equal(view.diagnostic.methods[1].captures,1);
});

test('pending progress uses dashes for unknown counts and never imports historical results', () => {
  const report=fixture();
  report.board_fallback_study={comparison:report.guarded_board_fallback_study.comparison};
  report.guarded_board_fallback_study={version:'sircus-guarded-board-fallback-study-v1',status:'running',phase:'waiting_for_4090',
    audit_status:'pending',physics_substeps:64,budgets:{positions:20,planning_s:30},progress:{prepared:40,planned:40}};
  const $=render(report);
  assert.match($('guardedBoardFallbackStatus').text,/waiting for 4090 · final scores pending/);
  assert.match($('guardedBoardFallbackProgress').text,/Fresh-position drops: — \/ —/);
  assert.equal($('guardedBoardFallbackTable').children.length,0);
  assert.equal(guardedBoardFallbackSummary({board_fallback_study:{}}),null);
  assert.equal(render({})('guardedBoardFallback').hidden,true);
});

test('new panel preserves historical sections, does not alter planner options and describes guarded rules', () => {
  const html=readFileSync(new URL('../web/index.html',import.meta.url),'utf8');
  for (const id of ['guardedBoardFallback','boardFallbackStudy','boardPoseRecovery','hybridMissCorrection','stalledReference']) {
    assert.equal((html.match(new RegExp(`id="${id}"`,'g'))||[]).length,1);
  }
  for (const id of ['boardPoseVideoAssets','hybridMissVideoAssets']) assert.match(html,new RegExp(`id="${id}"`));
  assert.match(html,/finishes its search without early interruption/);
  assert.match(html,/predict at least 5 mm of ball-to-arm clearance/);
  assert.match(html,/better predicted flight score/);
  assert.match(html,/The live planner is unchanged/);
  assert.match(app,/guardedBoardFallbackSummary = module.guardedBoardFallbackSummary/);
  assert.equal((app.match(/    showGuardedBoardFallback\(\);/g)||[]).length,2);
  assert.equal(physicsPlanners({planners:[{id:'hybrid'},{id:'fast'}],default_planner:'hybrid'}).defaultPlanner,'hybrid');
  assert.doesNotMatch(source,/innerHTML|fetch\(|physics\.drop|physics\.plan/);
});
