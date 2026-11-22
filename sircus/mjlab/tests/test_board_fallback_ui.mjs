import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {runInNewContext} from 'node:vm';
import {boardFallbackSummary} from '../web/interactive-physics.mjs';
const app=readFileSync(new URL('../web/app.js',import.meta.url),'utf8');
const source=`function showBoardFallbackStudy() {${app.split('function showBoardFallbackStudy() {')[1].split('function showBoardPoseRecovery()')[0]}`;
class Element {
  constructor(){this.children=[];this.textContent='';this.hidden=false;}
  append(...items){this.children.push(...items);}
  replaceChildren(...items){this.children=items;}
  get text(){return this.textContent+this.children.map(c=>c.text).join(' ');}
}
function render(report){const nodes=new Map(),$=id=>{if(!nodes.has(id))nodes.set(id,new Element());return nodes.get(id);};runInNewContext(`${source}\nshowBoardFallbackStudy();`,{report,boardFallbackSummary,$,document:{createElement:()=>new Element()}});return $;}
function fixture(){
  const positions=Array.from({length:50},(_,i)=>({id:`fresh-${i}`,bucket_xy_m:[.1+i*.02,-.2],outcomes:{baseline:i===0?'setup_rejected':'direct',fallback:i===1?'flight_miss':i===2?'table':'direct'}}));
  const timing={mean_s:15,median_s:12,p90_s:30,max_s:31};
  return {board_fallback_study:{version:'sircus-board-fallback-study-v1',status:'completed',phase:'completed',audit_status:'passed',physics_substeps:64,
    budgets:{positions:50,planning_s:30},controls:{controls_pre:{status:'passed'},controls_post:{status:'passed'}},
    progress:{prepared:100,planned:100,completed_releases:99,planned_releases:99},fallback:{activated:4,changed_choice:2},
    comparison:{methods:{baseline:{positions:50,admitted:49,captures:49,flight_misses:0,no_candidate:1},fallback:{positions:50,admitted:50,captures:49,flight_misses:1,no_candidate:0}},
      latency:{baseline:timing,fallback:{...timing,mean_s:14}},paired:{both:48,gained:1,lost:1,neither:0},positions},
    next:'Keep the current planner until review.'}};
}
test('fifty paired targets retain no-setup denominator and separate route counts and timing',()=>{
  const report=fixture(),v=boardFallbackSummary(report),$=render(report);assert.equal(v.complete,true);
  assert.match($('boardFallbackTable').text,/49 \/ 50 49 \/ 50/);
  assert.match($('boardFallbackTable').text,/Table-assisted captures 0 1/);
  assert.match($('boardFallbackTable').text,/Mean planning time 15.0 s 14.0 s/);
  assert.match($('boardFallbackPair').text,/gained 1 captures and lost 1/);
  assert.match($('boardFallbackUse').text,/activated at 4 positions/);
  assert.match($('boardFallbackPositions').text,/100.0, -200.0 No checked shot Direct capture/);
  assert.equal(v.promotion_eligible,false);
});
test('pending audit, failed controls, wrong substeps and inconsistent totals withhold final comparison',()=>{
  for(const mutate of [s=>s.audit_status='pending',s=>s.physics_substeps=128,s=>s.controls.controls_post.status='failed',s=>s.comparison.paired.gained=2,s=>s.comparison.positions[1].id='fresh-0',s=>s.comparison.positions[0].outcomes.fallback='predicted_capture',s=>s.progress.completed_releases=98,s=>s.budgets.positions=49]){
    const r=fixture();mutate(r.board_fallback_study);assert.equal(boardFallbackSummary(r).complete,false);assert.equal(render(r)('boardFallbackTable').children.length,0);
  }
});
test('known aiming target is separate from fresh comparison and may have failed confirmations',()=>{
  const r=fixture();r.board_fallback_study.heading_diagnostic={status:'completed',audit_status:'passed',physics_substeps:64,bucket_xy_m:[1.226,.1834],
    original:{released:3,captures:0,arm_assisted:0},search:{released:20,captures:2,arm_assisted:1},repeat:{released:6,captures:5,arm_assisted:0},confirmed_settings:1};
  const $=render(r);assert.match($('boardFallbackDiagnostic').text,/Fresh confirmations captured 5\/6/);assert.match($('boardFallbackDiagnostic').text,/excluded from the 50-position comparison/);
  assert.equal(boardFallbackSummary(r).total,50);
  r.board_fallback_study.heading_diagnostic.audit_status='pending';assert.equal(render(r)('boardFallbackDiagnostic').text,'');
});
test('absent results do not alter historical dashboard or publish scores from predictions',()=>{
  assert.equal(boardFallbackSummary({}),null);assert.equal(render({})('boardFallbackStudy').hidden,true);
  const r=fixture();r.board_fallback_study.status='running';r.board_fallback_study.phase='preparing';delete r.board_fallback_study.comparison;
  assert.equal(render(r)('boardFallbackTable').children.length,0);
  const html=readFileSync(new URL('../web/index.html',import.meta.url),'utf8');
  assert.match(html,/holds both arms at checked endpoints/);assert.match(html,/one 30-second decision limit/);
});
