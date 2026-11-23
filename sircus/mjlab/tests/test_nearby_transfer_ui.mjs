import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {runInNewContext} from 'node:vm';
import {nearbyShotTransferSummary, physicsPlanners} from '../web/interactive-physics.mjs';

const app = readFileSync(new URL('../web/app.js', import.meta.url), 'utf8');
const renderer = `function showNearbyShotTransfer() {${app.split('function showNearbyShotTransfer() {')[1].split('function showParallelBoardStudy()')[0]}`;
class Element {
  constructor() { this.children=[]; this.textContent=''; this.hidden=false; }
  append(...items) { this.children.push(...items); }
  replaceChildren(...items) { this.children=items; }
  get text() { return this.textContent + this.children.map(child => child.text).join(' '); }
}
function render(report) {
  const nodes=new Map();
  const $=id=>{ if(!nodes.has(id)) nodes.set(id,new Element()); return nodes.get(id); };
  runInNewContext(`${renderer}\nshowNearbyShotTransfer();`,{report,nearbyShotTransferSummary,$,document:{createElement:()=>new Element()}});
  return $;
}
function results(outcomes) {
  return {requested:outcomes.length,completed:outcomes.length,released:outcomes.length,
    captures:outcomes.filter(value=>['direct','table'].includes(value)).length,
    direct:outcomes.filter(value=>value==='direct').length,table:outcomes.filter(value=>value==='table').length,
    robot_contacts:outcomes.filter(value=>value==='arm_assisted_capture').length,outcomes};
}
function fixture() {
  const sources=Array.from({length:23},(_,i)=>({id:`source-${String(i).padStart(2,'0')}`,
    bucket_xy_m:[.90+i*.012,-.10+i*.003],study_key:i<7?'recovery':i===7?'parallel':'guarded',
    strength:i<8?'confirmed_3_of_3':'single_capture',repeat_captures:i<8?3:1,repeat_completed:i<8?3:1,
    report_sha256:'a'.repeat(64),audit_sha256:'b'.repeat(64),values:[.6,.15,.4,25,.12],
    board_heading_offset_deg:0,board_twist_deg:180}));
  const cases=['03','09','13','15','19'].map((suffix,i)=>{
    const id=`parallel-board-miss-${suffix}`,bucket_xy_m=[.902+i*.012,-.098+i*.003],seed=sources[i];
    const distance=Math.hypot(...bucket_xy_m.map((value,k)=>value-seed.bucket_xy_m[k]));
    const successful=i<2;
    const request={id:`${id}-transfer-00`,context_id:id,source_bank_id:seed.id,source_distance_m:distance,
      bucket_xy_m,values:[.6,.15,.4,25,.12],selected_q10:[0,0,0,0,0,0],selected_q14:[0,0,0,0,0,0],
      board_heading_offset_deg:0,board_twist_deg:180};
    const repeated_settings=successful?[{candidate_id:request.id,request,repeat_captures:i===0?3:2,repeat_completed:3}]:[];
    return {id,bucket_xy_m,proposed:50,checked:10,admitted:4,selected:2,nearest_seed_distance_m:distance,
      source_counts:{[seed.id]:2},results:{original:results(['flight_miss','flight_miss','flight_miss']),
        search:results(successful?[i===0?'direct':'table','flight_miss']:i===2?['arm_assisted_capture','flight_miss']:['flight_miss','flight_miss']),
        repeat:results(i===0?['direct','direct','direct']:i===1?['table','table','flight_miss']:[])},
      repeated_settings,confirmed_settings:repeated_settings.filter(row=>row.repeat_captures===3)};
  });
  const control={status:'passed',direct_captures:3,table_assisted_captures:3,negative_captures:0};
  return {nearby_shot_transfer:{version:'sircus-nearby-transfer-study-v1',status:'completed',phase:'completed',
    scope:'development',audit_status:'passed',physics_substeps:64,progress:{completed:31,planned:31},
    initial_requests:25,confirmation_requests:6,native_releases:49,
    controls:{controls_pre:{...control},controls_post:{...control}},cases,
    source_bank_summary:{settings:23,confirmed_settings:8,single_capture_settings:15,sources},
    evidence_manifest_sha256:'c'.repeat(64),published_provenance:{report_sha256:'d'.repeat(64),audit_sha256:'e'.repeat(64),
      evidence_manifest_sha256:'c'.repeat(64),plan_sha256:'f'.repeat(64),source_bank_sha256:'1'.repeat(64),contract_sha256:'2'.repeat(64)},
    next:'Review the repeated settings; keep the current live planner.'}};
}

test('known misses show CPU proposals, native search and fresh confirmations separately',()=>{
  const report=fixture(),before=structuredClone(report),view=nearbyShotTransferSummary(report),$=render(report);
  assert.equal(view.complete,true);assert.equal(view.promotion_eligible,false);assert.equal(view.cases.length,5);
  assert.match($('nearbyTransferResults').text,/\(902\.0, -98\.0\) 0 \/ 3 1 \/ 2 3 \/ 3 1/);
  assert.match($('nearbyTransferResults').text,/\(914\.0, -95\.0\) 0 \/ 3 1 \/ 2 2 \/ 3 0/);
  assert.match($('nearbyTransferSearch').text,/50 10 4 2 2/);
  assert.match($('nearbyTransferOutcome').text,/1 of the five known missed positions/);
  assert.match($('nearbyTransferOutcome').text,/Search found 2 capturing settings; 1 passed all three confirmations/);
  assert.match($('nearbyTransferOutcome').text,/1 arm-assisted search retentions are separate/);
  assert.match($('nearbyTransferConfirmed').text,/400\.0 mm 25\.00° 120\.0 mm 0\.00° 180\.00° 3 \/ 3/);
  assert.match($('nearbyTransferControls').text,/49 total releases, including 18 control drops/);
  assert.deepEqual(report,before);
});

test('source evidence distinguishes confirmed settings from single captures and recomputes distances',()=>{
  const report=fixture(),view=nearbyShotTransferSummary(report),$=render(report);
  assert.equal(view.sources.length,23);assert.equal(view.cases[0].nearest_source_id,'source-00');
  assert.match($('nearbyTransferSources').text,/S01 \(900\.0, -100\.0\) 3 \/ 3 prior repeats Board-position recovery/);
  assert.match($('nearbyTransferSources').text,/One prior capture · not repeated Guarded-search comparison/);
  assert.match($('nearbyTransferDistances').text,/\(902\.0, -98\.0\) S01 2\.8 mm S01/);
  assert.match($('nearbyTransferSourceNote').text,/does not prove success at the new target/);
});

test('final counts require exact accounting, valid source evidence, both controls and independent native64 audit',()=>{
  const mutations=[
    s=>s.status='running',s=>s.phase='auditing',s=>s.audit_status='pending',s=>s.audit_status='failed',
    s=>s.scope='fresh',s=>s.physics_substeps=128,s=>s.controls.controls_post.status='failed',
    s=>delete s.controls.controls_pre,s=>s.controls.controls_pre.negative_captures=1,
    s=>delete s.published_provenance,s=>s.published_provenance.audit_sha256='passed',
    s=>s.published_provenance.source_bank_sha256='',s=>s.published_provenance.plan_sha256='',
    s=>s.published_provenance.contract_sha256='',s=>s.published_provenance.evidence_manifest_sha256='f'.repeat(64),
    s=>s.progress.completed=30,s=>s.progress.planned=32,s=>s.native_releases=48,
    s=>s.initial_requests=24,s=>s.confirmation_requests=5,
    s=>s.cases[1].id=s.cases[0].id,s=>s.cases[1].bucket_xy_m=s.cases[0].bucket_xy_m,
    s=>s.cases.pop(),s=>s.cases[0].results.search.outcomes[0]='predicted_capture',
    s=>s.cases[0].results.search.captures=2,s=>s.cases[0].results.original.requested=4,
    s=>s.cases[0].results.repeat.released=2,s=>s.cases[0].results.repeat.robot_contacts=4,
    s=>s.cases[0].selected=3,s=>s.cases[0].admitted=1,
    s=>s.cases[0].confirmed_settings=[],s=>s.cases[0].repeated_settings[0].repeat_completed=2,
    s=>s.cases[0].repeated_settings[0].repeat_captures=2,
    s=>s.cases[0].repeated_settings[0].request.bucket_xy_m=[0,0],
    s=>s.cases[0].repeated_settings[0].request.context_id='another-target',
    s=>s.cases[0].repeated_settings[0].request.selected_q14=[0],
    s=>s.cases[0].repeated_settings[0].request.values[2]=Infinity,
    s=>s.cases[0].repeated_settings[0].request.source_bank_id='invented-source',
    s=>s.cases[0].repeated_settings[0].request.source_distance_m=0,
    s=>s.cases[0].nearest_seed_distance_m=0,s=>s.cases[0].source_counts={'source-00':1},
    s=>s.cases[0].source_counts={'invented-source':2},s=>s.source_bank_summary.sources[0].report_sha256='unknown',
    s=>s.source_bank_summary.sources[0].repeat_captures=2,s=>s.source_bank_summary.sources[8].repeat_completed=3,
    s=>s.source_bank_summary.sources[0].bucket_xy_m=[0,0],s=>s.source_bank_summary.sources.pop(),
    s=>s.source_bank_summary.confirmed_settings=9,
  ];
  for(const mutate of mutations) {
    const report=fixture();mutate(report.nearby_shot_transfer);const view=nearbyShotTransferSummary(report),$=render(report);
    assert.equal(view.complete,false,String(mutate));assert.deepEqual(view.cases,[],String(mutate));assert.deepEqual(view.sources,[],String(mutate));
    for(const id of ['nearbyTransferResults','nearbyTransferSearch','nearbyTransferSources','nearbyTransferDistances','nearbyTransferConfirmed']) assert.equal($(id).children.length,0,String(mutate));
    for(const id of ['nearbyTransferOutcome','nearbyTransferControls','nearbyTransferSourceNote']) assert.equal($(id).text,'',String(mutate));
    assert.equal($('nearbyTransferSearchDetails').hidden,true);assert.equal($('nearbyTransferConfirmedDetails').hidden,true);
  }
});

test('unknown progress stays unknown and historical scores cannot fill a pending study',()=>{
  const report=fixture();report.parallel_board_study=structuredClone(report.nearby_shot_transfer);
  report.nearby_shot_transfer={version:'sircus-nearby-transfer-study-v1',status:'running',phase:'waiting_for_4090'};
  const $=render(report);assert.match($('nearbyTransferProgress').text,/Recorded development drops: — \/ —/);
  assert.match($('nearbyTransferStatus').text,/waiting for 4090 · final results pending/);
  assert.equal($('nearbyTransferResults').children.length,0);
  assert.equal(nearbyShotTransferSummary({parallel_board_study:fixture().nearby_shot_transfer}),null);
  assert.equal(render({})('nearbyShotTransfer').hidden,true);
});

test('zero admitted settings reports no search drops without inventing confirmations',()=>{
  const report=fixture(),section=report.nearby_shot_transfer;
  for(const row of section.cases) {
    row.admitted=0;row.selected=0;row.source_counts={};row.repeated_settings=[];row.confirmed_settings=[];
    row.results.search=results([]);row.results.repeat=results([]);
  }
  Object.assign(section,{initial_requests:15,confirmation_requests:0,native_releases:33,progress:{completed:15,planned:15}});
  const view=nearbyShotTransferSummary(report),$=render(report);assert.equal(view.complete,true);
  assert.match($('nearbyTransferOutcome').text,/0 of the five known missed positions/);
  assert.match($('nearbyTransferResults').text,/0 \/ 3 0 \/ 0 0 \/ 0 0/);
  assert.equal($('nearbyTransferConfirmedDetails').hidden,true);
});

test('one proposal can admit multiple checked joint configurations without invalidating results',()=>{
  const report=fixture();report.nearby_shot_transfer.cases[0].admitted=18;
  assert.equal(nearbyShotTransferSummary(report).complete,true);
  const $=render(report);assert.match($('nearbyTransferSearch').text,/50 10 18 2 2/);
  assert.match($('nearbyTransferSearch').text,/Admitted arm poses/);
});

test('top panel is read-only and preserves historical panels, media and live planner choices',()=>{
  const html=readFileSync(new URL('../web/index.html',import.meta.url),'utf8');
  for(const id of ['nearbyShotTransfer','parallelBoardStudy','guardedBoardFallback','boardPoseRecovery','hybridMissCorrection'])
    assert.equal((html.match(new RegExp(`id="${id}"`,'g'))||[]).length,1);
  assert.ok(html.indexOf('id="nearbyShotTransfer"')<html.indexOf('id="parallelBoardStudy"'));
  for(const id of ['boardPoseVideoAssets','hybridMissVideoAssets']) assert.match(html,new RegExp(`id="${id}"`));
  const panel=html.split('<section id="nearbyShotTransfer"')[1].split('</section>')[0];
  assert.match(panel,/not an unseen-position evaluation/);assert.match(panel,/first capture is not one of those repeats/);
  assert.match(panel,/No retraining, camera perception, driven-arm validation/);assert.doesNotMatch(panel,/<button|<input|<select/);
  assert.match(app,/nearbyShotTransferSummary = module.nearbyShotTransferSummary/);
  assert.equal((app.match(/    showNearbyShotTransfer\(\);/g)||[]).length,2);
  assert.doesNotMatch(renderer,/innerHTML|fetch\(|physics\.drop|physics\.plan/);
  assert.deepEqual(physicsPlanners({planners:[{id:'hybrid'},{id:'fast'},{id:'nearby_transfer'}],default_planner:'hybrid'}).choices.map(row=>row.id),['hybrid','fast']);
});
