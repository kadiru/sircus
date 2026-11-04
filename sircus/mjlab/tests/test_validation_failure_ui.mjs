import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { VALIDATION_FAILURE_CATEGORIES, validationFailureAuditState, validationFailureCells,
  validationFailureSummary, validationFailureRepresentatives, validationFailureReplayPath,
  validationFailureReplayMatches, validationFailureShouldScroll, validationPoseNumber } from '../web/state.mjs';

const sha = 'a'.repeat(64), replayId = 'b'.repeat(32);
const categories = extra => ({ captured_clean: 1, captured_assisted: 2, setup_rejected: 2,
  invalid_physics: 1, external_disqualified_retention: 0, no_board_bounce: 1,
  entry_not_retained: 1, no_whole_ball_entry: 1, ...extra });
const pose = () => ({ requested: {board_center_m:[.6,-.2,.3],board_normal:[0,.5,.866],release_ball_center_m:[.6,-.2,.6]},
  native_initial: {board_center_m:[.60001,-.2,.3],board_normal:[0,.5,.866],release_ball_center_m:[.6,-.2,.60001]},
  errors:{release_position_mm:.01,flange_error_mm:.02},sampled_cached_board_drift_mm:.03,
  dynamic_settled_pose:'not_recorded',impact_board_orientation:'not_recorded' });
function fixture() {
  const layouts = Array.from({length:12},(_,i)=>({id:`val-${String(i).padStart(2,'0')}`,bucket_xy_m:[.2+i*.07,-.5+i*.04]}));
  const cells = [];
  for (const method of ['sweep','ppo']) for (const milestone of ['selected',0,256,512,1024]) for (const layout of layouts)
    cells.push({...structuredClone(layout),method,milestone,attempts:9,captures:3,native_releases:7,categories:categories(),
      miss_regions:{short:1,long:1,unknown:2},rejection_reasons:{finite_ik_unresolved:2},selected_samples_by_seed:{42:256,43:1024,44:1024},pose_audit:{dynamic_settled_pose:'not_recorded'},representative_ids:[replayId]});
  const representative = { id:replayId,replay_id:replayId,method:'ppo',seed:42,samples:256,context_id:layouts[0].id,
    category:'entry_not_retained',miss_region:{region:'short',reason:'observed_crossing'},retention_status:'horizon_censored',
    observed_events:{board_contact_observed:true,accepted_rebound_sample_time_s:.1,recording_horizon_reached:true},primary_selected:true,recorded_replay_available:true,
    path:`replays/${replayId}.json`,sha256:sha,pose_audit:pose(),source_row_id:'ppo-seed42-validation-record-000'};
  return { benchmark_read_only:true,sweep_rl_benchmark:{version:'sircus-calibrated-sweep-rl-benchmark-v2',status:'completed',
    contract_fingerprint:sha,test_disclosure:'verified',audit:{status:'passed',synthetic_fixture:false}},
    validation_failure_audit:{version:'sircus-calibrated-validation-failure-audit-v1',status:'passed',synthetic_fixture:false,fingerprint:sha,
      source:{contract_fingerprint:sha,report_sha256:sha,audit_sha256:sha,study_manifest_sha256:sha,
        source_sha256:{'native.py':sha},auditor_sha256:sha,auditor_dependencies:{'oracle.py':sha}},layouts,
      cohorts:{selected:{attempts:216},milestones:[0,256,512,1024].map(samples=>({samples,attempts:216}))},cells,
      representatives:[representative,{...structuredClone(representative),id:'c'.repeat(32),replay_id:'c'.repeat(32),path:`replays/${'c'.repeat(32)}.json`,
        samples:0,primary_selected:false},
      {...structuredClone(representative),id:'d'.repeat(32),replay_id:'d'.repeat(32),category:'setup_rejected',pose_audit:{requested:pose().requested,native_initial:null},
        path:`replays/${'d'.repeat(32)}.json`}] } };
}

test('readiness fails closed for missing, unaudited, synthetic and stale source evidence',()=>{
  assert.equal(validationFailureAuditState({}).present,false);assert.equal(validationFailureAuditState({}).ready,false);
  assert.equal(validationFailureAuditState({validation_failure_audit_error:'bad source'}).present,true);
  assert.equal(validationFailureAuditState(fixture()).ready,true);
  for (const change of [r=>r.validation_failure_audit.status='pending',r=>r.validation_failure_audit.synthetic_fixture=true,
    r=>delete r.validation_failure_audit.fingerprint,r=>delete r.validation_failure_audit.source.audit_sha256,
    r=>r.validation_failure_audit.source.contract_fingerprint='f'.repeat(64),r=>r.sweep_rl_benchmark.audit.status='pending',
    r=>r.sweep_rl_benchmark.audit.synthetic_fixture=true,r=>r.sweep_rl_benchmark.status='running',
    r=>delete r.sweep_rl_benchmark.audit,r=>r.validation_failure_audit.layouts.pop(),
    r=>r.validation_failure_audit.layouts[1]=structuredClone(r.validation_failure_audit.layouts[0])]) {
    const r=fixture();change(r);assert.equal(validationFailureAuditState(r).ready,false);
    assert.deepEqual(validationFailureCells(r),[]);assert.deepEqual(validationFailureRepresentatives(r),[]);
  }
});

test('audit hash scroll waits for readiness and happens only once instead of on polling',()=>{
  assert.equal(validationFailureShouldScroll({},'#validationFailureAudit',false),false);
  const report=fixture();assert.equal(validationFailureShouldScroll(report,'#validationFailureAudit',false),true);
  assert.equal(validationFailureShouldScroll(report,'#validationFailureAudit',true),false);
  assert.equal(validationFailureShouldScroll(report,'#sweepRlBenchmark',false),false);
  report.validation_failure_audit.status='pending';assert.equal(validationFailureShouldScroll(report,'#validationFailureAudit',false),false);
  const js=readFileSync(new URL('../web/app.js',import.meta.url),'utf8');
  assert.equal(js.split('validationAuditScrolled = true').length-1,1);
  assert.ok(js.includes('validationFailureShouldScroll(report, location.hash, validationAuditScrolled)'));
  assert.ok(js.includes("validationView ? 'Sircus · Validation failure audit'"));
});

test('capture tier wording stays marginal-aware and tiny pose errors remain distinguishable from zero/unknown',()=>{
  assert.equal(VALIDATION_FAILURE_CATEGORIES.captured_assisted,'Assisted or marginal capture');
  assert.equal(validationPoseNumber(.000042),'0.000042');
  assert.equal(validationPoseNumber(0),'0.000000');
  assert.equal(validationPoseNumber(undefined),'Unknown');assert.equal(validationPoseNumber(null),'Unknown');
  assert.equal(validationPoseNumber(NaN),'Unknown');
});

test('selected primary and all checkpoint slices stay separate repeated validation cohorts',()=>{
  const r=fixture(), original=JSON.stringify(r), selected=validationFailureSummary(r,'ppo','selected');
  assert.equal(selected.layouts,12);assert.equal(selected.reportedLayouts,12);
  assert.equal(selected.attempts,108);assert.equal(selected.captures,36);assert.equal(selected.nativeReleases,84);
  assert.equal(selected.primaryRequests,216);assert.equal(selected.milestoneRequests,864);
  for (const n of [0,256,512,1024]) assert.equal(validationFailureSummary(r,'sweep',n).attempts,108);
  assert.deepEqual(validationFailureCells(r,'ppo','unsupported'),[]);
  assert.deepEqual(validationFailureCells(r,'other','selected'),[]);
  assert.equal(JSON.stringify(r),original);
});

test('missing, duplicated, malformed and wrong-position cells are unknown instead of observed zeros',()=>{
  for (const change of [r=>r.cells.splice(r.cells.findIndex(c=>c.method==='ppo'&&c.milestone==='selected'),1),
    r=>r.cells.push(structuredClone(r.cells.find(c=>c.method==='ppo'&&c.milestone==='selected'))),
    r=>delete r.cells.find(c=>c.method==='ppo'&&c.milestone==='selected').categories.invalid_physics,
    r=>r.cells.find(c=>c.method==='ppo'&&c.milestone==='selected').bucket_xy_m[0]+=.001,
    r=>r.cells.find(c=>c.method==='ppo'&&c.milestone==='selected').native_releases=9]) {
    const r=fixture();change(r.validation_failure_audit);
    const rows=validationFailureCells(r);assert.equal(rows.length,12);assert.equal(rows[0].status,'unknown');assert.equal(rows[0].counts,null);
    assert.equal(validationFailureSummary(r).attempts,null);assert.equal(validationFailureSummary(r).captures,null);
  }
});

test('setup rejection stays in capture denominator and cannot masquerade as an observed trajectory',()=>{
  const r=fixture(), c=r.validation_failure_audit.cells.find(c=>c.method==='ppo'&&c.milestone==='selected');
  c.captures=0;c.native_releases=0;c.categories=Object.fromEntries(Object.keys(VALIDATION_FAILURE_CATEGORIES).map(k=>[k,k==='setup_rejected'?9:0]));
  const row=validationFailureCells(r)[0];assert.equal(row.status,'rejected');assert.equal(row.counts.attempts,9);
  assert.equal(row.counts.captures,0);assert.equal(row.counts.setupRejected,9);assert.equal(row.counts.rate,0);
  const rejected=validationFailureRepresentatives(r).find(c=>c.category==='setup_rejected');
  assert.equal(rejected.recordedReplayAvailable,false);assert.equal(rejected.resultUrl,null);assert.equal(rejected.poseAudit.native_initial,null);
});

test('representatives use exact method/checkpoint metadata without transferring captures or inventing poses',()=>{
  const r=fixture(), original=JSON.stringify(r), cards=validationFailureRepresentatives(r);
  assert.equal(cards.length,2);assert.equal(cards[0].replayId,replayId);assert.equal(cards[0].resultUrl,`/api/validation-failures/replay/${replayId}`);
  assert.equal(cards[0].poseAudit.dynamic_settled_pose,'not_recorded');assert.equal(cards[0].poseAudit.impact_board_orientation,'not_recorded');
  assert.equal(cards[0].missRegion,'short');assert.equal(cards[0].retentionStatus,'horizon_censored');
  assert.equal(cards[0].observedEvents.board_contact_observed,true);
  assert.equal(validationFailureRepresentatives(r,'ppo',0).length,1);
  assert.equal(validationFailureRepresentatives(r,'sweep','selected').length,0);
  r.validation_failure_audit.representatives[0].category='captured_clean';assert.equal(validationFailureRepresentatives(r).length,1);
  assert.notEqual(JSON.stringify(r),original);
});

test('replay endpoints accept only opaque same-origin IDs and require referenced hashes',()=>{
  assert.equal(validationFailureReplayPath(replayId),`/api/validation-failures/replay/${replayId}`);
  for (const id of ['','../secret','id/next','id?x=1','https://other/shot','%2e%2e','id#x','x'.repeat(161),null]) assert.equal(validationFailureReplayPath(id),null);
  for (const change of [v=>v.path='https://other/shot',v=>delete v.sha256,v=>v.recorded_replay_available=false]) {
    const r=fixture();change(r.validation_failure_audit.representatives[0]);assert.equal(validationFailureRepresentatives(r)[0].resultUrl,null);
  }
});

function replay(card) {return {request_id:card.sourceRowId,method:'validation_recorded_replay',success:false,bucket_xy_m:card.xy,
  configuration:{board_center:[.6,-.2,.3],board_normal:[0,0,1],release_center:[.6,-.2,.6]},
  trajectory:[{time:0,ball:[.6,-.2,.6]},{time:.01,ball:[.6,-.2,.599]}]};}
test('recorded replay checks exact row/bucket identity and ordered observed trajectory before viewer update',()=>{
  const card=validationFailureRepresentatives(fixture())[0];assert.equal(validationFailureReplayMatches(replay(card),card),true);
  for (const change of [r=>r.request_id='different',r=>r.method='search',r=>r.bucket_xy_m=[0,0],r=>r.success=true,
    r=>r.trajectory[0].time=.1,r=>r.trajectory[1].time=0,r=>r.trajectory[1].ball=[NaN,0,0],r=>r.configuration.board_center=null]) {
    const r=replay(card);change(r);assert.equal(validationFailureReplayMatches(r,card),false);
  }
});

test('panel preserves benchmark/history gates and exposes only recorded GET replays with honest pose/cohort scope',()=>{
  const html=readFileSync(new URL('../web/index.html',import.meta.url),'utf8'), js=readFileSync(new URL('../web/app.js',import.meta.url),'utf8');
  for (const id of ['validationFailureAudit','validationFailureMethod','validationFailureCheckpoint','validationFailureMap','validationFailureCellCounts','validationFailureCards','validationFailureReplayStatus'])
    assert.equal(html.split(`id="${id}"`).length-1,1);
  assert.ok(html.includes('Selected checkpoints · primary'));assert.ok(html.includes('already used for checkpoint selection'));
  assert.ok(html.includes('not fresh or unbiased policy evaluation'));assert.ok(html.includes('No new shots are collected'));
  assert.ok(html.includes('Settled and impact board poses were not recorded'));assert.ok(html.includes('Ball-center release error and flange error are separate'));
  assert.ok(html.includes('ball was spawned at p0'));assert.ok(html.includes('TEST outcomes are excluded'));
  const block=js.slice(js.indexOf('function validationCellDetail'),js.indexOf('const benchmarkLegacyNotes ='));
  assert.ok(block.includes("method: 'GET'"));assert.ok(!block.includes('POST')&&!block.includes('/api/solve'));
  assert.ok(block.indexOf('validationFailureReplayMatches(result, card)')<block.indexOf('applyResult(result,'));
  assert.ok(block.includes('generation !== validationReplayGeneration'));assert.ok(block.includes('audit?.fingerprint !== fingerprint'));
  assert.ok(block.includes('never add them together'));assert.ok(block.includes('No native release / trajectory'));
});
