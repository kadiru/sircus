import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {impactPoseSweepState,impactPoseSweepLayouts,impactPoseSweepSummary,impactPoseSweepCells,
  impactPoseReplayPath,impactPoseReplayMatches,impactPoseHistory,solveButtonState} from '../web/state.mjs';
const sha='a'.repeat(64),other='b'.repeat(64);
function fixture() {
  const layouts=[];
  for(const [index,xy] of [[3,[.9,-.4]],[5,[1.0,-.6]]])for(const method of ['ppo','sweep']) {
    const id=`validation-${index}-${method}`,baseline=[.7,-.3,.4,24,.3],trials=[];
    for(const [j,tilt] of [20,24,28].entries())for(const [k,gap] of [.3,.36,.42,.48].entries()) {
      const rejected=j===0&&k===3;
      trials.push({id:`${id}-g${j}${k}`,phase:'sweep',baseline:j===1&&k===0,values:[...baseline.slice(0,3),tilt,gap],
        native_release:!rejected,success:j===1&&k===2,invalid:rejected?'strict_endpoint_failed':null,
        replay_id:`shot-${index}-${method}-${j}${k}`,replay_sha256:sha,
        pose_summary:rejected?null:{status:'verified',native_joint_hz:15360,fk_uniform_hz:240,maximum_postintegration_board_drift_mm:.001}});
    }
    const selected=trials[6];
    for(let r=0;r<3;r++)trials.push({...structuredClone(selected),id:`${id}-confirmation-${r}`,phase:'confirmation',success:r<2,baseline:false,
      replay_id:`repeat-${index}-${method}-${r}`});
    layouts.push({id,context_id:`validation-${index}`,source_method:method,bucket_xy_m:xy,source_row_id:`${method}-seed42-validation-${index}-r0`,
      baseline_values:baseline,trials,selected:{id:selected.id,values:selected.values},confirmation:{attempts:3,native_releases:3,captures:2}});
  }
  return {benchmark_read_only:true,sweep_rl_benchmark:{version:'sircus-calibrated-sweep-rl-benchmark-v2',contract_fingerprint:sha},
    impact_pose_sweep:{version:'sircus-impact-pose-sweep-v1',status:'completed',synthetic_fixture:false,fingerprint:sha,audit:{status:'passed'},
      source:{benchmark_report_sha256:sha,validation_report_sha256:sha,contract_fingerprint:sha},protocol:{ball_radius_m:.0375},layouts}};
}
test('completed native sweep requires a passed source-bound audit; missing or running evidence remains pending',()=>{
  assert.equal(impactPoseSweepState({}).present,false);
  const r=fixture();assert.equal(impactPoseSweepState(r).ready,true);
  for(const change of [r=>r.impact_pose_sweep.status='running',r=>r.impact_pose_sweep.synthetic_fixture=true,r=>r.impact_pose_sweep.audit.status='pending',
    r=>delete r.impact_pose_sweep.fingerprint,r=>r.impact_pose_sweep.source.contract_fingerprint=other,
    r=>delete r.impact_pose_sweep.source.validation_report_sha256,r=>r.impact_pose_sweep.layouts.push(structuredClone(r.impact_pose_sweep.layouts[0])),
    r=>r.impact_pose_sweep_error='pinned source mismatch']) {
    const value=fixture();change(value);assert.equal(impactPoseSweepState(value).ready,false);assert.deepEqual(impactPoseSweepLayouts(value),[]);
    assert.equal(impactPoseSweepSummary(value),null);
  }
});
test('two bucket coordinates and four source postures stay distinct; all rejects remain in requested counts',()=>{
  const r=fixture(),before=JSON.stringify(r),summary=impactPoseSweepSummary(r);
  assert.equal(summary.positions,2);assert.equal(summary.postures,4);
  assert.deepEqual(summary.sweep,{attempts:48,nativeReleases:44,captures:4,setupRejected:4});
  assert.deepEqual(summary.confirmation,{attempts:12,nativeReleases:12,captures:8,setupRejected:0});
  assert.deepEqual(summary.counts,{attempts:60,nativeReleases:56,captures:12,setupRejected:4});
  assert.equal(JSON.stringify(r),before);
  assert.equal(solveButtonState(r,false).disabled,true);
});
test('selected repeat cohorts cannot inherit search successes or confirmation of a different setting',()=>{
  const r=fixture(),layout=impactPoseSweepLayouts(r)[0];
  assert.equal(layout.sweepCounts.captures,1);assert.equal(layout.confirmationCounts.captures,2);
  r.impact_pose_sweep.layouts[0].trials.at(-1).values[4]+=.01;
  assert.equal(impactPoseSweepLayouts(r)[0].confirmationCounts,null);assert.equal(impactPoseSweepSummary(r).confirmation,null);
  const value=fixture();value.impact_pose_sweep.layouts[0].confirmation.attempts=2;
  assert.equal(impactPoseSweepLayouts(value)[0].confirmationCounts,null);
  const missing=fixture();delete missing.impact_pose_sweep.layouts[0].selected;
  assert.equal(impactPoseSweepLayouts(missing)[0].confirmationCounts,null);
});
test('exact requested height and tilt grid retains a rejected point and missing squares are unknown',()=>{
  const r=fixture(),layout=impactPoseSweepLayouts(r)[0],cells=impactPoseSweepCells(layout);
  assert.equal(cells.length,12);assert.equal(cells.filter(c=>c.status==='captured').length,1);
  assert.equal(cells.filter(c=>c.status==='rejected').length,1);assert.equal(cells.filter(c=>c.trial?.baseline).length,1);
  assert.equal(layout.selected.release[2],.4+.42+.0375);
  assert.deepEqual(layout.selected.board,[.7,-.3,.4]);
  r.impact_pose_sweep.layouts[0].trials.splice(0,1);
  const hole=impactPoseSweepCells(impactPoseSweepLayouts(r)[0]);assert.equal(hole.length,12);assert.equal(hole.filter(c=>c.status==='unknown').length,1);
});
test('malformed counts, duplicate trials or impossible captured rejections do not become zeros',()=>{
  for(const change of [l=>delete l.trials[0].native_release,l=>l.trials.push(structuredClone(l.trials[0])),
    l=>l.trials[3].success=true,l=>l.trials[0].values[2]=NaN,l=>delete l.trials[0].success]) {
    const r=fixture();change(r.impact_pose_sweep.layouts[0]);
    const row=impactPoseSweepLayouts(r)[0];assert.equal(row.valid,false);assert.equal(row.counts,null);assert.equal(impactPoseSweepSummary(r).counts,null);
  }
});
test('only opaque same-origin hash-referenced native replay links can be opened',()=>{
  const r=fixture(),rows=impactPoseSweepLayouts(r)[0].trials;
  assert.ok(rows[0].resultUrl.startsWith('/api/impact-sweep/replay/'));assert.equal(rows[3].resultUrl,null);
  for(const id of ['',null,'../secret','shot-0000?next=x','https://other/shot','%2e%2e','x'.repeat(100),'shot/next'])assert.equal(impactPoseReplayPath(id),null);
  delete r.impact_pose_sweep.layouts[0].trials[0].replay_sha256;
  assert.equal(impactPoseSweepLayouts(r)[0].trials[0].resultUrl,null);
});
const history=()=>[0,.1,.2].map(time=>({time,board_center:[.7,-.3,.4],board_quaternion_wxyz:[1,0,0,0],
  link_positions10:[[0,0,0],[.7,-.3,.8]],link_positions14:[[1.33731,0,0],[.7,-.3,.4]]}));
function replay(trial) {return {method:'impact_sweep_recorded_replay',request_id:trial.id,bucket_xy_m:trial.xy,success:trial.success,
  configuration:{board_center:trial.board,board_normal:[0,.5,.866],release_center:trial.release},
  trajectory:[{time:0,ball:trial.release},{time:.2,ball:[.7,-.3,.6]}],pose_history:history()};}
test('recorded replay must retain exact trial, bucket and outcome and ordered actual pose samples',()=>{
  const trial=impactPoseSweepLayouts(fixture())[0].trials[0];assert.equal(impactPoseReplayMatches(replay(trial),trial),true);
  for(const change of [r=>r.request_id='other',r=>r.method='search',r=>r.success=!r.success,r=>r.bucket_xy_m=[0,0],
    r=>r.trajectory[0].time=.01,r=>r.trajectory[1].ball=[NaN,0,0],r=>r.pose_history[1].time=-1,
    r=>r.pose_history[1].board_center=null,r=>r.pose_history[1].link_positions14=[]]) {
    const r=replay(trial);change(r);assert.equal(impactPoseReplayMatches(r,trial),false);
  }
  const noHistory=replay(trial);delete noHistory.pose_history;assert.equal(impactPoseReplayMatches(noHistory,trial),true);
  assert.equal(impactPoseHistory(undefined),false);assert.equal(impactPoseHistory(history(),.15),false);
});
test('new panel is read-only, labels timing and spawned-ball limitations, and does not fabricate arm motion',()=>{
  const html=readFileSync(new URL('../web/index.html',import.meta.url),'utf8'),js=readFileSync(new URL('../web/app.js',import.meta.url),'utf8');
  for(const id of ['impactPoseSweep','impactPoseStatus','impactPoseLayout','impactPoseHeatmap','impactPoseConfirmation','impactPosePoseChecks','impactPoseReplay'])
    assert.equal(html.split(`id="${id}"`).length-1,1);
  for(const text of ['previously observed short-miss validation bucket positions','only vertical drop gap and board tilt vary','Completed benchmark results and checkpoints stay frozen',
    'not a capture-rate estimate','Contact observations belong to the preceding native interval','prescribed endpoints before every step','spawns the ball in free flight'])assert.ok(html.includes(text));
  const block=js.slice(js.indexOf('function impactPoseOutcome'),js.indexOf('const benchmarkLegacyNotes ='));
  assert.ok(block.includes("method:'GET'"));assert.ok(!block.includes('POST')&&!block.includes('/api/solve'));
  assert.ok(block.indexOf('impactPoseReplayMatches(result,trial)')<block.indexOf('applyResult(result,'));
  assert.ok(js.includes('currentPoseHistory = impactPoseHistory(result.pose_history'));
  assert.ok(js.includes('const pose = currentPoseHistory[k]'));assert.ok(js.includes('drawArms(pose)'));
});

test('short, long and unknown observed crossings remain distinct from contact-altered geometry and scored proximity',()=>{
  const r=fixture(),raw=r.impact_pose_sweep.layouts[0];
  raw.source_category='no_whole_ball_entry';raw.source_miss_region={region:'short'};
  for(const [index,region] of ['short','long','unknown'].entries())Object.assign(raw.trials[index],{
    category:'no_whole_ball_entry',miss_region:{region,geometric_region:index===2?'long':region,reason:index===2?'intervening_contact':'observed_crossing'},
    closest_opening_distance_m:.025+index*.01});
  const layout=impactPoseSweepLayouts(r)[0];
  assert.deepEqual(layout.trials.slice(0,3).map(t=>t.missRegion),['short','long','unknown']);
  assert.equal(layout.trials[2].geometricMissRegion,'long');assert.equal(layout.trials[2].missRegionReason,'intervening_contact');
  assert.equal(layout.trials[0].closestOpeningDistanceMm,25);assert.equal(layout.trials[2].closestOpeningDistanceMm,45);
  assert.equal(layout.trials[3].closestOpeningDistanceMm,null);
  assert.equal(layout.sourceCategory,'no_whole_ball_entry');assert.equal(layout.sourceMissRegion,'short');
  assert.equal(layout.baseline.id,layout.sweep.find(t=>t.baseline).id);
  delete raw.trials[0].miss_region;delete raw.trials[0].closest_opening_distance_m;
  const missing=impactPoseSweepLayouts(r)[0].trials[0];assert.equal(missing.missRegion,'unknown');assert.equal(missing.geometricMissRegion,null);
  assert.equal(missing.closestOpeningDistanceMm,null);
});
