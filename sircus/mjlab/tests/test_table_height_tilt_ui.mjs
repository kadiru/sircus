import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {VERSION,AUDIT,CASES,HEIGHTS,TILTS,PARENT,studyReady,cellState,validCounts,fallbackInterpretation} from '../web/table-height-tilt.js';
const h=x=>x.repeat(64);
const counts=(attempts=3,captures=0)=>({attempts,native_releases:attempts,captures,setup_rejections:0,native_misses:attempts-captures,robot_contact_shots:0,forbidden_contact_shots:0,invalid_physics:0});
function study(){
 const gate=()=>({status:'passed',counts:counts(9,6),direct_captures:3,table_assisted_captures:3,negative_captures:0});
 const cases=CASES.map((context_id,caseIndex)=>{
  const baseline_values=[.2,0,.4,20,.25],bucket_xy_m=[.8,0];
  const grid=HEIGHTS.flatMap((height,hi)=>TILTS.map((tilt,ti)=>({candidate_id:`${context_id}-${hi*5+ti}`,grid_index:caseIndex*30+hi*5+ti,case_grid_index:hi*5+ti,values:[.2,0,height,tilt,.25],bucket_xy_m:[...bucket_xy_m],admission:{status:'completed',passed:true,reason:'admitted',valid_ik_branches:[1,1],connection_sha256:h('a'),resolution_sha256:h('b')},discovery:{native_release:true,captured:caseIndex===0&&hi===0&&ti===0},selected:caseIndex===0&&hi===0&&ti===0})));
  return {context_id,baseline_values,bucket_xy_m,grid,selection:caseIndex===0?{candidate_id:grid[0].candidate_id,values:[...grid[0].values]}:null,repeats:{baseline:counts(),selected:caseIndex===0?counts(3,3):counts(0)}};
 });
 return {version:VERSION,status:'completed',contract_fingerprint:h('a'),audit:{version:AUDIT,status:'passed',full_checks_verified:true,contract_fingerprint:h('a'),report_sha256:h('b'),proof_sha256:h('c'),checks_count:100},parent_bindings:{...PARENT},admission_matrix_sha256:h('d'),training:false,noise:false,posthoc_development:true,grid_protocol:{heights_m:HEIGHTS,tilts_deg:TILTS,contexts:CASES,total_candidates:120,candidates_per_case:30,substeps:64,nominal_repeats:3},budget:{maximum_requested:162},cases,repeats:{baseline:counts(12),selected:counts(3,3)},control_gates:{pre:gate(),post:gate()}};
}
test('full matrix and separate nominal repeats require matching proof',()=>{const s=study();assert.equal(studyReady(s),true);s.audit.full_checks_verified=false;assert.equal(studyReady(s),false);});
test('wrong source, frozen grid or budget cannot appear verified',()=>{for(const edit of [s=>s.parent_bindings.report_sha256=h('e'),s=>s.grid_protocol.heights_m=[.32,.36,.4,.44,.48,.52],s=>s.budget.maximum_requested=200,s=>s.cases[1].grid.pop(),s=>s.cases[1].grid[0].values[4]=.3]){const s=study();edit(s);assert.equal(studyReady(s),false);}});
test('selected nominal results must follow a discovered capture at exact values',()=>{for(const edit of [s=>s.cases[0].grid[0].discovery.captured=false,s=>s.cases[0].selection.values[2]=.4,s=>s.cases[1].repeats.selected=counts(3),s=>s.repeats.selected.captures=2]){const s=study();edit(s);assert.equal(studyReady(s),false);}});
test('IK rejection, other rejection, admitted-unexecuted and native outcomes differ',()=>{
 const g=study().cases[0].grid[0];assert.equal(cellState(g,false).name,'Not verified');assert.equal(cellState(g,true).name,'Captured');g.discovery.captured=false;assert.equal(cellState(g,true).name,'Released, missed');g.discovery=null;assert.equal(cellState(g,true).name,'Admitted, no flight');g.admission={passed:false,reason:'finite_ik_unresolved'};assert.equal(cellState(g,true).name,'IK unresolved');g.admission.reason='fixed_base_clearance';assert.equal(cellState(g,true).name,'Admission rejected');g.admission={status:'not_attempted'};assert.equal(cellState(g,true).name,'Not attempted');
});
test('admission rejection and unattempted cells have no fabricated flight',()=>{const s=study(),g=s.cases[1].grid[0];g.admission={passed:false,status:'completed',reason:'finite_ik_unresolved'};assert.equal(studyReady(s),false);g.discovery=null;assert.equal(studyReady(s),true);g.admission={status:'not_attempted'};assert.equal(studyReady(s),true);});
test('setup rejects remain in nominal request denominator',()=>{const c=counts();c.native_releases=0;c.native_misses=0;c.setup_rejections=3;assert.equal(validCounts(c),true);c.attempts=0;assert.equal(validCounts(c),false);});
test('control failures and counts cannot pass final gate',()=>{const s=study();s.control_gates.post.negative_captures=1;assert.equal(studyReady(s),false);s.control_gates.post.negative_captures=0;s.control_gates.pre.counts.attempts=8;assert.equal(studyReady(s),false);});
test('panel is separate, read-only and stops final polling',()=>{const code=fs.readFileSync(new URL('../web/table-height-tilt.js',import.meta.url),'utf8');assert.equal(code.includes('/api/solve'),false);assert.equal(code.includes("method:'POST'"),false);assert.ok(code.includes('if(busy||finished)return'));const html=fs.readFileSync(new URL('../web/table-height-tilt.html',import.meta.url),'utf8');assert.ok(html.includes('id="tableHeightTiltStudy"'));assert.equal(html.includes('id="tableTrainingDiagnosis"'),false);assert.ok(html.includes('No selected posture is automatically installed'));});

test('verified fallback reports actual discovery contacts and no-capture repeat decision',()=>{
 const summary=JSON.parse(fs.readFileSync(new URL('../../../baselines/mjlab-table-height-tilt-20261108/summary.json',import.meta.url),'utf8')).study;
 assert.equal(studyReady(summary),true);
 const text=fallbackInterpretation(summary);
 assert.ok(text.includes('0/58 released shots'));
 assert.ok(text.includes('42/58 had recorded robot contact'));
 assert.ok(text.includes('no selected-shot repeats were run'));
 assert.ok(text.includes('do not change the original 50-position policy score'));
 summary.audit.full_checks_verified=false;
 assert.equal(fallbackInterpretation(summary),'Interpretation waits for verified results.');
});
