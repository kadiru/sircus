import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {runInNewContext} from 'node:vm';
import {clearShotRangeSummary,physicsPlanners} from '../web/interactive-physics.mjs';
const app=readFileSync(new URL('../web/app.js',import.meta.url),'utf8');
const renderer=`function showClearShotRange() {${app.split('function showClearShotRange() {')[1].split('function showRemainingLinkClearance()')[0]}`;
class Element{constructor(){this.children=[];this.textContent='';this.hidden=false;}append(...v){this.children.push(...v);}replaceChildren(...v){this.children=v;}get text(){return this.textContent+this.children.map(c=>c.text).join(' ');}}
function render(report){const nodes=new Map(),$=id=>{if(!nodes.has(id))nodes.set(id,new Element());return nodes.get(id);};runInNewContext(`${renderer}\nshowClearShotRange();`,{report,clearShotRangeSummary,$,document:{createElement:()=>new Element()}});return $;}
function result(outcomes,robot=0){return {requested:outcomes.length,completed:outcomes.length,released:outcomes.length,captures:outcomes.filter(v=>['direct','table'].includes(v)).length,direct:outcomes.filter(v=>v==='direct').length,table:outcomes.filter(v=>v==='table').length,robot_contacts:robot,outcomes};}
function fixture(){
 const source_summary={id:'parallel-board-miss-09-clearance-0016-b0',values:[.6,.15,.461868,25.469,.109439],bucket_xy_m:[1.192994,.147423],board_heading_offset_deg:0,board_twist_deg:135,selected_q10:[0,0,0,0,0,0],selected_q14:[0,0,0,0,0,0],prior_no_robot_contact:true};
 const grid=[];for(const tilt of [0,-3,3,-6,6])for(const gap of [0,.02,.04,.06,.08,.10,.13,.16,.20,.24])if(tilt!==0||gap!==0){const i=grid.length;
  grid.push({proposal_id:`proposal-${i}`,tilt_delta_deg:tilt,gap_delta_m:gap,tilt_deg:source_summary.values[3]+tilt,drop_gap_m:source_summary.values[4]+gap,
   setup_status:i<3?'admitted':i%2?'rejected':'not_checked',setup_reason:i<3?'admitted':i%2?'ik_rejected':'case_budget_not_attempted',candidate_id:i<3?`candidate-${i}`:null,
   initial_outcome:i<3?['direct','flight_miss','arm_assisted_capture'][i]:null,initial_capture:i<3?i===0:null,robot_contact:i<3?i>0:null,repeat_completed:i===0?3:0,repeat_captures:i===0?3:0,confirmed:i===0});
 }
 const request={...source_summary,id:'candidate-0',context_id:'parallel-board-miss-09',values:[...source_summary.values.slice(0,3),grid[0].tilt_deg,grid[0].drop_gap_m]};
 const repeated_settings=[{candidate_id:request.id,request,repeat_completed:3,repeat_captures:3}];
 const case_={id:'parallel-board-miss-09',bucket_xy_m:source_summary.bucket_xy_m,proposed:49,selected:3,grid,repeated_settings,confirmed_settings:repeated_settings,
  results:{original:result(['flight_miss','flight_miss','flight_miss']),search:result(['direct','flight_miss','arm_assisted_capture'],2),repeat:result(['direct','direct','direct'])}};
 const control={status:'passed',direct_captures:3,table_assisted_captures:3,negative_captures:0};
 const published_provenance=Object.fromEntries(['report_sha256','audit_sha256','evidence_manifest_sha256','plan_sha256','contract_sha256','source_bank_sha256'].map(key=>[key,'a'.repeat(64)]));
 return {clear_shot_range:{version:'sircus-clear-shot-range-v1',status:'completed',phase:'completed',audit_status:'passed',scope:'development',physics_substeps:64,
  source_summary,cases:[case_],initial_requests:6,confirmation_requests:3,native_releases:36,progress:{completed:9,planned:9},preserved_shots:{requested:9,completed:9,captures:9,passed:true},
  controls:{controls_pre:{...control},controls_post:{...control}},evidence_manifest_sha256:'a'.repeat(64),published_provenance}};
}

test('one target displays actual counts while all49 setup results remain distinct from drops',()=>{
 const report=fixture(),before=structuredClone(report),view=clearShotRangeSummary(report),$=render(report);
 assert.equal(view.complete,true);assert.equal(view.grid.length,49);assert.equal(view.promotion_eligible,false);
 assert.match($('clearRangeSource').text,/height 461\.9 mm · tilt 25\.47° · drop gap 109\.4 mm · board twist 135°/);
 assert.match($('clearRangeResults').text,/New tilt \/ gap settings 1 \/ 3 2 \/ 3/);
 assert.match($('clearRangeOutcome').text,/1 new settings captured; 1 passed all three fresh confirmations/);
 assert.match($('clearRangeGrid').text,/Rejected · ik_rejected No drop — No repeats/);
 assert.match($('clearRangeGrid').text,/Not fully checked · case_budget_not_attempted No drop — No repeats/);
 assert.match($('clearRangeOutcome').text,/other unresolved bucket was not tested/);
 assert.match($('clearRangeChecks').text,/36 total native releases/);
 assert.deepEqual(report,before);
});

test('audit, source, fullgrid, outcome and repeat inconsistencies withhold final data',()=>{
 const changes=[s=>s.audit_status='pending',s=>s.status='running',s=>s.phase='auditing',s=>s.physics_substeps=128,s=>s.scope='fresh',
  s=>s.preserved_shots.captures=8,s=>s.controls.controls_post.negative_captures=1,s=>s.native_releases++,s=>s.progress.completed++,s=>s.initial_requests++,s=>s.confirmation_requests++,
  s=>s.published_provenance.source_bank_sha256='',s=>s.evidence_manifest_sha256='b'.repeat(64),s=>s.source_summary.id='different',s=>s.source_summary.prior_no_robot_contact=false,
  s=>s.source_summary.board_twist_deg=180,s=>s.cases.push(s.cases[0]),s=>s.cases[0].id='parallel-board-miss-19',s=>s.cases[0].grid.pop(),
  s=>s.cases[0].grid[1]={...s.cases[0].grid[0]},s=>s.cases[0].grid[1].drop_gap_m+=.01,s=>s.cases[0].grid[3].initial_outcome='flight_miss',
  s=>s.cases[0].grid[0].robot_contact=null,s=>s.cases[0].grid[0].initial_capture=false,s=>s.cases[0].grid[0].repeat_completed=2,
  s=>s.cases[0].grid[0].confirmed=false,s=>s.cases[0].results.search.robot_contacts=1,s=>s.cases[0].results.search.captures=2,
  s=>s.cases[0].repeated_settings[0].request.selected_q14=[0],s=>s.cases[0].repeated_settings[0].request.values[0]+=.01,
  s=>s.cases[0].confirmed_settings=[],s=>s.cases[0].selected=4];
 for(const change of changes){const report=fixture();change(report.clear_shot_range);const view=clearShotRangeSummary(report),$=render(report);
  assert.equal(view.complete,false,String(change));assert.deepEqual(view.grid,[]);assert.deepEqual(view.confirmed,[]);
  for(const id of ['clearRangeResults','clearRangeGrid','clearRangeSettings'])assert.equal($(id).children.length,0,String(change));
  assert.equal($('clearRangeOutcome').text,'');assert.equal($('clearRangeGridDetails').hidden,true);
 }
});

test('partial and absent data never borrows prior target results',()=>{
 const $=render({remaining_link_clearance:{status:'completed'},clear_shot_range:{version:'sircus-clear-shot-range-v1',status:'running',phase:'waiting_for_4090'}});
 assert.match($('clearRangeProgress').text,/Recorded development drops: — \/ —/);assert.equal($('clearRangeResults').children.length,0);
 assert.equal(clearShotRangeSummary({remaining_link_clearance:fixture().clear_shot_range}),null);assert.equal(render({})('clearShotRange').hidden,true);
});

test('zero captures correctly omit repeat settings without turning setup rejections into flights',()=>{
 const report=fixture(),s=report.clear_shot_range,c=s.cases[0],g=c.grid[0];Object.assign(g,{initial_outcome:'flight_miss',initial_capture:false,confirmed:false,repeat_completed:0,repeat_captures:0});
 c.repeated_settings=[];c.confirmed_settings=[];c.results.search=result(['flight_miss','flight_miss','arm_assisted_capture'],2);c.results.repeat=result([]);
 s.confirmation_requests=0;s.progress={completed:6,planned:6};s.native_releases=33;
 assert.equal(clearShotRangeSummary(report).complete,true);const $=render(report);assert.match($('clearRangeOutcome').text,/0 new settings captured/);assert.equal($('clearRangeSettingsDetails').hidden,true);
});

test('new panel is read-only and preserves prior content and liveplanner choices',()=>{
 const html=readFileSync(new URL('../web/index.html',import.meta.url),'utf8');
 for(const id of ['clearShotRange','remainingLinkClearance','nearbyShotTransfer','boardPoseVideoAssets','hybridMissVideoAssets'])assert.equal((html.match(new RegExp(`id="${id}"`,'g'))||[]).length,1);
 assert.ok(html.indexOf('id="clearShotRange"')<html.indexOf('id="remainingLinkClearance"'));
 const panel=html.split('<section id="clearShotRange"')[1].split('</section>')[0];assert.doesNotMatch(panel,/<button|<input|<select/);assert.match(panel,/not tested in this pass/);
 assert.doesNotMatch(renderer,/innerHTML|fetch\(|physics\.drop|physics\.plan/);assert.equal((app.match(/    showClearShotRange\(\);/g)||[]).length,2);
 assert.match(app,/clearShotRangeSummary = module.clearShotRangeSummary/);
 assert.deepEqual(physicsPlanners({planners:[{id:'hybrid'},{id:'fast'},{id:'clear-range'}],default_planner:'hybrid'}).choices.map(row=>row.id),['hybrid','fast']);
});
