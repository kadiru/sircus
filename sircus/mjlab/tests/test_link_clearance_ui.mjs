import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {runInNewContext} from 'node:vm';
import {remainingLinkClearanceSummary,physicsPlanners} from '../web/interactive-physics.mjs';
const app=readFileSync(new URL('../web/app.js',import.meta.url),'utf8');
const renderer=`function showRemainingLinkClearance() {${app.split('function showRemainingLinkClearance() {')[1].split('function showNearbyShotTransfer()')[0]}`;
class Element {
  constructor(){this.children=[];this.textContent='';this.hidden=false;}
  append(...values){this.children.push(...values);}
  replaceChildren(...values){this.children=values;}
  get text(){return this.textContent+this.children.map(child=>child.text).join(' ');}
}
function render(report){
  const nodes=new Map(),$=id=>{if(!nodes.has(id))nodes.set(id,new Element());return nodes.get(id);};
  runInNewContext(`${renderer}\nshowRemainingLinkClearance();`,{report,remainingLinkClearanceSummary,$,document:{createElement:()=>new Element()}});
  return $;
}
function result(outcomes,robot=0){return {requested:outcomes.length,completed:outcomes.length,released:outcomes.length,
  captures:outcomes.filter(value=>['direct','table'].includes(value)).length,direct:outcomes.filter(value=>value==='direct').length,
  table:outcomes.filter(value=>value==='table').length,robot_contacts:robot,outcomes};}
function fixture(){
  const cases=['09','19'].map((suffix,index)=>{
    const id=`parallel-board-miss-${suffix}`,bucket_xy_m=index?[1.127555,-.038348]:[1.192994,.147423];
    const request={id:`${id}-alternative-01`,context_id:id,bucket_xy_m,values:[.6,.15,.4,25,.12],
      selected_q10:[0,0,0,0,0,0],selected_q14:[0,0,0,0,0,0],board_heading_offset_deg:0,board_twist_deg:180};
    const repeated_settings=index?[]:[{candidate_id:request.id,request,repeat_captures:3,repeat_completed:3}];
    return {id,bucket_xy_m,proposed:40,checked:20,admitted:5,selected:2,
      results:{original:result(['flight_miss','flight_miss','flight_miss'],3),search:result(index?['flight_miss','arm_assisted_capture']:['direct','flight_miss'],index?2:1),repeat:result(index?[]:['direct','direct','direct'])},
      repeated_settings,confirmed_settings:repeated_settings};
  });
  const diagnosis_rows=cases.map(row=>({id:row.id,bucket_xy_m:row.bucket_xy_m,trials:20,robot_contacts:20,
    first_contact_link_counts:{'Arm14/link3/collision/0':20},first_contact_time_s:{min:.4,max:.6},method:'native_force_cpu_reconstruction'}));
  const control={status:'passed',direct_captures:3,table_assisted_captures:3,negative_captures:0};
  const published_provenance=Object.fromEntries(['report_sha256','audit_sha256','evidence_manifest_sha256','plan_sha256','source_bank_sha256','contract_sha256','diagnosis_sha256'].map(key=>[key,'a'.repeat(64)]));
  return {remaining_link_clearance:{version:'sircus-link-clearance-study-v1',status:'completed',phase:'completed',scope:'development',audit_status:'passed',physics_substeps:64,
    progress:{completed:13,planned:13},initial_requests:10,confirmation_requests:3,native_releases:40,cases,diagnosis_rows,
    preserved_shots:{requested:9,completed:9,captures:9,passed:true},controls:{controls_pre:{...control},controls_post:{...control}},
    evidence_manifest_sha256:'a'.repeat(64),published_provenance,next:'Keep successful shots and review the remaining blocker.'}};
}

test('two known targets separate measured contacts, reconstructed links, captures and confirmations',()=>{
  const report=fixture(),before=structuredClone(report),view=remainingLinkClearanceSummary(report),$=render(report);
  assert.equal(view.complete,true);assert.equal(view.promotion_eligible,false);assert.equal(view.cases.length,2);
  assert.match($('remainingLinkDiagnosis').text,/\(1193\.0, 147\.4\) 20 \/ 20 Arm14\/link3\/collision\/0 · 20 flights 0\.400–0\.600 s/);
  assert.match($('remainingLinkDiagnosisNote').text,/link name is inferred/);
  assert.match($('remainingLinkResults').text,/0 \/ 3 1 \/ 2 1 \/ 2 3 \/ 3 1/);
  assert.match($('remainingLinkResults').text,/0 \/ 3 0 \/ 2 2 \/ 2 0 \/ 0 0/);
  assert.match($('remainingLinkOutcome').text,/1 of the two remaining positions/);
  assert.match($('remainingLinkOutcome').text,/1 arm-assisted search retentions are saved separately/);
  assert.match($('remainingLinkSettings').text,/400\.0 mm 25\.00° 120\.0 mm/);
  assert.match($('remainingLinkPreservation').text,/All nine captures passed/);
  assert.match($('remainingLinkControls').text,/40 total native releases/);
  assert.deepEqual(report,before);
});

test('uncertified, inconsistent or misbound final outcomes are never displayed as captures',()=>{
  const mutations=[s=>s.status='running',s=>s.audit_status='pending',s=>s.phase='auditing',s=>s.physics_substeps=128,s=>s.scope='fresh',
    s=>s.controls.controls_post.status='failed',s=>s.controls.controls_pre.negative_captures=1,s=>s.preserved_shots.captures=8,s=>s.preserved_shots.passed=false,
    s=>s.progress.planned++,s=>s.native_releases++,s=>s.initial_requests++,s=>s.confirmation_requests++,
    s=>delete s.published_provenance.diagnosis_sha256,s=>s.published_provenance.evidence_manifest_sha256='b'.repeat(64),
    s=>s.cases.reverse(),s=>s.cases.pop(),s=>s.cases[0].results.search.captures=2,s=>s.cases[0].results.search.robot_contacts=3,
    s=>s.cases[0].results.original.requested=2,s=>s.cases[0].selected=33,s=>s.cases[0].confirmed_settings=[],
    s=>s.cases[0].repeated_settings[0].request.selected_q14=[0],s=>s.cases[0].repeated_settings[0].request.bucket_xy_m=[0,0],
    s=>s.cases[0].repeated_settings[0].request.context_id='wrong',s=>s.cases[0].repeated_settings[0].request.values[2]=Infinity,
    s=>s.diagnosis_rows[0].first_contact_link_counts['Arm14/link3/collision/0']=19,s=>s.diagnosis_rows[0].method='sensor_link_label',
    s=>s.diagnosis_rows[0].first_contact_time_s.max=.2,s=>s.diagnosis_rows[0].bucket_xy_m=[0,0],s=>s.diagnosis_rows.pop()];
  for(const mutate of mutations){const report=fixture();mutate(report.remaining_link_clearance);const view=remainingLinkClearanceSummary(report),$=render(report);
    assert.equal(view.complete,false,String(mutate));assert.deepEqual(view.cases,[]);assert.deepEqual(view.diagnosis,[]);
    for(const id of ['remainingLinkDiagnosis','remainingLinkResults','remainingLinkChecks','remainingLinkSettings'])assert.equal($(id).children.length,0,String(mutate));
    for(const id of ['remainingLinkOutcome','remainingLinkPreservation','remainingLinkControls'])assert.equal($(id).text,'',String(mutate));
    assert.equal($('remainingLinkSettingDetails').hidden,true);
  }
});

test('queued progress remains unknown and cannot borrow the preceding study result',()=>{
  const report={nearby_shot_transfer:{status:'completed'},remaining_link_clearance:{version:'sircus-link-clearance-study-v1',status:'running',phase:'waiting_for_4090'}};
  const $=render(report);assert.match($('remainingLinkProgress').text,/Recorded development drops: — \/ —/);
  assert.match($('remainingLinkStatus').text,/waiting for 4090/);assert.equal($('remainingLinkResults').children.length,0);
  assert.equal(remainingLinkClearanceSummary({nearby_shot_transfer:fixture().remaining_link_clearance}),null);
  assert.equal(render({})('remainingLinkClearance').hidden,true);
});

test('valid zero-capture result displays zero and no invented repeat settings',()=>{
  const report=fixture(),s=report.remaining_link_clearance;
  s.cases[0].results.search=result(['flight_miss','flight_miss'],2);s.cases[0].results.repeat=result([]);
  s.cases[0].repeated_settings=[];s.cases[0].confirmed_settings=[];s.confirmation_requests=0;s.progress={completed:10,planned:10};s.native_releases=37;
  const $=render(report);assert.equal(remainingLinkClearanceSummary(report).complete,true);
  assert.match($('remainingLinkOutcome').text,/0 of the two remaining positions/);assert.equal($('remainingLinkSettingDetails').hidden,true);
});

test('new panel is read-only and retains all earlier panels, media and live planner behavior',()=>{
  const html=readFileSync(new URL('../web/index.html',import.meta.url),'utf8');
  for(const id of ['remainingLinkClearance','nearbyShotTransfer','parallelBoardStudy','boardPoseVideoAssets','hybridMissVideoAssets'])assert.equal((html.match(new RegExp(`id="${id}"`,'g'))||[]).length,1);
  assert.ok(html.indexOf('id="remainingLinkClearance"')<html.indexOf('id="nearbyShotTransfer"'));
  const panel=html.split('<section id="remainingLinkClearance"')[1].split('</section>')[0];
  assert.doesNotMatch(panel,/<button|<input|<select/);assert.match(panel,/not new evaluation targets/);assert.match(panel,/arms held at checked endpoints/);
  assert.doesNotMatch(renderer,/innerHTML|fetch\(|physics\.drop|physics\.plan/);
  assert.equal((app.match(/    showRemainingLinkClearance\(\);/g)||[]).length,2);
  assert.match(app,/remainingLinkClearanceSummary = module.remainingLinkClearanceSummary/);
  assert.deepEqual(physicsPlanners({planners:[{id:'hybrid'},{id:'fast'},{id:'link-clearance'}],default_planner:'hybrid'}).choices.map(row=>row.id),['hybrid','fast']);
});
