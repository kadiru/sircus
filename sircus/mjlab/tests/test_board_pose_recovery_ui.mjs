import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {runInNewContext} from 'node:vm';
import {boardPoseRecoverySummary} from '../web/interactive-physics.mjs';
const app=readFileSync(new URL('../web/app.js',import.meta.url),'utf8');
const source=`function showBoardPoseRecovery() {${app.split('function showBoardPoseRecovery() {')[1].split('function showHybridMissCorrection()')[0]}`;
const hash='a'.repeat(64),assets={clip:{url:'data:video/mp4;base64,AAAA',trace_sha256:hash}};
class Element {
  constructor(tag='div'){this.tag=tag;this.children=[];this.textContent='';this.hidden=false;}
  append(...n){this.children.push(...n);} replaceChildren(...n){this.children=n;} setAttribute(k,v){this[k]=v;}
  get text(){return this.textContent+this.children.map(n=>n.text).join(' ');}
  get videos(){return [...(this.tag==='video'?[this]:[]),...this.children.flatMap(n=>n.videos)];}
}
function render(report,videos=assets){const nodes=new Map(),$=id=>{if(!nodes.has(id))nodes.set(id,new Element());return nodes.get(id);};
  $('boardPoseVideoAssets').textContent=JSON.stringify(videos);
  runInNewContext(`${source}\nshowBoardPoseRecovery();`,{report,boardPoseRecoverySummary,$,document:{createElement:t=>new Element(t)}});return $;
}
function fixture(){
  const group=(outcomes)=>({requested:outcomes.length,completed:outcomes.length,released:outcomes.length,
    captures:outcomes.filter(x=>x==='direct'||x==='table').length,direct:outcomes.filter(x=>x==='direct').length,
    table:outcomes.filter(x=>x==='table').length,arm_assisted:outcomes.filter(x=>x==='arm_assisted_capture').length,
    robot_contacts:outcomes.filter(x=>x==='flight_miss'||x==='arm_assisted_capture').length,outcomes});
  const settings={board_xy_m:[.6,-.1],board_height_m:.4,tilt_deg:28,drop_gap_m:.3,heading_offset_deg:10,twist_deg:0,q10:[0,0,0,0,0,0],q14:[0,0,0,0,0,0]};
  return {board_pose_recovery:{version:'sircus-board-pose-recovery-v1',status:'completed',phase:'completed',audit_status:'passed',physics_substeps:64,
    controls:{status:'passed',completed:18,planned:18},progress:{completed:30,planned:30},cases:[0,1,2].map(i=>({id:`case${i}`,bucket_xy_m:[.9+i*.1,-.1],
      preparation:{proposed:20,checked:10,admitted:5,selected:4},settings:{original:structuredClone(settings),selected:structuredClone(settings)},
      confirmed_settings:[{candidate_id:`candidate${i}`,repeat_captures:3,repeat_completed:3,settings:structuredClone(settings)}],
      results:{original:group(['flight_miss','flight_miss','flight_miss']),search:group(['direct','flight_miss','flight_miss','arm_assisted_capture']),repeat:group(['direct','direct','direct'])},
      videos:{original:{asset_id:'clip',trace_sha256:hash,label:'Saved original'},selected:{asset_id:'clip',trace_sha256:hash,label:'Saved selected'}},
    })),next:'Preserve existing settings; no model training.'}};
}
test('search and repeats remain separate with preparation denominator, settings and six video players',()=>{
  const report=fixture(),view=boardPoseRecoverySummary(report,assets),$=render(report);
  assert.equal(view.complete,true);assert.equal(view.cases[0].confirmed.length,1);
  assert.match($('boardPoseResults').text,/900.0, -100.0 20 · 10 1 \/ 4 3 \/ 3 1/);
  assert.match($('boardPoseCases').text,/5 accepted arm solutions; 4 distinct shot settings selected/);
  assert.match($('boardPoseCases').text,/Robot-assisted retention, reported separately: 1 during search; 0 during fresh repeats/);
  assert.match($('boardPoseCases').text,/Heading offset \(°\) 10.00 10.00/);
  assert.equal($('boardPoseCases').videos.length,6);
});
test('pending audit, inconsistent counts, false confirmation or non64 physics withhold final results',()=>{
  for(const change of [s=>s.audit_status='pending',s=>s.physics_substeps=128,s=>s.controls.completed=17,
    s=>s.cases[0].results.search.captures=3,s=>s.cases[0].confirmed_settings[0].repeat_captures=2,
    s=>s.cases[0].results.repeat.requested=2,s=>s.cases[0].preparation.selected=5,s=>s.cases[1].id=s.cases[0].id]){
    const report=fixture();change(report.board_pose_recovery);
    assert.equal(boardPoseRecoverySummary(report,assets).complete,false);const $=render(report);
    assert.equal($('boardPoseResults').children.length,0);assert.equal($('boardPoseCases').videos.length,0);
  }
});
test('zero successful candidates requires no repeats and cannot be mislabeled confirmed',()=>{
  const report=fixture(),c=report.board_pose_recovery.cases[0];
  c.confirmed_settings=[];c.results.search={...c.results.search,captures:0,direct:0,outcomes:['flight_miss','flight_miss','flight_miss','arm_assisted_capture'],robot_contacts:4};
  c.results.repeat={requested:0,completed:0,released:0,captures:0,direct:0,table:0,arm_assisted:0,robot_contacts:0,outcomes:[]};
  assert.equal(boardPoseRecoverySummary(report).complete,true);assert.match(render(report)('boardPoseCases').text,/No setting captured all three/);
});
test('alternative joint solutions and repeated checks are not counted as distinct proposed shots',()=>{
  const report=fixture(),p=report.board_pose_recovery.cases[0].preparation;
  p.admitted=60;p.checked=80;
  assert.equal(boardPoseRecoverySummary(report).complete,true);
  assert.match(render(report)('boardPoseCases').text,/60 accepted arm solutions; 4 distinct shot settings/);
});
test('unsafe or mismatched media is never attached; optional panel preserves older pages',()=>{
  const report=fixture();for(const asset of [{url:'https://example.com/movie.mp4',trace_sha256:hash},{url:assets.clip.url,trace_sha256:'b'.repeat(64)}])
    assert.equal(render(report,{clip:asset})('boardPoseCases').videos.length,0);
  assert.equal(boardPoseRecoverySummary({}),null);assert.equal(render({})('boardPoseRecovery').hidden,true);
  const html=readFileSync(new URL('../web/index.html',import.meta.url),'utf8');
  assert.match(html,/id="boardPoseVideoAssets" type="application\/json">\{\}/);
  assert.match(html,/local development search, not a new policy score/);
});
