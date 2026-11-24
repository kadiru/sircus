import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {expandedSelectorReplaysReady,expandedSelectorReplayMovie,renderExpandedSelectorReplays} from '../web/interactive-physics.mjs';
const hash='a'.repeat(64), methods=['current','expanded'];
const example=()=>({version:'sircus-expanded-selector-replays-display-v1',status:'completed',audit_status:'passed',physics_substeps:64,
 provenance:{report_sha256:hash,audit_sha256:hash,diagnosis_sha256:hash},parent_summary:{positions:50,current_captures:44,expanded_captures:44},
 cases:['04','10','19','47'].map((n,i)=>({id:`expanded-selector-test-${n}`,bucket_xy_m:[i/10,0],change:i%2?'expanded_gained':'expanded_lost',findings:['One saved observation'],hypotheses:['Untested hypothesis'],
  methods:Object.fromEntries(methods.map(m=>[m,{summary:'Recorded observation',id:`expanded-selector-test-${n}-${m}`,outcome:{capture:m==='current'?i%2===0:i%2===1,route_class:'direct',max_retention_s:1},
   settings:{board_center_m:[0,0,.3],command_board_center_m:[0,0,.3],board_height_m:.3,tilt_deg:28,drop_gap_m:.35,release_ball_center_m:[0,0,.68],heading_offset_deg:0,twist_deg:0,q10_rad:[0,0,0,0,0,0],q14_rad:[0,0,0,0,0,0]},trace:{sha256:hash,column:i}}]))}))});
function addMovie(row){
 row.video={asset_id:row.id,video_sha256:hash,receipt_sha256:hash};
 return {[row.id]:{case_id:row.id,video_sha256:hash,receipt_sha256:hash,native_steps_added:0,held_arm_poses:true,url:'data:video/mp4;base64,AAAA',
  methods:Object.fromEntries(methods.map(m=>[m,{record_id:row.methods[m].id,trace_sha256:hash,column:row.methods[m].trace.column}]))}};
}
test('four historical pairs require audited outcomes, exact methods and finite arm settings',()=>{
 assert.equal(expandedSelectorReplaysReady(example()),true);
 for(const change of [{status:'running'},{audit_status:'pending'},{physics_substeps:128},{provenance:{}},{cases:[]}])assert.equal(expandedSelectorReplaysReady({...example(),...change}),false);
 for(const mutate of [x=>x.cases[0].methods.current.outcome.capture=false,x=>x.cases[0].methods.current.settings.q10_rad=[NaN],x=>x.cases[0].methods.current.trace.column=-1,x=>x.cases[0].change='expanded_gained',x=>x.parent_summary.positions=4,x=>x.cases[0].methods.current.settings.command_board_center_m=null]){const x=example();mutate(x);assert.equal(expandedSelectorReplaysReady(x),false);}
});
test('movie must match both trace columns, receipt, case and permitted local data URI',()=>{
 const row=example().cases[0],assets=addMovie(row);assert.ok(expandedSelectorReplayMovie(row,assets));
 for(const mutate of [a=>a[row.id].case_id='other',a=>a[row.id].methods.expanded.column=5,a=>a[row.id].methods.current.record_id='other',a=>a[row.id].native_steps_added=1,a=>a[row.id].receipt_sha256='b'.repeat(64),a=>a[row.id].url='https://example.com/movie.mp4',a=>a[row.id].url='javascript:alert(1)']){const a=structuredClone(assets);mutate(a);assert.equal(expandedSelectorReplayMovie(row,a),null);}
});
class Node{
 constructor(tag,doc){this.tagName=tag;this.ownerDocument=doc;this.children=[];this.textContent='';this.attrs={};this.hidden=false;}
 append(...nodes){this.children.push(...nodes);}
 replaceChildren(...nodes){this.children=[...nodes];}
 setAttribute(k,v){this.attrs[k]=v;}
 querySelector(selector){return this.slots?.[selector.match(/="([^"]+)"/)[1]];}
}
function root(){const d={createElement:tag=>new Node(tag,d)},r=new Node('section',d);r.slots=Object.fromEntries(['status','summary','proof','cases'].map(k=>[k,new Node('div',d)]));return r;}
function descendants(n){return [n,...n.children.flatMap(descendants)];}
test('rendering preserves playing elements across polls, with no automatic playback or raw HTML',()=>{
 const r=root(),raw=example(),assets=addMovie(raw.cases[0]);raw.cases[0].findings=['<script>untrusted</script>'];
 renderExpandedSelectorReplays(r,raw,assets);const first=r.slots.cases.children[0],all=descendants(r.slots.cases),videos=all.filter(n=>n.tagName==='video');
 assert.equal(videos.length,1);assert.equal(videos[0].controls,true);assert.equal(videos[0].preload,'none');assert.notEqual(videos[0].autoplay,true);
 assert.ok(all.find(n=>n.tagName==='li'&&n.textContent==='<script>untrusted</script>'));
 renderExpandedSelectorReplays(r,structuredClone(raw),assets);assert.equal(r.slots.cases.children[0],first);
 renderExpandedSelectorReplays(r,{...raw,audit_status:'pending'},assets);assert.equal(r.slots.cases.children.length,0);assert.equal(r.slots.summary.textContent,'');
});
test('additive panel and asset script are unique; both refresh paths update read-only view',()=>{
 const html=readFileSync(new URL('../web/index.html',import.meta.url),'utf8'),app=readFileSync(new URL('../web/app.js',import.meta.url),'utf8');
 for(const id of ['expandedSelectorReplays','expandedSelectorReplayVideoAssets','expandedSelectorStudy','boardPoseVideoAssets'])assert.equal((html.match(new RegExp(`id="${id}"`,'g'))||[]).length,1);
 assert.equal((app.match(/    showExpandedSelectorReplays\(\);/g)||[]).length,2);
 const source=readFileSync(new URL('../web/interactive-physics.mjs',import.meta.url),'utf8').split('// A read-only view of four audited saved flights.')[1];
 assert.doesNotMatch(source,/fetch\(|innerHTML|\.drop\(|\.plan\(|autoplay\s*=/);
 assert.match(source,/player\.play\(\)/);assert.match(source,/replay\.onclick=/);
});
