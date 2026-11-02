import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
const source=fs.readFileSync(new URL('../feasibility_panel.js',import.meta.url),'utf8');
class Node {
 constructor(tag){this.tag=tag;this.children=[];this.attrs={};this.textContent='';this.value='nominal';}
 replaceChildren(){this.children=[];}
 appendChild(n){this.children.push(n);return n;}
 setAttribute(k,v){this.attrs[k]=v;}
}
function setup(){const nodes={};const document={getElementById:id=>nodes[id]??=new Node('div'),createElement:t=>new Node(t),createElementNS:(_,t)=>new Node(t)};const context=vm.createContext({document,Math,JSON,Date,setInterval:()=>{},cancelAnimationFrame:()=>{},fetch:async()=>{throw Error('offline');}});vm.runInContext(source.replace('setInterval(feasibilityUpdate,20000);feasibilityUpdate();',''),context);return {nodes,context};}
const c={id:'target-00',index:0,bucket_xy_m:[1.1,-.47],winner:null,baseline:[],search:[],refinement:[],nominal:[],fixed_xy_ablation:[],confirmations:[]};
const scene={arms:{base_spacing_m:1.33731},ball:{radius_m:.0375},bucket:{assumed_bottom_thickness_m:.008,assumed_interior_depth_m:.175}};
const fixture={run:'fixture',phase:'completed',attempts:1,timestamp_utc:new Date().toISOString(),cells:[c],contract:{scene,scope:'Development only',unknown_rule:'Unresolved, not impossible',geometry_status:'Provisional'},probes:[]};
test('unresolved targets never become impossible or successful',()=>{const {nodes,context}=setup();context.r=fixture;vm.runInContext('renderFeasibility(r)',context);assert.match(nodes.detail.children[1].children[1].textContent,/No success found/);assert.match(nodes.outcome.textContent,/No fresh confirmation/);assert.equal(nodes.play.disabled,true);assert.match(nodes.limits.children[1].textContent,/not impossible/);});
test('measured repeats retain failures and separate old-XY comparison',()=>{const {nodes,context}=setup();const winner={id:'winner',board_x_m:.72,board_y_m:-.15,height_m:.35,gap_m:.2,tilt_deg:24};context.r={...fixture,cells:[{...c,winner,nominal:[{route_success:true}],fixed_xy_ablation:[{route_success:false}],confirmations:[{route_success:true,single_bounce_success:true,route_quality:2},{route_success:false,invalid:'ik'}]}]};vm.runInContext('renderFeasibility(r)',context);assert.match(nodes.outcome.textContent,/1\/2 captures/);assert.match(nodes.outcome.textContent,/1 endpoint rejects/);assert.match(nodes.comparison.textContent,/1\/1.*0\/1/);assert.equal(nodes.detail.children.find(r=>r.children[0].textContent==='Ball-center release Z').children[1].textContent,'587.5 mm');});
test('stale progress and missing endpoint audit are explicit',()=>{const {nodes,context}=setup();context.r={...fixture,phase:'search',timestamp_utc:'2020-01-01'};vm.runInContext('renderFeasibility(r)',context);assert.match(nodes.status.textContent,/stale/);assert.match(nodes.reach.textContent,/pending/);});
test('network failure is visible without controls',async()=>{const {nodes,context}=setup();await vm.runInContext('feasibilityUpdate()',context);assert.match(nodes.status.textContent,/unavailable/);assert.equal(source.includes('/api/'),false);});
test('live and compact reports separate search IK failures from released misses',()=>{
 const {context}=setup();
 const rows=[{route_success:false,invalid:'endpoint_ik_Arm14'},{route_success:false,external_contact_steps:3},{route_success:false},{route_success:true}];
 context.live={...c,search:rows,confirmations:[{invalid:'endpoint_ik_Arm14'}]};
 context.compact={...c,stages:{search:{failures:{endpoint_ik_Arm14:1,external_contact:1,released_miss:1}}}};
 const live=JSON.parse(vm.runInContext('JSON.stringify(searchFailures(live))',context));
 const compact=JSON.parse(vm.runInContext('JSON.stringify(searchFailures(compact))',context));
 assert.deepEqual(live,compact);assert.equal(live.reduce((n,r)=>n+r[1],0),3);
});
