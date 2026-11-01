import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
const source=fs.readFileSync(new URL('../plane_panel.js',import.meta.url),'utf8');
const preview=fs.readFileSync(new URL('../preview.py',import.meta.url),'utf8');
const helpers=preview.slice(preview.indexOf('function studyTable'),preview.indexOf('function studyAxes'));
class Node {
 constructor(tag){this.tag=tag;this.children=[];this.style={};this.attrs={};this.textContent='';}
 replaceChildren(){this.children=[];}
 appendChild(n){this.children.push(n);return n;}
 setAttribute(k,v){this.attrs[k]=v;}
}
function setup(){
 const nodes={};const document={getElementById:id=>nodes[id]??=new Node('div'),createElement:t=>new Node(t),createElementNS:(_,t)=>new Node(t)};
 const context=vm.createContext({document,Math,JSON,encodeURIComponent,setInterval:()=>{},fetch:async()=>{throw Error('offline');}});
 vm.runInContext(helpers+source.replace('setInterval(planeUpdate,5000);planeUpdate();',''),context);
 return {nodes,context};
}
const fixture={run:'fixture',phase:'grid',completed:2,planned:175,captures:1,invalid:1,contract:{distance_m:.4,board_xy_m:[.67,-.1],heights_m:[.4],gaps_m:[.12],tilts_deg:[17,22],scene:{bucket:{assumed_bottom_thickness_m:.008,assumed_interior_depth_m:.175,inner_top_diameter_m:.177,inner_bottom_diameter_m:.15},ball:{radius_m:.0375}}},rows:[{height_m:.4,gap_m:.12,tilt_deg:17,success:true},{height_m:.4,gap_m:.12,tilt_deg:22,success:false,invalid:'endpoint_ik'}],repeats:[],replays:[]};
test('incomplete sweep shows pending and rejected distinctly, no old image',()=>{
 const {nodes,context}=setup();context.report=fixture;vm.runInContext('planeRender(report)',context);
 assert.match(nodes.planeDecision.textContent,/incomplete/);assert.match(nodes.planeStatus.textContent,/2\/175/);assert.equal(nodes.planeSetup.style.display,'none');
 const cells=nodes.planeHeatmaps.children[0].children[1].children.filter(n=>n.tag==='rect');
 assert.deepEqual(cells.map(n=>n.attrs.fill),['#55c9ad','#74828a']);
});
test('completed replay is separately labeled and selectable without a command API',()=>{
 const {nodes,context}=setup();const c=fixture.rows[0];context.report={...fixture,phase:'completed',repeats:[{candidate:c,trials:[c,c]}],replays:[{candidate:c,result:c,trajectory_m:[[.67,-.1,.5575],[.67,-.5,.05]]}]};
 vm.runInContext('planeRender(report)',context);
 assert.equal(nodes.planeReplayButtons.children.length,1);assert.match(nodes.planeShot.textContent,/fresh replay: capture/);assert.match(nodes.planeDecision.textContent,/no learned policy/);assert.equal(nodes.planeSetup.style.display,'block');
});
