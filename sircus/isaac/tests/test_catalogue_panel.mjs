import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
const source=fs.readFileSync(new URL('../catalogue_panel.js',import.meta.url),'utf8');
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
 vm.runInContext(helpers+source.replace('setInterval(catalogueUpdate,5000);catalogueUpdate();',''),context);
 return {nodes,context};
}
const fixture={run:'fixture',phase:'grid',search_completed:3,grid_planned:495,refinement_planned:0,search_summary:{clean:1,assisted:0,misses:1,invalid:1},contract:{grid:{heights_m:[.4],gaps_m:[.12],tilts_deg:[12,15,18,21]},parent:{scene:{ball:{radius_m:.0375}}}},rows:[{height_m:.4,gap_m:.12,tilt_deg:12,success:true,quality_score:2,minimum_mouth_clearance_mm:20,trial_id:'a'},{height_m:.4,gap_m:.12,tilt_deg:15,success:false,quality_score:0},{height_m:.4,gap_m:.12,tilt_deg:18,invalid:'IK',success:false,quality_score:0}],confirmed:[]};
test('grid distinguishes clean, miss, rejected and pending; no completed claim',()=>{
 const {nodes,context}=setup();context.report=fixture;vm.runInContext('catalogueRender(report)',context);
 assert.match(nodes.catalogueDecision.textContent,/incomplete/);
 const cells=nodes.catalogueHeatmaps.children[0].children[1].children.filter(n=>n.tag==='rect');
 assert.deepEqual(cells.map(n=>n.attrs.fill),['#55c9ad','#bc7862','#74828a','#294047']);
 assert.equal(nodes.catalogueAll.children[0].children.length,2);
});
test('confirmation shows actual denominators and different height meanings',()=>{
 const {nodes,context}=setup();context.report={...fixture,phase:'completed',confirmed:[{candidate:{tilt_deg:30},board_height_mm:400,ball_bottom_release_height_mm:520,ball_center_release_height_mm:557.5,dropper_flange_height_mm:652.5,distance_mm:400,nominal:[],perturbed:[],nominal_summary:{clean:10,assisted:8,misses:1,invalid:1,trials:20}}]};vm.runInContext('catalogueRender(report)',context);
 const values=nodes.catalogueFinalists.children[0].children[1].children.map(n=>n.textContent);
 assert.deepEqual(values,['400.0','30.0','520.0','557.5','652.5','400.0','10 / 8 / 1 / 1 (20/20)','pending']);
 assert.match(nodes.catalogueDecision.textContent,/not guaranteed/);
});

test('independent confirmation must match source catalogue',()=>{
 const {nodes,context}=setup();context.report=fixture;context.fresh={phase:'completed',contract:{source_run:'old',candidate:{height_m:.4,gap_m:.13,tilt_deg:21}},summary:{captures:98,trials:100,clean:90,assisted:8,misses:2,invalid:0}};
 vm.runInContext('catalogueFresh(report,fresh)',context);assert.match(nodes.catalogueFresh.textContent,/No matching/);
 context.fresh.contract.source_run='fixture';vm.runInContext('catalogueFresh(report,fresh)',context);assert.match(nodes.catalogueFresh.textContent,/98\/100 captures/);assert.match(nodes.catalogueFresh.textContent,/2 misses/);
});
