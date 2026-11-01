import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
const source=fs.readFileSync(new URL('../parallel_panel.js',import.meta.url),'utf8');
class Node {
 constructor(tag){this.tag=tag;this.children=[];this.style={};this.attrs={};this.textContent='';}
 replaceChildren(){this.children=[];}
 appendChild(n){this.children.push(n);return n;}
 setAttribute(k,v){this.attrs[k]=v;}
 getAttribute(k){return this.attrs[k];}
 removeAttribute(k){delete this.attrs[k];}
}
function setup(){
 const nodes={};const document={getElementById:id=>nodes[id]??=new Node('div'),createElement:t=>new Node(t)};
 const context=vm.createContext({document,Math,JSON,Date,setInterval:()=>{},fetch:async()=>{throw Error('offline');}});
 vm.runInContext(source.replace('setInterval(parallelUpdate,4000);parallelUpdate();',''),context);
 return {nodes,context};
}
const base={id:'cell-00',index:0,bucket_xy_m:[.47,-.4],board_xy_m:[.67,-.1],heading_deg:-33.7,distance_m:.36,rows:[],confirmations:[],winner:null};
const fixture={phase:'completed',run:'example',timestamp_utc:new Date().toISOString(),contract:{num_envs:2,batches:28},attempts:60,batches_completed:28,cells:[base,{...base,id:'cell-01',index:1,bucket_xy_m:[.57,-.4],winner:{height_m:.35,gap_m:.2,tilt_deg:24},confirmations:[{route_success:true,single_bounce_success:true,route_quality:2},{route_success:false,single_bounce_success:false,route_quality:0}]}],probes:[],scope:'synthetic',view:{overview:'/overview.jpg',close:'/close.jpg',render_frame:1,labels:[]}};
test('cell selection distinguishes pending/no-setting from measured repeat results',()=>{
 const {nodes,context}=setup();context.r=fixture;vm.runInContext('parallelRender(r)',context);
 assert.equal(nodes.arms.textContent,4);assert.match(nodes.outcome.textContent,/No fresh confirmation/);
 nodes.cells.children[1].onclick();
 assert.equal(nodes.selectedId.textContent,'cell-01');assert.match(nodes.outcome.textContent,/1\/2 captured/);
 assert.equal(nodes.detail.children.find(r=>r.children[0].textContent==='Ball-bottom release height').children[1].textContent,'550 mm');
 assert.match(nodes.closeCaption.textContent,/table 1/);
});
test('stale and mismatched video cannot look like current evidence',()=>{
 const {nodes,context}=setup();context.r={...fixture,phase:'searching',timestamp_utc:'2020-01-01',video:{run:'older',url:'/wrong.mp4'}};vm.runInContext('parallelRender(r)',context);
 assert.match(nodes.status.textContent,/updates stale/);assert.equal(nodes.video.attrs.src,undefined);
 context.r={...fixture,video:{run:'example',url:'/right.mp4'},verification:{passed:true},csv_url:'/settings.csv'};vm.runInContext('parallelRender(r)',context);
 assert.equal(nodes.video.attrs.src,'/right.mp4');assert.equal(nodes.csvLink.href,'/settings.csv');
 context.r={...fixture,video:null};vm.runInContext('parallelRender(r)',context);assert.equal(nodes.video.attrs.src,undefined);
});
test('view zoom and labels are optional and do not change selected data',()=>{
 const {nodes,context}=setup();context.r=fixture;vm.runInContext('parallelRender(r)',context);
 nodes.zoom.onclick();assert.equal(nodes.sceneLayer.style.transform,'scale(1)');
 nodes.showLabels.onclick();assert.equal(nodes.labels.className,'');
 assert.equal(nodes.selectedId.textContent,'cell-00');
 nodes.showLabels.onclick();assert.equal(nodes.labels.className,'hidden');
});
