import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
const source=fs.readFileSync(new URL('../bucket_grid_panel.js',import.meta.url),'utf8');
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
 vm.runInContext(helpers+source.replace('setInterval(bucketGridUpdate,5000);bucketGridUpdate();',''),context);
 return {nodes,context};
}
const cells=[{id:'left',bucket_xy_m:[.47,-.4],distance_m:.36,heading_deg:-33.7},{id:'center',bucket_xy_m:[.67,-.4],distance_m:.3,heading_deg:0},{id:'right',bucket_xy_m:[.87,-.4],distance_m:.36,heading_deg:33.7}];
const fixture={phase:'confirming',run:'fixture',attempts:51,contract:{cells},cells:[{...cells[1],phase:'confirming',winner:{height_m:.35,gap_m:.2,tilt_deg:24},confirmation_summary:{trials:10,route_captures:7,single_bounce_captures:6,clean:5,assisted:2,invalid:0,historical_captures:9}}]};
test('grid distinguishes board-only versus legacy captures and unknown cells',()=>{
 const {nodes,context}=setup();context.report=fixture;vm.runInContext('bucketGridRender(report)',context);
 const rows=nodes.bucketGridTable.children[0].children;
 assert.equal(rows[1].children[5].textContent,'pending');
 assert.equal(rows[2].children[5].textContent,'7/10 (10/30 tested)');
 assert.equal(rows[2].children[7].textContent,'9/10');
 assert.equal(nodes.bucketGridDownload.style.display,'none');
 nodes.bucketGridButtons.children[1].onclick();
 assert.match(nodes.bucketGridDetail.textContent,/ball-bottom Z 550 mm/);
 assert.equal(nodes.bucketGridButtons.children[1].attrs['aria-pressed'],'true');
 assert.match(nodes.bucketGridDetail.textContent,/95% Wilson interval/);
});
test('no-setting and validated exports are explicit',()=>{
 const {nodes,context}=setup();context.report={...fixture,phase:'completed',verification:{},cells:[{...cells[0],phase:'no_route_setting_found'}]};
 vm.runInContext('bucketGridRender(report)',context);
 assert.equal(nodes.bucketGridTable.children[0].children[1].children[5].textContent,'none found');
 assert.match(nodes.bucketGridDetail.textContent,/not proof of impossibility/);
 assert.equal(nodes.bucketGridDownload.style.display,'block');
});
