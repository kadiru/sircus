import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
const source=fs.readFileSync(new URL('../distance_catalogue_panel.js',import.meta.url),'utf8');
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
 vm.runInContext(helpers+source.replace('setInterval(distanceCatalogueUpdate,5000);distanceCatalogueUpdate();',''),context);
 return {nodes,context};
}
const fixture={phase:'confirming',run:'fixture',attempts:15,contract:{distances_m:[.4,.3,.5]},distances:[{distance_m:.4,phase:'confirmation',rows:[],screened:[],winner:{height_m:.4,gap_m:.13,tilt_deg:21},confirmation_summary:{trials:10,captures:8,clean:6,assisted:2,misses:2,invalid:0}}]};
test('partial fresh results retain denominators and pending distances',()=>{
 const {nodes,context}=setup();context.report=fixture;vm.runInContext('distanceCatalogueRender(report)',context);
 const rows=nodes.distanceCatalogueTable.children[0].children;
 assert.equal(rows[1].children[1].textContent,'pending');
 assert.equal(rows[2].children[7].textContent,'8/10 of 50');
 assert.equal(rows[2].children[6].textContent,'567.5');
 assert.match(nodes.distanceCatalogueDecision.textContent,/incomplete/);
 assert.equal(nodes.distanceCatalogueDownload.style.display,'none');
});
test('no capture found is distinguished from a zero-rate confirmed setting',()=>{
 const {nodes,context}=setup();context.report={...fixture,phase:'completed',distances:[...fixture.distances,{distance_m:.5,phase:'no_capture_found',rows:[],screened:[]}]};vm.runInContext('distanceCatalogueRender(report)',context);
 const row=nodes.distanceCatalogueTable.children[0].children[3];assert.equal(row.children[7].textContent,'No captured setting found');
 assert.match(nodes.distanceCatalogueDecision.textContent,/not proof of impossibility/);assert.equal(nodes.distanceCatalogueDownload.style.display,'block');
});
test('selected setup uses matching scene dimensions and switches without motion',()=>{
 const {nodes,context}=setup();
 const scene=JSON.parse(fs.readFileSync(new URL('../scene.json',import.meta.url),'utf8'));
 context.report={...fixture,contract:{...fixture.contract,parent_quality:{parent:{scene}}},distances:[...fixture.distances,{distance_m:.6,phase:'completed',rows:[],screened:[],winner:{height_m:.5,gap_m:.2,tilt_deg:30}}]};
 vm.runInContext('distanceCatalogueRender(report)',context);
 assert.equal(nodes.distanceSetupButtons.children.length,2);
 let text=nodes.distanceSetupDiagram.children.map(n=>n.textContent).join(' ');
 assert.match(text,/Ball-center Z 567.5 mm/);assert.match(text,/distance: 400 mm/);
 nodes.distanceSetupButtons.children[1].onclick();
 text=nodes.distanceSetupDiagram.children.map(n=>n.textContent).join(' ');
 assert.match(text,/Ball-center Z 737.5 mm/);assert.match(text,/Board H 500.0 mm · tilt 30°/);
 assert.match(text,/distance: 600 mm/);assert.match(nodes.distanceSetupCaption.textContent,/not a trajectory/);
});
