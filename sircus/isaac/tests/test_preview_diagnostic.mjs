import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

const source=fs.readFileSync(new URL('../preview.py',import.meta.url),'utf8');
const helpers=source.slice(source.indexOf('function studyTable'),source.indexOf('function studyAxes'));
const update=source.slice(source.indexOf('async function diagnosticUpdate'),source.indexOf('setInterval(diagnosticUpdate'));
class Node {
  constructor(tag){this.tag=tag;this.children=[];this.style={};this.textContent='';this.attrs={};}
  replaceChildren(){this.children=[];}
  appendChild(child){this.children.push(child);return child;}
  setAttribute(k,v){this.attrs[k]=v;}
}
function setup(report,fail=false){
 const nodes={};
 const document={getElementById:id=>nodes[id]??=(new Node('div')),createElement:t=>new Node(t),createElementNS:(_,t)=>new Node(t)};
 const ctx=vm.createContext({document,fetch:async()=>{if(fail)throw Error('offline');return {json:async()=>report};},Date,Math});
 vm.runInContext(helpers+update,ctx);
 return {ctx,nodes};
}
test('diagnostic panel renders solver comparison and searched grid distinctly',async()=>{
 const report={state:'completed',completed:215,planned:215,study:'fixture',ik_audit:{stage_counts:{iteration:29,seed:25,unresolved:8},historical_failures:62,completed:78,planned:78},conclusion:'Diagnostic only',distances:[{distance_m:.2,policies:[{label:'ppo_100',solver:'original',success:false},{label:'ppo_100',solver:'extended',success:true}],grid_captures:10,grid_tested:35,grid_invalid:7,grid_best_action:[16,0,.2],repeats:[{success:true}],grid_cells:[{tilt_deg:16,gap_m:.2,success:true}]}]};
 const {ctx,nodes}=setup(report);await vm.runInContext('diagnosticUpdate()',ctx);
 assert.match(nodes.diagnosticStatus.textContent,/215\/215/);
 const rows=nodes.distanceSweepTable.children[0].children;
 assert.equal(rows[1].children[1].textContent,'miss → capture');
 assert.match(rows[1].children[4].textContent,/10\/35/);
 assert.equal(nodes.sweepHeatmaps.children.length,1);
 assert.equal(nodes.diagnosticConclusion.textContent,'Diagnostic only');
});
test('unavailable diagnostic does not display a completed result',async()=>{
 const {ctx,nodes}=setup(null,true);await vm.runInContext('diagnosticUpdate()',ctx);
 assert.match(nodes.diagnosticStatus.textContent,/available|No|unavailable/i);
});
