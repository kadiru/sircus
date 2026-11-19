import fs from 'node:fs';
import assert from 'node:assert/strict';
const source=fs.readFileSync(new URL('../shot_benchmark_panel.js',import.meta.url),'utf8');
const panel=await import('data:text/javascript;base64,'+Buffer.from(source).toString('base64'));
const pending={method:'ppo',seed:20261121,status:'pending',train:null,learning_curve:[],checkpoints:[],resources:{headless:null,render:null},videos:[]};
const train={attempts:1000,captures:100,clean:20,rejected:600,capture_rate:.1,failures:{ik:600}};
const complete={...pending,status:'running',train,learning_curve:[{...train,mean_reward:.2}],checkpoints:[{training_attempts:1000,checkpoint_sha256:'test',validation:{...train,attempts:35,captures:7,capture_rate:.2},test:null}]};
const report={runs:[pending,complete],seeds:[20261121,20261122,20261123]};
assert.equal(panel.runRows(report)[0][2],'pending');assert.match(panel.runRows(report)[1][3],/100 \/ 1000 · 10.0%/);
assert.equal(panel.frozenRows(report).length,1);assert.equal(panel.frozenRows(report)[0][2],'validation');assert.deepEqual(panel.curveSeries(report,'train')[1].points,[[1000,.1]]);
assert.deepEqual(panel.curveSeries(report,'reward')[1].points,[[1000,.2]]);assert.deepEqual(panel.curveSeries(report,'validation')[1].points,[[1000,.2]]);
assert.equal(panel.resourceRows(report)[0][2],'unknown');assert.deepEqual(panel.failureRows(report),[['ik','unknown',600]]);
assert.equal(panel.testIsFinal(report),true);complete.checkpoints[0].test={attempts:70};assert.equal(panel.testIsFinal(report),false);
assert.doesNotMatch(source,/innerHTML|WebSocket|method:\s*['"]POST|XMLHttpRequest/);
const html=fs.readFileSync(new URL('../shot_benchmark.html',import.meta.url),'utf8');for(const id of ['trainCurve','rewardCurve','validationCurve','frozenTable','resourceTable','layoutDetails','videoList','sourceIdentity'])assert.ok(html.includes(`id="${id}"`));
assert.match(html,/No per-layout physics search/);assert.match(html,/separate 70-position test/);assert.match(html,/Previous study viewers stay separate/);
console.log('Shot benchmark read-only panel checks passed');
// Render with a minimal DOM: no network, browser, simulation or service launch.
class Element{constructor(tag){this.tag=tag;this.children=[];this.attributes={};this.textContent='';}appendChild(x){this.children.push(x);return x;}replaceChildren(...x){this.children=x;}setAttribute(k,v){this.attributes[k]=v;}}
const nodes=new Map();globalThis.document={createElement:t=>new Element(t),createElementNS:(_,t)=>new Element(t),getElementById:id=>{if(!nodes.has(id))nodes.set(id,new Element(id));return nodes.get(id);}};
complete.checkpoints[0].test=null;
const display={...report,schema:'sircus-shot-benchmark-report-v1',experiment_id:'synthetic-test-only',status:'in_progress',observed_train_attempts:1000,source:{commit:'test'},split_manifest_sha256:'test',source_sha256:{},method_summary:[],scientific_status:'Synthetic test only',scope:'No actual data',resource_scope:'Unknown stays unknown',geometry_uncertainty:'Provisional'};
panel.render(display);assert.match(nodes.get('status').textContent,/synthetic-test-only/);assert.equal(nodes.get('metrics').children.length,3);assert.equal(nodes.get('runTable').children[0].tag,'table');assert.ok(nodes.get('trainCurve').children.length>0);assert.match(nodes.get('videoList').children[0].textContent,/pending/);
delete globalThis.document;
console.log('Shot benchmark DOM smoke passed without network or services');
